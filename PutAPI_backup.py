# -*- coding: utf-8 -*-
"""
核心優化：
1. [手動輪替] 程式啟動時自動檢查 Log 大小並搬移備份，維持 10MB 容量限制。
2. [狀態追蹤] 引入 State File，這批全成功才推進狀態，斷線重連不遺失。
3. [安全清理] 分批刪除舊資料，不卡死 AI 寫入。
4. [防並行鎖] fcntl File Lock，防止排程重疊。
"""

import os
import sys
import glob
import time
import sqlite3
import json
import datetime
import urllib3
import fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging 

import requests
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 系統設定區塊 =================
DB_DIR = "/home/nvidia/THI/DeepStream-TMC/output_db"
API_URL = "https://x235aiapi.thix180server.com:4004/AIDetect_Track_detection_stats"

MAX_WORKERS = 8          # 並發執行緒數
REQUEST_TIMEOUT = 10     # 單筆請求逾時秒數
MAX_RETRIES = 2          # 單筆失敗重試次數
RETENTION_DAYS = 7       # 本地 DB 保留天數

LOG_FILENAME = "PutAPI_run_old.log" # 維持你原本的檔名
STATE_FILENAME = "upload_state_track.json" 
LOCK_FILENAME = "PutAPI_run_track.lock"

MAX_FETCH_LIMIT = 50000  # 單次 SQL 撈取上限 (防 OOM)
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 3

# ================= ⭐ 手動 Log 輪替與 Logger 設定 =================
def manual_log_rotation():
    """程式啟動時手動檢查並輪替 Log 檔案，相容 Crontab 的 >> 寫入"""
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, LOG_FILENAME)
    
    if os.path.exists(log_path) and os.path.getsize(log_path) >= MAX_LOG_SIZE:
        oldest_backup = f"{log_path}.{BACKUP_COUNT}"
        if os.path.exists(oldest_backup):
            try: os.remove(oldest_backup)
            except OSError: pass
            
        for i in range(BACKUP_COUNT - 1, 0, -1):
            src = f"{log_path}.{i}"
            dst = f"{log_path}.{i+1}"
            if os.path.exists(src):
                try: os.rename(src, dst)
                except OSError: pass
                
        try: os.rename(log_path, f"{log_path}.1")
        except OSError: pass

manual_log_rotation()

logger = logging.getLogger("PutAPI_Track_Uploader")
logger.setLevel(logging.INFO)

def setup_logging():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 只輸出到 stdout，讓 Crontab 的 >> 去寫入檔案
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False 

setup_logging()

# ================= 狀態管理 =================
def load_state():
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILENAME)
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"讀取狀態檔失敗，將使用預設值: {e}")
    return {}

def save_state(state):
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILENAME)
    try:
        temp_path = state_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, state_path)
    except Exception as e:
        logger.error(f"儲存狀態檔失敗: {e}")

# ================= 資料庫操作 =================
def fetch_rows(db_file, since_str):
    name = os.path.basename(db_file)
    records = []
    max_time = None
    conn = None
    try:
        db_uri = f"{Path(db_file).as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=10) 
        c = conn.cursor()
        
        query = """
            SELECT CameraCode, DeviceCode, DetectClass, TrackID, 
                   FromRoadID, FromRoadName, ToRoadID, ToRoadName, 
                   Path, CollectTime 
            FROM AiTrafficFlowRawData 
            WHERE CollectTime > ? 
            ORDER BY CollectTime ASC
            LIMIT ?
        """
        rows = c.execute(query, (since_str, MAX_FETCH_LIMIT)).fetchall()

        for row in rows:
            records.append({
                'CameraCode': row[0], 'DeviceCode': row[1], 'DetectClass': row[2], 
                'TrackID': row[3], 'FromRoadID': row[4], 'FromRoadName': row[5], 
                'ToRoadID': row[6], 'ToRoadName': row[7], 'Path': row[8], 'CollectTime': row[9],
            })
            max_time = row[9] 

        if records:
            logger.info(f"[{name}] 撈取 {len(records)} 筆 (起點: {since_str}, 終點: {max_time})")
    except sqlite3.Error as sql_err:
        logger.error(f"[{name}] 資料庫存取失敗: {sql_err}")
    finally:
        if conn: conn.close()
    return records, max_time

def cleanup_db(db_file, max_uploaded_time, retention_days):
    name = os.path.basename(db_file)
    conn = None
    try:
        dt_max = datetime.datetime.strptime(max_uploaded_time, '%Y-%m-%d %H:%M:%S')
        dt_cleanup = dt_max - datetime.timedelta(days=retention_days)
        cleanup_time_str = dt_cleanup.strftime('%Y-%m-%d %H:%M:%S')

        db_uri = f"{Path(db_file).as_uri()}"
        conn = sqlite3.connect(db_uri, uri=True, timeout=30)
        c = conn.cursor()
        total_deleted = 0
        batch_size = 1000 
        delete_query = """
            DELETE FROM AiTrafficFlowRawData 
            WHERE rowid IN (SELECT rowid FROM AiTrafficFlowRawData WHERE CollectTime <= ? LIMIT ?)
        """
        while True:
            c.execute(delete_query, (cleanup_time_str, batch_size))
            deleted_count = c.rowcount
            total_deleted += deleted_count
            conn.commit() 
            if deleted_count < batch_size: break
            time.sleep(0.05) 
        if total_deleted > 0:
            logger.info(f"[{name}] 成功分批清理 {total_deleted} 筆舊資料 (保留 {retention_days} 天)")
    except Exception as e:
        logger.error(f"[{name}] 清理失敗: {e}")
    finally:
        if conn: conn.close()

def purge_history_data(db_files):
    since_str = compute_default_since_str()
    logger.info(f"【首次部署】將清理 {since_str} 之前的歷史包袱資料...")
    for db_file in db_files:
        name = os.path.basename(db_file)
        conn = None
        try:
            db_uri = f"{Path(db_file).as_uri()}"
            conn = sqlite3.connect(db_uri, uri=True, timeout=30)
            c = conn.cursor()
            total_deleted = 0
            while True:
                c.execute("DELETE FROM AiTrafficFlowRawData WHERE rowid IN (SELECT rowid FROM AiTrafficFlowRawData WHERE CollectTime <= ? LIMIT 2000)", (since_str,))
                deleted_count = c.rowcount
                total_deleted += deleted_count
                conn.commit()
                if deleted_count < 2000: break
                time.sleep(0.05)
            if total_deleted > 0:
                logger.info(f"[{name}] 首次部署清理：丟棄 {total_deleted} 筆歷史資料。")
        except Exception as e:
            logger.error(f"[{name}] 首次清理失敗: {e}")
        finally:
            if conn: conn.close()

# ================= API 上傳 =================
def build_session():
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = False
    session.headers.update({"Content-Type": "application/json"})
    return session

def upload_one(session, data):
    """多執行緒執行的單筆 PUT 上傳函數"""
    json_data = json.dumps(data)
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.put(API_URL, data=json_data, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return True
        except requests.exceptions.HTTPError as http_err:
            status = http_err.response.status_code if http_err.response is not None else None
            if status is not None and 400 <= status < 500:
                # 4xx 不重試，直接記錄並回傳 False
                return False
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            return False
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
                continue
            return False
    return False

def compute_default_since_str():
    dt = datetime.datetime.now().replace(microsecond=0) - datetime.timedelta(seconds=300)
    dt = dt.replace(second=0)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# ================= 主程式 =================
def main():
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOCK_FILENAME)
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.warning("偵測到另一個執行個體正在運行，本次結束。")
        lock_file.close() 
        return 

    try:
        db_files = sorted(glob.glob(os.path.join(DB_DIR, "*.db")))
        if not db_files:
            logger.warning(f"找不到任何 .db 檔: {DB_DIR}")
            return

        state = load_state()
        is_first_run = not bool(state) 
        
        if is_first_run:
            purge_history_data(db_files)
            for db_file in db_files:
                state[os.path.basename(db_file)] = compute_default_since_str()
            save_state(state)
            logger.info("首次部署清理完成，已初始化狀態檔。")

        session = build_session()
        
        grand_total_success = 0
        grand_total_fail = 0
        
        # === 逐個 DB 獨立處理 ===
        for db_file in db_files:
            name = os.path.basename(db_file)
            current_since = state.get(name, compute_default_since_str())
            
            records, max_time = fetch_rows(db_file, current_since)
            if not records:
                continue
            
            count = len(records)
            logger.info(f"[{name}] 準備使用 {MAX_WORKERS} 條執行緒並發上傳 {count} 筆...")
            
            success_count = 0
            fail_count = 0
            
            # === 多執行緒並發上傳 ===
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(upload_one, session, rec) for rec in records]
                for fut in as_completed(futures):
                    if fut.result():
                        success_count += 1
                    else:
                        fail_count += 1
                        
            logger.info(f"[{name}] 本批上傳結果：成功 {success_count} 筆，失敗 {fail_count} 筆。")
            
            # === 狀態推進邏輯 ===
            # 為了確保狀態機正確，只有當這批資料「100% 全數成功」時，才推進狀態並清理 DB。
            # 若有失敗，狀態不推進，下次排程會重傳 (PUT 具有冪等性，覆蓋重傳是安全的)。
            if fail_count == 0:
                state[name] = max_time
                cleanup_db(db_file, max_time, RETENTION_DAYS)
                logger.info(f"[{name}] 全數上傳成功，狀態已推進，舊資料已清理。")
            else:
                logger.warning(f"[{name}] 存在失敗紀錄，狀態不推進，等待下次排程重傳。")
                
            grand_total_success += success_count
            grand_total_fail += fail_count
                
        session.close()
        save_state(state)
        
        # === 總結報表 ===
        total_all = grand_total_success + grand_total_fail
        if total_all > 0:
            logger.info(f"最終統計：成功 {grand_total_success} 筆，失敗 {grand_total_fail} 筆，總共 {total_all} 筆")
        else:
            logger.info("最終統計：本次無新資料需要上傳。")
            
        logger.info("本次排程任務結束。")

    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

if __name__ == '__main__':
    try:
        main()
        logger.info("排程執行結束。\n" + "-" * 40)
    except Exception as e:
        logger.error(f"主程式執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
    finally:
        sys.exit(0)
