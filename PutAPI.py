# -*- coding: utf-8 -*-
"""
核心特色：
1. [手動輪替] 程式啟動時自動檢查 Log 大小並搬移備份，維持 10MB 容量限制
2. [一次打包] 恢復單一 JSON 陣列一次送出，簡化上傳邏輯
3. [狀態追蹤] 首次部署清理後【立即存檔】，確保後續正常運作
4. [安全清理] 分批刪除舊資料，不卡死 AI 寫入
5. [數據統計] 任務結束時自動統計成功/失敗/總筆數

注意事項：
- 本程式僅支援 Linux 環境（使用 fcntl 檔案鎖）
- 日誌輸出僅到 stdout，由 Crontab 的 '>>' 重定向到檔案
- 程式啟動時會手動輪替日誌檔案，避免無限增長
"""

import os
import sys
import glob
import time
import sqlite3
import json
import datetime
import urllib3
import threading
import fcntl          # Linux 環境用於檔案鎖
from pathlib import Path
import logging 

import requests
from requests.adapters import HTTPAdapter

# 關閉 SSL 驗證警告（內部 API 通常使用自簽憑證）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 系統設定區塊 =================
# 請根據實際部署環境調整以下設定值

# 資料庫相關設定
DB_DIR          = "/home/nvidia/THI/DeepStream-TMC/output_db"  # SQLite 資料庫存放目錄
RETENTION_DAYS  = 7                                            # 本地 DB 保留天數

# API 相關設定
API_URL  = "https://pingits.thix180server.com/pingits/api/AIData/AiTrafficFlowRawData"  # 上傳端點
AUTH_URL = "https://pingits.thix180server.com/pingits/api/Auth/login"                   # 認證端點

# 認證憑證
ENTERPRISE_ID = "THI"           # 企業 ID
USER_ID       = "AiAPI"         # 使用者 ID
PASSWORD      = "Msaj#aV6Lh"    # 密碼

# 連線與重試設定
TOKEN_TTL       = 120        # Token 有效時間（秒）
REQUEST_TIMEOUT = 30         # API 請求超時（秒）
MAX_RETRIES     = 2          # 最大重試次數
MAX_FETCH_LIMIT = 50000      # 單次從 DB 讀取的最大筆數（防 OOM）

# 檔案名稱設定
LOG_FILENAME   = "PutAPI.log"          # 日誌檔名（由 Crontab 重定向）
STATE_FILENAME = "upload_state.json"   # 狀態記錄檔
LOCK_FILENAME  = "PutAPI_run.lock"     # 執行鎖檔案

# 日誌輪替設定
MAX_LOG_SIZE   = 10 * 1024 * 1024   # 10 MB
BACKUP_COUNT   = 3                  # 保留備份數量


# ================= 手動 Log 輪替 =================
def manual_log_rotation():
    """
    手動日誌輪替函數
    
    功能說明：
    - 因為 Crontab 使用 >> 重定向，無法使用 RotatingFileHandler
    - 程式啟動時檢查日誌檔案大小，若超過限制則手動輪替
    - 保留 BACKUP_COUNT 個備份 (.1, .2, .3)
    - 刪除最舊的備份，將現有備份依序向後推移
    """
    log_dir  = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, LOG_FILENAME)
    
    if os.path.exists(log_path) and os.path.getsize(log_path) >= MAX_LOG_SIZE:
        # 1. 刪除最舊的備份（若存在）
        oldest_backup = f"{log_path}.{BACKUP_COUNT}"
        if os.path.exists(oldest_backup):
            try:
                os.remove(oldest_backup)
            except OSError:
                pass
                
        # 2. 將現有備份向後推移 (例如 .2 -> .3, .1 -> .2)
        for i in range(BACKUP_COUNT - 1, 0, -1):
            src = f"{log_path}.{i}"
            dst = f"{log_path}.{i+1}"
            if os.path.exists(src):
                try:
                    os.rename(src, dst)
                except OSError:
                    pass
                    
        # 3. 將當前日誌檔改名為 .1
        try:
            os.rename(log_path, f"{log_path}.1")
        except OSError:
            pass

# 執行日誌輪替（在 logger 初始化之前）
manual_log_rotation()

# 建立專屬 Logger（不繼承 root logger）
logger = logging.getLogger("PutAPI_Uploader")
logger.setLevel(logging.INFO)

def setup_logging():
    """
    設定 Logger 僅輸出到 stdout
    
    功能說明：
    - 只使用 StreamHandler 輸出到標準輸出
    - 由 Crontab 的 '>>' 重定向到日誌檔案
    - 避免重複寫入，徹底解決日誌重複列印問題
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # 清除既有的 handlers，避免重複
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 只加入 StreamHandler (輸出到 stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 切斷向上傳遞，避免重複輸出
    logger.propagate = False 

setup_logging()


# ================= Token 管理 =================
class TokenManager:
    """
    API 認證 Token 管理器
    
    功能：
    - 使用單調時鐘 (monotonic) 管理 Token 生命週期，不受系統時間變動影響
    - 執行緒安全（使用 Lock）
    - 自動在過期前刷新 Token
    """
    def __init__(self):
        self._token          = None         # 儲存 Token
        self._monotonic_time = None         # Token 獲取時的單調時間
        self._lock           = threading.Lock()

    def get_token(self):
        """
        獲取有效的 Token
        
        回傳值：
            str  : 有效的 Token
            None : 無法獲取（認證失敗或網路問題）
        """
        with self._lock:
            now = time.monotonic()
            elapsed = (now - self._monotonic_time) if self._monotonic_time else TOKEN_TTL + 1
            
            # 若無 Token 或已過期，則重新獲取
            if self._token is None or elapsed >= TOKEN_TTL:
                logger.info(f"Token 刷新觸發 (已使用 {elapsed:.0f} 秒)")
                self._token = self._fetch_token()
                self._monotonic_time = now if self._token else None
            return self._token

    def _fetch_token(self):
        """
        向認證伺服器請求新的 Token
        
        回傳值：
            str  : 新 Token
            None : 認證失敗
        """
        payload = {
            "EnterpriseId": ENTERPRISE_ID,
            "UserId": USER_ID,
            "Password": PASSWORD
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                AUTH_URL,
                data=json.dumps(payload),
                headers=headers,
                verify=False,
                timeout=10
            )
            response.raise_for_status()
            token_data = response.json()
            
            if not token_data.get("isPasswordValid", False):
                logger.error("登入失敗：帳號或密碼錯誤")
                return None
                
            token = token_data.get("AccessToken")
            if not token:
                logger.error("登入成功但找不到 AccessToken")
                return None
            return token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"取得 Token 失敗: {e}")
            return None

# 建立全域 Token 管理器實例
token_manager = TokenManager()


# ================= 狀態管理 =================
def load_state():
    """
    讀取上傳狀態檔案（用於斷點續傳）
    
    回傳值：
        dict: 狀態字典，格式為 {資料庫名稱: 最後上傳時間字串}
              若檔案不存在或損壞則返回空字典
    """
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILENAME)
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"讀取狀態檔失敗，將使用預設值: {e}")
    return {}

def save_state(state):
    """
    儲存上傳狀態檔案（使用原子寫入防止斷電損壞）
    
    參數：
        state (dict): 要儲存的狀態字典
    """
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), STATE_FILENAME)
    try:
        # 先寫入暫存檔，再取代原檔案
        temp_path = state_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
        os.replace(temp_path, state_path)
    except Exception as e:
        logger.error(f"儲存狀態檔失敗: {e}")


# ================= 資料庫操作 =================
def fetch_rows(db_file, since_str):
    """
    從資料庫讀取尚未上傳的資料（唯讀模式）
    
    參數：
        db_file (str)   : 資料庫檔案完整路徑
        since_str (str) : 起始時間點（只讀取此時間之後的資料）
        
    回傳值：
        tuple: (records, max_time)
            records (list) : 資料記錄列表（dict 格式）
            max_time (str) : 最後一筆資料的 CollectTime（用於更新狀態）
    
    注意：
        - 使用唯讀模式 (mode=ro) 避免影響 AI 寫入
        - 設定 10 秒 timeout，避免長期佔用鎖
        - 使用 LIMIT 限制最大讀取筆數 (MAX_FETCH_LIMIT) 防止 OOM
    """
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
                'CameraCode'   : row[0],   # 攝影機代碼
                'DeviceCode'   : row[1],   # 設備代碼
                'DetectClass'  : row[2],   # 偵測類別
                'TrackID'      : str(row[3]),  # 追蹤 ID
                'FromRoadID'   : row[4],   # 來源道路 ID
                'FromRoadName' : row[5],   # 來源道路名稱
                'ToRoadID'     : row[6],   # 目的地道路 ID
                'ToRoadName'   : row[7],   # 目的地道路名稱
                'Path'         : row[8],   # 路徑資訊
                'CollectTime'  : row[9],   # 收集時間
            })
            max_time = row[9]

        if records:
            logger.info(f"[{name}] 撈取 {len(records)} 筆 (起點: {since_str}, 終點: {max_time})")
            
    except sqlite3.Error as sql_err:
        logger.error(f"[{name}] 資料庫存取失敗: {sql_err}")
    finally:
        if conn:
            conn.close()
            
    return records, max_time

def cleanup_db(db_file, max_uploaded_time, retention_days):
    """
    分批刪除超過保留天數的舊資料（核心防禦機制）
    
    參數：
        db_file (str)            : 資料庫檔案完整路徑
        max_uploaded_time (str)  : 已成功上傳到的最新時間
        retention_days (int)     : 保留天數
    
    說明：
        - 每批刪除 1000 筆，立即 commit 釋放寫入鎖
        - 批次間 sleep 50ms，讓出 CPU 與 DB 鎖給 AI 寫入
        - 避免長時間佔用 DB 造成 AI 寫入阻塞
    """
    name = os.path.basename(db_file)
    conn = None
    try:
        # 計算清理的時間點：已上傳時間 - 保留天數
        dt_max = datetime.datetime.strptime(max_uploaded_time, '%Y-%m-%d %H:%M:%S')
        dt_cleanup = dt_max - datetime.timedelta(days=retention_days)
        cleanup_time_str = dt_cleanup.strftime('%Y-%m-%d %H:%M:%S')

        db_uri = f"{Path(db_file).as_uri()}"
        conn = sqlite3.connect(db_uri, uri=True, timeout=30)
        c = conn.cursor()
        
        total_deleted = 0
        batch_size = 1000   # 每批刪除 1000 筆
        delete_query = """
            DELETE FROM AiTrafficFlowRawData 
            WHERE rowid IN (
                SELECT rowid FROM AiTrafficFlowRawData 
                WHERE CollectTime <= ? 
                LIMIT ?
            )
        """
        while True:
            c.execute(delete_query, (cleanup_time_str, batch_size))
            deleted_count = c.rowcount
            total_deleted += deleted_count
            conn.commit()        # 立即提交，釋放鎖
            if deleted_count < batch_size:
                break
            time.sleep(0.05)     # 讓出資源

        if total_deleted > 0:
            logger.info(f"[{name}] 成功分批清理 {total_deleted} 筆舊資料 (保留 {retention_days} 天)")
            
    except Exception as e:
        logger.error(f"[{name}] 清理失敗: {e}")
    finally:
        if conn:
            conn.close()

def purge_history_data(db_files):
    """
    首次部署專用：刪除所有歷史包袱資料
    
    參數：
        db_files (list): 資料庫檔案路徑列表
    
    說明：
        - 刪除早於「現在往前推 5 分鐘」的所有資料
        - 採用分批刪除 (每批 2000 筆) 避免鎖死
        - 防止龐大歷史資料佔用 eMMC 空間
    """
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
                c.execute("""
                    DELETE FROM AiTrafficFlowRawData 
                    WHERE rowid IN (
                        SELECT rowid FROM AiTrafficFlowRawData 
                        WHERE CollectTime <= ? 
                        LIMIT 2000
                    )
                """, (since_str,))
                deleted_count = c.rowcount
                total_deleted += deleted_count
                conn.commit()
                if deleted_count < 2000:
                    break
                time.sleep(0.05)
                
            if total_deleted > 0:
                logger.info(f"[{name}] 首次部署清理：丟棄 {total_deleted} 筆歷史資料")
                
        except Exception as e:
            logger.error(f"[{name}] 首次清理失敗: {e}")
        finally:
            if conn:
                conn.close()


# ================= API 上傳 =================
def build_session():
    """
    建立 HTTP 會話（重用連線，關閉 SSL 驗證）
    
    回傳值：
        requests.Session: 設定好的會話物件
    """
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = False
    return session

def upload_batch(session, records):
    """
    將一批資料以 JSON 陣列形式上傳至 API
    
    參數：
        session (requests.Session) : HTTP 會話
        records (list)             : 要上傳的資料列表
        
    回傳值：
        bool: True 表示上傳成功，False 表示失敗
    
    說明：
        - 一次打包成單一 JSON 陣列
        - 支援重試機制（最多 MAX_RETRIES 次）
        - 4xx 錯誤不重試（用戶端錯誤），5xx 和網路錯誤會重試
    """
    total = len(records)
    json_data = json.dumps(records, ensure_ascii=False)
    
    for attempt in range(MAX_RETRIES + 1):
        token = token_manager.get_token()
        if not token:
            return False
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        try:
            logger.info(f"準備【一次打包】上傳，共 {total} 筆")
            response = session.post(API_URL, data=json_data, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.info(f"整批上傳成功，共 {total} 筆")
            return True
            
        except requests.exceptions.HTTPError as http_err:
            status = http_err.response.status_code if http_err.response is not None else None
            resp_text = http_err.response.text[:500] if http_err.response is not None else "無回應"
            
            if status is not None and 400 <= status < 500:
                # 4xx 錯誤不重試（例如認證失敗、資料格式錯誤）
                logger.error(f"整批上傳失敗 (HTTP {status}): {resp_text}")
                return False
                
            if attempt < MAX_RETRIES:
                wait_time = 1.0 * (attempt + 1)
                logger.warning(f"上傳暫時失敗 (HTTP {status})，{wait_time} 秒後重試伺服器回應: {resp_text}")
                time.sleep(wait_time)
                continue
                
            logger.error(f"整批上傳最終失敗 (HTTP {status})伺服器回應: {resp_text}")
            return False
            
        except requests.exceptions.RequestException as req_err:
            if attempt < MAX_RETRIES:
                wait_time = 1.0 * (attempt + 1)
                logger.warning(f"連線異常，{wait_time} 秒後重試: {req_err}")
                time.sleep(wait_time)
                continue
            logger.error(f"整批上傳最終失敗: {req_err}")
            return False
            
    return False


# ================= 主程式 =================
def main():
    """
    主執行流程（Crontab 相容版）
    
    執行順序：
    1. 取得檔案鎖（防止重疊執行）
    2. 掃描資料庫檔案
    3. 載入狀態，判斷是否首次部署
    4. 若是首次部署，清理歷史資料並立即儲存狀態
    5. 逐個 DB 處理：讀取新資料 → 一次上傳 → 成功後更新狀態並清理
    6. 統計成功/失敗筆數並記錄
    7. 儲存最終狀態，釋放鎖
    """
    # === 步驟 1：取得檔案鎖 ===
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOCK_FILENAME)
    lock_file = open(lock_path, 'w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        logger.warning("偵測到另一個執行個體正在運行，本次結束")
        lock_file.close()
        return

    try:
        # === 步驟 2：掃描資料庫檔案 ===
        db_files = sorted(glob.glob(os.path.join(DB_DIR, "*.db")))
        if not db_files:
            logger.warning(f"找不到任何 .db 檔: {DB_DIR}")
            return

        # === 步驟 3：載入狀態 ===
        state = load_state()
        is_first_run = not bool(state)   # 狀態為空表示首次執行

        # === 步驟 4：首次部署處理 ===
        if is_first_run:
            purge_history_data(db_files)
            # 清理後立即初始化狀態，避免下次重複清理
            for db_file in db_files:
                state[os.path.basename(db_file)] = compute_default_since_str()
            save_state(state)
            logger.info("首次部署清理完成，已初始化狀態檔")

        # === 步驟 5：建立 HTTP 會話 ===
        session = build_session()
        
        total_success = 0
        total_fail = 0
        
        # === 步驟 6：逐個 DB 處理 ===
        for db_file in db_files:
            name = os.path.basename(db_file)
            current_since = state.get(name, compute_default_since_str())
            
            # 讀取新資料
            records, max_time = fetch_rows(db_file, current_since)
            if not records:
                continue
            
            count = len(records)
            
            # 一次上傳整批資料
            if upload_batch(session, records):
                # 上傳成功：更新狀態並清理舊資料
                state[name] = max_time
                cleanup_db(db_file, max_time, RETENTION_DAYS)
                logger.info(f"[{name}] 上傳成功，狀態已更新，舊資料已清理")
                total_success += count
            else:
                logger.error(f"[{name}] 上傳失敗，等待下次排程重試")
                total_fail += count
                
        session.close()
        
        # === 步驟 7：儲存狀態並統計 ===
        save_state(state)
        
        total_all = total_success + total_fail
        if total_all > 0:
            logger.info(f"上傳完成：成功 {total_success} 筆，失敗 {total_fail} 筆，總共 {total_all} 筆")
        else:
            logger.info("上傳完成：本次無新資料需要上傳")
            
        logger.info("本次排程任務結束")

    finally:
        # 釋放檔案鎖
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def compute_default_since_str():
    """
    計算預設的查詢起點時間（當前時間往前推 5 分鐘，秒數歸零）
    
    回傳值：
        str: 格式為 'YYYY-MM-DD HH:MM:00' 的時間字串
    """
    dt = datetime.datetime.now().replace(microsecond=0) - datetime.timedelta(seconds=300)
    dt = dt.replace(second=0)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    """
    程式進入點
    """
    try:
        main()
        logger.info("排程執行結束\n" + "-" * 40)
    except Exception as e:
        # 捕捉所有未預期的錯誤，記錄詳細資訊
        logger.error(f"主程式執行時發生未預期的嚴重錯誤: {e}", exc_info=True)
    finally:
        # 只有最外層才使用 sys.exit 決定程式生命週期
        sys.exit(0)
