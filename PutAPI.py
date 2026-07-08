# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:11:15 2022
Updated: 整批打包上傳（改為單一 JSON 陣列一次送出）+ Session 連線重用 + 手動重試 + Bearer Token 驗證

@author: kuan
"""

import os
import sys
import glob
import time
import sqlite3
import json
import datetime
import urllib3
import logging
import threading
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ⭐ 要掃描的資料夾：DeepStream 會在這裡為每一路攝影機各產生一個 .db（camA.db、camB.db...）
DB_DIR = "/home/nvidia/THI/DeepStream-TMC/output_db"

# 中央伺服器 API 接口
API_URL  = "https://pingits.thix180server.com/pingits/api/AIData/AiTrafficFlowRawData"

# 登入取得 Token 的 API 接口
AUTH_URL      = "https://pingits.thix180server.com/pingits/api/Auth/login"
ENTERPRISE_ID = "THI"
USER_ID       = "AiAPI"
PASSWORD      = "Msaj#aV6Lh"   # 請替換為實際密碼

# Token 有效時間（秒）。伺服器 Token 存活 3 分鐘，提前 1 分鐘換，設 120 秒觸發刷新。
TOKEN_TTL = 120

# 單次批次請求逾時秒數（整批資料量較大，逾時設長一點）
REQUEST_TIMEOUT = 30

# 整批失敗後的最多重試次數（只對連線錯誤 / 5xx 重試；4xx 不重試）
MAX_RETRIES = 2

# ⭐ Log 檔名。為避免和「背景舊程式」搶同一個檔（造成權限不足或互相覆蓋），
#    測試期間建議用不同檔名；正式切換、舊程式停掉後再改回 PutAPI_run.log 也行。
LOG_FILENAME = "PutAPI_run_new.log"


# 自訂的由新至舊寫入 Handler
class PrependFileHandler(logging.Handler):
    def __init__(self, filename, max_lines=1000, encoding='utf-8'):
        super().__init__()
        self.filename = filename
        self.encoding = encoding
        self.max_lines = max_lines
        self.buffer = []

    def emit(self, record):
        # 收集本次執行的每一筆 log
        self.buffer.append(self.format(record) + '\n')

    def close(self):
        # 程式結束時，將新 log 排到舊 log 的最上方並寫回檔案
        if self.buffer:
            old_lines = []
            if os.path.exists(self.filename):
                try:
                    with open(self.filename, 'r', encoding=self.encoding) as f:
                        old_lines = f.readlines()
                except Exception:
                    pass

            # 將本次執行的 log 區塊放在最前面，但單次執行內保持正常時間順序 (較容易閱讀)
            new_lines = self.buffer

            all_lines = new_lines + old_lines
            # 最多保存 1000 行，避免邊緣設備長期執行撐爆記憶體或硬碟
            all_lines = all_lines[:self.max_lines]

            try:
                with open(self.filename, 'w', encoding=self.encoding) as f:
                    f.writelines(all_lines)
            except Exception as e:
                print(f"無法儲存 Log 到檔案 {self.filename}，請檢查權限: {e}", file=sys.stderr)
        super().close()


# 設定 Logging (同時輸出到檔案和命令列)
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        PrependFileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILENAME)),
        logging.StreamHandler(sys.stdout)
    ]
)


# ── Token 管理（執行緒安全，維持原設計；即使目前單執行緒也不影響）─────
class TokenManager:
    """
    集中管理 Bearer Token 的取得與刷新。
    使用 threading.Lock() 確保同一時間只有一個地方去打登入 API，不會重複刷新。
    """
    def __init__(self):
        self._token      = None
        self._token_time = None
        self._lock       = threading.Lock()

    def get_token(self):
        """
        回傳有效的 Token。
        若尚未取得、或已超過 TOKEN_TTL 秒，則重新向 Auth API 取得。
        """
        with self._lock:
            now = datetime.datetime.now()
            elapsed = (
                (now - self._token_time).total_seconds()
                if self._token_time else TOKEN_TTL + 1
            )

            if self._token is None or elapsed >= TOKEN_TTL:
                logging.info(
                    f"Token {'尚未取得' if self._token is None else f'已使用 {elapsed:.0f} 秒'}，"
                    f"重新取得 Token。"
                )
                self._token      = self._fetch_token()
                self._token_time = now if self._token else None

            return self._token

    def _fetch_token(self):
        """實際呼叫登入 API 取得 AccessToken。"""
        payload = {
            "EnterpriseId": ENTERPRISE_ID,
            "UserId":       USER_ID,
            "Password":     PASSWORD
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
                logging.error("登入失敗：帳號或密碼錯誤。")
                return None

            token = token_data.get("AccessToken")
            if not token:
                logging.error(f"登入成功但找不到 AccessToken，回傳內容: {token_data}")
                return None

            logging.info(f"成功取得 Token (UserId: {token_data['UserInfo']['UserId']})。")
            return token

        except requests.exceptions.RequestException as e:
            logging.error(f"取得 Token 失敗: {e}")
            return None


# 全域 TokenManager 實例
token_manager = TokenManager()


def build_session():
    """
    建立一條共用的 HTTP Session：重用 TCP 連線與 TLS 交握。
    """
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = False  # 沿用原本行為（自簽憑證）。若伺服器有正式憑證，建議改成 True。
    return session


def compute_since_str():
    """
    計算查詢起點時間：現在往前推 300 秒，並把秒數歸零（與原程式相同行為）。
    回傳格式: 'YYYY-MM-DD HH:MM:00'
    """
    dt = datetime.datetime.now().replace(microsecond=0) - datetime.timedelta(seconds=300)
    dt = dt.replace(second=0)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def fetch_rows(db_file, since_str):
    """
    第一階段：只負責「快速讀取」單一 db 的資料，撈完馬上關閉連線。
    回傳: list[dict]，每個 dict 就是一筆要上傳的資料。
    """
    name = os.path.basename(db_file)
    records = []
    conn = None
    try:
        logging.info(f"[{name}] 開始連接資料庫")
        # 以「唯讀」方式連線即可：不會動到 db、不影響 main.py 持續寫入。
        # 注意：不要用 immutable=1，會讀到過期/空白狀態。mode=ro 才正確。
        db_uri = f"{Path(db_file).as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=5)
        c = conn.cursor()

        jmode = c.execute("PRAGMA journal_mode;").fetchone()[0]
        logging.info(f"[{name}] journal_mode: {jmode}，查詢 CollectTime >= '{since_str}'")

        # 明確列出欄位（不要用 SELECT *），順序對齊中央 AiTrafficFlowRawData 規格。
        query = (
            "SELECT CameraCode, DeviceCode, DetectClass, TrackID, "
            "FromRoadID, FromRoadName, ToRoadID, ToRoadName, "
            "Path, CollectTime "
            "FROM AiTrafficFlowRawData WHERE CollectTime >= ?"
        )
        rows = c.execute(query, (since_str,)).fetchall()

        for row in rows:
            records.append({
                'CameraCode':   row[0],
                'DeviceCode':   row[1],
                'DetectClass':  row[2],
                'TrackID':      str(row[3]),
                'FromRoadID':   row[4],
                'FromRoadName': row[5],
                'ToRoadID':     row[6],
                'ToRoadName':   row[7],
                'Path':         row[8],
                'CollectTime':  row[9],
            })

        logging.info(f"[{name}] 查詢到 {len(records)} 筆資料")

    except sqlite3.Error as sql_err:
        logging.error(f"[{name}] 資料庫存取失敗: {sql_err}")
    except Exception as e:
        logging.error(f"[{name}] 發生未知的錯誤: {e}")
    finally:
        if conn:
            conn.close()
            logging.info(f"[{name}] 資料庫連線已關閉。")

    return records


def upload_batch(session, records):
    """
    將整批資料打包成單一 JSON 陣列，一次 POST 送出，含手動重試。
    - 4xx（例如 400）：重送也不會成功，直接記錄詳細原因並回 False。
    - 連線錯誤 / 逾時 / 5xx：重試 MAX_RETRIES 次。
    回傳: True 整批成功 / False 整批失敗。

    ⭐ 若中央 API 要求外層要包一個 key（例如 {"Data": [...]}），
      請把下方 json_data = json.dumps(records) 改成
      json_data = json.dumps({"Data": records})。
    """
    total = len(records)
    json_data = json.dumps(records, ensure_ascii=False)

    for attempt in range(MAX_RETRIES + 1):
        # 每次嘗試前向 TokenManager 取得 Token（過期時會自動刷新）
        token = token_manager.get_token()
        if not token:
            logging.error(f"無法取得 Token，中止本批 {total} 筆資料的上傳。")
            return False

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {token}"
        }
        try:
            logging.info(f"準備上傳整批資料，共 {total} 筆 → {API_URL}")
            response = session.post(API_URL, data=json_data, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            logging.info(f"整批上傳成功，共 {total} 筆。")
            return True

        except requests.exceptions.HTTPError as http_err:
            resp = http_err.response
            status = resp.status_code if resp is not None else None
            resp_body = resp.text if resp is not None else "(無回應內容)"

            # 4xx：客戶端資料問題，重送也不會成功，直接失敗
            if status is not None and 400 <= status < 500:
                logging.error(f"整批上傳失敗 (共 {total} 筆): HTTP {status}")
                logging.error(f"  ↳ 伺服器回應內容: {resp_body}")
                logging.error(f"  ↳ 我們送出的 JSON (前 500 字): {json_data[:500]}")
                return False

            # 5xx 或其他 HTTPError：還有重試次數就重試
            if attempt < MAX_RETRIES:
                logging.warning(
                    f"整批上傳暫時失敗 (HTTP {status})，準備重試第 {attempt + 1}/{MAX_RETRIES} 次"
                )
                logging.warning(f"  ↳ 伺服器回應內容: {resp_body}")
                time.sleep(0.5 * (attempt + 1))
                continue

            logging.error(f"整批上傳失敗(已重試 {MAX_RETRIES} 次) (共 {total} 筆): HTTP {status}")
            logging.error(f"  ↳ 伺服器回應內容: {resp_body}")
            logging.error(f"  ↳ 我們送出的 JSON (前 500 字): {json_data[:500]}")
            return False

        except requests.exceptions.RequestException as req_err:
            if attempt < MAX_RETRIES:
                logging.warning(
                    f"連線異常，準備重試第 {attempt + 1}/{MAX_RETRIES} 次: {req_err}"
                )
                time.sleep(0.5 * (attempt + 1))
                continue
            logging.error(f"整批上傳失敗(已重試 {MAX_RETRIES} 次) (共 {total} 筆): {req_err}")
            return False

    return False


def main():
    # 自動掃描 DB_DIR 底下所有 .db 檔（每路攝影機一個檔）
    db_files = sorted(glob.glob(os.path.join(DB_DIR, "*.db")))

    if not db_files:
        logging.warning(f"在 {DB_DIR} 找不到任何 .db 檔，本次無資料可送。請確認 DB_DIR 路徑是否正確。")
        return

    names = [os.path.basename(f) for f in db_files]
    logging.info(f"在 {DB_DIR} 找到 {len(db_files)} 個 db 檔：{names}")

    since_str = compute_since_str()

    # ===== 第一階段：快速讀取所有 db 的資料（本地操作，很快） =====
    all_records = []
    for db_file in db_files:
        logging.info(f"===== 讀取 {os.path.basename(db_file)} =====")
        all_records.extend(fetch_rows(db_file, since_str))

    total = len(all_records)
    if total == 0:
        logging.info("本次沒有需要上傳的資料。")
        return

    # ===== 第二階段：所有攝影機的資料打包成單一 JSON 陣列，一次送出 =====
    logging.info(f"準備將 {total} 筆資料打包成單一 JSON 一次上傳...")

    session = build_session()
    try:
        ok = upload_batch(session, all_records)
    finally:
        session.close()

    if ok:
        logging.info(f"上傳完成：成功 {total} 筆。")
    else:
        logging.info(f"上傳完成：失敗 {total} 筆。")


if __name__ == '__main__':
    try:
        main()
        logging.info("排程執行結束。\n" + "-" * 40)
    except Exception as e:
        logging.error(f"主程式執行時發生錯誤: {e}")
    finally:
        time.sleep(1)
        sys.exit(0)
