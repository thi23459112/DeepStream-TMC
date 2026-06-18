# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:11:15 2022

@author: kuan
"""

import os
import sys
import glob
import time
import sqlite3
import requests
import json
import datetime
import urllib3
import logging

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ⭐ 要掃描的資料夾：DeepStream 會在這裡為每一路攝影機各產生一個 .db（camA.db、camB.db...）
DB_DIR = "/home/nvidia/DeepStream-TMC/output_db"

# 中央伺服器 API 接口
API_URL = "https://x235aiapi.thix180server.com:4004/AIDetect_Track_detection_stats"


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
                import sys
                print(f"無法儲存 Log 到檔案 {self.filename}，請檢查權限: {e}", file=sys.stderr)
        super().close()

# 設定 Logging (同時輸出到檔案和命令列)
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        PrependFileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PutAPI_run.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

def putdata(NdbFile):

    DTL5 = datetime.datetime.strptime(str(datetime.datetime.now()).split(".")[0], '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=-300)
    # DTL5 = datetime.datetime.strptime('2024-09-13 16:00:00', '%Y-%m-%d %H:%M:%S') + datetime.timedelta(seconds=-300)
    D = str(DTL5).split(" ")[0]
    Ts = str(DTL5).split(" ")[1].split(":")
    T = Ts[0] + ":" + Ts[1] + ":00"

    DTL5r = D + " " + T

    conn = None
    try:
        logging.info(f"開始連接資料庫: {NdbFile}")
        from pathlib import Path
        # 以「唯讀」方式連線即可：不會動到 db、不影響 main.py 持續寫入。
        # 注意：不要再用 immutable=1。immutable 會把檔案當成「永不變動的靜止檔」，
        #       但這些 db 正被 main.py 即時寫入，immutable 會讀到過期/空白狀態，
        #       導致連表都查不到（no such table）。mode=ro 才會讀當下真實內容。
        db_uri = f"{Path(NdbFile).as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True)
        c = conn.cursor()
        # 診斷用：印出 journal_mode，確認是否為 WAL 模式
        jmode = c.execute("PRAGMA journal_mode;").fetchone()[0]
        logging.info(f"資料庫 journal_mode: {jmode}")
        logging.info(f"執行查詢，尋找 CollectTime >= '{DTL5r}' 的資料")

        # 明確列出要撈的欄位（不要用 SELECT *）：
        # 1. 不撈不需要的 RoiCount / VideoTime
        # 2. 欄位順序對齊中央 AiTrafficFlowRawData 規格，row[0]~row[9] 位置固定可靠
        query = (
            "SELECT CameraCode, DeviceCode, DetectClass, TrackID, "
            "FromRoadID, FromRoadName, ToRoadID, ToRoadName, "
            "Path, CollectTime "
            "FROM AiTrafficFlowRawData WHERE CollectTime >= ?"
        )
        cursor = c.execute(query, (DTL5r,))

        row_count = 0
        success_count = 0

        for row in cursor:
            row_count += 1
            data = {
                'CameraCode':   row[0],
                'DeviceCode':   row[1],
                'DetectClass':  row[2],
                'TrackID':      row[3],
                'FromRoadID':   row[4],
                'FromRoadName': row[5],
                'ToRoadID':     row[6],
                'ToRoadName':   row[7],
                'Path':         row[8],
                'CollectTime':  row[9],
            }

            json_data = json.dumps(data)
            headers = {"Content-Type": "application/json"}

            try:
                response = requests.put(API_URL, data=json_data, headers=headers, verify=False, timeout=10)
                response.raise_for_status()  # 檢查是否為 4xx 或 5xx 的錯誤狀態碼
                success_count += 1
            except requests.exceptions.RequestException as req_err:
                logging.error(f"上傳資料失敗 (DeviceCode: {row[1]}, CollectTime: {row[9]}): {req_err}")
                # 額外印出伺服器回應的詳細內容（400 的真正原因通常在這裡）
                resp = getattr(req_err, "response", None)
                if resp is not None:
                    logging.error(f"  ↳ 伺服器狀態碼: {resp.status_code}")
                    logging.error(f"  ↳ 伺服器回應內容: {resp.text}")
                    logging.error(f"  ↳ 我們送出的 JSON: {json_data}")

        logging.info(f"資料處理完成: 總共查詢到 {row_count} 筆資料，成功上傳 {success_count} 筆。")

    except sqlite3.Error as sql_err:
        logging.error(f"資料庫存取失敗: {sql_err}")
    except Exception as e:
        logging.error(f"發生未知的錯誤: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("資料庫連線已關閉。")


def main():
    # 自動掃描 DB_DIR 底下所有 .db 檔，逐一處理（每路攝影機一個檔）
    db_files = sorted(glob.glob(os.path.join(DB_DIR, "*.db")))

    if not db_files:
        logging.warning(f"在 {DB_DIR} 找不到任何 .db 檔，本次無資料可送。請確認 DB_DIR 路徑是否正確。")
        return

    names = [os.path.basename(f) for f in db_files]
    logging.info(f"在 {DB_DIR} 找到 {len(db_files)} 個 db 檔：{names}")

    for db_file in db_files:
        logging.info(f"===== 開始處理 {os.path.basename(db_file)} =====")
        putdata(db_file)


if __name__ == '__main__':
    try:
        main()
        logging.info("排程執行結束。\n" + "-"*40)
    except Exception as e:
        logging.error(f"主程式執行時發生錯誤: {e}")
    finally:
        time.sleep(1)
        sys.exit(0)