"""
SQLite 事件紀錄與軌跡狀態管理（轉向量 O-D 版）

主要功能：
1. 多 cam 獨立 SQLite DB：每路 cam 一個 .db 檔，避免寫入衝突
2. 軌跡狀態維護：每台車一份 track_history，記錄位置、ROI 命中次數、
   各 ROI 首次命中的 frame 編號（O-D 先後排序用）、車種投票
3. 消失時結算：物件連續 N 幀未出現 → 計算來向 (Origin) / 去向 (Destination)，
   一台車最多寫一筆 O-D 紀錄
   - 只看命中數 >= min_roi_hits 的「有效 ROI」（過濾擦邊噪音）
   - 依各 ROI 首次命中 frame 排序：最早進入 = 來向、最後進入 = 去向
   - 中間經過的 ROI 不入欄位，僅保留在 Path 供分析
4. 結算過濾（取代舊的 Y 軸方向過濾）：
   - 有效 ROI 少於 2 個（只經過一處）→ 不寫 DB
   - 來向 == 去向（折返 / 繞回原 ROI）→ 不寫 DB
5. 批次 flush 機制：累積在記憶體 pending_records，定期批次寫入 DB
6. save_output_db=false 旗標：純跑統計、不開連線、零 DB IO
7. local_id 循環機制：每路 cam 累積到 LOCAL_ID_MAX 後歸 1 重新計算
"""

import os
import time
import sqlite3
import threading
from datetime import timedelta

from logic.config import SOURCE_CONFIGS, LOCAL_ID_MAX
from logic.color import CLASS_MAP


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# --- 全域狀態字典 (供 probes.py 直接 import 使用) ---
track_history    = {}    # (pad_index, obj_id) → 軌跡狀態 dict
pending_records  = {}    # pad_index → 待寫入 DB 的 tuple list
last_flush_times = {}    # pad_index → 上次 flush 的時間戳
fps_streams      = {}    # pad_index → {"current_fps", "timestamps"}
local_id_maps    = {}    # pad_index → {global_id: local_id}
next_local_ids   = {}    # pad_index → 下一個可分配的 local_id（達 LOCAL_ID_MAX 後歸 1）

# --- SQLite 連線管理 ---
_db_conns = {}                    # pad_index → sqlite3.Connection
_db_lock  = threading.Lock()      # 寫入批次的執行緒鎖

# --- DB Schema ---
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS AiTrafficFlowRawData (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    DeviceCode   TEXT    NOT NULL,
    CameraCode   TEXT    NOT NULL,
    TrackID      INTEGER NOT NULL,
    DetectClass  TEXT,
    FromRoadID   TEXT    NOT NULL,
    FromRoadName TEXT,
    ToRoadID     TEXT    NOT NULL,
    ToRoadName   TEXT,
    Path         TEXT,
    VideoTime    TEXT,
    CollectTime  TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_camera_time
    ON AiTrafficFlowRawData (CameraCode, CollectTime);

CREATE INDEX IF NOT EXISTS idx_od
    ON AiTrafficFlowRawData (FromRoadID, ToRoadID);
"""

# ==========================================
# 2. DB 連線輔助 (Connection Helper)
# ==========================================

def _get_db_path(cfg, pad_index):
    """
    從 cfg["excel_path"] 推算 DB 路徑（向下相容鍵名）

    參數：
        cfg (dict): 該路 cam 的 YAML 設定
        pad_index (int): 哪一路 cam

    返回：
        str: .db 檔絕對路徑
    """
    excel_path = cfg.get("excel_path", f"output_db/cam_{pad_index}.db")
    base, _ = os.path.splitext(excel_path)
    return f"{base}.db"


def _open_db(pad_index, cfg):
    """
    為指定 cam 開啟 SQLite 連線並建立 schema

    使用 WAL 模式提升併發寫入效能，synchronous=NORMAL 兼顧速度與資料安全

    參數：
        pad_index (int): 哪一路 cam
        cfg (dict): 該路 cam 的 YAML 設定

    返回：
        sqlite3.Connection: 已建立 schema 的 DB 連線
    """
    db_path = _get_db_path(cfg, pad_index)
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_SCHEMA_SQL)
    print(f"[INFO] SQLite DB 開啟: {db_path}")
    return conn


def _format_video_time(vsec):
    """
    秒數轉成 HH:MM:SS 字串

    參數：
        vsec (float): 秒數

    返回：
        str: "HH:MM:SS" 格式；負值或 None 回傳 "00:00:00"
    """
    if vsec is None or vsec < 0:
        return "00:00:00"
    return time.strftime("%H:%M:%S", time.gmtime(int(vsec)))


# ==========================================
# 3. 啟動初始化 (Startup Initialization)
# ==========================================

def initialize_state_managers():
    """
    為每一路 cam 初始化狀態字典與 DB 連線

    處理流程：
    1. 為每路 cam 初始化所有狀態字典（pending / fps / local_id 等）
    2. 依 cfg["save_output_db"] 決定是否開 DB 連線
       - true（預設）：呼叫 _open_db 建立連線
       - false       ：跳過連線開啟，emit/flush 走 no-op 分支

    註：本函式應在 main.py 啟動時呼叫一次
    """
    for pad_index, cfg in SOURCE_CONFIGS.items():
        # 步驟 1: 狀態字典初始化
        pending_records[pad_index] = []
        last_flush_times[pad_index] = time.time()
        fps_streams[pad_index] = {"current_fps": 0.0}
        local_id_maps[pad_index] = {}
        next_local_ids[pad_index] = 1

        # 步驟 2: 依旗標決定是否開 DB
        if cfg.get("save_output_db", True):
            _db_conns[pad_index] = _open_db(pad_index, cfg)
        else:
            cam_name = cfg.get("source_id", f"cam_{pad_index}")
            print(f"[INFO] {cam_name} save_output_db=false，停用 DB 寫入（純跑統計）")


# ==========================================
# 4. ID 管理 (ID Mapping)
# ==========================================

def get_local_id(pad_index, global_id):
    """
    將追蹤器給的 global_id 映射成該路 cam 內的短 local_id

    循環機制：local_id 從 1 累加到 LOCAL_ID_MAX，下一個 global_id 拿到的會是 1
              （每路 cam 各自獨立循環，互不干擾）
              撞號的紀錄靠 DB CreateTime 區分，查詢時記得帶時間範圍

    參數：
        pad_index (int): 哪一路 cam
        global_id (int): 追蹤器給的物件 ID

    返回：
        int: 該路內遞增的短 ID（範圍 1 ~ LOCAL_ID_MAX，達上限後歸 1）
    """
    if global_id not in local_id_maps[pad_index]:
        local_id_maps[pad_index][global_id] = next_local_ids[pad_index]

        # 達上限歸 1，否則 +1
        if next_local_ids[pad_index] >= LOCAL_ID_MAX:
            cam_name = SOURCE_CONFIGS.get(pad_index, {}).get("source_id", f"cam_{pad_index}")
            print(f"[INFO] {cam_name} local_id 達上限 {LOCAL_ID_MAX}，下一個歸 1 重新計算")
            next_local_ids[pad_index] = 1
        else:
            next_local_ids[pad_index] += 1

    return local_id_maps[pad_index][global_id]


# ==========================================
# 5. 軌跡結算 (Trajectory Finalization)
# ==========================================

def _finalize_one(m_key, state, force=False):
    """
    結算單一車輛軌跡，產生「轉向量 (O-D)」紀錄：來向路段 → 去向路段

    結算規則（轉向量版）：
    1. 只看「有效命中」的 ROI：命中數 >= min_roi_hits（低於視為短暫飄過，不算數）
    2. 有效 ROI 少於 2 個 → 只經過一處或沒經過，不構成轉向 → 整筆丟掉
    3. 依各 ROI「第一次命中的 frame 編號」排序：
         來向 (FromRoadID)  = 最早進入的 ROI
         去向 (ToRoadID)    = 最後進入的 ROI
       中間經過的 ROI 不寫入欄位（保留在 Path 供分析）
    4. 來向 == 去向（折返 / 繞回原 ROI）→ 不記錄
    5. FromRoadName / ToRoadName：依 cfg["road_names"] 把 ROI id 換成路段名稱
       （查不到對照時，以 ROI id 當名稱 fallback）
    6. 已移除 Y 軸 IN/OUT/NA 方向過濾（轉向多為水平移動，沿用會誤殺）

    每筆紀錄欄位：
        DeviceCode / CameraCode / TrackID / DetectClass /
        FromRoadID / FromRoadName / ToRoadID / ToRoadName /
        Path / RoiCount / VideoTime / CollectTime

    參數：
        m_key (tuple): (pad_index, obj_id)
        state (dict): 該軌跡的狀態字典
        force (bool): 是否為強制結算（程式結束時用，影響 log 標籤）
    """
    pad_index, obj_id = m_key
    cfg = SOURCE_CONFIGS.get(pad_index, {})
    cam_name = cfg.get("source_id", f"cam_{pad_index}")
    min_hits = cfg.get("track_logic", {}).get("min_roi_hits", 2)

    # 步驟 1: 找出所有「有效命中」的 ROI（命中數 >= min_roi_hits）
    valid_rois = {
        roi_name: hits
        for roi_name, hits in state.get("roi_hits", {}).items()
        if hits >= min_hits
    }

    # 步驟 2: 有效 ROI 少於 2 個 → 不構成轉向，整筆丟掉
    if len(valid_rois) < 2:
        return

    # 步驟 3: 依「第一次命中的 frame 編號」排序，取頭尾當來向/去向
    roi_first_frame = state.get("roi_first_frame", {})
    ordered = sorted(valid_rois.keys(), key=lambda r: roi_first_frame.get(r, 0))
    origin = ordered[0]
    destination = ordered[-1]

    # 步驟 4: 來向 == 去向（折返）→ 不記錄
    if origin == destination:
        return

    # 步驟 5: ROI id → 路段名稱（查不到就用 ROI id 當 fallback）
    road_names = cfg.get("road_names", {})
    origin_name = road_names.get(origin, origin)
    dest_name = road_names.get(destination, destination)

    # 步驟 6: 共用欄位計算
    local_id = get_local_id(pad_index, obj_id)
    device_code = cfg.get("device_code", "UNKNOWN")
    path_str = ">".join(ordered)        # 完整經過順序（含中間 ROI），供除錯 / 進階分析
    roi_count = len(ordered)

    # 車種投票：取票數最多的類別
    if state.get("class_votes"):
        best_class_id = state["class_votes"].most_common(1)[0][0]
        cls_name = CLASS_MAP.get(best_class_id, f"Class_{best_class_id}")
    else:
        cls_name = "Unknown"

    # VideoTime：軌跡最後出現幀號 → 影片內秒數
    vsec = state["last_frame_num"] / cfg.get("stream_fps", 30.0)
    time_axis = _format_video_time(vsec)

    # CollectTime：檔案模式 = start_time + vsec；即時串流 = 系統當下時間
    start_dt = cfg.get("start_time_dt")
    if start_dt is not None:
        event_dt = start_dt + timedelta(seconds=vsec)
        create_time_str = event_dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        create_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

    tag = "[結算-強制]" if force else " "

    # 步驟 7: save_output_db=false → 只印 log，不累積、不寫 DB
    if not cfg.get("save_output_db", True):
        print(f"{tag}[{cam_name}] ID={local_id}, 車種={cls_name}, "
              f"來向={origin}-{origin_name}, 去向={destination}-{dest_name}, "
              f"路徑={path_str}, 時間軸={time_axis}, 時間點={create_time_str}  (DB 已停用)")
        return

    # 步驟 8: 一台車一筆 O-D 紀錄
    pending_records[pad_index].append((
        device_code,
        cam_name,
        local_id,
        cls_name,
        origin,          # FromRoadID
        origin_name,     # FromRoadName
        destination,     # ToRoadID
        dest_name,       # ToRoadName
        path_str,
        # roi_count,
        time_axis,
        create_time_str,
    ))

    print(f"{tag}[{cam_name}] ID={local_id}, 車種={cls_name}, "
          f"來向={origin}-{origin_name}, 去向={destination}-{dest_name}, "
          f"路徑={path_str}, 時間軸={time_axis}, 時間點={create_time_str}")

# ==========================================
# 6. DB 寫入 (DB Flush)
# ==========================================

def flush_pending_to_db(pad_index):
    """
    把 pending_records[pad_index] 批次寫入 SQLite

    使用單一 transaction (BEGIN/COMMIT) 提升寫入效能；
    失敗時 ROLLBACK，pending 保留在記憶體等下次重試

    參數：
        pad_index (int): 哪一路 cam

    返回：
        int: 實際寫入筆數；無 pending 或無連線回傳 0
    """
    records = pending_records.get(pad_index, [])
    if not records:
        return 0

    conn = _db_conns.get(pad_index)
    if conn is None:
        # save_output_db=false 模式下不會走到這（_finalize_one 早就跳過 append）
        # 萬一有殘留 records 也清掉，避免 memory leak
        records.clear()
        return 0

    with _db_lock:
        try:
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT INTO AiTrafficFlowRawData "
                "(DeviceCode, CameraCode, TrackID, DetectClass, FromRoadID, FromRoadName, ToRoadID, ToRoadName, Path, VideoTime, CollectTime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                records
            )
            conn.execute("COMMIT")
            n = len(records)
            records.clear()
            return n
        except sqlite3.Error as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            print(f"[ERROR] SQLite 寫入失敗 (pad_index={pad_index}): {e}")
            return 0


# ==========================================
# 7. 結束清理 (Shutdown Cleanup)
# ==========================================

def force_finalize_all():
    """
    程式結束前呼叫：強制結算所有殘留軌跡、flush 剩餘 pending、關閉所有 DB

    處理流程：
    1. 對所有 track_history 內殘留的軌跡呼叫 _finalize_one(force=True)
       （這些車是程式結束時還在畫面內、還沒消失到 cleanup_frames 的）
    2. 對每路 cam 強制 flush 一次，確保 pending 都進 DB
    3. 關閉所有 DB 連線（WAL checkpoint 也會跟著做）
    4. 清空 track_history 釋放記憶體
    """
    print("\n[INFO] 開始執行強制結算...")

    # 步驟 1: 殘留軌跡逐一結算
    for m_key, state in list(track_history.items()):
        _finalize_one(m_key, state, force=True)

    # 步驟 2: 強制 flush 所有 pending
    for pad_index, cfg in SOURCE_CONFIGS.items():
        n = flush_pending_to_db(pad_index)
        if n > 0:
            db_path = _get_db_path(cfg, pad_index)
            print(f"[檔案儲存] {cfg.get('source_id')}：已強制寫入 {n} 筆剩餘資料到 {db_path}")

    # 步驟 3: 關閉所有 DB 連線
    for pad_index, conn in list(_db_conns.items()):
        try:
            conn.close()
        except Exception as e:
            print(f"[WARNING] 關閉 DB 連線失敗 (pad_index={pad_index}): {e}")
    _db_conns.clear()

    # 步驟 4: 釋放記憶體
    track_history.clear()