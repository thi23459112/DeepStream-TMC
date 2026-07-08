"""
DeepStream Probe 探針集合（轉向量 O-D 版）

主要功能：
1. 兩種追蹤器模式共用同一份軌跡狀態邏輯：
   - NvDCF (nvdcf)  → tracker_src_pad_buffer_probe，掛在 tracker.src
   - BoxMOT 系列    → boxmot_pgie_src_probe，掛在 pgie.src
2. BoxMOT 模式接管偵測：清空 PGIE obj_meta → 餵給 BoxMOT → 用追蹤結果重建 obj_meta
3. 車輛軌跡狀態維護：多 ROI 命中累積、各 ROI 首次命中的 frame 編號
   （供 O-D 來向/去向先後排序）、車種投票
4. 消失時結算：物件 missing_frames 達門檻 → 呼叫 _finalize_one 計算
   來向 (Origin) / 去向 (Destination)，一台車最多寫一筆 O-D 紀錄
5. OSD 視覺化：bbox 用車種色（不變紅）、ID 標籤、左上角即時 FPS 顯示
"""

import time
import cv2
import numpy as np
from collections import Counter, deque
from gi.repository import Gst
import pyds

from logic.color import get_class_color, CLASS_MAP
from logic.config import SOURCE_CONFIGS
from logic.state_db import (
    get_local_id, _finalize_one, flush_pending_to_db,
    track_history, last_flush_times,
    fps_streams, local_id_maps
)


# ==========================================
# 1. 系統配置區 (System Configuration)
# ==========================================

# --- 模組執行期狀態 ---
g_last_fps_print_time = time.time()          # 上次印 FPS 報告的時間戳


# ==========================================
# 2. 共用核心邏輯 (Shared Tracking Logic)
# ==========================================

def _process_tracked_frame(frame_meta, current_frame_objects, pad_index, cfg):
    """
    遍歷 frame_meta 內所有 obj_meta，維護軌跡狀態、調整 OSD 顯示

    處理流程：
    1. 過濾追蹤器輸出與無效 ID
    2. 對每個物件計算 bbox 底部中心點 (cx, cy)
    3. 首次出現 → 初始化軌跡狀態（記下 start_y、初始化各欄位）
    4. 多 ROI 命中判斷 → 累加 roi_hits[roi_name] + class_votes
    5. 方向判斷（軌跡共用）：用 cy 與 start_y 的 Y 軸位移判斷 IN/OUT
    6. OSD 視覺化：bbox 永遠用車種色（不變紅）、ID 標籤

    參數：
        frame_meta (pyds.NvDsFrameMeta): 當前幀的 meta
        current_frame_objects (set): 本幀出現的 (pad_index, obj_id) 集合
        pad_index (int): 哪一路 cam
        cfg (dict): 該路 cam 的 YAML 設定

    註：呼叫方需先確保 obj_meta 已整理好（object_id 有效、rect_params 是要顯示的框）
    """
    cv_regions = cfg.get("cv_regions", {})
    movement_threshold = cfg.get("track_logic", {}).get("movement_threshold", 30)

    l_obj = frame_meta.obj_meta_list
    while l_obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
        except StopIteration:
            break

        # 步驟 1: 只處理追蹤器輸出的物件
        # nvdcf 模式：nvtracker 把追蹤結果標 unique_component_id=1
        # boxmot 模式：我們在 boxmot_pgie_src_probe 重建 obj_meta 時也設 =1
        if obj_meta.unique_component_id != 1:
            l_obj = l_obj.next
            continue

        obj_id = obj_meta.object_id
        if obj_id == -1:
            l_obj = l_obj.next
            continue

        unique_key = (pad_index, obj_id)
        current_frame_objects.add(unique_key)
        local_id = get_local_id(pad_index, obj_id)

        # 步驟 2: 計算 bbox 底部中心點
        cx = int(obj_meta.rect_params.left + (obj_meta.rect_params.width / 2))
        cy = int(obj_meta.rect_params.top + obj_meta.rect_params.height)

        # 步驟 3: 初始化軌跡狀態（首次出現）
        if unique_key not in track_history:
            track_history[unique_key] = {
                "start_y":         cy,                    # 起始 Y，判斷方向用
                "missing_frames":  0,
                "direction":       "NA",                  # IN / OUT / NA
                "class_votes":     Counter(),             # 車種投票
                "last_frame_num":  frame_meta.frame_num,
                "last_v_box":      None,                  # 最後 bbox（沿用原版欄位）
                "roi_hits":        {},                    # 多 ROI 各自累計 {roi_name: count}
                "roi_first_frame": {},                    # 每個 ROI「第一次命中」的 frame 編號（O-D 排序用）
            }

        state = track_history[unique_key]
        state["missing_frames"] = 0
        state["last_frame_num"] = frame_meta.frame_num

        # 紀錄最新的 bbox（沿用原版欄位，對效能影響極小）
        r = obj_meta.rect_params
        state["last_v_box"] = (
            float(r.left),
            float(r.top),
            float(r.left + r.width),
            float(r.top + r.height),
        )

        # 步驟 4: 多 ROI 命中判斷（命中時累加 roi_hits、class_votes、首次命中 frame）
        for roi_name, polygon in cv_regions.items():
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                state["roi_hits"][roi_name] = state["roi_hits"].get(roi_name, 0) + 1
                state["class_votes"][obj_meta.class_id] += 1
                # 記下第一次命中該 ROI 的 frame 編號（用於 O-D 來向/去向先後排序）
                if roi_name not in state["roi_first_frame"]:
                    state["roi_first_frame"][roi_name] = frame_meta.frame_num

        # 步驟 5: 方向判斷（軌跡共用，沿用原版邏輯）
        # 首次定向後就固定，不再翻轉（避免抖動造成方向反覆切換）
        if state["direction"] == "NA":
            dy = cy - state["start_y"]
            if dy > movement_threshold:
                state["direction"] = "IN"      # Y 增加 → 向下移動
            elif dy < -movement_threshold:
                state["direction"] = "OUT"     # Y 減少 → 向上移動

        # 步驟 6: OSD 視覺化（bbox 永遠用車種色，不變紅）
        cls_id = obj_meta.class_id
        cls_name = CLASS_MAP.get(cls_id, f"Class_{cls_id}")
        color = get_class_color(cls_id)

        r.border_width = 4
        r.border_color.set(*color)
        r.has_bg_color = 0

        txt = obj_meta.text_params
        txt.display_text = f"ID:{local_id} {cls_name}"
        txt.font_params.font_name = "Serif Bold"
        txt.font_params.font_size = 14
        txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        txt.set_bg_clr = 1
        txt.text_bg_clr.set(*color)

        text_h = int(14 * 1.4)
        txt.x_offset = max(0, int(r.left) + 0)
        txt.y_offset = max(0, int(r.top + r.height) - text_h - 10)

        l_obj = l_obj.next


def _post_frame_housekeeping(current_frame_objects):
    """
    每幀結束時的清理工作（兩種模式共用）

    處理流程：
    1. 偵測消失的軌跡（連續 missing_frames >= cleanup_frames）
       → 呼叫 _finalize_one 結算 → 從 track_history 移除
    2. 每 30 秒印一次 FPS 效能報告
    3. 依 flush_interval_seconds 定期把 pending 寫進 SQLite DB

    參數：
        current_frame_objects (set): 本幀出現的 (pad_index, obj_id) 集合
    """
    global g_last_fps_print_time

    # 步驟 1: 消失軌跡 → 結算 → 清理
    missing_keys = set(track_history.keys()) - current_frame_objects
    for m_key in missing_keys:
        pad_index, obj_id = m_key
        cfg = SOURCE_CONFIGS.get(pad_index, {})
        track_history[m_key]["missing_frames"] += 1
        cleanup_frames = cfg.get("session", {}).get("cleanup_frames", 30)

        if track_history[m_key]["missing_frames"] >= cleanup_frames:
            # ⭐ 結算後才刪
            _finalize_one(m_key, track_history[m_key], force=False)
            del track_history[m_key]
            if obj_id in local_id_maps[pad_index]:
                del local_id_maps[pad_index][obj_id]

    # 步驟 2: 每 30 秒印 FPS
    current_time = time.time()
    if current_time - g_last_fps_print_time >= 30:
        print("\n" + "=" * 35)
        print(f"[{time.strftime('%H:%M:%S')}] 即時處理效能報告 (FPS)：")
        for sid, stats in sorted(fps_streams.items()):
            c_name = SOURCE_CONFIGS[sid].get("source_id", f"cam_{sid}")
            print(f" • {c_name.ljust(10)}: {stats['current_fps']:.2f} FPS")
        print("=" * 35 + "\n")
        g_last_fps_print_time = current_time

    # 步驟 3: 定期 flush 到 SQLite DB
    for pad_index, cfg in SOURCE_CONFIGS.items():
        flush_interval = cfg.get("session", {}).get("flush_interval_seconds", 30)
        if current_time - last_flush_times[pad_index] >= flush_interval:
            flush_pending_to_db(pad_index)
            last_flush_times[pad_index] = current_time


def _update_fps(pad_index):
    """
    更新指定 pad 的即時 FPS 統計（兩種模式共用）

    使用滑動視窗（30 幀）計算瞬時 FPS

    參數：
        pad_index (int): 哪一路 cam
    """
    if "timestamps" not in fps_streams[pad_index]:
        fps_streams[pad_index]["timestamps"] = deque(maxlen=30)
    now = time.time()
    q = fps_streams[pad_index]["timestamps"]
    q.append(now)
    if len(q) > 1:
        fps_streams[pad_index]["current_fps"] = (len(q) - 1) / (q[-1] - q[0])


# ==========================================
# 3. NvDCF 模式探針 (NvDCF Tracker Probe)
# ==========================================

def tracker_src_pad_buffer_probe(pad, info, u_data):
    """
    NvDCF 模式專用探針：掛在 tracker.src

    obj_meta 由 nvtracker 提供，已包含有效的 object_id；
    本探針只負責 FPS 統計 + 軌跡狀態更新 + 收尾清理。

    參數：
        pad, info, u_data: GStreamer probe 標準參數

    返回：
        Gst.PadProbeReturn.OK
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        pad_index = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_index)
        if not cfg:
            l_frame = l_frame.next
            continue

        _update_fps(pad_index)
        _process_tracked_frame(frame_meta, current_frame_objects, pad_index, cfg)

        l_frame = l_frame.next

    _post_frame_housekeeping(current_frame_objects)
    return Gst.PadProbeReturn.OK


# ==========================================
# 4. BoxMOT 模式探針 (BoxMOT Tracker Probe)
# ==========================================

def boxmot_pgie_src_probe(pad, info, u_data):
    """
    BoxMOT 模式專用探針：掛在 pgie.src

    處理流程：
    1. 從 obj_meta 抽出所有 PGIE 偵測框（全車種都收，不過濾）
    2. 清空 frame_meta 內所有 PGIE obj_meta（回到 pool）
    3. 把偵測框餵給 BoxMOT，拿回追蹤結果（含 id、可能不同的框）
    4. 用 BoxMOT 輸出重建 obj_meta（框、id、conf、class）
    5. 走和 NvDCF 模式同一份的軌跡狀態邏輯

    重要前提：本探針必須在 pgie.src，下游不可有 nvtracker
              （否則 nvtracker 會覆寫掉我們重建的 meta）

    參數：
        pad, info, u_data: GStreamer probe 標準參數

    返回：
        Gst.PadProbeReturn.OK
    """
    # lazy import 避免 nvdcf 模式啟動時也載入 boxmot
    from logic.boxmot_adapter import track as boxmot_track

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    current_frame_objects = set()

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        pad_index = frame_meta.pad_index
        cfg = SOURCE_CONFIGS.get(pad_index)
        if not cfg:
            l_frame = l_frame.next
            continue

        _update_fps(pad_index)

        # 步驟 1: 抽出所有 PGIE 偵測框（車流版全車種都收，不做類別過濾）
        dets_list = []
        obj_metas_to_remove = []   # 先蒐集要刪的，等迴圈結束再 remove 才安全

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # pgie.src 用 detector_bbox_info 最準確（nvtracker 介入後才有 tracker_bbox_info）
            try:
                det_box = obj_meta.detector_bbox_info.org_bbox_coords
                x1 = float(det_box.left)
                y1 = float(det_box.top)
                x2 = float(det_box.left + det_box.width)
                y2 = float(det_box.top + det_box.height)
            except Exception:
                # 萬一拿不到 detector_bbox_info，退回 rect_params
                r = obj_meta.rect_params
                x1 = float(r.left)
                y1 = float(r.top)
                x2 = float(r.left + r.width)
                y2 = float(r.top + r.height)

            conf = float(obj_meta.confidence) if obj_meta.confidence > 0 else 0.5
            cls = int(obj_meta.class_id)

            # 車流版：所有車種都收（不像行人版只留 person）
            dets_list.append([x1, y1, x2, y2, conf, cls])

            obj_metas_to_remove.append(obj_meta)
            l_obj = l_obj.next

        # 步驟 2: 清空 frame_meta 內所有 obj_meta
        # 從 frame 移除後 obj_meta 自動回 pool，下面重新申請即可
        for om in obj_metas_to_remove:
            pyds.nvds_remove_obj_meta_from_frame(frame_meta, om)

        # 步驟 3: 餵給 BoxMOT 取得追蹤結果
        if dets_list:
            dets = np.asarray(dets_list, dtype=np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        # A/B 級追蹤器不需要 frame；C 級才需要從 NVMM 拷貝
        tracks = boxmot_track(pad_index, dets, frame=None)

        # 步驟 4: 用 BoxMOT 輸出重建 obj_meta
        # tracks 格式：[x1, y1, x2, y2, id, conf, cls, det_ind]，shape=(M, 8)
        for tr in tracks:
            x1, y1, x2, y2 = float(tr[0]), float(tr[1]), float(tr[2]), float(tr[3])
            tid = int(tr[4])
            conf = float(tr[5])
            cls = int(tr[6])

            new_obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            if new_obj is None:
                # pool 滿了就跳過這個 track（極少發生）
                continue

            # unique_component_id=1 與 nvtracker 預設一致，讓下游邏輯不需要分模式判斷
            new_obj.unique_component_id = 1
            new_obj.class_id = cls
            new_obj.object_id = tid
            new_obj.confidence = conf
            new_obj.obj_label = CLASS_MAP.get(cls, f"Class_{cls}")

            # rect_params 用 BoxMOT 自己給的框
            r = new_obj.rect_params
            r.left = x1
            r.top = y1
            r.width = max(1.0, x2 - x1)
            r.height = max(1.0, y2 - y1)
            r.border_width = 4
            r.has_bg_color = 0
            r.border_color.set(*get_class_color(cls))   # 預設車種色，後面 _process 還會覆寫一次

            pyds.nvds_add_obj_meta_to_frame(frame_meta, new_obj, None)

        # 步驟 5: 走和 nvdcf 一樣的軌跡狀態邏輯
        _process_tracked_frame(frame_meta, current_frame_objects, pad_index, cfg)

        l_frame = l_frame.next

    _post_frame_housekeeping(current_frame_objects)
    return Gst.PadProbeReturn.OK


# ==========================================
# 5. 每路畫面 OSD 探針 (Per-Cam FPS Overlay)
# ==========================================

def per_cam_osd_probe(pad, info, pad_index):
    """
    每路 nvosd.sink 上的 OSD 探針：左上角畫即時 FPS 文字

    兩種追蹤器模式共用，由該路 cam 的 display.show_fps_overlay 決定是否顯示

    參數：
        pad, info: GStreamer probe 標準參數
        pad_index (int): 哪一路 cam（由 main.py 用 add_probe 帶入）

    返回：
        Gst.PadProbeReturn.OK
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    cfg = SOURCE_CONFIGS.get(pad_index)
    if not cfg:
        return Gst.PadProbeReturn.OK

    show_fps = cfg.get("display", {}).get("show_fps_overlay", True)

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 0
        display_meta.num_lines = 0
        display_meta.num_rects = 0
        display_meta.num_circles = 0

        if show_fps and pad_index in fps_streams:
            display_meta.num_labels = 1
            txt_params = display_meta.text_params[0]
            txt_params.display_text = f"FPS: {fps_streams[pad_index]['current_fps']:.1f}"
            txt_params.x_offset = 5
            txt_params.y_offset = 5
            txt_params.font_params.font_name = "Serif Bold"
            txt_params.font_params.font_size = 25
            txt_params.font_params.font_color.set(0.0, 1.0, 0.0, 1.0)
            txt_params.set_bg_clr = 1
            txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.8)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK
