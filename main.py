#!/usr/bin/env python3
"""
DeepStream 7.1 車流計數主程式

主要功能：
1. 依 TRACKER_MODE 動態組裝 pipeline：
   - nvdcf  → PGIE → nvtracker → analytics
   - BoxMOT → PGIE → analytics（跳過 nvtracker，由 probe 接管追蹤）
2. 條件式掛載探針：
   - nvdcf  → tracker_src_pad_buffer_probe 掛在 tracker.src
   - BoxMOT → boxmot_pgie_src_probe 掛在 pgie.src
3. 多路 cam 各自的下游分支：本地預覽 / 影片存檔
4. RTSP 斷線防護（只作用於即時串流路，檔案來源不受影響）：
   第一層 nvurisrcbin 內建重連（無限重試）；第二層看門狗卡死單路重啟。
   兩層都持續到 EOS 為止：按 Q / SIGINT / SIGTERM 後看門狗立即停止。
5. 安全結束機制：Q 鍵 / Ctrl+C / systemctl stop 觸發 EOS，等影片封裝完成才退出
"""

import os
import sys
import time
import signal
import termios
import threading
import tty
# 啟用新版 nvstreammux：多路檔案來源時某一路先 EOS 不拖慢其餘來源。
# 必須在 import gi 之前，GStreamer 載入 nvstreammux 外掛時才讀得到。
os.environ.setdefault("USE_NEW_NVSTREAMMUX", "no")
# GLib/GIO 建立網路（RTSP）連線時會呼叫系統 libproxy 偵測 proxy。在 conda 環境下，
# conda 的 libstdc++ 與系統 libunwind ABI 不相容，libproxy 拋例外時無法正常 unwind
# 而導致行程 abort。改用 GIO 內建 dummy proxy resolver 完全繞過 libproxy。
# 須在匯入 gi 之前設定；系統 Python 不受影響，setdefault 也不覆蓋外部既有設定。
os.environ.setdefault("GIO_USE_PROXY_RESOLVER", "dummy")
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst

from logic.config import (
    SOURCE_CONFIGS, INFER_CONFIG, TRACKER_CONFIG,
    PREPROCESS_CONFIG, ANALYTICS_CONFIG,
    TRACKER_MODE,
)
from logic.state_db import initialize_state_managers, force_finalize_all, fps_streams
from logic.pipeline import (
    cb_newpad, cb_decodebin_child_added, make_elm, resolve_tracker_lib,
    _build_display_sink, setup_cam_branch,
)
from logic.probes import (
    tracker_src_pad_buffer_probe,
    boxmot_pgie_src_probe,
    per_cam_osd_probe,
)


# ==========================================
# 1. 全域狀態 (Global State)
# ==========================================

g_loop          = None     # GLib 主迴圈
g_pipeline      = None     # GStreamer 主 pipeline
g_eos_triggered = False    # EOS 是否已發送（避免重複觸發）

# ---- 看門狗（第二層防護：只針對「即時串流」路，檔案來源不適用）----
# 第一層：nvurisrcbin 內建 rtsp-reconnect，涵蓋「有斷線錯誤」的情況。
# 第二層：看門狗定期檢查各 RTSP 路最後吐幀時間，涵蓋「無錯誤但卡死不吐幀」的情況（單路重啟）。
# 兩層都只到 EOS 為止：按 Q / SIGINT / SIGTERM 觸發 EOS 後，看門狗立即停止，不干擾收尾。
g_sources      = {}    # pad_index -> {"src": nvurisrcbin, "streammux": mux, "cam": source_id}
g_last_restart = {}    # pad_index -> 上次重啟時間戳（防連環重啟）
WATCHDOG_STALL_SEC = 60    # 連續幾秒沒吐幀 → 判定卡死
WATCHDOG_GRACE_SEC = 60    # 重啟後寬限幾秒（期間不再判定）
WATCHDOG_CHECK_SEC = 10    # 每幾秒檢查一次

# --- 關閉行為逾時設定（可依需求調整）---
# EOS_WAIT_SECONDS：按 Q 後，給影片檔收尾的等待秒數，超過就強制結束主迴圈
# TEARDOWN_WAIT_SECONDS：關閉管線(set_state NULL)的最長等待秒數，
#                        即時 RTSP 來源收線常會卡住，超時就強制讓行程退出
EOS_WAIT_SECONDS      = 5
TEARDOWN_WAIT_SECONDS = 5

# --- RTSP 自動重連設定（nvurisrcbin 用，可依需求調整）---
# RTSP_RECONNECT_INTERVAL：RTSP 來源多久(秒)沒收到資料就嘗試重新連線（0=關閉重連）
# RTSP_RECONNECT_ATTEMPTS：最多重連次數（-1=無限重連；此屬性需 DeepStream 6.3 以上才有，
#                          舊版會自動略過、改用「每隔 interval 秒持續重試」的行為）
# RTSP_RTP_PROTOCOL      ：傳輸協定，4=TCP（較穩、對應原本 rtspsrc protocols=4）、0=UDP
RTSP_RECONNECT_INTERVAL = 5
RTSP_RECONNECT_ATTEMPTS = -1
RTSP_RTP_PROTOCOL       = 4


def _set_prop_if_exists(element, prop_name, value):
    """
    只有當該 element 真的有這個屬性時才設定，避免不同 DeepStream 版本
    缺少某屬性（例如舊版沒有 rtsp-reconnect-attempts）時整支程式報錯。

    回傳：
        bool: True 表示有設定成功；False 表示該版本無此屬性、已略過。
    """
    try:
        names = [p.name for p in element.list_properties()]
    except Exception:
        names = []
    if prop_name in names:
        try:
            element.set_property(prop_name, value)
            return True
        except Exception as e:
            print(f"[WARNING] 設定屬性 {prop_name}={value} 失敗：{e}")
            return False
    else:
        print(f"[INFO] 此 DeepStream 版本的來源元件沒有 '{prop_name}' 屬性，已略過。")
        return False


# ==========================================
# 2. 結束與訊息處理 (Lifecycle Callbacks)
# ==========================================

def force_quit_loop():
    """
    EOS 超時的強制退出 fallback

    用於：發送 EOS 後等待 8 秒影片仍未封裝完成時強制 quit，
          避免無限卡住

    返回：
        bool: False 讓 GLib timeout 不再重複觸發
    """
    global g_loop
    print("\n[WARNING] 等待影片封裝逾時，強制退出管線！")
    if g_loop and g_loop.is_running():
        g_loop.quit()
    return False


def keyboard_cb(fd, condition):
    """
    終端機按鍵處理：按 Q 觸發 EOS 安全退出

    處理流程：
    1. 讀一個字元
    2. 若是 q/Q 且尚未觸發 EOS → 發送 EOS event + 啟動 8 秒 timeout 保險

    參數：
        fd, condition: GLib io_add_watch 標準參數

    返回：
        bool: True 持續監聽；False 移除監聽（已觸發 EOS 後）
    """
    global g_eos_triggered, g_pipeline, g_loop

    ch = sys.stdin.read(1)
    if ch in ('q', 'Q') and not g_eos_triggered:
        g_eos_triggered = True
        print("\n[INFO] 收到 'Q' 鍵，正在安全發送 EOS 訊號 (等待影片寫入)...")
        if g_pipeline:
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(EOS_WAIT_SECONDS, force_quit_loop)
        return False
    return True


def bus_call(bus, message, loop):
    """
    GStreamer bus 訊息處理

    處理策略：
        EOS               → 正常結束主迴圈
        RTSP 相關錯誤     → 印 WARNING 但不退出，等待自動重連
        其它嚴重錯誤      → 印 ERROR 並退出

    參數：
        bus, message: GStreamer 標準參數
        loop (GLib.MainLoop): 要操作的主迴圈

    返回：
        bool: True 繼續接收訊息
    """
    t = message.type

    if t == Gst.MessageType.EOS:
        print("[INFO] 影像串流結束 (EOS 處理完畢)，準備安全退出...")
        loop.quit()

    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        # 同時檢查 err（錯誤短訊）與 debug（詳細訊息）：
        # 上次卡住的原因就是「rtsp」關鍵字只出現在 debug 裡，沒被檢查到，導致被誤判為致命錯誤。
        text = (str(err) + " " + (debug or "")).lower()

        # 來源/串流類錯誤（RTSP 斷線、串流停止等）→ 不退出，交給 nvurisrcbin 自動重連
        source_err_keywords = (
            "rtsp", "rtspsrc", "nvurisrcbin", "uridecodebin",
            "timeout", "resource not found", "could not read",
            "internal data stream error", "streaming stopped",
        )
        if any(k in text for k in source_err_keywords):
            print(f"[WARNING] 來源串流異常（{err}）。系統保持運行，交由自動重連處理...")
        else:
            print(f"[ERROR] 嚴重管線錯誤: {err}: {debug}")
            loop.quit()

    return True


# ==========================================
# 3. Pipeline 輔助 (Pipeline Helpers)
# ==========================================

def _restart_one_source(pad_index):
    """
    單獨重啟某一路 nvurisrcbin（看門狗判定卡死時由 GLib.idle_add 在主線程呼叫）。

    流程：source 設 NULL → 等狀態確實切換 → 重設 PLAYING。
    重啟後 nvurisrcbin 重連來源、重新吐 pad → cb_newpad 找回既有（未連結的）
    streammux sink pad 直接重接。不動其他路，互不影響。
    """
    info = g_sources.get(pad_index)
    if not info:
        return False
    if g_eos_triggered:      # 已在收尾流程 → 不再重啟，避免干擾影片封裝
        return False
    src, cam = info["src"], info["cam"]
    streammux = info["streammux"]
    print(f"[WATCHDOG] 重啟 {cam}（pad={pad_index}）...")
    try:
        # 1) 先斷開並 release streammux 的 sink pad —— 關鍵！
        #    若不先解開，nvurisrcbin 設 NULL 時 src pad 仍連在 streammux 上，
        #    bin 無法乾淨拆除，會殘留舊的 vsrc_0 幽靈 pad，重新 PLAYING 時報
        #    「Padname vsrc_0 is not unique」→ 新 pad 加不進去 → 永遠接不回來。
        sinkpad = streammux.get_static_pad(f"sink_{pad_index}")
        if sinkpad is not None:
            peer = sinkpad.get_peer()
            if peer is not None:
                peer.unlink(sinkpad)
            streammux.release_request_pad(sinkpad)

        # 2) 該路 source 設 NULL，等狀態確實切換
        src.set_state(Gst.State.NULL)
        src.get_state(Gst.CLOCK_TIME_NONE)

        # 3) 重新 PLAYING：nvurisrcbin 重連 RTSP、重新吐 pad → cb_newpad
        #    因 pad 已 release，會 get_request_pad 重新要一個乾淨的 sink_N 接回
        src.set_state(Gst.State.PLAYING)
        g_last_restart[pad_index] = time.time()
        print(f"[WATCHDOG] {cam} 已送出重啟（已釋放 streammux pad），等待重新連線...")
    except Exception as e:
        print(f"[WATCHDOG] 重啟 {cam} 發生例外: {e}")
    return False   # 給 idle_add 用，只跑一次


def _watchdog_check():
    """
    每 WATCHDOG_CHECK_SEC 秒檢查各「RTSP 路」最後吐幀時間，卡死超過門檻就單路重啟。

    只監控 g_sources 內的路（建立來源時只收錄即時串流，檔案來源播完不吐幀是正常現象，
    絕不能重啟——否則影片會從頭重播、DB 重複計數）。
    EOS 觸發（按 Q / SIGINT / SIGTERM）後回傳 False 停止本 timer，不干擾收尾封裝。
    """
    if g_eos_triggered:
        print("[WATCHDOG] 偵測到 EOS 收尾流程，看門狗停止")
        return False
    now = time.time()
    for pad_index, info in list(g_sources.items()):
        # 重啟寬限期內不判定
        if now - g_last_restart.get(pad_index, 0) < WATCHDOG_GRACE_SEC:
            continue
        stats = fps_streams.get(pad_index, {})
        ts = stats.get("timestamps")
        if not ts:
            continue   # 還沒收過任何幀（剛啟動/首連中），交給 nvurisrcbin 內建重連
        idle = now - ts[-1]
        if idle >= WATCHDOG_STALL_SEC:
            print(f"[WATCHDOG] {info['cam']}（pad={pad_index}）已 {idle:.0f} 秒無新幀，判定卡死 → 單路重啟")
            GLib.idle_add(_restart_one_source, pad_index)
    return True   # 回 True 讓 timer 持續


def _enlarge_queue(q, max_buffers=400):
    """
    放寬 queue 容量，避免下游處理偶發較慢時被反壓

    參數：
        q (Gst.Element): queue 元件
        max_buffers (int): 緩衝區最大數量
    """
    q.set_property("max-size-buffers", max_buffers)
    q.set_property("max-size-bytes", 0)
    q.set_property("max-size-time", 0)


def main():
    """
    主程式進入點

    處理流程：
    1. 印追蹤器模式並（BoxMOT 模式）初始化 tracker instance
    2. 建立 GStreamer pipeline 與 streammux
    3. 為每路 cam 建立 uridecodebin 來源
    4. 建立共用推論元件（preprocess / pgie / analytics）
    5. 依 TRACKER_MODE 條件式建立 nvtracker 與連結 pipeline 中段
    6. 條件式掛載對應的追蹤探針
    7. 建立 demux 並為每路 cam 組下游分支
    8. 啟動 RTSP server（若有 cam 啟用推流）
    9. 進入 GLib 主迴圈，等待 Q 鍵或 EOS 退出
    """
    global g_loop, g_pipeline, g_eos_triggered

    # ---- 步驟 1: 印追蹤器模式 + (BoxMOT) 初始化 tracker instance ----
    if TRACKER_MODE == "nvdcf":
        print("[INFO] 初始化 DeepStream 車流架構：PGIE → NvDCF (內建追蹤器) → Analytics")
    else:
        print(f"[INFO] 初始化 DeepStream 車流架構：PGIE → {TRACKER_MODE} (BoxMOT) → Analytics")
        print("[INFO]   pipeline 將跳過 nvtracker，追蹤交由 pgie.src 上的 BoxMOT 探針處理")

        from logic.boxmot_adapter import initialize_boxmot_trackers
        initialize_boxmot_trackers()

    # ---- 步驟 2: 建立 pipeline 與 streammux ----
    Gst.init(None)
    g_pipeline = Gst.Pipeline.new("traffic-pipeline")

    num_sources = len(SOURCE_CONFIGS)

    # 任一 cam 開啟 show_window 就建立本地預覽 sink
    show_window = any(
        cfg.get("display", {}).get("show_window", True)
        for cfg in SOURCE_CONFIGS.values()
    )

    streammux = make_elm("nvstreammux", "Stream-muxer")
    streammux.set_property("batch-size", num_sources)  # 新舊版 mux 皆支援

    if os.environ.get("USE_NEW_NVSTREAMMUX") == "yes":
        # 新版 mux：不接受 width/height/live-source 等舊屬性，改用 config_mux.txt
        _mux_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_mux.txt")
        if os.path.exists(_mux_cfg):
            streammux.set_property("config-file-path", _mux_cfg)
        else:
            print(f"[WARNING] 找不到 {_mux_cfg}，新版 mux 將用內建預設值")
    else:
        # 舊版 mux：維持原本設定
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batched-push-timeout", 70000)
        streammux.set_property("live-source", 1)
        streammux.set_property("nvbuf-memory-type", 0)
    g_pipeline.add(streammux)

    # ---- 步驟 3: 每路 cam 建一個 nvurisrcbin（RTSP 路啟用內建斷線自動重連）----
    # nvurisrcbin 是 DeepStream 自家的來源元件，輸出已是 NVMM 影格、可直接接 streammux，
    # 且支援 RTSP 斷線後自動重連 —— 單一攝影機掉線時整支程式不會死，會自己重連。
    for pad_index, cfg in SOURCE_CONFIGS.items():
        source = make_elm("nvurisrcbin", f"uri-decode-bin-{pad_index}")
        source.set_property("uri", cfg["source"])

        is_live = not cfg.get("is_file_source", False)
        if is_live:
            # 第一層防護：RTSP 自動重連與傳輸設定（缺屬性的舊版 DeepStream 自動略過）
            _set_prop_if_exists(source, "rtsp-reconnect-interval", RTSP_RECONNECT_INTERVAL)
            _set_prop_if_exists(source, "rtsp-reconnect-attempts", RTSP_RECONNECT_ATTEMPTS)
            _set_prop_if_exists(source, "select-rtp-protocol", RTSP_RTP_PROTOCOL)
            _set_prop_if_exists(source, "latency", 200)
            _set_prop_if_exists(source, "udp-buffer-size", 2000000)
            print(f"[INFO] {cfg.get('source_id', pad_index)} 為即時串流："
                  f"啟用自動重連（{RTSP_RECONNECT_INTERVAL}s 間隔、無限重試）")

        source.connect("pad-added", cb_newpad, {"streammux": streammux, "pad_index": pad_index})
        # 內部 rtspsrc 的 TCP / 逾時調校（nvurisrcbin 用 child-added 遞迴抓，取代 source-setup）
        source.connect("child-added", cb_decodebin_child_added, None)
        g_pipeline.add(source)

        # 第二層防護：看門狗只登記「即時串流」路（檔案來源播完不吐幀是正常現象，
        # 絕不能重啟——否則影片會從頭重播、DB 重複計數）
        if is_live:
            g_sources[pad_index] = {
                "src": source, "streammux": streammux,
                "cam": cfg.get("source_id", f"cam{pad_index}"),
            }

    # ---- 步驟 4: 共用推論元件 ----
    q1          = make_elm("queue", "q1")
    q2          = make_elm("queue", "q2")
    q3          = make_elm("queue", "q3")
    q_analytics = make_elm("queue", "q_analytics")
    q4          = make_elm("queue", "q4")
    _enlarge_queue(q_analytics, max_buffers=200)

    preprocess = make_elm("nvdspreprocess", "preprocess")
    preprocess.set_property("config-file", PREPROCESS_CONFIG)

    pgie = make_elm("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", INFER_CONFIG)
    pgie.set_property("input-tensor-meta", True)

    analytics = make_elm("nvdsanalytics", "analytics")
    analytics.set_property("config-file", ANALYTICS_CONFIG)

    # ---- 步驟 5: 依 TRACKER_MODE 條件式建立 nvtracker + 連結 pipeline 中段 ----
    tracker = None
    if TRACKER_MODE == "nvdcf":
        tracker = make_elm("nvtracker", "tracker")
        tracker.set_property("ll-config-file", TRACKER_CONFIG)
        tracker.set_property("ll-lib-file", resolve_tracker_lib())
        tracker.set_property("tracker-width", 640)
        tracker.set_property("tracker-height", 384)

    pipeline_elements = [q1, preprocess, q2, pgie, q3, q_analytics, analytics, q4]
    if tracker is not None:
        pipeline_elements.append(tracker)
    for elm in pipeline_elements:
        g_pipeline.add(elm)

    # 共用前段：streammux → q1 → preprocess → q2 → pgie → q3
    streammux.link(q1)
    q1.link(preprocess)
    preprocess.link(q2)
    q2.link(pgie)
    pgie.link(q3)

    if TRACKER_MODE == "nvdcf":
        # 原流程：pgie → q3 → nvtracker → q_analytics
        q3.link(tracker)
        tracker.link(q_analytics)
        print("[INFO] Pipeline 中段：pgie → q3 → nvtracker → q_analytics → analytics → q4")
    else:
        # BoxMOT 流程：跳過 nvtracker
        q3.link(q_analytics)
        print(f"[INFO] Pipeline 中段：pgie → q3 → q_analytics → analytics → q4  "
              f"({TRACKER_MODE} 模式，已跳過 nvtracker)")

    q_analytics.link(analytics)
    analytics.link(q4)

    # ---- 步驟 6: 條件式掛載追蹤探針 ----
    if TRACKER_MODE == "nvdcf":
        tracker.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0
        )
        print("[INFO] 已掛載探針：tracker_src_pad_buffer_probe → tracker.src")
    else:
        pgie.get_static_pad("src").add_probe(
            Gst.PadProbeType.BUFFER, boxmot_pgie_src_probe, 0
        )
        print(f"[INFO] 已掛載探針：boxmot_pgie_src_probe → pgie.src ({TRACKER_MODE})")

    # ---- 步驟 7: demux 並為每路 cam 組下游分支 ----
    demux = make_elm("nvstreamdemux", "demuxer")
    g_pipeline.add(demux)
    q4.link(demux)

    display_streammux = _build_display_sink(g_pipeline, num_sources) if show_window else None

    for pad_index, cfg in SOURCE_CONFIGS.items():
        setup_cam_branch(
            g_pipeline, pad_index, cfg, demux, display_streammux, per_cam_osd_probe
        )

    # ---- 步驟 9: 訊號處理 +（有終端機時才）鍵盤監聽 + 主迴圈 ----
    g_loop = GLib.MainLoop()

    # systemctl stop / restart 送的是 SIGTERM；Ctrl+C 送的是 SIGINT。
    # 用 GLib.unix_signal_add 在主迴圈內安全處理：先送 EOS 讓影片收尾，再排程強制退出，
    # 確保 force_finalize_all()（寫 DB）一定會跑到，不會在停止服務時掉資料。
    def _on_stop_signal(_user_data):
        global g_eos_triggered
        print("\n[INFO] 收到停止訊號（SIGTERM/SIGINT），準備安全退出並存檔...")
        if g_pipeline and not g_eos_triggered:
            g_eos_triggered = True
            g_pipeline.send_event(Gst.Event.new_eos())
            GLib.timeout_add_seconds(EOS_WAIT_SECONDS, force_quit_loop)
        return GLib.SOURCE_CONTINUE

    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGTERM, _on_stop_signal, None)
    GLib.unix_signal_add(GLib.PRIORITY_HIGH, signal.SIGINT, _on_stop_signal, None)

    # 只有「真的有終端機」（互動執行）時才設定鍵盤監聽。
    # systemd 服務底下沒有 TTY，這段必須跳過，否則 termios.tcgetattr 會直接報錯、程式還沒跑就掛。
    interactive = sys.stdin.isatty()
    fd = None
    old_settings = None
    if interactive:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        GLib.io_add_watch(fd, GLib.PRIORITY_DEFAULT, GLib.IOCondition.IN, keyboard_cb)
        print("\n[INFO] 💡 提示：在終端機按下 'q' 鍵即可優雅退出並存檔...\n")
    else:
        print("\n[INFO] 非互動模式（無終端機）：鍵盤監聽停用，請用 'systemctl stop' 安全停止。\n")

    try:
        bus = g_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, g_loop)

        g_pipeline.set_state(Gst.State.PLAYING)

        # 看門狗：只有存在 RTSP 路時才啟動（檔案批次跑不啟動、也不該監控）
        if g_sources:
            GLib.timeout_add_seconds(WATCHDOG_CHECK_SEC, _watchdog_check)
            print(f"[INFO] 看門狗啟動：監控 {len(g_sources)} 路即時串流，"
                  f"每 {WATCHDOG_CHECK_SEC}s 檢查，卡死門檻 {WATCHDOG_STALL_SEC}s，重啟寬限 {WATCHDOG_GRACE_SEC}s")

        g_loop.run()

    finally:
        # 還原終端機設定（只有互動模式才需要、也才安全）
        if interactive and fd is not None and old_settings is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        # 先把 DB 寫完、連線關掉。這步很快，且保證資料安全（之後就算強制退出也不怕掉資料）
        force_finalize_all()

        # 看門狗：即時 RTSP 來源在 set_state(NULL) 收線時常會卡住好幾分鐘。
        # 開一條背景執行緒，若超過 TEARDOWN_WAIT_SECONDS 秒管線還沒關乾淨，就強制讓行程退出。
        def _force_exit_watchdog():
            time.sleep(TEARDOWN_WAIT_SECONDS)
            print(f"\n[WARNING] 管線關閉超過 {TEARDOWN_WAIT_SECONDS} 秒未完成，強制結束行程。")
            os._exit(0)

        watchdog = threading.Thread(target=_force_exit_watchdog, daemon=True)
        watchdog.start()

        g_pipeline.set_state(Gst.State.NULL)

        # 正常情況下管線會在看門狗逾時前關好，走到這裡直接乾淨退出
        print("[INFO] 管線已關閉，程式結束。")
        os._exit(0)


if __name__ == '__main__':
    initialize_state_managers()
    main()
