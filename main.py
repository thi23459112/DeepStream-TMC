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
3. 多路 cam 各自的下游分支：本地預覽 / 影片存檔 / RTSP 推流
4. RTSP server：把每路 cam 的 udpsink 註冊成獨立 mount_path
5. 安全結束機制：Q 鍵 / Ctrl+C 觸發 EOS，等影片封裝完成才退出
"""

import os
import sys
import time
import select
import signal
import termios
import threading
import tty
# 啟用新版 nvstreammux：多路檔案來源時某一路先 EOS 不拖慢其餘來源。
# 必須在 import gi 之前，GStreamer 載入 nvstreammux 外掛時才讀得到。
os.environ.setdefault("USE_NEW_NVSTREAMMUX", "yes")
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer

from logic.color import load_labels, CLASS_MAP
from logic.config import (
    SOURCE_CONFIGS, INFER_CONFIG, TRACKER_CONFIG,
    PREPROCESS_CONFIG, ANALYTICS_CONFIG,
    TRACKER_MODE, BOXMOT_TRACKER_CONFIG,
)
from logic.state_db import initialize_state_managers, force_finalize_all
from logic.pipeline import (
    cb_newpad, cb_source_setup, make_elm,
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
g_rtsp_server   = None     # RTSP server 引用（持有避免被 GC）

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
RTSP_RECONNECT_INTERVAL = 10
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


def _start_rtsp_server(rtsp_routes):
    """
    啟動 GstRtspServer 並把每路 cam 的 udpsink 端口註冊成 mount_path

    處理流程：
    1. 依 port 分組（同 port 不同 mount_path 共用一台 server）
    2. 為每個 port 建一個 RTSPServer
    3. 為每路 cam 建一個 RTSPMediaFactory，套用對應的 udpsrc + rtp{h264,h265}pay

    參數：
        rtsp_routes (list[dict]): 每路 cam 的推流設定
            {pad_index, udp_port, port, mount_path, encoder}

    返回：
        list | None: 啟動的 RTSPServer 列表（None 表示無推流需求）
    """
    if not rtsp_routes:
        return None

    # 步驟 1: 依 port 分組
    routes_by_port = {}
    for r in rtsp_routes:
        routes_by_port.setdefault(r["port"], []).append(r)

    # 步驟 2: 每個 port 啟一台 RTSP server
    servers = []
    for port, routes in routes_by_port.items():
        server = GstRtspServer.RTSPServer()
        server.set_service(str(port))
        mounts = server.get_mount_points()

        # 步驟 3: 每路 cam 註冊一個 mount_path
        for r in routes:
            udp_port   = r["udp_port"]
            encoder    = r["encoder"]
            mount_path = "/" + r["mount_path"].lstrip("/")

            # encoding-name 必須對齊客戶端，否則 SDP 不匹配連不上
            enc_name = "H265" if encoder == "h265" else "H264"

            launch_str = (
                f"( udpsrc port={udp_port} caps=\"application/x-rtp, "
                f"media=video, clock-rate=90000, encoding-name={enc_name}, payload=96\" "
                f"! rtp{encoder}depay ! rtp{encoder}pay name=pay0 pt=96 )"
            )

            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_launch(launch_str)
            factory.set_shared(True)   # 多客戶端可同時連同一 mount
            mounts.add_factory(mount_path, factory)

            print(f"[INFO] RTSP 推流註冊: rtsp://<本機IP>:{port}{mount_path} "
                  f"(encoder={encoder}, udp_port={udp_port})")

        server.attach(None)
        servers.append(server)

    return servers


# ==========================================
# 4. 主程式 (Main)
# ==========================================

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
    global g_loop, g_pipeline, g_eos_triggered, g_rtsp_server

    # ---- 步驟 1: 印追蹤器模式 + (BoxMOT) 初始化 tracker instance ----
    if TRACKER_MODE == "nvdcf":
        print("[INFO] 初始化 DeepStream 車流架構：PGIE → NvDCF (內建追蹤器) → Analytics")
    else:
        print(f"[INFO] 初始化 DeepStream 車流架構：PGIE → {TRACKER_MODE} (BoxMOT) → Analytics")
        print(f"[INFO]   pipeline 將跳過 nvtracker，追蹤交由 pgie.src 上的 BoxMOT 探針處理")

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

    # ---- 步驟 3: 每路 cam 建一個 nvurisrcbin（內建 RTSP 自動重連）----
    # nvurisrcbin 是 DeepStream 自家的來源元件，輸出已是 NVMM 影格、可直接接 streammux，
    # 且支援 RTSP 斷線後自動重連 —— 單一攝影機掉線時整支程式不會死，會自己重連。
    for pad_index, cfg in SOURCE_CONFIGS.items():
        source = make_elm("nvurisrcbin", f"uri-decode-bin-{pad_index}")
        source.set_property("uri", cfg["source"])

        # RTSP 自動重連與傳輸設定（用 helper 設定，舊版 DeepStream 缺屬性也不會報錯）
        _set_prop_if_exists(source, "rtsp-reconnect-interval", RTSP_RECONNECT_INTERVAL)
        _set_prop_if_exists(source, "rtsp-reconnect-attempts", RTSP_RECONNECT_ATTEMPTS)
        _set_prop_if_exists(source, "select-rtp-protocol", RTSP_RTP_PROTOCOL)
        _set_prop_if_exists(source, "latency", 200)

        source.connect("pad-added", cb_newpad, {"streammux": streammux, "pad_index": pad_index})
        g_pipeline.add(source)

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
        tracker.set_property(
            "ll-lib-file",
            "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        )
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

    # 收集所有啟用 RTSP 推流的路，等下批次註冊到 RTSP server
    rtsp_routes = []
    for pad_index, cfg in SOURCE_CONFIGS.items():
        udp_port = setup_cam_branch(
            g_pipeline, pad_index, cfg, demux, display_streammux, per_cam_osd_probe
        )
        if udp_port is not None:
            rtsp_routes.append({
                "pad_index":  pad_index,
                "udp_port":   udp_port,
                "port":       cfg["rtsp_push"]["port"],
                "mount_path": cfg["rtsp_push"]["mount_path"],
                "encoder":    cfg["rtsp_push"]["encoder"],
            })

    # ---- 步驟 8: 啟動 RTSP server ----
    if rtsp_routes:
        g_rtsp_server = _start_rtsp_server(rtsp_routes)
        print(f"[INFO] 共 {len(rtsp_routes)} 條 RTSP 推流就緒")
    else:
        print("[INFO] 無 cam 啟用 RTSP 推流，跳過 RTSP server 啟動")

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
