# logic/pipeline.py
"""
pipeline.py
-----------
GStreamer pipeline 元件建構與分支邏輯。

對每路 cam，setup_cam_branch 依以下 YAML 欄位動態組裝分支：
    - output.save_output_video → 寫檔分支
    - display.show_window      → 本地預覽分支（合併到 tile 視窗）

平台自動適配：
    以 _is_jetson() 判斷平台（Jetson 或 dGPU/WSL2），在「顯示 sink、OSD 處理模式、
    NVENC 編碼器屬性」上自動選用該平台支援的設定。
    寫檔編碼器採延遲偵測：預設 NVENC，建不出時自動退回 CPU x264（USE_CPU_ENCODER 可覆寫）。
    nvtracker 的 ll-lib-file 由 resolve_tracker_lib() 自動解析（DS_TRACKER_LIB 可覆寫）。
    顯示 sink 找不到 NVIDIA 專用 sink 時退回標準 GStreamer sink（DS_DISPLAY_SINK 可指定）。

Tile 佈局：
    自動依 cam 數量計算 rows × cols，每格保持 16:9。
"""

import sys
from gi.repository import Gst
import os


def _is_jetson():
    """判斷是否為 Jetson（aarch64 或存在 /etc/nv_tegra_release）。"""
    import platform
    return (platform.machine() == "aarch64") or os.path.isfile("/etc/nv_tegra_release")


def _safe_set(elm, name, value):
    """只有當元件確實具有該屬性時才設定，避免跨平台屬性差異造成例外。回傳是否有設成功。"""
    if elm.find_property(name) is not None:
        elm.set_property(name, value)
        return True
    return False


# ==========================================
# 編碼器選擇（延遲偵測）與追蹤器路徑解析
# ==========================================

def _detect_cpu_encoder():
    """
    決定是否使用 CPU 軟體編碼器（x264）。

    規則（可用環境變數 USE_CPU_ENCODER 覆寫，1/true=強制 CPU，0/false=強制 NVENC）：
      - 環境變數有明確指定 → 依指定
      - 否則：預設優先使用 NVENC 硬體編碼（較快）；只有在「確實建不出 NVENC」時才退回 CPU，
              避免在無 NVENC 的環境（如部分 WSL）開存檔就直接中斷。
    """
    env = os.environ.get("USE_CPU_ENCODER")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    # 預設 = NVENC。用「實際建立元件」測試（比 ElementFactory.find 更可靠）
    test = Gst.ElementFactory.make("nvv4l2h264enc", None)
    if test is not None:
        print("[INFO] 預設使用 NVENC 硬體編碼（nvv4l2h264enc）")
        return False
    print("[INFO] 建不出 NVENC，退回 CPU 軟體編碼（x264）")
    return True


# 編碼器選擇「延遲判斷」：不在 import 當下決定，而是等第一次真正要建編碼器時才判斷並快取。
# 原因：main.py 是先 import 本模組、之後才呼叫 Gst.init(None)。若在 import 當下判斷，
# 會早於 Gst.init()，此時 GStreamer 尚未初始化、抓不到 NVENC，導致誤退 CPU。
_USE_CPU_ENCODER = None   # None=尚未判斷；True/False=已快取


def use_cpu_encoder():
    """回傳是否使用 CPU 編碼；第一次呼叫時才判斷並快取（此時 Gst.init() 已完成，能正確抓到 NVENC）。"""
    global _USE_CPU_ENCODER
    if _USE_CPU_ENCODER is None:
        _USE_CPU_ENCODER = _detect_cpu_encoder()
    return _USE_CPU_ENCODER


def resolve_tracker_lib():
    """
    自動解析 nvtracker 的 ll-lib-file 路徑（跨平台 / 跨機器）。

    順序：
      1. 環境變數 DS_TRACKER_LIB（若指定且存在）
      2. 依序搜尋常見安裝路徑（含 glob 掃版本號），回傳第一個存在的
      3. 都找不到 → 回傳標準 NVIDIA 路徑（讓 DS 自行報錯提示）
    """
    env = os.environ.get("DS_TRACKER_LIB", "").strip()
    if env and os.path.exists(env):
        return env
    import glob as _glob
    candidates = [
        "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "/opt/thi/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    ]
    candidates += sorted(_glob.glob(
        "/opt/nvidia/deepstream/deepstream*/lib/libnvds_nvmultiobjecttracker.so"))
    for p in candidates:
        if os.path.exists(p):
            return p
    return "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"


def cb_newpad(decodebin, decoder_src_pad, data):
    """
    nvurisrcbin 對外動態 pad 出現時，鏈到 streammux 的對應 sink_N。
    非 video pad（音訊等）接 fakesink 消化，避免未連結 pad 反壓卡整路。

    重啟支援：看門狗單路重啟（NULL→PLAYING）後會再次觸發本回呼；
    若該 sink_N 已存在但目前未 link，直接重新接上即完成復原。
    """
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()

    streammux = data["streammux"]
    pipeline = streammux.get_parent()
    pad_name = f"sink_{data['pad_index']}"

    is_video = caps.get_structure(0).get_name().find("video") != -1

    # 先取回既有 request pad；沒有才 request 新的（重啟後 pad 仍在，不可重複 request）
    sinkpad = streammux.get_static_pad(pad_name)
    if sinkpad is None:
        sinkpad = streammux.get_request_pad(pad_name)

    # 非影像流 → fakesink 消化
    if not is_video or sinkpad is None:
        _drain_pad_to_fakesink(pipeline, decoder_src_pad)
        return

    # 該 sink 已被接著（來源吐出第二條視訊流）→ 多餘的導 fakesink；
    # 未 link（首次接、或單路重啟後重接）→ 直接接上
    if sinkpad.is_linked():
        _drain_pad_to_fakesink(pipeline, decoder_src_pad)
        return

    decoder_src_pad.link(sinkpad)


def _drain_pad_to_fakesink(pipeline, src_pad):
    """把不需要的 pad（音訊 / 多餘視訊）接到 fakesink 消化，避免未連結 pad 造成反壓卡整路。"""
    fs = Gst.ElementFactory.make("fakesink", None)   # None = 自動命名，避免多路/重啟撞名
    if fs is None:
        return
    fs.set_property("sync", False)
    fs.set_property("async", False)
    pipeline.add(fs)
    fs.sync_state_with_parent()
    src_pad.link(fs.get_static_pad("sink"))


def cb_decodebin_child_added(child_proxy, obj, name, user_data):
    """
    nvurisrcbin 內部子元件建立時的回呼（取代 uridecodebin 的 source-setup 訊號，
    nvurisrcbin 沒有 source-setup，須用 child-added 遞迴往內抓）。
      - 內層還有 decodebin 時繼續往下掛，才追得到最底層的 rtspsrc。
      - 對 rtspsrc 強制 TCP、設抖動緩衝與連線逾時、超延遲丟幀。
    只在元件確實有該屬性時才設定（_safe_set），檔案來源的內部元件不受影響。
    """
    if name.find("decodebin") != -1:
        obj.connect("child-added", cb_decodebin_child_added, user_data)
    if name.find("source") != -1:
        _safe_set(obj, "protocols", 4)          # 4 = TCP
        _safe_set(obj, "latency", 200)          # 抖動緩衝 200ms
        _safe_set(obj, "timeout", 5000000)      # 連線逾時 5 秒（微秒）
        _safe_set(obj, "drop-on-latency", True)


def make_elm(gst_type, name):
    """建立 GStreamer 元件，失敗則整支程式退出。"""
    elm = Gst.ElementFactory.make(gst_type, name)
    if not elm:
        sys.exit(f"[ERROR] 無法建立 element: {gst_type} ({name})")
    return elm


# ==========================================
# 自動依 cam 數決定 tile 佈局
# ==========================================
def _get_tile_layout(num_sources):
    """
    根據 cam 總數回傳 (rows, cols, total_width, total_height)。
    每格保持 16:9，視窗總寬固定 1920，總高隨列數變動。

    對應表：
        1     → 1×1  (1920×1080)
        2     → 1×2  (1920× 540)
        3~4   → 2×2  (1920×1080)
        5~6   → 2×3  (1920× 720)
        7~9   → 3×3  (1920×1080)
        >9    → 4×4  (1920×1080)
    """
    if num_sources == 1:
        rows, cols = 1, 1
    elif num_sources == 2:
        rows, cols = 1, 2
    elif num_sources <= 4:
        rows, cols = 2, 2
    elif num_sources <= 6:
        rows, cols = 2, 3
    elif num_sources <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = 4, 4

    total_width = 1920
    cell_w = total_width // cols
    cell_h = int(cell_w * 9 / 16)
    total_height = cell_h * rows
    return rows, cols, total_width, total_height


# ==========================================
# 寫檔分支：file 來源
# ==========================================
def _build_save_branch_for_file(pipeline, pad_index, video_path, source_fps):
    """
    建立寫檔分支(本地檔案來源版)：
        nvvideoconvert → videorate → capsfilter(NV12, framerate=N/1)
        → nvv4l2h264enc → h264parse → qtmux → filesink

    ⭐ 修復「Buffer has no PTS」錯誤：
        - 插入 videorate 並用 capsfilter 明確指定 framerate，
          videorate 會以固定 framerate 重新產生 PTS，避免下游 qtmux 拒收。
        - qtmux 加 dts-method=1（從 PTS 推 DTS）作為雙保險。

    回傳起點 element(用於上游 link)。
    """
    i = pad_index

    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)

    # ⭐ 新增：videorate 確保 framerate / PTS 規律
    videorate = make_elm("videorate", f"videorate-save-{i}")

    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")
    fps_int = int(round(source_fps))
    if fps_int <= 0:
        fps_int = 30
    cpu_enc = use_cpu_encoder()
    caps_base = ("video/x-raw, format=I420" if cpu_enc
                 else "video/x-raw(memory:NVMM), format=NV12")
    caps = Gst.Caps.from_string(f"{caps_base}, framerate={fps_int}/1")
    cap_filter.set_property("caps", caps)

    if cpu_enc:
        encoder = make_elm("x264enc", f"encoder-{i}")
        _safe_set(encoder, "bitrate", 4000)      # x264enc 單位是 kbps
        _safe_set(encoder, "speed-preset", 1)    # 1=ultrafast，吞吐優先
        _safe_set(encoder, "tune", 4)            # 4=zerolatency
        _safe_set(encoder, "key-int-max", fps_int)
    else:
        encoder = make_elm("nvv4l2h264enc", f"encoder-{i}")
        _safe_set(encoder, "bitrate", 4000000)
        _safe_set(encoder, "profile", 0)
        _safe_set(encoder, "iframeinterval", fps_int)
        # preset-level / insert-sps-pps / maxperf-enable 為 Jetson NVENC 專有；
        # 若一個都設不成功，代表在 dGPU/WSL2，改設 dGPU NVENC 的調校屬性。
        if not (_safe_set(encoder, "preset-level", 1)
                | _safe_set(encoder, "insert-sps-pps", 1)
                | _safe_set(encoder, "maxperf-enable", 1)):
            _safe_set(encoder, "preset-id", 1)
            _safe_set(encoder, "tuning-info-id", 2)

    parser = make_elm("h264parse", f"h264-parser-{i}")

    muxer = make_elm("qtmux", f"muxer-{i}")
    # ⭐ 新增：DTS 由 PTS 推導(ASC)，避免 qtmux 對缺失 DTS 過度敏感
    muxer.set_property("dts-method", 1)

    filesink = make_elm("filesink", f"filesink-{i}")
    filesink.set_property("location", video_path)
    filesink.set_property("async", False)
    filesink.set_property("sync", False)

    for elm in [nvvidconv_s, videorate, cap_filter, encoder, parser, muxer, filesink]:
        pipeline.add(elm)

    nvvidconv_s.link(videorate)
    videorate.link(cap_filter)
    cap_filter.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)

    return nvvidconv_s


# ==========================================
# 寫檔分支：RTSP 來源（需要 videorate 把 FPS 固定）
# ==========================================
def _build_save_branch_for_rtsp(pipeline, pad_index, video_path, source_fps):
    """建立寫檔分支（RTSP 串流版），插入 videorate 穩定 FPS。"""
    i = pad_index

    nvvidconv_s = make_elm("nvvideoconvert", f"convertor-save-{i}")
    nvvidconv_s.set_property("nvbuf-memory-type", 0)

    videorate = make_elm("videorate", f"videorate-{i}")
    cap_filter = make_elm("capsfilter", f"cap-filter-save-{i}")

    fps_int = int(round(source_fps))
    if fps_int <= 0:
        fps_int = 30

    cpu_enc = use_cpu_encoder()
    caps_base = ("video/x-raw, format=I420" if cpu_enc
                 else "video/x-raw(memory:NVMM), format=NV12")
    caps = Gst.Caps.from_string(f"{caps_base}, framerate={fps_int}/1")
    cap_filter.set_property("caps", caps)

    if cpu_enc:
        encoder = make_elm("x264enc", f"encoder-{i}")
        _safe_set(encoder, "bitrate", 4000)      # x264enc 單位是 kbps
        _safe_set(encoder, "speed-preset", 1)    # 1=ultrafast，吞吐優先
        _safe_set(encoder, "tune", 4)            # 4=zerolatency
        _safe_set(encoder, "key-int-max", fps_int)
    else:
        encoder = make_elm("nvv4l2h264enc", f"encoder-{i}")
        _safe_set(encoder, "bitrate", 4000000)
        _safe_set(encoder, "profile", 0)
        _safe_set(encoder, "iframeinterval", fps_int)
        # Jetson 專有屬性設不成 → dGPU/WSL2，改設 dGPU NVENC 調校屬性
        if not (_safe_set(encoder, "preset-level", 1)
                | _safe_set(encoder, "insert-sps-pps", 1)
                | _safe_set(encoder, "maxperf-enable", 1)):
            _safe_set(encoder, "preset-id", 1)
            _safe_set(encoder, "tuning-info-id", 2)

    parser = make_elm("h264parse", f"h264-parser-{i}")
    muxer = make_elm("qtmux", f"muxer-{i}")
    muxer.set_property("dts-method", 1)
    filesink = make_elm("filesink", f"filesink-{i}")
    filesink.set_property("location", video_path)
    filesink.set_property("async", False)
    filesink.set_property("sync", False)

    for elm in [nvvidconv_s, videorate, cap_filter, encoder, parser, muxer, filesink]:
        pipeline.add(elm)

    nvvidconv_s.link(videorate)
    videorate.link(cap_filter)
    cap_filter.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)

    return nvvidconv_s


# ==========================================
# 本地顯示分支
# ==========================================
def _build_display_sink(pipeline, num_sources, has_live_source=False):
    """
    本地 tile 預覽視窗：
        streammux → tiler → nvvideoconvert → nvegltransform → nveglglessink
    回傳 streammux 作為各路顯示分支共用 sink_N 接點。

    ⭐ live-source 必須與主線 streammux 一致：
        - 檔案模式 (live-source=0)：依 buffer PTS 跑
        - RTSP 模式 (live-source=1)：依 wall clock 跑
        否則下游 streammux 會等永遠不會到的 buffer → 整個 pipeline 卡死。
    """
    rows, cols, total_w, total_h = _get_tile_layout(num_sources)
    print(f"[INFO] Tile 佈局: {rows}x{cols}, 視窗尺寸: {total_w}x{total_h}")

    streammux2 = make_elm("nvstreammux", "Stream-muxer-display")
    streammux2.set_property("batch-size", num_sources)  # 新舊版 mux 皆支援

    if os.environ.get("USE_NEW_NVSTREAMMUX") == "yes":
        # 新版 mux：不接受 width/height/live-source 等舊屬性，改用 config_mux.txt。
        # 來源皆 1080p、真正拼接由 tiler 完成，故新版 mux 不縮放也不影響顯示。
        _mux_cfg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_mux.txt"
        )
        if os.path.exists(_mux_cfg):
            streammux2.set_property("config-file-path", _mux_cfg)
        else:
            print(f"[WARNING] 找不到 {_mux_cfg}，顯示用 mux 將用內建預設值")
    else:
        # 舊版 mux：維持原本設定
        streammux2.set_property("width", 1920)
        streammux2.set_property("height", 1080)
        streammux2.set_property("batched-push-timeout", 70000)
        # ⭐ live-source 依主線是否有 live 來源決定，不可寫死 1
        streammux2.set_property("live-source", 1 if has_live_source else 0)
        streammux2.set_property("nvbuf-memory-type", 0)

    tiler = make_elm("nvmultistreamtiler", "nvtiler-display")
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", total_w)
    tiler.set_property("height", total_h)

    q_d1 = make_elm("queue", "q-display-1")

    nvvidconv = make_elm("nvvideoconvert", "convertor-display")
    nvvidconv.set_property("nvbuf-memory-type", 0)

    q_d2 = make_elm("queue", "q-display-2")
    q_d3 = make_elm("queue", "q-display-3")

    # ---- 路徑 A：NVIDIA 專用 sink（吃 NVMM 可直送；Jetson 視情況加 nvegltransform）----
    nv_sink = None
    use_egltransform = _is_jetson()
    if Gst.ElementFactory.find("nveglglessink") is not None:
        nv_sink = make_elm("nveglglessink", "nvvideo-renderer-display")
    elif Gst.ElementFactory.find("nv3dsink") is not None:
        nv_sink = make_elm("nv3dsink", "nvvideo-renderer-display")
        use_egltransform = False

    if nv_sink is not None:
        sink = nv_sink
        sink.set_property("sync", False)
        # ⭐ 顯示分支不要做 QoS drop，避免主線被反壓
        _safe_set(sink, "qos", False)
        if use_egltransform and Gst.ElementFactory.find("nvegltransform") is not None:
            transform = make_elm("nvegltransform", "nvegl-transform-display")
            elements = [streammux2, tiler, q_d1, nvvidconv, q_d2, transform, q_d3, sink]
        else:
            transform = None
            elements = [streammux2, tiler, q_d1, nvvidconv, q_d2, q_d3, sink]
        for elm in elements:
            pipeline.add(elm)
        streammux2.link(tiler)
        tiler.link(q_d1)
        q_d1.link(nvvidconv)
        nvvidconv.link(q_d2)
        if transform is not None:
            q_d2.link(transform)
            transform.link(q_d3)
        else:
            q_d2.link(q_d3)
        q_d3.link(sink)
        return streammux2

    # ---- 路徑 B：標準 GStreamer sink（dGPU / 純 WSLg / 無 NVIDIA sink）----
    # nvvideoconvert 先把畫面從 NVMM 轉到系統記憶體，再交給標準 sink 顯示。
    # 可用環境變數 DS_DISPLAY_SINK 指定要用哪個標準 sink（例如 ximagesink）。
    caps_sys = make_elm("capsfilter", "caps-display-sys")
    caps_sys.set_property("caps", Gst.Caps.from_string("video/x-raw, format=RGBA"))
    videoconv = make_elm("videoconvert", "videoconvert-display")

    forced = os.environ.get("DS_DISPLAY_SINK", "").strip()
    candidates = [forced] if forced else ["ximagesink", "glimagesink", "autovideosink"]
    std_sink = None
    for cand in candidates:
        if cand and Gst.ElementFactory.find(cand) is not None:
            std_sink = make_elm(cand, "nvvideo-renderer-display")
            print(f"[INFO] 使用標準顯示 sink：{cand}")
            break
    if std_sink is None:
        sys.exit(f"[ERROR] 找不到可用的顯示 sink（嘗試清單：{candidates}）")
    _safe_set(std_sink, "sync", False)
    sink = std_sink

    elements = [streammux2, tiler, q_d1, nvvidconv, caps_sys, videoconv, q_d2, sink]
    for elm in elements:
        pipeline.add(elm)
    streammux2.link(tiler)
    tiler.link(q_d1)
    q_d1.link(nvvidconv)
    nvvidconv.link(caps_sys)
    caps_sys.link(videoconv)
    videoconv.link(q_d2)
    q_d2.link(sink)

    return streammux2


# ==========================================
# 每路 cam 分支組裝（動態組合 save / show）
# ==========================================
def setup_cam_branch(pipeline, pad_index, cfg, demux, display_streammux, osd_probe_callback):
    """
    為單路 cam 建立完整下游分支。
    上游：demux.src_{pad_index}
    下游分支由兩個 YAML 旗標決定：
        save (output.save_output_video) : 寫檔
        show (display.show_window)      : 本地預覽

    流程：
        demux.src_N → queue → nvvideoconvert → caps(RGBA) → nvdsosd → tee
            tee ├─→ save 分支 (可選)
                └─→ show 分支 (可選，鏈到外部 display_streammux)
            若兩者皆關，後接 fakesink 吞掉 buffer。
    """
    i = pad_index
    src_pad = demux.get_request_pad(f"src_{i}")

    # === 共用前段：queue → RGBA 轉色 → OSD ===
    q_cam = make_elm("queue", f"q-cam-{i}")

    nvvidconv_osd = make_elm("nvvideoconvert", f"conv_osd_{i}")
    nvvidconv_osd.set_property("nvbuf-memory-type", 0)

    caps_osd = make_elm("capsfilter", f"caps_osd_{i}")
    caps_osd.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    nvosd_i = make_elm("nvdsosd", f"nvosd-{i}")
    # process-mode：Jetson 用 2（VIC/HW），dGPU/WSL 用 1（GPU）
    nvosd_i.set_property("process-mode", 2 if _is_jetson() else 1)

    for elm in [q_cam, nvvidconv_osd, caps_osd, nvosd_i]:
        pipeline.add(elm)

    src_pad.link(q_cam.get_static_pad("sink"))
    q_cam.link(nvvidconv_osd)
    nvvidconv_osd.link(caps_osd)
    caps_osd.link(nvosd_i)

    # OSD 探針（畫 FPS 等等）
    nvosd_i.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER,
        lambda pad, info, idx=i: osd_probe_callback(pad, info, idx),
        0
    )

    # === 分支開關 ===
    cam_save = cfg.get("output", {}).get("save_output_video", False)
    cam_show = cfg.get("display", {}).get("show_window", True)
    is_file = cfg.get("is_file_source", False)

    enabled_branches = sum([cam_save, cam_show])

    # ---------- 0 個分支：直接 fakesink ----------
    if enabled_branches == 0:
        fake = make_elm("fakesink", f"fake-{i}")
        fake.set_property("sync", False)
        fake.set_property("async", False)
        pipeline.add(fake)
        nvosd_i.link(fake)
        return

    # ---------- 1 個分支：不用 tee，直接 link ----------
    if enabled_branches == 1:
        if cam_save:
            if is_file:
                nvosd_i.link(_build_save_branch_for_file(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
            else:
                nvosd_i.link(_build_save_branch_for_rtsp(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
        elif cam_show:
            _link_show_branch(pipeline, i, nvosd_i, display_streammux)
        return

    # ---------- 存檔 + 顯示同時：用 tee 分流 ----------
    tee = make_elm("tee", f"tee-{i}")
    pipeline.add(tee)
    nvosd_i.link(tee)

    if cam_save:
        q_s = make_elm("queue", f"q-s-{i}")
        pipeline.add(q_s)
        tee.link(q_s)
        if is_file:
            q_s.link(_build_save_branch_for_file(pipeline, i, cfg["video_path"], cfg["stream_fps"]))
        else:
            q_s.link(_build_save_branch_for_rtsp(pipeline, i, cfg["video_path"], cfg["stream_fps"]))

    if cam_show:
        _link_show_branch(pipeline, i, tee, display_streammux)


def _link_show_branch(pipeline, i, upstream, display_streammux):
    """
    建立本地顯示子分支：queue → nvvideoconvert → display_streammux.sink_N
    upstream 可能是 nvosd（單一分支）或 tee（多分支）。
    """
    q_d = make_elm("queue", f"q-d-{i}")
    nv_d = make_elm("nvvideoconvert", f"nv-d-{i}")
    nv_d.set_property("nvbuf-memory-type", 0)

    pipeline.add(q_d)
    pipeline.add(nv_d)

    upstream.link(q_d)
    q_d.link(nv_d)
    nv_d.get_static_pad("src").link(display_streammux.get_request_pad(f"sink_{i}"))
