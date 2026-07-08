# DeepStream 路口轉向計數 (TMC) 專案 — 功能總結

基於 **DeepStream 7.1**（TensorRT 10.3），做路口 **O-D 轉向量統計**（車輛從哪個路段進、往哪個路段出），支援多路同時運算，跨平台自動適配（Jetson / dGPU / WSL2）。以下依模組列出功能。

---

## 一、O-D 轉向判定（核心）

* 多 ROI 對應路段：每個 ROI 代表一個路口分支，`region_names` 把 ROI id 對到路名（寫進 DB）。
* 來向 / 去向：一台車經過的 ROI 依「首次命中的 frame 編號」排序，**最早進入=來向（From）、最後進入=去向（To）**。
* 完整路徑：中間經過的 ROI 不入 From/To 欄，保留在 `Path`（如 `1600001>1600003>1600004`）供分析。
* 抖動過濾：`track_logic.min_roi_hits` 命中某 ROI 未達幀數視為擦邊噪音，不算經過。
* 消失才結算：ID 連續 `cleanup_frames` 幀未再出現才寫一筆；結束時強制結算殘留。

## 二、推論架構

* 動態組裝：`PGIE 車輛偵測 → [tracker] → nvdsanalytics`；BoxMOT 模式跳過 nvtracker，改在 `pgie.src` 探針接管追蹤。
* preprocess 裁切：`nvdspreprocess` 依 `crop_points` 只把 ROI 內畫面送進 PGIE，省算力。
* 設定檔自動產生：`traffic_count_txt.py` 讀 YAML 產出 6 份設定檔（preprocess / infer / analytics / tracker runtime / mux / app）。

## 三、追蹤器

* 雙模式：`nvdcf`（內建）或 BoxMOT（`bytetrack / ocsort / fasttracker / sfsort / cbiou`），由 `tracker.type` 決定。
* 追蹤器 `.so` 路徑自動偵測：`DS_TRACKER_LIB` → 常見路徑（含 `/opt/nvidia`、`/opt/thi`）→ glob 掃版本 → 保底。

## 四、解析度自動換算（mux 感知）

* 舊版 mux（`USE_NEW_NVSTREAMMUX=no`）：畫面統一縮到 1920×1080 → ROI/crop 依 `base_w/base_h` 自動換算成 1080P 座標，混合解析度來源（720P/4K）也對得齊。
* 新版 mux（預設 `yes`）：畫面保持原生解析度 → YAML 原生點位天然對齊，不縮放。
* 兩種模式的 `config.py`（執行期）與 `traffic_count_txt.py`（產生期）行為一致。

## 五、資料庫（每路獨立）

* 每路一個 DB：各 cam 各自寫進 `output_db/<source_id>.db`（非合併單檔），WAL 模式支援併發寫入。
* 表 `AiTrafficFlowRawData` 欄位：`DeviceCode / CameraCode / TrackID / DetectClass / FromRoadID / FromRoadName / ToRoadID / ToRoadName / Path / RoiCount / VideoTime / CollectTime`。
* CollectTime 雙模式：檔案 = `start_time + 影片虛擬秒數`；RTSP = 系統當下時間。
* `save_output_db=false`：只印 log、不開連線、零 DB IO。
* local_id 循環：每路各自累積到上限後歸 1（撞號靠 CollectTime 區分）。

## 六、跨平台 / 輸入 / 輸出 / 顯示

* 編碼器自動偵測：有 NVENC 走硬體編碼，否則退回 CPU（x264）；`USE_CPU_ENCODER=1/0` 可覆寫。
* 顯示 sink 退路：NVIDIA sink 找不到時退回 `ximagesink` / `glimagesink` / `autovideosink`，可用 `DS_DISPLAY_SINK` 指定。
* 來源類型：本地影片檔、RTSP、HTTP。
* HEADLESS：`run_batches.sh` 的 `HEADLESS=0/1`，或把 YAML `show_window` 全設 false。
* OSD 疊加：bbox 車種色與 ID、左上角 FPS（`show_fps_overlay`）、ROI 黃線（`show_roi`）、裁切框青綠線（`show_crop`）。

## 七、RTSP 斷線重連 / 優雅中斷

* 第一層：nvurisrcbin 內建 rtsp-reconnect（間隔重試、無限次），處理「有斷線錯誤」情況。
* 第二層：看門狗每 10 秒檢查各 RTSP 路最後吐幀時間，卡死 60 秒即單路重啟（其他路不受影響）。
* 只監控 RTSP 路：檔案來源播完不吐幀是正常現象，不會被誤重啟重播。
* 優雅中斷：按 `q` / SIGINT / `systemctl stop`(SIGTERM) → 送 EOS → 看門狗立即停止 → 影片封裝完才退出（收線逾時有強制退出保險）。無頭 / systemd 模式用訊號即可安全收尾。

---

## 執行流程

```bash
python traffic_count_txt.py   # 改過 ROI / crop / engine / tracker 後都要先重跑
python main.py                # 啟動計數
# 或
HEADLESS=1 ./run_batches.sh   # 批次（可切 HEADLESS）
```

> 無 NVENC 環境（如部分 WSL）走 CPU 編碼，需先裝 GStreamer 外掛：
> `sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav`

## 檔案結構

| 檔案 | 職責 |
|------|------|
| `main.py` | 建 pipeline、掛探針、mainloop、RTSP 重連 / 看門狗 / 安全退出 |
| `traffic_count_txt.py` | 讀 YAML 產 6 份設定檔（含 ROI/crop mux 感知換算、路徑自動偵測） |
| `logic/config.py` | 載入 YAML、`SOURCE_CONFIGS`、cv_regions、mux 感知座標對齊、追蹤器模式 |
| `logic/pipeline.py` | 元件建構、編碼器自動偵測、顯示 sink 退路、每路下游分支 |
| `logic/probes.py` | 追蹤探針（nvdcf / BoxMOT）、多 ROI 首次命中排序、OSD |
| `logic/state_db.py` | 每路獨立 SQLite、O-D 結算與寫入、local_id 管理 |
| `logic/boxmot_adapter.py` | BoxMOT 追蹤器介接 |
| `PutAPI.py` | DB 資料 API 上傳 |
| `ds_yaml/*.yaml` | 每路 cam 設定（來源、ROI、路名、追蹤器等） |

## YAML 重點欄位

```yaml
source_id: "jinde_chaolong_intersection_cam07"  # DB CameraCode / OSD / 輸出檔名前綴
device: {code: "ptits_dongang_edge05"}          # DB DeviceCode
source: "rtsp://user:pass@ip:port/path"          # 影片檔 / rtsp:// / http://
stream_fps: 15.0

geometry:
  base_w: 1920                    # 來源真實解析度（舊版 mux 縮放依據）
  base_h: 1080
  regions:                        # 每個 ROI = 一個路口分支
    1600001: [[0,405],[473,266],[920,870],[687,1033],[0,1033]]
  region_names:                   # ROI id → 路名（寫進 From/ToRoadName）
    1600001: "朝隆路(西)"
  crop_points: [[0,45],[1920,45],[1920,1033],[0,1033]]   # 裁切遮罩

session:
  cleanup_frames: 60              # 消失 N 幀後結算
  flush_interval_seconds: 0       # 0=即時寫 DB

track_logic:
  min_roi_hits: 1                 # 命中 ROI 最少幀數（過濾擦邊）

output:  {save_output_video: false, save_output_db: true, output_db_dir: "output_db"}
display: {show_window: false, show_fps_overlay: true, show_roi: true, show_crop: true}
tracker: {type: "fasttracker"}    # nvdcf / bytetrack / ocsort / fasttracker / sfsort / cbiou
```

## 與 LPR / 車流計數版的差異

| 面向 | 本專案（TMC 轉向） | LPR 車牌 | 多權重車流計數 |
|------|-------------------|----------|----------------|
| 主要產出 | O-D 轉向（From→To 路段 + Path） | 車牌字串 + 車種 | 分流計數 |
| DB 表 | `AiTrafficFlowRawData`（每路一 .db） | 每路一 .db | 合併單一 .db |
| 方向表示 | 來向/去向 ROI（路段名） | IN / OUT | flow_in / flow_out |
| 共通 | 解析度換算、跨平台編碼 / 顯示、HEADLESS、RTSP 重連 + 看門狗、追蹤器路徑自動偵測 | 同左 | 同左 |
