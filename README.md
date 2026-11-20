# DeepStream 路口轉向計數（TMC）

以 DeepStream 7.1 同時處理多路攝影機（RTSP 或影片檔），偵測、追蹤車輛，統計每台車從哪個路段進、往哪個路段出（O-D），結果寫入 SQLite。
Jetson（JetPack 6）與桌機 dGPU / WSL2 共用同一份程式碼。

本專案與「車流計數」（DeepStream-Multi-Model）共用部分程式碼：`logic/` 底下 `common.py`、`color.py`、`boxmot_adapter.py`、`pipeline.py`、`sources.py` 兩邊內容完全相同，修改後直接複製到另一邊；其餘檔案因統計方式不同，各自維護（見 8.1）。

---

## 一、快速開始

### 1.1 執行

```bash
conda activate tracking
python traffic_count_txt.py    # 依 ds_yaml 產生設定檔
python main.py                 # 啟動
```

- `ds_yaml` 有任何變動都要重跑 `traffic_count_txt.py`，否則 `main.py` 會拒絕啟動。
- 無終端機執行：`python main.py < /dev/null`。
- 環境建置、編譯 `nvdsinfer_custom_impl_Yolo`、模型轉換見 `說明.txt`。

### 1.2 結束方式

| 方式 | 結束碼 |
|---|---|
| 按 `q`、Ctrl+C、`kill -INT` | 130 |
| `kill -TERM`、`systemctl stop` | 0 |
| 影片檔全部播完 | 0 |
| 嚴重錯誤、啟動檢查失敗 | 1 |

- 每種方式都會先結算畫面上的車輛、寫完 DB 才結束；`q` / Ctrl+C / SIGTERM 最多再等 10 秒讓存檔影片收尾。
- SIGTERM 回 0：systemd 把非 0 視為失敗。SIGINT 回 130：`run_batches.sh` 靠它停止後續批次。

---

## 二、目錄結構

| 路徑 | 說明 |
|---|---|
| `ds_yaml/*.yaml` | 每路攝影機一個檔。子資料夾不會被讀取，可暫放不用的設定 |
| `car_fp16.engine` | 車輛模型（固定使用此檔名，放專案根目錄） |
| `labels_car.txt` | 類別名稱，一行一類，行號 = class_id |
| `output_db/` | 計數結果，每路一個 `<source_id>.db` |
| `output_video/` | 疊框影片 `<source_id>_output.mkv` |
| `config_*_group0.txt` | 產生器輸出，勿手改 |

---

## 三、YAML 設定

### 3.1 規則

1. 一個檔案 = 一路攝影機；新增攝影機就複製一個現有的檔案修改。
2. 欄位整行刪掉時使用表中的預設值；預設為「—」的不可省略。
3. 標示「共用」的欄位，只有**檔名排序第一個**檔案的值有效。
4. 點位類欄位以 `geometry.base_w × base_h` 標記，程式自動換算成 1920×1080。

### 3.2 欄位

#### 3.2.1 來源

| 欄位 | 預設 | 說明 |
|---|---|---|
| `source` | — | `rtsp://帳號:密碼@IP:埠/路徑`，或影片檔路徑（相對於專案根目錄）。密碼含 `@` 可直接寫 |
| `source_id` | 檔名 | 攝影機代號 → DB `CameraCode`、DB 檔名、影片檔名 |
| `device.code` | `UNKNOWN` | 邊緣機代號 → DB `DeviceCode` |
| `stream_fps` | 30 | RTSP 的 FPS，用於換算時間。影片檔自動讀取，不看此值 |
| `start_time` | 無 | 影片首幀的真實時間 `"YYYY-MM-DD HH:MM:SS"` → DB `CollectTime`。只對影片檔有效；不填則用系統時間 |

#### 3.2.2 模型

| 欄位 | 預設 | 說明 |
|---|---|---|
| `keep_classes` | 全部 | 要保留的 class_id，例如 `[2, 6]`；其餘不追蹤、不計數、不顯示 |
| `weight_batch_size` | 無 | 備援值，平常不生效（程式直接讀 engine）；只有環境沒有 `tensorrt` 時才用，此時必填 |
| `weight_imgsz` | 640 | 同上，模型輸入邊長 |
| `detect.car_conf` | 0.25 | 信心值門檻。共用 |
| `detect.car_iou` | 0.45 | NMS IoU 門檻。共用 |

engine 固定為 `car_fp16.engine`，YAML 不需要 `weight` 欄位（寫了也會被忽略）。

#### 3.2.3 區域

| 欄位 | 預設 | 說明 |
|---|---|---|
| `geometry.regions` | — | 每個 ROI = 路口的一個分支，`路段ID: [[x,y], …]`，至少 3 點。路段 ID → DB `FromRoadID` / `ToRoadID` |
| `geometry.region_names` | ROI 名稱 | `路段ID: "路名"` → DB `FromRoadName` / `ToRoadName` |
| `geometry.crop_points` | 整張畫面 | 送進模型的範圍（取外接矩形）。裁掉無關區域可讓小物件變大 |
| `geometry.base_w` / `base_h` | 1920 / 1080 | 標記上面點位時的畫面解析度 |

#### 3.2.4 計數規則

| 欄位 | 預設 | 說明 |
|---|---|---|
| `track_logic.min_roi_hits` | 2 | ROI 內至少幾幀才算經過，常用 1 ~ 3 |
| `session.cleanup_frames` | 30 | 車輛連續消失幾幀後結算 |
| `session.flush_interval_seconds` | 30 | 每幾秒批次寫入 DB；0 = 立即 |

#### 3.2.5 輸出與顯示

| 欄位 | 預設 | 說明 |
|---|---|---|
| `output.save_output_db` | `true` | `false` 時只在終端機印結果 |
| `output.output_db_dir` | `output_db` | DB 資料夾（舊名 `output_excel_dir` 仍可用） |
| `output.save_output_video` | `false` | 存疊框影片 |
| `output.output_video_dir` | `output_video` | 影片資料夾 |
| `display.show_window` | `true` | 預覽視窗；所有攝影機併成一個視窗 |
| `display.show_fps_overlay` | `true` | 左上角 FPS |
| `display.show_roi` | `true` | 畫 ROI |
| `display.show_crop` | `false` | 畫裁切範圍。只要一路開，全部都畫 |
| `tracker.type` | `nvdcf` | 追蹤器，共用。可用值見 4.4 |

---

## 四、運作原理

### 4.1 Pipeline

所有攝影機共用一條：

```
uridecodebin x N
 -> nvstreammux
 -> nvdspreprocess
 -> nvinfer
 -> nvtracker        (nvdcf only)
 -> nvdsanalytics
 -> nvstreamdemux
 -> per camera: nvdsosd -> file / window
```

| 元件 | 作用 |
|---|---|
| nvstreammux | 所有來源縮放成 1920×1080；ROI、裁切框、物件座標都在這個座標系 |
| nvdspreprocess | 依 `crop_points` 裁切、縮放成 engine 輸入 |
| nvinfer | 推論；使用 BoxMOT 時，追蹤在它輸出端的探針裡做 |
| nvtracker | 只有 `tracker.type: nvdcf` 才有 |
| nvdsanalytics | 只負責畫 ROI；ROI 命中在探針（`logic/probes.py`），O-D 結算在 `logic/state_db.py` |

攝影機數不必等於 engine 的 batch，多於 batch 時自動分批推論。

### 4.2 轉向判定（O-D）

1. **參考點**：bbox 底邊中點。
2. **ROI 命中**：參考點在 ROI 內的每一幀，該 ROI 命中數 +1，記下第一次命中的幀號，並為當下類別投一票。
3. **結算**：連續 `cleanup_frames` 幀沒出現時結算：
   - 命中數 ≥ `min_roi_hits` 的 ROI 才算經過。
   - 有效 ROI 少於 2 個 → 不記錄。
   - 依第一次命中的先後排序：最早 = 來向、最晚 = 去向；全部依序寫進 `Path`（如 `1600001>1600003>1600004`）。
   - 一台車最多一筆；類別取票數最多者。
4. 迴轉回原路段（A → B → A）會記成 A → B，因為排序看的是「第一次」命中。
5. 程式結束時，畫面上剩下的車輛依同樣規則立即結算（log 標示 `[結算-強制]`）。

### 4.3 RTSP 斷線重連

1. 一律走 TCP，延遲緩衝 2 秒。
2. 以下情況重建該路來源，其他攝影機不受影響：
   - 超過 20 秒沒有畫面。
   - 滴幀：近 60 秒平均低於 1 FPS。
   - 來源報錯：5 秒後重連，連續失敗時間隔遞增，最多 60 秒。
3. RTSP 的錯誤不會讓程式結束。

### 4.4 追蹤器

| `tracker.type` | 說明 |
|---|---|
| `nvdcf` | DeepStream 內建（GPU） |
| `bytetrack`、`ocsort`、`fasttracker`、`sfsort`、`cbiou` | BoxMOT（CPU），每路一個獨立追蹤器 |

參數檔：nvdcf 為 `config_tracker_NvDCF_accuracy.yml`，BoxMOT 為 `boxmot/configs/trackers/<type>.yaml`。

### 4.5 預覽視窗

1. 視窗卡住只會讓預覽掉幀，不影響推論、計數、存檔。
2. 沒有圖形顯示（未設 `DISPLAY`，例如 SSH）時自動不開視窗；要看畫面先 `export DISPLAY=:0`。

---

## 五、輸出

### 5.1 資料庫

每路一個 `output_db/<source_id>.db`，資料表 `AiTrafficFlowRawData`，一筆 = 一台車的一次轉向。

| 欄位 | 內容 |
|---|---|
| `DeviceCode` | `device.code` |
| `CameraCode` | `source_id` |
| `TrackID` | 車輛編號，與畫面上的 `ID` 相同；每路攝影機各自從 1 起算 |
| `DetectClass` | 類別名稱 |
| `FromRoadID` / `FromRoadName` | 來向路段 ID / 路名 |
| `ToRoadID` / `ToRoadName` | 去向路段 ID / 路名 |
| `Path` | 經過的所有路段，依先後以 `>` 連接 |
| `VideoTime` | 從啟動 / 影片開頭起算的時間 `HH:MM:SS` |
| `CollectTime` | RTSP：寫入時的系統時間；影片：`start_time` + 影片秒數 |

`kill -9` 會遺失最後 `flush_interval_seconds` 秒的紀錄。

### 5.2 產生的設定檔

| 檔案 | 內容 |
|---|---|
| `config_infer_group0.txt` | nvinfer：engine、batch、門檻。第一行是 ds_yaml 指紋，`main.py` 以此判斷是否過期 |
| `config_preprocess_group0.txt` | nvdspreprocess：tensor 形狀、裁切框 |
| `config_nvdsanalytics_group0.txt` | nvdsanalytics：ROI |

batch 與輸入尺寸直接讀 engine，不看 YAML。舊版的 `deepstream_app_config.txt`、`config_infer_primary_yolo11.txt`、`config_preprocess.txt`、`config_nvdsanalytics.txt`、`config_tracker_runtime.txt`、`config_mux.txt` 已不使用，執行產生器時會自動刪除。

---

## 六、部署

### 6.1 服務與腳本

| 項目 | 說明 |
|---|---|
| `deepstream.service` | systemd 服務，執行 `main.py` |
| `PutAPI.py` | 排程執行：讀 `output_db/*.db` 上傳到 API，成功後刪除已上傳的列 |
| `PutAPI_backup.py` | `PutAPI.py` 的舊版 |
| `status.py` | 終端機管理面板（看 log、啟停服務） |
| `ds_guard.py` | 守護腳本：服務出問題時自動重啟或重開機（見 6.2） |

`PutAPI.py` 依 `CollectTime`（`YYYY-MM-DD HH:MM:SS`）記錄進度，改 DB 欄位或時間格式前必須一併修改它。

### 6.2 守護腳本 ds_guard.py

只在邊緣機使用：由 root 定時執行，`deepstream.service` 出問題時自動重啟服務或重開機。兩個專案共用同一份。
輸出需導向腳本旁的 `ds_guard.log`，每日重開次數的上限靠這個檔計算。

#### 6.2.1 判斷規則

每輪依序檢查，命中一項就處理並結束本輪：

| 順序 | 條件 | 處置 |
|---|---|---|
| 1 | log 出現記憶體配置失敗（`NvMapMemAllocInternalTagged … error 12` 等） | 重開機：NVMM 配置不到時重啟服務無效 |
| 2 | 某路攝影機連續重連 50 次仍沒恢復（約 20 ~ 45 分鐘） | 重啟服務 |
| 3 | 系統可用記憶體低於 5%（服務啟動滿 10 分鐘後才判斷） | 重啟服務 |

1. 服務是手動停止的（`inactive`）時完全不處理。
2. 重開機的限制：開機滿 30 分鐘才會重開，24 小時內最多 3 次。
3. 重開機前會把記憶體狀態（`free`、`buddyinfo`、NVMM 各行程用量、記憶體前 5 名行程）寫進 `ds_guard.log`，供事後查原因。
4. 門檻都是檔案開頭的常數，每個常數旁有說明與調整的影響。

#### 6.2.2 與 main.py 的關係

條件 2 靠 `main.py` 的兩種 log 判斷（格式定義在 `logic/sources.py` 檔頭）：

```
[WARNING] [<source_id>] <原因>，重新連線（第 N 次）...
[INFO] [<source_id>] 已恢復畫面（先前連續重連 N 次）
```

修改這兩行的格式時，要一併修改 `ds_guard.py` 的 `RE_RECONNECT` / `RE_RECOVER`。

### 6.3 移到新機器

1. 不寫死任何路徑：專案路徑以程式所在目錄為準；DeepStream 安裝位置自動偵測（見 8.4）。
2. 要在新機器上重做：轉換 engine、編譯 `nvdsinfer_custom_impl_Yolo`、執行 `traffic_count_txt.py`。這些不能從其他機器複製。
3. systemd 的 service 檔、crontab 若寫了路徑，要自行修改。

---

## 七、疑難排解

| 現象 | 處理 |
|---|---|
| `設定檔與目前的 ds_yaml 不一致` | 重跑 `python traffic_count_txt.py` |
| `找不到 engine` | 專案根目錄沒有 `car_fp16.engine` |
| `GStreamer 找不到 nvinfer` | DeepStream 沒裝好；確認 `gst-inspect-1.0 nvinfer` 有輸出，或設 `DS_LIB_ROOT` |
| 某路 `0.00 FPS（無畫面）`、`重新連線` | RTSP 斷線，會自動重試。持續失敗請用 VLC / PotPlayer 以 TCP 測試網址 |
| `已 N 秒沒有新畫面` 並列出 queue | 標示「滿」的 queue 的下游卡住；沒列出任何 queue = 來源沒送資料 |
| 有框但沒有紀錄 | 車輛要在 2 個以上 ROI 各命中 `min_roi_hits` 幀；檢查 ROI 是否涵蓋 bbox 底邊中點 |
| `沒有 region_names` 警告 | 該 ROI 沒有路名對照，DB 路名欄位改填路段 ID |

可忽略的啟動訊息：`Failed to query video capabilities`、`Implicit layer support has been deprecated`、`gst-plugin-scanner … Failed to load plugin`、`Invalid mode selected, Falling back to GPU`（Jetson）。

---

## 八、開發說明

### 8.1 程式碼地圖

「共用」= 與車流計數專案內容完全相同；「本專案」= 因統計方式不同而各自維護。

| 檔案 | 來源 | 職責 |
|---|---|---|
| `logic/common.py` | 共用 | 座標換算、讀 engine 規格、找 DeepStream |
| `logic/color.py` | 共用 | labels 與顏色 |
| `logic/boxmot_adapter.py` | 共用 | BoxMOT 介接 |
| `logic/pipeline.py` | 共用 | 元件建構、預覽視窗、存檔 |
| `logic/sources.py` | 共用 | RTSP 來源、斷線重連 |
| `logic/config.py` | 本專案 | 讀 YAML（固定 engine、`region_names`、每路一個 DB） |
| `logic/probes.py` | 本專案 | 追蹤、ROI 命中與先後、畫面標註 |
| `logic/state_db.py` | 本專案 | O-D 結算、`AiTrafficFlowRawData` 寫入 |
| `main.py` | 本專案 | 建立 pipeline、結束流程、定時檢查（與車流計數只差檔頭說明） |
| `traffic_count_txt.py` | 本專案 | 產生設定檔（與車流計數只差檔頭說明與舊檔清單） |
| `ds_guard.py` | 共用 | 邊緣機守護腳本（見 6.2） |
| `run_batches.sh` | | 大量影片檔分批依序執行 |
| `onnx2engine.py` | | `.onnx` → `.engine` |

### 8.2 環境變數

| 變數 | 作用 |
|---|---|
| `DS_YAML_DIR` | 改讀其他 YAML 資料夾（產生器與 `main.py` 要用同一個） |
| `DS_RECONNECT_SECONDS` | 多久沒畫面就重連，預設 20；0 = 停用 |
| `DS_LIB_ROOT` | 手動指定 DeepStream 的 lib 目錄（自動偵測失敗時才用） |
| `DS_FORCE_DISPLAY` | 設 `1`：偵測不到 `DISPLAY` 也開視窗 |
| `DS_DISPLAY_SINK` | 指定顯示 sink，例如 `ximagesink` |
| `USE_CPU_ENCODER` | 存檔編碼器：`1` CPU、`0` NVENC；預設自動 |

### 8.3 修改前必讀

以下每項都曾造成難查的問題：

1. batch 與輸入尺寸必須等於 engine 實際值，所以從 engine 讀，不給手填。
2. RTSP 參數要用元件型別（`GstRTSPSrc`）判斷，名稱判斷會讓 TCP 設定失效。
3. 顯示分支的 queue 容量必須小於 4，且 nvvideoconvert 要關直通，否則視窗一卡推論就停。
4. RTSP 斷線的 EOS 必須攔下，否則該路永久黑畫面。
5. 被重連拆掉的舊來源仍會報錯，要從錯誤訊息的元件路徑認出來並忽略。
6. 開視窗前要先呼叫 `XInitThreads()`，否則多視窗可能死鎖。

### 8.4 平台差異

1. Jetson / 桌機差異（顯示 sink、OSD 模式、編碼器）由 `pipeline.py` 的 `is_jetson()`、`safe_set()` 處理。
2. DeepStream 位置：向 GStreamer 查 `nvinfer` plugin 檔，解開符號連結後往上兩層即為 lib 目錄（`common.find_ds_lib_root()`）。
3. 存檔預設用 NVENC，沒有時自動改用 CPU x264。
