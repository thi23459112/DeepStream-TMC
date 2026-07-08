#!/usr/bin/env python3
# =============================================================
#  DeepStream 即時管理面板 （自動偵測路徑版本）
#  一執行就同時顯示：服務狀態 + 辨識 log + PutAPI 上傳 log
#  按鍵即時生效（免按 Enter）：
#     1 開啟 = 啟動辨識服務 + 恢復上傳排程
#     2 關閉 = 停止辨識服務 + 暫停上傳排程
#     q 離開
# =============================================================

import subprocess
import sys
import threading
import time
import select
import termios
import tty
from collections import deque
from pathlib import Path
import os

# -------------------- 檢查 Rich 套件 --------------------
try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Console
except ImportError:
    raise SystemExit(
        "缺少 rich 套件，請先在 tracking 環境安裝：\n"
        "    conda activate tracking && pip install rich"
    )


# =============================================================
#  自動偵測設定函數
# =============================================================
def auto_detect_project_dir():
    """
    自動偵測 DeepStream-TMC 專案目錄
    
    偵測順序：
    1. 環境變數 DS_PROJECT_DIR
    2. 從腳本所在目錄往上找（找 PutAPI.py 或 main.py）
    3. 檢查常見安裝位置
    4. 詢問使用者
    5. 使用預設值
    """
    # 1. 檢查環境變數
    env_dir = os.environ.get("DS_PROJECT_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)
    
    # 2. 從腳本所在目錄往上找
    script_dir = Path(__file__).resolve().parent
    for parent in [script_dir] + list(script_dir.parents):
        # 檢查是否有特徵檔案
        if (parent / "PutAPI.py").exists() or (parent / "main.py").exists():
            return parent
    
    # 3. 檢查常見安裝位置
    common_paths = [
        Path("/home/nvidia/THI/DeepStream-TMC"),
        Path.home() / "DeepStream-TMC",
        Path.cwd() / "DeepStream-TMC",
    ]
    for path in common_paths:
        if path.exists():
            return path
    
    # 4. 都找不到就問使用者
    print("\n⚠️  找不到專案目錄（找不到 PutAPI.py 或 main.py）")
    print(f"   目前腳本位置: {script_dir}")
    user_input = input("   請輸入專案絕對路徑（Enter 使用預設）: ").strip()
    if user_input:
        return Path(user_input)
    
    # 5. 最後的預設值
    return Path("/home/nvidia/THI/DeepStream-TMC")


def auto_detect_conda_sh():
    """
    自動偵測 conda.sh 位置
    
    偵測順序：
    1. 環境變數 CONDA_SH
    2. 從 CONDA_PREFIX 環境變數推測
    3. 檢查常見安裝位置
    4. 使用 which conda 指令
    5. 使用預設值
    """
    # 1. 檢查環境變數
    env_conda = os.environ.get("CONDA_SH")
    if env_conda and Path(env_conda).exists():
        return Path(env_conda)
    
    # 2. 從 conda 環境變數推測
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_sh = Path(conda_prefix) / "etc/profile.d/conda.sh"
        if conda_sh.exists():
            return conda_sh
    
    # 3. 檢查常見安裝位置
    home = Path.home()
    common_paths = [
        home / "miniconda3/etc/profile.d/conda.sh",
        home / "anaconda3/etc/profile.d/conda.sh",
        Path("/opt/conda/etc/profile.d/conda.sh"),
        Path("/usr/local/anaconda3/etc/profile.d/conda.sh"),
    ]
    for path in common_paths:
        if path.exists():
            return path
    
    # 4. 使用 which conda 指令
    try:
        result = subprocess.run(
            ["which", "conda"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0 and result.stdout:
            conda_base = Path(result.stdout.strip()).parent.parent
            conda_sh = conda_base / "etc/profile.d/conda.sh"
            if conda_sh.exists():
                return conda_sh
    except Exception:
        pass
    
    # 5. 預設值
    return Path("/home/nvidia/miniconda3/etc/profile.d/conda.sh")


def auto_detect_conda_env():
    """
    自動偵測 Conda 環境名稱
    
    偵測順序：
    1. 環境變數 CONDA_ENV
    2. 從 CONDA_DEFAULT_ENV 環境變數
    3. 使用預設值 "tracking"
    """
    # 1. 檢查環境變數
    env = os.environ.get("CONDA_ENV")
    if env:
        return env
    
    # 2. 從 CONDA_DEFAULT_ENV
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env and env != "base":
        return env
    
    # 3. 預設值
    return "tracking"


def auto_config():
    """
    自動偵測所有路徑設定並回傳字典
    
    Returns:
        dict: 包含所有設定的字典
    """
    project_dir = auto_detect_project_dir()
    conda_sh    = auto_detect_conda_sh()
    conda_env   = auto_detect_conda_env()
    
    return {
        "project_dir": project_dir,                           # 專案目錄
        "conda_sh": conda_sh,                                 # conda.sh 路徑
        "conda_env": conda_env,                               # conda 環境名稱
        "putapi_log": project_dir / "PutAPI.log",            # PutAPI 日誌檔
    }


# =============================================================
#  執行自動偵測並顯示結果
# =============================================================
CONFIG = auto_config()

# 設定變數（轉為字串以便後續使用）
PROJECT_DIR      = str(CONFIG["project_dir"])                 # 專案目錄
CONDA_SH         = str(CONFIG["conda_sh"])                   # conda.sh 路徑
CONDA_ENV        = CONFIG["conda_env"]                       # conda 環境名稱
PUTAPI_LOG       = str(CONFIG["putapi_log"])                 # PutAPI 日誌檔

# 固定設定
MAX_KEEP          = 500                                      # 日誌保留行數
PUTAPI_PANEL_ROWS = 20                                       # PutAPI 面板行數
CRON_MARK         = "# DEEPSTREAM_PUTAPI"                    # Crontab 標記
CRON_LINE = (                                                # Crontab 排程指令
    f"*/5 * * * * /bin/bash -c 'source {CONDA_SH} && conda activate {CONDA_ENV} "
    f"&& cd {PROJECT_DIR} && python PutAPI.py >> {PUTAPI_LOG} 2>&1' {CRON_MARK}"
)

# 建立 Console 物件並顯示偵測結果
console = Console()
console.print("\n[bold cyan]╔════════════════════════════════════════════════╗[/]")
console.print("[bold cyan]║    DeepStream 即時管理面板 - 自動偵測模式    ║[/]")
console.print("[bold cyan]╚════════════════════════════════════════════════╝[/]\n")

console.print(f"[green]✓[/] 專案目錄: [yellow]{PROJECT_DIR}[/]")
console.print(f"[green]✓[/] Conda 腳本: [yellow]{CONDA_SH}[/]")
console.print(f"[green]✓[/] Conda 環境: [yellow]{CONDA_ENV}[/]")
console.print(f"[green]✓[/] PutAPI 日誌: [yellow]{PUTAPI_LOG}[/]")
console.print("")


# =============================================================
#  全域變數與緩衝區
# =============================================================
recog_buffer   = deque(maxlen=MAX_KEEP)                     # 辨識服務日誌緩衝
putapi_buffer  = deque(maxlen=MAX_KEEP)                     # PutAPI 日誌緩衝
stop_event     = threading.Event()                          # 停止事件
putapi_state   = {"on": None}                               # PutAPI 狀態快取


# =============================================================
#  背景日誌讀取執行緒
# =============================================================
def recog_reader():
    """
    背景執行緒：持續讀取 DeepStream 服務日誌
    使用 journalctl -f 即時追蹤
    """
    try:
        # 啟動 journalctl 即時追蹤
        proc = subprocess.Popen(
            ["journalctl", "-u", "deepstream.service", "-o", "cat", 
             "-n", "80", "-f", "--no-pager"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        recog_buffer.append("[錯誤] 找不到 journalctl 指令")
        return
    
    # 持續讀取日誌直到收到停止訊號
    for line in proc.stdout:
        if stop_event.is_set():
            break
        recog_buffer.append(line.rstrip("\n"))
    
    # 終止程序
    try:
        proc.terminate()
    except Exception:
        pass


def putapi_reader():
    """
    背景執行緒：持續讀取 PutAPI.log 日誌
    使用 tail -F 即時追蹤
    """
    try:
        # 啟動 tail 即時追蹤
        proc = subprocess.Popen(
            ["tail", "-n", "40", "-F", PUTAPI_LOG],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        putapi_buffer.append("[錯誤] 找不到 tail 指令")
        return
    except Exception as e:
        putapi_buffer.append(f"[錯誤] 無法讀取日誌: {e}")
        return
    
    # 持續讀取日誌直到收到停止訊號
    for line in proc.stdout:
        if stop_event.is_set():
            break
        putapi_buffer.append(line.rstrip("\n"))
    
    # 終止程序
    try:
        proc.terminate()
    except Exception:
        pass


# =============================================================
#  服務狀態與 Crontab 操作函數
# =============================================================
def get_service_status() -> str:
    """
    取得 DeepStream 服務狀態
    
    Returns:
        str: 'active', 'failed', 'inactive', 或其他狀態
    """
    result = subprocess.run(
        ["systemctl", "is-active", "deepstream.service"],
        capture_output=True,
        text=True,
        check=False
    )
    return result.stdout.strip()


def get_root_crontab() -> str:
    """
    取得 root 的 crontab 內容
    
    Returns:
        str: crontab 內容，若無則回傳空字串
    """
    result = subprocess.run(
        ["sudo", "crontab", "-l"],
        capture_output=True,
        text=True,
        check=False
    )
    return result.stdout if result.returncode == 0 else ""


def set_root_crontab(content: str) -> None:
    """
    設定 root 的 crontab 內容
    
    Args:
        content: 完整的 crontab 內容
    """
    subprocess.run(
        ["sudo", "crontab", "-"],
        input=content,
        text=True,
        check=False
    )


def putapi_is_enabled() -> bool:
    """
    檢查 PutAPI 排程是否已啟用
    
    Returns:
        bool: True 表示已啟用，False 表示未啟用
    """
    return CRON_MARK in get_root_crontab()


def putapi_enable() -> None:
    """
    啟用 PutAPI 排程（加入 crontab）
    會移除舊的標記行再新增
    """
    # 讀取現有 crontab，過濾掉舊的 PutAPI 行
    lines = [l for l in get_root_crontab().splitlines() if CRON_MARK not in l]
    # 加入新的排程
    lines.append(CRON_LINE)
    # 寫回 crontab
    set_root_crontab("\n".join(lines) + "\n")
    # 更新快取
    putapi_state["on"] = True


def putapi_disable() -> None:
    """
    停用 PutAPI 排程（從 crontab 移除）
    """
    # 讀取現有 crontab，過濾掉 PutAPI 行
    lines = [l for l in get_root_crontab().splitlines() if CRON_MARK not in l]
    # 寫回 crontab（若無內容則寫空字串）
    set_root_crontab(("\n".join(lines) + "\n") if lines else "")
    # 更新快取
    putapi_state["on"] = False


# =============================================================
#  畫面渲染函數
# =============================================================
def colorize_log_line(line: str) -> str:
    """
    為日誌行加上顏色標記
    
    Args:
        line: 原始日誌行
    
    Returns:
        str: 加上 Rich 顏色標記的字串
    """
    # 跳脫方括號以避免 Rich 誤判
    safe = line.replace("[", "\\[")
    low = line.lower()
    
    # 錯誤訊息 → 紅色
    if "error" in low or "traceback" in low or "fail" in low:
        return f"[red]{safe}[/]"
    
    # 警告訊息 → 黃色
    if "warn" in low or "逾時" in line:
        return f"[yellow]{safe}[/]"
    
    # 成功/資訊訊息 → 綠色
    if "info" in low or "上傳成功" in line or "success" in low:
        return f"[green]{safe}[/]"
    
    # 一般訊息 → 白色
    return safe


def make_log_panel(buf, lines_n: int, title: str, color: str) -> Panel:
    """
    建立日誌面板
    
    Args:
        buf: 日誌緩衝區 (deque)
        lines_n: 要顯示的行數
        title: 面板標題
        color: 面板顏色
    
    Returns:
        Panel: Rich 面板物件
    """
    # 取出最後 N 行
    rows = list(buf)[-lines_n:]
    
    if rows:
        # 將每行加上顏色後組合成文字
        colored_lines = "\n".join(colorize_log_line(x) for x in rows)
        text = Text.from_markup(colored_lines)
    else:
        text = Text("（等待 log 輸出中...）", style="dim")
    
    return Panel(
        text,
        title=f"[{color}]{title}[/]",
        border_style=color
    )


def render_dashboard(term_height: int) -> Layout:
    """
    渲染完整儀表板
    
    Args:
        term_height: 終端機高度（行數）
    
    Returns:
        Layout: Rich 版面配置物件
    """
    # ----- 取得服務狀態 -----
    status = get_service_status()
    if status == "active":
        service_text = "[bold green]● 運作中 (active)[/]"
    elif status == "failed":
        service_text = "[bold red]✕ 失敗 (failed)[/]"
    else:
        service_text = f"[bold red]○ 已停止 ({status or '未知'})[/]"
    
    # ----- 取得 PutAPI 狀態 -----
    if putapi_state["on"] is True:
        putapi_text = "[bold green]● 已啟用 (每5分鐘)[/]"
    elif putapi_state["on"] is False:
        putapi_text = "[bold red]○ 已暫停[/]"
    else:
        putapi_text = "[dim]未知[/]"
    
    # ----- 建立標題面板 -----
    header = Panel(
        f"辨識服務：{service_text}        PutAPI 上傳排程：{putapi_text}",
        title="[cyan]DeepStream 即時管理面板[/]",
        border_style="cyan",
    )
    
    # ----- 建立底部快捷鍵提示 -----
    footer = Panel(
        "[green]1[/] 開啟(辨識+上傳)     [red]2[/] 關閉(辨識+上傳)     "
        "[dim]q[/] 離開",
        border_style="dim",
    )
    
    # ----- 計算各面板高度 -----
    recog_lines    = max(3, term_height - 6 - PUTAPI_PANEL_ROWS - 2)
    putapi_lines   = max(2, PUTAPI_PANEL_ROWS - 2)
    
    # ----- 建立日誌面板 -----
    recog_panel   = make_log_panel(
        recog_buffer,
        recog_lines,
        "📡 辨識即時日誌",
        "bright_blue"
    )
    putapi_panel  = make_log_panel(
        putapi_buffer,
        putapi_lines,
        "📤 PutAPI 上傳日誌",
        "magenta"
    )
    
    # ----- 組裝版面 -----
    layout = Layout()
    layout.split_column(
        Layout(header, size=3),
        Layout(recog_panel),
        Layout(putapi_panel, size=PUTAPI_PANEL_ROWS),
        Layout(footer, size=3),
    )
    return layout


# =============================================================
#  鍵盤輸入與控制函數
# =============================================================
def read_key(timeout: float = 0.3) -> str:
    """
    讀取單一按鍵（非阻塞）
    
    Args:
        timeout: 等待超時時間（秒）
    
    Returns:
        str: 按下的字元，若超時則回傳 None
    """
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return None


def suspend_run_and_execute(live, old_termios, func):
    """
    暫停 Live 畫面，還原終端機模式，執行函數，然後恢復
    
    這個函數用於需要使用者輸入（如 sudo 密碼）的操作
    
    Args:
        live: Live 物件
        old_termios: 原始終端機設定
        func: 要執行的函數
    """
    # 停止 Live 畫面更新
    live.stop()
    
    # 還原終端機模式（讓使用者可以輸入密碼）
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)
    
    try:
        # 執行指定的函數
        func()
    finally:
        # 等待一秒確保輸出完成
        time.sleep(1.0)
        
        # 重新設定為 cbreak 模式
        tty.setcbreak(sys.stdin.fileno())
        
        # 恢復 Live 畫面更新
        live.start()


def action_open():
    """
    執行「開啟」動作：
    1. 啟動 DeepStream 服務
    2. 啟用 PutAPI 排程
    """
    console.print("\n[green]🔓 開啟：啟動辨識服務 + 恢復上傳排程...[/]")
    
    # 啟動服務
    subprocess.run(
        ["sudo", "systemctl", "start", "deepstream.service"],
        check=False
    )
    
    # 啟用 PutAPI 排程
    putapi_enable()
    
    console.print("[green]✅ 完成。[/]")


def action_close():
    """
    執行「關閉」動作：
    1. 停止 DeepStream 服務
    2. 停用 PutAPI 排程
    """
    console.print("\n[yellow]🔒 關閉：停止辨識服務 + 暫停上傳排程...[/]")
    
    # 停止服務
    subprocess.run(
        ["sudo", "systemctl", "stop", "deepstream.service"],
        check=False
    )
    
    # 停用 PutAPI 排程
    putapi_disable()
    
    console.print("[yellow]✅ 完成。[/]")


# =============================================================
#  主程式
# =============================================================
def main():
    """
    主程式進入點
    """
    # ----- 啟動背景日誌讀取執行緒 -----
    threading.Thread(target=recog_reader, daemon=True).start()
    threading.Thread(target=putapi_reader, daemon=True).start()
    
    # ----- 初始化時讀取 PutAPI 狀態 -----
    console.print("[dim]讀取目前狀態中（可能需要輸入 sudo 密碼）...[/]")
    try:
        putapi_state["on"] = putapi_is_enabled()
    except Exception:
        putapi_state["on"] = None
    
    # ----- 設定終端機為 cbreak 模式 -----
    fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(fd)
    
    try:
        # 設定為 cbreak 模式（按鍵即時響應）
        tty.setcbreak(fd)
        
        # 啟動 Live 畫面
        with Live(
            console=console,
            screen=True,
            auto_refresh=False,
            refresh_per_second=4
        ) as live:
            
            # 主事件迴圈
            while True:
                # 更新畫面
                live.update(
                    render_dashboard(console.size.height),
                    refresh=True
                )
                
                # 讀取按鍵（非阻塞）
                key = read_key(0.3)
                if not key:
                    continue
                
                key = key.lower()
                
                # 處理按鍵
                if key == "1":
                    # 開啟
                    suspend_run_and_execute(live, old_termios, action_open)
                    
                elif key == "2":
                    # 關閉
                    suspend_run_and_execute(live, old_termios, action_close)
                    
                elif key == "q":
                    # 離開
                    break
                    
    finally:
        # ----- 清理資源 -----
        # 通知背景執行緒停止
        stop_event.set()
        
        # 還原終端機設定
        termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)


# =============================================================
#  程式進入點
# =============================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # 優雅處理 Ctrl+C
        print("\n[dim]使用者中斷[/]")
    except Exception as e:
        print(f"\n[red]錯誤: {e}[/]")
        sys.exit(1)
