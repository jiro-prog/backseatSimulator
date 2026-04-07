import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datetime
import logging
import msvcrt
import queue
import sys
import threading
import time

import yaml

from ai.analyzer import AIAnalyzer, load_model  # torch を PyQt5 より先に読み込む
from capture.screen import ScreenCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def capture_loop(screen_capture: ScreenCapture, image_queue: queue.Queue,
                 config: dict, pause_event: threading.Event):
    """キャプチャスレッド: 一定間隔でスクリーンショットを取得。"""
    interval = config.get("capture_interval", 15)
    enable_focus = config.get("enable_focus", True)
    while True:
        if not pause_event.is_set():
            try:
                if enable_focus:
                    result = screen_capture.capture_with_focus()
                else:
                    img = screen_capture.capture()
                    result = {"full_image": img, "focus_image": None,
                              "focus_label": None, "focus_diff": 0.0,
                              "window_title": screen_capture._get_window_title()} if img else None

                if result is not None:
                    # 古い画像を捨てて最新のみ保持
                    while not image_queue.empty():
                        try:
                            image_queue.get_nowait()
                        except queue.Empty:
                            break
                    image_queue.put(result)
                    logger.info("キャプチャ完了、AI分析キューに追加 (focus=%s)",
                                result.get("focus_label"))
                else:
                    logger.debug("画面変化なし、スキップ")
            except Exception:
                logger.exception("キャプチャスレッドでエラー")
        time.sleep(interval)


def ai_loop(ai_analyzer: AIAnalyzer, image_queue: queue.Queue,
            comment_queue: queue.Queue):
    """AIスレッド: 画像を受け取りコメント生成。"""
    while True:
        try:
            data = image_queue.get()  # ブロッキング
            logger.info("AI分析開始...")

            comments = ai_analyzer.analyze(
                full_image=data["full_image"],
                focus_image=data.get("focus_image"),
                window_title=data.get("window_title", ""),
            )
            if comments:
                comment_queue.put(comments)
                logger.info("コメント生成: %d個 (comment_queue=%d)", len(comments), comment_queue.qsize())
                # コメントログをファイルに記録
                try:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    focus_info = data.get("focus_label") or "-"
                    focus_diff = data.get("focus_diff", 0.0)
                    scene = ai_analyzer._prev_scene or "-"
                    win_title = data.get("window_title", "")
                    with open("comments.log", "a", encoding="utf-8") as f:
                        f.write(f"[{now}] focus={focus_info} diff={focus_diff:.3f} window={win_title} scene={scene}\n")
                        for c in comments:
                            f.write(f"[{now}]   {c.get('text', '')}  (color: {c.get('color', '')})\n")
                except Exception:
                    logger.exception("コメントログ書き込み失敗")
            else:
                logger.info("コメントなし")
        except Exception:
            logger.exception("AIスレッドでエラー")


def main():
    # 多重起動防止
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lock")
    lock_file = open(lock_path, "w")
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("BackseatSimulator は既に起動しています。")
        sys.exit(1)

    # 設定読み込み
    try:
        with open("config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("config.yaml が見つかりません。デフォルト設定で起動します。")
        config = {}

    # モデルロード（AIスレッド起動前）
    print("モデルをロード中...")
    try:
        model, processor = load_model(config)
        print("モデルロード完了")
        os.environ["HF_HUB_OFFLINE"] = "1"  # ロード完了後に外部通信を遮断
    except Exception as e:
        print(f"モデルロード失敗: {e}")
        logger.exception("モデルロード失敗")
        sys.exit(1)

    # キュー作成
    image_queue: queue.Queue = queue.Queue(maxsize=1)
    comment_queue: queue.Queue = queue.Queue()

    # コンポーネント初期化
    screen_capture = ScreenCapture(config)
    ai_analyzer = AIAnalyzer(config, model, processor)

    # 一時停止イベント
    pause_event = threading.Event()

    # キャプチャスレッド起動
    cap_thread = threading.Thread(
        target=capture_loop,
        args=(screen_capture, image_queue, config, pause_event),
        daemon=True,
    )
    cap_thread.start()

    # AIスレッド起動
    ai_thread = threading.Thread(
        target=ai_loop,
        args=(ai_analyzer, image_queue, comment_queue),
        daemon=True,
    )
    ai_thread.start()

    # GUI（PyQt5はtorchロード後にimport — DLL競合回避）
    from PyQt5.QtWidgets import QApplication
    from overlay.window import OverlayWindow
    from tray.system_tray import SystemTray

    app = QApplication(sys.argv)
    overlay = OverlayWindow(config)
    overlay.comment_queue = comment_queue
    overlay.show()

    # システムトレイ
    def change_persona(key: str):
        ai_analyzer.persona = key
        logger.info("ペルソナ変更: %s", key)

    def resume():
        pause_event.clear()
        ai_analyzer.reset_scene()

    def restart():
        logger.info("再起動します...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    tray = SystemTray(
        app=app,
        config=config,
        on_pause=lambda: pause_event.set(),
        on_resume=resume,
        on_quit=app.quit,
        on_persona_change=change_persona,
        on_restart=restart,
    )

    logger.info("BackseatSimulator 起動完了")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
