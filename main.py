import datetime
import logging
import queue
import sys
import threading
import time

import yaml
from PyQt5.QtWidgets import QApplication

from ai.analyzer import AIAnalyzer
from capture.screen import ScreenCapture
from overlay.window import OverlayWindow
from tray.system_tray import SystemTray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def capture_loop(screen_capture: ScreenCapture, image_queue: queue.Queue,
                 config: dict, pause_event: threading.Event):
    """キャプチャスレッド: 一定間隔でスクリーンショットを取得。"""
    interval = config.get("capture_interval", 15)
    while True:
        if not pause_event.is_set():
            try:
                img = screen_capture.capture()
                if img is not None:
                    # 古い画像を捨てて最新のみ保持
                    while not image_queue.empty():
                        try:
                            image_queue.get_nowait()
                        except queue.Empty:
                            break
                    image_queue.put(img)
                    logger.info("キャプチャ完了、AI分析キューに追加")
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
            image_base64 = image_queue.get()  # ブロッキング
            logger.info("AI分析開始...")
            comments = ai_analyzer.analyze(image_base64)
            if comments:
                comment_queue.put(comments)
                logger.info("コメント生成: %d個", len(comments))
                # コメントログをファイルに記録
                try:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("comments.log", "a", encoding="utf-8") as f:
                        for c in comments:
                            f.write(f"[{now}] {c.get('text', '')}  (color: {c.get('color', '')})\n")
                except Exception:
                    logger.exception("コメントログ書き込み失敗")
            else:
                logger.info("コメントなし")
        except Exception:
            logger.exception("AIスレッドでエラー")


def main():
    # 設定読み込み
    try:
        with open("config.yaml", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("config.yaml が見つかりません。デフォルト設定で起動します。")
        config = {}

    # キュー作成
    image_queue: queue.Queue = queue.Queue(maxsize=1)
    comment_queue: queue.Queue = queue.Queue()

    # コンポーネント初期化
    screen_capture = ScreenCapture(config)
    ai_analyzer = AIAnalyzer(config)

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

    # GUI
    app = QApplication(sys.argv)
    overlay = OverlayWindow(config)
    overlay.comment_queue = comment_queue
    overlay.show()

    # システムトレイ
    tray = SystemTray(
        app=app,
        config=config,
        on_pause=lambda: pause_event.set(),
        on_resume=lambda: pause_event.clear(),
        on_quit=app.quit,
    )

    logger.info("BackseatSimulator 起動完了")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
