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

import numpy as np
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
            comment_queue: queue.Queue, audio_capture=None,
            capture_interval: int = 8):
    """AIスレッド: 画像を受け取りコメント生成。"""
    first_cycle = True
    while True:
        try:
            data = image_queue.get()  # ブロッキング（新鮮なフレームを待つ）
            logger.info("AI分析開始...")

            # 音声スナップショット取得
            audio_data = None
            if audio_capture is not None:
                audio_data = audio_capture.get_audio()
                if audio_data is not None:
                    rms = float(np.sqrt(np.mean(audio_data ** 2)))
                    logger.info("音声取得: %.1f秒, RMS=%.4f",
                                len(audio_data) / 16000, rms)

            comments = ai_analyzer.analyze(
                full_image=data["full_image"],
                focus_image=data.get("focus_image"),
                window_title=data.get("window_title", ""),
                audio_data=audio_data,
            )
            if comments:
                comment_queue.put(comments)
                logger.info("コメント生成: %d個 (comment_queue=%d)", len(comments), comment_queue.qsize())
                # コメントログをファイルに記録
                try:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    focus_info = data.get("focus_label") or "-"
                    focus_diff = data.get("focus_diff", 0.0)
                    win_title = data.get("window_title", "")
                    audio_info = f"{len(audio_data)/16000:.1f}s" if audio_data is not None else "-"
                    with open("comments.log", "a", encoding="utf-8") as f:
                        f.write(f"[{now}] focus={focus_info} diff={focus_diff:.3f} window={win_title} audio={audio_info}\n")
                        for c in comments:
                            f.write(f"[{now}]   {c.get('text', '')}  (color: {c.get('color', '')})\n")
                except Exception:
                    logger.exception("コメントログ書き込み失敗")
            else:
                logger.info("コメントなし")

            # 推論完了後: 滞留フレームを捨てて次の新鮮なキャプチャを待つ
            # （初回は滞留がないのでスキップ）
            if not first_cycle:
                skipped = 0
                while not image_queue.empty():
                    try:
                        image_queue.get_nowait()
                        skipped += 1
                    except queue.Empty:
                        break
                if skipped:
                    logger.info("滞留フレーム %d 枚スキップ", skipped)

            # コメント数が少なかった場合、画面が変わるまで待つ
            comment_count = len(comments) if comments else 0
            if not first_cycle and comment_count <= 1:
                logger.info("コメント少数(%d個)、%d秒待機", comment_count, capture_interval)
                time.sleep(capture_interval)
            first_cycle = False
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

    # 音声キャプチャ初期化
    audio_capture = None
    if config.get("enable_audio", False):
        from capture.audio import AudioCapture
        try:
            audio_capture = AudioCapture(config)
            audio_capture.start()
            print("音声キャプチャ開始")
        except Exception as e:
            logger.warning("音声キャプチャの初期化に失敗（音声なしで継続）: %s", e)
            audio_capture = None

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
        args=(ai_analyzer, image_queue, comment_queue, audio_capture,
              config.get("capture_interval", 8)),
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
