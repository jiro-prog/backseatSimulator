"""ai_loop() のサイクル制御テスト。

テーブル駆動で以下をカバー:
- 再起動フラグによる終了
- キュータイムアウト
- screen_changed + audio_data の組み合わせ
- 推論後の滞留フレーム破棄
- コメント少数時の待機
"""

import queue
import sys
import os
import threading
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import ai_loop


# ---------- ヘルパー ----------

class FakeAnalyzer:
    """analyze() が指定結果を順に返すスタブ。"""

    def __init__(self, results=None):
        self._results = list(results or [])
        self._call_count = 0
        self._called = threading.Event()
        self._gate = threading.Event()
        self._gate.set()  # デフォルトはブロックなし
        self.persona = "test"

    def analyze(self, **kwargs):
        self._gate.wait(timeout=10)
        self._call_count += 1
        self._called.set()
        if self._call_count <= len(self._results):
            return self._results[self._call_count - 1]
        return []


def make_data(screen_changed=True, full_image="dummybase64"):
    """ai_loop用のキューアイテムを生成。"""
    return {
        "full_image": full_image if screen_changed else None,
        "screen_changed": screen_changed,
        "window_title": "TestWindow",
    }


def run_ai_loop(analyzer, image_queue, comment_queue, audio_state,
                restart_flag, restart_ready, capture_interval=0, timeout=5):
    """ai_loopをスレッドで実行し、timeout秒で強制終了。"""
    t = threading.Thread(
        target=ai_loop,
        args=(analyzer, image_queue, comment_queue, audio_state,
              capture_interval, restart_flag, restart_ready),
        daemon=True,
    )
    t.start()
    t.join(timeout=timeout)
    return t


# ---------- テスト ----------

class TestAiLoop:

    def test_restart_flag_exits(self):
        """restart_flag → ai_loop終了 + restart_ready.set()。"""
        analyzer = FakeAnalyzer()
        image_queue = queue.Queue()
        comment_queue = queue.Queue()
        restart_flag = threading.Event()
        restart_ready = threading.Event()
        audio_state = {"enabled": False, "capture": None}

        restart_flag.set()
        t = run_ai_loop(analyzer, image_queue, comment_queue, audio_state,
                        restart_flag, restart_ready, timeout=5)
        assert not t.is_alive()
        assert restart_ready.is_set()

    def test_queue_timeout_then_restart(self):
        """空キュー→タイムアウトでループ継続→restart_flagで終了。"""
        analyzer = FakeAnalyzer()
        image_queue = queue.Queue()
        comment_queue = queue.Queue()
        restart_flag = threading.Event()
        restart_ready = threading.Event()
        audio_state = {"enabled": False, "capture": None}

        def delayed_restart():
            time.sleep(2)
            restart_flag.set()
        threading.Thread(target=delayed_restart, daemon=True).start()

        t = run_ai_loop(analyzer, image_queue, comment_queue, audio_state,
                        restart_flag, restart_ready, timeout=5)
        assert not t.is_alive()
        assert restart_ready.is_set()

    def test_no_screen_no_audio_skips(self):
        """screen_changed=False + audio無し → analyze()呼ばれない。"""
        analyzer = FakeAnalyzer()
        image_queue = queue.Queue()
        comment_queue = queue.Queue()
        restart_flag = threading.Event()
        restart_ready = threading.Event()
        audio_state = {"enabled": False, "capture": None}

        image_queue.put(make_data(screen_changed=False))

        def delayed_restart():
            time.sleep(1)
            restart_flag.set()
        threading.Thread(target=delayed_restart, daemon=True).start()

        run_ai_loop(analyzer, image_queue, comment_queue, audio_state,
                    restart_flag, restart_ready, timeout=5)
        assert analyzer._call_count == 0

    def test_screen_changed_calls_analyze(self):
        """有効データ → analyze()呼ばれる + comment_queueに結果。"""
        comments = [{"text": "テスト", "color": "#FFFFFF"}]
        analyzer = FakeAnalyzer(results=[comments])
        image_queue = queue.Queue()
        comment_queue = queue.Queue()
        restart_flag = threading.Event()
        restart_ready = threading.Event()
        audio_state = {"enabled": False, "capture": None}

        image_queue.put(make_data(screen_changed=True))

        def delayed_restart():
            analyzer._called.wait(timeout=5)
            time.sleep(0.5)
            restart_flag.set()
        threading.Thread(target=delayed_restart, daemon=True).start()

        run_ai_loop(analyzer, image_queue, comment_queue, audio_state,
                    restart_flag, restart_ready, timeout=10)

        assert analyzer._call_count >= 1
        assert not comment_queue.empty()

    def test_frame_skipping(self):
        """推論完了後に滞留フレームが破棄される。"""
        # 1回目は常に処理、2回目の後にキューが空になるはず
        analyzer = FakeAnalyzer(results=[
            [{"text": "first", "color": "#FFF"}],
            [{"text": "second", "color": "#FFF"}],
        ])
        # 1回目のanalyze中にブロック → その間にフレームを追加
        analyzer._gate.clear()

        image_queue = queue.Queue(maxsize=10)
        comment_queue = queue.Queue()
        restart_flag = threading.Event()
        restart_ready = threading.Event()
        audio_state = {"enabled": False, "capture": None}

        # 初回データ
        image_queue.put(make_data())

        t = threading.Thread(
            target=ai_loop,
            args=(analyzer, image_queue, comment_queue, audio_state,
                  0, restart_flag, restart_ready),
            daemon=True,
        )
        t.start()

        # analyze()がブロックされるのを待つ
        time.sleep(0.5)
        # ブロック中にフレームを3つ追加
        for _ in range(3):
            image_queue.put(make_data())

        # 最初のanalyzeを解放
        analyzer._gate.set()
        # 最初の呼び出し完了を待つ
        analyzer._called.wait(timeout=5)

        # 2回目のanalyzeが終わるのを待って終了
        time.sleep(1)
        restart_flag.set()
        t.join(timeout=5)

        # first_cycle=Trueの間はスキップなし、first_cycle=Falseになると滞留フレーム破棄
        # 結果: analyze()は2回呼ばれる（初回 + 滞留から1つ）、残り2つは破棄
        assert analyzer._call_count == 2

    def test_few_comments_causes_wait(self):
        """コメント0-1個(非初回) → capture_interval分の待機が発生する。"""
        restart_flag = threading.Event()
        restart_ready = threading.Event()

        class TimingAnalyzer:
            def __init__(self):
                self._call_count = 0
                self.persona = "test"
            def analyze(self, **kwargs):
                self._call_count += 1
                if self._call_count >= 2:
                    # sleep後にrestart_flagチェックが走るよう遅延セット
                    threading.Timer(0.5, restart_flag.set).start()
                if self._call_count == 1:
                    return [{"text": "a"}, {"text": "b"}, {"text": "c"}]
                return []  # 0コメント → sleep(capture_interval) が走る

        analyzer = TimingAnalyzer()
        image_queue = queue.Queue(maxsize=10)
        comment_queue = queue.Queue()
        audio_state = {"enabled": False, "capture": None}

        image_queue.put(make_data())
        image_queue.put(make_data())

        start = time.time()
        run_ai_loop(analyzer, image_queue, comment_queue, audio_state,
                    restart_flag, restart_ready, capture_interval=1, timeout=10)
        elapsed = time.time() - start

        assert analyzer._call_count == 2
        # capture_interval=1 の sleep が入るため最低0.8秒以上かかる
        assert elapsed >= 0.8
