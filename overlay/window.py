import collections
import logging
import queue
import random
import time

logger = logging.getLogger(__name__)

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget

from overlay.comment import Comment


class OverlayWindow(QWidget):
    # pending残量 → ドリップ間隔（秒）。残量が少ないほど温存
    DRIP_INTERVAL_TABLE = [
        (5, 2.0),   # 5個以上: 通常速度
        (3, 3.0),   # 3〜4個: 減速
        (1, 4.0),   # 1〜2個: 温存
    ]
    MAX_PENDING = 8  # pendingバッファ上限

    def __init__(self, config: dict):
        super().__init__()
        self.font_size = config.get("font_size", 36)
        self.scroll_speed = config.get("scroll_speed", 3.0)
        self.max_comments = config.get("max_comments", 20)

        self.comments: list[Comment] = []
        self.comment_queue: queue.Queue = queue.Queue()
        # ドリップ用: 個別コメントを溜めておくバッファ
        self._pending: collections.deque = collections.deque()
        self._last_drip_time: float = 0.0
        self._cycle_seconds = 20.0  # バッチ到着間隔（デバッグログ用）
        self._last_batch_time: float = 0.0

        # ウィンドウ属性
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # タスクバー非表示
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        # デスクトップ全体サイズ
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        # スロット管理（重なり回避）
        line_height = int(self.font_size * 1.6)
        usable_top = int(screen.height() * 0.1)
        usable_bottom = int(screen.height() * 0.9)
        self._slot_y = list(range(usable_top + line_height, usable_bottom, line_height))
        self._font_metrics = QFontMetrics(
            QFont("Meiryo UI", self.font_size, QFont.Bold)
        )

        # アニメーションタイマー（約60fps）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(16)

        # 最前面維持タイマー（5秒ごと）
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(self._raise_topmost)
        self._topmost_timer.start(5000)

        # 起動時の挨拶コメント
        greeting = [
            {"text": "わこつ", "color": "#FFFFFF"},
            {"text": "わこつです", "color": "#FFFFFF"},
            {"text": "わこつー", "color": "#87CEEB"},
            {"text": "きた", "color": "#FFB347"},
            {"text": "はじまった", "color": "#44FF44"},
        ]
        for c in greeting:
            self._pending.append(c)

    def _find_free_slot(self, text_width: int) -> int | None:
        """空きスロットを返す。なければNone。"""
        screen_w = QApplication.primaryScreen().geometry().width()
        gap = self.font_size * 2  # コメント間の最小余白

        # 各スロットの最右端を計算
        rightmost: dict[int, float] = {}
        for c in self.comments:
            if c.slot < 0:
                continue
            right_edge = c.x_pos + c.text_width
            if c.slot not in rightmost or right_edge > rightmost[c.slot]:
                rightmost[c.slot] = right_edge

        candidates = []
        for i in range(len(self._slot_y)):
            if i not in rightmost:
                candidates.append(i)
            elif rightmost[i] + gap < screen_w:
                candidates.append(i)

        if not candidates:
            return None
        return random.choice(candidates)

    def _try_add_comment(self, raw: dict) -> bool:
        """1個のコメントをスロットに配置。成功したらTrue。"""
        if len(self.comments) >= self.max_comments:
            return False
        text = raw.get("text", "")
        color = raw.get("color", "#FFFFFF")
        if not text:
            return True  # 空テキストは消化済み扱い

        text_width = self._font_metrics.horizontalAdvance(text)
        slot = self._find_free_slot(text_width)
        if slot is None:
            return False

        screen_w = QApplication.primaryScreen().geometry().width()
        comment = Comment(
            text=text,
            color=color,
            y_pos=self._slot_y[slot],
            x_pos=float(screen_w),
            speed=self.scroll_speed,
            font_size=self.font_size,
            slot=slot,
            text_width=text_width,
        )
        self.comments.append(comment)
        return True

    def add_comments(self, raw_comments: list[dict]):
        """AIからの生データをCommentオブジェクトに変換して追加。"""
        for raw in raw_comments:
            if not self._try_add_comment(raw):
                break

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for comment in self.comments:
            font = QFont("Meiryo UI", comment.font_size, QFont.Bold)
            painter.setFont(font)

            x = int(comment.x_pos)
            y = comment.y_pos

            # 黒アウトライン（4方向 + 斜め4方向）
            outline_pen = QPen(QColor(0, 0, 0))
            painter.setPen(outline_pen)
            offset = 2
            for dx, dy in [(-offset, 0), (offset, 0), (0, -offset), (0, offset),
                           (-offset, -offset), (offset, -offset),
                           (-offset, offset), (offset, offset)]:
                painter.drawText(x + dx, y + dy, comment.text)

            # 本体テキスト
            painter.setPen(QPen(QColor(comment.color)))
            painter.drawText(x, y, comment.text)

        painter.end()

    def _raise_topmost(self):
        """Win32 APIで直接最前面に設定する。"""
        try:
            import ctypes
            hwnd = int(self.winId())
            HWND_TOPMOST = -1
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_NOACTIVATE = 0x0010
            ctypes.windll.user32.SetWindowPos(
                hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE
            )
        except Exception:
            pass

    def _get_drip_interval(self) -> float | None:
        """pending残量に基づいてドリップ間隔を返す。Noneなら停止。"""
        count = len(self._pending)
        for threshold, interval in self.DRIP_INTERVAL_TABLE:
            if count >= threshold:
                return interval
        return None

    def _update_frame(self):
        # キューからコメントを取得してpendingバッファに追加
        while True:
            try:
                raw_comments = self.comment_queue.get_nowait()
                now = time.monotonic()
                if self._last_batch_time > 0:
                    self._cycle_seconds = now - self._last_batch_time
                self._last_batch_time = now
                # 溢れ分は古いpendingから捨て、それでも超過なら新バッチを切り詰め
                overflow = len(self._pending) + len(raw_comments) - self.MAX_PENDING
                if overflow > 0:
                    drop_old = min(overflow, len(self._pending))
                    for _ in range(drop_old):
                        self._pending.popleft()
                    drop_new = overflow - drop_old
                    if drop_new > 0:
                        raw_comments = raw_comments[:len(raw_comments) - drop_new]
                for raw in raw_comments:
                    self._pending.append(raw)
                logger.info("pending=%d (+%d) cycle=%.1fs",
                            len(self._pending), len(raw_comments), self._cycle_seconds)
            except queue.Empty:
                break

        # ドリップ: pending残量ベースで間隔を可変
        now = time.monotonic()
        drip_interval = self._get_drip_interval()
        if (drip_interval is not None
                and self._pending
                and (now - self._last_drip_time) >= drip_interval):
            if self._try_add_comment(self._pending[0]):
                self._pending.popleft()
                self._last_drip_time = now

        # 全コメント更新
        for comment in self.comments:
            comment.update()

        # 画面外コメント除去
        self.comments = [c for c in self.comments if not c.is_offscreen()]

        # 再描画
        self.update()
