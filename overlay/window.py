import collections
import queue
import random

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QWidget

from overlay.comment import Comment


class OverlayWindow(QWidget):
    def __init__(self, config: dict):
        super().__init__()
        self.font_size = config.get("font_size", 36)
        self.scroll_speed = config.get("scroll_speed", 3.0)
        self.max_comments = config.get("max_comments", 20)
        # コメントを流す間隔（秒）
        self.drip_interval = config.get("drip_interval", 2.0)

        self.comments: list[Comment] = []
        self.comment_queue: queue.Queue = queue.Queue()
        # ドリップ用: 個別コメントを溜めておくバッファ
        self._pending: collections.deque = collections.deque()
        self._drip_frames = max(1, int(self.drip_interval / 0.016))  # 秒→フレーム数
        self._frame_count = 0

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

    def _update_frame(self):
        # キューからコメントを取得してpendingバッファに追加
        while True:
            try:
                raw_comments = self.comment_queue.get_nowait()
                for raw in raw_comments:
                    self._pending.append(raw)
            except queue.Empty:
                break

        # ドリップ: 一定フレーム間隔で1個ずつ流す
        self._frame_count += 1
        if self._pending and self._frame_count >= self._drip_frames:
            self._frame_count = 0
            if self._try_add_comment(self._pending[0]):
                self._pending.popleft()

        # 全コメント更新
        for comment in self.comments:
            comment.update()

        # 画面外コメント除去
        self.comments = [c for c in self.comments if not c.is_offscreen()]

        # 再描画
        self.update()
