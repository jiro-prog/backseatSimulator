import collections
import queue
import random

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
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

    def add_comments(self, raw_comments: list[dict]):
        """AIからの生データをCommentオブジェクトに変換して追加。"""
        screen = QApplication.primaryScreen().geometry()
        screen_h = screen.height()
        screen_w = screen.width()

        for raw in raw_comments:
            if len(self.comments) >= self.max_comments:
                break
            text = raw.get("text", "")
            color = raw.get("color", "#FFFFFF")
            if not text:
                continue
            y = random.randint(int(screen_h * 0.1), int(screen_h * 0.9))
            comment = Comment(
                text=text,
                color=color,
                y_pos=y,
                x_pos=float(screen_w),
                speed=self.scroll_speed,
                font_size=self.font_size,
            )
            self.comments.append(comment)

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
            raw = self._pending.popleft()
            self.add_comments([raw])

        # 全コメント更新
        for comment in self.comments:
            comment.update()

        # 画面外コメント除去
        self.comments = [c for c in self.comments if not c.is_offscreen()]

        # 再描画
        self.update()
