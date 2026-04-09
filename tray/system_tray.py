from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (
    QSystemTrayIcon, QMenu, QAction, QActionGroup, QApplication,
)

PERSONA_LABELS = {
    "heckle": "ヤジ",
    "backseat": "指示厨",
    "hype": "ワイワイ",
}

CAPTURE_MODE_LABELS = {
    "active_window": "アクティブウィンドウ",
    "full_desktop": "全画面",
}


class SystemTray:
    def __init__(self, app: QApplication, config: dict,
                 on_pause: callable, on_resume: callable, on_quit: callable,
                 on_persona_change: callable = None,
                 on_capture_mode_change: callable = None,
                 on_audio_toggle: callable = None,
                 on_restart: callable = None):
        self.app = app
        self.is_paused = False
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.on_persona_change = on_persona_change
        self.on_capture_mode_change = on_capture_mode_change
        self.on_audio_toggle = on_audio_toggle
        self.on_restart = on_restart

        # 簡易アイコン生成（緑の四角）
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setBrush(QColor("#44FF44"))
        painter.setPen(QColor("#228822"))
        painter.drawRoundedRect(2, 2, 28, 28, 6, 6)
        painter.end()
        icon = QIcon(pixmap)

        self.tray = QSystemTrayIcon(icon, app)
        self.tray.setToolTip("BackseatSimulator")

        # メニュー作成
        menu = QMenu()

        self.pause_action = QAction("一時停止", menu)
        self.pause_action.triggered.connect(self._toggle_pause)
        menu.addAction(self.pause_action)

        # ペルソナ切替サブメニュー
        current_persona = config.get("persona", "heckle")
        if isinstance(current_persona, list):
            current_persona = current_persona[0]
        persona_menu = QMenu("ペルソナ", menu)
        persona_group = QActionGroup(persona_menu)
        persona_group.setExclusive(True)
        for key, label in PERSONA_LABELS.items():
            action = QAction(label, persona_menu, checkable=True)
            action.setChecked(key == current_persona)
            action.setData(key)
            action.triggered.connect(lambda checked, k=key: self._change_persona(k))
            persona_group.addAction(action)
            persona_menu.addAction(action)
        menu.addMenu(persona_menu)

        # キャプチャモード切替サブメニュー
        current_capture_mode = config.get("capture_mode", "active_window")
        capture_menu = QMenu("キャプチャ", menu)
        capture_group = QActionGroup(capture_menu)
        capture_group.setExclusive(True)
        for key, label in CAPTURE_MODE_LABELS.items():
            action = QAction(label, capture_menu, checkable=True)
            action.setChecked(key == current_capture_mode)
            action.triggered.connect(lambda checked, k=key: self._change_capture_mode(k))
            capture_group.addAction(action)
            capture_menu.addAction(action)
        menu.addMenu(capture_menu)

        # 音声キャプチャ トグル
        self.audio_action = QAction("音声キャプチャ", menu, checkable=True)
        self.audio_action.setChecked(config.get("enable_audio", False))
        self.audio_action.triggered.connect(self._toggle_audio)
        menu.addAction(self.audio_action)

        menu.addSeparator()

        restart_action = QAction("再起動", menu)
        restart_action.triggered.connect(self._restart)
        menu.addAction(restart_action)

        quit_action = QAction("終了", menu)
        quit_action.triggered.connect(on_quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.show()

    def _toggle_pause(self):
        if self.is_paused:
            self.is_paused = False
            self.pause_action.setText("一時停止")
            self.on_resume()
        else:
            self.is_paused = True
            self.pause_action.setText("再開")
            self.on_pause()

    def _restart(self):
        if self.on_restart:
            self.on_restart()

    def _change_capture_mode(self, mode: str):
        if self.on_capture_mode_change:
            self.on_capture_mode_change(mode)

    def _toggle_audio(self, checked: bool):
        if self.on_audio_toggle:
            self.on_audio_toggle(checked)

    def _change_persona(self, key: str):
        if self.on_persona_change:
            self.on_persona_change(key)
