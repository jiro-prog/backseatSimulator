from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction, QApplication


class SystemTray:
    def __init__(self, app: QApplication, config: dict,
                 on_pause: callable, on_resume: callable, on_quit: callable):
        self.app = app
        self.is_paused = False
        self.on_pause = on_pause
        self.on_resume = on_resume

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

        settings_action = QAction("設定...", menu)
        settings_action.setEnabled(False)
        menu.addAction(settings_action)

        menu.addSeparator()

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
