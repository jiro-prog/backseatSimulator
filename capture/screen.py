import base64
import io
import logging
import threading

import mss
import numpy as np
from PIL import Image

try:
    import win32gui
except ImportError:
    win32gui = None

try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    _wgc_available = True
except ImportError:
    _wgc_available = False

logger = logging.getLogger(__name__)


class ScreenCapture:
    def __init__(self, config: dict):
        self.capture_mode = config.get("capture_mode", "full_desktop")
        self.image_max_size = config.get("image_max_size", 1280)
        self.change_threshold = config.get("change_threshold", 0.05)
        self.max_skip_count = config.get("max_skip_count", 4)
        self._prev_array: np.ndarray | None = None
        self._skip_count = 0

    def capture(self) -> str | None:
        """スクリーンショットを取得し、base64エンコードして返す。差分が閾値以下ならNone。"""
        try:
            img = self._grab()
        except Exception:
            logger.exception("キャプチャ失敗")
            return None

        if img is None:
            return None

        # リサイズ
        img = self._resize(img)

        # 差分検知（比較用に縮小）
        small = img.resize((256, 256), Image.LANCZOS)
        arr = np.array(small, dtype=np.float32)

        if self._prev_array is not None:
            diff = self._compute_diff(self._prev_array, arr)
            if diff < self.change_threshold:
                self._skip_count += 1
                if self._skip_count < self.max_skip_count:
                    logger.debug("差分 %.4f < 閾値 %.4f — スキップ (%d/%d)",
                                 diff, self.change_threshold, self._skip_count, self.max_skip_count)
                    return None
                logger.debug("スキップ上限到達、強制キャプチャ")

        self._skip_count = 0
        self._prev_array = arr

        # base64エンコード
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _get_window_title(self) -> str:
        """アクティブウィンドウのタイトルを取得。"""
        if win32gui is None:
            return ""
        try:
            hwnd = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(hwnd) or ""
        except Exception:
            return ""

    def _grab(self) -> Image.Image | None:
        if self.capture_mode == "active_window" and win32gui is not None:
            return self._grab_active_window()
        return self._grab_full_desktop()

    def _grab_full_desktop(self) -> Image.Image:
        with mss.mss() as sct:
            if len(sct.monitors) < 2:
                raise RuntimeError("モニターが検出されません (monitors=%d)" % len(sct.monitors))
            monitor = sct.monitors[1]  # プライマリモニター
            shot = sct.grab(monitor)
            return Image.frombytes("RGB", shot.size, shot.rgb)

    def _grab_active_window(self) -> Image.Image:
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd) or ""

            # WGC: ウィンドウ単体をキャプチャ（他ウィンドウの被りなし）
            if _wgc_available and title:
                img = self._grab_wgc(title)
                if img is not None:
                    return img
                logger.debug("WGCキャプチャ失敗、mssにフォールバック")

            # mss フォールバック: 矩形領域の合成後画像
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                logger.warning("アクティブウィンドウのサイズが不正、フルデスクトップにフォールバック")
                return self._grab_full_desktop()

            with mss.mss() as sct:
                region = {"left": left, "top": top, "width": width, "height": height}
                shot = sct.grab(region)
                return Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception:
            logger.warning("アクティブウィンドウ取得失敗、フルデスクトップにフォールバック", exc_info=True)
            return self._grab_full_desktop()

    @staticmethod
    def _grab_wgc(window_title: str, timeout: float = 3.0) -> Image.Image | None:
        """Windows Graphics Capture でウィンドウ単体をキャプチャする。"""
        result = {}
        event = threading.Event()

        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            window_name=window_title,
        )

        @capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
            buf = frame.frame_buffer  # (H, W, 4) BGRA uint8
            result["img"] = Image.fromarray(buf[:, :, :3][:, :, ::-1])  # BGR -> RGB
            capture_control.stop()
            event.set()

        @capture.event
        def on_closed():
            event.set()

        try:
            capture.start_free_threaded()
            event.wait(timeout=timeout)
        except Exception:
            logger.debug("WGCセッション開始失敗", exc_info=True)
            return None

        return result.get("img")

    def _resize(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        longest = max(w, h)
        if longest <= self.image_max_size:
            return img
        scale = self.image_max_size / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    def _compute_diff(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """ピクセル差分率を計算（0.0〜1.0）"""
        if img1.shape != img2.shape:
            return 1.0
        diff = np.abs(img1 - img2)
        return float(np.mean(diff) / 255.0)
