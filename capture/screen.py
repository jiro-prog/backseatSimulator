import base64
import io
import logging

import mss
import numpy as np
from PIL import Image

try:
    import win32gui
except ImportError:
    win32gui = None

logger = logging.getLogger(__name__)


_GRID_LABELS = [
    ["左上", "上", "右上"],
    ["左",   "中央", "右"],
    ["左下", "下", "右下"],
]


class ScreenCapture:
    FOCUS_DIFF_THRESHOLD = 0.10
    UNIFORM_DIFF_RATIO = 0.85

    def __init__(self, config: dict):
        self.capture_mode = config.get("capture_mode", "full_desktop")
        self.image_max_size = config.get("image_max_size", 1280)
        self.change_threshold = config.get("change_threshold", 0.05)
        self.max_skip_count = config.get("max_skip_count", 4)
        self.enable_focus = config.get("enable_focus", True)
        self.focus_grid = config.get("focus_grid", [3, 3])
        self.focus_diff_threshold = config.get("focus_diff_threshold", 0.10)
        self.focus_crop_size = config.get("focus_crop_size", 640)
        self._prev_array: np.ndarray | None = None
        self._prev_image: Image.Image | None = None
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
        self._prev_image = img

        # base64エンコード
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def capture_with_focus(self) -> dict | None:
        """スクリーンショットを取得し、差分検知＋フォーカスクロップを行う。"""
        prev_image = self._prev_image
        prev_array = self._prev_array

        try:
            img = self._grab()
        except Exception:
            logger.exception("キャプチャ失敗")
            return None
        if img is None:
            return None

        img = self._resize(img)
        small = img.resize((256, 256), Image.LANCZOS)
        arr = np.array(small, dtype=np.float32)

        if prev_array is not None:
            diff = self._compute_diff(prev_array, arr)
            if diff < self.change_threshold:
                self._skip_count += 1
                if self._skip_count < self.max_skip_count:
                    logger.debug("差分 %.4f < 閾値 %.4f — スキップ (%d/%d)",
                                 diff, self.change_threshold, self._skip_count, self.max_skip_count)
                    return None
                logger.debug("スキップ上限到達、強制キャプチャ")

        self._skip_count = 0
        self._prev_array = arr
        self._prev_image = img

        # base64エンコード（全体画像）
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        full_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # フォーカス判定
        focus_image = None
        focus_label = None
        focus_diff = 0.0

        if self.enable_focus and prev_array is not None:
            rows, cols = self.focus_grid
            grid = self._compute_grid_diff(prev_array, arr, rows, cols)
            if grid:
                logger.info("グリッドdiff top3: %s",
                            [(c["row"], c["col"], f"{c['diff']:.4f}") for c in grid[:3]])

            if grid and grid[0]["diff"] >= self.focus_diff_threshold:
                top1 = grid[0]
                top2 = grid[1] if len(grid) > 1 else {"diff": 0.0}

                # 均等変化チェック
                if top2["diff"] > 0 and top2["diff"] / top1["diff"] >= self.UNIFORM_DIFF_RATIO:
                    logger.debug("均等変化（%.3f / %.3f）、フォーカスなし",
                                 top2["diff"], top1["diff"])
                else:
                    # 隣接マージ
                    bbox = self._merge_adjacent(grid, top1, rows, cols)
                    focus_image = self._crop_and_encode(img, bbox)
                    focus_label = self._grid_to_label(top1["row"], top1["col"])
                    focus_diff = top1["diff"]
                    logger.info("フォーカス: %s (diff=%.3f) bbox=%s",
                                focus_label, focus_diff, bbox)

        return {
            "full_image": full_b64,
            "focus_image": focus_image,
            "focus_label": focus_label,
            "focus_diff": focus_diff,
            "window_title": self._get_window_title(),
        }

    def _compute_grid_diff(
        self, img1: np.ndarray, img2: np.ndarray, rows: int = 3, cols: int = 3
    ) -> list[dict]:
        """画像をrows x colsのグリッドに分割し、各セルの差分率を計算。diff降順。"""
        h, w = img1.shape[:2]
        cell_h = h // rows
        cell_w = w // cols
        # 元画像とのスケール比
        orig_w, orig_h = self._prev_image.size if self._prev_image else (w, h)
        sx = orig_w / w
        sy = orig_h / h

        cells = []
        for r in range(rows):
            for c in range(cols):
                y1 = r * cell_h
                y2 = (r + 1) * cell_h if r < rows - 1 else h
                x1 = c * cell_w
                x2 = (c + 1) * cell_w if c < cols - 1 else w

                cell1 = img1[y1:y2, x1:x2]
                cell2 = img2[y1:y2, x1:x2]
                diff = self._compute_diff(cell1, cell2)

                cells.append({
                    "row": r,
                    "col": c,
                    "diff": diff,
                    "bbox": (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
                })

        cells.sort(key=lambda x: x["diff"], reverse=True)
        return cells

    def _merge_adjacent(self, grid: list[dict], top: dict, rows: int, cols: int) -> tuple:
        """最大diffセルに隣接する高diff領域をマージ。最大2x2。"""
        grid_map = {(c["row"], c["col"]): c for c in grid}
        tr, tc = top["row"], top["col"]

        # 隣接セルのうちdiffが閾値超えのものを収集
        merge_cells = [(tr, tc)]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                adj = grid_map.get((nr, nc))
                if adj and adj["diff"] >= self.focus_diff_threshold:
                    merge_cells.append((nr, nc))

        # 2x2制限: row/colの範囲が2以内に収める
        min_r = min(r for r, c in merge_cells)
        max_r = max(r for r, c in merge_cells)
        min_c = min(c for r, c in merge_cells)
        max_c = max(c for r, c in merge_cells)

        if max_r - min_r > 1:
            min_r = tr
            max_r = tr + 1 if tr + 1 < rows else tr
        if max_c - min_c > 1:
            min_c = tc
            max_c = tc + 1 if tc + 1 < cols else tc

        # マージ対象セルからbboxを統合
        final_cells = [
            grid_map[(r, c)] for r in range(min_r, max_r + 1)
            for c in range(min_c, max_c + 1) if (r, c) in grid_map
        ]

        x1 = min(c["bbox"][0] for c in final_cells)
        y1 = min(c["bbox"][1] for c in final_cells)
        x2 = max(c["bbox"][2] for c in final_cells)
        y2 = max(c["bbox"][3] for c in final_cells)
        return (x1, y1, x2, y2)

    def _crop_and_encode(self, img: Image.Image, bbox: tuple) -> str:
        """画像からbbox領域を切り出し、リサイズしてbase64エンコード。"""
        cropped = img.crop(bbox)
        w, h = cropped.size
        longest = max(w, h)
        if longest > self.focus_crop_size:
            scale = self.focus_crop_size / longest
            cropped = cropped.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _grid_to_label(row: int, col: int) -> str:
        """グリッド座標を日本語ラベルに変換。"""
        try:
            return _GRID_LABELS[row][col]
        except IndexError:
            return f"({row},{col})"

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
