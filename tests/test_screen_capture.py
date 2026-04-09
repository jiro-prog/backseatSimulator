"""ScreenCapture のユニットテスト。

テーブル駆動で以下をカバー:
- _compute_diff: ピクセル差分率の計算
- _grid_to_label: グリッド座標→日本語ラベル
- _compute_grid_diff: グリッド分割差分
- _merge_adjacent: 隣接セルマージ
- capture: 差分検知・スキップ制御
"""

import base64
import sys
import os
import threading

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capture.screen import ScreenCapture


# ---------- ヘルパー ----------

def make_capture(**overrides) -> ScreenCapture:
    """最小限の設定でScreenCaptureを生成。"""
    config = {
        "capture_mode": "full_desktop",
        "image_max_size": 256,
        "change_threshold": 0.05,
        "max_skip_count": 4,
        "enable_focus": False,
        "focus_grid": [3, 3],
        "focus_diff_threshold": 0.10,
        "focus_crop_size": 640,
    }
    config.update(overrides)
    return ScreenCapture(config)


def solid_array(value: float, shape=(256, 256, 3)) -> np.ndarray:
    """指定値で埋めたfloat32配列を返す。"""
    return np.full(shape, value, dtype=np.float32)


def solid_image(color: tuple, size=(100, 100)) -> Image.Image:
    """指定色のPIL Imageを返す。"""
    return Image.new("RGB", size, color)


# ---------- _compute_diff ----------

class TestComputeDiff:

    def test_identical_images_zero_diff(self):
        """同一画像→差分0.0。"""
        sc = make_capture()
        arr = solid_array(128.0)
        assert sc._compute_diff(arr, arr.copy()) == 0.0

    def test_completely_different_images(self):
        """全黒vs全白→差分1.0。"""
        sc = make_capture()
        black = solid_array(0.0)
        white = solid_array(255.0)
        assert sc._compute_diff(black, white) == pytest.approx(1.0, abs=0.01)

    def test_half_changed(self):
        """上半分同一、下半分反転→差分約0.5。"""
        sc = make_capture()
        arr1 = solid_array(0.0)
        arr2 = arr1.copy()
        arr2[128:, :, :] = 255.0
        diff = sc._compute_diff(arr1, arr2)
        assert diff == pytest.approx(0.5, abs=0.05)

    def test_shape_mismatch_returns_one(self):
        """shape不一致→1.0。"""
        sc = make_capture()
        a = solid_array(0.0, shape=(256, 256, 3))
        b = solid_array(0.0, shape=(128, 128, 3))
        assert sc._compute_diff(a, b) == 1.0

    def test_small_perturbation(self):
        """1ピクセルだけ変更→0より大きいが小さい。"""
        sc = make_capture()
        arr1 = solid_array(100.0)
        arr2 = arr1.copy()
        arr2[0, 0, 0] = 200.0
        diff = sc._compute_diff(arr1, arr2)
        assert 0.0 < diff < 0.01


# ---------- _grid_to_label ----------

class TestGridToLabel:

    def test_center(self):
        assert ScreenCapture._grid_to_label(1, 1) == "中央"

    def test_corners(self):
        assert ScreenCapture._grid_to_label(0, 0) == "左上"
        assert ScreenCapture._grid_to_label(0, 2) == "右上"
        assert ScreenCapture._grid_to_label(2, 0) == "左下"
        assert ScreenCapture._grid_to_label(2, 2) == "右下"

    def test_out_of_range(self):
        """範囲外→フォールバック文字列。"""
        assert ScreenCapture._grid_to_label(5, 5) == "(5,5)"


# ---------- _compute_grid_diff ----------

class TestComputeGridDiff:

    def test_uniform_images_all_zero(self):
        """同一画像→全セルdiff==0。"""
        sc = make_capture()
        sc._prev_image = solid_image((0, 0, 0), size=(256, 256))
        arr = solid_array(128.0)
        grid = sc._compute_grid_diff(arr, arr.copy(), 3, 3)
        assert len(grid) == 9
        assert all(c["diff"] == 0.0 for c in grid)

    def test_single_cell_change(self):
        """左上だけ変更→grid[0]が(0,0)で最大diff。"""
        sc = make_capture()
        sc._prev_image = solid_image((0, 0, 0), size=(256, 256))
        arr1 = solid_array(0.0)
        arr2 = arr1.copy()
        # 左上セル (row=0, col=0) のみ変更
        cell_h = 256 // 3
        cell_w = 256 // 3
        arr2[:cell_h, :cell_w, :] = 255.0
        grid = sc._compute_grid_diff(arr1, arr2, 3, 3)
        assert grid[0]["row"] == 0
        assert grid[0]["col"] == 0
        assert grid[0]["diff"] > 0.5

    def test_bbox_scale_ratio(self):
        """_prev_imageが512x512、入力が256x256→bboxが2倍スケール。"""
        sc = make_capture()
        sc._prev_image = solid_image((0, 0, 0), size=(512, 512))
        arr = solid_array(0.0)
        grid = sc._compute_grid_diff(arr, arr.copy(), 3, 3)
        # cell (0,0) の bbox は (0, 0, cell_w*2, cell_h*2)
        cell = next(c for c in grid if c["row"] == 0 and c["col"] == 0)
        assert cell["bbox"][0] == 0
        assert cell["bbox"][1] == 0
        # 512 / 256 = 2x scale, cell_w = 256//3 = 85 → 85*2 = 170
        assert cell["bbox"][2] == int((256 // 3) * 2)

    def test_sorted_descending(self):
        """結果がdiff降順でソートされている。"""
        sc = make_capture()
        sc._prev_image = solid_image((0, 0, 0), size=(256, 256))
        arr1 = solid_array(0.0)
        arr2 = arr1.copy()
        arr2[:86, :86, :] = 100.0  # 左上に変化
        grid = sc._compute_grid_diff(arr1, arr2, 3, 3)
        diffs = [c["diff"] for c in grid]
        assert diffs == sorted(diffs, reverse=True)


# ---------- _merge_adjacent ----------

class TestMergeAdjacent:

    def _make_grid(self, diffs_2d, cell_size=100):
        """2Dリストからgridデータを生成。"""
        grid = []
        rows = len(diffs_2d)
        cols = len(diffs_2d[0])
        for r in range(rows):
            for c in range(cols):
                grid.append({
                    "row": r, "col": c,
                    "diff": diffs_2d[r][c],
                    "bbox": (c * cell_size, r * cell_size,
                             (c + 1) * cell_size, (r + 1) * cell_size),
                })
        grid.sort(key=lambda x: x["diff"], reverse=True)
        return grid

    def test_single_cell_no_merge(self):
        """1セルだけ閾値超え→そのセルのbboxのみ。"""
        sc = make_capture(focus_diff_threshold=0.10)
        diffs = [
            [0.50, 0.01, 0.01],
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
        ]
        grid = self._make_grid(diffs)
        top = grid[0]
        bbox = sc._merge_adjacent(grid, top, 3, 3)
        assert bbox == (0, 0, 100, 100)

    def test_adjacent_merge(self):
        """隣接セルが閾値超え→マージされたbbox。"""
        sc = make_capture(focus_diff_threshold=0.10)
        diffs = [
            [0.50, 0.30, 0.01],
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
        ]
        grid = self._make_grid(diffs)
        top = grid[0]  # (0,0) diff=0.50
        bbox = sc._merge_adjacent(grid, top, 3, 3)
        # (0,0)と(0,1)がマージ → x: 0-200, y: 0-100
        assert bbox == (0, 0, 200, 100)

    def test_2x2_cap(self):
        """L字型3セルが閾値超え→2x2に制限。"""
        sc = make_capture(focus_diff_threshold=0.10)
        diffs = [
            [0.50, 0.30, 0.01],
            [0.20, 0.01, 0.01],
            [0.15, 0.01, 0.01],
        ]
        grid = self._make_grid(diffs)
        top = grid[0]  # (0,0)
        bbox = sc._merge_adjacent(grid, top, 3, 3)
        # (0,0), (0,1), (1,0) → row span=2, col span=2 → OK
        # (2,0) は隣接だが row span=3 になるので 2x2制限で切られる
        x1, y1, x2, y2 = bbox
        assert (y2 - y1) <= 200  # 最大2セル分
        assert (x2 - x1) <= 200

    def test_non_adjacent_not_merged(self):
        """離れた2セルが閾値超え→topセルのみ。"""
        sc = make_capture(focus_diff_threshold=0.10)
        diffs = [
            [0.50, 0.01, 0.01],
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.30],
        ]
        grid = self._make_grid(diffs)
        top = grid[0]  # (0,0) diff=0.50
        bbox = sc._merge_adjacent(grid, top, 3, 3)
        assert bbox == (0, 0, 100, 100)


# ---------- capture (差分検知・スキップ制御) ----------

class TestCapture:

    def test_first_capture_always_succeeds(self):
        """初回キャプチャは常にbase64文字列を返す。"""
        sc = make_capture()
        sc._grab = lambda: solid_image((128, 128, 128))
        result = sc.capture()
        assert result is not None
        # base64デコードできることを確認
        base64.b64decode(result)

    def test_identical_frames_skip(self):
        """同一フレーム2回→2回目はNone。"""
        img = solid_image((100, 100, 100))
        sc = make_capture()
        sc._grab = lambda: img.copy()
        first = sc.capture()
        assert first is not None
        second = sc.capture()
        assert second is None

    def test_skip_count_forces_capture(self):
        """max_skip_count回スキップ後は強制キャプチャ。"""
        img = solid_image((100, 100, 100))
        sc = make_capture(max_skip_count=2)
        sc._grab = lambda: img.copy()
        # 初回: 成功, skip_count=0
        assert sc.capture() is not None
        # 2回目: diff<threshold, skip_count=1 < 2 → skip
        assert sc.capture() is None
        # 3回目: diff<threshold, skip_count=2 (not < 2) → 強制キャプチャ
        assert sc.capture() is not None

    def test_different_frames_no_skip(self):
        """異なるフレーム→スキップしない。"""
        colors = iter([(0, 0, 0), (255, 255, 255)])
        sc = make_capture()
        sc._grab = lambda: solid_image(next(colors))
        first = sc.capture()
        assert first is not None
        second = sc.capture()
        assert second is not None

    def test_grab_returns_none(self):
        """_grabがNone→capture()もNone。"""
        sc = make_capture()
        sc._grab = lambda: None
        assert sc.capture() is None

    def test_grab_raises_exception(self):
        """_grabが例外→capture()はNone（クラッシュしない）。"""
        sc = make_capture()
        sc._grab = lambda: (_ for _ in ()).throw(RuntimeError("test"))
        assert sc.capture() is None
