"""ScreenCapture のユニットテスト。

テーブル駆動で以下をカバー:
- _compute_diff: ピクセル差分率の計算
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
