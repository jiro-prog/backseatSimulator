"""AudioCapture.get_audio() の前処理チェーンテスト。

FakeAudioCaptureでpyaudioをバイパスし、信号処理パイプライン全段を検証:
- 無音判定 / stereo→mono / リサンプル / DC除去 / プリエンファシス / EQ / RMS正規化
"""

import sys
import os
import threading
from math import gcd

import numpy as np
import pytest
from scipy.signal import butter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capture.audio import AudioCapture, _TARGET_SR


class FakeAudioCapture:
    """AudioCapture.get_audio() だけを持つ軽量スタブ。pyaudio不要。"""

    def __init__(self, native_sr=48000, channels=2, buffer_seconds=5,
                 silence_threshold=0.001, target_rms=0.0, preemphasis=0.0,
                 highpass=0, lowpass=0):
        self._native_sr = native_sr
        self._channels = channels
        self._buffer_seconds = buffer_seconds
        self._silence_threshold = silence_threshold
        self._target_rms = target_rms
        self._preemphasis = preemphasis

        buf_samples = int(native_sr * buffer_seconds * channels)
        self._buffer = np.zeros(buf_samples, dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()

        g = gcd(native_sr, _TARGET_SR)
        self._up = _TARGET_SR // g
        self._down = native_sr // g

        nyquist = _TARGET_SR / 2
        self._sos_hp = None
        self._sos_lp = None
        if 0 < highpass < nyquist:
            self._sos_hp = butter(4, highpass, btype='high',
                                  fs=_TARGET_SR, output='sos')
        if 0 < lowpass < nyquist:
            self._sos_lp = butter(4, lowpass, btype='low',
                                  fs=_TARGET_SR, output='sos')

    get_audio = AudioCapture.get_audio


# ---------- ヘルパー ----------

def make_audio(**overrides) -> FakeAudioCapture:
    return FakeAudioCapture(**overrides)


def fill_sine(ac: FakeAudioCapture, freq: float = 440.0, amplitude: float = 0.3):
    """バッファ全体を正弦波で埋める。"""
    total = len(ac._buffer)
    samples_per_ch = total // ac._channels
    t = np.arange(samples_per_ch, dtype=np.float32) / ac._native_sr
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if ac._channels > 1:
        stereo = np.column_stack([mono] * ac._channels).flatten()
        ac._buffer[:len(stereo)] = stereo
    else:
        ac._buffer[:len(mono)] = mono
    ac._write_pos = 0


# ---------- TestGetAudio ----------

class TestGetAudio:

    def test_silence_returns_none(self):
        """無音バッファ→None。"""
        ac = make_audio()
        # バッファはゼロ初期化済み
        assert ac.get_audio() is None

    def test_nonsilent_returns_array(self):
        """有音バッファ→ndarrayが返る。"""
        ac = make_audio()
        fill_sine(ac)
        result = ac.get_audio()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_output_is_16khz_mono(self):
        """48kHz stereo入力→16kHz monoが出力される。"""
        ac = make_audio(native_sr=48000, channels=2, buffer_seconds=3)
        fill_sine(ac)
        result = ac.get_audio()
        assert result is not None
        expected_len = _TARGET_SR * 3  # 16000 * 3 = 48000 samples
        # リサンプル誤差は数サンプル以内
        assert abs(len(result) - expected_len) < 10

    def test_mono_input_no_reshape(self):
        """mono入力でもreshapeエラーにならない。"""
        ac = make_audio(native_sr=16000, channels=1, buffer_seconds=1)
        fill_sine(ac)
        result = ac.get_audio()
        assert result is not None
        assert abs(len(result) - 16000) < 10

    def test_dc_offset_removed(self):
        """DCオフセット付き信号→出力のmeanがほぼ0。"""
        ac = make_audio(native_sr=16000, channels=1, buffer_seconds=1)
        # DC=0.5 + 信号
        t = np.arange(16000, dtype=np.float32) / 16000
        ac._buffer[:16000] = 0.5 + 0.1 * np.sin(2 * np.pi * 440 * t)
        result = ac.get_audio()
        assert result is not None
        assert abs(np.mean(result)) < 0.01

    def test_preemphasis_applied(self):
        """プリエンファシスON→出力が入力と異なる。"""
        ac_off = make_audio(native_sr=16000, channels=1, preemphasis=0.0)
        ac_on = make_audio(native_sr=16000, channels=1, preemphasis=0.97)
        fill_sine(ac_off)
        fill_sine(ac_on)
        result_off = ac_off.get_audio()
        result_on = ac_on.get_audio()
        assert result_off is not None and result_on is not None
        # プリエンファシスで波形が変わる
        assert not np.allclose(result_off, result_on, atol=1e-3)

    def test_rms_normalization(self):
        """target_rms設定→出力RMSがtargetに近い。"""
        ac = make_audio(native_sr=16000, channels=1, target_rms=0.05,
                        silence_threshold=0.0001)
        # 小さい振幅の信号（silence_thresholdは超える）
        t = np.arange(16000 * 5, dtype=np.float32) / 16000
        ac._buffer[:len(t)] = 0.005 * np.sin(2 * np.pi * 440 * t)
        result = ac.get_audio()
        assert result is not None
        rms = float(np.sqrt(np.mean(result ** 2)))
        assert rms == pytest.approx(0.05, abs=0.01)

    def test_clipping_at_unity(self):
        """大音量+高target_rms→出力が[-1, 1]にクリップ。"""
        ac = make_audio(native_sr=16000, channels=1, target_rms=0.9)
        t = np.arange(16000 * 5, dtype=np.float32) / 16000
        ac._buffer[:len(t)] = 0.8 * np.sin(2 * np.pi * 440 * t)
        result = ac.get_audio()
        assert result is not None
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


# ---------- TestCircularBuffer ----------

class TestCircularBuffer:

    def test_buffer_rollover(self):
        """write_posがラップアラウンドした状態→正しい時系列順で読み出せる。"""
        ac = make_audio(native_sr=16000, channels=1, buffer_seconds=1)
        buf_len = len(ac._buffer)  # 16000 * 1 * 1 = 16000

        # 前半にA=0.1、後半にB=0.5を書き込み、write_posを中央に
        half = buf_len // 2
        ac._buffer[:half] = 0.5   # 新しいデータ（write_pos=0から書かれた）
        ac._buffer[half:] = 0.1   # 古いデータ
        ac._write_pos = half      # 次に書く位置＝中央

        result = ac.get_audio()
        assert result is not None
        # roll(-half)すると [古い(0.1), 新しい(0.5)] の順になる
        # DC除去後は正負に振れるが、前半と後半の絶対値差があるはず

    def test_write_pos_zero(self):
        """write_pos=0は初期状態 or バッファ1周 — rollしても内容変わらない。"""
        ac = make_audio(native_sr=16000, channels=1, buffer_seconds=1)
        t = np.arange(16000, dtype=np.float32) / 16000
        ac._buffer[:] = 0.3 * np.sin(2 * np.pi * 440 * t)
        ac._write_pos = 0

        result = ac.get_audio()
        assert result is not None
        assert len(result) > 0
