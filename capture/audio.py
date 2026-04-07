import logging
import struct
import threading

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

# デフォルト設定
_DEFAULT_BUFFER_SECONDS = 5
_DEFAULT_SILENCE_THRESHOLD = 0.001
_TARGET_SR = 16000
_CHUNK = 1024


class AudioCapture:
    """WASAPI loopbackでデスクトップ音声をキャプチャし、ローリングバッファに蓄積する。

    get_audio()で直近N秒の16kHz monoデータを取得できる。
    """

    def __init__(self, config: dict):
        self._buffer_seconds = config.get("audio_buffer_seconds", _DEFAULT_BUFFER_SECONDS)
        self._silence_threshold = config.get("audio_silence_threshold", _DEFAULT_SILENCE_THRESHOLD)

        self._pa = pyaudio.PyAudio()

        # WASAPI loopbackデバイスを検出
        device_cfg = config.get("audio_device")
        self._device_info = self._find_loopback_device(device_cfg)
        self._native_sr = int(self._device_info["defaultSampleRate"])
        self._channels = self._device_info["maxInputChannels"]

        # ネイティブレートの循環バッファ（stereo or mono）
        buf_samples = int(self._native_sr * self._buffer_seconds * self._channels)
        self._buffer = np.zeros(buf_samples, dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()
        self._stream = None
        self._running = False
        self._thread = None

        # リサンプル比の計算（48000→16000 = 1:3）
        from math import gcd
        g = gcd(self._native_sr, _TARGET_SR)
        self._up = _TARGET_SR // g
        self._down = self._native_sr // g

        logger.info("AudioCapture初期化: device=%s, sr=%d, ch=%d, buffer=%ds, resample=%d:%d",
                     self._device_info["name"], self._native_sr, self._channels,
                     self._buffer_seconds, self._up, self._down)

    def _find_loopback_device(self, device_cfg) -> dict:
        """WASAPI loopbackデバイスを検出する。"""
        try:
            wasapi_info = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            raise RuntimeError("WASAPI host APIが見つかりません")

        # config指定あり
        if device_cfg is not None:
            return self._pa.get_device_info_by_index(device_cfg)

        # デフォルト出力デバイスを取得
        default_output = self._pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        logger.info("デフォルト出力デバイス: [%d] %s",
                     default_output["index"], default_output["name"])

        # loopbackデバイスを探す
        if default_output.get("isLoopbackDevice"):
            logger.info("ループバックデバイス: [%d] %s (sr=%d, ch=%d)",
                         default_output["index"], default_output["name"],
                         int(default_output["defaultSampleRate"]),
                         default_output["maxInputChannels"])
            return default_output

        # デフォルトがloopbackでなければ、同名のloopbackデバイスを探す
        for loopback in self._pa.get_loopback_device_info_generator():
            if default_output["name"] in loopback["name"]:
                logger.info("ループバックデバイス: [%d] %s (sr=%d, ch=%d)",
                             loopback["index"], loopback["name"],
                             int(loopback["defaultSampleRate"]),
                             loopback["maxInputChannels"])
                return loopback

        raise RuntimeError(f"WASAPIループバックデバイスが見つかりません (output: {default_output['name']})")

    def _capture_thread(self):
        """読み取りスレッド。streamからデータを読んでバッファに書き込む。"""
        while self._running:
            try:
                data = self._stream.read(_CHUNK, exception_on_overflow=False)
                # bytes → float32 numpy
                samples = np.frombuffer(data, dtype=np.float32)
                n = len(samples)

                with self._lock:
                    buf_len = len(self._buffer)
                    end = self._write_pos + n
                    if end <= buf_len:
                        self._buffer[self._write_pos:end] = samples
                    else:
                        first = buf_len - self._write_pos
                        self._buffer[self._write_pos:] = samples[:first]
                        self._buffer[:n - first] = samples[first:]
                    self._write_pos = end % buf_len
            except Exception:
                if self._running:
                    logger.exception("音声読み取りエラー")
                break

    def start(self):
        """キャプチャを開始する。"""
        if self._running:
            return

        try:
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._channels,
                rate=self._native_sr,
                input=True,
                input_device_index=self._device_info["index"],
                frames_per_buffer=_CHUNK,
            )
            self._running = True
            self._thread = threading.Thread(target=self._capture_thread, daemon=True)
            self._thread.start()
            logger.info("Audio capture開始")
        except Exception:
            logger.exception("Audio streamの開始に失敗")
            raise

    def stop(self):
        """キャプチャを停止する。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        logger.info("Audio capture停止")

    def get_audio(self) -> np.ndarray | None:
        """バッファから直近N秒の音声を取得する。

        Returns:
            16kHz mono float32 numpy配列。無音の場合はNone。
        """
        with self._lock:
            # バッファを時系列順にコピー
            raw = np.roll(self._buffer, -self._write_pos).copy()

        # stereo → mono
        if self._channels > 1:
            raw = raw.reshape(-1, self._channels).mean(axis=1)

        # 無音判定
        rms = float(np.sqrt(np.mean(raw ** 2)))
        if rms < self._silence_threshold:
            return None

        # リサンプル（ネイティブレート → 16kHz）
        if self._native_sr != _TARGET_SR:
            raw = resample_poly(raw, self._up, self._down).astype(np.float32)

        # ピークノーマライズ（音量を最大化してモデルの認識精度を上げる）
        peak = np.max(np.abs(raw))
        if peak > 1e-6:
            raw = raw / peak

        return raw
