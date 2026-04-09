import logging
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import butter, resample_poly, sosfilt

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
        self._target_rms = config.get("audio_target_rms", 0.0)
        self._preemphasis = config.get("audio_preemphasis", 0.0)

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

        # EQフィルタ係数の事前計算（16kHz基準）
        nyquist = _TARGET_SR / 2
        self._highpass_freq = config.get("audio_highpass", 0)
        self._lowpass_freq = config.get("audio_lowpass", 0)
        self._sos_hp = None
        self._sos_lp = None
        if 0 < self._highpass_freq < nyquist:
            self._sos_hp = butter(4, self._highpass_freq, btype='high',
                                  fs=_TARGET_SR, output='sos')
        elif self._highpass_freq >= nyquist:
            logger.warning("audio_highpass=%dHz >= ナイキスト%dHz、無効化",
                           self._highpass_freq, nyquist)
            self._highpass_freq = 0
        if 0 < self._lowpass_freq < nyquist:
            self._sos_lp = butter(4, self._lowpass_freq, btype='low',
                                  fs=_TARGET_SR, output='sos')
        elif self._lowpass_freq >= nyquist:
            logger.info("audio_lowpass=%dHz >= ナイキスト%dHz、リサンプルLPFで十分なため無効化",
                        self._lowpass_freq, nyquist)
            self._lowpass_freq = 0

        logger.info("AudioCapture初期化: device=%s, sr=%d, ch=%d, buffer=%ds, resample=%d:%d, "
                     "preemph=%.2f, HP=%sHz, LP=%sHz",
                     self._device_info["name"], self._native_sr, self._channels,
                     self._buffer_seconds, self._up, self._down,
                     self._preemphasis,
                     self._highpass_freq or "off", self._lowpass_freq or "off")

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
        backoff = 0.5
        max_backoff = 30
        while self._running:
            try:
                data = self._stream.read(_CHUNK, exception_on_overflow=False)
                backoff = 0.5  # 成功時リセット
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
                if not self._running:
                    break
                logger.exception("音声読み取りエラー、%.1f秒後に再接続", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                try:
                    if self._stream is not None:
                        try:
                            self._stream.stop_stream()
                            self._stream.close()
                        except Exception:
                            pass
                    self._stream = self._pa.open(
                        format=pyaudio.paFloat32,
                        channels=self._channels,
                        rate=self._native_sr,
                        input=True,
                        input_device_index=self._device_info["index"],
                        frames_per_buffer=_CHUNK,
                    )
                    logger.info("音声ストリーム再接続成功")
                except Exception:
                    logger.exception("音声ストリーム再接続失敗")

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

        # DCオフセット除去
        raw = (raw - np.mean(raw)).astype(np.float32)

        # プリエンファシス（高周波のSNR改善）
        if self._preemphasis > 0:
            raw = np.append(raw[0], raw[1:] - self._preemphasis * raw[:-1]).astype(np.float32)

        # EQフィルタ（ランブル除去 + 高域ノイズ除去）
        if self._sos_hp is not None:
            raw = sosfilt(self._sos_hp, raw).astype(np.float32)
        if self._sos_lp is not None:
            raw = sosfilt(self._sos_lp, raw).astype(np.float32)

        # RMS正規化（loopbackの低音量を補正）
        if self._target_rms > 0:
            current_rms = float(np.sqrt(np.mean(raw ** 2)))
            if current_rms > 1e-6:
                gain = self._target_rms / current_rms
                raw = np.clip(raw * gain, -1.0, 1.0).astype(np.float32)

        return raw
