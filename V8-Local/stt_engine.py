"""
Speech-to-Text Engine for Driver Drowsiness Detection System V7-Local.
Uses Vosk + PyAudio — fully local, zero network calls.

Audio capture: PyAudio raw PCM stream → Vosk KaldiRecognizer.
Silence detection: 2 seconds of quiet after last recognized text = done.

V7-Local: Returns (text, raw_pcm_bytes) so voice features can be extracted
from the raw 16-bit PCM audio (same interface as V7's Deepgram-based version).
"""

import json
import time
from vosk import Model, KaldiRecognizer
import pyaudio

from config import Config


class STTEngine:
    """Vosk-based STT — fully local speech recognition via PyAudio."""

    def __init__(self):
        self.model = None
        self.recognizer = None
        self.mic = None
        self.stream = None
        self.metrics_logger = None   # Set externally for benchmarking
        self._initialize()

    def _initialize(self):
        """Initialize Vosk model and microphone."""
        try:
            if not Config.VOSK_MODEL_PATH.exists():
                print(f"❌ Vosk model not found: {Config.VOSK_MODEL_PATH}")
                return

            print("📦 Loading Vosk model...")
            self.model = Model(str(Config.VOSK_MODEL_PATH))
            self.recognizer = KaldiRecognizer(self.model, Config.VOSK_SAMPLE_RATE)
            print("✓ Vosk model loaded")

            print("🎤 Setting up microphone...")
            self.mic = pyaudio.PyAudio()
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=Config.VOSK_SAMPLE_RATE,
                input=True,
                frames_per_buffer=Config.VOSK_BUFFER_SIZE,
            )
            self.stream.start_stream()
            print("✓ Microphone ready")

        except Exception as e:
            print(f"⚠️ STT init failed: {e}")
            self.model = None

    def _clear_buffer(self):
        """Drain ALL buffered audio so stale mic input is discarded before listening."""
        if not self.stream:
            return
        try:
            while self.stream.get_read_available() >= Config.VOSK_BUFFER_SIZE:
                self.stream.read(Config.VOSK_BUFFER_SIZE, exception_on_overflow=False)
        except Exception:
            pass

    def listen(self, timeout=20, show_diagnostics=False):
        """Listen for speech and return (text, raw_pcm_bytes) or (None, None).

        Args:
            timeout: Maximum seconds to wait for speech.
            show_diagnostics: Unused (kept for API compatibility with V7).

        Returns:
            tuple: (recognized_text, raw_pcm_bytes) on success.
                   raw_pcm_bytes is signed 16-bit PCM at VOSK_SAMPLE_RATE Hz,
                   ready for VoiceFeatureExtractor.extract_features().
                   (None, None) on timeout or no speech recognized.
        """
        if not self.model:
            print("⚠️ STT not available")
            return None, None

        print(f"\n🎤 Listening (timeout: {timeout}s)...")

        self._clear_buffer()
        self.recognizer.Reset()

        t_start = time.perf_counter()
        collected_text = []
        collected_audio = bytearray()
        last_text_time = None

        iterations_per_second = Config.VOSK_SAMPLE_RATE / Config.VOSK_BUFFER_SIZE
        max_iterations = int(timeout * iterations_per_second)

        try:
            for _ in range(max_iterations):
                # stream.read() blocks ~512ms per call (8192 samples @ 16kHz)
                # max_iterations already caps total time to `timeout` seconds
                data = self.stream.read(
                    Config.VOSK_BUFFER_SIZE,
                    exception_on_overflow=False,
                )
                collected_audio.extend(data)

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        collected_text.append(text)
                        last_text_time = time.perf_counter()
                        print(f"   📝 Heard: '{text}'")

                # 3 seconds of silence after last recognized phrase = done
                # Also require at least 4s total listen time so user has time to start
                if (last_text_time
                        and (time.perf_counter() - last_text_time) > 3.0
                        and (time.perf_counter() - t_start) > 4.0):
                    break

            # Collect any partial result the recognizer is still holding
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                collected_text.append(final_text)

            full_text = " ".join(collected_text).strip()

            if not full_text:
                print("⚠️  No speech recognized")
                return None, None

            latency_ms = (time.perf_counter() - t_start) * 1000
            print(f"✓ You said: '{full_text}'  ({latency_ms:.0f}ms)")

            if self.metrics_logger:
                self.metrics_logger.log_stt(latency_ms, full_text)

            return full_text, bytes(collected_audio)

        except Exception as e:
            print(f"⚠️ STT error: {e}")
            return None, None

    def cleanup(self):
        """Cleanup audio resources."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.mic:
                self.mic.terminate()
        except Exception:
            pass
