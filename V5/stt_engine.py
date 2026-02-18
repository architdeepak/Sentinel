"""
Speech-to-Text Engine for Driver Drowsiness Detection System V5.
Uses Google Speech Recognition via the SpeechRecognition library.

V5: Returns (text, audio_data) tuple so voice features can be
extracted from the raw audio for drowsiness assessment.
"""

import time
import speech_recognition as sr


class STTEngine:
    """speech_recognition-based STT using Google's online recognizer."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.metrics_logger = None   # Set externally for benchmarking
        self._initialize()

    def _initialize(self):
        """Initialize microphone and calibrate ambient noise."""
        try:
            print("🎤 Initializing microphone...")
            self.mic = sr.Microphone()

            # Allow longer pauses before considering speech "done"
            # Default 0.8s is way too short — drivers pause to think
            self.recognizer.pause_threshold = 2.5    # seconds of silence before phrase ends
            self.recognizer.phrase_threshold = 0.3    # min seconds of audio to consider speech
            self.recognizer.non_speaking_duration = 1.5  # seconds of silence to keep on buffer edges

            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Microphone ready")
        except Exception as e:
            print(f"⚠️ STT init failed: {e}")
            self.mic = None

    def listen(self, timeout=20, show_diagnostics=False):
        """Listen for speech and return (text, audio_data) or (None, None).

        Returns:
            tuple: (recognized_text, AudioData) on success, (None, None) on failure.
                   AudioData is the raw audio for voice feature extraction.
        """
        if not self.mic:
            print("⚠️ STT not available")
            return None, None

        print(f"\n🎤 Listening (timeout: {timeout}s)...")
        if show_diagnostics:
            print("   [Listening with Google STT]")

        t_start = time.perf_counter()
        try:
            with self.mic as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)

            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            latency_ms = (time.perf_counter() - t_start) * 1000
            print(f"✓ You said: '{text}'  ({latency_ms:.0f}ms)")

            # Report to metrics logger if attached
            if self.metrics_logger:
                self.metrics_logger.log_stt(latency_ms, text)

            return text, audio

        except sr.WaitTimeoutError:
            print("⏱️ No speech detected (timeout)")
            return None, None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None, None
        except sr.RequestError as e:
            print(f"⚠️ Speech recognition service error: {e}")
            return None, None
        except Exception as e:
            print(f"⚠️ Microphone error: {e}")
            return None, None

    def cleanup(self):
        """Cleanup resources."""
        pass
