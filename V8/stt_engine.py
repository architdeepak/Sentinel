"""
Speech-to-Text Engine for Driver Drowsiness Detection System V6.
Uses Deepgram Nova-3 via their REST pre-recorded API.

Audio capture: SpeechRecognition library (mic + VAD/silence detection).
Transcription: Deepgram pre-recorded endpoint (replaces Google STT).

V6: Returns (text, audio_data) tuple so voice features can be
extracted from the raw audio for drowsiness assessment.
"""

import time
import requests
import speech_recognition as sr

from config import Config


class STTEngine:
    """Deepgram-based STT — mic capture via SpeechRecognition, transcription via Deepgram."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.metrics_logger = None   # Set externally for benchmarking
        self._session = requests.Session()
        self._dg_url = (
            f"https://api.deepgram.com/v1/listen"
            f"?model={Config.DEEPGRAM_STT_MODEL}"
            f"&smart_format=true&punctuate=true"
        )
        self._session.headers.update({
            "Authorization": f"Token {Config.DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav",
        })
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

            if not Config.DEEPGRAM_API_KEY:
                print("⚠️ DEEPGRAM_API_KEY not set — STT will fail")
        except Exception as e:
            print(f"⚠️ STT init failed: {e}")
            self.mic = None

    def _transcribe_deepgram(self, audio):
        """Send audio to Deepgram pre-recorded API and return transcript text.

        Args:
            audio: SpeechRecognition AudioData object.

        Returns:
            str or None: Recognized text, or None on failure.
        """
        # Convert to WAV bytes (16-bit PCM) for Deepgram
        wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)

        resp = self._session.post(self._dg_url, data=wav_data, timeout=15)
        resp.raise_for_status()

        result = resp.json()
        transcript = (
            result.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
        return transcript.strip() if transcript.strip() else None

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
            print("   [Listening with Deepgram STT]")

        t_start = time.perf_counter()
        try:
            with self.mic as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)

            print("🔄 Processing speech (Deepgram)...")
            text = self._transcribe_deepgram(audio)

            if text is None:
                print("❓ Could not understand audio")
                return None, None

            latency_ms = (time.perf_counter() - t_start) * 1000
            print(f"✓ You said: '{text}'  ({latency_ms:.0f}ms)")

            # Report to metrics logger if attached
            if self.metrics_logger:
                self.metrics_logger.log_stt(latency_ms, text)

            return text, audio

        except sr.WaitTimeoutError:
            print("⏱️ No speech detected (timeout)")
            return None, None
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Deepgram API error: {e}")
            return None, None
        except Exception as e:
            print(f"⚠️ Microphone error: {e}")
            return None, None

    def cleanup(self):
        """Cleanup resources."""
        self._session.close()
