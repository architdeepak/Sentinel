"""
Speech-to-Text Engine for Driver Drowsiness Detection System V3.2
Uses Google Speech Recognition via the SpeechRecognition library.
"""

import speech_recognition as sr


class STTEngine:
    """speech_recognition-based STT using Google's online recognizer."""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = None
        self._initialize()

    def _initialize(self):
        """Initialize microphone and calibrate ambient noise."""
        try:
            print("🎤 Initializing microphone...")
            self.mic = sr.Microphone()
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Microphone ready")
        except Exception as e:
            print(f"⚠️ STT init failed: {e}")
            self.mic = None

    def listen(self, timeout=20, show_diagnostics=False):
        """Listen for speech and return recognized text (or None)."""
        if not self.mic:
            print("⚠️ STT not available")
            return None

        print(f"\n🎤 Listening (timeout: {timeout}s)...")
        if show_diagnostics:
            print("   [Listening with Google STT]")

        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)

            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"✓ You said: '{text}'")
            return text

        except sr.WaitTimeoutError:
            print("⏱️ No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"⚠️ Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Microphone error: {e}")
            return None

    def cleanup(self):
        """Cleanup resources."""
        pass
