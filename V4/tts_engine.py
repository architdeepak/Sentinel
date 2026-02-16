"""
Text-to-Speech Engine for Driver Drowsiness Detection System V3.2
Uses Microsoft Edge-TTS for high-quality neural voice synthesis.
"""

import queue
import threading
import asyncio
from pathlib import Path

import edge_tts

from config import Config


class TTSEngine:
    """High-quality TTS using Microsoft Edge-TTS API."""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        self.is_speaking = False
        self.temp_audio = Path("/tmp/tts_temp.mp3")

    def _audio_worker(self):
        """Process TTS queue."""
        while True:
            text = self.audio_queue.get()
            if text is None:
                break

            self.is_speaking = True
            try:
                # Generate speech with Edge-TTS
                asyncio.run(self._generate_speech(text))

                # Play using mpg123 (lightweight MP3 player for RPi)
                import subprocess
                subprocess.run(
                    ["mpg123", "-q", str(self.temp_audio)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Clean up temp file
                if self.temp_audio.exists():
                    self.temp_audio.unlink()

            except Exception as e:
                print(f"⚠️ TTS error: {e}")

            self.is_speaking = False
            self.audio_queue.task_done()

    async def _generate_speech(self, text):
        """Generate speech file using Edge-TTS."""
        communicate = edge_tts.Communicate(
            text,
            Config.EDGE_TTS_VOICE,
            rate=Config.EDGE_TTS_RATE
        )
        await communicate.save(str(self.temp_audio))

    def speak(self, text):
        """Queue text for speech."""
        if text and text.strip():
            self.audio_queue.put(text.strip())

    def wait_until_done(self):
        """Wait for all speech to finish."""
        self.audio_queue.join()

    def shutdown(self):
        """Shutdown TTS."""
        self.audio_queue.put(None)
