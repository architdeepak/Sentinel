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
    """High-quality TTS using Microsoft Edge-TTS API with sentence-level streaming."""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        self.is_speaking = False
        self._chunk_counter = 0  # Unique temp file per chunk to avoid conflicts
        self._lock = threading.Lock()

    def _get_temp_path(self):
        """Get a unique temp file path for each audio chunk."""
        with self._lock:
            self._chunk_counter += 1
            return Path(f"/tmp/tts_chunk_{self._chunk_counter}.mp3")

    def _audio_worker(self):
        """Process TTS queue — plays each sentence chunk as it arrives."""
        # Create a persistent event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            text = self.audio_queue.get()
            if text is None:
                loop.close()
                break

            self.is_speaking = True
            temp_file = self._get_temp_path()
            try:
                # Generate speech with Edge-TTS
                loop.run_until_complete(self._generate_speech(text, temp_file))

                # Play using mpg123 (lightweight MP3 player for RPi)
                import subprocess
                subprocess.run(
                    ["mpg123", "-q", str(temp_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

            except Exception as e:
                print(f"⚠️ TTS error: {e}")

            finally:
                # Clean up this chunk's temp file
                if temp_file.exists():
                    temp_file.unlink()

            self.is_speaking = self.audio_queue.qsize() > 0
            self.audio_queue.task_done()

        loop.close()

    async def _generate_speech(self, text, output_path):
        """Generate speech file using Edge-TTS."""
        communicate = edge_tts.Communicate(
            text,
            Config.EDGE_TTS_VOICE,
            rate=Config.EDGE_TTS_RATE
        )
        await communicate.save(str(output_path))

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
