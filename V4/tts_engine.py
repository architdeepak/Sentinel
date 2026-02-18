"""
Text-to-Speech Engine for Driver Drowsiness Detection System V3.2
Uses Microsoft Edge-TTS for high-quality neural voice synthesis.
"""

import queue
import threading
import asyncio
import time
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
        self.metrics_logger = None  # Set externally for benchmarking
        self._turn_tts_start = None  # Track TTS timing across chunks in a turn

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
                t_gen_start = time.perf_counter()
                loop.run_until_complete(self._generate_speech(text, temp_file))
                gen_ms = (time.perf_counter() - t_gen_start) * 1000

                # Play using mpg123 (lightweight MP3 player for RPi)
                import subprocess
                t_play_start = time.perf_counter()
                subprocess.run(
                    ["mpg123", "-q", str(temp_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                play_ms = (time.perf_counter() - t_play_start) * 1000

                # Report to metrics logger if attached
                if self.metrics_logger:
                    self.metrics_logger.log_tts_generation(gen_ms)
                    self.metrics_logger.log_tts_playback(play_ms)

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
            # Mark turn TTS start on first chunk
            if self._turn_tts_start is None:
                self._turn_tts_start = time.perf_counter()
            self.audio_queue.put(text.strip())

    def wait_until_done(self):
        """Wait for all speech to finish."""
        self.audio_queue.join()
        # Log total TTS time for this turn
        if self._turn_tts_start is not None and self.metrics_logger:
            total_ms = (time.perf_counter() - self._turn_tts_start) * 1000
            self.metrics_logger.log_tts_total(total_ms)
        self._turn_tts_start = None

    def shutdown(self):
        """Shutdown TTS."""
        self.audio_queue.put(None)
