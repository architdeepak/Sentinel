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
    """High-quality TTS using Edge-TTS with pipelined generation + playback.
    
    Two-stage pipeline:
      1. Generator thread: takes text from audio_queue, generates .mp3, puts path into playback_queue
      2. Playback thread: takes .mp3 paths from playback_queue, plays them back-to-back
    
    This means chunk N+1 is being generated while chunk N is playing = no gaps.
    """

    def __init__(self):
        self.audio_queue = queue.Queue()      # text in
        self._playback_queue = queue.Queue()  # (mp3_path, text) ready to play
        self._done_event = threading.Event()  # signals all chunks generated for current batch

        self._gen_thread = threading.Thread(target=self._generator_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._gen_thread.start()
        self._play_thread.start()

        self.is_speaking = False
        self._chunk_counter = 0
        self._lock = threading.Lock()
        self._pending_chunks = 0  # track how many chunks are in-flight
        self._pending_lock = threading.Lock()
        self.metrics_logger = None
        self._turn_tts_start = None

    def _get_temp_path(self):
        """Get a unique temp file path for each audio chunk."""
        with self._lock:
            self._chunk_counter += 1
            return Path(f"/tmp/tts_chunk_{self._chunk_counter}.mp3")

    def _generator_worker(self):
        """Stage 1: Generate audio files from text chunks."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            text = self.audio_queue.get()
            if text is None:
                # Signal playback thread to stop after finishing remaining items
                self._playback_queue.put(None)
                loop.close()
                break

            temp_file = self._get_temp_path()
            try:
                t_gen_start = time.perf_counter()
                loop.run_until_complete(self._generate_speech(text, temp_file))
                gen_ms = (time.perf_counter() - t_gen_start) * 1000

                self._playback_queue.put((temp_file, gen_ms))
            except Exception as e:
                print(f"⚠️ TTS generation error: {e}")
                with self._pending_lock:
                    self._pending_chunks -= 1

            self.audio_queue.task_done()

    def _playback_worker(self):
        """Stage 2: Play audio files as soon as they're generated."""
        import subprocess

        while True:
            item = self._playback_queue.get()
            if item is None:
                break

            temp_file, gen_ms = item
            self.is_speaking = True

            try:
                t_play_start = time.perf_counter()
                subprocess.run(
                    ["mpg123", "-q", str(temp_file)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                play_ms = (time.perf_counter() - t_play_start) * 1000

                if self.metrics_logger:
                    self.metrics_logger.log_tts_generation(gen_ms)
                    self.metrics_logger.log_tts_playback(play_ms)

            except Exception as e:
                print(f"⚠️ TTS playback error: {e}")

            finally:
                if temp_file.exists():
                    temp_file.unlink()
                with self._pending_lock:
                    self._pending_chunks -= 1
                    if self._pending_chunks <= 0:
                        self.is_speaking = False
                        self._done_event.set()

            self._playback_queue.task_done()

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
            if self._turn_tts_start is None:
                self._turn_tts_start = time.perf_counter()
            with self._pending_lock:
                self._pending_chunks += 1
                self._done_event.clear()
            self.audio_queue.put(text.strip())

    def wait_until_done(self):
        """Wait for all speech (generation + playback) to finish."""
        # Wait for generator to process all text
        self.audio_queue.join()
        # Wait for playback to finish all generated chunks
        self._done_event.wait(timeout=60)

        if self._turn_tts_start is not None and self.metrics_logger:
            total_ms = (time.perf_counter() - self._turn_tts_start) * 1000
            self.metrics_logger.log_tts_total(total_ms)
        self._turn_tts_start = None

    def shutdown(self):
        """Shutdown TTS."""
        self.audio_queue.put(None)
