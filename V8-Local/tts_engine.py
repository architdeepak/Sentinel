"""
Text-to-Speech Engine for Driver Drowsiness Detection System V7-Local.
Uses espeak-ng — fully local, zero network calls.

Single-thread pipeline:
  Worker receives text → runs espeak-ng subprocess → blocks until audio done.
  Queue ensures no overlap — sentences play one at a time.
"""

import queue
import subprocess
import threading
import time

from config import Config


class TTSEngine:
    """espeak-ng TTS with single worker queue (no overlap)."""

    def __init__(self):
        self.audio_queue = queue.Queue()
        self._done_event = threading.Event()
        self._done_event.set()  # start as "done" (nothing pending)

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        self.is_speaking = False
        self._pending_chunks = 0
        self._pending_lock = threading.Lock()
        self.metrics_logger = None
        self._turn_tts_start = None

    def _worker(self):
        """Single worker: espeak-ng subprocess per text item.

        Blocks until espeak-ng finishes each item — guarantees no overlap.
        """
        while True:
            text = self.audio_queue.get()
            if text is None:
                self.audio_queue.task_done()
                break

            self.is_speaking = True
            t_start = time.perf_counter()

            try:
                cmd = [
                    "espeak-ng",
                    "-s", str(Config.ESPEAK_SPEED),
                    "-v", Config.ESPEAK_VOICE,
                    text,
                ]
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                gen_ms = (time.perf_counter() - t_start) * 1000
                if self.metrics_logger:
                    self.metrics_logger.log_tts_first_audio()
                    self.metrics_logger.log_tts_generation(gen_ms)

            except Exception as e:
                print(f"⚠️ TTS error: {e}")

            finally:
                with self._pending_lock:
                    self._pending_chunks -= 1
                    if self._pending_chunks <= 0:
                        self._pending_chunks = 0
                        self.is_speaking = False
                        self._done_event.set()

                self.audio_queue.task_done()

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
        self.audio_queue.join()
        self._done_event.wait(timeout=60)

        if self._turn_tts_start is not None and self.metrics_logger:
            total_ms = (time.perf_counter() - self._turn_tts_start) * 1000
            self.metrics_logger.log_tts_total(total_ms)
        self._turn_tts_start = None

    def shutdown(self):
        """Shutdown TTS."""
        self.audio_queue.put(None)
