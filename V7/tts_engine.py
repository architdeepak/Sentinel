"""
Text-to-Speech Engine for Driver Drowsiness Detection System V6
Uses Deepgram Aura TTS REST API — streams audio bytes directly
to mpg123 via stdin pipe — zero file I/O on the SD card.

Single-thread pipeline:
  Worker receives text → HTTP-streams Deepgram audio → pipes each chunk
  directly into mpg123's stdin as it arrives.  Playback starts after the
  FIRST chunk (~100-200ms) instead of waiting for the full download.
"""

import queue
import subprocess
import threading
import time
import requests

from config import Config


class TTSEngine:
    """Deepgram TTS with direct-pipe streaming (lowest latency)."""

    def __init__(self):
        self.audio_queue = queue.Queue()       # text in
        self._done_event = threading.Event()
        self._done_event.set()  # start as "done" (nothing pending)

        self._dg_url = (
            f"https://api.deepgram.com/v1/speak"
            f"?model={Config.DEEPGRAM_TTS_VOICE}"
            f"&encoding=mp3&speed={Config.DEEPGRAM_TTS_SPEED}"
        )
        self._dg_headers = {
            "Authorization": f"Token {Config.DEEPGRAM_API_KEY}",
            "Content-Type": "application/json",
        }

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        self.is_speaking = False
        self._pending_chunks = 0
        self._pending_lock = threading.Lock()
        self.metrics_logger = None
        self._turn_tts_start = None

    def _worker(self):
        """Single worker: Deepgram HTTP stream → mpg123 stdin in real-time.

        For each text chunk:
          1. POST to Deepgram TTS with stream=True
          2. Start mpg123
          3. Pipe each HTTP chunk directly to mpg123's stdin
        Audio begins playing after the first ~1KB arrives (~100-200ms).
        No intermediate buffering.
        """
        session = requests.Session()
        session.headers.update(self._dg_headers)
        while True:
            text = self.audio_queue.get()
            if text is None:
                self.audio_queue.task_done()
                break

            self.is_speaking = True
            t_start = time.perf_counter()

            try:
                # Stream=True: iter_content yields chunks as they arrive
                resp = session.post(
                    self._dg_url, json={"text": text},
                    stream=True, timeout=15,
                )
                resp.raise_for_status()

                # Start player BEFORE we have all audio — pipe as we receive
                proc = self._start_player()
                if proc is None:
                    resp.close()
                    continue

                gen_ms = (time.perf_counter() - t_start) * 1000  # time-to-first-byte approx
                first_byte_logged = False

                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        proc.stdin.write(chunk)
                        if not first_byte_logged:
                            first_byte_logged = True
                            if self.metrics_logger:
                                self.metrics_logger.log_tts_first_audio()

                proc.stdin.close()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

                total_ms = (time.perf_counter() - t_start) * 1000

                if self.metrics_logger:
                    self.metrics_logger.log_tts_generation(gen_ms)
                    self.metrics_logger.log_tts_playback(total_ms - gen_ms)

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

    @staticmethod
    def _start_player():
        """Start mpg123 (or ffplay fallback) and return the Popen handle."""
        for cmd in [
            ["mpg123", "-q", "--scale", "32768", "-"],   # --scale amplifies output (max 32768)
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", "100", "-"],
        ]:
            try:
                return subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (FileNotFoundError, OSError):
                continue
        print("⚠️ No audio player found (mpg123/ffplay)")
        return None

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
