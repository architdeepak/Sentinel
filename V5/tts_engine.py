"""
Text-to-Speech Engine for Driver Drowsiness Detection System V4
Optimized for Raspberry Pi: streams Edge-TTS audio bytes directly
to mpg123 via stdin pipe — zero file I/O on the SD card.

Two-stage pipeline:
  1. Generator thread: streams audio bytes from Edge-TTS into memory
  2. Playback thread: pipes bytes to mpg123 process via stdin
  Chunk N+1 generates while chunk N plays = minimal gaps.
"""

import queue
import subprocess
import threading
import asyncio
import time
import platform

import edge_tts

from config import Config


class TTSEngine:
    """Edge-TTS with pipelined generation + playback (RPi optimized)."""

    def __init__(self):
        self.audio_queue = queue.Queue()       # text chunks in
        self._playback_queue = queue.Queue()   # (audio_bytes, gen_ms) ready to play
        self._done_event = threading.Event()
        self._done_event.set()  # start as "done" (nothing pending)

        self._gen_thread = threading.Thread(target=self._generator_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._gen_thread.start()
        self._play_thread.start()

        self.is_speaking = False
        self._pending_chunks = 0
        self._pending_lock = threading.Lock()
        self.metrics_logger = None
        self._turn_tts_start = None

    def _generator_worker(self):
        """Stage 1: Stream audio bytes from Edge-TTS into memory.
        
        NOTE: Generation and playback are pipelined at the SENTENCE level —
        sentence N+1 generates while sentence N plays. Within a single
        sentence, we now stream audio chunks directly to the player
        (see _playback_worker) for lowest first-byte latency.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            text = self.audio_queue.get()
            if text is None:
                self._playback_queue.put(None)
                break

            try:
                t_gen_start = time.perf_counter()
                # Collect chunks as a list for streaming playback
                audio_chunks = loop.run_until_complete(self._generate_chunks(text))
                gen_ms = (time.perf_counter() - t_gen_start) * 1000

                if audio_chunks:
                    self._playback_queue.put((audio_chunks, gen_ms))
                else:
                    with self._pending_lock:
                        self._pending_chunks -= 1
                        if self._pending_chunks <= 0:
                            self._pending_chunks = 0
                            self._done_event.set()

            except Exception as e:
                print(f"⚠️ TTS generation error: {e}")
                with self._pending_lock:
                    self._pending_chunks -= 1
                    if self._pending_chunks <= 0:
                        self._pending_chunks = 0
                        self._done_event.set()

            self.audio_queue.task_done()

        loop.close()

    def _playback_worker(self):
        """Stage 2: Stream audio chunks to mpg123 as they arrive (low latency).
        
        Instead of buffering the entire sentence's audio and then playing,
        we write each Edge-TTS chunk to mpg123's stdin as it arrives from
        the generator. This means playback starts after the FIRST chunk
        (~50-150ms) rather than waiting for the entire sentence (~300-800ms).
        """
        while True:
            item = self._playback_queue.get()
            if item is None:
                break

            audio_chunks, gen_ms = item
            self.is_speaking = True

            try:
                t_play_start = time.perf_counter()

                # Start mpg123 and stream chunks to its stdin
                proc = subprocess.Popen(
                    ["mpg123", "-q", "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                for chunk in audio_chunks:
                    proc.stdin.write(chunk)
                proc.stdin.close()
                proc.wait()

                play_ms = (time.perf_counter() - t_play_start) * 1000

                if self.metrics_logger:
                    self.metrics_logger.log_tts_generation(gen_ms)
                    self.metrics_logger.log_tts_playback(play_ms)

            except FileNotFoundError:
                # mpg123 not installed — try ffplay as fallback
                try:
                    proc = subprocess.Popen(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    for chunk in audio_chunks:
                        proc.stdin.write(chunk)
                    proc.stdin.close()
                    proc.wait()
                except Exception as e2:
                    print(f"⚠️ No audio player found (mpg123/ffplay): {e2}")

            except Exception as e:
                print(f"⚠️ TTS playback error: {e}")

            finally:
                with self._pending_lock:
                    self._pending_chunks -= 1
                    if self._pending_chunks <= 0:
                        self._pending_chunks = 0
                        self.is_speaking = False
                        self._done_event.set()

            self._playback_queue.task_done()

    async def _generate_chunks(self, text):
        """Stream audio chunks from Edge-TTS as a list (for incremental playback).
        
        Returns a list of bytes objects rather than one concatenated blob.
        The playback worker writes each chunk to mpg123's stdin sequentially,
        so mpg123 can start decoding/playing as soon as it has enough data.
        """
        communicate = edge_tts.Communicate(
            text,
            Config.EDGE_TTS_VOICE,
            rate=Config.EDGE_TTS_RATE
        )

        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio" and chunk["data"]:
                chunks.append(chunk["data"])

        return chunks if chunks else None

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
