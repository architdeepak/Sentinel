#!/usr/bin/env python3
"""
ACTUALLY WORKING Streaming TTS (eSpeak NG)
Audio starts IMMEDIATELY while LLM generates
Optimized for Raspberry Pi
"""

import time
import subprocess
import threading
from pathlib import Path
from llama_cpp import Llama

# ============================================
# CONFIGURATION
# ============================================
HOME = Path.home()
LLM_MODEL_PATH = HOME / "Sentinel" / "modls" / "granite-3.0-1b-a400m-instruct.Q4_K_M.gguf"

LLM_THREADS = 3
CHUNK_SIZE = 8  # smaller = faster speech start

# ============================================
# INSTANT AUDIO (eSpeak NG)
# ============================================

def play_audio_async(text, chunk_id):
    """
    Fire-and-forget speech.
    Returns immediately.
    """
    def _speak():
        if not text.strip():
            return

        # Slight pause smoothing
        text_clean = text.replace("\n", " ").strip()

        cmd = [
            "espeak-ng",
            "-s", "165",        # speed (words/min)
            "-v", "en-us",      # voice
            text_clean
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
    return thread

# ============================================
# STREAMING LLM + AUDIO
# ============================================

def stream_with_immediate_audio(llm, user_message):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an in-car safety assistant. "
                "Keep responses under 2 sentences. "
                "Speak clearly and urgently but calm."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    print("\n" + "─" * 60)
    print(f"User: {user_message}")
    print("─" * 60)
    print("Assistant: ", end="", flush=True)

    start_time = time.time()
    first_audio_time = None
    generation_done = None

    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.6,
        max_tokens=60,
        stream=True
    )

    buffer = []
    audio_threads = []
    chunk_count = 0
    full_response = ""

    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" not in delta:
            continue

        token = delta["content"]
        full_response += token
        print(token, end="", flush=True)

        words = token.split()
        buffer.extend(words)

        if len(buffer) >= CHUNK_SIZE:
            chunk_text = " ".join(buffer[:CHUNK_SIZE])
            buffer = buffer[CHUNK_SIZE:]
            chunk_count += 1

            if first_audio_time is None:
                first_audio_time = time.time()
                print("\n[🔊 Speaking while generating...]\nAssistant: ", end="", flush=True)

            t = play_audio_async(chunk_text, chunk_count)
            audio_threads.append(t)

    generation_done = time.time()

    # Speak leftover words
    if buffer:
        chunk_count += 1
        t = play_audio_async(" ".join(buffer), chunk_count)
        audio_threads.append(t)

    print("\n")

    for t in audio_threads:
        t.join(timeout=3)

    end_time = time.time()

    print("\n" + "=" * 60)
    print("⏱️ TIMING")
    print("=" * 60)
    print(f"LLM Generation: {(generation_done - start_time)*1000:.0f} ms")
    print(f"First Audio:     {(first_audio_time - start_time)*1000:.0f} ms")
    print(f"Total Time:     {(end_time - start_time)*1000:.0f} ms")
    print(f"Chunks Spoken:  {chunk_count}")

    if first_audio_time and first_audio_time < generation_done:
        print("\n✅ TRUE STREAMING CONFIRMED")
    else:
        print("\n⚠️ Audio waited for generation")

    return full_response

# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "#" * 60)
    print("# STREAMING LLM + eSpeak NG")
    print("# Ultra-low latency TTS for Raspberry Pi")
    print("#" * 60)

    print("\nLoading LLM...")
    start = time.time()

    llm = Llama(
        model_path=str(LLM_MODEL_PATH),
        n_ctx=2048,
        n_threads=LLM_THREADS,
        n_gpu_layers=0,
        verbose=False
    )

    print(f"LLM loaded in {time.time() - start:.1f}s")

    # Warm-up
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )

    prompts = [
        "I'm feeling sleepy while driving",
        "Help me stay awake",
        "I need to stay alert right now"
    ]

    for p in prompts:
        stream_with_immediate_audio(llm, p)
        time.sleep(2)

if __name__ == "__main__":
    main()
