#!/usr/bin/env python3
"""
CLEAN Streaming TTS (NO OVERLAP)
- Uses eSpeak NG
- Sentence-based splitting
- Single playback queue
- Audio is ALWAYS understandable
"""

import time
import subprocess
import threading
import queue
from llama_cpp import Llama
from pathlib import Path

# ===============================
# CONFIG
# ===============================
LLM_MODEL_PATH = Path.home() / "Sentinel" / "modls" / "granite-3.0-1b-a400m-instruct.Q4_K_M.gguf"

LLM_THREADS = 3

# ===============================
# AUDIO QUEUE WORKER
# ===============================
audio_queue = queue.Queue()

def audio_worker():
    """Plays sentences one at a time — NO overlap."""
    while True:
        sentence = audio_queue.get()
        if sentence is None:
            break

        cmd = [
            "espeak-ng",
            "-s", "165",      # speed
            "-v", "en-us",
            sentence
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        audio_queue.task_done()

# Start audio worker thread
threading.Thread(target=audio_worker, daemon=True).start()

# ===============================
# STREAMING + SENTENCE SPLIT
# ===============================
def stream_with_clean_audio(llm, user_message):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an in-car assistant keeping a driver awake. "
                "Be calm, brief, and supportive. Max 2 sentences."
            )
        },
        {"role": "user", "content": user_message}
    ]

    print("\nUser:", user_message)
    print("Assistant:", end=" ", flush=True)

    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.6,
        max_tokens=80,
        stream=True
    )

    buffer = ""
    full_response = ""
    first_audio_time = None
    start_time = time.time()

    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" not in delta:
            continue

        token = delta["content"]
        print(token, end="", flush=True)

        buffer += token
        full_response += token

        # Sentence boundary detected
        if any(p in buffer for p in [".", "!", "?"]):
            sentence = buffer.strip()
            buffer = ""

            if sentence:
                audio_queue.put(sentence)

                if first_audio_time is None:
                    first_audio_time = time.time()
                    print("\n[🔊 Speaking...]\nAssistant:", end=" ", flush=True)

    # Speak remaining buffer
    if buffer.strip():
        audio_queue.put(buffer.strip())

    audio_queue.join()

    end_time = time.time()

    print("\n")
    print("⏱ Timing:")
    print(f"  First audio: {(first_audio_time - start_time)*1000:.0f} ms")
    print(f"  Total time: {(end_time - start_time)*1000:.0f} ms")

    return full_response

# ===============================
# MAIN
# ===============================
def main():
    print("\nLoading LLM...")
    llm = Llama(
        model_path=str(LLM_MODEL_PATH),
        n_ctx=2048,
        n_threads=LLM_THREADS,
        n_gpu_layers=0,
        verbose=False
    )

    # Warm-up
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )

    prompts = [
        "I'm feeling drowsy while driving",
        "I've been on the road for hours",
        "Help me stay awake"
    ]

    for p in prompts:
        stream_with_clean_audio(llm, p)
        time.sleep(2)

if __name__ == "__main__":
    main()
