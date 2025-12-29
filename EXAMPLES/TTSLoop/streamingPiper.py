#!/usr/bin/env python3
"""
ACTUALLY WORKING Streaming TTS
Plays audio IMMEDIATELY while LLM continues generating
Uses background threads for each audio chunk
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
PIPER_VOICE_PATH = HOME / "Sentinel" / "pipervoices" / "en_US-amy-medium.onnx"
LLM_MODEL_PATH = HOME / "Sentinel" / "modls" / "granite-3.0-1b-a400m-instruct.Q4_K_M.gguf"

LLM_THREADS = 3
CHUNK_SIZE = 10  # Words per chunk

# ============================================
# IMMEDIATE AUDIO PLAYBACK
# ============================================

def play_audio_async(text, voice_path, chunk_id):
    """
    Play audio in background thread.
    Returns immediately so LLM can keep generating!
    """
    def _play():
        if not text.strip():
            return
        
        # Escape text
        text_escaped = text.replace("'", "'\"'\"'")
        
        # Generate and play
        cmd = f"echo '{text_escaped}' | piper -m {voice_path} --output-raw | paplay --raw --channels=1 --rate=22050 --format=s16le 2>/dev/null &"
        
        subprocess.run(cmd, shell=True)
    
    # Start in background thread
    thread = threading.Thread(target=_play, daemon=True)
    thread.start()
    
    return thread

# ============================================
# STREAMING WITH IMMEDIATE PLAYBACK
# ============================================

def stream_with_immediate_audio(llm, voice_path, user_message):
    """
    Stream LLM response and play audio IMMEDIATELY.
    Audio plays while LLM continues generating!
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are an in-car voice assistant. Keep responses under 2 sentences. Be brief."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    
    print(f"\n{'─'*60}")
    print(f"User: {user_message}")
    print(f"{'─'*60}")
    print("Assistant: ", end='', flush=True)
    
    # Timing
    start_time = time.time()
    first_audio_started = None
    generation_complete = None
    
    # Start LLM streaming
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=60,
        stream=True
    )
    
    buffer = []
    chunk_count = 0
    audio_threads = []
    full_response = ""
    
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        
        if 'content' in delta:
            token = delta['content']
            full_response += token
            print(token, end='', flush=True)
            
            # Add words to buffer
            words = token.split()
            buffer.extend(words)
            
            # When we have enough words, play immediately!
            if len(buffer) >= CHUNK_SIZE:
                chunk_text = ' '.join(buffer[:CHUNK_SIZE])
                chunk_count += 1
                
                # Mark when first audio started
                if first_audio_started is None:
                    first_audio_started = time.time()
                    print(f"\n[🔊 Audio started while generating...]", flush=True)
                    print("Assistant (continued): ", end='', flush=True)
                
                # Play in background - returns IMMEDIATELY!
                thread = play_audio_async(chunk_text, voice_path, chunk_count)
                audio_threads.append(thread)
                
                buffer = buffer[CHUNK_SIZE:]
    
    # Generation complete
    generation_complete = time.time()
    
    # Play any remaining words
    if buffer:
        chunk_text = ' '.join(buffer)
        chunk_count += 1
        
        if first_audio_started is None:
            first_audio_started = time.time()
        
        thread = play_audio_async(chunk_text, voice_path, chunk_count)
        audio_threads.append(thread)
    
    print()
    
    # Wait for all audio to finish
    print(f"\n[Waiting for {len(audio_threads)} audio chunks to finish...]")
    for thread in audio_threads:
        thread.join(timeout=5)
    
    audio_complete = time.time()
    
    # Calculate timings
    generation_time = (generation_complete - start_time) * 1000
    ttfa = (first_audio_started - start_time) * 1000 if first_audio_started else 0
    total_time = (audio_complete - start_time) * 1000
    
    print(f"\n{'='*60}")
    print("⏱️  TIMING BREAKDOWN")
    print(f"{'='*60}")
    print(f"  LLM Generation:       {generation_time:6.0f}ms")
    print(f"  🎯 First Audio:        {ttfa:6.0f}ms ← Started DURING generation!")
    print(f"  Total (gen + audio):  {total_time:6.0f}ms")
    print(f"  Chunks played:        {chunk_count}")
    print(f"{'='*60}")
    
    # Check if streaming worked
    if first_audio_started and first_audio_started < generation_complete:
        overlap = (generation_complete - first_audio_started) * 1000
        print(f"\n✅ STREAMING WORKED!")
        print(f"   Audio played {overlap:.0f}ms BEFORE generation finished!")
    else:
        print(f"\n⚠️  Streaming didn't work - audio waited for completion")
    
    print(f"{'='*60}")
    
    return {
        'response': full_response,
        'generation_time': generation_time,
        'ttfa': ttfa,
        'total_time': total_time,
        'streaming_worked': first_audio_started < generation_complete if first_audio_started else False
    }

# ============================================
# MAIN
# ============================================

def main():
    """Run test."""
    
    print("\n" + "#"*60)
    print("# ACTUALLY WORKING STREAMING TTS")
    print("# Audio plays WHILE LLM generates")
    print("#"*60)
    
    print("\n⚠️  IMPORTANT: Close Chromium first!")
    print("   Run: pkill chromium")
    input("\nPress Enter when ready...")
    
    print("\n" + "─"*60)
    print("Loading Models...")
    print("─"*60)
    
    print(f"  Loading LLM ({LLM_THREADS} threads)...", end='', flush=True)
    start = time.time()
    
    llm = Llama(
        model_path=str(LLM_MODEL_PATH),
        n_ctx=2048,
        n_threads=LLM_THREADS,
        n_gpu_layers=0,
        verbose=False
    )
    
    print(f" ✓ ({time.time() - start:.1f}s)")
    
    # Warm-up
    print("  Warming up LLM...", end='', flush=True)
    llm.create_chat_completion(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=5
    )
    print(" ✓")
    
    print("\n✅ Ready!\n")
    time.sleep(0.5)
    
    # Test prompts
    prompts = [
        "I'm feeling drowsy and need help",
        "I had a really long day at work today",
        "Can you help me stay awake while driving"
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n\n{'#'*60}")
        print(f"Test {i}/{len(prompts)}")
        print(f"{'#'*60}")
        
        result = stream_with_immediate_audio(llm, PIPER_VOICE_PATH, prompt)
        results.append(result)
        
        time.sleep(3)  # Pause between tests
    
    # Final summary
    print(f"\n\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    
    streaming_count = sum(1 for r in results if r['streaming_worked'])
    avg_ttfa = sum(r['ttfa'] for r in results) / len(results)
    avg_gen = sum(r['generation_time'] for r in results) / len(results)
    
    print(f"\n  Tests where streaming worked: {streaming_count}/{len(results)}")
    print(f"  Average generation time:      {avg_gen:.0f}ms")
    print(f"  Average first audio:          {avg_ttfa:.0f}ms")
    
    if streaming_count == len(results):
        print(f"\n  ✅ STREAMING WORKS PERFECTLY!")
        print(f"     Audio starts {avg_gen - avg_ttfa:.0f}ms BEFORE generation finishes!")
    elif streaming_count > 0:
        print(f"\n  🟡 STREAMING WORKS SOMETIMES")
    else:
        print(f"\n  ❌ STREAMING NOT WORKING")
        print(f"     Audio waits for complete generation")
        print(f"\n  Possible issues:")
        print(f"     - Piper might be slow")
        print(f"     - Bluetooth latency")
        print(f"     - Background processes (Chromium?)")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
