#!/usr/bin/env python3
"""
Standalone TTS + LLM + STT Test Script
Test the conversation flow without drowsiness detection
"""

import time
import subprocess
import threading
import queue
import json
from pathlib import Path
from llama_cpp import Llama
from vosk import Model, KaldiRecognizer
import pyaudio

# =========================
# CONFIGURATION
# =========================
class Config:
    # Model paths
    LLM_MODEL_PATH = Path.home() / "Sentinel" / "modls" / "granite-3.0-1b-a400m-instruct.Q4_K_M.gguf"
    VOSK_MODEL_PATH = Path.home() / "Sentinel" / "Sentinel" / "vosk-model-small-en-us-0.15"
    
    # LLM settings
    LLM_THREADS = 3
    LLM_CONTEXT = 2048
    LLM_MAX_TOKENS = 150
    
    # Audio settings
    ESPEAK_SPEED = 165
    ESPEAK_VOICE = "en-us"
    VOSK_SAMPLE_RATE = 16000
    VOSK_BUFFER_SIZE = 8192

# =========================
# TEXT-TO-SPEECH
# =========================
class TTSEngine:
    """Simple TTS using espeak-ng."""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        self.is_speaking = False
    
    def _audio_worker(self):
        """Process TTS queue."""
        while True:
            sentence = self.audio_queue.get()
            if sentence is None:
                break
            
            self.is_speaking = True
            cmd = [
                "espeak-ng",
                "-s", str(Config.ESPEAK_SPEED),
                "-v", Config.ESPEAK_VOICE,
                sentence
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.is_speaking = False
            self.audio_queue.task_done()
    
    def speak(self, text):
        """Queue text for speech."""
        if text and text.strip():
            print(f"🔊 Speaking: {text}")
            self.audio_queue.put(text.strip())
    
    def wait_until_done(self):
        """Wait for all speech to finish."""
        self.audio_queue.join()
    
    def shutdown(self):
        """Shutdown TTS."""
        self.audio_queue.put(None)

# =========================
# SPEECH-TO-TEXT
# =========================
class STTEngine:
    """Vosk STT with diagnostics."""
    
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.mic = None
        self.stream = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Vosk."""
        try:
            if not Config.VOSK_MODEL_PATH.exists():
                print(f"❌ Vosk model not found: {Config.VOSK_MODEL_PATH}")
                return
            
            print(f"✓ Vosk model found")
            print("📦 Loading Vosk model...")
            self.model = Model(str(Config.VOSK_MODEL_PATH))
            self.recognizer = KaldiRecognizer(self.model, Config.VOSK_SAMPLE_RATE)
            print("✓ Vosk loaded")
            
            print("🎤 Setting up microphone...")
            self.mic = pyaudio.PyAudio()
            
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=Config.VOSK_SAMPLE_RATE,
                input=True,
                frames_per_buffer=Config.VOSK_BUFFER_SIZE
            )
            self.stream.start_stream()
            print("✓ Microphone ready\n")
            
        except Exception as e:
            print(f"⚠️ STT init failed: {e}")
            self.model = None
    
    def _clear_buffer(self):
        """Clear audio buffer."""
        if not self.stream:
            return
        try:
            for _ in range(5):
                self.stream.read(Config.VOSK_BUFFER_SIZE, exception_on_overflow=False)
        except:
            pass
    
    def listen(self, timeout=20, show_diagnostics=True):
        """
        Listen with timeout - DIAGNOSTIC VERSION.
        
        Args:
            timeout: Max seconds to listen
            show_diagnostics: Show what's happening
        """
        if not self.model:
            print("⚠️ STT not available")
            return None

        print(f"\n🎤 Listening (timeout: {timeout}s)...")
        if show_diagnostics:
            print("   [Will show dots for each 0.5s chunk]")
        
        # Clear buffer
        self._clear_buffer()
        self.recognizer.Reset()
        
        start_time = time.time()
        collected_text = []
        last_text_time = None
        chunks_processed = 0
        
        iterations_per_second = Config.VOSK_SAMPLE_RATE / Config.VOSK_BUFFER_SIZE
        max_iterations = int(timeout * iterations_per_second)
        
        try:
            for iteration in range(max_iterations):
                elapsed = time.time() - start_time
                
                # Diagnostic: show progress
                if show_diagnostics and iteration % 10 == 0:
                    print(".", end="", flush=True)
                
                # Read audio
                data = self.stream.read(
                    Config.VOSK_BUFFER_SIZE,
                    exception_on_overflow=False
                )
                chunks_processed += 1
                
                # Process
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    
                    if text:
                        collected_text.append(text)
                        last_text_time = time.time()
                        print(f"\n   📝 Heard: '{text}'")
                        if show_diagnostics:
                            print("   [Continuing to listen...]")
                
                # Silence detection - only after we've heard something
                if last_text_time:
                    silence_duration = time.time() - last_text_time
                    if silence_duration > 2.0:
                        if show_diagnostics:
                            print(f"\n   ⏸️  2 seconds of silence - ending (chunks: {chunks_processed})")
                        break
                
                # Timeout check
                if elapsed > timeout:
                    if show_diagnostics:
                        print(f"\n   ⏱️  Timeout reached (chunks: {chunks_processed})")
                    break
            
            # Final result
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                collected_text.append(final_text)
                print(f"   📝 Final: '{final_text}'")
            
            full_text = " ".join(collected_text).strip()
            
            if full_text:
                print(f"✓ Complete: '{full_text}'")
                return full_text
            else:
                print("⚠️  No speech recognized")
                return None
                
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.mic:
                self.mic.terminate()
        except:
            pass

# =========================
# LLM ASSISTANT
# =========================
class LLMAssistant:
    """LLM with streaming TTS."""
    
    def __init__(self, tts):
        self.tts = tts
        self.llm = None
        self.messages = []
        self._initialize()
    
    def _initialize(self):
        """Load LLM."""
        try:
            print("🧠 Loading LLM...")
            self.llm = Llama(
                model_path=str(Config.LLM_MODEL_PATH),
                n_ctx=Config.LLM_CONTEXT,
                n_threads=Config.LLM_THREADS,
                n_gpu_layers=0,
                verbose=False
            )
            
            # Warm-up
            self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            print("✓ LLM ready\n")
            
        except Exception as e:
            print(f"⚠️ LLM init failed: {e}")
            self.llm = None
    
    def start_conversation(self):
        """Start new conversation."""
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. "
                    "Keep responses SHORT (1-3 sentences). "
                    "Be conversational and friendly."
                )
            }
        ]
    
    def get_response(self, user_message):
        """Get response with streaming TTS."""
        if not self.llm:
            return "Sorry, I'm not available."
        
        self.messages.append({"role": "user", "content": user_message})
        print(f"\n👤 You: {user_message}")
        print("🤖 Assistant: ", end="", flush=True)
        
        stream = self.llm.create_chat_completion(
            messages=self.messages,
            temperature=0.7,
            max_tokens=Config.LLM_MAX_TOKENS,
            stream=True
        )
        
        buffer = ""
        full_response = ""
        
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            if "content" not in delta:
                continue
            
            token = delta["content"]
            print(token, end="", flush=True)
            
            buffer += token
            full_response += token
            
            # Speak on sentence boundaries
            if any(p in buffer for p in [".", "!", "?"]):
                sentence = buffer.strip()
                buffer = ""
                if sentence:
                    self.tts.speak(sentence)
        
        # Speak remaining
        if buffer.strip():
            self.tts.speak(buffer.strip())
        
        print()
        
        self.messages.append({"role": "assistant", "content": full_response})
        return full_response

# =========================
# CONVERSATION LOOP
# =========================
def run_conversation(tts, stt, llm):
    """Run test conversation."""
    print("\n" + "="*60)
    print("🎙️  CONVERSATION TEST")
    print("="*60)
    print("Say 'exit' or 'quit' to end\n")
    
    llm.start_conversation()
    
    # Initial greeting
    tts.speak("Hello! I'm ready to chat. What would you like to talk about?")
    tts.wait_until_done()
    
    turn = 0
    max_turns = 5
    
    while turn < max_turns:
        turn += 1
        print(f"\n--- Turn {turn}/{max_turns} ---")
        
        # Listen
        user_input = stt.listen(timeout=20, show_diagnostics=True)
        
        if user_input is None:
            print("\n⚠️ No input detected")
            tts.speak("I didn't hear anything. Try again?")
            tts.wait_until_done()
            continue
        
        # Check for exit
        if any(word in user_input.lower() for word in ['exit', 'quit', 'bye', 'goodbye']):
            print("\n👋 User ended conversation")
            tts.speak("Goodbye!")
            tts.wait_until_done()
            break
        
        # Get response
        llm.get_response(user_input)
        tts.wait_until_done()
    
    if turn >= max_turns:
        print(f"\n🔚 Reached {max_turns} turns")
        tts.speak("That's all for now. Goodbye!")
        tts.wait_until_done()
    
    print("\n" + "="*60)
    print("✓ Conversation ended")
    print("="*60)

# =========================
# MAIN
# =========================
def main():
    """Test the TTS + LLM + STT flow."""
    print("\n" + "#"*60)
    print("# TTS + LLM + STT CONVERSATION TEST")
    print("#"*60 + "\n")
    
    # Initialize
    tts = TTSEngine()
    stt = STTEngine()
    llm = LLMAssistant(tts)
    
    if not stt.model or not llm.llm:
        print("❌ Failed to initialize components")
        return
    
    print("✓ All components ready!\n")
    input("Press ENTER to start conversation...")
    
    try:
        run_conversation(tts, stt, llm)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
    finally:
        print("\n🔄 Cleaning up...")
        stt.cleanup()
        tts.shutdown()
        print("✓ Done!")

if __name__ == "__main__":
    main()
