#!/usr/bin/env python3
"""
Sesame CSM TTS Test Script
Testing contextual speech generation on Raspberry Pi
"""

import torch
import torchaudio
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# Optional: LLM integration for conversation
try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except:
    HAS_LLAMA = False
    print("⚠️ llama-cpp-python not found - will use predefined responses")

# =========================
# CONFIGURATION
# =========================
class Config:
    # CSM Model
    CSM_MODEL = "sesame/csm-1b"  # or local path
    
    # LLM (optional - for generating conversation)
    LLM_MODEL_PATH = Path.home() / "Sentinel" / "modls" / "granite-3.0-1b-a400m-instruct.Q4_K_M.gguf"
    
    # Audio settings
    SAMPLE_RATE = 24000
    OUTPUT_DIR = Path("/tmp/csm_audio")
    
    # Device
    DEVICE = "cpu"  # RPi doesn't have GPU

# =========================
# CSM TTS ENGINE
# =========================
class CSMEngine:
    """Contextual Speech Model TTS Engine."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = Config.DEVICE
        Config.OUTPUT_DIR.mkdir(exist_ok=True)
        self._initialize()
    
    def _initialize(self):
        """Initialize CSM model."""
        print("🔄 Loading Sesame CSM model...")
        print("⚠️ This may take a while on RPi (downloading ~400MB model)...")
        
        start = time.time()
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(Config.CSM_MODEL)
            self.model = AutoModel.from_pretrained(
                Config.CSM_MODEL,
                torch_dtype=torch.float32,  # CPU requires float32
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            load_time = time.time() - start
            print(f"✓ CSM loaded in {load_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Failed to load CSM: {e}")
            self.model = None
    
    def generate_speech(self, text, context_audio=None, output_path=None):
        """
        Generate speech from text with optional context.
        
        Args:
            text: Text to speak
            context_audio: Optional audio tensor for contextual adaptation
            output_path: Where to save audio (auto-generated if None)
        """
        if not self.model:
            print("❌ CSM not loaded")
            return None
        
        if output_path is None:
            output_path = Config.OUTPUT_DIR / f"speech_{int(time.time()*1000)}.wav"
        
        print(f"🎤 Generating: '{text}'")
        start = time.time()
        
        try:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate audio
            with torch.no_grad():
                if context_audio is not None:
                    # Use contextual generation
                    output = self.model.generate(
                        **inputs,
                        audio_context=context_audio,
                        max_new_tokens=1024
                    )
                else:
                    # Basic generation
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=1024
                    )
            
            # Decode audio (CSM outputs Mimi codes that need decoding)
            # Note: This is simplified - actual implementation may vary
            audio_codes = output[0]
            
            # Save audio
            # CSM should have a decode method, but may need vocoder
            # For now, we'll try direct output
            torchaudio.save(
                str(output_path),
                audio_codes.cpu().unsqueeze(0),
                Config.SAMPLE_RATE
            )
            
            gen_time = time.time() - start
            print(f"✓ Generated in {gen_time:.2f}s → {output_path.name}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
    
    def play_audio(self, audio_path):
        """Play generated audio."""
        import subprocess
        try:
            subprocess.run(
                ["aplay", str(audio_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"⚠️ Playback error: {e}")
            print(f"   Try: aplay {audio_path}")

# =========================
# LLM FOR CONVERSATION
# =========================
class SimpleLLM:
    """Simple LLM wrapper for conversation generation."""
    
    def __init__(self):
        self.llm = None
        self.messages = []
        
        if HAS_LLAMA and Config.LLM_MODEL_PATH.exists():
            self._initialize()
    
    def _initialize(self):
        """Load LLM."""
        print("🧠 Loading LLM...")
        try:
            self.llm = Llama(
                model_path=str(Config.LLM_MODEL_PATH),
                n_ctx=2048,
                n_threads=3,
                n_gpu_layers=0,
                verbose=False
            )
            print("✓ LLM loaded")
        except Exception as e:
            print(f"⚠️ LLM load failed: {e}")
            self.llm = None
    
    def get_response(self, user_input):
        """Get LLM response."""
        if not self.llm:
            # Fallback responses
            responses = [
                "That's interesting! Tell me more about that.",
                "I understand. How are you feeling now?",
                "Let's talk about something to keep you alert. What's your destination today?",
                "Good point! Have you tried adjusting the temperature or opening a window?",
            ]
            import random
            return random.choice(responses)
        
        self.messages.append({"role": "user", "content": user_input})
        
        response = self.llm.create_chat_completion(
            messages=self.messages,
            max_tokens=80,
            temperature=0.7
        )
        
        reply = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        
        return reply

# =========================
# TEST SCENARIOS
# =========================
def test_basic_tts(csm):
    """Test 1: Basic text-to-speech."""
    print("\n" + "="*60)
    print("TEST 1: Basic Text-to-Speech")
    print("="*60)
    
    test_texts = [
        "Hello! I'm your in-car assistant.",
        "I notice you might be feeling drowsy. Let's have a conversation to keep you alert.",
        "How was your day today?",
        "That sounds interesting! Tell me more about that.",
    ]
    
    for text in test_texts:
        audio_path = csm.generate_speech(text)
        if audio_path:
            print("▶️ Playing audio...")
            csm.play_audio(audio_path)
            time.sleep(1)

def test_contextual_speech(csm):
    """Test 2: Contextual speech generation."""
    print("\n" + "="*60)
    print("TEST 2: Contextual Speech Generation")
    print("="*60)
    
    print("\n📝 Testing how CSM adapts tone with context...")
    
    # First utterance (establishes context)
    text1 = "I understand you're feeling tired."
    print(f"\n1️⃣ First utterance: '{text1}'")
    audio1_path = csm.generate_speech(text1)
    
    if audio1_path:
        csm.play_audio(audio1_path)
        time.sleep(1)
        
        # Load as context
        try:
            context_audio, sr = torchaudio.load(str(audio1_path))
            
            # Second utterance (uses first as context)
            text2 = "Let's chat to help you stay awake."
            print(f"\n2️⃣ Second utterance with context: '{text2}'")
            audio2_path = csm.generate_speech(text2, context_audio=context_audio)
            
            if audio2_path:
                csm.play_audio(audio2_path)
                time.sleep(1)
                
        except Exception as e:
            print(f"⚠️ Context test failed: {e}")

def test_conversation(csm, llm):
    """Test 3: Full conversation loop."""
    print("\n" + "="*60)
    print("TEST 3: Conversation with LLM + CSM")
    print("="*60)
    
    # Simulated user inputs (or use real STT)
    user_inputs = [
        "I'm driving to work",
        "I didn't sleep well last night",
        "Maybe I should pull over",
    ]
    
    context_audio = None
    
    for user_text in user_inputs:
        print(f"\n👤 User: {user_text}")
        
        # Get LLM response
        response = llm.get_response(user_text)
        print(f"🤖 Assistant: {response}")
        
        # Generate speech with context
        audio_path = csm.generate_speech(response, context_audio=context_audio)
        
        if audio_path:
            csm.play_audio(audio_path)
            
            # Use this as context for next utterance
            try:
                context_audio, _ = torchaudio.load(str(audio_path))
            except:
                pass
            
            time.sleep(2)

def test_performance(csm):
    """Test 4: Performance benchmarks."""
    print("\n" + "="*60)
    print("TEST 4: Performance Benchmark")
    print("="*60)
    
    test_sentences = [
        "Short test.",
        "This is a medium length sentence for testing.",
        "This is a longer sentence that will help us understand how the model performs with more text and whether it maintains quality.",
    ]
    
    for i, text in enumerate(test_sentences, 1):
        print(f"\n📊 Test {i}: {len(text)} chars")
        start = time.time()
        audio_path = csm.generate_speech(text)
        gen_time = time.time() - start
        
        print(f"   Generation: {gen_time:.2f}s")
        print(f"   Speed: {len(text)/gen_time:.1f} chars/sec")
        
        if audio_path:
            # Get audio duration
            try:
                info = torchaudio.info(str(audio_path))
                duration = info.num_frames / info.sample_rate
                print(f"   Audio duration: {duration:.2f}s")
                print(f"   Real-time factor: {gen_time/duration:.2f}x")
            except:
                pass

# =========================
# MAIN
# =========================
def main():
    print("\n" + "="*60)
    print("🎤 Sesame CSM TTS Test Suite")
    print("   Testing on Raspberry Pi 4B")
    print("="*60)
    
    # Check PyTorch
    print(f"\n✓ PyTorch version: {torch.__version__}")
    print(f"✓ Device: {Config.DEVICE}")
    
    # Initialize engines
    csm = CSMEngine()
    if not csm.model:
        print("\n❌ CSM failed to load. Cannot continue.")
        return
    
    llm = SimpleLLM()
    
    # Run tests
    print("\n" + "="*60)
    print("Select test to run:")
    print("  1 - Basic TTS")
    print("  2 - Contextual speech")
    print("  3 - Full conversation")
    print("  4 - Performance benchmark")
    print("  5 - Run all tests")
    print("="*60)
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            test_basic_tts(csm)
        elif choice == "2":
            test_contextual_speech(csm)
        elif choice == "3":
            test_conversation(csm, llm)
        elif choice == "4":
            test_performance(csm)
        elif choice == "5":
            test_basic_tts(csm)
            test_contextual_speech(csm)
            test_conversation(csm, llm)
            test_performance(csm)
        else:
            print("Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted")
    
    print("\n✓ Tests complete!")
    print(f"📁 Audio files saved to: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
