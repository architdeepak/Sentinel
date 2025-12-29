#!/usr/bin/env python3
"""
Vosk STT Test Script - Debugging Version
Uses EXACT same logic as the main drowsiness detection system
Helps diagnose inconsistent recognition issues
"""

import json
import time
from pathlib import Path
from vosk import Model, KaldiRecognizer
import pyaudio

# =========================
# CONFIGURATION
# =========================
VOSK_MODEL_PATH = Path.home() / "Sentinel" / "Sentinel" / "vosk-model-small-en-us-0.15"
VOSK_SAMPLE_RATE = 16000
VOSK_BUFFER_SIZE = 8192
STT_TIMEOUT = 20  # Same as main program

# =========================
# VOSK STT ENGINE (Exact copy from main program)
# =========================
class STTEngine:
    """Vosk-based speech recognition using exact logic from main program."""
    
    def __init__(self):
        self.model = None
        self.recognizer = None
        self.mic = None
        self.stream = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Vosk model and microphone."""
        try:
            # Check model exists
            if not VOSK_MODEL_PATH.exists():
                print(f"❌ Vosk model not found: {VOSK_MODEL_PATH}")
                print("   Download model to continue")
                return
            
            print(f"✓ Vosk model found: {VOSK_MODEL_PATH}")
            
            # Load model
            print("📦 Loading Vosk model...")
            self.model = Model(str(VOSK_MODEL_PATH))
            self.recognizer = KaldiRecognizer(self.model, VOSK_SAMPLE_RATE)
            print("✓ Vosk model loaded")
            
            # Setup microphone
            print("🎤 Setting up microphone...")
            self.mic = pyaudio.PyAudio()
            
            # List available microphones
            print("\nAvailable microphones:")
            for i in range(self.mic.get_device_count()):
                info = self.mic.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    is_default = " (DEFAULT)" if i == self.mic.get_default_input_device_info()['index'] else ""
                    print(f"  {i}: {info['name']}{is_default}")
            
            # Open stream with exact settings from main program
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=VOSK_SAMPLE_RATE,
                input=True,
                frames_per_buffer=VOSK_BUFFER_SIZE
            )
            self.stream.start_stream()
            print("✓ Microphone ready")
            print("✓ STT Engine initialized successfully\n")
            
        except Exception as e:
            print(f"⚠️ STT initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.cleanup()
    
    def listen(self, timeout=None, debug=False):
        """
        Listen for speech and return transcribed text.
        Uses the exact logic from the main program.
        
        Args:
            timeout: Listening timeout in seconds (default: STT_TIMEOUT)
            debug: If True, show detailed debug information
        
        Returns:
            str: Transcribed text, or None if no speech detected
        """
        if not self.model:
            print("⚠️ STT not available")
            return None
        
        if timeout is None:
            timeout = STT_TIMEOUT
        
        print(f"\n🎤 Listening... (you have {timeout} seconds)")
        print("   Speak clearly into your microphone")
        
        start_time = time.time()
        recognized_text = ""
        
        try:
            # Calculate number of iterations based on timeout
            # Formula from main program: int(SAMPLE_RATE / BUFFER_SIZE * TIMEOUT)
            iterations = int(VOSK_SAMPLE_RATE / VOSK_BUFFER_SIZE * timeout)
            
            if debug:
                print(f"\n[DEBUG] Total iterations: {iterations}")
                print(f"[DEBUG] Each iteration = {VOSK_BUFFER_SIZE / VOSK_SAMPLE_RATE:.3f}s")
            
            partial_results_count = 0
            accept_count = 0
            
            for i in range(iterations):
                # Read audio data
                data = self.stream.read(VOSK_BUFFER_SIZE, exception_on_overflow=False)
                
                if debug and i % 10 == 0:  # Print every 10th iteration
                    elapsed = time.time() - start_time
                    print(f"[DEBUG] Iteration {i}/{iterations} | Elapsed: {elapsed:.1f}s | Data size: {len(data)}")
                
                # Process audio
                if self.recognizer.AcceptWaveform(data):
                    accept_count += 1
                    result = json.loads(self.recognizer.Result())
                    text = result.get('text', '')
                    
                    if debug:
                        print(f"[DEBUG] AcceptWaveform #{accept_count} | Text: '{text}'")
                    
                    if text:
                        recognized_text = text
                        elapsed = time.time() - start_time
                        print(f"✓ You said: '{text}'")
                        print(f"✓ Recognition time: {elapsed:.2f}s")
                        return recognized_text
                else:
                    # Check partial results for debugging
                    if debug:
                        partial = json.loads(self.recognizer.PartialResult())
                        partial_text = partial.get('partial', '')
                        if partial_text and i % 5 == 0:
                            print(f"[DEBUG] Partial result: '{partial_text}'")
                            partial_results_count += 1
            
            if debug:
                print(f"\n[DEBUG] Loop finished:")
                print(f"  - AcceptWaveform calls: {accept_count}")
                print(f"  - Partial results with text: {partial_results_count}")
            
            # Check final result if nothing detected in loop
            print("[DEBUG] Checking final result...")
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get('text', '')
            
            if debug:
                print(f"[DEBUG] Final result: '{final_text}'")
            
            if final_text:
                elapsed = time.time() - start_time
                print(f"✓ You said: '{final_text}'")
                print(f"✓ Recognition time: {elapsed:.2f}s")
                return final_text
            
            # No speech detected
            elapsed = time.time() - start_time
            print(f"⏱️ No speech detected after {elapsed:.1f}s")
            return None
            
        except Exception as e:
            print(f"⚠️ Listening error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Cleanup audio resources."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        except:
            pass
        
        try:
            if self.mic:
                self.mic.terminate()
        except:
            pass

# =========================
# TEST FUNCTIONS
# =========================
def test_basic_recognition(stt):
    """Test basic speech recognition."""
    print("\n" + "="*60)
    print("TEST 1: Basic Recognition")
    print("="*60)
    print("Try saying: 'Hello, I am feeling drowsy'")
    
    result = stt.listen(timeout=10, debug=False)
    
    if result:
        print("✅ TEST PASSED - Speech recognized")
    else:
        print("❌ TEST FAILED - No speech detected")
    
    return result is not None

def test_with_debug(stt):
    """Test with detailed debug output."""
    print("\n" + "="*60)
    print("TEST 2: Debug Mode (detailed output)")
    print("="*60)
    print("Try saying: 'I need help staying awake'")
    
    result = stt.listen(timeout=10, debug=True)
    
    if result:
        print("✅ TEST PASSED - Speech recognized")
    else:
        print("❌ TEST FAILED - No speech detected")
    
    return result is not None

def test_different_timeouts(stt):
    """Test with different timeout values."""
    print("\n" + "="*60)
    print("TEST 3: Different Timeouts")
    print("="*60)
    
    timeouts = [5, 10, 20]
    results = []
    
    for timeout in timeouts:
        print(f"\n--- Testing with {timeout}s timeout ---")
        print(f"Try saying: 'Testing {timeout} seconds'")
        
        result = stt.listen(timeout=timeout, debug=False)
        results.append(result is not None)
        
        if result:
            print(f"✅ {timeout}s timeout - SUCCESS")
        else:
            print(f"❌ {timeout}s timeout - FAILED")
        
        time.sleep(1)
    
    return any(results)

def test_continuous_listening(stt):
    """Test continuous listening (multiple phrases)."""
    print("\n" + "="*60)
    print("TEST 4: Continuous Listening (3 attempts)")
    print("="*60)
    
    attempts = 3
    successes = 0
    
    for i in range(attempts):
        print(f"\n--- Attempt {i+1}/{attempts} ---")
        print(f"Try saying: 'Attempt number {i+1}'")
        
        result = stt.listen(timeout=10, debug=False)
        
        if result:
            successes += 1
            print(f"✅ Attempt {i+1} - SUCCESS")
        else:
            print(f"❌ Attempt {i+1} - FAILED")
        
        time.sleep(0.5)
    
    print(f"\n--- Results: {successes}/{attempts} successful ---")
    return successes > 0

def test_audio_levels(stt):
    """Test audio input levels."""
    print("\n" + "="*60)
    print("TEST 5: Audio Level Test")
    print("="*60)
    print("Checking if microphone is receiving audio...")
    
    if not stt.stream:
        print("❌ No audio stream available")
        return False
    
    print("Recording 2 seconds of audio...")
    samples = []
    
    for i in range(int(2 * VOSK_SAMPLE_RATE / VOSK_BUFFER_SIZE)):
        data = stt.stream.read(VOSK_BUFFER_SIZE, exception_on_overflow=False)
        samples.extend(list(data))
    
    # Calculate simple metrics
    import struct
    audio_ints = struct.unpack(f'{len(samples)}b', bytes(samples))
    max_val = max(abs(x) for x in audio_ints)
    avg_val = sum(abs(x) for x in audio_ints) / len(audio_ints)
    
    print(f"\nAudio Levels:")
    print(f"  Max amplitude: {max_val}/127 ({max_val/127*100:.1f}%)")
    print(f"  Avg amplitude: {avg_val:.1f}/127 ({avg_val/127*100:.1f}%)")
    
    if max_val < 10:
        print("⚠️  WARNING: Audio levels very low - check microphone")
        print("   - Is the correct microphone selected?")
        print("   - Is the microphone muted?")
        print("   - Try speaking louder")
        return False
    elif max_val < 30:
        print("⚠️  Audio levels low but present - try speaking louder")
        return True
    else:
        print("✅ Audio levels good")
        return True

# =========================
# MAIN PROGRAM
# =========================
def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# VOSK STT DEBUG TEST SUITE")
    print("# Uses exact logic from main drowsiness detection system")
    print("#"*60)
    
    # Initialize STT
    print("\nInitializing STT Engine...")
    stt = STTEngine()
    
    if not stt.model:
        print("\n❌ STT initialization failed. Cannot continue.")
        return
    
    print("\n" + "="*60)
    print("READY TO TEST")
    print("="*60)
    print("\nWhat would you like to test?")
    print("  1. Basic recognition")
    print("  2. Debug mode (detailed output)")
    print("  3. Different timeouts")
    print("  4. Continuous listening")
    print("  5. Audio level check")
    print("  6. Run all tests")
    print("  7. Custom test (keep testing until you quit)")
    print("  q. Quit")
    
    while True:
        choice = input("\nEnter choice (1-7 or q): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '1':
            test_basic_recognition(stt)
        elif choice == '2':
            test_with_debug(stt)
        elif choice == '3':
            test_different_timeouts(stt)
        elif choice == '4':
            test_continuous_listening(stt)
        elif choice == '5':
            test_audio_levels(stt)
        elif choice == '6':
            # Run all tests
            results = {
                "Basic Recognition": test_basic_recognition(stt),
                "Debug Mode": test_with_debug(stt),
                "Different Timeouts": test_different_timeouts(stt),
                "Continuous Listening": test_continuous_listening(stt),
                "Audio Levels": test_audio_levels(stt)
            }
            
            print("\n" + "="*60)
            print("TEST SUMMARY")
            print("="*60)
            for test, passed in results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{test:25s} {status}")
            print("="*60)
        elif choice == '7':
            # Custom test mode
            print("\n" + "="*60)
            print("CUSTOM TEST MODE")
            print("="*60)
            print("Keep testing until you type 'done'")
            
            test_num = 1
            while True:
                print(f"\n--- Test #{test_num} ---")
                print("Say anything, or type 'done' to stop")
                print("Press Enter to start listening (or type 'done'):")
                
                cmd = input().strip().lower()
                if cmd == 'done':
                    break
                
                result = stt.listen(timeout=15, debug=False)
                
                if result:
                    print(f"✅ Test #{test_num} - SUCCESS")
                else:
                    print(f"❌ Test #{test_num} - FAILED")
                    print("\nTroubleshooting tips:")
                    print("  - Check if you're speaking into the right microphone")
                    print("  - Try speaking louder")
                    print("  - Reduce background noise")
                    print("  - Run Test 5 (audio levels) to check microphone")
                
                test_num += 1
        else:
            print("Invalid choice. Please enter 1-7 or q.")
    
    # Cleanup
    print("\n🔄 Cleaning up...")
    stt.cleanup()
    print("✓ Done")

if __name__ == "__main__":
    main()