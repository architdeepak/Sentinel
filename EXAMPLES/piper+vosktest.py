#!/usr/bin/env python3
"""
Complete TTS and STT Test Script for Raspberry Pi
Tests Piper TTS and Vosk STT
"""

import os
import sys
import wave
import json
import subprocess
import time

# ============================================
# CONFIGURATION - Change paths if needed
# ============================================
PIPER_VOICE_PATH = os.path.expanduser("~/Sentinel/pipervoices/en_US-amy-medium.onnx")
VOSK_MODEL_PATH = os.path.expanduser("~/Sentinel/Sentinel/vosk-model-small-en-us-0.15")

def test_tts_piper():
    """Test Piper TTS."""
    print("\n" + "="*60)
    print("TESTING PIPER TTS")
    print("="*60)
    
    try:
        from piper import PiperVoice
        
        # Check voice file exists
        if not os.path.exists(PIPER_VOICE_PATH):
            print(f"❌ Voice file not found: {PIPER_VOICE_PATH}")
            print("   Run setup commands to download voice")
            return False
        
        print(f"✓ Voice file found: {PIPER_VOICE_PATH}")
        
        # Load voice
        print("Loading voice...")
        voice = PiperVoice.load(PIPER_VOICE_PATH)
        print("✓ Voice loaded")
        
        # Test phrases
        test_phrases = [
            "Hello, this is a test of text to speech.",
            "I notice you're feeling drowsy. Let me help you stay alert.",
            "Would you like to chat to help you stay awake?"
        ]
        
        for i, text in enumerate(test_phrases, 1):
            print(f"\n[Test {i}/{len(test_phrases)}]")
            print(f"Speaking: {text}")
            
            output_file = f"/tmp/piper_test_{i}.wav"
            
            # Generate speech
            start = time.time()
            with wave.open(output_file, "wb") as wav_file:
                voice.synthesize(text, wav_file)
            gen_time = time.time() - start
            
            # Play speech
            subprocess.run(["aplay", "-q", output_file], check=False)
            
            # Cleanup
            os.remove(output_file)
            
            print(f"✓ Generation time: {gen_time:.2f}s")
        
        print("\n" + "="*60)
        print("✓ PIPER TTS: ALL TESTS PASSED")
        print("="*60)
        return True
        
    except ImportError:
        print("❌ Piper not installed. Run: pip install piper-tts")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_stt_vosk():
    """Test Vosk STT."""
    print("\n" + "="*60)
    print("TESTING VOSK STT")
    print("="*60)
    
    try:
        from vosk import Model, KaldiRecognizer
        import pyaudio
        
        # Check model exists
        if not os.path.exists(VOSK_MODEL_PATH):
            print(f"❌ Model not found: {VOSK_MODEL_PATH}")
            print("   Run setup commands to download model")
            return False
        
        print(f"✓ Model found: {VOSK_MODEL_PATH}")
        
        # Load model
        print("Loading model...")
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)
        print("✓ Model loaded")
        
        # Setup microphone
        print("Setting up microphone...")
        mic = pyaudio.PyAudio()
        
        # List available microphones
        print("\nAvailable microphones:")
        for i in range(mic.get_device_count()):
            info = mic.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']}")
        
        # Open stream
        stream = mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8192
        )
        stream.start_stream()
        print("✓ Microphone ready")
        
        # Test 1: Simple recognition
        print("\n[Test 1] Basic Recognition")
        print("🎤 Say something (you have 5 seconds)...")
        print("   Try saying: 'I am feeling drowsy'")
        
        start = time.time()
        recognized_text = ""
        
        for i in range(0, int(16000 / 8192 * 5)):
            data = stream.read(8192, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get('text', '')
                if text:
                    recognized_text = text
                    print(f"✓ You said: '{text}'")
                    break
        
        elapsed = time.time() - start
        
        if recognized_text:
            print(f"✓ Recognition time: {elapsed:.2f}s")
            print("✓ BASIC TEST PASSED")
        else:
            print("⚠️  No speech detected (try speaking louder)")
        
        # Test 2: Continuous listening
        print("\n[Test 2] Continuous Listening")
        print("🎤 Say 3 different things (press Ctrl+C when done)...")
        print("   Examples: 'hello', 'I am tired', 'help me stay awake'")
        
        count = 0
        try:
            while count < 3:
                data = stream.read(8192, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '')
                    if text:
                        count += 1
                        print(f"  [{count}] You said: '{text}'")
        except KeyboardInterrupt:
            print("\n  Stopped by user")
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        mic.terminate()
        
        print("\n" + "="*60)
        print("✓ VOSK STT: ALL TESTS PASSED")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run: pip install vosk pyaudio")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stt_google():
    """Test Google STT (requires internet)."""
    print("\n" + "="*60)
    print("TESTING GOOGLE STT (requires internet)")
    print("="*60)
    
    try:
        import speech_recognition as sr
        
        r = sr.Recognizer()
        
        with sr.Microphone() as source:
            print("🎤 Adjusting for ambient noise (wait 2 seconds)...")
            r.adjust_for_ambient_noise(source, duration=2)
            
            print("🎤 Say something (you have 5 seconds)...")
            print("   Try saying: 'I need help staying awake'")
            
            try:
                audio = r.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing...")
                
                start = time.time()
                text = r.recognize_google(audio)
                elapsed = time.time() - start
                
                print(f"✓ You said: '{text}'")
                print(f"✓ Recognition time: {elapsed:.2f}s")
                print("\n" + "="*60)
                print("✓ GOOGLE STT: TEST PASSED")
                print("="*60)
                return True
                
            except sr.WaitTimeoutError:
                print("⚠️  No speech detected (timeout)")
                return False
            except sr.UnknownValueError:
                print("⚠️  Could not understand audio")
                return False
            except sr.RequestError as e:
                print(f"❌ API error: {e} (check internet connection)")
                return False
        
    except ImportError:
        print("❌ SpeechRecognition not installed. Run: pip install SpeechRecognition")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_full_loop():
    """Test complete TTS → STT loop."""
    print("\n" + "="*60)
    print("TESTING COMPLETE TTS → STT LOOP")
    print("="*60)
    print("This simulates the drowsiness assistant workflow")
    
    try:
        from piper import PiperVoice
        from vosk import Model, KaldiRecognizer
        import pyaudio
        
        # Load TTS
        print("\n1. Loading TTS voice...")
        voice = PiperVoice.load(PIPER_VOICE_PATH)
        print("✓ TTS ready")
        
        # Load STT
        print("2. Loading STT model...")
        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)
        print("✓ STT ready")
        
        # Setup mic
        print("3. Setting up microphone...")
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                         input=True, frames_per_buffer=8192)
        stream.start_stream()
        print("✓ Microphone ready")
        
        # Test loop
        print("\n" + "─"*60)
        print("SIMULATED CONVERSATION")
        print("─"*60)
        
        # Assistant speaks
        assistant_msg = "I notice you're feeling drowsy. How are you doing?"
        print(f"\n🤖 Assistant: {assistant_msg}")
        
        with wave.open("/tmp/loop_test.wav", "wb") as wav_file:
            voice.synthesize(assistant_msg, wav_file)
        subprocess.run(["aplay", "-q", "/tmp/loop_test.wav"])
        os.remove("/tmp/loop_test.wav")
        
        # Listen for response
        print("🎤 Your turn to respond (5 seconds)...")
        
        for i in range(0, int(16000 / 8192 * 5)):
            data = stream.read(8192, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get('text', '')
                if text:
                    print(f"👤 You: {text}")
                    break
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        mic.terminate()
        
        print("\n" + "="*60)
        print("✓ FULL LOOP TEST COMPLETE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# TTS & STT COMPLETE TEST SUITE")
    print("#"*60)
    
    results = {}
    
    # Test TTS
    results['Piper TTS'] = test_tts_piper()
    
    time.sleep(1)
    
    # Test STT
    results['Vosk STT'] = test_stt_vosk()
    
    time.sleep(1)
    
    # Test Google STT (optional)
    print("\nTest Google STT? (requires internet) [y/N]: ", end='')
    if input().lower() == 'y':
        results['Google STT'] = test_stt_google()
    
    time.sleep(1)
    
    # Test full loop
    print("\nTest full TTS→STT loop? [y/N]: ", end='')
    if input().lower() == 'y':
        results['Full Loop'] = test_full_loop()
    
    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test:20s} {status}")
    
    print("="*60)
    
    if all(results.values()):
        print("\n🎉 ALL SYSTEMS GO! Ready for drowsiness detection!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")

if __name__ == "__main__":
    main()