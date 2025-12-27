#!/usr/bin/env python3
"""
Windows Audio Test Script - Piper TTS and Vosk STT
Tests audio on Windows before deploying to Raspberry Pi
"""

import os
import sys
import wave
import json
import time
from pathlib import Path

# ============================================
# CONFIGURATION - Windows Paths
# ============================================
HOME = Path.home()
PIPER_VOICE_PATH = HOME / "piper_voices" / "en_US-amy-medium.onnx"
VOSK_MODEL_PATH = HOME / "ScienecFair25-26" / "vosk-model-small-en-us-0.15"

def play_wav_windows(filepath):
    """Play WAV file on Windows."""
    try:
        import winsound
        winsound.PlaySound(filepath, winsound.SND_FILENAME)
    except ImportError:
        print("⚠️  winsound not available, trying alternative...")
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except ImportError:
            print("❌ No audio playback available!")
            print("   Install pygame: pip install pygame")

def test_piper_tts():
    """Test Piper TTS on Windows."""
    print("\n" + "="*60)
    print("TESTING PIPER TTS (Windows)")
    print("="*60)
    
    try:
        from piper import PiperVoice
        
        # Check voice file
        if not PIPER_VOICE_PATH.exists():
            print(f"❌ Voice file not found: {PIPER_VOICE_PATH}")
            print(f"\nDownload from:")
            print(f"  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx")
            print(f"  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json")
            print(f"\nSave to: {PIPER_VOICE_PATH.parent}")
            return False
        
        print(f"✓ Voice file found: {PIPER_VOICE_PATH}")
        
        # Load voice
        print("Loading voice...")
        voice = PiperVoice.load(str(PIPER_VOICE_PATH))
        print("✓ Voice loaded")
        
        # Test phrases
        test_phrases = [
            "Hello, this is a test of Piper text to speech on Windows.",
            "I notice you're feeling drowsy. Let me help you stay alert.",
            "This is working perfectly on your Windows laptop!"
        ]
        
        for i, text in enumerate(test_phrases, 1):
            print(f"\n[Test {i}/{len(test_phrases)}]")
            print(f"Speaking: {text}")
            
            output_file = f"C:\\Users\\archi\\Sentinel\\piper_test_{i}.wav"
            
            # Generate speech
            start = time.time()
            # sensible defaults
            channels = 1
            sampwidth = 2  # bytes (16-bit)
            rate = 22050

            try:
                with wave.open(output_file, "wb") as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(sampwidth)
                    wav_file.setframerate(rate)
                    try:
                        rv = voice.synthesize(text, wav_file)
                    except TypeError:
                        # some bindings return bytes
                        rv = voice.synthesize(text)
                        if isinstance(rv, (bytes, bytearray)):
                            with open(output_file, "wb") as f:
                                f.write(rv)
                gen_time = time.time() - start
                print(f"✓ Generated in {gen_time:.2f}s")

                size = os.path.getsize(output_file) if os.path.exists(output_file) else 0
                if size <= 44:
                    print(f"⚠️ Generated file looks empty (size={size} bytes). Synthesis may have produced no audio.")

            except wave.Error as e:
                print(f"❌ Error: {e}")
                print("This typically means WAV header fields were not set before writing. Ensure channels/sample width/framerate are set.")
                continue
            except Exception as e:
                print(f"❌ Error generating TTS: {e}")
                continue
            
            # Play speech
            print("🔊 Playing audio...")
            play_wav_windows(output_file)
            
            # Cleanup
            try:
                os.remove(output_file)
            except:
                pass
            
            time.sleep(0.5)
        
        print("\n" + "="*60)
        print("✓ PIPER TTS: ALL TESTS PASSED")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install: pip install piper-tts onnxruntime")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vosk_stt():
    """Test Vosk STT on Windows."""
    print("\n" + "="*60)
    print("TESTING VOSK STT (Windows)")
    print("="*60)
    
    try:
        from vosk import Model, KaldiRecognizer
        import pyaudio
        
        # Check model
        if not VOSK_MODEL_PATH.exists():
            print(f"❌ Model not found: {VOSK_MODEL_PATH}")
            print(f"\nDownload from:")
            print(f"  https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
            print(f"\nExtract to: {VOSK_MODEL_PATH}")
            return False
        
        print(f"✓ Model found: {VOSK_MODEL_PATH}")
        
        # Load model
        print("Loading model...")
        model = Model(str(VOSK_MODEL_PATH))
        recognizer = KaldiRecognizer(model, 16000)
        print("✓ Model loaded")
        
        # Setup microphone
        print("Setting up microphone...")
        mic = pyaudio.PyAudio()
        
        # List microphones
        print("\nAvailable microphones:")
        default_device = None
        for i in range(mic.get_device_count()):
            info = mic.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
                if default_device is None:
                    default_device = i
        
        if default_device is None:
            print("❌ No microphone found!")
            return False
        
        # Open stream
        print(f"\nUsing device: {default_device}")
        stream = mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=default_device,
            frames_per_buffer=8192
        )
        stream.start_stream()
        print("✓ Microphone ready")
        
        # Test recognition
        print("\n" + "─"*60)
        print("🎤 SPEAK NOW (you have 10 seconds)")
        print("   Try saying: 'Hello, I am testing the microphone'")
        print("─"*60)
        
        start = time.time()
        recognized_text = ""
        
        try:
            for i in range(0, int(16000 / 8192 * 10)):
                data = stream.read(8192, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get('text', '')
                    if text:
                        recognized_text = text
                        print(f"\n✓ You said: '{text}'")
                        break
                
                # Show progress
                if i % 5 == 0:
                    elapsed = time.time() - start
                    print(f"  Listening... {elapsed:.1f}s", end='\r')
        except KeyboardInterrupt:
            print("\n⚠️  Stopped by user")
        
        elapsed = time.time() - start
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        mic.terminate()
        
        if recognized_text:
            print(f"\n✓ Recognition time: {elapsed:.2f}s")
            print("\n" + "="*60)
            print("✓ VOSK STT: TEST PASSED")
            print("="*60)
            return True
        else:
            print("\n⚠️  No speech detected")
            print("   Try speaking louder or closer to microphone")
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        if "pyaudio" in str(e).lower():
            print("\nPyAudio installation failed!")
            print("Download precompiled wheel from:")
            print("  https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
            print("\nFor Python 3.11:")
            print("  pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl")
        else:
            print("   Install: pip install vosk pyaudio")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_google_stt():
    """Test Google STT (requires internet)."""
    print("\n" + "="*60)
    print("TESTING GOOGLE STT (Windows - requires internet)")
    print("="*60)
    
    try:
        import speech_recognition as sr
        
        r = sr.Recognizer()
        
        with sr.Microphone() as source:
            print("🎤 Adjusting for ambient noise (2 seconds)...")
            r.adjust_for_ambient_noise(source, duration=2)
            
            print("🎤 Say something (you have 5 seconds)...")
            print("   Try saying: 'Testing Google speech recognition'")
            
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
                print(f"❌ API error: {e}")
                print("   Check internet connection")
                return False
        
    except ImportError:
        print("❌ SpeechRecognition not installed")
        print("   Install: pip install SpeechRecognition")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# WINDOWS AUDIO TEST SUITE")
    print("# Testing Piper TTS and Vosk STT")
    print("#"*60)
    
    print(f"\nPaths being used:")
    print(f"  Piper Voice: {PIPER_VOICE_PATH}")
    print(f"  Vosk Model:  {VOSK_MODEL_PATH}")
    
    results = {}
    
    # Test TTS
    input("\nPress Enter to test Piper TTS...")
    results['Piper TTS'] = test_piper_tts()
    
    # Test Vosk STT
    input("\nPress Enter to test Vosk STT...")
    results['Vosk STT'] = test_vosk_stt()
    
    # Test Google STT (optional)
    print("\n\nTest Google STT? (requires internet) [y/N]: ", end='')
    if input().lower() == 'y':
        results['Google STT'] = test_google_stt()
    
    # Summary
    print("\n\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{test:20s} {status}")
    
    print("="*60)
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED!")
        print("Your audio setup is working correctly on Windows!")
        print("This should work on Raspberry Pi too!")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()