#!/usr/bin/env python3
"""
Simple STT Test - Just speak and see what it recognizes
"""

import json
import time
import numpy as np
from pathlib import Path
from vosk import Model, KaldiRecognizer
import pyaudio

# =========================
# CONFIGURATION
# =========================
class Config:
    VOSK_MODEL_PATH = Path.home() / "Sentinel" / "Sentinel" / "vosk-model-small-en-us-0.15"
    VOSK_SAMPLE_RATE = 16000
    VOSK_BUFFER_SIZE = 8192

# =========================
# SPEECH-TO-TEXT ENGINE
# =========================
class STTEngine:
    """Vosk-based speech recognition with proper timeout and silence detection."""
    
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
            if not Config.VOSK_MODEL_PATH.exists():
                print(f"❌ Vosk model not found: {Config.VOSK_MODEL_PATH}")
                print("   Download model to continue")
                return
            
            print(f"✓ Vosk model found: {Config.VOSK_MODEL_PATH}")
            
            # Load model
            print("📦 Loading Vosk model...")
            self.model = Model(str(Config.VOSK_MODEL_PATH))
            self.recognizer = KaldiRecognizer(self.model, Config.VOSK_SAMPLE_RATE)
            print("✓ Vosk model loaded")
            
            # Setup microphone
            print("🎤 Setting up microphone...")
            self.mic = pyaudio.PyAudio()
            
            # List available microphones
            print("\nAvailable microphones:")
            for i in range(self.mic.get_device_count()):
                info = self.mic.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  {i}: {info['name']}")
            
            # Open stream
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
            print(f"⚠️ STT initialization failed: {e}")
            self.model = None
            self.cleanup()
    
    def _clear_audio_buffer(self):
        """Clear any buffered audio from the stream."""
        if not self.stream:
            return
        
        # Read and discard buffered audio
        try:
            for _ in range(5):  # Clear ~5 buffers worth
                self.stream.read(Config.VOSK_BUFFER_SIZE, exception_on_overflow=False)
        except:
            pass
    
    def _detect_speech_energy(self, audio_data):
        """Check if audio contains actual speech (simple energy threshold)."""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_array**2))
        # Threshold (adjust if needed - higher = more selective)
        return energy > 300  # Typical speech is 500-3000, background noise is 50-200
    
    def listen(self, timeout=15, wait_for_speech=True):
        """
        Listen for speech with proper timeout and silence detection.
        
        Args:
            timeout: Maximum time to wait for speech (seconds)
            wait_for_speech: If True, waits for actual speech before accepting input
        
        Returns:
            Recognized text or None if timeout/no speech
        """
        if not self.model:
            print("⚠️ STT not available")
            return None

        print("\n🎤 Listening... (start speaking)")
        
        # Clear any buffered audio
        self._clear_audio_buffer()
        
        # Reset recognizer for fresh start
        self.recognizer.Reset()
        
        start_time = time.time()
        speech_detected = False
        silence_start = None
        collected_text = []
        
        # Calculate number of iterations for timeout
        iterations_per_second = Config.VOSK_SAMPLE_RATE / Config.VOSK_BUFFER_SIZE
        max_iterations = int(timeout * iterations_per_second)
        
        try:
            for iteration in range(max_iterations):
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(f"⏱️  Timeout ({timeout}s) - no speech detected")
                    break
                
                # Read audio
                data = self.stream.read(
                    Config.VOSK_BUFFER_SIZE,
                    exception_on_overflow=False
                )
                
                # Check if this audio contains speech energy
                has_energy = self._detect_speech_energy(data)
                
                # Wait for actual speech to start
                if wait_for_speech and not speech_detected:
                    if has_energy:
                        speech_detected = True
                        silence_start = None
                        print("🎤 Speech detected, listening...")
                    continue  # Keep waiting for speech
                
                # Once speech is detected, process normally
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    
                    if text:
                        collected_text.append(text)
                        silence_start = time.time()
                        print(f"   Heard: '{text}'")
                
                # Check for silence after speech (end of utterance)
                if silence_start and not has_energy:
                    silence_duration = time.time() - silence_start
                    if silence_duration > 1.5:  # 1.5 seconds of silence = done speaking
                        break
                elif has_energy:
                    silence_start = None  # Reset if we hear more speech
            
            # Get any final partial results
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                collected_text.append(final_text)
            
            # Combine all collected text
            full_text = " ".join(collected_text).strip()
            
            if full_text:
                print(f"✓ You said: '{full_text}'")
                return full_text
            else:
                print("⚠️  No speech recognized")
                return None
                
        except Exception as e:
            print(f"⚠️ STT error: {e}")
            return None
    
    def cleanup(self):
        """Cleanup audio resources."""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.mic:
                self.mic.terminate()
        except:
            pass

# =========================
# MAIN TEST
# =========================
def main():
    """Simple test - just speak and see what it recognizes."""
    
    print("\n" + "="*60)
    print("🎤 SIMPLE STT TEST")
    print("="*60 + "\n")
    
    # Initialize STT
    stt = STTEngine()
    
    if not stt.model:
        print("❌ Failed to initialize STT")
        return
    
    print("\n" + "="*60)
    print("Ready! Say something...")
    print("="*60)
    
    try:
        # Listen once
        result = stt.listen(timeout=20, wait_for_speech=True)
        
        print("\n" + "="*60)
        print("RESULT:")
        print("="*60)
        if result:
            print(f"✅ Recognized: '{result}'")
        else:
            print("❌ Nothing recognized")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        print("\n🔄 Cleaning up...")
        stt.cleanup()
        print("✓ Done!")

if __name__ == "__main__":
    main()
