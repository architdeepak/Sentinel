#!/usr/bin/env python3
"""
Driver Drowsiness Detection System V2
Optimized for Raspberry Pi 4B

Changes from V1:
- llama.cpp with Granite model (replaced ollama)
- espeak-ng for TTS (replaced pyttsx3)
- Vosk for STT with speech_recognition fallback
- Performance optimizations for RPi
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import subprocess
import threading
import queue
import json
from pathlib import Path
from llama_cpp import Llama

# =========================
# CONFIGURATION
# =========================
class Config:
    # Model paths
    LLM_MODEL_PATH = Path.home() / "Sentinel" / "modls" / "granite-3.0-1b-a400m-instruct.Q4_K_M.gguf"
    VOSK_MODEL_PATH = Path.home() / "Sentinel" / "vosk-model-small-en-us-0.15"
    
    # LLM settings (no hard limits - use prompting instead)
    LLM_THREADS = 3
    LLM_CONTEXT = 2048
    LLM_MAX_TOKENS = 150  # Reasonable max, but prompt guides length
    
    # Audio settings
    ESPEAK_SPEED = 165
    ESPEAK_VOICE = "en-us"
    VOSK_SAMPLE_RATE = 16000
    VOSK_BUFFER_SIZE = 4096  # Smaller for better responsiveness
    
    # Camera settings (RPi optimization)
    CAMERA_WIDTH = 480
    CAMERA_HEIGHT = 360
    CAMERA_FPS = 20
    
    # Detection thresholds
    EAR_THRESH = 0.25
    MAR_THRESH = 0.6
    MICROSLEEP_TIME = 1.5
    SLOW_BLINK_TIME = 0.4
    HEAD_DOWN_THRESH = 0.12
    HEAD_DOWN_TIME = 1.2
    HEAD_ROLL_THRESH = 15
    ROLL_TIME = 1.2
    WINDOW_TIME = 10
    DROWSY_THRESHOLD = 0.47
    DROWSY_TRIGGER_COUNT = 10

# =========================
# LANDMARKS CONSTANTS
# =========================
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
MOUTH_LANDMARKS = [13, 14, 61, 291]

# =========================
# TEXT-TO-SPEECH (espeak-ng)
# =========================
class TTSEngine:
    """Clean TTS using espeak-ng with queue-based playback."""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        self.is_speaking = False
    
    def _audio_worker(self):
        """Plays sentences one at a time - NO overlap."""
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
        """Add text to speaking queue."""
        if text and text.strip():
            self.audio_queue.put(text.strip())
    
    def wait_until_done(self):
        """Wait for all speech to finish."""
        self.audio_queue.join()
    
    def shutdown(self):
        """Shutdown the TTS engine."""
        self.audio_queue.put(None)

# =========================
# SPEECH-TO-TEXT (Hybrid)
# =========================
class STTEngine:
    """Hybrid STT: Tries Vosk first, falls back to speech_recognition."""
    
    def __init__(self):
        self.mode = None
        self.vosk_model = None
        self.vosk_recognizer = None
        self.mic = None
        self.stream = None
        
        # Try Vosk first
        if self._initialize_vosk():
            self.mode = "vosk"
            print("✓ Using Vosk for STT")
        else:
            # Fall back to speech_recognition
            if self._initialize_speech_recognition():
                self.mode = "speech_recognition"
                print("✓ Using speech_recognition library for STT")
            else:
                print("⚠️ STT not available")
                self.mode = None
    
    def _initialize_vosk(self):
        """Try to initialize Vosk."""
        try:
            from vosk import Model, KaldiRecognizer
            import pyaudio
            
            if not Config.VOSK_MODEL_PATH.exists():
                print(f"⚠️ Vosk model not found at {Config.VOSK_MODEL_PATH}")
                return False
            
            print("📦 Loading Vosk model...")
            self.vosk_model = Model(str(Config.VOSK_MODEL_PATH))
            self.vosk_recognizer = KaldiRecognizer(self.vosk_model, Config.VOSK_SAMPLE_RATE)
            self.vosk_recognizer.SetWords(True)  # Enable word-level timestamps
            
            print("🎤 Initializing microphone for Vosk...")
            self.mic = pyaudio.PyAudio()
            
            # Find best input device
            default_device = self.mic.get_default_input_device_info()
            
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=Config.VOSK_SAMPLE_RATE,
                input=True,
                input_device_index=default_device['index'],
                frames_per_buffer=Config.VOSK_BUFFER_SIZE
            )
            self.stream.start_stream()
            
            # Test the stream
            test_data = self.stream.read(Config.VOSK_BUFFER_SIZE, exception_on_overflow=False)
            if len(test_data) == 0:
                raise Exception("No audio data from microphone")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Vosk initialization failed: {e}")
            self._cleanup_vosk()
            return False
    
    def _initialize_speech_recognition(self):
        """Try to initialize speech_recognition library."""
        try:
            import speech_recognition as sr
            self.sr = sr
            self.recognizer = sr.Recognizer()
            
            # Test microphone
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            return True
            
        except Exception as e:
            print(f"⚠️ speech_recognition initialization failed: {e}")
            return False
    
    def listen(self, timeout=10):
        """Listen for speech and return transcribed text."""
        if self.mode == "vosk":
            return self._listen_vosk(timeout)
        elif self.mode == "speech_recognition":
            return self._listen_sr(timeout)
        else:
            print("⚠️ No STT engine available")
            return None
    
    def _listen_vosk(self, timeout):
        """Listen using Vosk."""
        print("\n🎤 Listening (Vosk)... speak now")
        start_time = time.time()
        
        # Clear any old data in the stream
        try:
            while self.stream.get_read_available() > 0:
                self.stream.read(self.stream.get_read_available(), exception_on_overflow=False)
        except:
            pass
        
        silence_threshold = 1.5  # seconds of silence before giving up
        last_speech_time = time.time()
        has_spoken = False
        accumulated_text = []
        
        try:
            while time.time() - start_time < timeout:
                try:
                    data = self.stream.read(Config.VOSK_BUFFER_SIZE, exception_on_overflow=False)
                except Exception as e:
                    print(f"⚠️ Stream read error: {e}")
                    time.sleep(0.1)
                    continue
                
                if self.vosk_recognizer.AcceptWaveform(data):
                    result = json.loads(self.vosk_recognizer.Result())
                    text = result.get('text', '').strip()
                    
                    if text:
                        accumulated_text.append(text)
                        last_speech_time = time.time()
                        has_spoken = True
                        print(f"  Recognized: '{text}'")
                else:
                    # Check partial results
                    partial = json.loads(self.vosk_recognizer.PartialResult())
                    partial_text = partial.get('partial', '')
                    
                    if partial_text:
                        last_speech_time = time.time()
                        has_spoken = True
                
                # If we've heard speech and then silence, finish
                if has_spoken and (time.time() - last_speech_time > silence_threshold):
                    break
            
            # Get final result
            final_result = json.loads(self.vosk_recognizer.FinalResult())
            final_text = final_result.get('text', '').strip()
            if final_text:
                accumulated_text.append(final_text)
            
            full_text = ' '.join(accumulated_text).strip()
            
            if full_text:
                print(f"✓ Full transcription: '{full_text}'")
                return full_text
            else:
                print("⏱️ No speech detected (Vosk)")
                return None
                
        except Exception as e:
            print(f"⚠️ Vosk listening error: {e}")
            return None
    
    def _listen_sr(self, timeout):
        """Listen using speech_recognition library."""
        print("\n🎤 Listening (Google Speech Recognition)... speak now")
        
        try:
            with self.sr.Microphone() as source:
                # Quick ambient noise adjustment
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=15
                )
            
            print("🔄 Processing speech...")
            
            # Try Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            print(f"✓ You said: '{text}'")
            return text
            
        except self.sr.WaitTimeoutError:
            print("⏱️ No speech detected (timeout)")
            return None
        except self.sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except self.sr.RequestError as e:
            print(f"⚠️ Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Listening error: {e}")
            return None
    
    def _cleanup_vosk(self):
        """Cleanup Vosk resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if self.mic:
            try:
                self.mic.terminate()
            except:
                pass
    
    def cleanup(self):
        """Cleanup audio resources."""
        if self.mode == "vosk":
            self._cleanup_vosk()

# =========================
# LLM ASSISTANT (llama.cpp)
# =========================
class LLMAssistant:
    """Streaming LLM assistant with espeak-ng TTS."""
    
    def __init__(self, tts_engine):
        self.tts = tts_engine
        self.llm = None
        self.messages = []
        self._initialize()
    
    def _initialize(self):
        """Initialize LLM model."""
        try:
            print("🧠 Loading LLM model...")
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
            print("✓ LLM ready")
            
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}")
            self.llm = None
    
    def start_conversation(self):
        """Start a new conversation with system prompt."""
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are an in-car voice assistant designed to keep a driver awake and alert.\n\n"
                    "YOUR SPEAKING STYLE:\n"
                    "- Keep responses SHORT (1-3 sentences) unless the driver asks for details\n"
                    "- Be calm, supportive, and conversational\n"
                    "- Ask ONE simple question at a time\n"
                    "- Use natural, friendly language\n"
                    "- Never be alarmist or stressful\n\n"
                    "YOUR STRATEGIES:\n"
                    "- Engage the driver in light conversation\n"
                    "- Suggest gentle activities: rolling down window, adjusting temperature, light stretches\n"
                    "- Ask about their destination, plans, interests\n"
                    "- Recommend taking a break if very drowsy\n"
                    "- Keep them mentally active with simple questions\n\n"
                    "Remember: Short and sweet is better. Only elaborate if asked."
                )
            },
            {
                "role": "user",
                "content": "I am feeling drowsy while driving"
            }
        ]
    
    def get_response_streaming(self, user_message=None):
        """Get streaming response from LLM with real-time TTS."""
        if not self.llm:
            return "Sorry, the assistant is not available."
        
        if user_message:
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
            
            # Speak complete sentences immediately
            if any(p in buffer for p in [".", "!", "?"]):
                sentence = buffer.strip()
                buffer = ""
                if sentence:
                    self.tts.speak(sentence)
        
        # Speak remaining buffer
        if buffer.strip():
            self.tts.speak(buffer.strip())
        
        print()  # New line
        
        # Add to conversation history
        self.messages.append({"role": "assistant", "content": full_response})
        
        return full_response

# =========================
# HELPER FUNCTIONS
# =========================
def euclidean(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, idx):
    """Calculate Eye Aspect Ratio (EAR)."""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))

def mouth_aspect_ratio(landmarks):
    """Calculate Mouth Aspect Ratio (MAR)."""
    return euclidean(landmarks[13], landmarks[14]) / euclidean(landmarks[61], landmarks[291])

# =========================
# INITIALIZATION
# =========================
def initialize_mediapipe():
    """Initialize MediaPipe face mesh (optimized for RPi)."""
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return face_mesh

def initialize_camera():
    """Initialize camera with RPi-optimized settings."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def initialize_state_variables():
    """Initialize all state tracking variables."""
    return {
        'ear_window': deque(),
        'pitch_window': deque(),
        'closed_window': deque(),
        'blink_times': deque(),
        'blink_durations': deque(),
        'yawn_times': deque(),
        'eye_closed_start': None,
        'blink_start': None,
        'yawn_start': None,
        'head_down_start': None,
        'head_roll_start': None,
        'llm_triggered': False,
        'drowsy_count': 0,
        'window_start_time': time.time()
    }

# =========================
# DETECTION FUNCTIONS
# =========================
def process_eye_metrics(landmarks, state, now):
    """Process eye-related metrics and detect microsleep."""
    ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
           eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2

    state['ear_window'].append((now, ear))
    state['closed_window'].append((now, ear < Config.EAR_THRESH))

    if ear < Config.EAR_THRESH:
        if state['eye_closed_start'] is None:
            state['eye_closed_start'] = now
            state['blink_start'] = now
    else:
        if state['eye_closed_start']:
            duration = now - state['blink_start']
            state['blink_times'].append(now)
            state['blink_durations'].append(duration)
            state['eye_closed_start'] = None

    microsleep = (state['eye_closed_start'] is not None and 
                  (now - state['eye_closed_start'] >= Config.MICROSLEEP_TIME))
    return ear, microsleep

def process_mouth_metrics(landmarks, state, now):
    """Process mouth metrics and detect yawning."""
    mar = mouth_aspect_ratio(landmarks)
    if mar > Config.MAR_THRESH:
        if state['yawn_start'] is None:
            state['yawn_start'] = now
    else:
        if state['yawn_start'] and now - state['yawn_start'] > 1.0:
            state['yawn_times'].append(now)
        state['yawn_start'] = None
    return mar

def process_head_pitch(landmarks, state, now):
    """Process head pitch (downward tilt) detection."""
    nose_y = landmarks[NOSE_TIP][1]
    eye_mid_y = (landmarks[LEFT_EYE_CORNER][1] + landmarks[RIGHT_EYE_CORNER][1]) / 2
    face_height = abs(landmarks[FOREHEAD][1] - landmarks[CHIN][1])

    pitch = (nose_y - eye_mid_y) / face_height
    state['pitch_window'].append((now, pitch))

    head_down = False
    if pitch > Config.HEAD_DOWN_THRESH:
        if state['head_down_start'] is None:
            state['head_down_start'] = now
        elif now - state['head_down_start'] > Config.HEAD_DOWN_TIME:
            head_down = True
    else:
        state['head_down_start'] = None

    return head_down

def process_head_roll(landmarks, state, now):
    """Process head roll (tilt) detection."""
    dx = landmarks[RIGHT_EYE_CORNER][0] - landmarks[LEFT_EYE_CORNER][0]
    dy = landmarks[RIGHT_EYE_CORNER][1] - landmarks[LEFT_EYE_CORNER][1]
    roll = abs(np.degrees(np.arctan2(dy, dx)))

    head_roll = False
    if roll > Config.HEAD_ROLL_THRESH:
        if state['head_roll_start'] is None:
            state['head_roll_start'] = now
        elif now - state['head_roll_start'] > Config.ROLL_TIME:
            head_roll = True
    else:
        state['head_roll_start'] = None

    return head_roll

def cleanup_windows(state, now):
    """Remove old entries from time windows."""
    while state['ear_window'] and now - state['ear_window'][0][0] > Config.WINDOW_TIME:
        state['ear_window'].popleft()
    while state['pitch_window'] and now - state['pitch_window'][0][0] > Config.WINDOW_TIME:
        state['pitch_window'].popleft()
    while state['closed_window'] and now - state['closed_window'][0][0] > Config.WINDOW_TIME:
        state['closed_window'].popleft()
    while state['blink_times'] and now - state['blink_times'][0] > Config.WINDOW_TIME:
        state['blink_times'].popleft()
    while state['blink_durations'] and len(state['blink_durations']) > len(state['blink_times']):
        state['blink_durations'].popleft()
    while state['yawn_times'] and now - state['yawn_times'][0] > Config.WINDOW_TIME:
        state['yawn_times'].popleft()

def calculate_metrics(state, microsleep, head_down, head_roll):
    """Calculate drowsiness metrics and score."""
    perclos = (sum(v for _, v in state['closed_window']) / len(state['closed_window']) 
               if state['closed_window'] else 0)
    blink_rate = len(state['blink_times'])
    slow_blinks = sum(d > Config.SLOW_BLINK_TIME for d in state['blink_durations'])
    ear_std = np.std([v for _, v in state['ear_window']]) if state['ear_window'] else 0
    pitch_var = np.var([v for _, v in state['pitch_window']]) if state['pitch_window'] else 0

    drowsy_score = min(1.0, (
        0.20 * perclos +
        0.15 * int(microsleep) +
        0.15 * min(slow_blinks / 5, 1.0) +
        0.10 * min(ear_std / 0.12, 1.0) +
        0.10 * min(pitch_var / 0.015, 1.0) +
        0.20 * int(head_down) +
        0.10 * int(head_roll)
    ))

    return {
        'perclos': perclos,
        'blink_rate': blink_rate,
        'slow_blinks': slow_blinks,
        'ear_std': ear_std,
        'pitch_var': pitch_var,
        'drowsy_score': drowsy_score
    }

def draw_overlay(frame, metrics, ear, mar, microsleep, head_down, head_roll, state, drowsy_state):
    """Draw metrics overlay on the frame."""
    y = 25
    font_scale = 0.5
    thickness = 1
    
    EYE_COLOR = (255, 0, 0)
    MOUTH_COLOR = (0, 0, 255)
    HEAD_COLOR = (0, 255, 0)

    texts = [
        (f"PERCLOS: {metrics['perclos']:.2f}", EYE_COLOR),
        (f"Blinks: {metrics['blink_rate']}", EYE_COLOR),
        (f"EAR: {ear:.3f}", EYE_COLOR),
        (f"MAR: {mar:.3f}", MOUTH_COLOR),
        (f"Yawns: {len(state['yawn_times'])}", MOUTH_COLOR),
        (f"Microsleep: {microsleep}", EYE_COLOR),
        (f"Head Down: {head_down}", HEAD_COLOR),
    ]

    drowsy_color = (0, 255, 0) if drowsy_state == "ALERT" else (0, 0, 255)
    texts.append((f"Score: {metrics['drowsy_score']:.2f} ({drowsy_state})", drowsy_color))

    for text, color in texts:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, thickness)
        y += 20

# =========================
# MAIN DETECTION LOOP
# =========================
def run_detection_loop(cap, face_mesh, state):
    """Run the main drowsiness detection loop."""
    frame_skip = 2
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        now = time.time()
        
        small_frame = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        h, w = Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH

        microsleep = False
        head_down = False
        head_roll = False
        ear = 0
        mar = 0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]

            ear, microsleep = process_eye_metrics(landmarks, state, now)
            mar = process_mouth_metrics(landmarks, state, now)
            head_down = process_head_pitch(landmarks, state, now)
            head_roll = process_head_roll(landmarks, state, now)

        cleanup_windows(state, now)
        metrics = calculate_metrics(state, microsleep, head_down, head_roll)

        drowsy_state = "ALERT"
        if metrics['drowsy_score'] > Config.DROWSY_THRESHOLD:
            drowsy_state = "DROWSY"
            state['drowsy_count'] += 1

        draw_overlay(frame, metrics, ear, mar, microsleep, head_down, 
                     head_roll, state, drowsy_state)

        cv2.imshow("Driver Drowsiness Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("📊 Monitoring ended by user")
            return False

        if (state['drowsy_count'] >= Config.DROWSY_TRIGGER_COUNT and 
            drowsy_state == "DROWSY" and 
            not state['llm_triggered']):
            state['llm_triggered'] = True
            return True

    return False

# =========================
# LLM CONVERSATION
# =========================
def run_llm_conversation(tts, stt, llm_assistant):
    """Run interactive voice conversation with LLM."""
    print("\n" + "="*60)
    print("🚨 DROWSINESS DETECTED - Starting Assistant")
    print("="*60)
    
    tts.speak("I notice you're feeling drowsy. Let me help you stay alert.")
    
    llm_assistant.start_conversation()
    llm_assistant.get_response_streaming()
    tts.wait_until_done()
    
    max_turns = 8
    turn = 0
    
    while turn < max_turns:
        user_input = stt.listen(timeout=15)
        
        if user_input is None:
            tts.speak("Are you still there?")
            tts.wait_until_done()
            user_input = stt.listen(timeout=10)
            
            if user_input is None:
                print("⚠️ No response - ending conversation")
                tts.speak("Okay, resuming monitoring. Stay safe!")
                tts.wait_until_done()
                break
        
        if any(word in user_input.lower() for word in ['exit', 'quit', 'bye', 'stop', 'goodbye', 'done', 'enough']):
            print("🔚 User ended conversation")
            tts.speak("Alright, drive safely!")
            tts.wait_until_done()
            break
        
        llm_assistant.get_response_streaming(user_input)
        tts.wait_until_done()
        
        turn += 1
    
    print("\n" + "="*60)
    print("✓ Conversation ended - Resuming monitoring")
    print("="*60)

# =========================
# MAIN FUNCTION
# =========================
def main():
    """Main function to run the drowsiness detection system."""
    print("\n" + "="*60)
    print("🚗 Driver Drowsiness Detection System V2")
    print("   Optimized for Raspberry Pi 4B")
    print("="*60 + "\n")
    
    tts = TTSEngine()
    stt = STTEngine()
    llm_assistant = LLMAssistant(tts)
    
    cap = initialize_camera()
    face_mesh = initialize_mediapipe()
    state = initialize_state_variables()

    print("✓ System ready - Starting monitoring...")
    tts.speak("Drowsiness monitoring system activated.")

    try:
        while True:
            should_trigger_llm = run_detection_loop(cap, face_mesh, state)

            if not should_trigger_llm:
                break

            cap.release()
            face_mesh.close()
            cv2.destroyAllWindows()

            run_llm_conversation(tts, stt, llm_assistant)

            try:
                cap = initialize_camera()
                face_mesh = initialize_mediapipe()
                state['llm_triggered'] = False
                state['drowsy_count'] = 0
                time.sleep(0.5)
                print("\n✓ Monitoring resumed\n")
            except Exception as e:
                print(f'⚠️ Failed to restart: {e}')
                break

    except KeyboardInterrupt:
        print("\n\n⚠️ System interrupted by user")
    
    finally:
        print("\n🔄 Cleaning up resources...")
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        stt.cleanup()
        tts.shutdown()
        print("✓ Cleanup complete")

if __name__ == "__main__":
    main()