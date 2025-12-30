#!/usr/bin/env python3
"""
Driver Drowsiness Detection System V3
With Edge-TTS and Groq API Integration
Optimized for Raspberry Pi 4B
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import threading
import queue
import asyncio
from pathlib import Path
import speech_recognition as sr

# New imports for APIs
import edge_tts
from groq import Groq

# =========================
# CONFIGURATION
# =========================
class Config:
    # API Keys (set these as environment variables or here)
    GROQ_API_KEY = "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"  # Get free at console.groq.com
    
    # API settings
    GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and good quality
    EDGE_TTS_VOICE = "en-US-JennyNeural"  # Options: AriaNeural, GuyNeural, JennyNeural
    EDGE_TTS_RATE = "+25%"  # Speed adjustment
    
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
# TEXT-TO-SPEECH (Edge-TTS)
# =========================
class TTSEngine:
    """High-quality TTS using Microsoft Edge-TTS API."""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()
        self.is_speaking = False
        self.temp_audio = Path("/tmp/tts_temp.mp3")
    
    def _audio_worker(self):
        """Process TTS queue."""
        while True:
            text = self.audio_queue.get()
            if text is None:
                break
            
            self.is_speaking = True
            try:
                # Generate speech with Edge-TTS
                asyncio.run(self._generate_speech(text))
                
                # Play using mpg123 (lightweight MP3 player for RPi)
                import subprocess
                subprocess.run(
                    ["mpg123", "-q", str(self.temp_audio)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Clean up temp file
                if self.temp_audio.exists():
                    self.temp_audio.unlink()
                    
            except Exception as e:
                print(f"⚠️ TTS error: {e}")
            
            self.is_speaking = False
            self.audio_queue.task_done()
    
    async def _generate_speech(self, text):
        """Generate speech file using Edge-TTS."""
        communicate = edge_tts.Communicate(
            text,
            Config.EDGE_TTS_VOICE,
            rate=Config.EDGE_TTS_RATE
        )
        await communicate.save(str(self.temp_audio))
    
    def speak(self, text):
        """Queue text for speech."""
        if text and text.strip():
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
    """speech_recognition-based STT using Google's online recognizer."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = None
        self._initialize()

    def _initialize(self):
        """Initialize microphone and calibrate ambient noise."""
        try:
            print("🎤 Initializing microphone...")
            self.mic = sr.Microphone()
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Microphone ready")
        except Exception as e:
            print(f"⚠️ STT init failed: {e}")
            self.mic = None

    def listen(self, timeout=20, show_diagnostics=False):
        """Listen for speech and return recognized text (or None)."""
        if not self.mic:
            print("⚠️ STT not available")
            return None

        print(f"\n🎤 Listening (timeout: {timeout}s)...")
        if show_diagnostics:
            print("   [Listening with Google STT]")

        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=timeout)

            print("🔄 Processing speech...")
            text = self.recognizer.recognize_google(audio)
            print(f"✓ You said: '{text}'")
            return text

        except sr.WaitTimeoutError:
            print("⏱️ No speech detected (timeout)")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"⚠️ Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Microphone error: {e}")
            return None

    def cleanup(self):
        """Cleanup resources."""
        pass

# =========================
# LLM ASSISTANT (Groq API)
# =========================
# =========================
# LLM ASSISTANT WITH BINARY STATE
# =========================
class LLMAssistant:
    """LLM using Groq API with drowsiness metrics as context."""
    
    def __init__(self, tts_engine):
        self.tts = tts_engine
        self.client = None
        self.messages = []
        self.initial_metrics = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Groq client."""
        try:
            print("🧠 Connecting to Groq API...")
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            print("✓ Groq API ready")
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}")
            self.client = None
    
    def start_conversation(self, metrics, state):
        """
        Start a new conversation with detection metrics as context.
        
        Args:
            metrics: Dict with drowsy_score, perclos, blink_rate, etc.
            state: State dict with yawn_times, eye_closed_start, etc.
        """
        # Store initial metrics
        self.initial_metrics = {
            'drowsy_score': metrics['drowsy_score'],
            'perclos': metrics['perclos'],
            'blink_rate': metrics['blink_rate'],
            'yawn_count': len(state['yawn_times']),
            'microsleep': state['eye_closed_start'] is not None
        }
        
        # Build system prompt with metrics
        system_prompt = f"""You are Sentinel, an AI safety companion in a car. Your ONLY job is to help drowsy drivers regain alertness through engaging conversation. You were just activated because the driver crossed the drowsiness threshold.

## Current Driver State (Detection Metrics)
```
STATUS: DROWSY (threshold exceeded)
DROWSINESS SCORE: {metrics['drowsy_score']:.2f} / 1.00
EYE CLOSURE (PERCLOS): {metrics['perclos']:.2f}
BLINK RATE (last 10s): {metrics['blink_rate']}
YAWN COUNT (last 10s): {len(state['yawn_times'])}
MICROSLEEP DETECTED: {state['eye_closed_start'] is not None}
```

**What this means:** The driver is showing clear signs of drowsiness and needs engagement to regain full alertness.

## Your Mission
Engage the driver in conversation to restore their alertness. Use a combination of mental activation, light conversation, and physical prompts. Adapt your intensity based on the metrics above and their responses.

## Core Rules (ALWAYS follow):
1. **Keep responses SHORT** - Maximum 2-3 sentences per response
2. **Ask ONE clear question** per turn
3. **Be warm and supportive** - Never alarming, panicky, or lecturing
4. **Vary your approach** - Don't repeat the same types of questions
5. **Read the room** - If they sound slow/tired, increase engagement; if alert, maintain current level
6. **Know when to exit** - If they sound consistently alert, prepare to end conversation

## Engagement Toolkit (use variety!)

### 🧠 Mental Activation
Keep their mind working with quick, easy tasks:
- "Quick - what exit number are you passing?"
- "Name 3 things you can see that are blue"
- "What's half of 26?"
- "Count backwards from 15 by 2s"
- "What's playing on the radio?"
- "Spell your destination backwards"

### 💬 Light Conversation
Simple topics that don't require deep thought:
- "How much longer until you arrive?"
- "What's the first thing you'll do when you get there?"
- "Is the traffic moving okay?"
- "What's the weather like ahead?"
- "Planning to stop for coffee soon?"

### 💪 Physical Activation
Suggest simple actions that increase alertness:
- "Try rolling down your window for some fresh air"
- "Take a deep breath in... hold it... exhale slowly"
- "Adjust your seat position if you're getting stiff"
- "Crank up the AC or open a window"
- "Can you grip the steering wheel tighter for 5 seconds?"

### ⚡ Quick Decision Making
Simple choices that engage their brain:
- "Coffee or energy drink at the next stop?"
- "Want me to tell you a fact or ask you a trivia question?"
- "Music up or window down?"

## Conversation Strategy

### Opening (First Response)
Start warm and check in:
- "Hey! I noticed you're showing signs of drowsiness. How are you feeling?"
- "I'm here to help keep you alert. How's the drive going?"

### Building Engagement
Mix it up based on their responses:
- If they respond quickly/clearly → Light conversation
- If they respond slowly/quietly → Mental tasks, physical prompts
- If they don't respond → Simpler yes/no questions

### Closing (When They're Alert)
- "You sound much more alert now! I'll keep monitoring quietly."
- "Great! You're doing well. I'll stay quiet but I'm still watching."

## Adapting to Metrics

**If PERCLOS is high (>0.30):** Eyes closing frequently
→ "Open that window right now, get some air!"

**If multiple yawns detected:**
→ "I counted those yawns. How about pulling over for 5 minutes?"

**If microsleep detected:**
→ "Hey! I need you alert right now. Tell me where you are."

**If score is very high (>0.70):**
→ Suggest pulling over: "I'm concerned. Can you pull over safely in the next mile?"

## What NOT to Do
❌ Don't ask complex questions
❌ Don't discuss emotional/heavy topics
❌ Don't lecture or scold
❌ Don't mention technical metrics
❌ Don't be repetitive
❌ Don't create anxiety
❌ Don't talk for too long

## Remember
You're a supportive companion. Keep it brief, varied, and engaging. Your goal is to restore alertness without becoming a distraction."""

        # Initialize conversation
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "I am feeling drowsy while driving"
            }
        ]
    
    def get_response_streaming(self, user_message=None):
        """Get response from Groq and speak it."""
        if not self.client:
            return "Sorry, the assistant is not available."
        
        if user_message:
            self.messages.append({"role": "user", "content": user_message})
            print(f"\n👤 You: {user_message}")
        
        print("🤖 Assistant: ", end="", flush=True)
        
        try:
            # Get response from Groq
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=self.messages,
                temperature=0.7,
                max_tokens=150,
                stream=False
            )
            
            full_response = response.choices[0].message.content
            print(full_response)
            
            # Speak the response
            self.tts.speak(full_response)
            
            # Add to conversation history
            self.messages.append({"role": "assistant", "content": full_response})
            
            return full_response
            
        except Exception as e:
            print(f"\n⚠️ API error: {e}")
            fallback = "I'm having trouble connecting. How are you feeling right now?"
            self.tts.speak(fallback)
            return fallback
    
    def should_end_conversation(self, turn_count, max_turns=8):
        """
        Determine if conversation should end.
        
        Simple version - can enhance with response analysis.
        """
        if turn_count >= max_turns:
            return True, "max_turns_reached"
        
        return False, None

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
    final_metrics = None  # Store metrics when drowsiness triggers
    
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
            # Store the metrics for when we trigger conversation
            final_metrics = metrics.copy()

        draw_overlay(frame, metrics, ear, mar, microsleep, head_down, 
                     head_roll, state, drowsy_state)

        cv2.imshow("Driver Drowsiness Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("📊 Monitoring ended by user")
            return False, None

        if (state['drowsy_count'] >= Config.DROWSY_TRIGGER_COUNT and 
            drowsy_state == "DROWSY" and 
            not state['llm_triggered']):
            state['llm_triggered'] = True
            return True, final_metrics  # Return the metrics!

    return False, None

# =========================
# LLM CONVERSATION
# =========================
def run_llm_conversation(tts, stt, llm_assistant, final_metrics, state):
    """
    Run interactive voice conversation with LLM.
    
    Args:
        tts: TTS engine
        stt: STT engine  
        llm_assistant: LLM assistant instance
        final_metrics: The metrics dict when drowsiness was detected
        state: The state dict with yawn_times, eye_closed_start, etc.
    """
    print("\n" + "="*60)
    print("🚨 DROWSINESS DETECTED - Starting Assistant")
    print("="*60)
    print(f"Drowsiness Score: {final_metrics['drowsy_score']:.2f}")
    print(f"PERCLOS: {final_metrics['perclos']:.2f}")
    print(f"Blinks: {final_metrics['blink_rate']}")
    print(f"Yawns: {len(state['yawn_times'])}")
    print("="*60)
    
    # Initial TTS alert
    tts.speak("I notice you're feeling drowsy. Let me help you stay alert.")
    
    # Start conversation with metrics as context
    llm_assistant.start_conversation(final_metrics, state)
    
    # Get first LLM response (uses the "I am feeling drowsy" user message)
    llm_assistant.get_response_streaming()
    tts.wait_until_done()
    
    max_turns = 8
    turn = 0
    
    while turn < max_turns:
        turn += 1
        print(f"\n--- Turn {turn}/{max_turns} ---")
        
        # Listen for user response
        user_input = stt.listen(timeout=20, show_diagnostics=False)
        
        if user_input is None:
            print("⚠️ No input detected")
            tts.speak("I didn't hear anything. Are you still there?")
            tts.wait_until_done()
            
            # Second attempt
            user_input = stt.listen(timeout=10, show_diagnostics=False)
            
            if user_input is None:
                print("⚠️ No response after two attempts - ending conversation")
                tts.speak("Okay, resuming monitoring. Stay safe!")
                tts.wait_until_done()
                break
        
        # Check for exit keywords
        if any(word in user_input.lower() for word in ['exit', 'quit', 'bye', 'stop', 'goodbye', 'done', 'enough', 'fine', 'alert', "i'm good", "im good"]):
            print("🔚 User indicated they're alert")
            tts.speak("Great! You sound much better. I'll keep monitoring quietly.")
            tts.wait_until_done()
            break
        
        # Get LLM response based on user input
        llm_assistant.get_response_streaming(user_input)
        tts.wait_until_done()
        
        # Check if should end (can be enhanced with response analysis)
        should_end, reason = llm_assistant.should_end_conversation(turn, max_turns)
        if should_end:
            if reason == "max_turns_reached":
                print(f"🔚 Reached {max_turns} turns")
                tts.speak("You're sounding much more alert. I'll keep monitoring quietly. Drive safe!")
                tts.wait_until_done()
            break
    
    print("\n" + "="*60)
    print("✓ Conversation ended - Resuming monitoring")
    print("="*60)

# =========================
# MAIN FUNCTION
# =========================
def main():
    """Main function to run the drowsiness detection system."""
    print("\n" + "="*60)
    print("🚗 Driver Drowsiness Detection System V3")
    print("   With Edge-TTS & Groq API")
    print("   Context-Aware Engagement")
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
            should_trigger_llm, final_metrics = run_detection_loop(cap, face_mesh, state)

            if not should_trigger_llm:
                break

            cap.release()
            face_mesh.close()
            cv2.destroyAllWindows()

            # Pass metrics to conversation!
            run_llm_conversation(tts, stt, llm_assistant, final_metrics, state)

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