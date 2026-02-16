#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import threading
import queue
import asyncio
import json
from pathlib import Path
import speech_recognition as sr

import edge_tts
from groq import Groq

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

class Config:
    # API Keys (loaded from .env file)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
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
# MEMORY MANAGER
# =========================
class MemoryManager:
    """Manages persistent driver profile and conversation history."""
    
    def __init__(self, profile_path="sentinel_driver_profile.json"):
        self.profile_path = Path.home() / profile_path
        self.profile = self.load_profile()
        self.current_session_learnings = []  # Track learnings during this conversation
    
    def load_profile(self):
        """Load driver profile from disk."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading profile: {e}")
                return self.create_default_profile()
        else:
            print("📝 Creating new driver profile...")
            return self.create_default_profile()
    
    def save_profile(self):
        """Save profile to disk."""
        try:
            with open(self.profile_path, 'w') as f:
                json.dump(self.profile, f, indent=2)
            print(f"✓ Profile saved to {self.profile_path}")
        except Exception as e:
            print(f"⚠️ Error saving profile: {e}")
    
    def create_default_profile(self):
        """Initialize empty profile structure."""
        return {
            "personal": {
                "name": None,
                "occupation": None,
                "family": [],
                "location": None
            },
            "interests": {
                "hobbies": [],
                "sports_teams": [],
                "music_preferences": [],
                "topics_engaged_with": []
            },
            "driving_patterns": {
                "common_routes": [],
                "typical_times": [],
                "usual_destinations": []
            },
            "conversation_history": {
                "last_topics": [],
                "successful_engagement_types": [],
                "things_to_avoid": []
            },
            "system_metadata": {
                "total_conversations": 0,
                "last_conversation": None,
                "profile_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_drowsy_episodes": 0
            }
        }
    
    def update_field(self, category, key, value):
        """Update a specific field in the profile."""
        if category not in self.profile:
            self.profile[category] = {}
        self.profile[category][key] = value
        self.save_profile()
    
    def append_to_list(self, category, key, value, max_items=10):
        """Append to a list field, maintaining max size."""
        if category not in self.profile:
            self.profile[category] = {}
        if key not in self.profile[category]:
            self.profile[category][key] = []
        
        # Avoid duplicates
        if value not in self.profile[category][key]:
            self.profile[category][key].append(value)
            # Keep only most recent items
            if len(self.profile[category][key]) > max_items:
                self.profile[category][key] = self.profile[category][key][-max_items:]
        self.save_profile()
    
    def get_profile_summary(self):
        """Get formatted profile summary for LLM context."""
        summary_lines = []
        
        # Personal information
        personal = self.profile.get("personal", {})
        if personal.get("name"):
            summary_lines.append(f"Driver's name: {personal['name']}")
        if personal.get("occupation"):
            summary_lines.append(f"Occupation: {personal['occupation']}")
        if personal.get("family") and len(personal['family']) > 0:
            summary_lines.append(f"Family: {', '.join(personal['family'])}")
        if personal.get("location"):
            summary_lines.append(f"Location: {personal['location']}")
        
        # Interests
        interests = self.profile.get("interests", {})
        if interests.get("hobbies") and len(interests['hobbies']) > 0:
            summary_lines.append(f"Hobbies/Interests: {', '.join(interests['hobbies'])}")
        if interests.get("sports_teams") and len(interests['sports_teams']) > 0:
            summary_lines.append(f"Favorite sports teams: {', '.join(interests['sports_teams'])}")
        if interests.get("music_preferences") and len(interests['music_preferences']) > 0:
            summary_lines.append(f"Music preferences: {', '.join(interests['music_preferences'])}")
        
        # Driving patterns
        driving = self.profile.get("driving_patterns", {})
        if driving.get("common_routes") and len(driving['common_routes']) > 0:
            summary_lines.append(f"Common routes: {', '.join(driving['common_routes'][:3])}")
        if driving.get("usual_destinations") and len(driving['usual_destinations']) > 0:
            summary_lines.append(f"Frequent destinations: {', '.join(driving['usual_destinations'][:3])}")
        
        # Conversation history
        conv_history = self.profile.get("conversation_history", {})
        if conv_history.get("last_topics") and len(conv_history['last_topics']) > 0:
            recent_topics = conv_history['last_topics'][-3:]
            summary_lines.append(f"Recent conversation topics: {', '.join(recent_topics)}")
        if conv_history.get("successful_engagement_types") and len(conv_history['successful_engagement_types']) > 0:
            summary_lines.append(f"Engagement types that work well: {', '.join(conv_history['successful_engagement_types'][:3])}")
        
        # Metadata
        metadata = self.profile.get("system_metadata", {})
        if metadata.get("total_conversations", 0) > 0:
            summary_lines.append(f"Total conversations: {metadata['total_conversations']}")
        if metadata.get("last_conversation"):
            summary_lines.append(f"Last conversation: {metadata['last_conversation']}")
        
        if len(summary_lines) == 0:
            return "**NEW DRIVER** - No profile information yet. This is your first conversation! Learn about the driver by asking friendly questions."
        else:
            return "\n".join(summary_lines)
    
    def extract_learnings_from_text(self, user_text, assistant_text=None):
        """Simple pattern-based extraction of information from conversation."""
        learnings = []
        text_lower = user_text.lower()
        
        # Extract name
        if "my name is" in text_lower or "i'm" in text_lower or "call me" in text_lower:
            # Simple extraction - look for patterns
            if "my name is " in text_lower:
                potential_name = user_text.split("my name is ", 1)[1].split()[0].strip(".,!?")
                if len(potential_name) < 20 and potential_name.isalpha():
                    learnings.append(("personal", "name", potential_name.title()))
        
        # Extract occupation
        if "i work as" in text_lower or "i'm a " in text_lower or "i am a " in text_lower:
            if "i work as " in text_lower:
                occupation = user_text.lower().split("i work as ", 1)[1].split()[0].strip(".,!?")
                learnings.append(("personal", "occupation", occupation))
        
        # Extract interests/hobbies
        hobby_phrases = ["i like", "i love", "i enjoy", "i'm into", "my hobby"]
        for phrase in hobby_phrases:
            if phrase in text_lower:
                # Extract what comes after
                interest = user_text.lower().split(phrase, 1)[1].strip().split('.')[0].strip()
                if len(interest) < 50:
                    learnings.append(("interests", "hobbies", interest))
        
        # Extract destination/route info
        if "going to" in text_lower or "heading to" in text_lower or "driving to" in text_lower:
            for phrase in ["going to", "heading to", "driving to"]:
                if phrase in text_lower:
                    destination = user_text.lower().split(phrase, 1)[1].strip().split('.')[0].split(',')[0].strip()
                    if len(destination) < 30:
                        learnings.append(("driving_patterns", "usual_destinations", destination))
        
        # Store learnings for this session
        self.current_session_learnings.extend(learnings)
        
        return learnings
    
    def apply_session_learnings(self):
        """Apply all learnings from this conversation session."""
        count = len(self.current_session_learnings)
        for category, key, value in self.current_session_learnings:
            if key in ["hobbies", "sports_teams", "music_preferences", "topics_engaged_with",
                      "common_routes", "usual_destinations", "last_topics", "successful_engagement_types"]:
                self.append_to_list(category, key, value)
            elif key in ["family"]:
                # Family is a list but we want to check before adding
                if value not in self.profile[category][key]:
                    self.profile[category][key].append(value)
            else:
                # Single value fields
                self.update_field(category, key, value)
        
        # Clear session learnings
        self.current_session_learnings = []
        print(f"✓ Applied {count} learnings to profile")
    
    def log_conversation_metadata(self, metrics, conversation_length):
        """Log metadata about this conversation."""
        metadata = self.profile.get("system_metadata", {})
        metadata["total_conversations"] = metadata.get("total_conversations", 0) + 1
        metadata["last_conversation"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata["total_drowsy_episodes"] = metadata.get("total_drowsy_episodes", 0) + 1
        self.profile["system_metadata"] = metadata
        self.save_profile()

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
# LLM ASSISTANT WITH MEMORY
# =========================
class LLMAssistant:
    """LLM using Groq API with drowsiness metrics and driver memory as context."""
    
    def __init__(self, tts_engine, memory_manager):
        self.tts = tts_engine
        self.memory_manager = memory_manager  # Memory manager for driver profile
        self.client = None
        self.messages = []
        self.initial_metrics = {}
        self.conversation_turns = 0  # Track conversation length
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
        Start a new conversation with detection metrics AND driver profile as context.
        
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
        
        # Reset conversation turn counter
        self.conversation_turns = 0
        
        # Get driver profile summary
        profile_summary = self.memory_manager.get_profile_summary()
        
        # Build enhanced system prompt with memory
        system_prompt = f"""You are Sentinel, an AI safety companion in a car. Your ONLY job is to help drowsy drivers regain alertness through engaging conversation. You were just activated because the driver crossed the drowsiness threshold.

## Driver Profile (What You Know About This Driver)
{profile_summary}

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
Engage the driver in conversation to restore their alertness. **USE WHAT YOU KNOW ABOUT THEM** to make the conversation personal and engaging. Reference their interests, ask about topics they've mentioned before, use their name if you know it.

## Core Rules (ALWAYS follow):
1. **Keep responses SHORT** - Maximum 2-3 sentences per response
2. **Ask ONE clear question** per turn that relates to THEIR interests/life when possible
3. **Be warm and supportive** - Never alarming, panicky, or lecturing
4. **Use personalization** - Reference their profile information naturally
5. **Vary your approach** - Don't repeat the same types of questions
6. **Read the room** - If they sound slow/tired, increase engagement; if alert, maintain current level
7. **Learn as you go** - If this is a new driver, ask friendly questions to learn about them
8. **Know when to exit** - If they sound consistently alert, prepare to end conversation

## Engagement Toolkit (PRIORITIZE PERSONAL TOPICS)

### 🎯 Personalized Engagement (USE THIS FIRST if you have profile info)
- Reference their hobbies: "How's that [hobby] going?"
- Ask about family: "How's [family member]?"
- Reference destinations: "Still heading to [destination]?"
- Follow up on past topics: "Last time we talked about [topic]..."
- Use their name: "[Name], tell me about..."

### 🧠 Mental Activation (if no personal info or as backup)
Keep their mind working with quick, easy tasks:
- "Quick - what exit number are you passing?"
- "Name 3 things you can see that are blue"
- "What's half of 26?"
- "Count backwards from 15 by 2s"

### 💬 Light Conversation & Learning (for new drivers)
Simple topics that help you learn about them:
- "What do you do for work?"
- "Got any fun plans this weekend?"
- "What kind of music do you like?"
- "Where are you headed today?"
- "Do you have family nearby?"

### 💪 Physical Activation
Suggest simple actions that increase alertness:
- "Try rolling down your window for some fresh air"
- "Take a deep breath in... hold it... exhale slowly"
- "Can you grip the steering wheel tighter for 5 seconds?"

## Conversation Strategy

### Opening (First Response)
**If you know their name:** "Hey [Name]! I noticed you're showing signs of drowsiness. How are you doing?"
**If you don't know them yet:** "Hey! I noticed you're showing signs of drowsiness. I'm Sentinel, and I'm here to help. What's your name?"

### Building Engagement
- Start with personal topics if you have profile information
- Learn about them if you don't have much profile information
- Mix it up based on their responses

### Closing (When They're Alert)
- "You sound much more alert now, [Name]! I'll keep monitoring quietly."
- "Great job staying with me! I'll stay quiet but I'm still watching."

## What NOT to Do
❌ Don't ask complex questions requiring deep thought
❌ Don't discuss emotional/heavy topics
❌ Don't lecture or scold
❌ Don't mention technical metrics or detection scores
❌ Don't be repetitive - vary your questions
❌ Don't create anxiety

## Remember
You're a supportive companion who KNOWS this driver. Use that knowledge to create engaging, personalized conversations that keep them alert without being distracting."""

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
        """Get response from Groq, extract learnings, and speak it."""
        if not self.client:
            return "Sorry, the assistant is not available."
        
        if user_message:
            self.messages.append({"role": "user", "content": user_message})
            print(f"\n👤 You: {user_message}")
            
            # Extract learnings from user input
            learnings = self.memory_manager.extract_learnings_from_text(user_message)
            if learnings:
                print(f"📚 Learned: {len(learnings)} new facts")
        
        print("🤖 Assistant: ", end="", flush=True)
        
        try:
            # Get complete response
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=self.messages,
                temperature=0.7,
                max_tokens=150,
                stream=False
            )
            
            full_response = response.choices[0].message.content
            print(full_response)
            
            # Track conversation turns
            self.conversation_turns += 1
            
            # Speak the entire response smoothly
            self.tts.speak(full_response)
            
            # Add to conversation history
            self.messages.append({"role": "assistant", "content": full_response})
            
            return full_response
            
        except Exception as e:
            print(f"\n⚠️ API error: {e}")
            return "Sorry, I'm having trouble connecting right now."
    
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
def run_llm_conversation(tts, stt, llm_assistant, metrics, state):
    """Run conversation loop with memory extraction."""
    print("\n" + "="*60)
    print("💬 STARTING CONVERSATION (with memory)")
    print("="*60)
    
    # Start conversation with metrics AND memory
    llm_assistant.start_conversation(metrics, state)
    
    # Get opening message
    opening = llm_assistant.get_response_streaming()
    tts.wait_until_done()
    
    # Conversation loop
    max_turns = 15
    for turn in range(max_turns):
        # Listen for user response
        user_input = stt.listen(timeout=20, show_diagnostics=False)
        
        if user_input:
            # Check for exit keywords
            if any(word in user_input.lower() for word in ['exit', 'quit', 'bye', 'stop', 'goodbye', 'done', 'enough', 'fine', 'alert', "i'm good", "im good"]):
                print("🔚 User indicated they're alert")
                tts.speak("Great! You sound much better. I'll keep monitoring quietly.")
                tts.wait_until_done()
                break
            
            # Get LLM response
            response = llm_assistant.get_response_streaming(user_input)
            tts.wait_until_done()
            
            time.sleep(0.5)
        else:
            print("⚠️  No response detected")
            tts.speak("Are you still there? Give me a quick response if you can hear me.")
            tts.wait_until_done()
            
            retry_input = stt.listen(timeout=15, show_diagnostics=False)
            if not retry_input:
                print("⚠️  Still no response - ending conversation")
                tts.speak("I'll keep monitoring. Stay safe!")
                tts.wait_until_done()
                break
            else:
                response = llm_assistant.get_response_streaming(retry_input)
                tts.wait_until_done()
    
    # After conversation ends, apply all learnings
    print("\n💾 Saving conversation learnings to driver profile...")
    llm_assistant.memory_manager.apply_session_learnings()
    llm_assistant.memory_manager.log_conversation_metadata(metrics, llm_assistant.conversation_turns)
    
    print("\n" + "="*60)
    print("✓ Conversation ended - Memory updated - Resuming monitoring")
    print("="*60)

# =========================
# MAIN FUNCTION
# =========================
def main():
    """Main function to run the drowsiness detection system."""
    print("\n" + "="*60)
    print("🚗 Driver Drowsiness Detection System V3.2")
    print("   With Edge-TTS & Groq API")
    print("   Context-Aware Engagement + Memory")
    print("="*60 + "\n")
    
    # Initialize memory manager
    memory_manager = MemoryManager()
    print(f"📊 Driver Profile Status:")
    print(f"   Total conversations: {memory_manager.profile['system_metadata']['total_conversations']}")
    if memory_manager.profile['system_metadata']['last_conversation']:
        print(f"   Last conversation: {memory_manager.profile['system_metadata']['last_conversation']}")
    print()
    
    tts = TTSEngine()
    stt = STTEngine()
    llm_assistant = LLMAssistant(tts, memory_manager)  # Pass memory manager
    
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

def view_driver_profile():
    """Utility function to view the current driver profile."""
    memory_manager = MemoryManager()
    print("\n" + "="*60)
    print("📊 CURRENT DRIVER PROFILE")
    print("="*60)
    print(json.dumps(memory_manager.profile, indent=2))
    print("="*60)

# Add command line argument handling
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--view-profile":
        view_driver_profile()
    else:
        main()
