import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import ollama
import pyttsx3
import threading
import speech_recognition as sr

# =========================
# Text-to-Speech Setup
# =========================
tts_lock = threading.Lock()

def get_best_voice():
    """Get the most realistic-sounding voice available."""
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.stop()
        del engine
        
        # Priority order for voice selection (platform-specific)
        preferred_names = [
            'zira', 'david',  # Windows voices
            'samantha', 'alex',  # macOS voices
            'fiona', 'daniel',  # Other common voices
        ]
        
        # First try: Look for preferred voice names
        for preferred in preferred_names:
            for voice in voices:
                if preferred in voice.name.lower():
                    return voice.id
        
        # Second try: Look for "enhanced" or "premium" voices
        for voice in voices:
            voice_lower = voice.name.lower()
            if 'enhanced' in voice_lower or 'premium' in voice_lower:
                return voice.id
        
        # Third try: Prefer female voices (often sound clearer)
        for voice in voices:
            voice_lower = voice.name.lower()
            if 'female' in voice_lower or 'woman' in voice_lower:
                return voice.id
        
        # Default: Use second voice if available (first is often lower quality)
        if len(voices) > 1:
            return voices[1].id
        
        # Fallback: Use first available voice
        return voices[0].id if voices else None
        
    except Exception as e:
        print(f"⚠️ Voice selection error: {e}")
        return None

def speak(text):
    """Speak the given text using TTS (reinitializes engine each time)."""
    try:
        with tts_lock:
            engine = pyttsx3.init()
            
            # Set voice to most realistic option
            best_voice = get_best_voice()
            if best_voice:
                engine.setProperty('voice', best_voice)
            
            engine.setProperty('rate', 175)  # Faster speech (was 150)
            engine.setProperty('volume', 0.95)  # Slightly louder
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine  # Force cleanup
    except Exception as e:
        print(f"⚠️ TTS error: {e}")

def speak_async(text):
    """Speak text in a separate thread to avoid blocking."""
    thread = threading.Thread(target=speak, args=(text,))
    thread.daemon = True
    thread.start()

# =========================
# Speech-to-Text Setup
# =========================
def listen_for_speech(timeout=5, phrase_time_limit=10):
    """
    Listen for user speech and convert to text.
    
    Args:
        timeout: Seconds to wait for speech to start
        phrase_time_limit: Maximum seconds for the phrase
    
    Returns:
        str: Transcribed text, or None if failed
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("\n🎤 Listening... (speak now)")
            
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen for speech
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            
            print("🔄 Processing speech...")
            
            # Convert speech to text using Google's speech recognition
            text = recognizer.recognize_google(audio)
            print(f"✓ You said: {text}")
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

def initialize_microphone():
    """Test and initialize the microphone."""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Microphone initialized successfully")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("✓ Ambient noise calibration complete")
            return True
    except Exception as e:
        print(f"⚠️ Microphone initialization failed: {e}")
        return False

# =========================
# Helper Functions
# =========================
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, idx):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))

def mouth_aspect_ratio(landmarks):
    return euclidean(landmarks[13], landmarks[14]) / euclidean(landmarks[61], landmarks[291])

# =========================
# Constants
# =========================
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
MOUTH_LANDMARKS = [13, 14, 61, 291]

EAR_THRESH = 0.25
MAR_THRESH = 0.6
MICROSLEEP_TIME = 1.5
SLOW_BLINK_TIME = 0.4

HEAD_DOWN_THRESH = 0.12
HEAD_DOWN_TIME = 1.2
HEAD_ROLL_THRESH = 15
ROLL_TIME = 1.2

WINDOW_TIME = 10

# =========================
# Initialization Functions
# =========================
def initialize_mediapipe():
    """Initialize MediaPipe face mesh."""
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return face_mesh

def initialize_state_variables():
    """Initialize all state tracking variables."""
    state = {
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
        'drowsyonce': 0,
        'window_start_time': time.time()
    }
    return state

# =========================
# Detection Functions
# =========================
def process_eye_metrics(landmarks, state, now):
    """Process eye-related metrics and detect microsleep."""
    ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
           eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2

    state['ear_window'].append((now, ear))
    state['closed_window'].append((now, ear < EAR_THRESH))

    if ear < EAR_THRESH:
        if state['eye_closed_start'] is None:
            state['eye_closed_start'] = now
            state['blink_start'] = now
    else:
        if state['eye_closed_start']:
            duration = now - state['blink_start']
            state['blink_times'].append(now)
            state['blink_durations'].append(duration)
            state['eye_closed_start'] = None

    microsleep = state['eye_closed_start'] is not None and (now - state['eye_closed_start'] >= MICROSLEEP_TIME)
    return ear, microsleep

def process_mouth_metrics(landmarks, state, now):
    """Process mouth metrics and detect yawning."""
    mar = mouth_aspect_ratio(landmarks)
    if mar > MAR_THRESH:
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
    if pitch > HEAD_DOWN_THRESH:
        if state['head_down_start'] is None:
            state['head_down_start'] = now
        elif now - state['head_down_start'] > HEAD_DOWN_TIME:
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
    if roll > HEAD_ROLL_THRESH:
        if state['head_roll_start'] is None:
            state['head_roll_start'] = now
        elif now - state['head_roll_start'] > ROLL_TIME:
            head_roll = True
    else:
        state['head_roll_start'] = None

    return head_roll

def cleanup_windows(state, now):
    """Remove old entries from time windows."""
    while state['ear_window'] and now - state['ear_window'][0][0] > WINDOW_TIME:
        state['ear_window'].popleft()
    while state['pitch_window'] and now - state['pitch_window'][0][0] > WINDOW_TIME:
        state['pitch_window'].popleft()
    while state['closed_window'] and now - state['closed_window'][0][0] > WINDOW_TIME:
        state['closed_window'].popleft()
    while state['blink_times'] and now - state['blink_times'][0] > WINDOW_TIME:
        state['blink_times'].popleft()
    while state['blink_durations'] and len(state['blink_durations']) > len(state['blink_times']):
        state['blink_durations'].popleft()
    while state['yawn_times'] and now - state['yawn_times'][0] > WINDOW_TIME:
        state['yawn_times'].popleft()

def calculate_metrics(state, microsleep, head_down, head_roll):
    """Calculate drowsiness metrics and score."""
    perclos = sum(v for _, v in state['closed_window']) / len(state['closed_window']) if state['closed_window'] else 0
    blink_rate = len(state['blink_times'])
    slow_blinks = sum(d > SLOW_BLINK_TIME for d in state['blink_durations'])
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

def draw_landmarks(frame, landmarks):
    """Draw facial landmarks on the frame."""
    for idx in LEFT_EYE:
        cv2.circle(frame, landmarks[idx], 2, (0, 255, 255), -1)
    for idx in RIGHT_EYE:
        cv2.circle(frame, landmarks[idx], 2, (255, 0, 0), -1)
    for idx in MOUTH_LANDMARKS:
        cv2.circle(frame, landmarks[idx], 2, (0, 0, 255), -1)
    cv2.circle(frame, landmarks[NOSE_TIP], 3, (0, 255, 0), -1)

def draw_overlay(frame, metrics, ear, mar, microsleep, head_down, head_roll, state, drowsy_state):
    """Draw metrics overlay on the frame."""
    y = 30
    EYE_COLOR = (255, 0, 0)
    MOUTH_COLOR = (0, 0, 255)
    HEAD_COLOR = (0, 255, 0)

    texts = [
        (f"PERCLOS: {metrics['perclos']:.2f}", EYE_COLOR),
        (f"Blink Rate: {metrics['blink_rate']}/min", EYE_COLOR),
        (f"Slow Blinks: {metrics['slow_blinks']}", EYE_COLOR),
        (f"EAR STD: {metrics['ear_std']:.3f}", EYE_COLOR),
        (f"EAR: {ear:.3f}", EYE_COLOR),
        (f"MAR: {mar:.3f}", MOUTH_COLOR),
        (f"Yawns: {len(state['yawn_times'])}", MOUTH_COLOR),
        (f"Microsleep: {microsleep}", EYE_COLOR),
        (f"Head Down: {head_down}", HEAD_COLOR),
        (f"Head Roll: {head_roll}", HEAD_COLOR),
    ]

    drowsy_color = (0, 255, 0) if drowsy_state == "ALERT" else (0, 0, 255)
    texts.append((f"Drowsy Score: {metrics['drowsy_score']:.2f} ({drowsy_state})", drowsy_color))

    for text, color in texts:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 25

# =========================
# Main Detection Loop
# =========================
def run_detection_loop(cap, face_mesh, state):
    """Run the main drowsiness detection loop."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape

        microsleep = False
        head_down = False
        head_roll = False
        ear = 0
        mar = 0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]

            # Process all detections
            ear, microsleep = process_eye_metrics(landmarks, state, now)
            mar = process_mouth_metrics(landmarks, state, now)
            head_down = process_head_pitch(landmarks, state, now)
            head_roll = process_head_roll(landmarks, state, now)

            draw_landmarks(frame, landmarks)

        # Cleanup old data
        cleanup_windows(state, now)

        # Calculate metrics
        metrics = calculate_metrics(state, microsleep, head_down, head_roll)

        # Determine state
        drowsy_state = "ALERT"
        if metrics['drowsy_score'] > 0.47:
            drowsy_state = "DROWSY"
            state['drowsyonce'] += 1

        # Draw overlay
        draw_overlay(frame, metrics, ear, mar, microsleep, head_down, head_roll, state, drowsy_state)

        cv2.imshow("Driver Drowsiness Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("📊 Monitoring window ENDED")
            return False

        # Check if LLM should be triggered
        if state['drowsyonce'] >= 10 and drowsy_state == "DROWSY" and not state['llm_triggered']:
            state['llm_triggered'] = True
            return True

    return False

# =========================
# LLM Chat Function
# =========================
def run_llm_chat():
    """Run the interactive LLM chat assistant with voice input."""
    print("Drowsiness detected! Triggering LLM assistant...")
    speak_async("I notice you're feeling drowsy. Let me help you stay alert.")
    print("🔔 Detection loop stopped. Starting interactive voice assistant (say 'exit' to finish).")
    
    # Initialize microphone
    if not initialize_microphone():
        print("⚠️ Microphone not available. Falling back to text input.")
        use_voice = False
    else:
        use_voice = True
        print("✓ Voice mode activated")

    messages = [
        {
            'role': 'system',
            'content': (
                "You are an in-car voice assistant designed to keep a driver awake and attentive.\n"
                "Rules you MUST follow:\n"
                "- Start convos with 'hey I see that you are feeling drowsy' and offer to help.\n"
                "- Speak in short, calm sentences.\n"
                "- Try to ease the driver's drowsiness with gentle suggestions.\n"
                "- Ask questions about engaging in conversations or other ways to stay alert.\n"
                "- Ask only ONE simple question at a time.\n"
                "- Keep responses under 3 sentences.\n"
                "- Use a friendly, conversational tone.\n"
                "- Driver state: Mildly Drowsy."
            )
        },
        {
            'role': 'user',
            'content': 'I am feeling a little drowsy'
        }
    ]

    try:
        while True:
            # Get LLM response
            response = ollama.chat(model='qwen:4b', messages=messages)
            assistant_msg = response.get('message', {}).get('content', '')
            messages.append({'role': 'assistant', 'content': assistant_msg})
            print(f"\n🤖 Assistant: {assistant_msg}")
            
            # Speak the assistant's response
            speak(assistant_msg)
            
            # Get user input (voice or text)
            if use_voice:
                user_input = listen_for_speech(timeout=10, phrase_time_limit=15)
                
                # Handle failed speech recognition
                if user_input is None:
                    print("💬 Say something, or press Enter to skip...")
                    # Give one more chance
                    user_input = listen_for_speech(timeout=10, phrase_time_limit=15)
                    
                    if user_input is None:
                        print("⚠️ No speech detected. Type your response or 'exit' to end:")
                        user_input = input("You (text): ").strip()
            else:
                # Fallback to text input
                user_input = input("\nYou (text - say 'exit' to end): ").strip()
            
            # Check for exit commands
            if user_input and user_input.lower() in ('exit', 'quit', 'bye', 'goodbye', 'stop'):
                print("🔚 Ending chat.")
                speak_async("Okay, drive safely!")
                break
            
            # If we got valid input, add to conversation
            if user_input:
                messages.append({'role': 'user', 'content': user_input})
            else:
                # No input received, prompt again
                print("⚠️ No input received. Let me ask again...")
                
    except Exception as e:
        print("⚠️ Chat error:", e)

    print("🔔 Chat finished — resuming monitoring.")
    speak_async("Resuming monitoring. Stay alert!")


# =========================
# Main Function
# =========================
def main():
    """Main function to run the drowsiness detection system."""
    cap = cv2.VideoCapture(0)
    face_mesh = initialize_mediapipe()
    state = initialize_state_variables()

    print("Monitoring window Started")
    speak_async("Drowsiness monitoring system activated.")

    try:
        while True:
            # Run detection loop
            should_trigger_llm = run_detection_loop(cap, face_mesh, state)

            if not should_trigger_llm:
                break

            # Stop detection resources
            cap.release()
            face_mesh.close()
            cv2.destroyAllWindows()

            # Run LLM chat
            run_llm_chat()

            # Restart detection
            try:
                cap = cv2.VideoCapture(0)
                face_mesh = initialize_mediapipe()
                state['llm_triggered'] = False
                state['drowsyonce'] = 0
                time.sleep(0.5)
            except Exception as e:
                print('⚠️ Failed to restart camera/face mesh:', e)
                break

    finally:
        # Final cleanup
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()