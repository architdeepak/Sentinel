import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import ollama

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, idx):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    return (euclidean(p2, p6) + euclidean(p3, p5)) / (2.0 * euclidean(p1, p4))

def mouth_aspect_ratio(landmarks):
    return euclidean(landmarks[13], landmarks[14]) / euclidean(landmarks[61], landmarks[291])

# =========================
# Landmark Indices
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
# Thresholds
# =========================
EAR_THRESH = 0.25
MAR_THRESH = 0.6
MICROSLEEP_TIME = 1.5
SLOW_BLINK_TIME = 0.4

HEAD_DOWN_THRESH = 0.12
HEAD_DOWN_TIME = 1.2
HEAD_ROLL_THRESH = 15
ROLL_TIME = 1.2

WINDOW_TIME = 10  # seconds (monitoring window)

# =========================
# Rolling Buffers
# =========================
ear_window = deque()
pitch_window = deque()
closed_window = deque()

blink_times = deque()
blink_durations = deque()
yawn_times = deque()

# =========================
# State Variables
# =========================
eye_closed_start = None
blink_start = None
yawn_start = None
head_down_start = None
head_roll_start = None
llm_triggered = False
drowsyonce = 0

window_start_time = time.time()

# =========================
# MediaPipe
# =========================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Monitoring window Started")

# =========================
# Main Loop
# =========================
while cap.isOpened():
    while not llm_triggered:
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

        # Ensure variables exist even when no face is detected
        ear = 0.0
        mar = 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]

            # =========================
            # EAR + Blink Logic
            # =========================
            ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
                eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2

            ear_window.append((now, ear))
            closed_window.append((now, ear < EAR_THRESH))

            if ear < EAR_THRESH:
                if eye_closed_start is None:
                    eye_closed_start = now
                    blink_start = now
            else:
                if eye_closed_start:
                    duration = now - blink_start
                    blink_times.append(now)
                    blink_durations.append(duration)
                    eye_closed_start = None

            microsleep = eye_closed_start is not None and (now - eye_closed_start >= MICROSLEEP_TIME)

            # =========================
            # MAR (Yawning)
            # =========================
            mar = mouth_aspect_ratio(landmarks)
            if mar > MAR_THRESH:
                if yawn_start is None:
                    yawn_start = now
            else:
                if yawn_start and now - yawn_start > 1.0:
                    yawn_times.append(now)
                yawn_start = None

            # =========================
            # Head Pitch (Down)
            # =========================
            nose_y = landmarks[NOSE_TIP][1]
            eye_mid_y = (landmarks[LEFT_EYE_CORNER][1] + landmarks[RIGHT_EYE_CORNER][1]) / 2
            face_height = abs(landmarks[FOREHEAD][1] - landmarks[CHIN][1])

            pitch = (nose_y - eye_mid_y) / face_height
            pitch_window.append((now, pitch))

            if pitch > HEAD_DOWN_THRESH:
                if head_down_start is None:
                    head_down_start = now
                elif now - head_down_start > HEAD_DOWN_TIME:
                    head_down = True
            else:
                head_down_start = None

            # =========================
            # Head Roll
            # =========================
            dx = landmarks[RIGHT_EYE_CORNER][0] - landmarks[LEFT_EYE_CORNER][0]
            dy = landmarks[RIGHT_EYE_CORNER][1] - landmarks[LEFT_EYE_CORNER][1]
            roll = abs(np.degrees(np.arctan2(dy, dx)))

            if roll > HEAD_ROLL_THRESH:
                if head_roll_start is None:
                    head_roll_start = now
                elif now - head_roll_start > ROLL_TIME:
                    head_roll = True
            else:
                head_roll_start = None

            # Draw points (colored per region)
            for idx in LEFT_EYE:
                cv2.circle(frame, landmarks[idx], 2, (0,255,255), -1)  # left eye - yellow (BGR)
            for idx in RIGHT_EYE:
                cv2.circle(frame, landmarks[idx], 2, (255,0,0), -1)    # right eye - blue (BGR)
            for idx in MOUTH_LANDMARKS:
                cv2.circle(frame, landmarks[idx], 2, (0,0,255), -1)    # mouth - red (BGR)
            cv2.circle(frame, landmarks[NOSE_TIP], 3, (0,255,0), -1)    # nose tip - green (BGR)

        # =========================
        # Window Cleanup
        # =========================
        while ear_window and now - ear_window[0][0] > WINDOW_TIME:
            ear_window.popleft()
        while pitch_window and now - pitch_window[0][0] > WINDOW_TIME:
            pitch_window.popleft()
        while closed_window and now - closed_window[0][0] > WINDOW_TIME:
            closed_window.popleft()
        while blink_times and now - blink_times[0] > WINDOW_TIME:
            blink_times.popleft()
        while blink_durations and len(blink_durations) > len(blink_times):
            blink_durations.popleft()
        while yawn_times and now - yawn_times[0] > WINDOW_TIME:
            yawn_times.popleft()

        # =========================
        # Metrics
        # =========================
        perclos = sum(v for _, v in closed_window) / len(closed_window) if closed_window else 0
        blink_rate = len(blink_times)
        slow_blinks = sum(d > SLOW_BLINK_TIME for d in blink_durations)
        ear_std = np.std([v for _, v in ear_window]) if ear_window else 0
        pitch_var = np.var([v for _, v in pitch_window]) if pitch_window else 0

        # =========================
        # Final Drowsiness Score
        # =========================
        drowsy_score = min(1.0, (
            0.20 * perclos +
            0.15 * int(microsleep) +
            0.15 * min(slow_blinks / 5, 1.0) +
            0.10 * min(ear_std / 0.12, 1.0) +
            0.10 * min(pitch_var / 0.015, 1.0) +
            0.20 * int(head_down) +
            0.10 * int(head_roll)
        ))


        state = "ALERT"
        if drowsy_score > 0.47:
            state = "DROWSY"
            drowsyonce += 1

        # =========================
        # Overlay (colored per region)
        # =========================
        y = 30

        # Colors (BGR)
        EYE_COLOR = (255, 0, 0)    # blue (matches right eye)
        MOUTH_COLOR = (0, 0, 255)  # red
        HEAD_COLOR = (0, 255, 0)   # green

        texts = [
            (f"PERCLOS: {perclos:.2f}", EYE_COLOR),
            (f"Blink Rate: {blink_rate}/min", EYE_COLOR),
            (f"Slow Blinks: {slow_blinks}", EYE_COLOR),
            (f"EAR STD: {ear_std:.3f}", EYE_COLOR),
            (f"EAR: {ear:.3f}", EYE_COLOR),
            (f"MAR: {mar:.3f}", MOUTH_COLOR),
            (f"Yawns: {len(yawn_times)}", MOUTH_COLOR),
            (f"Microsleep: {microsleep}", EYE_COLOR),
            (f"Head Down: {head_down}", HEAD_COLOR),
            (f"Head Roll: {head_roll}", HEAD_COLOR),
        ]

        # Drowsy score: green if ALERT, red if DROWSY
        drowsy_color = (0, 255, 0) if state == "ALERT" else (0, 0, 255)
        texts.append((f"Drowsy Score: {drowsy_score:.2f} ({state})", drowsy_color))

        for text, color in texts:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y += 25

        cv2.imshow("Driver Drowsiness Monitor", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("📊 Monitoring window ENDED")
            break
        
        if drowsyonce >= 10 and state == "DROWSY" and not llm_triggered:
            llm_triggered = True
            print("Drowsiness detected! Triggering LLM assistant...")

            # Stop the detection loop's camera and windows before starting the chat
            cap.release()
            face_mesh.close()
            cv2.destroyAllWindows()
            print("🔔 Detection loop stopped. Starting interactive assistant chat (type 'exit' to finish).")

            # Initial conversation context
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

            # Enter interactive multi-turn chat
            try:
                while True:
                    response = ollama.chat(model='qwen:4b', messages=messages)
                    assistant_msg = response.get('message', {}).get('content', '')
                    messages.append({'role': 'assistant', 'content': assistant_msg})
                    print(assistant_msg)
                    user_input = input("You (type 'exit' to end): ").strip()
                    if user_input.lower() in ('exit', 'quit', 'bye'):
                        print("🔚 Ending chat.")
                        break

                    messages.append({'role': 'user', 'content': user_input})
            except Exception as e:
                print("⚠️ Chat error:", e)

            print("🔔 Chat finished — resuming monitoring.")
            try:
                speak("I see that you are no longer drowsy. Resuming monitoring.")
            except Exception:
                pass

            try:
                cap = cv2.VideoCapture(0)
                face_mesh = mp_face.FaceMesh(
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                llm_triggered = False
                drowsyonce = 0
                time.sleep(0.5)
            except Exception as e:
                print('⚠️ Failed to restart camera/face mesh:', e)
                break

# Final cleanup
try:
    shutdown_tts()
except Exception:
    pass
cap.release()
face_mesh.close()
cv2.destroyAllWindows()