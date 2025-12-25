import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

# =========================
# Utility Functions
# =========================
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

print("📊 Monitoring window STARTED")

# =========================
# Main Loop
# =========================
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

        # Draw points
        for idx in LEFT_EYE + RIGHT_EYE + MOUTH_LANDMARKS + [NOSE_TIP]:
            cv2.circle(frame, landmarks[idx], 2, (0,255,0), -1)

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
    if drowsy_score > 0.75:
        state = "DROWSY"
    elif drowsy_score > 0.45:
        state = "DROWSY"

    # =========================
    # Overlay
    # =========================
    y = 30
    for text in [
        f"PERCLOS: {perclos:.2f}",
        f"Blink Rate: {blink_rate}/min",
        f"Slow Blinks: {slow_blinks}",
        f"EAR STD: {ear_std:.3f}",
        f"EAR: {ear:.3f}",
        f"MAR: {mar:.3f}",
        f"Yawns: {len(yawn_times)}",
        f"Microsleep: {microsleep}",
        f"Head Down: {head_down}",
        f"Head Roll: {head_roll}",
        f"Drowsy Score: {drowsy_score:.2f} ({state})"
    ]:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
        y += 25

    cv2.imshow("Driver Drowsiness Monitor", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("📊 Monitoring window ENDED")
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
