import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, idx):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def mouth_aspect_ratio(landmarks):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[61]
    right = landmarks[291]
    vertical = euclidean(top, bottom)
    horizontal = euclidean(left, right)
    return vertical / horizontal

# -------------------------
# Landmark indices
# -------------------------
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 1
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
MOUTH_LANDMARKS = [13, 14, 61, 291]

EAR_THRESH = 0.25
MAR_THRESH = 0.6
MICROSLEEP_TIME = 1.5
YAWN_TIME = 1.0
WINDOW_TIME = 30
HEAD_NOD_ANGLE_THRESH = 75  # degrees
NOD_TIME = 1.0  # seconds

# -------------------------
# State variables
# -------------------------
eye_closed_start = None
yawn_start = None
head_nod_start = None
closed_time_window = deque()
blink_times = deque()
yawn_times = deque()
nose_positions = deque(maxlen=15)

# -------------------------
# MediaPipe
# -------------------------
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    nod_detected = False  #reset per frame

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        landmarks = [(int(p.x * w), int(p.y * h)) for p in lm]

        #EAR
        ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
               eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0

        #eye close
        if ear < EAR_THRESH:
            if eye_closed_start is None:
                eye_closed_start = now
        else:
            if eye_closed_start is not None:
                duration = now - eye_closed_start
                if duration < 0.8:
                    blink_times.append(now)
                eye_closed_start = None

        #Track closed eye time for PERCLOS
        closed_time_window.append((now, 1 if ear < EAR_THRESH else 0))

        #MAR 
        mar = mouth_aspect_ratio(landmarks)
        if mar > MAR_THRESH:
            if yawn_start is None:
                yawn_start = now
        else:
            if yawn_start is not None:
                if now - yawn_start >= YAWN_TIME:
                    yawn_times.append(now)
                yawn_start = None

        nose = landmarks[NOSE_TIP]
        eye_mid = ((landmarks[LEFT_EYE_CORNER][0] + landmarks[RIGHT_EYE_CORNER][0]) // 2,
                   (landmarks[LEFT_EYE_CORNER][1] + landmarks[RIGHT_EYE_CORNER][1]) // 2)
        delta_x = nose[0] - eye_mid[0]
        delta_y = nose[1] - eye_mid[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        if angle < HEAD_NOD_ANGLE_THRESH:
            if head_nod_start is None:
                head_nod_start = now
            elif now - head_nod_start > NOD_TIME:
                nod_detected = True
        else:
            head_nod_start = None
            nod_detected = False


        for idx in LEFT_EYE + RIGHT_EYE + MOUTH_LANDMARKS + [NOSE_TIP]:
            x, y = landmarks[idx]
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    while closed_time_window and now - closed_time_window[0][0] > WINDOW_TIME:
        closed_time_window.popleft()
    while blink_times and now - blink_times[0] > 60:
        blink_times.popleft()
    while yawn_times and now - yawn_times[0] > 60:
        yawn_times.popleft()

    total_samples = len(closed_time_window)
    closed_samples = sum(v for _, v in closed_time_window)
    perclos = closed_samples / total_samples if total_samples > 0 else 0
    blink_rate = len(blink_times)
    yawn_rate = len(yawn_times)
    microsleep = (eye_closed_start is not None) and (now - eye_closed_start >= MICROSLEEP_TIME)

    cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Blinks/min: {blink_rate}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Yawns/min: {yawn_rate}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Microsleep: {microsleep}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame, f"Head Nod: {nod_detected}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (10,210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    cv2.putText(frame, f"Angle: {angle:.2f} degrees", (10,240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
