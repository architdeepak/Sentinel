#!/usr/bin/env python3
"""Standalone Windows-friendly MediaPipe Face Mesh visualizer.

- Draws every detected face landmark point (all Face Mesh indices)
- Shows live detection/drowsiness metrics directly in this script
- Press 'q' to quit
"""

import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


# Face mesh landmark indices used for derived metrics.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152

# Local thresholds (independent from project Config).
EAR_THRESH = 0.20
MAR_THRESH = 0.60
MICROSLEEP_TIME = 1.5
SLOW_BLINK_TIME = 0.55
HEAD_DOWN_TIME = 1.2
HEAD_ROLL_THRESH_DEG = 15.0
ROLL_TIME = 1.2
WINDOW_TIME = 10.0
DROWSY_THRESHOLD = 0.55

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30


def init_state():
    win = 200
    return {
        "ear_window": deque(maxlen=win),
        "pitch_window": deque(maxlen=win),
        "closed_window": deque(maxlen=win),
        "blink_times": deque(maxlen=50),
        "blink_durations": deque(maxlen=50),
        "yawn_times": deque(maxlen=20),
        "eye_closed_start": None,
        "blink_start": None,
        "yawn_start": None,
        "head_down_start": None,
        "head_roll_start": None,
        "window_start_time": time.time(),
    }


def eye_aspect_ratio(landmarks, idx):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    denom = 2.0 * math.dist(p1, p4)
    if denom < 1e-6:
        return 0.25
    return (math.dist(p2, p6) + math.dist(p3, p5)) / denom


def mouth_aspect_ratio(landmarks):
    denom = math.dist(landmarks[61], landmarks[291])
    if denom < 1e-6:
        return 0.0
    return math.dist(landmarks[13], landmarks[14]) / denom


def process_eye_metrics(landmarks, state, now):
    ear = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2
    state["ear_window"].append((now, ear))
    state["closed_window"].append((now, ear < EAR_THRESH))

    if ear < EAR_THRESH:
        if state["eye_closed_start"] is None:
            state["eye_closed_start"] = now
            state["blink_start"] = now
    else:
        if state["eye_closed_start"] is not None:
            duration = now - state["blink_start"]
            state["blink_times"].append(now)
            state["blink_durations"].append(duration)
            state["eye_closed_start"] = None
            state["blink_start"] = None

    microsleep = state["eye_closed_start"] is not None and (now - state["eye_closed_start"] >= MICROSLEEP_TIME)
    return ear, microsleep


def process_mouth_metrics(landmarks, state, now):
    mar = mouth_aspect_ratio(landmarks)
    if mar > MAR_THRESH:
        if state["yawn_start"] is None:
            state["yawn_start"] = now
    else:
        if state["yawn_start"] is not None and now - state["yawn_start"] > 1.0:
            state["yawn_times"].append(now)
        state["yawn_start"] = None
    return mar


def process_head_pitch(landmarks, state, now):
    nose_y = landmarks[NOSE_TIP][1]
    chin_y = landmarks[CHIN][1]
    forehead_y = landmarks[FOREHEAD][1]

    face_height = abs(forehead_y - chin_y)
    if face_height < 10:
        state["head_down_start"] = None
        return False

    upper_ratio = (nose_y - forehead_y) / face_height
    state["pitch_window"].append((now, upper_ratio))

    calib = state.setdefault("_pitch_calib", [])
    baseline = state.get("_pitch_baseline")
    if baseline is None:
        calib.append(upper_ratio)
        if len(calib) >= 30:
            state["_pitch_baseline"] = float(np.median(calib))
        state["head_down_start"] = None
        return False

    head_dropping = (upper_ratio - baseline) > 0.10
    if head_dropping:
        if state["head_down_start"] is None:
            state["head_down_start"] = now
        elif now - state["head_down_start"] > HEAD_DOWN_TIME:
            return True
    else:
        state["head_down_start"] = None
    return False


def process_head_roll(landmarks, state, now):
    dx = landmarks[RIGHT_EYE_CORNER][0] - landmarks[LEFT_EYE_CORNER][0]
    dy = landmarks[RIGHT_EYE_CORNER][1] - landmarks[LEFT_EYE_CORNER][1]
    roll = abs(math.degrees(math.atan2(dy, dx)))

    if roll > HEAD_ROLL_THRESH_DEG:
        if state["head_roll_start"] is None:
            state["head_roll_start"] = now
        elif now - state["head_roll_start"] > ROLL_TIME:
            return True
    else:
        state["head_roll_start"] = None
    return False


def cleanup_windows(state, now):
    cutoff = now - WINDOW_TIME
    while state["ear_window"] and state["ear_window"][0][0] < cutoff:
        state["ear_window"].popleft()
    while state["pitch_window"] and state["pitch_window"][0][0] < cutoff:
        state["pitch_window"].popleft()
    while state["closed_window"] and state["closed_window"][0][0] < cutoff:
        state["closed_window"].popleft()
    while state["blink_times"] and state["blink_times"][0] < cutoff:
        state["blink_times"].popleft()
        if state["blink_durations"]:
            state["blink_durations"].popleft()
    while state["yawn_times"] and state["yawn_times"][0] < cutoff:
        state["yawn_times"].popleft()


def calculate_metrics(state, microsleep, head_down, head_roll):
    perclos = sum(v for _, v in state["closed_window"]) / len(state["closed_window"]) if state["closed_window"] else 0.0
    blink_rate = len(state["blink_times"])
    slow_blinks = sum(d > SLOW_BLINK_TIME for d in state["blink_durations"])

    if state["ear_window"]:
        ear_vals = np.array([v for _, v in state["ear_window"]], dtype=np.float32)
        ear_std = float(np.std(ear_vals))
    else:
        ear_std = 0.0

    if state["pitch_window"]:
        pitch_vals = np.array([v for _, v in state["pitch_window"]], dtype=np.float32)
        pitch_var = float(np.var(pitch_vals))
    else:
        pitch_var = 0.0

    drowsy_score = min(
        1.0,
        (
            0.30 * perclos
            + 0.20 * int(microsleep)
            + 0.20 * min(slow_blinks / 8, 1.0)
            + 0.15 * min(ear_std / 0.20, 1.0)
            + 0.05 * min(pitch_var / 0.015, 1.0)
            + 0.05 * int(head_down)
            + 0.05 * int(head_roll)
        ),
    )

    return {
        "perclos": perclos,
        "blink_rate": blink_rate,
        "slow_blinks": slow_blinks,
        "ear_std": ear_std,
        "pitch_var": pitch_var,
        "drowsy_score": drowsy_score,
    }


def draw_all_landmarks(frame, landmarks, width, height):
    for idx, lm in enumerate(landmarks):
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
        if idx % 10 == 0:
            cv2.putText(
                frame,
                str(idx),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.25,
                (255, 255, 255),
                1,
            )


def draw_metrics_panel(
    frame,
    metrics,
    face_detected,
    landmark_count,
    ear,
    mar,
    microsleep,
    head_down,
    head_roll,
    yawns,
    detection_score,
    fps,
    min_det_conf,
    min_track_conf,
):
    state_label = "DROWSY" if metrics["drowsy_score"] >= DROWSY_THRESHOLD else "ALERT"
    color = (0, 0, 255) if state_label == "DROWSY" else (0, 255, 0)

    lines = [
        f"Face detected: {face_detected}",
        f"Landmarks drawn: {landmark_count}",
        f"Detection score (30f): {detection_score:.2f}",
        f"FPS: {fps:.1f}",
        f"min_det_conf: {min_det_conf:.2f}",
        f"min_track_conf: {min_track_conf:.2f}",
        f"EAR: {ear:.3f}",
        f"MAR: {mar:.3f}",
        f"PERCLOS: {metrics['perclos']:.3f}",
        f"Blinks(window): {metrics['blink_rate']}",
        f"Slow blinks: {metrics['slow_blinks']}",
        f"Yawns(window): {yawns}",
        f"EAR std: {metrics['ear_std']:.4f}",
        f"Pitch var: {metrics['pitch_var']:.4f}",
        f"Microsleep: {microsleep}",
        f"Head down: {head_down}",
        f"Head roll: {head_roll}",
        f"Drowsy score: {metrics['drowsy_score']:.3f} ({state_label})",
        "Press q to quit",
    ]

    y = 24
    for i, text in enumerate(lines):
        text_color = color if i == len(lines) - 2 else (255, 255, 255)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        y += 20


def run_face_mesh_all_points():
    min_det_conf = 0.50
    min_track_conf = 0.50

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index 0).")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
    )

    state = init_state()
    detect_hist = deque(maxlen=30)
    fps_times = deque(maxlen=30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            now = time.time()
            fps_times.append(now)

            face_detected = bool(results.multi_face_landmarks)
            detect_hist.append(1 if face_detected else 0)
            detection_score = sum(detect_hist) / len(detect_hist)

            ear = 0.0
            mar = 0.0
            microsleep = False
            head_down = False
            head_roll = False
            landmark_count = 0

            if face_detected:
                lm = results.multi_face_landmarks[0].landmark
                landmark_count = len(lm)
                draw_all_landmarks(frame, lm, w, h)

                landmarks_px = [(int(p.x * w), int(p.y * h)) for p in lm]
                ear, microsleep = process_eye_metrics(landmarks_px, state, now)
                mar = process_mouth_metrics(landmarks_px, state, now)
                head_down = process_head_pitch(landmarks_px, state, now)
                head_roll = process_head_roll(landmarks_px, state, now)

            cleanup_windows(state, now)
            metrics = calculate_metrics(state, microsleep, head_down, head_roll)

            if len(fps_times) >= 2:
                elapsed = fps_times[-1] - fps_times[0]
                fps = (len(fps_times) - 1) / elapsed if elapsed > 0 else 0.0
            else:
                fps = 0.0

            draw_metrics_panel(
                frame=frame,
                metrics=metrics,
                face_detected=face_detected,
                landmark_count=landmark_count,
                ear=ear,
                mar=mar,
                microsleep=microsleep,
                head_down=head_down,
                head_roll=head_roll,
                yawns=len(state["yawn_times"]),
                detection_score=detection_score,
                fps=fps,
                min_det_conf=min_det_conf,
                min_track_conf=min_track_conf,
            )

            cv2.imshow("V8 Face Mesh - All Points + Metrics", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_mesh_all_points()
