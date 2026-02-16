"""
Detection functions for Driver Drowsiness Detection System V3.2
Handles face landmark processing, drowsiness metric calculation, and overlay drawing.
"""

import cv2
import numpy as np

from config import Config

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
