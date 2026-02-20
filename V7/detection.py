"""
Detection module for Driver Drowsiness Detection System V7.
Handles face landmark processing, drowsiness metric calculation,
overlay drawing, AND background detection thread.

V5: Added DetectionThread for parallel detection during conversation.
V7: Added format_detection_for_llm() — passes raw metrics to the LLM
    with no pre-interpreted severity labels. The LLM reasons about
    drowsiness from raw numbers + baseline context.
"""

import cv2
import math
import time
import threading
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

# Pre-computed threshold in radians (avoids np.degrees conversion per frame)
_ROLL_THRESH_RAD = math.radians(Config.HEAD_ROLL_THRESH)


# =========================
# HELPER FUNCTIONS
# =========================
def eye_aspect_ratio(landmarks, idx):
    """Calculate Eye Aspect Ratio (EAR)."""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    return (math.dist(p2, p6) + math.dist(p3, p5)) / (2.0 * math.dist(p1, p4))


def mouth_aspect_ratio(landmarks):
    """Calculate Mouth Aspect Ratio (MAR)."""
    return math.dist(landmarks[13], landmarks[14]) / math.dist(landmarks[61], landmarks[291])


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
    roll = abs(math.atan2(dy, dx))

    head_roll = False
    if roll > _ROLL_THRESH_RAD:
        if state['head_roll_start'] is None:
            state['head_roll_start'] = now
        elif now - state['head_roll_start'] > Config.ROLL_TIME:
            head_roll = True
    else:
        state['head_roll_start'] = None

    return head_roll


def cleanup_windows(state, now):
    """Remove old entries from time windows."""
    cutoff = now - Config.WINDOW_TIME
    while state['ear_window'] and state['ear_window'][0][0] < cutoff:
        state['ear_window'].popleft()
    while state['pitch_window'] and state['pitch_window'][0][0] < cutoff:
        state['pitch_window'].popleft()
    while state['closed_window'] and state['closed_window'][0][0] < cutoff:
        state['closed_window'].popleft()
    while state['blink_times'] and state['blink_times'][0] < cutoff:
        state['blink_times'].popleft()
        # Keep blink_durations in sync with blink_times (1-to-1 pairing)
        if state['blink_durations']:
            state['blink_durations'].popleft()
    while state['yawn_times'] and state['yawn_times'][0] < cutoff:
        state['yawn_times'].popleft()


def calculate_metrics(state, microsleep, head_down, head_roll):
    """Calculate drowsiness metrics and score."""
    if state['closed_window']:
        perclos = sum(v for _, v in state['closed_window']) / len(state['closed_window'])
    else:
        perclos = 0.0

    blink_rate = len(state['blink_times'])
    slow_blinks = sum(d > Config.SLOW_BLINK_TIME for d in state['blink_durations'])

    if state['ear_window']:
        ear_vals = np.fromiter((v for _, v in state['ear_window']), dtype=np.float32,
                               count=len(state['ear_window']))
        ear_std = float(np.std(ear_vals))
    else:
        ear_std = 0.0

    if state['pitch_window']:
        pitch_vals = np.fromiter((v for _, v in state['pitch_window']), dtype=np.float32,
                                 count=len(state['pitch_window']))
        pitch_var = float(np.var(pitch_vals))
    else:
        pitch_var = 0.0

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


# Landmark indices to visualize (eyes, mouth, nose, brows, jawline subset)
_VIS_LANDMARKS = {
    'eye':   LEFT_EYE + RIGHT_EYE,                       # 12 pts
    'mouth': MOUTH_LANDMARKS,                             # 4 pts
    'head':  [NOSE_TIP, FOREHEAD, CHIN,                   # nose/forehead/chin
              LEFT_EYE_CORNER, RIGHT_EYE_CORNER],         # eye corners for roll
}


def draw_overlay(frame, metrics, ear, mar, microsleep, head_down,
                 head_roll, state, drowsy_state, landmarks=None,
                 proc_size=(320, 240)):
    """Draw metrics overlay and landmark points on the frame."""
    h, w = frame.shape[:2]
    y = 25
    font_scale = 0.5
    thickness = 1

    EYE_COLOR = (255, 0, 0)
    MOUTH_COLOR = (0, 0, 255)
    HEAD_COLOR = (0, 255, 0)

    # ── Draw landmark points ──
    if landmarks:
        sx = w / proc_size[0]
        sy = h / proc_size[1]
        for group, color in [('eye', EYE_COLOR), ('mouth', MOUTH_COLOR),
                              ('head', HEAD_COLOR)]:
            for idx in _VIS_LANDMARKS[group]:
                if idx < len(landmarks):
                    px = int(landmarks[idx][0] * sx)
                    py = int(landmarks[idx][1] * sy)
                    cv2.circle(frame, (px, py), 2, color, -1)

    # ── Draw text overlay ──
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
# BACKGROUND DETECTION THREAD (V5)
# =========================
class DetectionThread:
    """Runs drowsiness detection in a background thread during conversation.

    Creates its OWN camera capture and FaceMesh instance — MediaPipe is NOT
    thread-safe, so we cannot share the main thread's instances. The main
    loop must release its camera BEFORE starting this thread (RPi only has
    one camera device, can't open two handles).

    Uses DETECTION_FRAME_SKIP to reduce CPU during conversation (detection is
    secondary while speech/LLM are active).
    """

    def __init__(self, show_display=False):
        self._show_display = show_display
        # Shared metrics (protected by lock)
        self._lock = threading.Lock()
        self._latest_metrics = {
            'drowsy_score': 0.0,
            'perclos': 0.0,
            'blink_rate': 0,
            'slow_blinks': 0,
            'ear_std': 0.0,
            'pitch_var': 0.0,
        }
        self._microsleep = False
        self._head_down = False

        self._running = False
        self._thread = None

    def _init_state(self):
        """Create a fresh state dict for the detection thread (no sharing)."""
        from collections import deque
        _win = 200  # safety cap; time-based cleanup handles normal pruning
        return {
            'ear_window': deque(maxlen=_win),
            'pitch_window': deque(maxlen=_win),
            'closed_window': deque(maxlen=_win),
            'blink_times': deque(maxlen=50),
            'blink_durations': deque(maxlen=50),
            'yawn_times': deque(maxlen=20),
            'eye_closed_start': None,
            'blink_start': None,
            'yawn_start': None,
            'head_down_start': None,
            'head_roll_start': None,
            'window_start_time': time.time(),
        }

    def start(self):
        """Open own camera + FaceMesh, start background detection thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        print("🔍 Background detection thread started (own camera + FaceMesh)")

    def stop(self):
        """Stop detection, release camera and FaceMesh."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Resources are released inside _detection_loop's finally block
        print("🔍 Background detection thread stopped")

    def get_full_state(self):
        """Return metrics + microsleep/head_down flags (thread-safe)."""
        with self._lock:
            m = self._latest_metrics.copy()
            m['microsleep'] = self._microsleep
            m['head_down'] = self._head_down
            return m

    def _detection_loop(self):
        """Background loop with its own camera and FaceMesh (thread-safe)."""
        import mediapipe as mp

        # Create thread-local resources with retry
        # (camera may not be released instantly by the main thread)
        cap = None
        for attempt in range(5):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            cap.release()
            print(f"🔍 Camera not ready, retrying ({attempt + 1}/5)...")
            time.sleep(0.5)

        if cap is None or not cap.isOpened():
            print("⚠️ DetectionThread: failed to open camera after 5 attempts — running without live detection")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        state = self._init_state()
        frame_count = 0
        frame_skip = Config.DETECTION_FRAME_SKIP
        # Landmarks are normalized to the processed frame size (320x240),
        # NOT the capture resolution — must scale to processed dimensions.
        PROC_W, PROC_H = 320, 240

        try:
            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                now = time.time()

                # Process at reduced resolution for speed
                small_frame = cv2.resize(frame, (PROC_W, PROC_H))
                rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                microsleep = False
                head_down = False
                head_roll = False
                ear = 0
                mar = 0
                landmarks = None

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    landmarks = [(int(p.x * PROC_W), int(p.y * PROC_H)) for p in lm]

                    ear, microsleep = process_eye_metrics(landmarks, state, now)
                    mar = process_mouth_metrics(landmarks, state, now)
                    head_down = process_head_pitch(landmarks, state, now)
                    head_roll = process_head_roll(landmarks, state, now)

                cleanup_windows(state, now)
                metrics = calculate_metrics(state, microsleep, head_down, head_roll)

                # Update shared metrics (only thing accessed cross-thread)
                with self._lock:
                    self._latest_metrics = metrics
                    self._microsleep = microsleep
                    self._head_down = head_down

                # Show camera overlay if display mode is on
                if self._show_display:
                    drowsy_state = "DROWSY" if metrics['drowsy_score'] > Config.DROWSY_THRESHOLD else "ALERT"
                    draw_overlay(frame, metrics, ear, mar, microsleep, head_down,
                                 head_roll, state, drowsy_state, landmarks=landmarks,
                                 proc_size=(PROC_W, PROC_H))
                    cv2.imshow("Driver Drowsiness Monitor", frame)
                    cv2.waitKey(1)

                # Throttle to target FPS
                time.sleep(1.0 / Config.DETECTION_THREAD_FPS)

        finally:
            # Always release thread-local resources
            cap.release()
            face_mesh.close()
            if self._show_display:
                try:
                    cv2.destroyWindow("Driver Drowsiness Monitor")
                except Exception:
                    pass


# =========================
# RAW METRICS FORMATTING FOR LLM (V7)
# =========================
def format_detection_for_llm(metrics, microsleep=False, head_down=False):
    """Format detection metrics as a compact raw-number string for LLM injection.

    V7: No severity labels or threshold interpretation. The LLM gets raw
    numbers and reasons about severity itself using the system prompt's
    metric explanations + the driver's personal baselines.

    Args:
        metrics: dict from calculate_metrics() or DetectionThread.get_full_state()
        microsleep: bool — True if eyes closed > 1.5s
        head_down: bool — True if head tilted down for extended period

    Returns:
        Single-line string like:
        "DETECTION: score=0.62, perclos=0.18, blinks=4, slow_blinks=2, ..."
    """
    parts = [
        f"score={metrics.get('drowsy_score', 0):.3f}",
        f"perclos={metrics.get('perclos', 0):.3f}",
        f"blink_rate={metrics.get('blink_rate', 0)}",
        f"slow_blinks={metrics.get('slow_blinks', 0)}",
        f"ear_std={metrics.get('ear_std', 0):.4f}",
        f"pitch_var={metrics.get('pitch_var', 0):.5f}",
        f"microsleep={microsleep}",
        f"head_down={head_down}",
    ]
    return "DETECTION: " + ", ".join(parts)
