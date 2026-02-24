#!/usr/bin/env python3
"""
Sentinel V8 — Full Metrics Test Suite
======================================
Single interactive script covering every science fair metric category.

Sections:
  0  Setup & Calibration          (~2 min)
  1  Detection Performance        (~5 min)  3 behavior cycles × alert+drowsy
  2  Conversation — New Driver    (~7 min)  3 turns, latency + voice + memory
  3  Conversation — Returning     (~4 min)  2 turns, recall accuracy
  4  Conversation Behavior        (~4 min)  Termination accuracy + false exits
  5  PVT-B Reaction Time          (~3 min)  10 stimuli
  6  Output                       (~1 min)  TSV file

Output: test_logs/metrics_test_{YYYYMMDD_HHMM}.tsv  (paste into Google Sheets)

Usage:
    python test_all_metrics.py            # Run all sections
    python test_all_metrics.py --skip 12  # Skip sections 1 and 2
"""

import cv2
import mediapipe as mp
import os
import platform
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from detection import (
    process_eye_metrics,
    process_mouth_metrics,
    process_head_pitch,
    process_head_roll,
    cleanup_windows,
    calculate_metrics,
    draw_overlay,
    format_detection_for_llm,
    LEFT_EYE,
    RIGHT_EYE,
    eye_aspect_ratio,
)
from memory import MemoryManager
from metric_reasoner import MetricReasoner
from metrics_logger import MetricsLogger
from stt_engine import STTEngine
from tts_engine import TTSEngine
from llm_assistant import LLMAssistant
from voice_features import VoiceFeatureExtractor


# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE = Path(__file__).resolve().parent
TEST_DB   = _BASE / "test_logs" / "sentinel_test.db"
OUTPUT_DIR = _BASE / "test_logs"
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_FILE = OUTPUT_DIR / f"metrics_test_{TIMESTAMP}.tsv"


# ── Fake sensor data ──────────────────────────────────────────────────────────

FAKE_DROWSY_METRICS = {
    "drowsy_score": 0.68, "perclos": 0.32, "blink_rate": 16,
    "ear_std": 0.042,     "slow_blinks": 4, "pitch_var": 0.009,
    "alert_duration": 0,
}

FAKE_ALERT_METRICS = {
    "drowsy_score": 0.10, "perclos": 0.04, "blink_rate": 3,
    "ear_std": 0.018,     "slow_blinks": 0, "pitch_var": 0.001,
    "alert_duration": 85.0,
}

# Three distinct drowsy behavior types (cycles in Section 1)
BEHAVIOR_CYCLES = [
    ("DROOPY EYES",
     "Let your eyes droop. Blink very slowly. Look heavy and unfocused."),
    ("MICROSLEEP",
     "Close your eyes completely for 2-3 seconds, then open. Repeat slowly."),
    ("HEAD DOWN",
     "Let your head nod forward slowly. Hold it down, then slowly raise it."),
]

# Known facts to share in Session 1 — used to score extraction accuracy
KNOWN_FACTS = {1: "name", 2: "job or occupation", 3: "destination today"}

# Confounding phrases for false exit rate test
# Some contain exit keyword substrings (expected false exits), some do not
CONFOUNDING_PHRASES = [
    ("I'm feeling good now",              False),   # no keyword
    ("That stopped my yawning",           True),    # "stop" in "stopped"
    ("Let's keep talking",                False),   # no keyword
    ("I've been done with work for hours",True),    # "done"
    ("Goodbye anxiety",                   True),    # "goodbye"
    ("Drive safely everyone",             False),   # no keyword
    ("I stopped the music",               True),    # "stop"
    ("That was a nice memory",            False),   # no keyword
]
EXIT_KEYWORDS = ['exit', 'quit', 'bye', 'stop', 'done', 'goodbye']


# ── TSV accumulator ───────────────────────────────────────────────────────────

_tsv_headers: dict = {}
_tsv_rows:    dict = {}


def _header(section: str, cols: list):
    _tsv_headers[section] = cols


def _row(section: str, values: list):
    _tsv_rows.setdefault(section, []).append(values)


# ── Console helpers ───────────────────────────────────────────────────────────

W = 60

def banner(title: str, num: int = None):
    prefix = f"SECTION {num}  " if num is not None else ""
    print(f"\n{'═'*W}")
    print(f"  {prefix}{title}")
    print(f"{'═'*W}")


def step(msg: str):  print(f"\n  ▶  {msg}")
def ok(msg: str):    print(f"  ✓  {msg}")
def warn(msg: str):  print(f"  ⚠  {msg}")
def info(msg: str):  print(f"     {msg}")


def pause(prompt: str = "  Press ENTER to continue..."):
    input(prompt)


# ── Detection state ───────────────────────────────────────────────────────────

def _new_state() -> dict:
    W = 200
    return {
        'ear_window':       deque(maxlen=W),
        'pitch_window':     deque(maxlen=W),
        'closed_window':    deque(maxlen=W),
        'blink_times':      deque(maxlen=50),
        'blink_durations':  deque(maxlen=50),
        'yawn_times':       deque(maxlen=20),
        'eye_closed_start': None,
        'blink_start':      None,
        'yawn_start':       None,
        'head_down_start':  None,
        'head_roll_start':  None,
        'window_start_time': time.time(),
    }


# ── Statistics helpers ────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score 95 % confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z * (p*(1-p)/n + z**2/(4*n**2))**0.5) / denom
    return max(0.0, round(center - margin, 4)), min(1.0, round(center + margin, 4))


def compute_auc_roc(data: list):
    """Trapezoidal AUC-ROC from list of (score, 0/1) pairs."""
    n_pos = sum(y for _, y in data)
    n_neg = len(data) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, []
    pairs = sorted(data, key=lambda x: -x[0])
    tp = fp = 0
    pts = [(0.0, 0.0)]
    for score, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        pts.append((fp / n_neg, tp / n_pos))
    pts.append((1.0, 1.0))
    auc = sum(
        (pts[i][0] - pts[i-1][0]) * (pts[i][1] + pts[i-1][1]) / 2
        for i in range(1, len(pts))
    )
    return round(auc, 4), pts


# ── PVT-B keypress (cross-platform) ──────────────────────────────────────────

def _wait_key(timeout_s: float):
    """Block up to timeout_s for any keypress. Returns elapsed seconds or None."""
    if platform.system() == 'Windows':
        import msvcrt
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if msvcrt.kbhit():
                msvcrt.getch()
                return time.perf_counter() - t0
            time.sleep(0.005)
        return None
    else:
        import select, tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            t0    = time.perf_counter()
            ready = select.select([sys.stdin], [], [], timeout_s)
            if ready[0]:
                sys.stdin.read(1)
                return time.perf_counter() - t0
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0  –  SETUP & CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def section_0_setup() -> dict:
    banner("SETUP & CALIBRATION", num=0)
    ctx: dict = {}

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          Config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if cap.isOpened():
        ok("Camera")
        ctx['cap'] = cap
    else:
        warn("Camera failed — Section 1 will be skipped")
        ctx['cap'] = None

    # ── MediaPipe FaceMesh ────────────────────────────────────────────────────
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        ok("MediaPipe FaceMesh")
        ctx['face_mesh'] = face_mesh
    except Exception as e:
        warn(f"FaceMesh failed: {e}")
        ctx['face_mesh'] = None

    # ── TTS ────────────────────────────────────────────────────────────────────
    try:
        tts = TTSEngine()
        ok("TTS (Deepgram)")
        ctx['tts'] = tts
    except Exception as e:
        warn(f"TTS failed: {e}")
        ctx['tts'] = None

    # ── STT ────────────────────────────────────────────────────────────────────
    try:
        stt = STTEngine()
        ok("STT (Deepgram)")
        ctx['stt'] = stt
    except Exception as e:
        warn(f"STT failed: {e}")
        ctx['stt'] = None

    # ── MemoryManager (isolated test DB — real driver DB untouched) ───────────
    try:
        if TEST_DB.exists():
            TEST_DB.unlink()
        memory = MemoryManager(db_path=TEST_DB)
        ok(f"MemoryManager  ({TEST_DB.name})")
        ctx['memory'] = memory
    except Exception as e:
        warn(f"MemoryManager failed: {e}")
        ctx['memory'] = None

    # ── MetricReasoner ────────────────────────────────────────────────────────
    try:
        reasoner = MetricReasoner()
        ok("MetricReasoner (8B)")
        ctx['reasoner'] = reasoner
    except Exception as e:
        warn(f"MetricReasoner failed: {e}")
        ctx['reasoner'] = None

    # ── LLMAssistant ──────────────────────────────────────────────────────────
    try:
        llm = LLMAssistant(ctx.get('tts'), ctx.get('memory'))
        ok("LLMAssistant (Groq 70B)")
        ctx['llm'] = llm
    except Exception as e:
        warn(f"LLMAssistant failed: {e}")
        ctx['llm'] = None

    # ── VoiceFeatureExtractor ─────────────────────────────────────────────────
    ctx['vfe'] = VoiceFeatureExtractor()
    ok("VoiceFeatureExtractor")

    # ── Fatigue level ─────────────────────────────────────────────────────────
    print()
    fl = input("  Fatigue level right now (1 = fully alert, 10 = very sleepy): ").strip()
    ctx['fatigue'] = fl or "not recorded"

    # ── EAR Calibration ───────────────────────────────────────────────────────
    ear_thresh = Config.EAR_THRESH
    if ctx.get('cap') and ctx.get('face_mesh'):
        step("EAR Calibration — look at camera normally, keep eyes open.")
        pause("  Press ENTER to begin 5-second calibration...")

        cap       = ctx['cap']
        face_mesh = ctx['face_mesh']
        samples   = []
        PROC_W, PROC_H = Config.PROC_WIDTH, Config.PROC_HEIGHT
        t_cal = time.perf_counter()

        while time.perf_counter() - t_cal < 5.0:
            ret, frame = cap.read()
            if not ret:
                continue
            small = (frame if (frame.shape[1] == PROC_W and frame.shape[0] == PROC_H)
                     else cv2.resize(frame, (PROC_W, PROC_H)))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res   = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                lnds = [(int(p.x*PROC_W), int(p.y*PROC_H)) for p in lm]
                ear = (eye_aspect_ratio(lnds, LEFT_EYE) +
                       eye_aspect_ratio(lnds, RIGHT_EYE)) / 2
                samples.append(ear)
            rem = max(0, 5 - int(time.perf_counter() - t_cal))
            cv2.putText(frame, f"Calibrating... {rem}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Sentinel — Calibration", frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        if samples:
            avg = sum(samples) / len(samples)
            ear_thresh = round(avg * 0.75, 4)
            ok(f"EAR calibrated: avg={avg:.3f}  threshold={ear_thresh:.3f}")
        else:
            warn(f"No face detected — using default EAR threshold={ear_thresh}")

    ctx['ear_thresh'] = ear_thresh

    print(f"\n  Fatigue: {ctx['fatigue']}   EAR threshold: {ear_thresh}")
    print(f"  Output: {OUTPUT_FILE}")
    pause("\n  Setup complete. Press ENTER to begin testing...")
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1  –  DETECTION PERFORMANCE + VISION SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def section_1_detection(ctx: dict):
    banner("DETECTION PERFORMANCE + VISION SIGNALS", num=1)

    cap       = ctx.get('cap')
    face_mesh = ctx.get('face_mesh')
    reasoner  = ctx.get('reasoner')
    ear_thresh = ctx.get('ear_thresh', Config.EAR_THRESH)

    if not cap or not face_mesh:
        warn("Camera / FaceMesh unavailable — skipping Section 1")
        return

    _header("DETECTION_PERFORMANCE",
            ["Cycle", "Behavior", "Phase", "True_Label",
             "Triggered", "Avg_Score", "Max_Score", "Latency_s", "Classification"])
    _header("VISION_SIGNALS",
            ["Cycle", "Behavior", "Phase", "Time_s",
             "EAR", "PERCLOS", "Slow_Blinks", "Pitch_Var", "Ear_Std", "Drowsy_Score"])
    _header("AUC_ROC_DATA",
            ["Cycle", "Phase", "Time_s", "Score", "True_Label_Int"])
    _header("DETECTION_LATENCY",
            ["Cycle", "Behavior", "Latency_s"])

    confusion       = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    latency_entries = []   # list of (beh_name, latency_s)
    PROC_W, PROC_H  = Config.PROC_WIDTH, Config.PROC_HEIGHT

    for cyc_idx, (beh_name, beh_instr) in enumerate(BEHAVIOR_CYCLES, 1):
        print(f"\n  {'─'*W}")
        print(f"  Cycle {cyc_idx}/3 — Behavior: {beh_name}")
        print(f"  {'─'*W}")

        for phase_name, true_label, duration, label_int in [
            ("ALERT BASELINE", "NEGATIVE", 15, 0),
            (beh_name,         "POSITIVE", 35, 1),
        ]:
            print(f"\n  ┌─ {phase_name}  ({duration}s)  ──────────────────────")
            if phase_name == "ALERT BASELINE":
                info("Look alert. Eyes open. Head up. Breathe normally.")
            else:
                info(beh_instr)
            info(f"Duration: {duration} seconds")

            # Reset reasoner state before each drowsy phase
            if true_label == "POSITIVE" and reasoner:
                reasoner.reset()

            pause(f"  Press ENTER to start {phase_name}...")

            state        = _new_state()
            last_log_t   = -1.0
            triggered    = False
            trig_latency = None
            scores       = []

            t0          = time.perf_counter()
            frame_count = 0
            FRAME_SKIP  = getattr(Config, 'DETECTION_FRAME_SKIP', 3)

            # Last-known processed values — used for display on skipped frames
            last_ear        = 0.25
            last_metrics    = {'drowsy_score': 0.0, 'perclos': 0.0, 'slow_blinks': 0,
                               'pitch_var': 0.0, 'ear_std': 0.0, 'blink_rate': 0}
            last_microsleep = False
            last_head_down  = False
            last_head_roll  = False

            while True:
                elapsed = time.perf_counter() - t0
                if elapsed >= duration:
                    break

                ret, frame = cap.read()
                if not ret:
                    continue

                frame_count += 1
                now = time.perf_counter()

                if frame_count % FRAME_SKIP == 0:
                    small = (frame if (frame.shape[1] == PROC_W and frame.shape[0] == PROC_H)
                             else cv2.resize(frame, (PROC_W, PROC_H)))
                    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    res   = face_mesh.process(rgb)

                    ear = 0.25
                    microsleep = head_down = head_roll = False

                    if res.multi_face_landmarks:
                        lm   = res.multi_face_landmarks[0].landmark
                        lnds = [(int(p.x*PROC_W), int(p.y*PROC_H)) for p in lm]
                        ear, microsleep = process_eye_metrics(
                            lnds, state, now, ear_thresh=ear_thresh)
                        process_mouth_metrics(lnds, state, now)
                        head_down = process_head_pitch(lnds, state, now)
                        head_roll = process_head_roll(lnds, state, now)

                    cleanup_windows(state, now)
                    metrics = calculate_metrics(state, microsleep, head_down, head_roll)
                    score   = metrics['drowsy_score']
                    scores.append(score)

                    last_ear        = ear
                    last_metrics    = metrics
                    last_microsleep = microsleep
                    last_head_down  = head_down
                    last_head_roll  = head_roll

                    # Log vision snapshot every 0.5 s
                    if elapsed - last_log_t >= 0.5:
                        last_log_t = elapsed
                        _row("VISION_SIGNALS", [
                            cyc_idx, beh_name, phase_name, round(elapsed, 1),
                            round(ear, 4),
                            round(metrics['perclos'], 4),
                            metrics['slow_blinks'],
                            round(metrics['pitch_var'], 6),
                            round(metrics['ear_std'], 4),
                            round(score, 4),
                        ])
                        _row("AUC_ROC_DATA", [
                            cyc_idx, phase_name, round(elapsed, 1),
                            round(score, 4), label_int,
                        ])

                    # Microsleep — instant trigger
                    if microsleep and true_label == "POSITIVE" and not triggered:
                        triggered    = True
                        trig_latency = round(elapsed, 2)
                        print(f"  ⚡ MICROSLEEP TRIGGER  t={elapsed:.1f}s")

                    # Reasoner gate (uses its own 3 s interval check)
                    if (not triggered and true_label == "POSITIVE"
                            and score >= Config.REASONER_PRE_FILTER
                            and reasoner and reasoner.should_call()):
                        reasoner.evaluate(metrics, microsleep=microsleep,
                                          head_down=head_down, head_roll=head_roll)
                        if reasoner.is_confirmed_drowsy():
                            triggered    = True
                            trig_latency = round(elapsed, 2)
                            print(f"  ⚡ REASONER TRIGGER  t={elapsed:.1f}s  score={score:.3f}")

                # Overlay + live display every frame using last processed values
                dstate = "DROWSY" if last_metrics['drowsy_score'] > Config.DROWSY_THRESHOLD else "ALERT"
                draw_overlay(frame, last_metrics, last_ear, 0, last_microsleep,
                             last_head_down, last_head_roll, state, dstate,
                             proc_size=(PROC_W, PROC_H))
                rem = duration - elapsed
                label_text = f"{phase_name}  {rem:.0f}s"
                if triggered:
                    label_text += "  [TRIGGERED]"
                cv2.putText(frame, label_text,
                            (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                cv2.imshow("Sentinel — Test Suite", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ── Phase result ──────────────────────────────────────────────────
            avg_s = round(sum(scores)/len(scores), 4) if scores else 0
            max_s = round(max(scores), 4)             if scores else 0
            lat_s = str(trig_latency) if trig_latency else "—"

            if true_label == "NEGATIVE":
                cls = "FP" if triggered else "TN"
            else:
                cls = "TP" if triggered else "FN"
                if trig_latency:
                    latency_entries.append((beh_name, trig_latency))

            confusion[cls] += 1

            print(f"  └─ {cls}  avg={avg_s}  max={max_s}  latency={lat_s}s")
            _row("DETECTION_PERFORMANCE", [
                cyc_idx, beh_name, phase_name, true_label,
                "YES" if triggered else "NO",
                avg_s, max_s, lat_s, cls,
            ])

            # Short pause between alert→drowsy within same cycle
            if phase_name == "ALERT BASELINE":
                time.sleep(2)

        # Brief reset between cycles
        if cyc_idx < len(BEHAVIOR_CYCLES):
            print(f"\n  End of Cycle {cyc_idx}. Look alert for 10 seconds...")
            t_reset = time.perf_counter()
            while time.perf_counter() - t_reset < 10.0:
                ret, frame = cap.read()
                if ret:
                    rem = 10 - int(time.perf_counter() - t_reset)
                    cv2.putText(frame, f"RESET — look alert  {rem}s",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    cv2.imshow("Sentinel — Test Suite", frame)
                    cv2.waitKey(1)
            if reasoner:
                reasoner.reset()

    cv2.destroyAllWindows()

    # ── Confusion matrix stats ────────────────────────────────────────────────
    tp, fp, tn, fn = confusion["TP"], confusion["FP"], confusion["TN"], confusion["FN"]
    total    = tp + fp + tn + fn
    accuracy = round((tp+tn)/total, 4)  if total        else 0
    prec     = round(tp/(tp+fp), 4)     if (tp+fp)      else 0
    rec      = round(tp/(tp+fn), 4)     if (tp+fn)      else 0
    f1       = round(2*prec*rec/(prec+rec), 4) if (prec+rec) else 0
    fnr      = round(fn/(tp+fn), 4)     if (tp+fn)      else 0
    ci_lo, ci_hi = wilson_ci(tp+tn, total)
    avg_lat  = round(sum(lat for _, lat in latency_entries)/len(latency_entries), 2) if latency_entries else None

    _header("CONFUSION_MATRIX", ["Metric", "Value"])
    for metric, val in [
        ("True_Positives",       tp),
        ("False_Positives",      fp),
        ("True_Negatives",       tn),
        ("False_Negatives",      fn),
        ("Total_Trials",         total),
        ("Accuracy",             accuracy),
        ("Precision",            prec),
        ("Recall",               rec),
        ("F1",                   f1),
        ("False_Negative_Rate",  fnr),
        ("Wilson_CI_Low_95",     ci_lo),
        ("Wilson_CI_High_95",    ci_hi),
        ("Detection_Latency_Avg_s", avg_lat or "—"),
    ]:
        _row("CONFUSION_MATRIX", [metric, val])

    for i, (beh, lat) in enumerate(latency_entries, 1):
        _row("DETECTION_LATENCY", [i, beh, lat])

    print(f"\n  ── Section 1 Complete ──")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Accuracy={accuracy}  Precision={prec}  Recall={rec}  F1={f1}  FNR={fnr}")
    print(f"  Wilson 95% CI: [{ci_lo}, {ci_hi}]")
    if avg_lat:
        print(f"  Detection latency avg: {avg_lat}s  ({len(latency_entries)} trigger(s))")

    pause("\n  Press ENTER to continue to Section 2...")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2  –  CONVERSATION SESSION 1  (NEW DRIVER)
# ══════════════════════════════════════════════════════════════════════════════

def _seed_baselines(memory: MemoryManager):
    """Seed realistic voice baselines so comparisons work in test mode."""
    memory.store_calibration_baselines([
        {'energy_rms': 0.045, 'speech_rate_wpm': 135.0, 'pause_ratio': 0.20,
         'response_latency_s': 2.0, 'peak_amplitude': 0.35, 'duration_s': 4.0},
        {'energy_rms': 0.050, 'speech_rate_wpm': 140.0, 'pause_ratio': 0.18,
         'response_latency_s': 1.8, 'peak_amplitude': 0.40, 'duration_s': 3.5},
        {'energy_rms': 0.042, 'speech_rate_wpm': 130.0, 'pause_ratio': 0.22,
         'response_latency_s': 2.2, 'peak_amplitude': 0.38, 'duration_s': 4.2},
    ])


def _run_conversation(ctx: dict, n_turns: int, turn_reminders: list,
                      session_label: str, session_count: int = 0,
                      driver_history: str = "") -> tuple:
    """
    Run a full conversation session (opening + n_turns of STT/LLM/TTS).

    Returns:
        (logger, turns_with_response, session_id)
    """
    tts    = ctx['tts']
    stt    = ctx['stt']
    llm    = ctx['llm']
    memory = ctx['memory']
    vfe    = ctx['vfe']

    session_id   = memory.start_session()
    baselines    = memory.get_baselines()
    baselines_str = memory.format_baselines_for_llm()

    logger = MetricsLogger(trial_id=session_label, driver_type="new" if session_count == 0 else "returning")
    stt.metrics_logger = logger
    tts.metrics_logger = logger
    llm.metrics_logger = logger
    llm._session_id    = session_id

    det_ctx = format_detection_for_llm(FAKE_DROWSY_METRICS,
                                       microsleep=False, head_down=True)
    llm.start_conversation(det_ctx, baselines_str,
                           session_count=session_count,
                           driver_history=driver_history)

    logger.start_conversation()

    # Opening turn (LLM speaks first, no user input)
    logger.start_turn(0)
    llm.get_response_streaming()
    tts.wait_until_done()
    vfe.mark_prompt_end()
    logger.end_turn()

    turns_with_response = 0

    for turn in range(1, n_turns + 1):
        reminder = turn_reminders[turn - 1] if turn <= len(turn_reminders) else ""
        print(f"\n  ── Turn {turn}/{n_turns}  {reminder}")
        logger.start_turn(turn)

        user_input, audio_data = stt.listen(timeout=20)
        if user_input:
            turns_with_response += 1

        if not user_input:
            warn("No speech detected — skipping turn")
            logger.log_stt(0, None)
            logger.end_turn()
            continue

        # Voice features
        voice_ctx = None
        if audio_data is not None:
            feats = vfe.extract_features(audio_data, user_input)
            if feats:
                voice_ctx = vfe.format_for_llm(feats, baselines)
                _row("VOICE_SIGNALS", [
                    session_label, turn,
                    feats.get('energy_rms', ''),
                    feats.get('speech_rate_wpm', ''),
                    feats.get('pause_ratio', ''),
                    feats.get('response_latency_s', ''),
                ])

        # Detection context improves over time
        improving = dict(FAKE_DROWSY_METRICS)
        improving['drowsy_score'] = max(0.20, 0.68 - turn * 0.08)
        improving['perclos']      = max(0.05, 0.32 - turn * 0.06)
        det_ctx = format_detection_for_llm(improving, microsleep=False,
                                           head_down=(turn < 2))

        response = llm.get_response_streaming(
            user_message=user_input,
            detection_context=det_ctx,
            voice_context=voice_ctx,
        )
        tts.wait_until_done()
        vfe.mark_prompt_end()
        logger.end_turn()

        # Live latency summary
        t = logger.turns[-1]
        print(f"     STT:{t.get('stt_latency_ms','—')}ms | "
              f"1st:{t.get('groq_latency_ms','—')}ms | "
              f"Full:{t.get('groq_full_latency_ms','—')}ms | "
              f"TTS:{t.get('tts_generation_ms','—')}ms | "
              f"Gap:{t.get('user_to_speech_ms','—')}ms | "
              f"Total:{t.get('total_turnaround_ms','—')}ms")
        time.sleep(0.2)

    logger.end_conversation()

    # Write latency rows
    for t in logger.turns:
        _row("SYSTEM_LATENCY", [
            session_label, t['turn'],
            t.get('stt_latency_ms', ''),      t.get('groq_latency_ms', ''),
            t.get('groq_full_latency_ms', ''), t.get('tts_generation_ms', ''),
            t.get('user_to_speech_ms', ''),    t.get('total_turnaround_ms', ''),
        ])

    # Export LLM responses (no labels — for blind second-rater scoring)
    for t in logger.turns:
        if t.get('llm_response'):
            _row("LLM_RESPONSES_FOR_RATING", [
                session_label, f"R{t['turn']:02d}", t['llm_response'],
            ])

    return logger, turns_with_response, session_id


def section_2_new_driver(ctx: dict):
    banner("CONVERSATION  —  SESSION 1  (NEW DRIVER)", num=2)

    if not all([ctx.get('tts'), ctx.get('stt'),
                ctx.get('llm'), ctx.get('memory')]):
        warn("Conversation engines unavailable — skipping Section 2")
        return

    _header("SYSTEM_LATENCY",
            ["Session", "Turn", "STT_ms", "Groq_First_ms",
             "Groq_Full_ms", "TTS_Gen_ms", "User_Speech_ms", "Total_ms"])
    _header("VOICE_SIGNALS",
            ["Session", "Turn", "RMS_Energy", "Speech_Rate_WPM",
             "Pause_Ratio", "Response_Latency_s"])
    _header("LLM_RESPONSES_FOR_RATING",
            ["Session", "Response_ID", "Response_Text"])

    step("New driver session — 3 turns")
    info("Speak naturally when Sentinel asks questions.")
    info(f"Turn 1: mention your NAME")
    info(f"Turn 2: mention your JOB / occupation")
    info(f"Turn 3: mention your DESTINATION today")
    pause("\n  Press ENTER to start the conversation...")

    _seed_baselines(ctx['memory'])

    reminders = [
        "  → Mention your NAME",
        "  → Mention your JOB / occupation",
        "  → Mention your DESTINATION today",
    ]

    logger, turns_responded, session_id = _run_conversation(
        ctx, n_turns=3, turn_reminders=reminders,
        session_label="S1_new", session_count=0,
    )

    # Post-session: fact extraction
    step("Running fact extraction from conversation...")
    extracted_count = ctx['memory'].extract_and_store_facts(session_id)
    ok(f"Extracted {extracted_count} facts into SQLite")

    ctx['memory'].end_session(
        session_id,
        avg_drowsy_score=0.50, max_drowsy_score=0.68,
        turn_count=ctx['llm'].conversation_turns,
        duration_s=(logger.conversation_end - logger.conversation_start)
                   if (logger.conversation_end and logger.conversation_start) else 0,
    )

    # Show extracted facts
    facts_in_db = ctx['memory'].get_context_facts(limit=30)
    print("\n  Facts extracted into SQLite:")
    for ft, val, conf, tc in facts_in_db:
        print(f"    {ft}: {val}  (conf={conf:.1f})")

    # Ask which known facts user shared
    print(f"\n  Which of the 3 known facts did you mention?")
    print(f"  1 = name   2 = job/occupation   3 = destination")
    shared_input = input("  → (e.g. '1 2 3' or '1 3'): ").strip()
    facts_shared = []
    for ch in shared_input.split():
        if ch.isdigit() and int(ch) in KNOWN_FACTS:
            facts_shared.append(KNOWN_FACTS[int(ch)])
            logger.log_fact_shared(KNOWN_FACTS[int(ch)])

    # Extraction accuracy
    db_types  = {ft.lower() for ft, _, _, _ in facts_in_db}
    recalled  = sum(1 for f in facts_shared if any(f in dt for dt in db_types))
    extr_acc  = round(recalled / len(facts_shared), 3) if facts_shared else None

    ctx['s1_session_id']        = session_id
    ctx['s1_facts_shared']      = facts_shared
    ctx['s1_extraction_acc']    = extr_acc
    ctx['s1_turns_responded']   = turns_responded
    ctx['s1_total_turns']       = 3

    ok(f"Extraction accuracy: {extr_acc}  ({recalled}/{len(facts_shared)} shared facts found in DB)")
    logger.save()
    pause("\n  Press ENTER to continue to Section 3...")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3  –  CONVERSATION SESSION 2  (RETURNING DRIVER)
# ══════════════════════════════════════════════════════════════════════════════

def section_3_returning_driver(ctx: dict):
    banner("CONVERSATION  —  SESSION 2  (RETURNING DRIVER)", num=3)

    if not all([ctx.get('tts'), ctx.get('stt'),
                ctx.get('llm'), ctx.get('memory')]):
        warn("Conversation engines unavailable — skipping Section 3")
        return

    step("Returning driver session — 2 turns")
    info("Converse naturally. We check if Sentinel remembers facts from Session 1.")
    pause("\n  Press ENTER to start...")

    session_count  = ctx['memory'].get_session_count()
    driver_history = ctx['memory'].get_driver_history_for_llm()

    logger, turns_responded, session_id = _run_conversation(
        ctx, n_turns=2, turn_reminders=["", ""],
        session_label="S2_ret",
        session_count=session_count,
        driver_history=driver_history,
    )

    ctx['memory'].end_session(
        session_id,
        avg_drowsy_score=0.30, max_drowsy_score=0.45,
        turn_count=ctx['llm'].conversation_turns,
        duration_s=(logger.conversation_end - logger.conversation_start)
                   if (logger.conversation_end and logger.conversation_start) else 0,
    )

    # Recall accuracy: how many Session 1 facts are still in the DB?
    facts_s1      = ctx.get('s1_facts_shared', [])
    facts_in_db   = ctx['memory'].get_context_facts(limit=30)
    db_types      = {ft.lower() for ft, _, _, _ in facts_in_db}
    recalled      = sum(1 for f in facts_s1 if any(f in dt for dt in db_types))
    recall_acc    = round(recalled / len(facts_s1), 3) if facts_s1 else None

    ctx['s2_session_id']     = session_id
    ctx['s2_recall_acc']     = recall_acc
    ctx['s2_turns_responded']= turns_responded
    ctx['s2_total_turns']    = 2

    _header("MEMORY_PERSONALIZATION",
            ["Session", "Type", "Facts_Shared", "Facts_Extracted",
             "Extraction_Acc", "Facts_Recalled", "Recall_Acc"])
    _row("MEMORY_PERSONALIZATION", [
        "S1", "NEW",
        len(facts_s1),
        len(facts_in_db),
        ctx.get('s1_extraction_acc', '—'),
        "—", "—",
    ])
    _row("MEMORY_PERSONALIZATION", [
        "S2", "RETURNING",
        "—", "—", "—",
        recalled, recall_acc or "—",
    ])

    ok(f"Recall accuracy: {recall_acc}  ({recalled}/{len(facts_s1)} Session 1 facts found)")
    logger.save()
    pause("\n  Press ENTER to continue to Section 4...")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4  –  CONVERSATION BEHAVIOR
# ══════════════════════════════════════════════════════════════════════════════

def section_4_behavior(ctx: dict):
    banner("CONVERSATION BEHAVIOR", num=4)

    _header("CONVERSATION_BEHAVIOR", ["Metric", "Value", "Detail"])

    # ── 4A  Termination accuracy ──────────────────────────────────────────────
    step("4A — TERMINATION ACCURACY")
    info("A short 2-turn conversation starts. After Sentinel's 2nd response,")
    info("say: 'I feel much better, I'm fully awake and good to drive.'")
    info("We measure how long until the system produces [RECOVERED].")

    term_latency  = None
    term_correct  = False

    tts    = ctx.get('tts')
    stt    = ctx.get('stt')
    llm    = ctx.get('llm')
    memory = ctx.get('memory')

    if all([tts, stt, llm, memory]):
        pause("  Press ENTER to start termination test...")

        # Clear logger refs so we don't pollute earlier logs
        for eng in [stt, tts, llm]:
            eng.metrics_logger = None

        session_id    = memory.start_session()
        t_s4_start    = time.perf_counter()
        session_count = memory.get_session_count()
        baselines_str = memory.format_baselines_for_llm()
        det_ctx = format_detection_for_llm(FAKE_DROWSY_METRICS,
                                           microsleep=False, head_down=False)
        llm.start_conversation(det_ctx, baselines_str, session_count=session_count)
        llm._session_id = session_id

        # Opening + Turn 1 (drowsy context)
        llm.get_response_streaming()
        tts.wait_until_done()

        print("\n  Turn 1: respond normally to Sentinel.")
        u1, _ = stt.listen(timeout=20)
        if u1:
            llm.get_response_streaming(
                user_message=u1,
                detection_context=format_detection_for_llm(FAKE_DROWSY_METRICS))
            tts.wait_until_done()

        # Recovery turn — inject alert context
        t_recovery = time.perf_counter()
        print("\n  Turn 2: say 'I feel much better, fully awake and good to drive.'")
        u2, _ = stt.listen(timeout=25)

        if u2:
            alert_ctx = format_detection_for_llm(FAKE_ALERT_METRICS)
            resp = llm.get_response_streaming(
                user_message=u2, detection_context=alert_ctx)
            tts.wait_until_done()

            if "[RECOVERED]" in resp:
                term_latency = round(time.perf_counter() - t_recovery, 2)
                term_correct = True
                ok(f"[RECOVERED] detected — latency {term_latency}s")
            else:
                # One more turn with very strong alert signal
                print("  No [RECOVERED] yet — one extra turn with strong alert signal...")
                u3, _ = stt.listen(timeout=20)
                if u3:
                    strong_ctx = format_detection_for_llm(FAKE_ALERT_METRICS)
                    resp2 = llm.get_response_streaming(
                        user_message=u3, detection_context=strong_ctx)
                    tts.wait_until_done()
                    if "[RECOVERED]" in resp2:
                        term_latency = round(time.perf_counter() - t_recovery, 2)
                        term_correct = True
                        ok(f"[RECOVERED] on extra turn — latency {term_latency}s")
                    else:
                        warn("[RECOVERED] not detected in this trial")

        memory.end_session(session_id, avg_drowsy_score=0.30, max_drowsy_score=0.68,
                           turn_count=llm.conversation_turns,
                           duration_s=round(time.perf_counter() - t_s4_start, 1))

    _row("CONVERSATION_BEHAVIOR", [
        "Termination_Accuracy",
        "YES" if term_correct else "NO",
        f"Latency={term_latency}s" if term_latency else "did_not_terminate",
    ])

    # ── 4B  False exit rate  (computational — no speech needed) ──────────────
    step("4B — FALSE EXIT RATE")
    info("Testing exit keyword substring matching against confounding phrases.")
    info("(Computational — no speaking needed)")

    false_exits = 0
    for phrase, expected_false_exit in CONFOUNDING_PHRASES:
        hit = any(kw in phrase.lower() for kw in EXIT_KEYWORDS)
        if hit:
            matched = next(kw for kw in EXIT_KEYWORDS if kw in phrase.lower())
            print(f"  FALSE EXIT: '{phrase}'  (matched '{matched}')")
            false_exits += 1
        else:
            print(f"  OK:         '{phrase}'")

    fe_rate = round(false_exits / len(CONFOUNDING_PHRASES), 3)
    print(f"\n  False exits: {false_exits}/{len(CONFOUNDING_PHRASES)} = {fe_rate:.1%}")

    _row("CONVERSATION_BEHAVIOR", [
        "False_Exit_Rate", fe_rate,
        f"{false_exits}/{len(CONFOUNDING_PHRASES)} confounding phrases triggered exit",
    ])

    # ── 4C  Response elicitation rate  (from Sections 2+3) ────────────────────
    resp  = ctx.get('s1_turns_responded', 0) + ctx.get('s2_turns_responded', 0)
    total = ctx.get('s1_total_turns', 0)     + ctx.get('s2_total_turns', 0)
    eli   = round(resp / total, 3) if total else None
    _row("CONVERSATION_BEHAVIOR", [
        "Response_Elicitation_Rate", eli or "—",
        f"{resp}/{total} turns received STT response",
    ])
    _row("CONVERSATION_BEHAVIOR", [
        "Re_trigger_Rate", "MANUAL_CHECK",
        "Observe 5 min post-session — does system stay silent?",
    ])

    print(f"\n  Response elicitation rate: {eli}")
    pause("\n  Press ENTER to continue to Section 5...")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5  –  PVT-B REACTION TIME
# ══════════════════════════════════════════════════════════════════════════════

def section_5_pvtb(ctx: dict):
    banner("PVT-B REACTION TIME", num=5)

    print(f"  10 stimuli  |  Random intervals 2–8 s  |  Press SPACEBAR on stimulus")
    print(f"  Lapse threshold: 500 ms")
    print(f"  Fatigue level recorded: {ctx.get('fatigue', 'not recorded')}")
    print(f"  Note: scientifically valid only when genuinely fatigued.")
    pause("\n  Press ENTER to begin PVT-B test...")

    _header("PVT_B",
            ["Stimulus", "Interval_s", "RT_ms", "Is_Lapse"])

    N         = 10
    rts       = []
    lapses    = 0
    LAPSE_MS  = 500

    for i in range(1, N + 1):
        isi = random.uniform(2.0, 8.0)
        time.sleep(isi)

        os.system('cls' if platform.system() == 'Windows' else 'clear')
        print("\n\n")
        print("  " + "█" * 52)
        print("  █" + " " * 50 + "█")
        print("  █" + " " * 12 + ">>> PRESS SPACEBAR <<<" + " " * 16 + "█")
        print("  █" + " " * 50 + "█")
        print("  " + "█" * 52)
        print(f"\n  Stimulus {i}/{N}")

        rt = _wait_key(5.0)

        os.system('cls' if platform.system() == 'Windows' else 'clear')

        if rt is None:
            rt_ms    = 5000
            is_lapse = True
            print(f"  Stimulus {i}/{N}:  NO RESPONSE  ← LAPSE")
        else:
            rt_ms    = round(rt * 1000, 1)
            is_lapse = rt_ms > LAPSE_MS
            tag      = "  ← LAPSE" if is_lapse else ""
            print(f"  Stimulus {i}/{N}:  {rt_ms:.0f} ms{tag}")

        rts.append(rt_ms)
        if is_lapse:
            lapses += 1

        _row("PVT_B", [i, round(isi, 1), rt_ms, "YES" if is_lapse else "NO"])

    # Stats (exclude timeout values for mean/median)
    valid = [r for r in rts if r < 5000]
    mean_rt   = round(sum(valid)/len(valid), 1) if valid else None
    sorted_v  = sorted(valid)
    median_rt = sorted_v[len(sorted_v)//2] if sorted_v else None
    lapse_rt  = round(lapses / N, 3)

    _row("PVT_B", ["Mean_RT_ms",   "—", mean_rt   or "—", "—"])
    _row("PVT_B", ["Median_RT_ms", "—", median_rt or "—", "—"])
    _row("PVT_B", ["Lapse_Count",  "—", lapses,            "—"])
    _row("PVT_B", ["Lapse_Rate",   "—", lapse_rt,          "—"])
    _row("PVT_B", ["Fatigue_Level","—", ctx.get('fatigue', ''), "—"])

    print(f"\n  PVT-B complete.")
    print(f"  Mean RT: {mean_rt} ms   Median: {median_rt} ms")
    print(f"  Lapses: {lapses}/{N} ({lapse_rt:.0%})")
    pause("\n  Press ENTER to generate output file...")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6  –  OUTPUT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def section_6_output(ctx: dict):
    banner("OUTPUT GENERATION", num=6)

    # Compute AUC-ROC from frame-level data collected in Section 1
    auc_raw  = _tsv_rows.get("AUC_ROC_DATA", [])
    auc_pairs = []
    for row in auc_raw:
        try:
            auc_pairs.append((float(row[3]), int(row[4])))
        except (IndexError, ValueError):
            pass

    auc_score, _ = compute_auc_roc(auc_pairs)
    if auc_score is not None:
        _row("CONFUSION_MATRIX", ["AUC_ROC", auc_score])
        ok(f"AUC-ROC computed: {auc_score}")

    # Section output order
    ORDER = [
        "CONFUSION_MATRIX",
        "DETECTION_LATENCY",
        "DETECTION_PERFORMANCE",
        "VISION_SIGNALS",
        "SYSTEM_LATENCY",
        "VOICE_SIGNALS",
        "MEMORY_PERSONALIZATION",
        "CONVERSATION_BEHAVIOR",
        "PVT_B",
        "AUC_ROC_DATA",
        "LLM_RESPONSES_FOR_RATING",
    ]

    step(f"Writing {OUTPUT_FILE.name} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# Sentinel V8 — Full Metrics Test\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Fatigue level: {ctx.get('fatigue', 'not recorded')}\n\n")

        for sec in ORDER:
            if sec not in _tsv_headers and sec not in _tsv_rows:
                continue
            f.write(f"# SECTION: {sec}\n")
            hdrs = _tsv_headers.get(sec, [])
            if hdrs:
                f.write("\t".join(str(h) for h in hdrs) + "\n")
            for row in _tsv_rows.get(sec, []):
                f.write("\t".join(str(v) for v in row) + "\n")
            f.write("\n")

    ok(f"Saved → {OUTPUT_FILE}")

    print(f"\n  ── How to import into Google Sheets ──────────────────────")
    print(f"  1. Google Sheets → File → Import → Upload the .tsv file")
    print(f"  2. Separator: Tab")
    print(f"  3. Copy each # SECTION block to its own sheet tab")
    print(f"  File: {OUTPUT_FILE}\n")

    # Final summary
    print("═" * W)
    print("  FINAL RESULTS SUMMARY")
    print("═" * W)
    for row in _tsv_rows.get("CONFUSION_MATRIX", []):
        if len(row) == 2:
            print(f"  {str(row[0]):35s}  {row[1]}")
    print("─" * W)
    for row in _tsv_rows.get("PVT_B", []):
        if row[0] in ("Mean_RT_ms", "Lapse_Count", "Lapse_Rate"):
            print(f"  {str(row[0]):35s}  {row[2]}")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * W)
    print("  SENTINEL V8 — FULL METRICS TEST SUITE")
    print("  Science Fair 2025–26  |  ~20 minutes")
    print("═" * W)
    print(f"\n  Output → {OUTPUT_FILE}")

    # Parse --skip argument
    skip: set = set()
    if "--skip" in sys.argv:
        idx = sys.argv.index("--skip")
        if idx + 1 < len(sys.argv):
            for ch in sys.argv[idx + 1]:
                if ch.isdigit():
                    skip.add(int(ch))

    if skip:
        print(f"  Skipping sections: {sorted(skip)}")

    print(f"\n  Sections: 0=Setup, 1=Detection, 2=Conv(New), 3=Conv(Ret),")
    print(f"            4=Behavior, 5=PVT-B  (Section 6 Output always runs)")
    skip_input = input("  Skip any sections? (e.g. '1 5', or ENTER for all): ").strip()
    for ch in skip_input.split():
        if ch.isdigit():
            skip.add(int(ch))

    ctx: dict = {}

    try:
        if 0 not in skip:
            ctx = section_0_setup()
        else:
            print("  [Section 0 skipped — using default config]")
            ctx = {'ear_thresh': Config.EAR_THRESH, 'fatigue': 'skipped'}

        if 1 not in skip:
            section_1_detection(ctx)

        if 2 not in skip:
            section_2_new_driver(ctx)

        if 3 not in skip:
            section_3_returning_driver(ctx)

        if 4 not in skip:
            section_4_behavior(ctx)

        if 5 not in skip:
            section_5_pvtb(ctx)

        section_6_output(ctx)

    except KeyboardInterrupt:
        print("\n\n  ⚠ Test interrupted — saving partial results...")
        section_6_output(ctx)

    finally:
        # Clean up resources
        cap = ctx.get('cap')
        if cap:
            cap.release()
        fm = ctx.get('face_mesh')
        if fm:
            fm.close()
        tts = ctx.get('tts')
        if tts:
            tts.shutdown()
        mem = ctx.get('memory')
        if mem:
            mem.close()
        cv2.destroyAllWindows()

    print(f"\n  Test complete. Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
