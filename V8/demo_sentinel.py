#!/usr/bin/env python3
"""
demo_sentinel.py — Science Fair Demo for Sentinel (V8)

A reliable ~90-second live demonstration of all Sentinel core features:
  - Real camera + MediaPipe FaceMesh drowsiness detection
  - 8B LLM reasoning gate (Groq llama-3.1-8b-instant) with confirmation count
  - 70B personalized conversation (Groq llama-3.3-70b-versatile + Deepgram STT/TTS)
  - Voice biomarker tracking with personal baseline comparison
  - SQLite memory: pre-seeded driver profile, live fact extraction post-session
  - Comprehensive color-coded live metrics dashboard

Flow:
  1. DB Card    — pre-seeded Archit profile, START MONITORING button
  2. Monitoring — real camera, metric boost (B key), state panel ticking live
  3. Trigger    — 8B confirms drowsiness, Sentinel speaks opening line
  4. Convo      — 2 real STT turns; Turn 1 = real LLM; Turn 2 = crafted recovery
  5. Post-Card  — updated DB showing facts Sentinel just extracted

Controls during monitoring:
  B   — activate metric boost (helps trigger while acting drowsy)
  ESC — exit demo
"""

import cv2
import sys
import time
import threading
import numpy as np
import sqlite3
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

import mediapipe as mp

# ── V8 module imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from memory import MemoryManager
from tts_engine import TTSEngine
from stt_engine import STTEngine
from llm_assistant import LLMAssistant
from voice_features import VoiceFeatureExtractor
from detection import (
    eye_aspect_ratio, LEFT_EYE, RIGHT_EYE,
    process_eye_metrics, process_mouth_metrics,
    process_head_pitch, process_head_roll,
    cleanup_windows, calculate_metrics,
    draw_overlay, DetectionThread,
    format_detection_for_llm,
)
from metric_reasoner import MetricReasoner, ReasonerResult


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_DB_PATH       = Path(__file__).parent / "demo_sentinel.db"
DEMO_CONFIRM_COUNT = 2          # Confirm on 2 consecutive DROWSY/CRITICAL (not 3)
BOOST_PERCLOS_MULT = 1.70       # PERCLOS multiplier when boost active
BOOST_SLOW_ADD     = 2          # Extra slow blinks injected when boost active

# Window titles (stable names — never recreate unnecessarily)
WIN_CAMERA = "Sentinel  ·  Camera Feed"
WIN_DASH   = "Sentinel  ·  Live Dashboard"
WIN_DB     = "Sentinel  ·  Driver Database"

# ── Color palette (BGR) ───────────────────────────────────────────────────────
C_BG     = (16, 16, 16)
C_PANEL  = (26, 26, 26)
C_WHITE  = (235, 235, 235)
C_GRAY   = (80, 80, 80)
C_LGRAY  = (138, 138, 138)
C_GREEN  = (55, 210, 55)
C_YELLOW = (0,  195, 240)
C_RED    = (45,  45, 215)
C_CYAN   = (195, 200, 0)
C_BLUE   = (220, 145, 20)
C_TEAL   = (130, 200, 75)
C_ORANGE = (0,  150, 245)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

STATE_COLORS = {
    "MONITORING":   C_GREEN,
    "DETECTING":    C_YELLOW,
    "DROWSY":       C_RED,
    "CONVERSATION": C_BLUE,
    "RECOVERED":    C_TEAL,
}

# Dashboard dimensions
DASH_W, DASH_H = 520, 780

# DB card dimensions
DB_W, DB_H = 720, 720
DB_BTN_X1, DB_BTN_Y1 = 220, 658
DB_BTN_X2, DB_BTN_Y2 = 500, 694


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _t(img, text, x, y, scale=0.44, color=C_WHITE, bold=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, bold, cv2.LINE_AA)


def _bar(img, x, y, w, h, value, max_val=1.0, color=C_GREEN):
    cv2.rectangle(img, (x, y), (x + w, y + h), (46, 46, 46), -1)
    fill = int((min(float(value), float(max_val)) / float(max_val)) * (w - 2))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + h - 1), color, -1)


def _divider(img, y, x1=12, x2=None, w=DASH_W):
    x2 = x2 or w - 12
    cv2.line(img, (x1, y), (x2, y), C_GRAY, 1)


def _arrow(cur, prev, threshold=0.005):
    if prev is None or abs(float(cur) - float(prev)) < threshold:
        return "  ="
    return " ^" if float(cur) > float(prev) else " v"


def _arrow_inv(cur, prev, threshold=0.1):
    """Inverted: higher = worse → up arrow means bad."""
    if prev is None or abs(float(cur) - float(prev)) < threshold:
        return "  ="
    return "  ^" if float(cur) > float(prev) else "  v"


def _section_header(img, text, x, y, right_text="", right_color=C_LGRAY):
    _t(img, text, x, y, 0.50, C_CYAN, 1)
    if right_text:
        _t(img, right_text, DASH_W - 10 - len(right_text) * 8, y, 0.38, right_color)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE SEEDING
# ═══════════════════════════════════════════════════════════════════════════════

def seed_demo_database(db_path: Path) -> MemoryManager:
    """Wipe and re-seed the demo database to a clean, consistent starting state."""
    # Always start fresh so every demo run looks identical
    if db_path.exists():
        db_path.unlink()

    mm = MemoryManager(db_path=db_path)
    now  = datetime.now()
    ts   = now.isoformat()
    yest = (now - timedelta(hours=22)).isoformat()
    yend = (now - timedelta(hours=22, minutes=-3)).isoformat()

    conn = mm._conn

    # ── Prior session (yesterday, camera-triggered) ──────────────────────────
    conn.execute("""
        INSERT INTO sessions
        (started_at, ended_at, avg_drowsy_score, max_drowsy_score,
         turn_count, duration_s, recovery_time_s, peak_perclos,
         peak_slow_blinks, avg_energy_rms, avg_speech_rate,
         avg_response_latency, trigger_reason)
        VALUES (?, ?, 0.64, 0.81, 4, 104.0, 43.0, 0.29,
                5, 0.034, 131.0, 2.3, 'camera')
    """, (yest, yend))

    # ── Driver facts ─────────────────────────────────────────────────────────
    facts = [
        ("name",              "Archit",                                              1.0, 3),
        ("occupation",        "High school student and science fair researcher",     0.95, 2),
        ("driving_habit",     "Drives late at night after long study sessions",      0.90, 2),
        ("project",           "Building Sentinel — AI drowsiness detection system",  0.95, 2),
        ("drowsy_pattern",    "Eyes drop first; PERCLOS rises before head movement", 0.85, 2),
        ("recovery_style",    "Recovers quickly once engaged in conversation",       0.80, 2),
    ]
    for fact_type, value, confidence, times in facts:
        conn.execute("""
            INSERT INTO facts
            (fact_type, value, confidence, session_id,
             first_seen, last_seen, times_confirmed)
            VALUES (?, ?, ?, 1, ?, ?, ?)
        """, (fact_type, value, confidence, yest, yest, times))

    # ── Camera + Voice baselines ──────────────────────────────────────────────
    # EAR: Eye Aspect Ratio — Archit's personal open-eye geometry
    # Stored from the 5-second EAR calibration run at first session startup.
    # Threshold = avg * 0.75 (75% of personal open-eye EAR = eyes-closed cutoff).
    baselines = {
        "ear_open_avg":       (0.288, 0.271, 0.304,  5),   # avg open-eye EAR
        "ear_threshold":      (0.216, 0.203, 0.228,  5),   # personal closed-eye cutoff
        "energy_rms":         (0.038, 0.024, 0.058, 15),
        "speech_rate_wpm":    (138.0, 110.0, 170.0, 15),
        "pause_ratio":        (0.210, 0.130, 0.330, 15),
        "response_latency_s": (1.55,  0.75,  2.90,  15),
        "peak_amplitude":     (0.335, 0.175, 0.520, 15),
    }
    for metric, (avg, mn, mx, count) in baselines.items():
        conn.execute("""
            INSERT INTO baselines
            (metric_name, avg_value, min_value, max_value, sample_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (metric, avg, mn, mx, count, ts))

    conn.commit()
    print(f"✓ Demo database seeded → {db_path.name}")
    return mm


# ═══════════════════════════════════════════════════════════════════════════════
#  DB CARD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

def render_db_card(mm: MemoryManager, mode: str = "before") -> np.ndarray:
    """Render the database overview card as an OpenCV image."""
    img = np.full((DB_H, DB_W, 3), C_BG, dtype=np.uint8)

    def t(text, x, y, sc=0.44, col=C_WHITE, b=1):
        cv2.putText(img, text, (x, y), FONT, sc, col, b, cv2.LINE_AA)

    def div(y):
        cv2.line(img, (14, y), (DB_W - 14, y), C_GRAY, 1)

    # ── Header bar ────────────────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (DB_W, 54), (28, 28, 28), -1)
    t("SENTINEL", 16, 36, 0.88, C_CYAN, 2)
    t("Driver Intelligence Database", 152, 36, 0.58, C_WHITE, 1)
    live_col  = C_GREEN if mode == "before" else C_TEAL
    live_text = "BEFORE SESSION" if mode == "before" else "AFTER SESSION  — UPDATED"
    t(f"● {live_text}", DB_W - 260, 36, 0.42, live_col)

    y = 74

    # ── Driver Profile ────────────────────────────────────────────────────────
    sessions = mm.get_session_count()
    last_t   = mm.get_last_session_time()
    t("DRIVER PROFILE", 16, y, 0.50, C_CYAN, 1)
    info_str = f"● {sessions} session{'s' if sessions != 1 else ''}"
    if last_t:
        info_str += f"  ·  last {last_t[:10]}"
    t(info_str, DB_W - 16 - len(info_str) * 7, y, 0.38, C_LGRAY)
    y += 8
    div(y); y += 18

    facts = mm.get_context_facts(limit=20)
    shown = 0
    for fact_type, value, confidence, times in facts:
        if shown >= 7:
            break
        label = fact_type.replace("_", " ").title()
        t(f"  {label}", 16, y, 0.41, C_LGRAY)
        t(value[:72], 196, y, 0.42, C_WHITE)
        if times >= 3:
            t("✓", DB_W - 30, y, 0.38, C_GREEN)
        y += 22
        shown += 1

    # If after-mode: highlight newly extracted facts
    if mode == "after":
        new_found = [(ft, v) for ft, v, cf, ti in facts if ti == 1]
        if new_found:
            y += 4
            t("  ↳ Newly learned this session:", 16, y, 0.40, C_TEAL)
            y += 18
            for ft, v in new_found[:4]:
                lbl = ft.replace("_", " ").title()
                t(f"      + {lbl}:  {v[:60]}", 16, y, 0.40, C_TEAL)
                y += 18

    y += 8
    div(y); y += 20

    # ── Stored Baselines ─────────────────────────────────────────────────────
    baselines = mm.get_baselines()
    t("STORED BASELINES  (learned from Archit)", 16, y, 0.50, C_CYAN, 1)
    cal_text = "● calibrated" if baselines else "● not calibrated"
    cal_col  = C_GREEN if baselines else C_YELLOW
    t(cal_text, DB_W - 130, y, 0.40, cal_col)
    y += 8
    div(y); y += 14

    # Camera baselines (EAR — eye geometry)
    t("  Camera / Eye", 16, y, 0.40, C_ORANGE)
    y += 17
    cam_bl_rows = [
        ("ear_open_avg",   "EAR open-eye avg",  "{:.3f}",   "personal eye geometry"),
        ("ear_threshold",  "EAR closed cutoff", "{:.3f}",   "= open_avg × 0.75"),
    ]
    for metric, label, fmt, note in cam_bl_rows:
        bl = baselines.get(metric)
        if not bl:
            continue
        val_s = fmt.format(bl["avg"])
        rng_s = f"range {bl['min']:.3f}–{bl['max']:.3f}   ({note})"
        t(f"    {label:<22}", 16, y, 0.40, C_LGRAY)
        t(val_s, 220, y, 0.42, C_WHITE)
        t(rng_s, 290, y, 0.35, C_GRAY)
        y += 19

    y += 4
    # Voice baselines
    t("  Voice", 16, y, 0.40, C_BLUE)
    y += 17
    voice_bl_rows = [
        ("energy_rms",         "Energy RMS",     "{:.4f}",  "avg speech volume"),
        ("speech_rate_wpm",    "Speech Rate",    "{:.0f} wpm", "words per minute"),
        ("pause_ratio",        "Pause Ratio",    "{:.3f}",  "fraction of silence"),
        ("response_latency_s", "Response Delay", "{:.2f}s", "time to start speaking"),
    ]
    for metric, label, fmt, note in voice_bl_rows:
        bl = baselines.get(metric)
        if not bl:
            continue
        val_s = fmt.format(bl["avg"])
        rng_s = f"range {bl['min']:.3f}–{bl['max']:.3f}   n={bl['sample_count']}"
        t(f"    {label:<22}", 16, y, 0.40, C_LGRAY)
        t(val_s, 220, y, 0.42, C_WHITE)
        t(rng_s, 290, y, 0.35, C_GRAY)
        y += 19

    y += 8
    div(y); y += 20

    # ── Session History ───────────────────────────────────────────────────────
    t("SESSION HISTORY", 16, y, 0.50, C_CYAN, 1)
    y += 8
    div(y); y += 18

    hist = mm.get_driver_history_for_llm()
    if hist:
        lines_out = 0
        for hl in hist.split("\n"):
            hl = hl.strip()
            if not hl or hl.startswith("Recent") or hl.startswith("  "):
                continue
            t(f"  {hl[:88]}", 16, y, 0.39, C_LGRAY)
            y += 17
            lines_out += 1
            if lines_out >= 4:
                break

    y = max(y + 10, DB_H - 86)
    div(y); y += 14

    # ── START button ─────────────────────────────────────────────────────────
    btn_text  = "START MONITORING" if mode == "before" else "RETURN TO MONITORING"
    btn_color = C_GREEN if mode == "before" else C_TEAL
    cv2.rectangle(img, (DB_BTN_X1, DB_BTN_Y1), (DB_BTN_X2, DB_BTN_Y2), btn_color, -1)
    cv2.rectangle(img, (DB_BTN_X1, DB_BTN_Y1), (DB_BTN_X2, DB_BTN_Y2), C_WHITE, 1)
    (tw, th), _ = cv2.getTextSize(btn_text, FONT, 0.58, 2)
    tx = DB_BTN_X1 + ((DB_BTN_X2 - DB_BTN_X1) - tw) // 2
    ty = DB_BTN_Y1 + ((DB_BTN_Y2 - DB_BTN_Y1) + th) // 2
    cv2.putText(img, btn_text, (tx, ty), FONT, 0.58, (10, 10, 10), 2, cv2.LINE_AA)

    # ── Footer ────────────────────────────────────────────────────────────────
    footer = ("Sentinel v8  ·  Groq llama-3.3-70b + llama-3.1-8b  "
              "·  Deepgram Nova-3  ·  MediaPipe FaceMesh  ·  SQLite")
    t(footer, 14, DB_H - 10, 0.30, C_GRAY)

    return img


def show_db_card(mm: MemoryManager, mode: str = "before") -> bool:
    """Display DB card. Returns True when button clicked, False on ESC."""
    clicked = [False]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if DB_BTN_X1 <= x <= DB_BTN_X2 and DB_BTN_Y1 <= y <= DB_BTN_Y2:
                clicked[0] = True

    cv2.namedWindow(WIN_DB, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_DB, DB_W, DB_H)
    cv2.setMouseCallback(WIN_DB, on_mouse)
    cv2.moveWindow(WIN_DB, 180, 80)

    img = render_db_card(mm, mode)
    while not clicked[0]:
        cv2.imshow(WIN_DB, img)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            cv2.destroyWindow(WIN_DB)
            cv2.waitKey(1)
            return False

    cv2.destroyWindow(WIN_DB)
    cv2.waitKey(1)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

class DemoDashboard:
    """
    Custom dashboard for the demo. Shows state badge, detection metrics,
    8B reasoner with confirmation boxes, and voice biomarkers per turn.
    Thread-safe: update methods can be called from any thread.
    """

    def __init__(self, baselines: dict = None):
        self._baselines   = baselines or {}
        self._lock        = threading.Lock()
        self._state       = "MONITORING"
        self._r_level     = "ALERT"
        self._r_conf      = 0.0
        self._r_text      = ""
        self._confirm     = 0
        self._confirm_max = DEMO_CONFIRM_COUNT
        self._voice       = {}
        self._prev_voice  = {}
        self._prev_det    = {}
        self._turn        = 0
        self._boost       = False

    def set_state(self, state: str):
        with self._lock:
            self._state = state

    def set_reasoner(self, result: ReasonerResult, confirm: int):
        with self._lock:
            self._r_level   = result.level
            self._r_conf    = result.confidence
            self._r_text    = result.reasoning
            self._confirm   = confirm

    def update_voice(self, features: dict):
        with self._lock:
            self._prev_voice = self._voice.copy()
            self._voice      = features.copy() if features else {}
            self._turn      += 1

    def set_boost(self, active: bool):
        with self._lock:
            self._boost = active

    # ------------------------------------------------------------------

    def render(self, det: dict) -> np.ndarray:
        with self._lock:
            state      = self._state
            r_level    = self._r_level
            r_conf     = self._r_conf
            r_text     = self._r_text
            confirm    = self._confirm
            cmax       = self._confirm_max
            voice      = self._voice.copy()
            prev_voice = self._prev_voice.copy()
            baselines  = self._baselines.copy()
            turn       = self._turn
            boost      = self._boost

        img = np.full((DASH_H, DASH_W, 3), C_BG, dtype=np.uint8)

        def t(text, x, y, sc=0.44, col=C_WHITE, b=1):
            cv2.putText(img, text, (x, y), FONT, sc, col, b, cv2.LINE_AA)

        def div(y):
            cv2.line(img, (12, y), (DASH_W - 12, y), C_GRAY, 1)

        # ── Header ────────────────────────────────────────────────────────────
        cv2.rectangle(img, (0, 0), (DASH_W, 48), (26, 26, 26), -1)
        t("SENTINEL  LIVE DASHBOARD", 12, 32, 0.65, C_CYAN, 2)

        y = 62

        # ── State badge ───────────────────────────────────────────────────────
        badge_col = STATE_COLORS.get(state, C_WHITE)
        bx1, by1, bx2, by2 = 12, y, DASH_W - 12, y + 44
        cv2.rectangle(img, (bx1, by1), (bx2, by2), badge_col, -1)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        (tw, th), _ = cv2.getTextSize(state, FONT, 0.78, 2)
        tx = bx1 + ((bx2 - bx1) - tw) // 2
        ty = by1 + ((by2 - by1) + th) // 2
        cv2.putText(img, state, (tx, ty), FONT, 0.78, (12, 12, 12), 2, cv2.LINE_AA)
        y += 56

        # ── Detection section ─────────────────────────────────────────────────
        t("DETECTION  (Camera / MediaPipe)", 12, y, 0.50, C_CYAN, 1)
        y += 8;  div(y);  y += 17

        score   = det.get("drowsy_score", 0.0)
        sc_col  = C_GREEN if score < 0.35 else (C_YELLOW if score < 0.55 else C_RED)
        trend_s = _arrow(score, self._prev_det.get("drowsy_score"))
        t(f"Drowsy Score   {score:.3f}", 14, y, 0.45, sc_col)
        _bar(img, 232, y - 13, 238, 15, score, 1.0, sc_col)
        t(trend_s, 475, y, 0.42, C_WHITE)
        y += 23

        perclos = det.get("perclos", 0.0)
        pc_col  = C_GREEN if perclos < 0.10 else (C_YELLOW if perclos < 0.20 else C_RED)
        trend_p = _arrow(perclos, self._prev_det.get("perclos"))
        t(f"PERCLOS        {perclos:.3f}", 14, y, 0.45, C_WHITE)
        _bar(img, 232, y - 13, 238, 15, perclos, 1.0, pc_col)
        t(trend_p, 475, y, 0.42, C_WHITE)
        y += 22

        blinks = det.get("blink_rate", 0)
        slow   = det.get("slow_blinks", 0)
        sl_col = C_GREEN if slow == 0 else (C_YELLOW if slow <= 2 else C_RED)
        t(f"Blink Rate     {blinks}", 14, y, 0.44, C_WHITE)
        t(f"Slow Blinks  {slow}", 264, y, 0.44, sl_col)
        y += 20

        t(f"EAR std   {det.get('ear_std', 0):.4f}", 14, y, 0.40, C_LGRAY)
        t(f"Pitch var  {det.get('pitch_var', 0):.5f}", 210, y, 0.40, C_LGRAY)
        y += 20

        ms   = det.get("microsleep", False)
        hd   = det.get("head_down",  False)
        ms_c = C_RED if ms else C_GREEN
        hd_c = C_RED if hd else C_GREEN
        t(f"Microsleep  {'YES' if ms else 'No'}", 14, y, 0.44, ms_c)
        t(f"Head Down  {'YES' if hd else 'No'}",  264, y, 0.44, hd_c)
        y += 20

        alert_dur = det.get("alert_duration", 0.0)
        if alert_dur > 0:
            mins, secs = int(alert_dur // 60), int(alert_dur % 60)
            dur_s = f"{mins}:{secs:02d}" if mins else f"{secs}s"
            t(f"Alert Timer    {dur_s}", 14, y, 0.44, C_GREEN)
            prog = min(alert_dur / Config.ALERT_RECOVERY_SECS, 1.0)
            _bar(img, 232, y - 13, 238, 15, prog, 1.0, C_GREEN)
            if prog >= 1.0:
                t("RECOVERED", 476, y, 0.40, C_TEAL)
        else:
            t("Alert Timer    --", 14, y, 0.44, C_GRAY)
        y += 26

        self._prev_det = det.copy()

        # ── 8B Reasoner section ───────────────────────────────────────────────
        div(y);  y += 16
        t("8B REASONER  (Groq llama-3.1-8b-instant)", 12, y, 0.48, C_CYAN, 1)
        y += 24

        lc = {"ALERT": C_GREEN, "MILD": C_YELLOW,
               "DROWSY": C_RED,  "CRITICAL": C_RED}.get(r_level, C_WHITE)

        # Level label + confidence
        t(r_level, 14, y, 0.65, lc, 2)
        conf_str = f"{r_conf:.0%} confidence"
        t(conf_str, 124, y, 0.44, C_WHITE)

        # Confirmation boxes
        box_x = DASH_W - 14 - cmax * 42
        for i in range(cmax):
            bx = box_x + i * 42
            filled = i < confirm
            fc     = lc if filled else C_GRAY
            cv2.rectangle(img, (bx, y - 16), (bx + 34, y + 6),
                          fc, -1 if filled else 1)
            inner = (12, 12, 12) if filled else C_GRAY
            cv2.putText(img, str(i + 1), (bx + 11, y + 2),
                        FONT, 0.36, inner, 1, cv2.LINE_AA)
        t(f"{confirm}/{cmax}", box_x + cmax * 42 + 4, y, 0.40, C_LGRAY)
        y += 22

        # Reasoning text (word-wrapped)
        if r_text:
            words, line_buf = r_text.split(), ""
            for w in words:
                test = (line_buf + " " + w).strip()
                (tw2, _), _ = cv2.getTextSize(test, FONT, 0.37, 1)
                if tw2 > DASH_W - 32 and line_buf:
                    t(line_buf.strip(), 14, y, 0.37, C_LGRAY)
                    y += 15;  line_buf = w
                else:
                    line_buf = test
            if line_buf.strip():
                t(line_buf.strip(), 14, y, 0.37, C_LGRAY)
                y += 15
        y += 12

        # ── Voice Biomarkers section ──────────────────────────────────────────
        div(y);  y += 16
        turn_lbl = f"Turn {turn}" if turn > 0 else "Waiting for conversation..."
        t(f"VOICE BIOMARKERS  ({turn_lbl})", 12, y, 0.48, C_BLUE, 1)
        y += 24

        if voice:
            # Energy RMS
            rms    = voice.get("energy_rms", 0.0)
            rms_bl = baselines.get("energy_rms", {})
            rms_r  = rms / rms_bl["avg"] if rms_bl and rms_bl.get("avg", 0) > 0 else None
            rms_c  = C_GREEN if (rms_r is None or rms_r >= 0.80) else \
                     (C_YELLOW if rms_r >= 0.60 else C_RED)
            trend_r = _arrow(rms, prev_voice.get("energy_rms"))
            t(f"Energy RMS     {rms:.4f}", 14, y, 0.44, C_WHITE)
            if rms_r is not None:
                t(f"({rms_r:.0%} baseline)", 265, y, 0.38, C_LGRAY)
            _bar(img, 438, y - 11, 58, 13, rms, 0.12, rms_c)
            t(trend_r, 500, y, 0.40, C_WHITE)
            y += 21

            # Speech Rate
            rate = voice.get("speech_rate_wpm")
            if rate is not None:
                rate_bl = baselines.get("speech_rate_wpm", {})
                rate_r  = rate / rate_bl["avg"] if rate_bl and rate_bl.get("avg", 0) > 0 else None
                rate_c  = C_GREEN if (rate_r is None or rate_r >= 0.80) else \
                          (C_YELLOW if rate_r >= 0.65 else C_RED)
                trend_rt = _arrow(rate, prev_voice.get("speech_rate_wpm"))
                t(f"Speech Rate    {rate:.0f} wpm", 14, y, 0.44, C_WHITE)
                if rate_r is not None:
                    t(f"({rate_r:.0%} baseline)", 265, y, 0.38, C_LGRAY)
                t(trend_rt, 500, y, 0.40, C_WHITE)
                y += 21

            # Response Latency
            lat = voice.get("response_latency_s")
            if lat is not None:
                lat_bl  = baselines.get("response_latency_s", {})
                lat_dif = lat - lat_bl.get("avg", 1.55) if lat_bl else 0
                lat_c   = C_RED if lat > 4.0 else (C_YELLOW if lat > 2.5 else C_WHITE)
                sign    = "+" if lat_dif >= 0 else ""
                trend_l = _arrow_inv(lat, prev_voice.get("response_latency_s"))
                t(f"Resp Latency   {lat:.1f}s", 14, y, 0.44, lat_c)
                t(f"({sign}{lat_dif:.1f}s vs base)", 265, y, 0.38, C_LGRAY)
                t(trend_l, 500, y, 0.40, C_WHITE)
                y += 21

            # Pause Ratio
            pr   = voice.get("pause_ratio", 0.0)
            pr_c = C_GREEN if pr < 0.30 else (C_YELLOW if pr < 0.50 else C_RED)
            t(f"Pause Ratio    {pr:.3f}", 14, y, 0.44, C_WHITE)
            _bar(img, 265, y - 11, 150, 13, pr, 1.0, pr_c)
            y += 21

            # Word count / duration
            wc  = voice.get("word_count", 0)
            dur = voice.get("duration_s", 0.0)
            if wc > 0:
                t(f"Words: {wc}    Duration: {dur:.1f}s", 14, y, 0.39, C_GRAY)
                y += 18
        else:
            t("  Listening for first voice response...", 14, y, 0.40, C_GRAY)
            y += 18

        # ── Footer bar ────────────────────────────────────────────────────────
        cv2.line(img, (0, DASH_H - 30), (DASH_W, DASH_H - 30), C_GRAY, 1)
        b_text = "BOOST: ACTIVE" if boost else "Press  B  to enhance detection"
        b_col  = C_YELLOW if boost else C_GRAY
        t(b_text, 14, DASH_H - 12, 0.36, b_col)
        t("ESC: exit", DASH_W - 82, DASH_H - 12, 0.36, C_GRAY)

        return img


# ═══════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE + STATE INIT
# ═══════════════════════════════════════════════════════════════════════════════

def _init_mediapipe():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _init_state():
    return {
        "ear_window":       deque(maxlen=200),
        "pitch_window":     deque(maxlen=200),
        "closed_window":    deque(maxlen=200),
        "blink_times":      deque(maxlen=50),
        "blink_durations":  deque(maxlen=50),
        "yawn_times":       deque(maxlen=20),
        "eye_closed_start": None,
        "blink_start":      None,
        "yawn_start":       None,
        "head_down_start":  None,
        "head_roll_start":  None,
        "llm_triggered":    False,
        "window_start_time": time.time(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  OPENING LINE (inlined from main.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_opening_line(freq: dict) -> tuple:
    today   = freq["today_count"]
    last_2h = freq["last_2h_count"]
    severity = freq["severity"]

    if severity == "critical":
        line = (f"Warning — this is now a serious concern. "
                f"You've been drowsy {today} times today. "
                "Please think about pulling over at the nearest safe spot.")
    elif severity == "serious":
        if last_2h >= 2:
            line = (f"Hey Archit — that's {last_2h + 1} times in the last two hours. "
                    "I'm getting concerned. Let's talk.")
        else:
            line = (f"Hey Archit — this is the {today + 1}th time today. "
                    "I'm here, but this is becoming a pattern.")
    elif severity == "elevated":
        line = ("Hey Archit — looks like you're feeling drowsy again. "
                "I'm here with you — let's talk.")
    elif today == 0:
        line = ("Hey Archit — I'm detecting some drowsiness. "
                "I'm Sentinel, I'm here to help.")
    else:
        line = ("Hey Archit — feeling drowsy again? I've got you — "
                "let's get you back on track.")

    return line, severity in ("serious", "critical")


# ═══════════════════════════════════════════════════════════════════════════════
#  MONITORING PHASE
# ═══════════════════════════════════════════════════════════════════════════════

def run_monitoring_phase(
    reasoner: MetricReasoner,
    dashboard: DemoDashboard,
) -> tuple:
    """
    Real-time monitoring loop.
    Returns (True, final_metrics) when drowsiness confirmed, or (False, None) on exit.
    Controls: B = metric boost, ESC = exit.
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          Config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    face_mesh    = _init_mediapipe()
    state        = _init_state()
    PROC_W, PROC_H = Config.PROC_WIDTH, Config.PROC_HEIGHT
    boost_active = False
    frame_count  = 0
    dashboard.set_state("MONITORING")
    dashboard.set_boost(False)

    cv2.namedWindow(WIN_CAMERA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAMERA, 480, 360)
    cv2.moveWindow(WIN_CAMERA,  40, 120)
    cv2.namedWindow(WIN_DASH, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_DASH, DASH_W, DASH_H)
    cv2.moveWindow(WIN_DASH,  550, 60)

    print("\n" + "─" * 58)
    print("  MONITORING  —  act drowsy, then press B to enhance")
    print("─" * 58)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % Config.DETECTION_FRAME_SKIP != 0:
            # Still show last rendered frame at full rate
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key in (ord("b"), ord("B")):
                boost_active = True
                dashboard.set_boost(True)
                print("🔥  Boost active — metrics enhanced")
            continue

        now      = time.time()
        small    = cv2.resize(frame, (PROC_W, PROC_H))
        rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results  = face_mesh.process(rgb)

        microsleep = head_down = head_roll = False
        ear = 0.25;  mar = 0.0;  landmarks = None

        if results.multi_face_landmarks:
            lm        = results.multi_face_landmarks[0].landmark
            landmarks = [(int(p.x * PROC_W), int(p.y * PROC_H)) for p in lm]
            ear, microsleep = process_eye_metrics(
                landmarks, state, now, ear_thresh=Config.EAR_THRESH)
            mar       = process_mouth_metrics(landmarks, state, now)
            head_down = process_head_pitch(landmarks, state, now)
            head_roll = process_head_roll(landmarks, state, now)

        cleanup_windows(state, now)
        raw = calculate_metrics(state, microsleep, head_down, head_roll)

        # ── Metric boost ──────────────────────────────────────────────────────
        if boost_active:
            m = raw.copy()
            m["perclos"]    = min(1.0, raw["perclos"] * BOOST_PERCLOS_MULT)
            m["slow_blinks"] = raw["slow_blinks"] + BOOST_SLOW_ADD
            m["drowsy_score"] = min(1.0,
                0.30 * m["perclos"] +
                0.20 * int(microsleep) +
                0.20 * min(m["slow_blinks"] / 8, 1.0) +
                0.15 * min(raw["ear_std"] / 0.20, 1.0) +
                0.05 * min(raw["pitch_var"] / 0.015, 1.0) +
                0.05 * int(head_down) +
                0.05 * int(head_roll)
            )
            metrics = m
        else:
            metrics = raw

        # ── Microsleep bypass ─────────────────────────────────────────────────
        if microsleep and not state["llm_triggered"]:
            print("\n🚨  MICROSLEEP — instant trigger")
            state["llm_triggered"] = True
            fm = metrics.copy()
            fm["microsleep"] = True
            fm["head_down"]  = head_down
            dashboard.set_state("DROWSY")
            _flash_trigger(cap, face_mesh, frame, metrics, ear, mar,
                           microsleep, head_down, head_roll, state, landmarks,
                           PROC_W, PROC_H, dashboard)
            return True, fm

        # ── 8B gate ───────────────────────────────────────────────────────────
        if (metrics["drowsy_score"] > Config.REASONER_PRE_FILTER
                and reasoner.should_call()):
            result  = reasoner.evaluate(metrics, microsleep, head_down, head_roll)
            confirm = reasoner.get_confirmation_count()
            dashboard.set_reasoner(result, confirm)
            new_state = "DETECTING" if result.is_drowsy() else "MONITORING"
            dashboard.set_state(new_state)
            print(f"  🧠  8B: {result.level} ({result.confidence:.0%})  "
                  f"[{confirm}/{DEMO_CONFIRM_COUNT}]  —  {result.reasoning[:60]}")

        # ── Demo trigger (uses DEMO_CONFIRM_COUNT, not Config) ────────────────
        confirm_now = reasoner.get_confirmation_count()
        if confirm_now >= DEMO_CONFIRM_COUNT and not state["llm_triggered"]:
            print("\n🚨  DROWSINESS CONFIRMED — triggering conversation")
            state["llm_triggered"] = True
            fm = metrics.copy()
            fm["microsleep"] = microsleep
            fm["head_down"]  = head_down
            dashboard.set_state("DROWSY")
            _flash_trigger(cap, face_mesh, frame, metrics, ear, mar,
                           microsleep, head_down, head_roll, state, landmarks,
                           PROC_W, PROC_H, dashboard)
            return True, fm

        # ── Render ────────────────────────────────────────────────────────────
        rr = reasoner.get_last_result()
        ds = rr.level if rr.is_drowsy() else "ALERT"
        draw_overlay(frame, metrics, ear, mar, microsleep, head_down,
                     head_roll, state, ds, landmarks=landmarks,
                     proc_size=(PROC_W, PROC_H))
        _stamp_state(frame, dashboard._state)

        dash_img = dashboard.render(metrics)
        cv2.imshow(WIN_CAMERA, frame)
        cv2.imshow(WIN_DASH,   dash_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in (ord("b"), ord("B")):
            boost_active = True
            dashboard.set_boost(True)
            print("🔥  Boost active — metrics enhanced")

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return False, None


def _flash_trigger(cap, face_mesh, frame, metrics, ear, mar,
                   microsleep, head_down, head_roll, state, landmarks,
                   PROC_W, PROC_H, dashboard):
    """Flash DROWSY state 5× then release resources."""
    for _ in range(5):
        draw_overlay(frame, metrics, ear, mar, microsleep, head_down,
                     head_roll, state, "CRITICAL", landmarks=landmarks,
                     proc_size=(PROC_W, PROC_H))
        _stamp_state(frame, "DROWSY")
        dash_img = dashboard.render(metrics)
        cv2.imshow(WIN_CAMERA, frame)
        cv2.imshow(WIN_DASH,   dash_img)
        cv2.waitKey(120)
    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def _stamp_state(frame, state: str):
    """Overlay a color-coded state badge onto the camera frame in-place."""
    col = STATE_COLORS.get(state, C_WHITE)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (w - 220, 8), (w - 8, 40), (0, 0, 0), -1)
    cv2.rectangle(frame, (w - 220, 8), (w - 8, 40), col, 2)
    cv2.putText(frame, f"● {state}", (w - 212, 30),
                FONT, 0.58, col, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO LLM ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════

class DemoLLMAssistant(LLMAssistant):
    """
    LLMAssistant with demo guidance appended to the system prompt.
    Keeps the LLM genuine (real Groq call, real metrics) while making
    Turn 1 behaviour consistent for a live audience.
    """

    def start_conversation(self, *args, **kwargs):
        super().start_conversation(*args, **kwargs)

        demo_hint = (
            "\n\n## DEMO GUIDANCE — Science Fair  (READ BEFORE TURN 1)\n"
            "You are live on stage in front of science fair judges right now.\n\n"
            "RULES — follow these exactly:\n"
            "1. Keep every response to 2–3 short sentences. No more.\n"
            "2. Always address Archit by name at least once.\n"
            "3. TURN 1: He will respond somewhat drowsily / slowly. "
            "Acknowledge what your sensors are showing. Ask one engaging personal "
            "question. Reference something specific from his profile "
            "(science fair project, late-night driving, etc.).\n"
            "4. TURN 2: He will sound noticeably more alert. "
            "Remark on the improvement naturally — say something like "
            "'you sound a lot more present now' or 'that response was sharp'. "
            "Do NOT mention sensor names to him. "
            "End your ENTIRE turn-2 response with [RECOVERED] as the very last token.\n"
            "5. Do not mention PERCLOS, EAR, or technical terms to the driver.\n"
        )

        self._system_message["content"] += demo_hint
        self.messages[0] = self._system_message


# ═══════════════════════════════════════════════════════════════════════════════
#  TURN-2 RECOVERY RESPONSE (crafted from real voice data)
# ═══════════════════════════════════════════════════════════════════════════════

def craft_recovery_response(
    user_text: str,
    voice_features: dict,
    baselines: dict,
) -> str:
    """
    Build a Sentinel-style recovery line that references the driver's ACTUAL
    measured voice improvement from Turn 2. Always ends with [RECOVERED].
    """
    rate    = voice_features.get("speech_rate_wpm") or 0
    latency = voice_features.get("response_latency_s") or 0
    rate_bl = (baselines.get("speech_rate_wpm") or {}).get("avg", 130)
    lat_bl  = (baselines.get("response_latency_s") or {}).get("avg", 1.6)

    # Natural-language commentary based on measured numbers
    if latency <= lat_bl + 0.6:
        engagement = "you jumped right in without missing a beat"
    elif latency <= lat_bl + 1.8:
        engagement = "that response came a lot quicker"
    else:
        engagement = "you're sounding much more engaged"

    if rate >= rate_bl * 0.90:
        voice_note = "and your voice is back at full strength"
    elif rate >= rate_bl * 0.75:
        voice_note = "and your speech is clearing up nicely"
    else:
        voice_note = "I can hear you're more awake"

    return (
        f"Hey Archit — {engagement}, {voice_note}. "
        f"Everything is looking a lot better on my end. "
        f"I'll keep watching, but I think you've got it from here — stay sharp. "
        f"[RECOVERED]"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION PHASE
# ═══════════════════════════════════════════════════════════════════════════════

def run_conversation_phase(
    tts:              TTSEngine,
    stt:              STTEngine,
    llm_assistant:    DemoLLMAssistant,
    initial_metrics:  dict,
    memory_manager:   MemoryManager,
    reasoner:         MetricReasoner,
    voice_extractor:  VoiceFeatureExtractor,
    dashboard:        DemoDashboard,
) -> int:
    """
    2-turn conversation with real STT/TTS.
    Turn 1: real 70B Groq LLM (guided by demo system-prompt hint).
    Turn 2: crafted response using actual measured voice data.
    Returns session_id for post-processing.
    """
    print("\n" + "=" * 58)
    print("  CONVERSATION PHASE")
    print("=" * 58)

    session_id    = memory_manager.start_session()
    llm_assistant._session_id = session_id
    session_start = time.perf_counter()
    baselines     = memory_manager.get_baselines()
    baselines_str = memory_manager.format_baselines_for_llm()

    dashboard.set_state("CONVERSATION")
    dashboard.set_boost(False)

    # Background detection thread (own camera)
    det_thread = DetectionThread(dashboard=None, ear_thresh=Config.EAR_THRESH)
    det_thread.start()

    # LLM conversation setup
    det_ctx      = format_detection_for_llm(
        initial_metrics,
        microsleep=initial_metrics.get("microsleep", False),
        head_down=initial_metrics.get("head_down", False),
    )
    reasoner_ctx = reasoner.get_reasoning_for_llm()
    driver_hist  = memory_manager.get_driver_history_for_llm()
    freq         = memory_manager.get_drowsy_frequency()
    session_cnt  = max(0, memory_manager.get_session_count() - 1)

    llm_assistant.start_conversation(
        det_ctx, baselines_str, session_cnt,
        reasoner_context=reasoner_ctx,
        driver_history=driver_hist,
        drowsy_freq=freq,
    )

    # Shared accumulators (written from conv thread, read after join)
    voice_accum = []
    score_accum = []
    conv_done   = threading.Event()

    # ------------------------------------------------------------------
    def conv_worker():
        nonlocal voice_accum, score_accum
        try:
            # ── Opening message (real 70B LLM) ────────────────────────────────
            print("\n🤖  Sentinel opening…")
            llm_assistant.get_response_streaming()
            tts.wait_until_done()
            voice_extractor.mark_prompt_end()

            # ── Turn 1 ────────────────────────────────────────────────────────
            print("\n🎤  Turn 1 — listening (act drowsy)…")
            user_t1, audio_t1 = stt.listen(timeout=25, show_diagnostics=False)
            if not user_t1:
                user_t1 = "I don't know… I'm just really tired."

            feat_t1 = None
            if audio_t1 is not None:
                feat_t1 = voice_extractor.extract_features(audio_t1, user_t1)
                if feat_t1:
                    voice_accum.append(feat_t1)
                    dashboard.update_voice(feat_t1)

            det1  = det_thread.get_full_state()
            score_accum.append(det1.get("drowsy_score", 0))
            dctx1 = format_detection_for_llm(
                det1,
                microsleep=det1.get("microsleep", False),
                head_down=det1.get("head_down", False),
            )
            vctx1 = (voice_extractor.format_for_llm(feat_t1, baselines)
                     if feat_t1 else None)

            print(f"\n👤  Archit (T1): {user_t1}")
            llm_assistant.get_response_streaming(
                user_message=user_t1,
                detection_context=dctx1,
                voice_context=vctx1,
            )
            tts.wait_until_done()
            voice_extractor.mark_prompt_end()

            # ── Turn 2 ────────────────────────────────────────────────────────
            print("\n🎤  Turn 2 — listening (respond normally)…")
            user_t2, audio_t2 = stt.listen(timeout=25, show_diagnostics=False)
            if not user_t2:
                user_t2 = "Yeah, I feel much better now, thanks."

            feat_t2 = None
            if audio_t2 is not None:
                feat_t2 = voice_extractor.extract_features(audio_t2, user_t2)
                if feat_t2:
                    voice_accum.append(feat_t2)
                    dashboard.update_voice(feat_t2)

            det2 = det_thread.get_full_state()
            score_accum.append(det2.get("drowsy_score", 0))

            print(f"\n👤  Archit (T2): {user_t2}")

            # Crafted recovery referencing real voice measurements
            resp_t2 = craft_recovery_response(user_t2, feat_t2 or {}, baselines)
            print(f"\n🤖  Sentinel (recovery): {resp_t2}")

            # Keep transcripts consistent for fact extraction
            memory_manager.add_to_transcript("user", user_t2)
            tts_text = resp_t2.replace("[RECOVERED]", "").strip()
            tts.speak(tts_text)
            # Manually add to message history and transcript
            llm_assistant.messages.append(
                {"role": "assistant", "content": resp_t2})
            memory_manager.add_to_transcript("assistant", resp_t2)
            llm_assistant.conversation_turns += 1

            tts.wait_until_done()

            # Show RECOVERED state briefly
            dashboard.set_state("RECOVERED")
            time.sleep(2.5)

        finally:
            # ── Post-session bookkeeping ───────────────────────────────────────
            dur     = time.perf_counter() - session_start
            avg_sc  = sum(score_accum) / len(score_accum) if score_accum else 0
            max_sc  = max(score_accum)  if score_accum else 0

            avg_rms = avg_rate = avg_lat = None
            if voice_accum:
                rms_v  = [v["energy_rms"]        for v in voice_accum if v.get("energy_rms")]
                rate_v = [v["speech_rate_wpm"]   for v in voice_accum if v.get("speech_rate_wpm")]
                lat_v  = [v["response_latency_s"] for v in voice_accum if v.get("response_latency_s")]
                if rms_v:   avg_rms  = round(sum(rms_v)  / len(rms_v),  4)
                if rate_v:  avg_rate = round(sum(rate_v) / len(rate_v), 1)
                if lat_v:   avg_lat  = round(sum(lat_v)  / len(lat_v),  1)

            memory_manager.end_session(
                session_id,
                avg_drowsy_score=round(avg_sc, 3),
                max_drowsy_score=round(max_sc, 3),
                turn_count=llm_assistant.conversation_turns,
                duration_s=round(dur, 1),
                trigger_reason="camera",
                avg_energy_rms=avg_rms,
                avg_speech_rate=avg_rate,
                avg_response_latency=avg_lat,
            )

            print("\n💾  Extracting facts from conversation…")
            memory_manager.extract_and_store_facts(session_id)

            if voice_accum:
                avg_voice = {}
                for key in ["energy_rms", "speech_rate_wpm",
                             "pause_ratio", "response_latency_s"]:
                    vals = [v[key] for v in voice_accum if v.get(key) is not None]
                    if vals:
                        avg_voice[key] = sum(vals) / len(vals)
                if avg_voice:
                    memory_manager.update_baselines_bulk(avg_voice)
                    print(f"✓  Voice baselines updated from {len(voice_accum)} samples")

            conv_done.set()

    # ------------------------------------------------------------------
    conv_thread = threading.Thread(target=conv_worker, daemon=True)
    conv_thread.start()

    # ── Main thread: display loop ─────────────────────────────────────────────
    cv2.namedWindow(WIN_CAMERA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAMERA, 480, 360)
    cv2.moveWindow(WIN_CAMERA,  40, 120)
    cv2.namedWindow(WIN_DASH, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_DASH, DASH_W, DASH_H)
    cv2.moveWindow(WIN_DASH,  550, 60)

    while not conv_done.is_set():
        cam_f, _ = det_thread.get_display_frames()
        det_st   = det_thread.get_full_state()

        if cam_f is not None:
            _stamp_state(cam_f, dashboard._state)
            cv2.imshow(WIN_CAMERA, cam_f)

        dash_img = dashboard.render(det_st)
        cv2.imshow(WIN_DASH, dash_img)

        key = cv2.waitKey(33) & 0xFF
        if key == 27:
            conv_done.set()
            break

    conv_thread.join(timeout=35.0)   # Allow time for fact extraction API call
    det_thread.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return session_id


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not Config.GROQ_API_KEY:
        print("❌  GROQ_API_KEY not set in .env")
        return
    if not Config.DEEPGRAM_API_KEY:
        print("❌  DEEPGRAM_API_KEY not set in .env")
        return

    print("\n" + "=" * 58)
    print("  SENTINEL — Science Fair Demo")
    print("  Groq 70B + 8B  ·  Deepgram Nova-3  ·  MediaPipe")
    print("=" * 58 + "\n")

    # ── Seed demo database ────────────────────────────────────────────────────
    print("🗄   Seeding demo database…")
    memory_manager = seed_demo_database(DEMO_DB_PATH)

    # ── Init engines ──────────────────────────────────────────────────────────
    print("🔊  Initializing TTS…")
    tts = TTSEngine()
    print("🎤  Initializing STT…")
    stt = STTEngine()

    voice_extractor = VoiceFeatureExtractor()
    llm_assistant   = DemoLLMAssistant(tts, memory_manager)

    # Load baselines into reasoner
    baselines = memory_manager.get_baselines()
    reasoner  = MetricReasoner()
    reasoner.set_voice_baselines(baselines)
    patterns = memory_manager.get_driver_patterns_for_reasoner()
    if patterns:
        reasoner.set_driver_patterns(patterns)

    # Shared dashboard (both monitoring and conversation use it)
    dashboard = DemoDashboard(baselines=baselines)

    # ── ACT 1: DB Card (before) ───────────────────────────────────────────────
    print("\n📊  Showing database card…")
    if not show_db_card(memory_manager, mode="before"):
        print("Demo exited.")
        _shutdown(tts, stt, memory_manager)
        return

    # ── ACT 2–3: Monitoring + trigger ────────────────────────────────────────
    print("\n🎥  Starting monitoring…")
    triggered, final_metrics = run_monitoring_phase(reasoner, dashboard)

    if not triggered or final_metrics is None:
        print("Demo exited during monitoring.")
        _shutdown(tts, stt, memory_manager)
        return

    # ── Opening line (spoken, not skipped) ───────────────────────────────────
    freq        = memory_manager.get_drowsy_frequency()
    opening, _  = _build_opening_line(freq)
    print(f"\n🔊  Opening: {opening}")
    tts.speak(opening)
    tts.wait_until_done()

    # ── ACT 4: Conversation ───────────────────────────────────────────────────
    run_conversation_phase(
        tts, stt, llm_assistant, final_metrics,
        memory_manager, reasoner, voice_extractor, dashboard,
    )

    # ── ACT 5: DB Card (after) ────────────────────────────────────────────────
    time.sleep(0.5)
    print("\n📊  Showing post-session database card…")
    # Re-open memory manager to get freshest data from disk
    memory_manager.close()
    mm_after = MemoryManager(db_path=DEMO_DB_PATH)
    show_db_card(mm_after, mode="after")
    mm_after.close()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n✓  Demo complete.")
    tts.shutdown()
    stt.cleanup()


def _shutdown(tts, stt, mm):
    try:
        tts.shutdown()
    except Exception:
        pass
    try:
        stt.cleanup()
    except Exception:
        pass
    try:
        mm.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
