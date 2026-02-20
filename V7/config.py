"""Configuration for Driver Drowsiness Detection System V7

V7 changes:
  - LLM as reasoning layer: raw metrics + baselines replace hardcoded thresholds
  - SQLite replaces JSON profile (facts, sessions, baselines tables)
  - Calibration script for personal voice baselines
  - Dynamic fact extraction with free-form types
  - Deviation-based reasoning: LLM compares current vs personal baseline
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (parent of V7/)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Config:
    # ── API Keys ──
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

    # ── API settings ──
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_EXTRACTION_MODEL = "llama-3.1-8b-instant"  # Cheap model for post-session extraction

    # ── Deepgram STT settings ──
    DEEPGRAM_STT_MODEL = "nova-3"

    # ── Deepgram TTS settings ──
    DEEPGRAM_TTS_VOICE = "aura-2-thalia-en"
    DEEPGRAM_TTS_SPEED = 1.15

    # ── Camera settings (RPi optimization) ──
    CAMERA_WIDTH = 480
    CAMERA_HEIGHT = 360
    CAMERA_FPS = 20

    # ── Detection thresholds (trigger only — LLM reasons about severity) ──
    EAR_THRESH = 0.25
    MAR_THRESH = 0.6
    MICROSLEEP_TIME = 1.5
    SLOW_BLINK_TIME = 0.4
    HEAD_DOWN_THRESH = 0.12
    HEAD_DOWN_TIME = 1.2
    HEAD_ROLL_THRESH = 15
    ROLL_TIME = 1.2
    WINDOW_TIME = 10
    DROWSY_THRESHOLD = 0.47  # Used only for DetectionThread overlay label

    # ── Conversation settings ──
    MAX_HISTORY_TURNS = 8
    MAX_CONVERSATION_TURNS = 15

    # ── Parallel detection during conversation ──
    DETECTION_THREAD_FPS = 10
    DETECTION_FRAME_SKIP = 3

    # ── 8B MetricReasoner (replaces hardcoded drowsy_score formula) ──
    GROQ_REASONER_MODEL = "llama-3.1-8b-instant"  # Same cheap model as extraction
    REASONER_PRE_FILTER = 0.30       # Local score must exceed this before calling 8B
    REASONER_INTERVAL_S = 3.0        # Min seconds between 8B API calls
    REASONER_CONFIRM_COUNT = 3       # Consecutive DROWSY/CRITICAL before triggering
    REASONER_HISTORY_SIZE = 10       # Rolling history snapshots for trend analysis

    # ── Calibration ──
    CALIBRATION_SENTENCES = 5  # Number of sentences for voice baseline calibration
