"""Configuration for Driver Drowsiness Detection System V5

V5 additions:
  - Voice feature extraction for LLM context
  - Parallel detection during conversation
  - LLM-based post-session memory extraction
  - Conversation history capping
  - Multi-threaded architecture
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (parent of V5/)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Config:
    # ── API Keys ──
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # ── API settings ──
    GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_EXTRACTION_MODEL = "llama-3.1-8b-instant"  # Cheap/fast model for post-session extraction
    EDGE_TTS_VOICE = "en-US-JennyNeural"
    EDGE_TTS_RATE = "+35%"

    # ── Camera settings (RPi optimization) ──
    CAMERA_WIDTH = 480
    CAMERA_HEIGHT = 360
    CAMERA_FPS = 20

    # ── Detection thresholds ──
    EAR_THRESH = 0.25
    MAR_THRESH = 0.6
    MICROSLEEP_TIME = 1.5
    SLOW_BLINK_TIME = 0.4
    HEAD_DOWN_THRESH = 0.12
    HEAD_DOWN_TIME = 1.2
    HEAD_ROLL_THRESH = 15
    ROLL_TIME = 1.2
    WINDOW_TIME = 10
    DROWSY_THRESHOLD = 0.47
    DROWSY_TRIGGER_COUNT = 10

    # ── Conversation settings ──
    MAX_HISTORY_TURNS = 8        # Keep only last N turn-pairs in context window
    MAX_CONVERSATION_TURNS = 15  # Absolute max turns per conversation

    # ── Voice feature thresholds ──
    VOICE_LOW_ENERGY_THRESH = 0.02   # RMS below this = very quiet/drowsy speech
    VOICE_LOW_RATE_THRESH = 80       # Words per minute below this = sluggish
    VOICE_SLOW_RESPONSE_THRESH = 8.0 # Seconds to start speaking — above = very slow

    # ── Parallel detection during conversation ──
    DETECTION_THREAD_FPS = 10   # Lower FPS during conversation to save CPU
    DETECTION_FRAME_SKIP = 3    # Process every Nth frame during conversation
