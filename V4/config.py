"""Configuration for Driver Drowsiness Detection System V4"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (parent of V4/)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Config:
    # API Keys (loaded from .env file — never hardcode these!)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # API settings
    GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast and good quality
    EDGE_TTS_VOICE = "en-US-JennyNeural"  # Options: AriaNeural, GuyNeural, JennyNeural
    EDGE_TTS_RATE = "+25%"  # Speed adjustment

    # Camera settings (RPi optimization)
    CAMERA_WIDTH = 480
    CAMERA_HEIGHT = 360
    CAMERA_FPS = 20

    # Detection thresholds
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
