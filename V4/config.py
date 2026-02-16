"""
Configuration for Driver Drowsiness Detection System V3.2
"""


class Config:
    # API Keys (set these as environment variables or here)
    GROQ_API_KEY = "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"  # Get free at console.groq.com

    # API settings
    GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and good quality
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
