"""
Configuration management for Driver Drowsiness Detection System V3
Handles API keys and environment variables securely
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """
    Application configuration with secure API key management.
    API keys are loaded from environment variables (.env file).
    """
    
    # API Keys (loaded from .env file - NEVER hardcode these!)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    
    # Validate that required API keys are present
    @classmethod
    def validate(cls):
        """Validate that all required API keys are configured."""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "❌ GROQ_API_KEY not found in environment variables.\n"
                "   Please create a .env file in the V3 folder with your API key.\n"
                "   See .env.example for template."
            )
        return True
    
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
