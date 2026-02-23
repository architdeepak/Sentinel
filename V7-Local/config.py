"""Configuration for Driver Drowsiness Detection System V7-Local.

V7-Local changes from V7:
  - All API keys removed — fully local operation
  - LLM: llama-cpp (local GGUF model) replaces Groq 70B/8B
  - STT: Vosk + PyAudio replaces Deepgram Nova-3
  - TTS: espeak-ng replaces Deepgram Aura
  - MetricReasoner: hardcoded thresholds replace Groq 8B
"""

from pathlib import Path


class Config:
    # ── Local model paths ──
    LLM_MODEL_PATH = Path.home() / "Sentinel" / "modls" / "granite-3.0-1b-a400m-instruct.Q4_K_M.gguf"
    VOSK_MODEL_PATH = Path.home() / "Sentinel" / "Sentinel" / "vosk-model-small-en-us-0.15"

    # ── LLM settings (llama-cpp) ──
    LLM_THREADS = 3
    LLM_CONTEXT = 4096   # Granite 1B Q4 uses ~800MB RAM; 4096 context fits RPi 4 (4GB)
    LLM_MAX_TOKENS = 120  # Keep responses short — 2-3 sentences is ~80-100 tokens

    # ── espeak-ng TTS settings ──
    ESPEAK_SPEED = 165
    ESPEAK_VOICE = "en-us"

    # ── Vosk STT settings ──
    VOSK_SAMPLE_RATE = 16000
    VOSK_BUFFER_SIZE = 8192

    # ── Camera settings (RPi optimization) ──
    CAMERA_WIDTH = 480
    CAMERA_HEIGHT = 360
    CAMERA_FPS = 20

    # ── Processing resolution (detection runs at this size) ──
    PROC_WIDTH = 320
    PROC_HEIGHT = 240

    # ── Detection thresholds (trigger only — reasoner decides severity) ──
    EAR_THRESH = 0.20          # Default fallback; overridden by EAR calibration at startup
    MAR_THRESH = 0.6
    MICROSLEEP_TIME = 1.5
    SLOW_BLINK_TIME = 0.55       # Only genuinely slow/droopy blinks (normal ≤ 0.4s)
    HEAD_DOWN_THRESH = 0.12
    HEAD_DOWN_TIME = 1.2
    HEAD_ROLL_THRESH = 15
    ROLL_TIME = 1.2
    WINDOW_TIME = 10
    DROWSY_THRESHOLD = 0.55  # Used only for DetectionThread overlay label

    # ── Conversation settings ──
    MAX_HISTORY_TURNS = 4    # Small model: 4 turn-pairs = ~400 tokens of history
    MAX_CONVERSATION_TURNS = 15

    # ── Parallel detection during conversation ──
    DETECTION_THREAD_FPS = 10
    DETECTION_FRAME_SKIP = 3

    # ── MetricReasoner (local thresholds replacing 8B) ──
    REASONER_PRE_FILTER = 0.55       # Local score must exceed this before calling reasoner
    REASONER_INTERVAL_S = 3.0        # Min seconds between reasoner calls
    REASONER_CONFIRM_COUNT = 3       # Consecutive DROWSY/CRITICAL before triggering
    REASONER_HISTORY_SIZE = 10       # Rolling history snapshots for trend analysis

    # ── Auto-recovery (end conversation when driver is alert) ──
    ALERT_RECOVERY_SECS = 90   # Seconds of sustained alert before auto-ending conversation

    # ── Calibration ──
    CALIBRATION_SENTENCES = 5  # Number of sentences for voice baseline calibration
