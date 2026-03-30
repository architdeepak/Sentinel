# Sentinel: Driver Drowsiness Detection + Conversational Safety Assistant

Sentinel is a real-time driver safety system that combines camera-based drowsiness detection, voice analysis, and conversational intervention.

The project includes two main runtime variants:

- `V8/`: cloud-assisted mode (Groq + Deepgram)
- `V8-Local/`: fully offline mode (llama-cpp + Vosk + espeak-ng)

It was built for Raspberry Pi deployment, with development and testing often done on Windows.

## What Sentinel Does

1. Monitors the driver in real time using a camera (eye closure, blink behavior, head motion, etc.).
2. Computes rolling drowsiness metrics (for example `perclos`, `slow_blinks`, `drowsy_score`).
3. Uses a reasoning gate to reduce false positives before triggering intervention.
4. Starts a short, engaging safety conversation when drowsiness is confirmed.
5. Tracks voice features during conversation (energy, speech rate, response latency, pauses).
6. Stores sessions, learned facts, baselines, and patterns in SQLite to personalize future runs.

## High-Level Architecture

### Shared pipeline

- **Vision detection**: `detection.py`
- **Voice feature extraction**: `voice_features.py`
- **Session memory + SQLite**: `memory.py`
- **Conversation orchestration**: `main.py`
- **Live dashboard**: `dashboard.py`

### `V8` (Cloud)

- **Reasoner**: Groq 8B model (`metric_reasoner.py`)
- **Conversation model**: Groq 70B (`llm_assistant.py`)
- **STT/TTS**: Deepgram (`stt_engine.py`, `tts_engine.py`)
- **Uses `.env` keys**: `GROQ_API_KEY`, `DEEPGRAM_API_KEY`

### `V8-Local` (Offline)

- **Reasoner**: local threshold logic (`metric_reasoner.py`)
- **Conversation model**: local GGUF via `llama-cpp`
- **STT**: Vosk
- **TTS**: `espeak-ng`
- **No external API calls** once local models are installed and configured

## Repository Layout

- `V1` ... `V8`: project evolution by version
- `V8/`: latest cloud-connected runtime
- `V8-Local/`: offline/local runtime
- `EXAMPLES/`: experiments and utility examples
- `requirements.txt`: Python dependencies
- `V8_V8Local_Deep_Guide.txt`: detailed comparison notes

## Requirements

### Hardware

- Camera (USB or Pi camera)
- Microphone
- Speaker/audio output
- Raspberry Pi 4+ recommended for deployment

### Software

- Python 3.10+
- OpenCV-compatible camera drivers
- For cloud mode (`V8`): internet access + API keys
- For local mode (`V8-Local`): local model files and native audio deps (`espeak-ng`, PyAudio-compatible stack)

### Python Dependencies

Install from the root:

```bash
pip install -r requirements.txt
```

Note: Some packages are mode-specific (for example `groq`/Deepgram-related for cloud mode, `vosk`/`llama-cpp-python` for local mode).

## Environment Setup (Cloud mode)

Create/update `.env` in the repo root (`ScienecFair25-26/.env`):

```env
GROQ_API_KEY=your_groq_key
DEEPGRAM_API_KEY=your_deepgram_key
```

`V8/config.py` reads this file automatically.

## Running the Project

### Cloud Mode (`V8`)

```bash
cd V8
python main.py
```

Optional commands:

```bash
python main.py --calibrate   # voice baseline calibration only
python main.py --dump-db     # print SQLite contents
python main.py --reset-db    # reset SQLite database
```

### Offline Mode (`V8-Local`)

```bash
cd V8-Local
python main.py
```

Same optional flags are supported:

```bash
python main.py --calibrate
python main.py --dump-db
python main.py --reset-db
```

Before running local mode, verify model paths in `V8-Local/config.py`:

- `LLM_MODEL_PATH`
- `VOSK_MODEL_PATH`

## How Detection and Conversation Work

1. **Startup calibration**
- Voice baseline calibration runs to capture personal normal ranges.
- EAR (eye aspect ratio) baseline calibration runs for personalized eye-closure threshold.

2. **Continuous monitoring loop**
- Frames are processed with MediaPipe FaceMesh.
- Metrics are updated over rolling windows.
- A reasoning gate confirms drowsiness before conversation starts.

3. **Conversation mode**
- Main loop switches to an interactive conversation.
- A background detection thread keeps monitoring while the assistant talks.
- Voice metrics are injected each turn for richer assessment.

4. **Post-session learning**
- Session analytics are stored in SQLite.
- Facts and patterns are extracted and reused for personalization.
- Baselines update over time.

## Database / Memory

Sentinel stores long-term memory in SQLite (default file in the user home directory).

Main tables include:

- `sessions`
- `facts`
- `baselines`
- `reasoner_evaluations`
- `driver_patterns`

Use `--dump-db` to inspect saved data.

## Safety Notes

- This is a student/research prototype, not a certified medical or automotive safety device.
- It should be treated as an assistive layer, not a replacement for safe-driving practices.
- Real-world deployment requires additional validation, hardware hardening, and fail-safe design.

## Troubleshooting

- **Camera not opening**: check camera index/device permissions and whether another app is holding the camera.
- **No STT/TTS output**: verify microphone/speaker routing and required local audio packages.
- **Cloud mode API errors**: confirm `.env` keys and internet connectivity.
- **Local mode model load failures**: verify `LLM_MODEL_PATH` and `VOSK_MODEL_PATH` in `V8-Local/config.py`.

## Project Status

Active development. New versions and tuning changes are tracked in versioned folders (`V1` to `V8`, plus local variants).
