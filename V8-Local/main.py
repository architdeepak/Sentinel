#!/usr/bin/env python3
"""
Driver Drowsiness Detection System V7-Local
Main entry point — multi-threaded architecture.

V7-Local architecture (fully offline — zero API calls):
  - LLM: llama-cpp local GGUF model (replaces Groq 70B)
  - STT: Vosk + PyAudio (replaces Deepgram Nova-3)
  - TTS: espeak-ng (replaces Deepgram Aura)
  - MetricReasoner: hardcoded thresholds (replaces Groq 8B)
  - SQLite memory: facts, sessions, baselines
  - Post-session: local LLM extracts facts, baselines updated

Usage:
    python main.py                  # Run the system
    python main.py --calibrate      # Force voice baseline calibration
    python main.py --dump-db        # Print all SQLite data
"""

import cv2
import mediapipe as mp
import signal
import subprocess
import time
import threading
import numpy as np
from collections import deque

from config import Config
from memory import MemoryManager
from tts_engine import TTSEngine
from stt_engine import STTEngine
from llm_assistant import LLMAssistant
from voice_features import VoiceFeatureExtractor
from detection import (
    eye_aspect_ratio,
    LEFT_EYE, RIGHT_EYE,
    process_eye_metrics,
    process_mouth_metrics,
    process_head_pitch,
    process_head_roll,
    cleanup_windows,
    calculate_metrics,
    draw_overlay,
    DetectionThread,
    format_detection_for_llm,
)
from metric_reasoner import MetricReasoner
from dashboard import DashboardRenderer


# =========================
# ALARM + OPENING LINE
# =========================
def play_alarm(duration=4.0):
    """Play an urgent two-tone siren through ALSA (aplay) — no files needed.

    Generates a 880/1760 Hz alternating sine wave as raw PCM in memory
    and pipes it directly to aplay. Works on RPi with no extra packages.
    Falls back silently if aplay is unavailable.
    """
    try:
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Alternate between 880 Hz and 1760 Hz every 0.25 s (urgent police-style)
        freq = np.where((t * 4).astype(int) % 2 == 0, 880.0, 1760.0)
        wave = (np.sin(2 * np.pi * freq * t) * 28000).astype(np.int16)
        proc = subprocess.Popen(
            ["aplay", "-r", str(sample_rate), "-f", "S16_LE", "-c", "1", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(wave.tobytes())
        proc.stdin.close()
        try:
            proc.wait(timeout=duration + 2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    except (FileNotFoundError, OSError):
        print("⚠️ play_alarm: aplay not found — skipping alarm sound")
    except Exception as e:
        print(f"⚠️ play_alarm error: {e}")


def build_opening_line(freq):
    """Return a dynamic drowsy-detection announcement based on session history.

    Args:
        freq: dict from MemoryManager.get_drowsy_frequency()

    Returns:
        (opening_tts: str, is_serious: bool)
    """
    today = freq["today_count"]
    last_2h = freq["last_2h_count"]
    severity = freq["severity"]

    if severity == "critical":
        line = (
            "Warning. This is now a serious concern — "
            f"you've been drowsy {today} times today. "
            "You really need to pull over at the nearest safe spot."
        )
    elif severity == "serious":
        if last_2h >= 2:
            line = (
                f"Hey — that's the {last_2h + 1} time in the last two hours. "
                "I'm getting concerned. Let's talk, but please think about pulling over."
            )
        else:
            line = (
                f"Hey, this is the {today + 1}th time today you've been drowsy. "
                "I'm here to help, but this is becoming a pattern."
            )
    elif severity == "elevated":
        line = (
            "Hey, looks like you're feeling drowsy again. I'm here with you — let's talk."
        )
    elif today == 0:
        line = (
            "Hey, I'm detecting some drowsiness. I'm Sentinel — I'm here to help."
        )
    else:
        line = "Hey, feeling drowsy again? I've got you — let's get you back on track."

    is_serious = severity in ("serious", "critical")
    return line, is_serious


# =========================
# INITIALIZATION
# =========================
def initialize_mediapipe():
    """Initialize MediaPipe face mesh."""
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return face_mesh


def initialize_camera():
    """Initialize camera with optimized settings."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def initialize_state_variables():
    """Initialize all state tracking variables."""
    _win = 200
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
        'llm_triggered': False,
        'window_start_time': time.time()
    }


def run_ear_calibration(cap, face_mesh, duration=5.0):
    """Measure the driver's personal EAR baseline over `duration` seconds.

    Returns a calibrated EAR closed-eye threshold, or the config default
    if calibration fails (e.g. face not detected).
    """
    print(f"\n👁️  EAR Calibration — look at the camera with eyes open for {duration:.0f}s...")
    PROC_W, PROC_H = Config.PROC_WIDTH, Config.PROC_HEIGHT
    ear_samples = []
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        small = cv2.resize(frame, (PROC_W, PROC_H))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks = [(int(p.x * PROC_W), int(p.y * PROC_H)) for p in lm]
            ear = (eye_aspect_ratio(landmarks, LEFT_EYE) +
                   eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2
            ear_samples.append(ear)

        # Show progress
        elapsed = time.time() - start
        cv2.putText(frame, f"EAR Calibration: {elapsed:.1f}/{duration:.0f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Driver Drowsiness Monitor", frame)
        cv2.waitKey(1)

    if len(ear_samples) < 20:
        print(f"  ⚠️ Not enough face detections ({len(ear_samples)}) — using default EAR threshold {Config.EAR_THRESH}")
        return Config.EAR_THRESH

    import numpy as np
    ear_arr = np.array(ear_samples)
    avg_ear = float(np.mean(ear_arr))
    std_ear = float(np.std(ear_arr))

    # Threshold = 75% of their average open-eye EAR
    # This accounts for individual eye geometry
    calibrated_thresh = round(avg_ear * 0.75, 3)

    # Safety clamp: never go below 0.15 or above 0.28
    calibrated_thresh = max(0.15, min(0.28, calibrated_thresh))

    print(f"  ✓ EAR Calibration complete: avg={avg_ear:.3f}, std={std_ear:.3f}")
    print(f"  ✓ Personal EAR threshold: {calibrated_thresh} (vs default {Config.EAR_THRESH})")
    return calibrated_thresh


# =========================
# VOICE CALIBRATION
# =========================
def run_calibration(stt, voice_extractor, memory_manager):
    """Record baseline voice metrics from the driver speaking normally.

    Runs at first launch (no baselines in DB) or when --calibrate is passed.
    Records CALIBRATION_SENTENCES sentences and stores the averages.
    """
    print("\n" + "=" * 60)
    print("🎙️  VOICE BASELINE CALIBRATION")
    print("=" * 60)
    print("   I need to learn what your voice sounds like when you're alert.")
    print(f"   Please read {Config.CALIBRATION_SENTENCES} sentences in your normal voice.")
    print("=" * 60)

    prompts = [
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "I'm driving to my destination and everything looks clear ahead.",
        "Today has been a really productive day at work, I got a lot done.",
        "My favorite thing to do on weekends is relax with friends and family.",
        "The weather forecast says it will be sunny and warm tomorrow afternoon.",
        "I just picked up some groceries and I'm heading home for dinner now.",
        "There's a great restaurant downtown that serves amazing Italian food.",
    ]

    samples = []

    for i in range(Config.CALIBRATION_SENTENCES):
        prompt = prompts[i % len(prompts)]
        print(f"\n  [{i+1}/{Config.CALIBRATION_SENTENCES}] Please read aloud:")
        print(f'  "{prompt}"')
        print("  🎤 Listening...")

        voice_extractor.mark_prompt_end()
        text, audio = stt.listen(timeout=15, show_diagnostics=False)

        if audio is not None:
            features = voice_extractor.extract_features(audio, text)
            if features:
                samples.append(features)
                print(f"  ✓ Captured: RMS={features['energy_rms']:.4f}, "
                      f"rate={features.get('speech_rate_wpm', 'N/A')} wpm, "
                      f"pause_ratio={features['pause_ratio']:.3f}")
            else:
                print("  ⚠️ Couldn't extract features — try again")
        else:
            print("  ⚠️ No audio captured — try again")

    if len(samples) >= 2:
        memory_manager.store_calibration_baselines(samples)
        print(f"\n✓ Calibration complete — stored baselines from {len(samples)} samples")
    else:
        print("\n⚠️ Not enough samples for calibration (need at least 2)")
        print("   Run `python main.py --calibrate` to try again")

    print("=" * 60 + "\n")


# =========================
# MAIN DETECTION LOOP
# =========================
def run_detection_loop(cap, face_mesh, state, reasoner, ear_thresh=None):
    """Run the main drowsiness detection loop with 8B reasoning gate.

    Flow:
      1. Every frame: compute local metrics (fast, no API)
      2. Pre-filter: if local drowsy_score > REASONER_PRE_FILTER
         AND enough time since last 8B call → call MetricReasoner
      3. Reasoner confirms DROWSY/CRITICAL N consecutive times → trigger
      4. Microsleep bypass: instant trigger (too critical for API latency)
    """
    frame_skip = Config.DETECTION_FRAME_SKIP
    frame_count = 0
    final_metrics = None
    PROC_W, PROC_H = Config.PROC_WIDTH, Config.PROC_HEIGHT

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        now = time.time()

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

            ear, microsleep = process_eye_metrics(landmarks, state, now, ear_thresh=ear_thresh)
            mar = process_mouth_metrics(landmarks, state, now)
            head_down = process_head_pitch(landmarks, state, now)
            head_roll = process_head_roll(landmarks, state, now)

        cleanup_windows(state, now)
        metrics = calculate_metrics(state, microsleep, head_down, head_roll)

        # ── 8B Reasoning Gate ──
        drowsy_state = "ALERT"
        reasoner_result = reasoner.get_last_result()

        # Microsleep bypass — instant trigger, no API latency
        if microsleep and not state['llm_triggered']:
            print("\n🚨 MICROSLEEP DETECTED — instant trigger (bypassing 8B)")
            state['llm_triggered'] = True
            final_metrics = metrics.copy()
            final_metrics['microsleep'] = True
            final_metrics['head_down'] = head_down
            drowsy_state = "CRITICAL"
            draw_overlay(frame, metrics, ear, mar, microsleep, head_down,
                         head_roll, state, drowsy_state, landmarks=landmarks,
                         proc_size=(PROC_W, PROC_H))
            cv2.imshow("Driver Drowsiness Monitor", frame)
            cv2.waitKey(1)
            return True, final_metrics

        # Pre-filter gate: only call 8B when local score suggests possible drowsiness
        if (metrics['drowsy_score'] > Config.REASONER_PRE_FILTER and
                reasoner.should_call()):
            result = reasoner.evaluate(metrics, microsleep, head_down, head_roll)
            reasoner_result = result
            level_label = f"{result.level} (conf={result.confidence:.2f})"
            confirm = reasoner.get_confirmation_count()
            print(f"\n🧠 8B Reasoner: {level_label} [{confirm}/{Config.REASONER_CONFIRM_COUNT}] — {result.reasoning}")

        # Update display state from reasoner
        if reasoner_result.is_drowsy():
            drowsy_state = reasoner_result.level  # DROWSY or CRITICAL

        draw_overlay(frame, metrics, ear, mar, microsleep, head_down,
                     head_roll, state, drowsy_state, landmarks=landmarks,
                     proc_size=(PROC_W, PROC_H))

        cv2.imshow("Driver Drowsiness Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("📊 Monitoring ended by user")
            return False, None

        # Trigger conversation when 8B confirms drowsiness N consecutive times
        if (reasoner.is_confirmed_drowsy() and not state['llm_triggered']):
            print("\n🚨 DROWSINESS DETECTED — triggering conversation")
            state['llm_triggered'] = True
            final_metrics = metrics.copy()
            final_metrics['microsleep'] = microsleep
            final_metrics['head_down'] = head_down
            return True, final_metrics

    return False, None


# =========================
# LLM CONVERSATION (V7 — raw metrics + baselines + SQLite)
# =========================
def run_llm_conversation(tts, stt, llm_assistant, metrics, state,
                         voice_extractor, memory_manager, reasoner=None,
                         ear_thresh=None, drowsy_freq=None):
    """Run conversation loop with raw metric injection and SQLite session tracking.

    Architecture (Linux/RPi compatible):
      - Main thread: display loop (cv2.imshow) — required on Linux
      - Conversation thread: STT/LLM/TTS blocking calls
      - DetectionThread: camera + metrics + renders frames to shared buffer
      - DashboardRenderer: renders metrics panel (called by DetectionThread)
    """
    print("\n" + "=" * 60)
    print("💬 STARTING CONVERSATION (V7 — raw metrics + baselines + SQLite)")
    print("=" * 60)

    # Start SQLite session
    session_id = memory_manager.start_session()
    llm_assistant._session_id = session_id
    session_start = time.perf_counter()

    # Track detection scores for session averages
    score_accumulator = []
    voice_accumulator = []
    perclos_accumulator = []
    slow_blinks_accumulator = []

    # Get baselines for voice comparison
    baselines = memory_manager.get_baselines()
    baselines_str = memory_manager.format_baselines_for_llm()

    # Dashboard renderer (stateless — called by DetectionThread each frame)
    dashboard = DashboardRenderer(baselines=baselines)

    # Detection thread (own camera + FaceMesh) — renders frames to buffer
    detection_thread = DetectionThread(dashboard=dashboard, ear_thresh=ear_thresh)
    detection_thread.start()

    # Format initial detection context (raw numbers)
    detection_context = format_detection_for_llm(
        metrics,
        microsleep=metrics.get('microsleep', False),
        head_down=metrics.get('head_down', False),
    )

    # Get 8B reasoning context for the 70B conversation model
    reasoner_context = ""
    if reasoner is not None:
        reasoner_context = reasoner.get_reasoning_for_llm()

    # Get driver drowsiness history for LLM personalization
    driver_history = memory_manager.get_driver_history_for_llm()

    # Start conversation with raw metrics + baselines + 8B analysis + session history
    session_count = memory_manager.get_session_count() - 1  # -1 because we just started this one
    llm_assistant.start_conversation(
        detection_context, baselines_str, max(0, session_count),
        reasoner_context=reasoner_context,
        driver_history=driver_history,
        drowsy_freq=drowsy_freq,
    )

    # ── Conversation worker (runs in background thread) ──
    conversation_done = threading.Event()

    def _conversation_worker():
        """Full conversation logic — runs in a background thread so the
        main thread can keep refreshing the display on Linux/RPi."""

        # Get opening message
        opening = llm_assistant.get_response_streaming()
        tts.wait_until_done()
        voice_extractor.mark_prompt_end()

        max_turns = Config.MAX_CONVERSATION_TURNS

        def _handle_user_turn(text, audio):
            """Extract raw features, get live detection, inject both to LLM.
            Also feeds voice features into the 8B reasoner for combined reasoning."""
            voice_context = None
            latest_voice_features = None
            if audio is not None:
                features = voice_extractor.extract_features(audio, text)
                if features:
                    voice_context = voice_extractor.format_for_llm(features, baselines)
                    voice_accumulator.append(features)
                    dashboard.update_voice(features)
                    latest_voice_features = features

            det_state = detection_thread.get_full_state()
            det_context = format_detection_for_llm(
                det_state,
                microsleep=det_state.get('microsleep', False),
                head_down=det_state.get('head_down', False),
            )
            score_accumulator.append(det_state.get('drowsy_score', 0))

            # Track peak vision metrics for session analytics
            perclos_accumulator.append(det_state.get('perclos', 0))
            slow_blinks_accumulator.append(det_state.get('slow_blinks', 0))

            # Feed voice features into the 8B reasoner for combined reasoning
            if (latest_voice_features and reasoner is not None
                    and reasoner.should_call()):
                r_result = reasoner.evaluate(
                    det_state,
                    microsleep=det_state.get('microsleep', False),
                    head_down=det_state.get('head_down', False),
                    voice_features=latest_voice_features,
                )
                dashboard.update_reasoner(r_result)

            response = llm_assistant.get_response_streaming(
                user_message=text,
                detection_context=det_context,
                voice_context=voice_context,
            )
            tts.wait_until_done()
            voice_extractor.mark_prompt_end()
            return response

        hard_exit = {'exit', 'quit', 'bye', 'goodbye', 'bye bye', 'stop talking'}
        consecutive_no_response = 0  # Tracks back-to-back silent turns

        def _check_alert_recovery():
            """Check if driver has been alert long enough to auto-end."""
            det = detection_thread.get_full_state()
            dur = det.get('alert_duration', 0)
            if dur >= Config.ALERT_RECOVERY_SECS:
                mins = int(dur // 60)
                secs = int(dur % 60)
                print(f"\n🟢 Driver alert for {mins}m{secs}s — auto-ending conversation")
                tts.speak(
                    "Hey, you've been looking sharp for a while now! "
                    "I'm going to hop off, but I'll keep watching. Stay safe!"
                )
                tts.wait_until_done()
                return True
            return False

        for turn in range(max_turns):
            user_input, audio_data = stt.listen(timeout=20, show_diagnostics=False)

            if user_input:
                consecutive_no_response = 0  # Reset on any response
                if user_input.lower().strip() in hard_exit:
                    print("🔚 User requested exit")
                    tts.speak("Alright, I'll be here if you need me. Stay safe!")
                    tts.wait_until_done()
                    break

                response = _handle_user_turn(user_input, audio_data)
                if response and "[RECOVERED]" in response:
                    print("🟢 LLM determined driver has recovered")
                    break
                if _check_alert_recovery():
                    break
            else:
                consecutive_no_response += 1
                print(f"⚠️  No response detected (consecutive: {consecutive_no_response})")

                if consecutive_no_response >= 2:
                    # Driver has not responded twice in a row — trigger emergency alarm
                    print("🚨 Driver unresponsive — triggering alarm")
                    tts.speak(
                        "ALERT. Driver unresponsive. "
                        "Please pull over at the nearest safe location immediately."
                    )
                    tts.wait_until_done()
                    # Play siren in a background thread so it doesn't block
                    alarm_thread = threading.Thread(
                        target=play_alarm, args=(5.0,), daemon=True
                    )
                    alarm_thread.start()
                    alarm_thread.join(timeout=6.0)
                    break

                tts.speak("Hey — are you still with me? Give me a quick response.")
                tts.wait_until_done()
                voice_extractor.mark_prompt_end()

                retry_input, retry_audio = stt.listen(timeout=15, show_diagnostics=False)
                if not retry_input:
                    # Count the retry miss toward consecutive counter too
                    consecutive_no_response += 1
                    if consecutive_no_response >= 2:
                        print("🚨 Driver unresponsive after retry — triggering alarm")
                        tts.speak(
                            "ALERT. Driver unresponsive. "
                            "Please pull over at the nearest safe location immediately."
                        )
                        tts.wait_until_done()
                        alarm_thread = threading.Thread(
                            target=play_alarm, args=(5.0,), daemon=True
                        )
                        alarm_thread.start()
                        alarm_thread.join(timeout=6.0)
                    else:
                        tts.speak("I'll keep monitoring. Stay alert!")
                        tts.wait_until_done()
                    break
                consecutive_no_response = 0
                response = _handle_user_turn(retry_input, retry_audio)
                if response and "[RECOVERED]" in response:
                    print("🟢 LLM determined driver has recovered")
                    break
                if _check_alert_recovery():
                    break

        conversation_done.set()

    # Start conversation in background thread
    conv_thread = threading.Thread(target=_conversation_worker, daemon=True)
    conv_thread.start()

    # ── Main thread: display loop (required on Linux/RPi) ──
    while not conversation_done.is_set():
        cam_frame, dash_frame = detection_thread.get_display_frames()
        if cam_frame is not None:
            cv2.imshow("Driver Drowsiness Monitor", cam_frame)
        if dash_frame is not None:
            cv2.imshow("Sentinel Dashboard", dash_frame)
        key = cv2.waitKey(33) & 0xFF  # ~30 FPS
        if key == 27:
            print("⚠️ ESC pressed — ending conversation")
            conversation_done.set()
            break

    # Wait for conversation thread to finish
    conv_thread.join(timeout=5.0)

    # Close conversation windows
    detection_thread.stop()
    cv2.destroyWindow("Driver Drowsiness Monitor")
    cv2.destroyWindow("Sentinel Dashboard")
    cv2.waitKey(1)

    # ── Post-session processing ──
    session_duration = time.perf_counter() - session_start
    avg_score = sum(score_accumulator) / len(score_accumulator) if score_accumulator else 0
    max_score = max(score_accumulator) if score_accumulator else 0

    # Compute richer session analytics
    peak_perclos = max(perclos_accumulator) if perclos_accumulator else None
    peak_slow = max(slow_blinks_accumulator) if slow_blinks_accumulator else None

    # Voice averages for the session
    avg_rms = avg_rate = avg_lat = None
    if voice_accumulator:
        rms_vals = [v['energy_rms'] for v in voice_accumulator if v.get('energy_rms') is not None]
        rate_vals = [v['speech_rate_wpm'] for v in voice_accumulator if v.get('speech_rate_wpm') is not None]
        lat_vals = [v['response_latency_s'] for v in voice_accumulator if v.get('response_latency_s') is not None]
        if rms_vals:
            avg_rms = round(sum(rms_vals) / len(rms_vals), 4)
        if rate_vals:
            avg_rate = round(sum(rate_vals) / len(rate_vals), 1)
        if lat_vals:
            avg_lat = round(sum(lat_vals) / len(lat_vals), 1)

    # Recovery time = alert_duration at conversation end (0 if never recovered)
    final_det = detection_thread.get_full_state()
    recovery_time = final_det.get('alert_duration', 0)

    # Determine what triggered the conversation
    trigger_reason = "camera"
    if metrics.get('microsleep', False):
        trigger_reason = "microsleep"

    # End session in SQLite
    memory_manager.end_session(
        session_id,
        avg_drowsy_score=round(avg_score, 3),
        max_drowsy_score=round(max_score, 3),
        turn_count=llm_assistant.conversation_turns,
        duration_s=round(session_duration, 1),
        recovery_time_s=round(recovery_time, 1) if recovery_time else None,
        peak_perclos=round(peak_perclos, 4) if peak_perclos is not None else None,
        peak_slow_blinks=peak_slow,
        avg_energy_rms=avg_rms,
        avg_speech_rate=avg_rate,
        avg_response_latency=avg_lat,
        trigger_reason=trigger_reason,
    )

    # LLM-based fact extraction (8B model — cheap)
    print("\n💾 Running LLM-based fact extraction...")
    memory_manager.extract_and_store_facts(session_id)

    # Store 8B reasoner evaluations and learn driver patterns
    if reasoner is not None:
        eval_log = reasoner.get_evaluation_log()
        if eval_log:
            memory_manager.store_reasoner_evaluations(session_id, eval_log)
            print(f"🧠 Running driver pattern learning ({len(eval_log)} evaluations)...")
            memory_manager.learn_driver_patterns(session_id)
            # Reload patterns into reasoner for next session
            new_patterns = memory_manager.get_driver_patterns_for_reasoner()
            if new_patterns:
                reasoner.set_driver_patterns(new_patterns)

    # Update voice baselines from this session's voice data
    if voice_accumulator:
        avg_voice = {}
        for key in ['energy_rms', 'speech_rate_wpm', 'pause_ratio',
                     'response_latency_s']:
            vals = [v[key] for v in voice_accumulator if v.get(key) is not None]
            if vals:
                avg_voice[key] = sum(vals) / len(vals)
        if avg_voice:
            memory_manager.update_baselines_bulk(avg_voice)
            print(f"✓ Voice baselines updated from {len(voice_accumulator)} samples")
            # Refresh reasoner's baselines with latest data
            if reasoner is not None:
                reasoner.set_voice_baselines(memory_manager.get_baselines())

    print("\n" + "=" * 60)
    print("✓ Session complete — Facts extracted — Baselines updated — Resuming monitoring")
    print("=" * 60)


# =========================
# MAIN FUNCTION
# =========================
def main():
    """Main function — V7-Local drowsiness detection with local LLM + SQLite memory."""
    # ── Graceful shutdown on SIGTERM (e.g. systemd stop) ──
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("\n" + "=" * 60)
    print("🚗 Driver Drowsiness Detection System V7-Local")
    print("   Local LLM (llama-cpp) | Vosk STT | espeak-ng TTS")
    print("   SQLite memory | Hardcoded reasoner | Zero API calls")
    print("=" * 60 + "\n")

    # Initialize SQLite memory
    memory_manager = MemoryManager()
    session_count = memory_manager.get_session_count()
    last_time = memory_manager.get_last_session_time()
    print(f"📊 Database Status:")
    print(f"   Sessions: {session_count}")
    if last_time:
        print(f"   Last session: {last_time}")
    fact_count = len(memory_manager.get_context_facts(limit=100))
    if fact_count:
        print(f"   Known facts: {fact_count}")
    print()

    tts = TTSEngine()
    stt = STTEngine()
    voice_extractor = VoiceFeatureExtractor()
    llm_assistant = LLMAssistant(tts, memory_manager)
    # Share the single Llama instance with MemoryManager (avoids loading two models)
    memory_manager.set_llm(llm_assistant.llm)

    # Voice calibration — always recalibrate on startup for fresh baselines
    print("🎙️ Running voice baseline calibration...")
    tts.speak("Quick voice calibration. Please read a few sentences in your normal voice.")
    tts.wait_until_done()
    run_calibration(stt, voice_extractor, memory_manager)
    bls = memory_manager.get_baselines()
    rms_bl = bls.get('energy_rms', {})
    rate_bl = bls.get('speech_rate_wpm', {})
    print(f"✓ Voice baselines updated "
          f"(RMS avg={rms_bl.get('avg', 0):.4f}, "
          f"rate avg={rate_bl.get('avg', 0):.1f} wpm)")

    # Initialize 8B MetricReasoner (replaces hardcoded drowsy_score gate)
    reasoner = MetricReasoner()

    # Load learned driver patterns and voice baselines into reasoner
    driver_patterns = memory_manager.get_driver_patterns_for_reasoner()
    if driver_patterns:
        reasoner.set_driver_patterns(driver_patterns)
    baselines_for_reasoner = memory_manager.get_baselines()
    if baselines_for_reasoner:
        reasoner.set_voice_baselines(baselines_for_reasoner)

    cap = initialize_camera()
    face_mesh = initialize_mediapipe()
    state = initialize_state_variables()

    # Calibrate EAR threshold for this driver's eye geometry
    ear_thresh = run_ear_calibration(cap, face_mesh)

    print("\n✓ System ready — Starting monitoring (8B reasoning gate active)...")
    tts.speak("Drowsiness monitoring system activated.")

    try:
        while True:
            should_trigger_llm, final_metrics = run_detection_loop(
                cap, face_mesh, state, reasoner, ear_thresh=ear_thresh
            )

            if not should_trigger_llm:
                break

            # Dynamic opening based on how many times driver has been drowsy today
            freq = memory_manager.get_drowsy_frequency()
            opening_line, is_serious = build_opening_line(freq)
            tts.speak(opening_line)
            tts.wait_until_done()

            # Release main camera before detection thread opens its own
            cap.release()
            cap = None
            face_mesh.close()
            face_mesh = None
            cv2.destroyAllWindows()

            # Run conversation with raw metric injection + 8B reasoning + SQLite session
            run_llm_conversation(
                tts, stt, llm_assistant, final_metrics, state,
                voice_extractor, memory_manager, reasoner=reasoner,
                ear_thresh=ear_thresh,
                drowsy_freq=freq,
            )

            # Reopen camera for monitoring (retry — device may take time to release)
            cv2.destroyAllWindows()
            time.sleep(1.0)
            reopen_ok = False
            for attempt in range(5):
                try:
                    cap = initialize_camera()
                    if cap.isOpened():
                        face_mesh = initialize_mediapipe()
                        state['llm_triggered'] = False
                        reasoner.reset()  # Reset 8B confirmation counter
                        reopen_ok = True
                        print("\n✓ Monitoring resumed\n")
                        break
                    else:
                        cap.release()
                except Exception:
                    pass
                print(f"⚠️ Camera not ready, retrying ({attempt + 1}/5)...")
                time.sleep(1.0)

            if not reopen_ok:
                print("⚠️ Failed to restart camera after 5 attempts")
                break

    except KeyboardInterrupt:
        print("\n\n⚠️ System interrupted by user")

    finally:
        print("\n🔄 Cleaning up resources...")
        # Final extraction if there's un-processed transcript
        if memory_manager.conversation_transcript:
            print("💾 Running final fact extraction before exit...")
            try:
                sid = llm_assistant._session_id or memory_manager.start_session()
                memory_manager.extract_and_store_facts(sid)
            except Exception as e:
                print(f"⚠️ Final extraction failed: {e}")
        if cap is not None:
            cap.release()
        if face_mesh is not None:
            face_mesh.close()
        cv2.destroyAllWindows()
        stt.cleanup()
        tts.shutdown()
        memory_manager.close()
        print("✓ Cleanup complete")


def dump_database():
    """Utility: print all SQLite data."""
    memory_manager = MemoryManager()
    memory_manager.dump_all()
    memory_manager.close()


# Add command line argument handling
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--calibrate":
            memory = MemoryManager()
            tts = TTSEngine()
            stt = STTEngine()
            ve = VoiceFeatureExtractor()
            tts.speak("Starting voice calibration.")
            tts.wait_until_done()
            run_calibration(stt, ve, memory)
            stt.cleanup()
            tts.shutdown()
            memory.close()
        elif sys.argv[1] == "--dump-db":
            dump_database()
        elif sys.argv[1] == "--reset-db":
            memory = MemoryManager()
            print(f"⚠️  This will DELETE all data in {memory.db_path}")
            confirm = input("Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                memory.reset_database()
            else:
                print("Cancelled.")
            memory.close()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python main.py [--calibrate | --dump-db | --reset-db]")
    else:
        main()
