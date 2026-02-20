#!/usr/bin/env python3
"""
Driver Drowsiness Detection System V7
Main entry point — multi-threaded architecture.

V7 architecture:
  - LLM as reasoning layer: raw metrics + baselines injected every turn
  - SQLite memory: facts, sessions, baselines (replaces flat JSON profile)
  - Calibration check at startup — records voice baselines if none exist
  - Post-session: 8B model extracts facts into SQLite, baselines updated
  - Detection + voice metrics passed as raw numbers (no hardcoded thresholds)
  - Deepgram STT/TTS, live detection thread, voice feature extraction

Usage:
    python main.py                  # Run the system
    python main.py --calibrate      # Force voice baseline calibration
    python main.py --dump-db        # Print all SQLite data
"""

import cv2
import mediapipe as mp
import time
from collections import deque

from config import Config
from memory import MemoryManager
from tts_engine import TTSEngine
from stt_engine import STTEngine
from llm_assistant import LLMAssistant
from voice_features import VoiceFeatureExtractor
from detection import (
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
from dashboard import MetricsDashboard


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
        'drowsy_count': 0,
        'window_start_time': time.time()
    }


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
def run_detection_loop(cap, face_mesh, state):
    """Run the main drowsiness detection loop."""
    frame_skip = 2
    frame_count = 0
    final_metrics = None
    PROC_W, PROC_H = 320, 240

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

            ear, microsleep = process_eye_metrics(landmarks, state, now)
            mar = process_mouth_metrics(landmarks, state, now)
            head_down = process_head_pitch(landmarks, state, now)
            head_roll = process_head_roll(landmarks, state, now)

        cleanup_windows(state, now)
        metrics = calculate_metrics(state, microsleep, head_down, head_roll)

        drowsy_state = "ALERT"
        if metrics['drowsy_score'] > Config.DROWSY_THRESHOLD:
            drowsy_state = "DROWSY"
            state['drowsy_count'] += 1
            final_metrics = metrics.copy()
            final_metrics['microsleep'] = microsleep
            final_metrics['head_down'] = head_down

        draw_overlay(frame, metrics, ear, mar, microsleep, head_down,
                     head_roll, state, drowsy_state, landmarks=landmarks,
                     proc_size=(PROC_W, PROC_H))

        cv2.imshow("Driver Drowsiness Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("📊 Monitoring ended by user")
            return False, None

        if (state['drowsy_count'] >= Config.DROWSY_TRIGGER_COUNT and
                drowsy_state == "DROWSY" and
                not state['llm_triggered']):
            state['llm_triggered'] = True
            return True, final_metrics

    return False, None


# =========================
# LLM CONVERSATION (V7 — raw metrics + baselines + SQLite)
# =========================
def run_llm_conversation(tts, stt, llm_assistant, metrics, state,
                         voice_extractor, memory_manager):
    """Run conversation loop with raw metric injection and SQLite session tracking.

    V7 flow:
      1. Start SQLite session
      2. DetectionThread provides live raw metrics each turn
      3. VoiceFeatureExtractor provides raw voice metrics + baseline comparison
      4. Both injected to LLM as raw numbers — LLM reasons about severity
      5. Post-session: 8B model extracts facts → SQLite, baselines updated
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

    # Detection thread (own camera + FaceMesh) — with live camera overlay
    detection_thread = DetectionThread(show_display=True)
    detection_thread.start()

    # Get baselines for voice comparison
    baselines = memory_manager.get_baselines()
    baselines_str = memory_manager.format_baselines_for_llm()

    # Live metrics dashboard
    dashboard = MetricsDashboard(detection_thread, baselines=baselines)
    dashboard.start()

    # Format initial detection context (raw numbers)
    detection_context = format_detection_for_llm(
        metrics,
        microsleep=metrics.get('microsleep', False),
        head_down=metrics.get('head_down', False),
    )

    # Start conversation with raw metrics + baselines + session history
    session_count = memory_manager.get_session_count() - 1  # -1 because we just started this one
    llm_assistant.start_conversation(detection_context, baselines_str, max(0, session_count))

    # Get opening message
    opening = llm_assistant.get_response_streaming()
    tts.wait_until_done()
    voice_extractor.mark_prompt_end()

    # Conversation loop
    max_turns = Config.MAX_CONVERSATION_TURNS

    def _handle_user_turn(text, audio):
        """Extract raw features, get live detection, inject both to LLM.
        Returns the LLM response text."""
        # Voice features (raw)
        voice_context = None
        if audio is not None:
            features = voice_extractor.extract_features(audio, text)
            if features:
                voice_context = voice_extractor.format_for_llm(features, baselines)
                voice_accumulator.append(features)
                dashboard.update_voice(features)

        # Detection (raw)
        det_state = detection_thread.get_full_state()
        det_context = format_detection_for_llm(
            det_state,
            microsleep=det_state.get('microsleep', False),
            head_down=det_state.get('head_down', False),
        )
        score_accumulator.append(det_state.get('drowsy_score', 0))

        # LLM gets raw numbers for both
        response = llm_assistant.get_response_streaming(
            user_message=text,
            detection_context=det_context,
            voice_context=voice_context,
        )
        tts.wait_until_done()
        voice_extractor.mark_prompt_end()
        return response

    # Only explicit exit phrases — NOT common words like "fine" or "alert"
    hard_exit = {'exit', 'quit', 'bye', 'goodbye', 'bye bye', 'stop talking'}

    for turn in range(max_turns):
        user_input, audio_data = stt.listen(timeout=20, show_diagnostics=False)

        if user_input:
            if user_input.lower().strip() in hard_exit:
                print("🔚 User requested exit")
                tts.speak("Alright, I'll be here if you need me. Stay safe!")
                tts.wait_until_done()
                break

            response = _handle_user_turn(user_input, audio_data)

            # LLM decides when driver has recovered (via [RECOVERED] tag)
            if response and "[RECOVERED]" in response:
                print("🟢 LLM determined driver has recovered")
                break
        else:
            print("⚠️  No response detected")
            tts.speak("Are you still there? Give me a quick response if you can hear me.")
            tts.wait_until_done()
            voice_extractor.mark_prompt_end()

            retry_input, retry_audio = stt.listen(timeout=15, show_diagnostics=False)
            if not retry_input:
                print("⚠️  Still no response — ending conversation")
                tts.speak("I'll keep monitoring. Stay safe!")
                tts.wait_until_done()
                break
            response = _handle_user_turn(retry_input, retry_audio)
            if response and "[RECOVERED]" in response:
                print("🟢 LLM determined driver has recovered")
                break

    # Stop background detection + dashboard
    detection_thread.stop()
    dashboard.stop()

    # ── Post-session processing ──
    session_duration = time.perf_counter() - session_start
    avg_score = sum(score_accumulator) / len(score_accumulator) if score_accumulator else 0
    max_score = max(score_accumulator) if score_accumulator else 0

    # End session in SQLite
    memory_manager.end_session(
        session_id,
        avg_drowsy_score=round(avg_score, 3),
        max_drowsy_score=round(max_score, 3),
        turn_count=llm_assistant.conversation_turns,
        duration_s=round(session_duration, 1),
    )

    # LLM-based fact extraction (8B model — cheap)
    print("\n💾 Running LLM-based fact extraction...")
    memory_manager.extract_and_store_facts(session_id)

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

    print("\n" + "=" * 60)
    print("✓ Session complete — Facts extracted — Baselines updated — Resuming monitoring")
    print("=" * 60)


# =========================
# MAIN FUNCTION
# =========================
def main():
    """Main function — V7 drowsiness detection with LLM reasoning + SQLite memory."""
    print("\n" + "=" * 60)
    print("🚗 Driver Drowsiness Detection System V7")
    print("   LLM as reasoning layer | SQLite memory")
    print("   Raw metrics + baselines | Dynamic fact extraction")
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

    # Voice calibration check — run if no baselines exist
    if memory_manager.needs_calibration():
        print("⚠️ No voice baselines found — running calibration...")
        tts.speak("Welcome! I need to calibrate your voice first. "
                  "Please read a few sentences in your normal speaking voice.")
        tts.wait_until_done()
        run_calibration(stt, voice_extractor, memory_manager)
    else:
        bls = memory_manager.get_baselines()
        rms_bl = bls.get('energy_rms', {})
        rate_bl = bls.get('speech_rate_wpm', {})
        print(f"✓ Voice baselines loaded "
              f"(RMS avg={rms_bl.get('avg', 0):.4f}, "
              f"rate avg={rate_bl.get('avg', 0):.1f} wpm)")

    cap = initialize_camera()
    face_mesh = initialize_mediapipe()
    state = initialize_state_variables()

    print("\n✓ System ready — Starting monitoring...")
    tts.speak("Drowsiness monitoring system activated.")

    try:
        while True:
            should_trigger_llm, final_metrics = run_detection_loop(
                cap, face_mesh, state
            )

            if not should_trigger_llm:
                break

            # Release main camera before detection thread opens its own
            cap.release()
            face_mesh.close()
            cv2.destroyAllWindows()

            # Run conversation with raw metric injection + SQLite session
            run_llm_conversation(
                tts, stt, llm_assistant, final_metrics, state,
                voice_extractor, memory_manager,
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
                        state['drowsy_count'] = 0
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
        cap.release()
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
        elif sys.argv[1] == "--dump-db":
            dump_database()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python main.py [--calibrate | --dump-db]")
    else:
        main()
