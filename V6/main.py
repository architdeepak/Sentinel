#!/usr/bin/env python3
"""
Driver Drowsiness Detection System V6
Main entry point — multi-threaded architecture.

V6 architecture:
  - Deepgram STT (replaces Google Speech Recognition)
  - Deepgram Aura TTS (replaces Edge-TTS)
  - Detection thread runs continuously (even during conversation)
  - Voice features extracted from each STT response
  - Live drowsy score + voice analysis injected into every LLM turn
  - LLM-based post-session extraction (replaces regex)
  - Conversation history capped at MAX_HISTORY_TURNS

Usage:
    python main.py                  # Run the system
    python main.py --view-profile   # View saved driver profile
"""

import cv2
import mediapipe as mp
import json
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
)


# =========================
# INITIALIZATION
# =========================
def initialize_mediapipe():
    """Initialize MediaPipe face mesh (optimized for RPi)."""
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return face_mesh


def initialize_camera():
    """Initialize camera with RPi-optimized settings."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def initialize_state_variables():
    """Initialize all state tracking variables."""
    # maxlen caps memory: at 20fps/2skip = 10fps for 10s window = 100 entries max
    _win = 200  # generous cap — cleanup_windows prunes by time, this is a safety net
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
# MAIN DETECTION LOOP
# =========================
def run_detection_loop(cap, face_mesh, state):
    """Run the main drowsiness detection loop."""
    frame_skip = 2
    frame_count = 0
    final_metrics = None  # Store metrics when drowsiness triggers
    # Landmarks normalized to the frame MediaPipe actually processes (320x240)
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
            # Store the metrics for when we trigger conversation
            final_metrics = metrics.copy()

        draw_overlay(frame, metrics, ear, mar, microsleep, head_down,
                     head_roll, state, drowsy_state)

        cv2.imshow("Driver Drowsiness Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("📊 Monitoring ended by user")
            return False, None

        if (state['drowsy_count'] >= Config.DROWSY_TRIGGER_COUNT and
                drowsy_state == "DROWSY" and
                not state['llm_triggered']):
            state['llm_triggered'] = True
            return True, final_metrics  # Return the metrics!

    return False, None


# =========================
# LLM CONVERSATION (V5 — with live detection + voice features)
# =========================
def run_llm_conversation(tts, stt, llm_assistant, metrics, state, voice_extractor):
    """Run conversation loop with parallel detection, voice analysis, and memory.

    V5 architecture:
      - DetectionThread creates its OWN camera + FaceMesh (MediaPipe isn't thread-safe)
      - Main thread's camera must be released BEFORE calling this
      - VoiceFeatureExtractor analyzes each STT response
      - Live drowsy score + voice context injected every LLM turn
      - Transcript logged for post-session LLM extraction
    """
    print("\n" + "=" * 60)
    print("💬 STARTING CONVERSATION (V6 — Deepgram STT/TTS + live detection)")
    print("=" * 60)

    # Detection thread creates its own camera + FaceMesh (fully isolated)
    detection_thread = DetectionThread()
    detection_thread.start()

    # Start conversation with initial metrics + memory
    llm_assistant.start_conversation(metrics, state)

    # Get opening message (no user input yet → no voice/score context)
    opening = llm_assistant.get_response_streaming()
    tts.wait_until_done()

    # Mark prompt end → voice extractor starts measuring response latency
    voice_extractor.mark_prompt_end()

    # Conversation loop
    max_turns = Config.MAX_CONVERSATION_TURNS
    def _handle_user_turn(text, audio):
        """Extract voice features, get live score, call LLM, wait for TTS."""
        voice_context = None
        if audio is not None:
            features = voice_extractor.extract_features(audio, text)
            if features:
                voice_context = voice_extractor.format_for_llm(features)
        live_score = detection_thread.get_latest_score()
        llm_assistant.get_response_streaming(
            user_message=text,
            live_score=live_score,
            voice_context=voice_context,
        )
        tts.wait_until_done()
        voice_extractor.mark_prompt_end()

    exit_words = {'exit', 'quit', 'bye', 'stop', 'goodbye', 'done',
                  'enough', 'fine', 'alert', "i'm good", "im good"}

    for turn in range(max_turns):
        # Listen for user response (returns text + raw audio)
        user_input, audio_data = stt.listen(timeout=20, show_diagnostics=False)

        if user_input:
            # Check for exit keywords
            if any(word in user_input.lower() for word in exit_words):
                print("🔚 User indicated they're alert")
                tts.speak("Great! You sound much better. I'll keep monitoring quietly.")
                tts.wait_until_done()
                break

            _handle_user_turn(user_input, audio_data)
            time.sleep(0.3)
        else:
            # No response detected
            print("⚠️  No response detected")
            tts.speak("Are you still there? Give me a quick response if you can hear me.")
            tts.wait_until_done()
            voice_extractor.mark_prompt_end()

            retry_input, retry_audio = stt.listen(timeout=15, show_diagnostics=False)
            if not retry_input:
                print("⚠️  Still no response - ending conversation")
                tts.speak("I'll keep monitoring. Stay safe!")
                tts.wait_until_done()
                break
            _handle_user_turn(retry_input, retry_audio)

    # Stop background detection
    detection_thread.stop()

    # Post-session: LLM-based extraction (replaces old regex)
    print("\n💾 Running LLM-based profile extraction...")
    llm_assistant.memory_manager.extract_and_apply_learnings()
    llm_assistant.memory_manager.log_conversation_metadata(metrics, llm_assistant.conversation_turns)

    print("\n" + "=" * 60)
    print("✓ Conversation ended — Memory updated — Resuming monitoring")
    print("=" * 60)


# =========================
# MAIN FUNCTION
# =========================
def main():
    """Main function — V6 multi-threaded drowsiness detection system."""
    print("\n" + "=" * 60)
    print("🚗 Driver Drowsiness Detection System V6")
    print("   Deepgram STT + TTS | Live detection + voice analysis")
    print("   LLM-based memory extraction")
    print("=" * 60 + "\n")

    # Initialize memory manager
    memory_manager = MemoryManager()
    print(f"📊 Driver Profile Status:")
    print(f"   Total conversations: {memory_manager.profile['system_metadata']['total_conversations']}")
    if memory_manager.profile['system_metadata']['last_conversation']:
        print(f"   Last conversation: {memory_manager.profile['system_metadata']['last_conversation']}")
    print()

    tts = TTSEngine()
    stt = STTEngine()
    voice_extractor = VoiceFeatureExtractor()
    llm_assistant = LLMAssistant(tts, memory_manager)

    cap = initialize_camera()
    face_mesh = initialize_mediapipe()
    state = initialize_state_variables()

    print("✓ System ready — Starting monitoring...")
    tts.speak("Drowsiness monitoring system activated.")

    try:
        while True:
            should_trigger_llm, final_metrics = run_detection_loop(cap, face_mesh, state)

            if not should_trigger_llm:
                break

            # Release main camera + FaceMesh BEFORE detection thread opens its own
            # (RPi has one camera device — can't open two handles simultaneously)
            cap.release()
            face_mesh.close()
            cv2.destroyAllWindows()

            # Run conversation — detection thread creates its own isolated instances
            run_llm_conversation(
                tts, stt, llm_assistant, final_metrics, state,
                voice_extractor,
            )

            # Reopen main camera + FaceMesh for monitoring
            try:
                cap = initialize_camera()
                face_mesh = initialize_mediapipe()
                state['llm_triggered'] = False
                state['drowsy_count'] = 0
                time.sleep(0.5)
                print("\n✓ Monitoring resumed\n")
            except Exception as e:
                print(f'⚠️ Failed to restart camera: {e}')
                break

    except KeyboardInterrupt:
        print("\n\n⚠️ System interrupted by user")

    finally:
        print("\n🔄 Cleaning up resources...")
        # Run LLM extraction if there's un-extracted transcript
        if llm_assistant.memory_manager.conversation_transcript:
            print("💾 Running final LLM extraction before exit...")
            try:
                llm_assistant.memory_manager.extract_and_apply_learnings()
            except Exception as e:
                print(f"⚠️ Final extraction failed: {e}")
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        stt.cleanup()
        tts.shutdown()
        print("✓ Cleanup complete")


def view_driver_profile():
    """Utility function to view the current driver profile."""
    memory_manager = MemoryManager()
    print("\n" + "=" * 60)
    print("📊 CURRENT DRIVER PROFILE")
    print("=" * 60)
    print(json.dumps(memory_manager.profile, indent=2))
    print("=" * 60)


# Add command line argument handling
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--view-profile":
        view_driver_profile()
    else:
        main()
