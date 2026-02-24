#!/usr/bin/env python3
"""
Conversation Pipeline End-to-End Test (V7)
=================================================
Tests the full conversation loop: STT → Groq LLM → Deepgram TTS
WITHOUT needing a camera or drowsiness detection.

V7 changes:
  - Uses SQLite memory (facts/sessions/baselines) instead of JSON profile
  - Raw metric injection (no severity labels) — matches V7 architecture
  - Returning driver seeds SQLite with facts instead of flat JSON
  - Post-session: LLM extracts facts → SQLite with free-form types
  - Simulated detection metrics formatted via format_detection_for_llm()

Measures per-turn: STT latency, Groq first-token & full latency,
TTS generation & playback time, total turnaround, memory accuracy.

Usage:
    python test_conversation.py                       # Single trial, new driver
    python test_conversation.py --returning            # Single trial, returning driver
    python test_conversation.py --batch                # Full 20-trial batch
    python test_conversation.py --summary              # Print summary of all logs

After each LLM response you'll be asked to rate quality (1-4):
    1 = Poor (off-topic, ignores drowsiness, robotic)
    2 = Fair (addresses drowsiness but generic)
    3 = Good (personalized, uses an alertness technique)
    4 = Excellent (personalized, creative, perfect prompt adherence)

Press Enter to skip rating. Say "exit" or "quit" to end a conversation.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from memory import MemoryManager
from tts_engine import TTSEngine
from stt_engine import STTEngine
from llm_assistant import LLMAssistant
from metrics_logger import MetricsLogger
from voice_features import VoiceFeatureExtractor
from detection import format_detection_for_llm


# ── Fake drowsiness metrics (simulated trigger) ────────────────

FAKE_METRICS = {
    "drowsy_score": 0.62,
    "perclos": 0.35,
    "blink_rate": 18,
    "ear_std": 0.04,
    "slow_blinks": 4,
    "pitch_var": 0.008,
}

FAKE_MICROSLEEP = False
FAKE_HEAD_DOWN = True


# ── Facts to seed SQLite for a "returning driver" ──────────────

RETURNING_DRIVER_FACTS = [
    ("name", "Alex"),
    ("occupation", "software engineer"),
    ("family_member", "Mom"),
    ("family_member", "brother Jake"),
    ("location", "Austin"),
    ("engagement_preference", "conversation"),
    ("hobby", "hiking"),
    ("hobby", "cooking"),
    ("hobby", "video games"),
    ("sports_team", "Dallas Cowboys"),
    ("music_preference", "indie rock"),
    ("music_preference", "lo-fi beats"),
    ("topic_discussed", "weekend plans"),
    ("topic_discussed", "cooking recipes"),
    ("driving_route", "I-35 Austin to Dallas"),
    ("destination", "Dallas"),
    ("destination", "work"),
    ("topic_discussed", "hiking trip"),
    ("topic_discussed", "new recipe"),
    ("engagement_that_worked", "personal questions"),
    ("engagement_that_worked", "mental challenges"),
]


# ── Facts the tester should try to share (for accuracy tracking) ──

NEW_DRIVER_EXPECTED_FACTS = [
    "name",
    "occupation or what they do",
    "a hobby or interest",
    "engagement preference (conversation vs actions)",
]

RETURNING_DRIVER_EXPECTED_FACTS = [
    "a new hobby or interest",
    "a destination they're heading to",
]


def _seed_returning_profile(memory, session_id):
    """Populate SQLite with facts for a returning driver (Alex)."""
    for fact_type, value in RETURNING_DRIVER_FACTS:
        memory._store_fact(fact_type, value, session_id)
    # Bump confidence on a few facts (simulate multiple confirmations)
    for _ in range(3):
        memory._store_fact("name", "Alex", session_id)
        memory._store_fact("occupation", "software engineer", session_id)


def run_single_trial(trial_id, driver_type="new", max_turns=8):
    """Run one conversation trial with full metrics logging.

    Args:
        trial_id: Unique identifier for this trial
        driver_type: "new" or "returning"
        max_turns: Maximum conversation turns

    Returns:
        MetricsLogger with all collected data
    """
    print("\n" + "=" * 70)
    print(f"🧪 TRIAL: {trial_id}  |  Driver type: {driver_type}  |  Max turns: {max_turns}")
    print("=" * 70)

    # ── Setup SQLite memory ──
    # Use a test-specific DB so we don't pollute the real one
    test_db = Path(__file__).resolve().parent / "test_logs" / f"test_{trial_id}.db"
    test_db.parent.mkdir(exist_ok=True)
    memory = MemoryManager(db_path=test_db)

    # Create a session for this test
    session_id = memory.start_session()

    if driver_type == "returning":
        _seed_returning_profile(memory, session_id)
        # Also create some fake prior sessions
        for _ in range(5):
            sid = memory.start_session()
            memory.end_session(sid, avg_drowsy_score=0.55, max_drowsy_score=0.72,
                               turn_count=6, duration_s=120.0)
        print("📋 Loaded returning driver profile (Alex) into SQLite")
    else:
        print("📋 Fresh SQLite database (new driver)")

    # Seed some fake baselines (so voice comparison works in test mode)
    memory.store_calibration_baselines([
        {'energy_rms': 0.045, 'speech_rate_wpm': 135.0, 'pause_ratio': 0.20,
         'response_latency_s': 2.0, 'peak_amplitude': 0.35, 'duration_s': 4.0},
        {'energy_rms': 0.050, 'speech_rate_wpm': 140.0, 'pause_ratio': 0.18,
         'response_latency_s': 1.8, 'peak_amplitude': 0.40, 'duration_s': 3.5},
        {'energy_rms': 0.042, 'speech_rate_wpm': 130.0, 'pause_ratio': 0.22,
         'response_latency_s': 2.2, 'peak_amplitude': 0.38, 'duration_s': 4.2},
    ])

    # ── Initialize engines ──
    tts = TTSEngine()
    stt = STTEngine()
    voice_extractor = VoiceFeatureExtractor()
    llm = LLMAssistant(tts, memory)
    llm._session_id = session_id

    # ── Create and attach metrics logger ──
    logger = MetricsLogger(trial_id=trial_id, driver_type=driver_type)
    stt.metrics_logger = logger
    tts.metrics_logger = logger
    llm.metrics_logger = logger

    # ── Remind user what facts to share ──
    facts_to_share = NEW_DRIVER_EXPECTED_FACTS if driver_type == "new" else RETURNING_DRIVER_EXPECTED_FACTS
    print(f"\n📝 Try to share these facts during conversation:")
    for i, fact in enumerate(facts_to_share, 1):
        print(f"   {i}. {fact}")
    print()

    # ── Build initial detection context (raw metrics) ──
    detection_context = format_detection_for_llm(
        FAKE_METRICS, microsleep=FAKE_MICROSLEEP, head_down=FAKE_HEAD_DOWN
    )
    baselines_str = memory.format_baselines_for_llm()
    baselines = memory.get_baselines()

    # ── Start conversation ──
    logger.start_conversation()
    llm.start_conversation(detection_context, baselines_str)

    # Opening turn (LLM speaks first)
    print("─" * 40)
    logger.start_turn(0)
    opening = llm.get_response_streaming()
    tts.wait_until_done()
    voice_extractor.mark_prompt_end()

    _ask_quality_rating(logger)
    logger.end_turn()

    # ── Conversation loop ──
    for turn in range(1, max_turns + 1):
        print(f"\n─── Turn {turn} ───")
        logger.start_turn(turn)

        user_input, audio_data = stt.listen(timeout=20, show_diagnostics=False)

        if not user_input:
            print("⚠️  No speech detected")
            logger.log_stt(0, None)

            tts.speak("Are you still there? Give me a quick response.")
            tts.wait_until_done()
            voice_extractor.mark_prompt_end()

            user_input, audio_data = stt.listen(timeout=15, show_diagnostics=False)
            if not user_input:
                print("⚠️  Still nothing — ending conversation")
                logger.end_turn()
                break

        # Check exit keywords
        if user_input and any(w in user_input.lower() for w in
                              ['exit', 'quit', 'bye', 'stop', 'done', 'goodbye']):
            print("🔚 User ended conversation")
            logger.end_turn()
            break

        # Extract voice features (raw — V7)
        voice_context = None
        if audio_data is not None:
            features = voice_extractor.extract_features(audio_data, user_input)
            if features:
                voice_context = voice_extractor.format_for_llm(features, baselines)

        # Simulated detection metrics (score drifts down to simulate recovery)
        simulated_metrics = FAKE_METRICS.copy()
        simulated_metrics["drowsy_score"] = max(0.30, 0.62 - (turn * 0.04))
        simulated_metrics["perclos"] = max(0.08, 0.35 - (turn * 0.03))

        sim_detection_context = format_detection_for_llm(
            simulated_metrics,
            microsleep=FAKE_MICROSLEEP,
            head_down=FAKE_HEAD_DOWN if turn < 3 else False,
        )

        # LLM response (V7: raw detection + voice context)
        response = llm.get_response_streaming(
            user_message=user_input,
            detection_context=sim_detection_context,
            voice_context=voice_context,
        )
        tts.wait_until_done()
        voice_extractor.mark_prompt_end()

        _ask_quality_rating(logger)
        logger.end_turn()

        time.sleep(0.3)

    # ── End conversation ──
    logger.end_conversation()

    # Post-session: LLM fact extraction → SQLite
    print("\n💾 Running post-session fact extraction...")
    extracted = memory.extract_and_store_facts(session_id)

    # End session
    memory.end_session(
        session_id,
        avg_drowsy_score=0.50,
        max_drowsy_score=0.62,
        turn_count=llm.conversation_turns,
        duration_s=(time.perf_counter() - logger.conversation_start) if logger.conversation_start else 0,
    )

    # Ask which facts the user actually shared
    _ask_facts_shared(logger, facts_to_share)

    # Print what was extracted into SQLite
    print("\n📊 Facts in SQLite after extraction:")
    facts = memory.get_context_facts(limit=30)
    for ft, val, conf, tc in facts:
        print(f"   {ft}: {val} (confidence={conf:.1f}, confirmed={tc}x)")

    # Save and print results
    logger.print_summary()
    logger.save()

    # Cleanup
    tts.shutdown()

    return logger


def _ask_quality_rating(logger):
    """Prompt user for LLM quality rating (1-4)."""
    try:
        rating_input = input("   ⭐ Rate LLM quality (1-4, Enter=skip): ").strip()
        if rating_input and rating_input.isdigit():
            rating = int(rating_input)
            if 1 <= rating <= 4:
                logger.log_quality_rating(rating)
                return
    except (EOFError, KeyboardInterrupt):
        pass


def _ask_facts_shared(logger, expected_facts):
    """After conversation, ask which facts the user actually shared."""
    print(f"\n📝 Which of these facts did you share? (comma-separated numbers, or 'all')")
    for i, fact in enumerate(expected_facts, 1):
        print(f"   {i}. {fact}")

    try:
        answer = input("   → ").strip().lower()
        if answer == "all":
            for fact in expected_facts:
                logger.log_fact_shared(fact)
        elif answer:
            for n in answer.replace(",", " ").split():
                if n.isdigit() and 1 <= int(n) <= len(expected_facts):
                    logger.log_fact_shared(expected_facts[int(n) - 1])
    except (EOFError, KeyboardInterrupt):
        pass


def run_batch():
    """Run full 20-trial batch: 10 new + 10 returning."""
    print("\n" + "=" * 70)
    print("🧪 BATCH TEST: 20 conversations (10 new + 10 returning)")
    print("=" * 70)

    all_loggers = []

    for i in range(1, 11):
        print(f"\n\n{'#' * 70}")
        print(f"  NEW DRIVER TRIAL {i}/10")
        print(f"{'#' * 70}")
        lg = run_single_trial(f"new_{i:02d}", driver_type="new", max_turns=8)
        all_loggers.append(lg)

    for i in range(1, 11):
        print(f"\n\n{'#' * 70}")
        print(f"  RETURNING DRIVER TRIAL {i}/10")
        print(f"{'#' * 70}")
        lg = run_single_trial(f"returning_{i:02d}", driver_type="returning", max_turns=8)
        all_loggers.append(lg)

    print_aggregate_summary(all_loggers)


def print_aggregate_summary(loggers=None):
    """Print aggregate stats across all trial logs."""
    log_dir = Path(__file__).resolve().parent / "test_logs"

    if loggers is None:
        if not log_dir.exists():
            print("No test_logs/ directory found.")
            return
        loggers_data = []
        for f in sorted(log_dir.glob("*.json")):
            with open(f) as fh:
                loggers_data.append(json.load(fh))
    else:
        loggers_data = [lg.get_summary() for lg in loggers]
        loggers_data = [{"summary": s} if "summary" not in s else s for s in loggers_data]

    if not loggers_data:
        print("No logs found.")
        return

    summaries = []
    for d in loggers_data:
        s = d.get("summary", d)
        summaries.append(s)

    new_summaries = [s for s in summaries if s.get("driver_type") == "new"]
    ret_summaries = [s for s in summaries if s.get("driver_type") == "returning"]

    def _agg(slist, key):
        vals = []
        for s in slist:
            stat = s.get(key, {})
            if isinstance(stat, dict) and stat.get("avg") is not None:
                vals.append(stat["avg"])
        if not vals:
            return None
        return round(sum(vals) / len(vals), 1)

    print("\n" + "=" * 70)
    print("📊 AGGREGATE RESULTS")
    print("=" * 70)

    for label, slist in [("ALL", summaries), ("NEW DRIVER", new_summaries), ("RETURNING", ret_summaries)]:
        if not slist:
            continue
        print(f"\n  ── {label} ({len(slist)} trials) ──")
        for metric_label, key in [
            ("STT Latency", "stt_latency_ms"),
            ("Groq First Token", "groq_first_token_ms"),
            ("Groq Full Response", "groq_full_response_ms"),
            ("TTS Generation", "tts_generation_ms"),
            ("TTS Total", "tts_total_ms"),
            ("User→Speech Gap", "user_to_speech_ms"),
            ("Total Turnaround", "total_turnaround_ms"),
            ("LLM Quality (1-4)", "llm_quality_rating"),
        ]:
            avg = _agg(slist, key)
            if avg is not None:
                unit = "" if "quality" in key.lower() else " ms"
                print(f"     {metric_label:>20s}:  avg across trials = {avg}{unit}")

        total_shared = sum(s.get("memory_facts_shared", 0) for s in slist)
        total_extracted = sum(s.get("memory_facts_extracted", 0) for s in slist)
        acc = round(total_extracted / total_shared * 100, 1) if total_shared > 0 else None
        print(f"     {'Memory Accuracy':>20s}:  {total_extracted}/{total_shared} facts", end="")
        if acc is not None:
            print(f" ({acc}%)")
        else:
            print()

    print("=" * 70)


# ── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--summary" in args:
        print_aggregate_summary()

    elif "--batch" in args:
        run_batch()

    elif "--returning" in args:
        trial_id = f"returning_{int(time.time()) % 10000}"
        run_single_trial(trial_id, driver_type="returning")

    else:
        trial_id = f"new_{int(time.time()) % 10000}"
        run_single_trial(trial_id, driver_type="new")
