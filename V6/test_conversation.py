#!/usr/bin/env python3
"""
Conversation Pipeline End-to-End Test (V5)
=================================================
Tests the full conversation loop: STT → Groq LLM → Edge-TTS
WITHOUT needing a camera or drowsiness detection.

Measures per-turn: STT latency, Groq first-token & full latency,
Edge-TTS generation & playback time, total turnaround, memory accuracy.

Usage:
    python test_conversation.py                      # Single trial, new driver
    python test_conversation.py --returning           # Single trial, returning driver
    python test_conversation.py --batch               # Full 20-trial batch
    python test_conversation.py --summary             # Print summary of all logs

After each LLM response you'll be asked to rate quality (1-4):
    1 = Poor (off-topic, ignores drowsiness, robotic)
    2 = Fair (addresses drowsiness but generic)
    3 = Good (personalized, uses an alertness technique)
    4 = Excellent (personalized, creative, perfect prompt adherence)

Press Enter to skip rating. Say "exit" or "quit" to end a conversation.
"""

import sys
import os
import json
import time
import shutil
from pathlib import Path

# Ensure V5 modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from memory import MemoryManager
from tts_engine import TTSEngine
from stt_engine import STTEngine
from llm_assistant import LLMAssistant
from metrics_logger import MetricsLogger
from voice_features import VoiceFeatureExtractor


# ── Fake drowsiness metrics (simulated trigger) ────────────────

FAKE_METRICS = {
    "drowsy_score": 0.62,
    "perclos": 0.35,
    "blink_rate": 18,
    "ear_std": 0.04,
    "slow_blinks": 4,
    "microsleep_flag": False,
    "head_down_flag": True,
    "head_roll_flag": False,
    "pitch_var": 0.08,
}

FAKE_STATE = {
    "yawn_times": [time.time() - 30, time.time() - 10],
    "eye_closed_start": None,
}


# ── Populated profile for "returning driver" trials ────────────

RETURNING_PROFILE = {
    "personal": {
        "name": "Alex",
        "occupation": "software engineer",
        "family": ["Mom", "brother Jake"],
        "location": "Austin"
    },
    "preferences": {
        "engagement_style": "conversation"
    },
    "interests": {
        "hobbies": ["hiking", "cooking", "video games"],
        "sports_teams": ["Dallas Cowboys"],
        "music_preferences": ["indie rock", "lo-fi beats"],
        "topics_engaged_with": ["weekend plans", "cooking recipes"]
    },
    "driving_patterns": {
        "common_routes": ["I-35 Austin to Dallas"],
        "typical_times": [],
        "usual_destinations": ["Dallas", "work"]
    },
    "conversation_history": {
        "last_topics": ["hiking trip", "new recipe"],
        "successful_engagement_types": ["personal questions", "mental challenges"],
        "things_to_avoid": []
    },
    "system_metadata": {
        "total_conversations": 5,
        "last_conversation": "2026-02-15 22:30:00",
        "profile_created": "2026-01-20 10:00:00",
        "total_drowsy_episodes": 5
    }
}


# ── Helper: facts the user plans to share (for accuracy tracking) ──

NEW_DRIVER_FACTS = [
    "name",
    "occupation or what they do",
    "a hobby or interest",
    "engagement preference (conversation vs actions)",
]

RETURNING_DRIVER_FACTS = [
    "a new hobby or interest",
    "a destination they're heading to",
]


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

    # ── Setup memory ──
    memory = MemoryManager()

    if driver_type == "returning":
        # Inject the populated profile
        memory.profile = json.loads(json.dumps(RETURNING_PROFILE))
        memory.save_profile()
        print("📋 Loaded returning driver profile (Alex)")
    else:
        # Reset to blank profile
        memory.profile = memory.create_default_profile()
        memory.save_profile()
        print("📋 Fresh profile (new driver)")

    # ── Initialize engines ──
    tts = TTSEngine()
    stt = STTEngine()
    voice_extractor = VoiceFeatureExtractor()
    llm = LLMAssistant(tts, memory)

    # ── Create and attach metrics logger ──
    logger = MetricsLogger(trial_id=trial_id, driver_type=driver_type)
    stt.metrics_logger = logger
    tts.metrics_logger = logger
    llm.metrics_logger = logger

    # ── Remind user what facts to share ──
    facts_to_share = NEW_DRIVER_FACTS if driver_type == "new" else RETURNING_DRIVER_FACTS
    print(f"\n📝 Try to share these facts during conversation:")
    for i, fact in enumerate(facts_to_share, 1):
        print(f"   {i}. {fact}")
    print()

    # ── Start conversation ──
    logger.start_conversation()
    llm.start_conversation(FAKE_METRICS, FAKE_STATE)

    # Opening turn (LLM speaks first, no STT)
    print("─" * 40)
    logger.start_turn(0)
    opening = llm.get_response_streaming()
    tts.wait_until_done()
    voice_extractor.mark_prompt_end()
    
    # Ask for quality rating
    _ask_quality_rating(logger)
    logger.end_turn()

    # ── Conversation loop ──
    for turn in range(1, max_turns + 1):
        print(f"\n─── Turn {turn} ───")
        logger.start_turn(turn)

        # Listen (V5: returns (text, audio) tuple)
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

        # Extract voice features (V5)
        voice_context = None
        if audio_data is not None:
            features = voice_extractor.extract_features(audio_data, user_input)
            if features:
                voice_context = voice_extractor.format_for_llm(features)

        # Simulated live drowsy score (no camera in test mode)
        # Starts at the fake trigger score and drifts down slightly each turn
        # to simulate the driver getting more alert during conversation
        simulated_score = max(0.30, FAKE_METRICS["drowsy_score"] - (turn * 0.04))

        # LLM response (V5: with voice context + simulated live score)
        response = llm.get_response_streaming(
            user_message=user_input,
            live_score=simulated_score,
            voice_context=voice_context,
        )
        tts.wait_until_done()
        voice_extractor.mark_prompt_end()

        # Quality rating
        _ask_quality_rating(logger)
        logger.end_turn()

        time.sleep(0.3)

    # ── End conversation ──
    logger.end_conversation()

    # Apply memory learnings (V5: LLM-based extraction)
    memory.extract_and_apply_learnings()
    memory.log_conversation_metadata(FAKE_METRICS, llm.conversation_turns)

    # Ask which facts the user actually shared
    _ask_facts_shared(logger, facts_to_share)

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

    # ── Aggregate summary ──
    print_aggregate_summary(all_loggers)


def print_aggregate_summary(loggers=None):
    """Print aggregate stats across all trial logs."""
    log_dir = Path(__file__).resolve().parent / "test_logs"

    if loggers is None:
        # Load from disk
        if not log_dir.exists():
            print("No test_logs/ directory found.")
            return
        loggers_data = []
        for f in sorted(log_dir.glob("*.json")):
            with open(f) as fh:
                loggers_data.append(json.load(fh))
    else:
        loggers_data = [lg.get_summary() for lg in loggers]
        # Wrap in expected format
        loggers_data = [{"summary": s} if "summary" not in s else s for s in loggers_data]

    if not loggers_data:
        print("No logs found.")
        return

    # Collect all summaries
    summaries = []
    for d in loggers_data:
        s = d.get("summary", d)
        summaries.append(s)

    new_summaries = [s for s in summaries if s.get("driver_type") == "new"]
    ret_summaries = [s for s in summaries if s.get("driver_type") == "returning"]

    def _agg(slist, key):
        """Aggregate a stat across summaries."""
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
            ("Total Turnaround", "total_turnaround_ms"),
            ("LLM Quality (1-4)", "llm_quality_rating"),
        ]:
            avg = _agg(slist, key)
            if avg is not None:
                unit = "" if "quality" in key.lower() else " ms"
                print(f"     {metric_label:>20s}:  avg across trials = {avg}{unit}")

        # Memory accuracy
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
