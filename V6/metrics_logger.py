"""
Metrics Logger for Conversation Pipeline End-to-End Testing (V5).
Logs per-turn and per-conversation timing, accuracy, and quality metrics.

Output: JSON log files in V5/test_logs/ directory.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime


class MetricsLogger:
    """Collects and persists per-turn and per-conversation metrics."""

    def __init__(self, trial_id=None, driver_type="new"):
        self.log_dir = Path(__file__).resolve().parent / "test_logs"
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trial_id = trial_id or f"{driver_type}_{timestamp}"
        self.driver_type = driver_type  # "new" or "returning"

        # Per-conversation metadata
        self.conversation_start = None
        self.conversation_end = None

        # Per-turn metrics (list of dicts)
        self.turns = []

        # Current turn accumulator
        self._current_turn = {}

        # Memory tracking
        self.facts_shared = []       # Facts the user explicitly shared
        self.facts_extracted = []    # Facts the system extracted

        # LLM quality ratings (filled in manually after each turn)
        self._quality_ratings = []

        print(f"📊 MetricsLogger initialized — trial: {self.trial_id}")

    # ── Conversation-level ──────────────────────────────────────

    def start_conversation(self):
        """Mark conversation start."""
        self.conversation_start = time.perf_counter()

    def end_conversation(self):
        """Mark conversation end."""
        self.conversation_end = time.perf_counter()

    # ── Turn-level ──────────────────────────────────────────────

    def start_turn(self, turn_number):
        """Begin a new turn measurement."""
        self._current_turn = {
            "turn": turn_number,
            "turn_start": time.perf_counter(),
            "stt_latency_ms": None,
            "stt_text": None,
            "stt_done_time": None,               # perf_counter when STT finished
            "groq_latency_ms": None,           # Time to first token
            "groq_full_latency_ms": None,       # Time to full response
            "groq_tokens": 0,
            "tts_generation_ms": None,
            "tts_playback_ms": None,
            "tts_total_ms": None,
            "user_to_speech_ms": None,           # STT done → TTS first audio byte
            "total_turnaround_ms": None,
            "llm_response": None,
            "llm_quality_rating": None,         # 1-4 manual rating
            "memory_facts_extracted_this_turn": 0,
        }

    def end_turn(self):
        """Finalize the current turn and append to turns list."""
        now = time.perf_counter()
        if self._current_turn.get("turn_start"):
            self._current_turn["total_turnaround_ms"] = round(
                (now - self._current_turn["turn_start"]) * 1000, 1
            )
        self.turns.append(self._current_turn)
        self._current_turn = {}

    # ── STT timing ──────────────────────────────────────────────

    def log_stt(self, latency_ms, text):
        """Log STT result and timing."""
        self._current_turn["stt_latency_ms"] = round(latency_ms, 1)
        self._current_turn["stt_text"] = text
        self._current_turn["stt_done_time"] = time.perf_counter()

    # ── Groq / LLM timing ──────────────────────────────────────

    def log_groq_first_token(self, latency_ms):
        """Log time from API call to first streamed token."""
        self._current_turn["groq_latency_ms"] = round(latency_ms, 1)

    def log_groq_complete(self, full_latency_ms, token_count, response_text):
        """Log full Groq response timing."""
        self._current_turn["groq_full_latency_ms"] = round(full_latency_ms, 1)
        self._current_turn["groq_tokens"] = token_count
        self._current_turn["llm_response"] = response_text

    # ── TTS timing ──────────────────────────────────────────────

    def log_tts_generation(self, gen_ms):
        """Log Deepgram TTS audio generation time (cumulative for all chunks)."""
        prev = self._current_turn.get("tts_generation_ms") or 0
        self._current_turn["tts_generation_ms"] = round(prev + gen_ms, 1)

    def log_tts_first_audio(self):
        """Log the moment TTS first audio byte is piped to the player.
        Calculates user_to_speech_ms = now - stt_done_time."""
        stt_done = self._current_turn.get("stt_done_time")
        if stt_done is not None:
            gap_ms = (time.perf_counter() - stt_done) * 1000
            self._current_turn["user_to_speech_ms"] = round(gap_ms, 1)

    def log_tts_playback(self, play_ms):
        """Log mpg123 playback time (cumulative for all chunks)."""
        prev = self._current_turn.get("tts_playback_ms") or 0
        self._current_turn["tts_playback_ms"] = round(prev + play_ms, 1)

    def log_tts_total(self, total_ms):
        """Log total TTS time (generation + playback, all chunks)."""
        self._current_turn["tts_total_ms"] = round(total_ms, 1)

    # ── Memory tracking ─────────────────────────────────────────

    def log_fact_shared(self, fact_description):
        """Log a fact the user intentionally shared (for accuracy calc)."""
        self.facts_shared.append(fact_description)

    def log_facts_extracted(self, learnings_list):
        """Log facts the system extracted this turn."""
        self.facts_extracted.extend(learnings_list)
        self._current_turn["memory_facts_extracted_this_turn"] = len(learnings_list)

    # ── Quality rating ──────────────────────────────────────────

    def log_quality_rating(self, rating):
        """Log manual LLM quality rating (1-4) for current turn."""
        self._current_turn["llm_quality_rating"] = rating
        self._quality_ratings.append(rating)

    # ── Summary & persistence ───────────────────────────────────

    def get_summary(self):
        """Generate summary statistics for this conversation."""
        if not self.turns:
            return {"error": "No turns logged"}

        stt_latencies = [t["stt_latency_ms"] for t in self.turns if t.get("stt_latency_ms") is not None]
        groq_latencies = [t["groq_latency_ms"] for t in self.turns if t.get("groq_latency_ms") is not None]
        groq_full = [t["groq_full_latency_ms"] for t in self.turns if t.get("groq_full_latency_ms") is not None]
        tts_gen = [t["tts_generation_ms"] for t in self.turns if t.get("tts_generation_ms") is not None]
        tts_total = [t["tts_total_ms"] for t in self.turns if t.get("tts_total_ms") is not None]
        user_to_speech = [t["user_to_speech_ms"] for t in self.turns if t.get("user_to_speech_ms") is not None]
        turnarounds = [t["total_turnaround_ms"] for t in self.turns if t.get("total_turnaround_ms") is not None]
        qualities = [t["llm_quality_rating"] for t in self.turns if t.get("llm_quality_rating") is not None]

        def _stats(arr):
            if not arr:
                return {"min": None, "max": None, "avg": None, "count": 0}
            return {
                "min": round(min(arr), 1),
                "max": round(max(arr), 1),
                "avg": round(sum(arr) / len(arr), 1),
                "count": len(arr),
            }

        total_time = None
        if self.conversation_start and self.conversation_end:
            total_time = round((self.conversation_end - self.conversation_start) * 1000, 1)

        return {
            "trial_id": self.trial_id,
            "driver_type": self.driver_type,
            "total_turns": len(self.turns),
            "total_conversation_ms": total_time,
            "stt_latency_ms": _stats(stt_latencies),
            "groq_first_token_ms": _stats(groq_latencies),
            "groq_full_response_ms": _stats(groq_full),
            "tts_generation_ms": _stats(tts_gen),
            "tts_total_ms": _stats(tts_total),
            "user_to_speech_ms": _stats(user_to_speech),
            "total_turnaround_ms": _stats(turnarounds),
            "llm_quality_rating": _stats(qualities),
            "memory_facts_shared": len(self.facts_shared),
            "memory_facts_extracted": len(self.facts_extracted),
            "memory_extraction_accuracy": (
                round(len(self.facts_extracted) / len(self.facts_shared), 2)
                if self.facts_shared else None
            ),
        }

    def save(self):
        """Save full log (all turns + summary) to JSON file."""
        log = {
            "trial_id": self.trial_id,
            "driver_type": self.driver_type,
            "timestamp": datetime.now().isoformat(),
            "turns": self.turns,
            "facts_shared": self.facts_shared,
            "facts_extracted": [
                str(f) for f in self.facts_extracted
            ],
            "summary": self.get_summary(),
        }

        filepath = self.log_dir / f"{self.trial_id}.json"
        with open(filepath, "w") as f:
            json.dump(log, f, indent=2)

        print(f"\n📊 Metrics saved → {filepath}")
        return filepath

    def print_summary(self):
        """Print a formatted summary to console."""
        s = self.get_summary()
        print("\n" + "=" * 60)
        print(f"📊 TRIAL SUMMARY: {s['trial_id']}")
        print(f"   Driver type: {s['driver_type']}")
        print(f"   Total turns: {s['total_turns']}")
        if s['total_conversation_ms']:
            print(f"   Total conversation time: {s['total_conversation_ms']:.0f} ms ({s['total_conversation_ms']/1000:.1f}s)")
        print("-" * 60)

        for label, key in [
            ("STT Latency", "stt_latency_ms"),
            ("Groq First Token", "groq_first_token_ms"),
            ("Groq Full Response", "groq_full_response_ms"),
            ("TTS Generation", "tts_generation_ms"),
            ("TTS Total", "tts_total_ms"),
            ("User→Speech Gap", "user_to_speech_ms"),
            ("Total Turnaround", "total_turnaround_ms"),
        ]:
            st = s[key]
            if st["count"] > 0:
                print(f"   {label:>20s}:  avg={st['avg']:.0f}ms  min={st['min']:.0f}ms  max={st['max']:.0f}ms  (n={st['count']})")

        if s["llm_quality_rating"]["count"] > 0:
            q = s["llm_quality_rating"]
            print(f"   {'LLM Quality (1-4)':>20s}:  avg={q['avg']:.1f}  min={q['min']}  max={q['max']}  (n={q['count']})")

        print(f"\n   Memory: {s['memory_facts_extracted']} extracted / {s['memory_facts_shared']} shared", end="")
        if s["memory_extraction_accuracy"] is not None:
            print(f" ({s['memory_extraction_accuracy']*100:.0f}% accuracy)")
        else:
            print()
        print("=" * 60)
