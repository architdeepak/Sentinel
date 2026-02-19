"""
Memory Manager for Driver Drowsiness Detection System V7.
SQLite-backed persistent storage with three tables:
  - facts: driver personal facts with free-form types and confidence scoring
  - sessions: per-session metadata (scores, duration, turn count)
  - baselines: personal voice metric averages (from calibration + session updates)

V7 changes:
  - Replaces flat JSON profile with dynamic SQLite schema
  - Facts extracted by LLM with whatever types it decides are relevant
  - Deduplication: same fact reappearing increases confidence
  - Baseline metrics enable deviation-based reasoning (not absolute thresholds)
  - Context injection pulls most recent + most frequently confirmed facts
"""

import json
import time
import sqlite3
from pathlib import Path
from datetime import datetime

from groq import Groq
from config import Config


class MemoryManager:
    """SQLite-backed memory with dynamic facts, sessions, and baselines."""

    def __init__(self, db_path=None):
        self.db_path = db_path or (Path.home() / "sentinel_driver.db")
        self.conversation_transcript = []
        self._groq_client = None
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    session_id INTEGER,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    times_confirmed INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    avg_drowsy_score REAL,
                    max_drowsy_score REAL,
                    turn_count INTEGER DEFAULT 0,
                    duration_s REAL
                );

                CREATE TABLE IF NOT EXISTS baselines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT UNIQUE NOT NULL,
                    avg_value REAL,
                    min_value REAL,
                    max_value REAL,
                    sample_count INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
            """)

    def _connect(self):
        """Return a new SQLite connection."""
        return sqlite3.connect(str(self.db_path))

    # ═══════════════════════════════════════════════════════════
    #  Sessions
    # ═══════════════════════════════════════════════════════════

    def start_session(self):
        """Create a new session row, return its ID."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO sessions (started_at) VALUES (?)", (now,)
            )
            session_id = cursor.lastrowid
        print(f"📝 Session {session_id} started")
        return session_id

    def end_session(self, session_id, avg_drowsy_score=None, max_drowsy_score=None,
                    turn_count=0, duration_s=None):
        """Finalize a session with aggregate stats."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE sessions
                SET ended_at=?, avg_drowsy_score=?, max_drowsy_score=?,
                    turn_count=?, duration_s=?
                WHERE id=?
            """, (now, avg_drowsy_score, max_drowsy_score, turn_count,
                  duration_s, session_id))
        print(f"📝 Session {session_id} ended (turns={turn_count}, "
              f"avg_score={avg_drowsy_score:.2f})" if avg_drowsy_score else "")

    def get_session_count(self):
        """Return total number of completed sessions."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            return row[0] if row else 0

    def get_last_session_time(self):
        """Return ISO timestamp of the last completed session, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ended_at FROM sessions WHERE ended_at IS NOT NULL "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None

    # ═══════════════════════════════════════════════════════════
    #  Baselines
    # ═══════════════════════════════════════════════════════════

    def get_baselines(self):
        """Return all baselines as {metric_name: {avg, min, max, sample_count}}."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT metric_name, avg_value, min_value, max_value, sample_count "
                "FROM baselines"
            ).fetchall()
        return {
            row[0]: {
                "avg": row[1],
                "min": row[2],
                "max": row[3],
                "sample_count": row[4],
            }
            for row in rows
        }

    def update_baseline(self, metric_name, new_value):
        """Update a single baseline metric with a new observation (running average)."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT avg_value, min_value, max_value, sample_count "
                "FROM baselines WHERE metric_name=?",
                (metric_name,)
            ).fetchone()

            if existing:
                avg, mn, mx, count = existing
                new_count = count + 1
                new_avg = avg + (new_value - avg) / new_count
                new_min = min(mn, new_value)
                new_max = max(mx, new_value)
                conn.execute("""
                    UPDATE baselines
                    SET avg_value=?, min_value=?, max_value=?, sample_count=?, updated_at=?
                    WHERE metric_name=?
                """, (new_avg, new_min, new_max, new_count, now, metric_name))
            else:
                conn.execute("""
                    INSERT INTO baselines
                    (metric_name, avg_value, min_value, max_value, sample_count, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?)
                """, (metric_name, new_value, new_value, new_value, now))

    def update_baselines_bulk(self, metrics_dict):
        """Update multiple baselines at once. metrics_dict = {name: value}."""
        for name, value in metrics_dict.items():
            if value is not None:
                self.update_baseline(name, value)

    def store_calibration_baselines(self, samples):
        """Store baseline metrics from calibration (list of feature dicts).
        Replaces any existing baselines for the given metrics.
        """
        import numpy as np
        if not samples:
            return

        now = datetime.now().isoformat()

        # Aggregate across all calibration samples
        metrics_to_store = {}
        for key in ['energy_rms', 'speech_rate_wpm', 'pause_ratio',
                     'response_latency_s', 'peak_amplitude', 'duration_s']:
            values = [s[key] for s in samples if s.get(key) is not None]
            if values:
                metrics_to_store[key] = {
                    'avg': float(np.mean(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                }

        with self._connect() as conn:
            for name, stats in metrics_to_store.items():
                conn.execute("""
                    INSERT INTO baselines
                    (metric_name, avg_value, min_value, max_value, sample_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(metric_name) DO UPDATE SET
                        avg_value=excluded.avg_value, min_value=excluded.min_value,
                        max_value=excluded.max_value, sample_count=excluded.sample_count,
                        updated_at=excluded.updated_at
                """, (name, stats['avg'], stats['min'], stats['max'],
                      stats['count'], now))

        print(f"✓ Calibration baselines stored ({len(metrics_to_store)} metrics)")

    def needs_calibration(self):
        """Return True if no voice baselines exist yet."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM baselines "
                "WHERE metric_name IN ('energy_rms', 'speech_rate_wpm', 'pause_ratio')"
            ).fetchone()
            return row[0] < 3  # need at least the core 3

    def format_baselines_for_llm(self):
        """Format baselines as a compact string for system prompt context."""
        baselines = self.get_baselines()
        if not baselines:
            return ("No personal baselines recorded yet. "
                    "Judge metrics by absolute values until calibration data is available.")

        lines = []
        for metric, stats in baselines.items():
            label = metric.replace("_", " ")
            lines.append(
                f"  {label}: avg={stats['avg']:.4f} "
                f"(range {stats['min']:.4f}–{stats['max']:.4f}, "
                f"n={stats['sample_count']})"
            )
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    #  Facts
    # ═══════════════════════════════════════════════════════════

    def get_context_facts(self, limit=20):
        """Pull the most relevant facts for context injection.

        Prioritizes: most frequently confirmed, then most recently seen.
        Returns list of (fact_type, value, confidence, times_confirmed).
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT fact_type, value, confidence, times_confirmed
                FROM facts
                ORDER BY times_confirmed DESC, last_seen DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return rows

    def get_profile_summary(self):
        """Format driver facts as a readable summary for the system prompt."""
        facts = self.get_context_facts(limit=25)
        session_count = self.get_session_count()
        last_time = self.get_last_session_time()

        if not facts:
            if session_count == 0:
                return ("**NEW DRIVER** — No profile information yet. "
                        "This is your first conversation. Learn about "
                        "the driver naturally.")
            return (f"**RETURNING DRIVER** ({session_count} prior sessions) "
                    f"— No specific facts recorded yet.")

        # Group facts by type
        grouped = {}
        for fact_type, value, confidence, times_confirmed in facts:
            if fact_type not in grouped:
                grouped[fact_type] = []
            marker = ""
            if times_confirmed >= 3:
                marker = " (confirmed multiple times)"
            elif confidence < 0.5:
                marker = " (uncertain)"
            grouped[fact_type].append(f"{value}{marker}")

        lines = []
        lines.append(
            f"Sessions: {session_count}"
            + (f" | Last: {last_time}" if last_time else "")
        )

        for fact_type, values in grouped.items():
            label = fact_type.replace("_", " ").title()
            lines.append(f"{label}: {', '.join(values)}")

        return "\n".join(lines)

    def _store_fact(self, fact_type, value, session_id):
        """Store or deduplicate a single fact."""
        now = datetime.now().isoformat()
        fact_type = fact_type.strip().lower().replace(" ", "_")
        value = value.strip()

        with self._connect() as conn:
            # Check for existing fact with same type and value (case-insensitive)
            existing = conn.execute(
                "SELECT id, confidence, times_confirmed FROM facts "
                "WHERE fact_type=? AND LOWER(value)=LOWER(?)",
                (fact_type, value)
            ).fetchone()

            if existing:
                fid, conf, count = existing
                new_conf = min(conf + 0.2, 2.0)  # cap confidence at 2.0
                conn.execute("""
                    UPDATE facts
                    SET confidence=?, times_confirmed=?, last_seen=?, session_id=?
                    WHERE id=?
                """, (new_conf, count + 1, now, session_id, fid))
            else:
                conn.execute("""
                    INSERT INTO facts
                    (fact_type, value, confidence, session_id, first_seen, last_seen)
                    VALUES (?, ?, 1.0, ?, ?, ?)
                """, (fact_type, value, session_id, now, now))

    # ═══════════════════════════════════════════════════════════
    #  Transcript tracking
    # ═══════════════════════════════════════════════════════════

    def add_to_transcript(self, role, text):
        """Add a message to the conversation transcript for post-session extraction."""
        self.conversation_transcript.append((role, text))

    # ═══════════════════════════════════════════════════════════
    #  LLM-based fact extraction
    # ═══════════════════════════════════════════════════════════

    def _get_groq_client(self):
        """Lazy-init Groq client (only needed at end of conversation)."""
        if self._groq_client is None:
            self._groq_client = Groq(api_key=Config.GROQ_API_KEY)
        return self._groq_client

    def extract_and_store_facts(self, session_id):
        """Use LLM to extract facts from transcript → SQLite with dedup.

        Sends the full conversation transcript to an 8B model with a
        dynamic extraction prompt. Returns count of facts stored/updated.
        The LLM decides what fact types are relevant — not us.
        """
        if not self.conversation_transcript:
            print("ℹ️ No transcript to extract from")
            return 0

        transcript = "\n".join(
            f"{'DRIVER' if role == 'user' else 'SENTINEL'}: {text}"
            for role, text in self.conversation_transcript
        )

        # Include existing facts so the LLM can avoid near-duplicates
        existing_facts = self.get_context_facts(limit=30)
        existing_str = ""
        if existing_facts:
            existing_str = "\n\nEXISTING KNOWN FACTS (for deduplication):\n"
            for ft, val, conf, tc in existing_facts:
                existing_str += f"  - {ft}: {val} (confirmed {tc}x)\n"

        extraction_prompt = f"""Analyze this conversation between a drowsy driver and an AI assistant (Sentinel). Extract any personal facts the DRIVER revealed.
{existing_str}
CONVERSATION:
{transcript}

Return a JSON array of facts. Each fact object should have:
- "type": a short lowercase category (e.g. "name", "occupation", "hobby", "family_member", "music_preference", "destination", "vehicle", "food_preference", "pet", "schedule", "habit", "driving_routine", etc.)
- "value": the specific fact

Use whatever types you think best describe the information. Don't force facts into categories that don't fit — invent new types if needed.

Only include things the DRIVER stated or clearly implied. Skip anything Sentinel said or assumed.
If a fact already exists in EXISTING KNOWN FACTS with the same meaning, include it anyway (it will be deduplicated automatically).

Return ONLY the JSON array, no explanations. Return [] if no new facts found.

Example:
[
  {{"type": "name", "value": "Marcus"}},
  {{"type": "occupation", "value": "software engineer"}},
  {{"type": "hobby", "value": "rock climbing"}},
  {{"type": "family_member", "value": "daughter named Sophie"}},
  {{"type": "driving_routine", "value": "commutes 45 mins each way"}}
]"""

        try:
            client = self._get_groq_client()
            response = client.chat.completions.create(
                model=Config.GROQ_EXTRACTION_MODEL,
                messages=[
                    {"role": "system",
                     "content": ("You extract structured personal facts from "
                                 "conversations. Return only a valid JSON array.")},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=500,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences
            if raw.startswith("```"):
                raw = raw[3:]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
                if raw.endswith("```"):
                    raw = raw[:-3]
            raw = raw.strip()

            facts = json.loads(raw)
            if not isinstance(facts, list):
                facts = [facts]

            count = 0
            for fact in facts:
                ft = fact.get("type", "").strip()
                val = fact.get("value", "").strip()
                if ft and val:
                    self._store_fact(ft, val, session_id)
                    count += 1

            print(f"✓ LLM extraction: stored/updated {count} facts")
            self.conversation_transcript = []
            return count

        except json.JSONDecodeError as e:
            print(f"⚠️ LLM extraction returned invalid JSON: {e}")
            return 0
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return 0

    # ═══════════════════════════════════════════════════════════
    #  Utility
    # ═══════════════════════════════════════════════════════════

    def dump_all(self):
        """Debug utility: print everything in the database."""
        with self._connect() as conn:
            print("\n═══ SESSIONS ═══")
            for row in conn.execute("SELECT * FROM sessions").fetchall():
                print(f"  {row}")

            print("\n═══ FACTS ═══")
            for row in conn.execute(
                "SELECT * FROM facts ORDER BY times_confirmed DESC"
            ).fetchall():
                print(f"  {row}")

            print("\n═══ BASELINES ═══")
            for row in conn.execute("SELECT * FROM baselines").fetchall():
                print(f"  {row}")
