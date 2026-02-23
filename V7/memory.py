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
import sqlite3
import threading
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
        self._db_lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
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
                    duration_s REAL,
                    recovery_time_s REAL,
                    peak_perclos REAL,
                    peak_slow_blinks INTEGER,
                    avg_energy_rms REAL,
                    avg_speech_rate REAL,
                    avg_response_latency REAL,
                    trigger_reason TEXT
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

                CREATE TABLE IF NOT EXISTS reasoner_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    confidence REAL,
                    reasoning TEXT,
                    perclos REAL,
                    blink_rate INTEGER,
                    slow_blinks INTEGER,
                    ear_std REAL,
                    pitch_var REAL,
                    microsleep INTEGER DEFAULT 0,
                    head_down INTEGER DEFAULT 0,
                    head_roll INTEGER DEFAULT 0,
                    energy_rms REAL,
                    speech_rate_wpm REAL,
                    pause_ratio REAL,
                    response_latency_s REAL
                );

                CREATE TABLE IF NOT EXISTS driver_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    times_observed INTEGER DEFAULT 1,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
            """)
            # Migrations: add columns that may be missing from older DBs
            existing = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            migrations = [
                "ALTER TABLE sessions ADD COLUMN recovery_time_s REAL",
                "ALTER TABLE sessions ADD COLUMN peak_perclos REAL",
                "ALTER TABLE sessions ADD COLUMN peak_slow_blinks INTEGER",
                "ALTER TABLE sessions ADD COLUMN avg_energy_rms REAL",
                "ALTER TABLE sessions ADD COLUMN avg_speech_rate REAL",
                "ALTER TABLE sessions ADD COLUMN avg_response_latency REAL",
                "ALTER TABLE sessions ADD COLUMN trigger_reason TEXT",
            ]
            for sql in migrations:
                col = sql.split("ADD COLUMN ")[1].split()[0]
                if col not in existing:
                    conn.execute(sql)

    def _connect(self):
        """Return the persistent SQLite connection."""
        return self._conn

    def close(self):
        """Close the persistent DB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def reset_database(self):
        """Drop all tables and recreate them. Irreversible."""
        with self._connect() as conn:
            conn.executescript("""
                DROP TABLE IF EXISTS facts;
                DROP TABLE IF EXISTS sessions;
                DROP TABLE IF EXISTS baselines;
                DROP TABLE IF EXISTS reasoner_evaluations;
                DROP TABLE IF EXISTS driver_patterns;
            """)
        self._init_db()
        print(f"\u2713 Database reset: {self.db_path}")

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
                    turn_count=0, duration_s=None, recovery_time_s=None,
                    peak_perclos=None, peak_slow_blinks=None,
                    avg_energy_rms=None, avg_speech_rate=None,
                    avg_response_latency=None, trigger_reason=None):
        """Finalize a session with aggregate stats."""
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute("""
                UPDATE sessions
                SET ended_at=?, avg_drowsy_score=?, max_drowsy_score=?,
                    turn_count=?, duration_s=?, recovery_time_s=?,
                    peak_perclos=?, peak_slow_blinks=?,
                    avg_energy_rms=?, avg_speech_rate=?,
                    avg_response_latency=?, trigger_reason=?
                WHERE id=?
            """, (now, avg_drowsy_score, max_drowsy_score, turn_count,
                  duration_s, recovery_time_s, peak_perclos, peak_slow_blinks,
                  avg_energy_rms, avg_speech_rate, avg_response_latency,
                  trigger_reason, session_id))
        print(f"📝 Session {session_id} ended (turns={turn_count}, "
              f"avg_score={avg_drowsy_score:.2f})" if avg_drowsy_score is not None else 
              f"📝 Session {session_id} ended (turns={turn_count})")

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

    def get_driver_history_for_llm(self):
        """Build a rich driver history summary from past sessions for LLM context.

        Returns a formatted string covering:
          - Session frequency and patterns
          - Average/worst drowsiness metrics across sessions
          - Typical recovery times
          - Voice metric trends
          - Time-of-day patterns
        """
        with self._connect() as conn:
            sessions = conn.execute("""
                SELECT started_at, ended_at, avg_drowsy_score, max_drowsy_score,
                       turn_count, duration_s, recovery_time_s,
                       peak_perclos, peak_slow_blinks,
                       avg_energy_rms, avg_speech_rate, avg_response_latency,
                       trigger_reason
                FROM sessions WHERE ended_at IS NOT NULL
                ORDER BY id DESC LIMIT 20
            """).fetchall()

        if not sessions:
            return ""

        total = len(sessions)

        # Aggregate stats
        avg_scores = [s[2] for s in sessions if s[2] is not None]
        max_scores = [s[3] for s in sessions if s[3] is not None]
        durations = [s[5] for s in sessions if s[5] is not None]
        recovery_times = [s[6] for s in sessions if s[6] is not None]
        peak_perclos = [s[7] for s in sessions if s[7] is not None]
        peak_slow = [s[8] for s in sessions if s[8] is not None]
        avg_rms = [s[9] for s in sessions if s[9] is not None]
        avg_rate = [s[10] for s in sessions if s[10] is not None]
        avg_lat = [s[11] for s in sessions if s[11] is not None]
        triggers = [s[12] for s in sessions if s[12] is not None]

        lines = []
        lines.append(f"Total activations: {total}")

        # Time-of-day patterns
        hours = []
        for s in sessions:
            try:
                h = int(s[0][11:13])  # Extract hour from ISO timestamp
                hours.append(h)
            except (ValueError, IndexError):
                pass
        if hours:
            from collections import Counter
            hour_counts = Counter(hours)
            common_hours = hour_counts.most_common(3)
            time_strs = []
            for h, c in common_hours:
                period = "morning" if 5 <= h < 12 else "afternoon" if 12 <= h < 17 else "evening" if 17 <= h < 21 else "night"
                time_strs.append(f"{h}:00 ({period}, {c}x)")
            lines.append(f"Most common times: {', '.join(time_strs)}")

        # Drowsiness severity
        if avg_scores:
            lines.append(f"Avg drowsy score across sessions: {sum(avg_scores)/len(avg_scores):.3f}")
        if max_scores:
            lines.append(f"Worst peak score ever: {max(max_scores):.3f}")
        if peak_perclos:
            lines.append(f"Worst PERCLOS ever: {max(peak_perclos):.3f}")
        if peak_slow:
            lines.append(f"Worst slow blinks in a session: {max(peak_slow)}")

        # Recovery patterns
        if recovery_times:
            avg_rec = sum(recovery_times) / len(recovery_times)
            fastest = min(recovery_times)
            lines.append(f"Avg recovery time: {avg_rec:.0f}s (fastest: {fastest:.0f}s)")
        if durations:
            avg_dur = sum(durations) / len(durations)
            lines.append(f"Avg conversation duration: {avg_dur:.0f}s")

        # Voice patterns across sessions
        voice_parts = []
        if avg_rms:
            voice_parts.append(f"avg energy={sum(avg_rms)/len(avg_rms):.4f}")
        if avg_rate:
            voice_parts.append(f"avg speech rate={sum(avg_rate)/len(avg_rate):.0f}wpm")
        if avg_lat:
            voice_parts.append(f"avg response latency={sum(avg_lat)/len(avg_lat):.1f}s")
        if voice_parts:
            lines.append(f"Voice across sessions: {', '.join(voice_parts)}")

        # Trigger reasons
        if triggers:
            from collections import Counter
            trigger_counts = Counter(triggers)
            parts = [f"{reason}: {count}x" for reason, count in trigger_counts.most_common()]
            lines.append(f"Trigger reasons: {', '.join(parts)}")

        # Recent session summaries (last 3)
        lines.append("")
        lines.append("Recent sessions:")
        for i, s in enumerate(sessions[:3]):
            started = s[0][:16].replace("T", " ") if s[0] else "?"
            score_str = f"avg_score={s[2]:.3f}" if s[2] is not None else ""
            dur_str = f"duration={s[5]:.0f}s" if s[5] is not None else ""
            rec_str = f"recovery={s[6]:.0f}s" if s[6] is not None else ""
            parts = [p for p in [score_str, dur_str, rec_str] if p]
            lines.append(f"  {started}: {', '.join(parts)}")

        return "\n".join(lines)

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

    def update_baselines_bulk(self, metrics_dict):
        """Update multiple baselines in a single transaction."""
        now = datetime.now().isoformat()
        with self._db_lock:
            for name, value in metrics_dict.items():
                if value is None:
                    continue
                existing = self._conn.execute(
                    "SELECT avg_value, min_value, max_value, sample_count "
                    "FROM baselines WHERE metric_name=?",
                    (name,)
                ).fetchone()
                if existing:
                    avg, mn, mx, count = existing
                    new_count = count + 1
                    new_avg = avg + (value - avg) / new_count
                    self._conn.execute("""
                        UPDATE baselines
                        SET avg_value=?, min_value=?, max_value=?,
                            sample_count=?, updated_at=?
                        WHERE metric_name=?
                    """, (new_avg, min(mn, value), max(mx, value),
                          new_count, now, name))
                else:
                    self._conn.execute("""
                        INSERT INTO baselines
                        (metric_name, avg_value, min_value, max_value,
                         sample_count, updated_at)
                        VALUES (?, ?, ?, ?, 1, ?)
                    """, (name, value, value, value, now))
            self._conn.commit()

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
    #  Reasoner Evaluations & Driver Pattern Learning
    # ═══════════════════════════════════════════════════════════

    def store_reasoner_evaluations(self, session_id, evaluations):
        """Store a batch of 8B reasoner evaluations from a session.

        Args:
            session_id: current session ID
            evaluations: list of dicts with keys:
                level, confidence, reasoning, perclos, blink_rate,
                slow_blinks, ear_std, pitch_var, microsleep, head_down,
                head_roll, energy_rms, speech_rate_wpm, pause_ratio,
                response_latency_s, timestamp
        """
        if not evaluations:
            return
        with self._connect() as conn:
            conn.executemany("""
                INSERT INTO reasoner_evaluations
                (session_id, timestamp, level, confidence, reasoning,
                 perclos, blink_rate, slow_blinks, ear_std, pitch_var,
                 microsleep, head_down, head_roll,
                 energy_rms, speech_rate_wpm, pause_ratio, response_latency_s)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    session_id,
                    ev.get('timestamp', datetime.now().isoformat()),
                    ev.get('level', 'ALERT'),
                    ev.get('confidence', 0.5),
                    ev.get('reasoning', ''),
                    ev.get('perclos'),
                    ev.get('blink_rate'),
                    ev.get('slow_blinks'),
                    ev.get('ear_std'),
                    ev.get('pitch_var'),
                    int(ev.get('microsleep', False)),
                    int(ev.get('head_down', False)),
                    int(ev.get('head_roll', False)),
                    ev.get('energy_rms'),
                    ev.get('speech_rate_wpm'),
                    ev.get('pause_ratio'),
                    ev.get('response_latency_s'),
                )
                for ev in evaluations
            ])
        print(f"✓ Stored {len(evaluations)} reasoner evaluations for session {session_id}")

    def learn_driver_patterns(self, session_id):
        """Use 8B model to analyze stored evaluations and learn driver-specific patterns.

        Runs post-session: looks at all historical evaluations across sessions
        to identify what metric combinations reliably indicate drowsiness for
        THIS specific driver.
        """
        # Get all evaluations (recent first, cap at last 100)
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT level, confidence, reasoning,
                       perclos, blink_rate, slow_blinks, ear_std, pitch_var,
                       microsleep, head_down, head_roll,
                       energy_rms, speech_rate_wpm, pause_ratio, response_latency_s
                FROM reasoner_evaluations
                ORDER BY id DESC LIMIT 100
            """).fetchall()

        if len(rows) < 5:
            print("ℹ️ Not enough evaluations for pattern learning yet "
                  f"({len(rows)}/5 needed)")
            return

        # Get existing patterns for dedup
        existing_patterns = self._get_existing_patterns()

        # Build summary for 8B analysis
        drowsy_cases = []
        alert_cases = []
        for r in rows:
            level = r[0]
            entry = {
                'level': level, 'confidence': r[1],
                'perclos': r[3], 'blink_rate': r[4], 'slow_blinks': r[5],
                'ear_std': r[6], 'pitch_var': r[7],
                'microsleep': bool(r[8]), 'head_down': bool(r[9]),
                'head_roll': bool(r[10]),
                'energy_rms': r[11], 'speech_rate_wpm': r[12],
                'pause_ratio': r[13], 'response_latency_s': r[14],
            }
            if level in ('DROWSY', 'CRITICAL'):
                drowsy_cases.append(entry)
            else:
                alert_cases.append(entry)

        prompt = self._build_pattern_learning_prompt(
            drowsy_cases, alert_cases, existing_patterns
        )

        try:
            client = self._get_groq_client()
            response = client.chat.completions.create(
                model=Config.GROQ_EXTRACTION_MODEL,
                messages=[
                    {"role": "system",
                     "content": ("You are a pattern recognition expert analyzing "
                                 "driver drowsiness data. Extract recurring patterns "
                                 "specific to this driver. Return ONLY valid JSON.")},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=600,
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw[3:]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
                if raw.endswith("```"):
                    raw = raw[:-3]
            raw = raw.strip()

            patterns = json.loads(raw)
            if not isinstance(patterns, list):
                patterns = [patterns]

            count = 0
            now = datetime.now().isoformat()
            for p in patterns:
                ptype = p.get('type', '').strip().lower().replace(' ', '_')
                desc = p.get('description', '').strip()
                if ptype and desc:
                    self._store_pattern(ptype, desc, now)
                    count += 1

            print(f"✓ Pattern learning: stored/updated {count} driver patterns")

        except Exception as e:
            print(f"⚠️ Pattern learning failed: {e}")

    def _build_pattern_learning_prompt(self, drowsy_cases, alert_cases,
                                       existing_patterns):
        """Build prompt for 8B to discover driver-specific patterns."""
        def _metric_summary(cases, key, fmt=".3f"):
            """Return 'min – max (avg X)' string for a metric, or None."""
            vals = [c[key] for c in cases if c.get(key) is not None]
            if not vals:
                return None
            avg = sum(vals) / len(vals)
            return f"{min(vals):{fmt}} – {max(vals):{fmt}} (avg {avg:{fmt}})"

        parts = []
        parts.append("Analyze this driver's drowsiness detection history and identify "
                     "PERSONAL patterns — things that are specific to THIS driver.")
        parts.append("")

        # Drowsy cases summary
        parts.append(f"## DROWSY/CRITICAL Cases ({len(drowsy_cases)} evaluations)")
        if drowsy_cases:
            for key, label, fmt in [
                ('perclos', 'PERCLOS range', '.3f'),
                ('slow_blinks', 'Slow blinks range', '.1f'),
                ('energy_rms', 'Voice energy (RMS) range', '.4f'),
                ('speech_rate_wpm', 'Speech rate range', '.1f'),
                ('pause_ratio', 'Pause ratio range', '.3f'),
                ('response_latency_s', 'Response latency range', '.1f'),
            ]:
                s = _metric_summary(drowsy_cases, key, fmt)
                if s:
                    unit = ' wpm' if 'rate' in key else ('s' if 'latency' in key else '')
                    parts.append(f"  {label}: {s}{unit}")
            # Count head/microsleep events
            ms_count = sum(1 for c in drowsy_cases if c['microsleep'])
            hd_count = sum(1 for c in drowsy_cases if c['head_down'])
            hr_count = sum(1 for c in drowsy_cases if c['head_roll'])
            if ms_count or hd_count or hr_count:
                parts.append(f"  Events: microsleep={ms_count}, head_down={hd_count}, head_roll={hr_count}")
        else:
            parts.append("  (no drowsy cases recorded yet)")

        # Alert cases summary
        parts.append(f"")
        parts.append(f"## ALERT/MILD Cases ({len(alert_cases)} evaluations)")
        if alert_cases:
            for key, label, fmt in [
                ('perclos', 'PERCLOS range', '.3f'),
                ('energy_rms', 'Voice energy (RMS) range', '.4f'),
                ('speech_rate_wpm', 'Speech rate range', '.1f'),
            ]:
                s = _metric_summary(alert_cases, key, fmt)
                if s:
                    unit = ' wpm' if 'rate' in key else ''
                    parts.append(f"  {label}: {s}{unit}")

        # Existing patterns for dedup
        if existing_patterns:
            parts.append("")
            parts.append("## Already Known Patterns (avoid duplicating these)")
            for pt, desc, times in existing_patterns:
                parts.append(f"  - [{pt}] {desc} (observed {times}x)")

        parts.append("")
        parts.append("""Identify NEW patterns from this data. Look for:
- What PERCLOS level reliably indicates drowsiness for this driver?
- Does this driver show voice changes (quieter, slower, more pauses) when drowsy?
- Are there audio-visual combinations unique to this driver (e.g. slow blinks + quiet voice)?
- Does this driver have specific pre-drowsiness patterns (e.g. speech slows before eyes close)?
- Any unusual patterns (e.g. high blink rate when drowsy instead of low)?

Return a JSON array of patterns:
[{"type": "visual_pattern|voice_pattern|combined_pattern|threshold_pattern|sequence_pattern", "description": "specific finding"}]

Be specific with numbers. e.g. "Driver becomes drowsy when PERCLOS > 0.18 AND speech rate drops below 120 wpm" not "Driver shows drowsiness with high PERCLOS".
Return [] if not enough data for meaningful patterns.""")

        return "\n".join(parts)

    def _store_pattern(self, pattern_type, description, timestamp):
        """Store or update a driver pattern with dedup."""
        with self._connect() as conn:
            # Check for similar existing pattern (same type)
            existing = conn.execute(
                "SELECT id, times_observed FROM driver_patterns "
                "WHERE pattern_type=? AND LOWER(description)=LOWER(?)",
                (pattern_type, description)
            ).fetchone()

            if existing:
                pid, count = existing
                conn.execute("""
                    UPDATE driver_patterns
                    SET times_observed=?, last_seen=?,
                        confidence=MIN(confidence + 0.1, 2.0)
                    WHERE id=?
                """, (count + 1, timestamp, pid))
            else:
                conn.execute("""
                    INSERT INTO driver_patterns
                    (pattern_type, description, confidence, times_observed,
                     first_seen, last_seen)
                    VALUES (?, ?, 1.0, 1, ?, ?)
                """, (pattern_type, description, timestamp, timestamp))

    def _get_existing_patterns(self):
        """Get all driver patterns as [(type, description, times_observed)]."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT pattern_type, description, times_observed "
                "FROM driver_patterns ORDER BY times_observed DESC"
            ).fetchall()

    def get_driver_patterns_for_reasoner(self):
        """Format driver patterns as a string for the 8B reasoner's system prompt.

        Returns a multi-line summary of what the system has learned about
        this specific driver's drowsiness indicators.
        """
        patterns = self._get_existing_patterns()
        if not patterns:
            return ""

        lines = []
        for ptype, desc, times in patterns:
            label = ptype.replace('_', ' ').title()
            reliability = "" if times < 2 else f" (observed {times}x)"
            lines.append(f"- [{label}] {desc}{reliability}")
        return "\n".join(lines)

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
                if not isinstance(fact, dict):
                    continue
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

            print("\n═══ DRIVER PATTERNS ═══")
            for row in conn.execute(
                "SELECT * FROM driver_patterns ORDER BY times_observed DESC"
            ).fetchall():
                print(f"  {row}")

            print("\n═══ REASONER EVALUATIONS (last 20) ═══")
            for row in conn.execute(
                "SELECT id, session_id, level, confidence, reasoning "
                "FROM reasoner_evaluations ORDER BY id DESC LIMIT 20"
            ).fetchall():
                print(f"  {row}")
