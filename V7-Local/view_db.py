#!/usr/bin/env python3
"""
Sentinel Database Viewer — V7

Displays all contents of the sentinel_driver.db SQLite database
in a formatted, readable way. Works on Raspberry Pi via terminal.

Usage:
    python view_db.py              # View all data
    python view_db.py --facts      # View only facts
    python view_db.py --sessions   # View only sessions
    python view_db.py --baselines  # View only baselines
    python view_db.py --reset      # Delete all data (with confirmation)
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path.home() / "sentinel_driver.db"


def connect():
    if not DB_PATH.exists():
        print(f"  Database not found at: {DB_PATH}")
        print("  Run the main system first to create it.")
        sys.exit(1)
    return sqlite3.connect(str(DB_PATH))


def show_summary(conn):
    print("\n" + "=" * 70)
    print("  SENTINEL DATABASE SUMMARY")
    print(f"  Path: {DB_PATH}")
    print("=" * 70)

    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    facts = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    baselines = conn.execute("SELECT COUNT(*) FROM baselines").fetchone()[0]
    fact_types = conn.execute(
        "SELECT COUNT(DISTINCT fact_type) FROM facts"
    ).fetchone()[0]

    print(f"  Sessions:  {sessions}")
    print(f"  Facts:     {facts} ({fact_types} types)")
    print(f"  Baselines: {baselines} metrics")

    last = conn.execute(
        "SELECT ended_at FROM sessions WHERE ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if last:
        print(f"  Last session: {last[0]}")


def show_sessions(conn):
    print("\n" + "=" * 70)
    print("  SESSIONS")
    print("=" * 70)

    rows = conn.execute(
        "SELECT id, started_at, ended_at, avg_drowsy_score, "
        "max_drowsy_score, turn_count, duration_s "
        "FROM sessions ORDER BY id"
    ).fetchall()

    if not rows:
        print("  (no sessions recorded)")
        return

    for row in rows:
        sid, started, ended, avg_s, max_s, turns, dur = row
        print(f"\n  Session #{sid}")
        print(f"    Started:   {started}")
        print(f"    Ended:     {ended or '(in progress)'}")
        if avg_s is not None:
            print(f"    Avg Score: {avg_s:.3f}  |  Max Score: {max_s:.3f}")
        if turns:
            print(f"    Turns:     {turns}")
        if dur:
            print(f"    Duration:  {dur:.1f}s ({dur / 60:.1f} min)")

    print(f"\n  Total: {len(rows)} session(s)")


def show_facts(conn):
    print("\n" + "=" * 70)
    print("  FACTS (Driver Profile)")
    print("=" * 70)

    rows = conn.execute(
        "SELECT fact_type, value, confidence, times_confirmed, "
        "first_seen, last_seen, session_id "
        "FROM facts ORDER BY times_confirmed DESC, last_seen DESC"
    ).fetchall()

    if not rows:
        print("  (no facts recorded)")
        return

    # Group by type
    grouped = {}
    for ft, val, conf, tc, first, last, sid in rows:
        if ft not in grouped:
            grouped[ft] = []
        grouped[ft].append((val, conf, tc, first, last, sid))

    for fact_type, entries in grouped.items():
        label = fact_type.replace("_", " ").title()
        print(f"\n  [{label}]")
        for val, conf, tc, first, last, sid in entries:
            confirmed = f" (confirmed {tc}x)" if tc > 1 else ""
            print(f"    * {val}{confirmed}")
            print(f"      confidence={conf:.1f}  |  "
                  f"first={first[:16]}  |  last={last[:16]}  |  "
                  f"session={sid}")

    print(f"\n  Total: {len(rows)} fact(s) across {len(grouped)} type(s)")


def show_baselines(conn):
    print("\n" + "=" * 70)
    print("  VOICE BASELINES")
    print("=" * 70)

    rows = conn.execute(
        "SELECT metric_name, avg_value, min_value, max_value, "
        "sample_count, updated_at FROM baselines ORDER BY metric_name"
    ).fetchall()

    if not rows:
        print("  (no baselines recorded — run calibration first)")
        return

    for name, avg, mn, mx, count, updated in rows:
        label = name.replace("_", " ").title()
        print(f"\n  {label}:")
        print(f"    Average:  {avg:.4f}")
        print(f"    Range:    {mn:.4f} - {mx:.4f}")
        print(f"    Samples:  {count}")
        print(f"    Updated:  {updated}")

    print(f"\n  Total: {len(rows)} baseline metric(s)")


def reset_database(conn):
    confirm = input(
        "\n  WARNING: This will DELETE ALL DATA. Type 'yes' to confirm: "
    )
    if confirm.strip().lower() == 'yes':
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM baselines")
        conn.commit()
        print("  All data deleted.")
    else:
        print("  Cancelled.")


if __name__ == "__main__":
    conn = connect()

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--facts":
            show_facts(conn)
        elif arg == "--sessions":
            show_sessions(conn)
        elif arg == "--baselines":
            show_baselines(conn)
        elif arg == "--reset":
            reset_database(conn)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python view_db.py [--facts | --sessions | --baselines | --reset]")
    else:
        show_summary(conn)
        show_sessions(conn)
        show_facts(conn)
        show_baselines(conn)

    conn.close()
    print()
