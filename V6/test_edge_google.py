#!/usr/bin/env python3
"""
Edge TTS + Google STT Benchmark Test
======================================
Tests Google STT accuracy and Edge-TTS latency (the V5 stack).

STT test: Displays reference phrases → you read them → Google STT transcribes
          → computes Word Error Rate (WER) and latency.
TTS test: Sends phrases to Edge-TTS → measures generation time, streams to
          mpg123, measures total playback time.

Results saved to V6/test_logs/edge_google_benchmark_<timestamp>.json

Usage:
    python test_edge_google.py            # Run full benchmark (STT + TTS)
    python test_edge_google.py --stt      # STT only
    python test_edge_google.py --tts      # TTS only
"""

import sys
import os
import json
import time
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

import edge_tts
import speech_recognition as sr

# Ensure V6 modules are importable (for shared config/.env loading)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config

# ── Settings (V5 defaults) ─────────────────────────────────────
EDGE_TTS_VOICE = "en-US-JennyNeural"
EDGE_TTS_RATE = "+35%"

# ── Test phrases (SAME as Deepgram test for fair comparison) ───
TEST_PHRASES = [
    "I'm feeling a little tired but I think I can keep driving.",
    "Can you turn up the music? Something with a fast beat.",
    "I had about six hours of sleep last night, maybe less.",
    "My exit is in about fifteen miles, I think I can make it.",
    "Yeah I'm fine, just yawned because it's warm in here.",
    "I've been on the road for three hours now without a break.",
    "The highway is pretty empty right now, not much traffic.",
    "I grabbed a coffee at the last rest stop but it didn't help much.",
    "My eyes feel heavy but I don't want to pull over yet.",
    "OK let me try that breathing exercise you mentioned.",
]


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute Word Error Rate between reference and hypothesis.
    Returns dict with wer, substitutions, insertions, deletions, ref_words, hyp_words.
    Uses dynamic programming (Levenshtein on words).
    """
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()

    r = len(ref_words)
    h = len(hyp_words)

    d = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub = d[i - 1][j - 1] + 1
                ins = d[i][j - 1] + 1
                dlt = d[i - 1][j] + 1
                d[i][j] = min(sub, ins, dlt)

    i, j = r, h
    subs = ins = dels = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dels += 1
            i -= 1

    wer = d[r][h] / r if r > 0 else 0.0

    return {
        "wer": round(wer, 4),
        "errors": d[r][h],
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "ref_word_count": r,
        "hyp_word_count": len(hyp_words),
    }


# ── STT Test (Google) ─────────────────────────────────────────

def run_stt_test(phrases=None):
    """Run Google STT accuracy + latency test."""
    phrases = phrases or TEST_PHRASES

    print("\n" + "=" * 60)
    print("🎤 GOOGLE STT BENCHMARK")
    print("=" * 60)
    print(f"   Phrases: {len(phrases)}")
    print(f"   Engine: Google Speech Recognition (free tier)")
    print()

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.5
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 1.5

    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("✓ Microphone calibrated\n")

    results = []

    for i, phrase in enumerate(phrases):
        print(f"─── Phrase {i + 1}/{len(phrases)} ───")
        print(f"📖 READ THIS: \"{phrase}\"")
        input("   Press Enter when ready, then speak...")

        print("   🎤 Listening...")
        t_start = time.perf_counter()

        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)

            t_api = time.perf_counter()
            transcript = recognizer.recognize_google(audio)
            api_ms = (time.perf_counter() - t_api) * 1000
            total_ms = (time.perf_counter() - t_start) * 1000

            wer_result = compute_wer(phrase, transcript)

            print(f"   ✓ Got: \"{transcript}\"")
            print(f"   📊 WER: {wer_result['wer']:.1%}  |  API: {api_ms:.0f}ms  |  Total: {total_ms:.0f}ms")

            results.append({
                "phrase_index": i + 1,
                "reference": phrase,
                "hypothesis": transcript,
                "wer": wer_result["wer"],
                "wer_details": wer_result,
                "confidence": None,  # Google free tier doesn't return confidence
                "api_latency_ms": round(api_ms, 1),
                "total_latency_ms": round(total_ms, 1),
            })

        except sr.WaitTimeoutError:
            print("   ⏱️ Timeout — no speech detected")
            results.append({"phrase_index": i + 1, "reference": phrase, "error": "timeout"})
        except sr.UnknownValueError:
            print("   ❓ Could not understand audio")
            results.append({
                "phrase_index": i + 1,
                "reference": phrase,
                "hypothesis": "",
                "wer": 1.0,
                "wer_details": {"wer": 1.0, "errors": len(phrase.split()),
                                "substitutions": 0, "insertions": 0,
                                "deletions": len(phrase.split()),
                                "ref_word_count": len(phrase.split()), "hyp_word_count": 0},
                "confidence": None,
                "api_latency_ms": None,
                "total_latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
                "error": "unknown_value",
            })
        except sr.RequestError as e:
            print(f"   ⚠️ Google API error: {e}")
            results.append({"phrase_index": i + 1, "reference": phrase, "error": str(e)})
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            results.append({"phrase_index": i + 1, "reference": phrase, "error": str(e)})

        print()

    return results


# ── TTS Test (Edge-TTS) ───────────────────────────────────────

async def _generate_edge_tts(text: str) -> list:
    """Generate audio chunks from Edge-TTS."""
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE, rate=EDGE_TTS_RATE)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio" and chunk["data"]:
            chunks.append(chunk["data"])
    return chunks


def run_tts_test(phrases=None):
    """Run Edge-TTS latency test."""
    phrases = phrases or TEST_PHRASES

    print("\n" + "=" * 60)
    print("🔊 EDGE-TTS BENCHMARK")
    print("=" * 60)
    print(f"   Phrases: {len(phrases)}")
    print(f"   Voice: {EDGE_TTS_VOICE}")
    print(f"   Rate: {EDGE_TTS_RATE}")
    print()

    loop = asyncio.new_event_loop()
    results = []

    for i, phrase in enumerate(phrases):
        print(f"─── Phrase {i + 1}/{len(phrases)} ───")
        print(f"   \"{phrase}\"")

        try:
            # Generation
            t_start = time.perf_counter()
            chunks = loop.run_until_complete(_generate_edge_tts(phrase))
            generation_ms = (time.perf_counter() - t_start) * 1000

            if not chunks:
                print("   ⚠️ No audio generated")
                results.append({"phrase_index": i + 1, "text": phrase, "error": "no_audio"})
                continue

            total_bytes = sum(len(c) for c in chunks)

            # Playback — stream to player
            t_play = time.perf_counter()
            proc = None
            for cmd in [["mpg123", "-q", "-"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"]]:
                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except FileNotFoundError:
                    continue

            if proc:
                for chunk in chunks:
                    proc.stdin.write(chunk)
                proc.stdin.close()
                proc.wait()

            playback_ms = (time.perf_counter() - t_play) * 1000
            total_ms = (time.perf_counter() - t_start) * 1000

            print(f"   📊 Gen: {generation_ms:.0f}ms  |  Playback: {playback_ms:.0f}ms  |  Total: {total_ms:.0f}ms  |  Size: {total_bytes / 1024:.1f}KB")

            results.append({
                "phrase_index": i + 1,
                "text": phrase,
                "char_count": len(phrase),
                "word_count": len(phrase.split()),
                "generation_ms": round(generation_ms, 1),
                "playback_ms": round(playback_ms, 1),
                "total_ms": round(total_ms, 1),
                "audio_bytes": total_bytes,
            })

        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            results.append({"phrase_index": i + 1, "text": phrase, "error": str(e)})

        print()

    loop.close()
    return results


# ── Summary + Save ─────────────────────────────────────────────

def print_and_save(stt_results, tts_results):
    """Print summary and save to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("📊 EDGE-TTS + GOOGLE STT BENCHMARK RESULTS")
    print("=" * 60)

    report = {
        "engine": "edge_google",
        "timestamp": datetime.now().isoformat(),
        "stt_engine": "Google Speech Recognition",
        "tts_voice": EDGE_TTS_VOICE,
        "tts_rate": EDGE_TTS_RATE,
    }

    if stt_results:
        valid = [r for r in stt_results if "wer" in r]
        if valid:
            wers = [r["wer"] for r in valid]
            api_lats = [r["api_latency_ms"] for r in valid if r.get("api_latency_ms") is not None]
            total_lats = [r["total_latency_ms"] for r in valid if r.get("total_latency_ms") is not None]

            stt_summary = {
                "phrases_tested": len(stt_results),
                "phrases_successful": len(valid),
                "wer_avg": round(sum(wers) / len(wers), 4),
                "wer_min": round(min(wers), 4),
                "wer_max": round(max(wers), 4),
                "wer_zero_count": sum(1 for w in wers if w == 0),
                "accuracy_avg": round(1 - sum(wers) / len(wers), 4),
                "api_latency_avg_ms": round(sum(api_lats) / len(api_lats), 1) if api_lats else None,
                "api_latency_min_ms": round(min(api_lats), 1) if api_lats else None,
                "api_latency_max_ms": round(max(api_lats), 1) if api_lats else None,
                "total_latency_avg_ms": round(sum(total_lats) / len(total_lats), 1) if total_lats else None,
            }
            report["stt_summary"] = stt_summary
            report["stt_results"] = stt_results

            print(f"\n  STT (Google Speech Recognition):")
            print(f"    Phrases: {stt_summary['phrases_successful']}/{stt_summary['phrases_tested']}")
            print(f"    WER:   avg={stt_summary['wer_avg']:.1%}  min={stt_summary['wer_min']:.1%}  max={stt_summary['wer_max']:.1%}")
            print(f"    Accuracy: {stt_summary['accuracy_avg']:.1%}  ({stt_summary['wer_zero_count']} perfect)")
            if stt_summary["api_latency_avg_ms"]:
                print(f"    API Latency: avg={stt_summary['api_latency_avg_ms']:.0f}ms  min={stt_summary['api_latency_min_ms']:.0f}ms  max={stt_summary['api_latency_max_ms']:.0f}ms")

    if tts_results:
        valid = [r for r in tts_results if "generation_ms" in r]
        if valid:
            gens = [r["generation_ms"] for r in valid]
            tots = [r["total_ms"] for r in valid]

            tts_summary = {
                "phrases_tested": len(tts_results),
                "phrases_successful": len(valid),
                "generation_avg_ms": round(sum(gens) / len(gens), 1),
                "generation_min_ms": round(min(gens), 1),
                "generation_max_ms": round(max(gens), 1),
                "total_avg_ms": round(sum(tots) / len(tots), 1),
            }
            report["tts_summary"] = tts_summary
            report["tts_results"] = tts_results

            print(f"\n  TTS (Edge-TTS {EDGE_TTS_VOICE} @ {EDGE_TTS_RATE}):")
            print(f"    Generation: avg={tts_summary['generation_avg_ms']:.0f}ms  min={tts_summary['generation_min_ms']:.0f}ms  max={tts_summary['generation_max_ms']:.0f}ms")
            print(f"    Total (incl playback): avg={tts_summary['total_avg_ms']:.0f}ms")

    # Save
    log_dir = Path(__file__).resolve().parent / "test_logs"
    log_dir.mkdir(exist_ok=True)
    filepath = log_dir / f"edge_google_benchmark_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  💾 Saved → {filepath}")
    print("=" * 60)


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run_stt = "--tts" not in sys.argv
    run_tts = "--stt" not in sys.argv

    stt_results = run_stt_test() if run_stt else []
    tts_results = run_tts_test() if run_tts else []

    print_and_save(stt_results, tts_results)
