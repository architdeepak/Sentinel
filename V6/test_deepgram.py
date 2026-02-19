#!/usr/bin/env python3
"""
Deepgram STT + TTS Benchmark Test
==================================
Tests Deepgram Nova-3 STT accuracy and Deepgram Aura TTS latency.

STT test: Displays reference phrases → you read them → Deepgram transcribes
          → computes Word Error Rate (WER) and latency.
TTS test: Sends phrases to Deepgram TTS → measures time-to-first-byte,
          total generation time, and plays audio.

Results saved to V6/test_logs/deepgram_benchmark_<timestamp>.json

Usage:
    python test_deepgram.py            # Run full benchmark (STT + TTS)
    python test_deepgram.py --stt      # STT only
    python test_deepgram.py --tts      # TTS only
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

import requests
import speech_recognition as sr

# Ensure V6 modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config

# ── Test phrases (diverse difficulty) ──────────────────────────
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

    # DP matrix
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

    # Backtrace for S/I/D counts
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


# ── STT Test ───────────────────────────────────────────────────

def run_stt_test(phrases=None):
    """Run Deepgram STT accuracy + latency test."""
    phrases = phrases or TEST_PHRASES

    print("\n" + "=" * 60)
    print("🎤 DEEPGRAM STT BENCHMARK (Nova-3)")
    print("=" * 60)
    print(f"   Phrases: {len(phrases)}")
    print(f"   Model: {Config.DEEPGRAM_STT_MODEL}")
    print(f"   API key: {'set' if Config.DEEPGRAM_API_KEY else 'MISSING'}")
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

            # Send to Deepgram
            wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)

            url = (
                f"https://api.deepgram.com/v1/listen"
                f"?model={Config.DEEPGRAM_STT_MODEL}"
                f"&smart_format=true"
                f"&punctuate=true"
            )
            headers = {
                "Authorization": f"Token {Config.DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            }

            t_api = time.perf_counter()
            resp = requests.post(url, headers=headers, data=wav_data, timeout=15)
            resp.raise_for_status()
            api_ms = (time.perf_counter() - t_api) * 1000

            result = resp.json()
            transcript = (
                result.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            ).strip()

            total_ms = (time.perf_counter() - t_start) * 1000
            confidence = (
                result.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("confidence", 0)
            )

            wer_result = compute_wer(phrase, transcript) if transcript else {
                "wer": 1.0, "errors": len(phrase.split()),
                "substitutions": 0, "insertions": 0,
                "deletions": len(phrase.split()),
                "ref_word_count": len(phrase.split()),
                "hyp_word_count": 0,
            }

            print(f"   ✓ Got: \"{transcript}\"")
            print(f"   📊 WER: {wer_result['wer']:.1%}  |  API: {api_ms:.0f}ms  |  Total: {total_ms:.0f}ms  |  Conf: {confidence:.3f}")

            results.append({
                "phrase_index": i + 1,
                "reference": phrase,
                "hypothesis": transcript,
                "wer": wer_result["wer"],
                "wer_details": wer_result,
                "confidence": confidence,
                "api_latency_ms": round(api_ms, 1),
                "total_latency_ms": round(total_ms, 1),
                "audio_size_bytes": len(wav_data),
            })

        except sr.WaitTimeoutError:
            print("   ⏱️ Timeout — no speech detected")
            results.append({"phrase_index": i + 1, "reference": phrase, "error": "timeout"})
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            results.append({"phrase_index": i + 1, "reference": phrase, "error": str(e)})

        print()

    return results


# ── TTS Test ───────────────────────────────────────────────────

def run_tts_test(phrases=None):
    """Run Deepgram TTS latency test."""
    phrases = phrases or TEST_PHRASES

    print("\n" + "=" * 60)
    print("🔊 DEEPGRAM TTS BENCHMARK (Aura-2)")
    print("=" * 60)
    print(f"   Phrases: {len(phrases)}")
    print(f"   Voice: {Config.DEEPGRAM_TTS_VOICE}")
    print(f"   Speed: {Config.DEEPGRAM_TTS_SPEED}")
    print()

    results = []

    for i, phrase in enumerate(phrases):
        print(f"─── Phrase {i + 1}/{len(phrases)} ───")
        print(f"   \"{phrase}\"")

        try:
            url = (
                f"https://api.deepgram.com/v1/speak"
                f"?model={Config.DEEPGRAM_TTS_VOICE}"
                f"&encoding=mp3"
                f"&speed={Config.DEEPGRAM_TTS_SPEED}"
            )
            headers = {
                "Authorization": f"Token {Config.DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {"text": phrase}

            t_start = time.perf_counter()
            resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=15)
            resp.raise_for_status()

            # Stream to player, measure first-byte and total
            first_byte_ms = None
            total_bytes = 0

            # Start player
            proc = None
            for cmd in [["mpg123", "-q", "-"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-"]]:
                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    break
                except FileNotFoundError:
                    continue

            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_ms is None:
                        first_byte_ms = (time.perf_counter() - t_start) * 1000
                    total_bytes += len(chunk)
                    if proc:
                        proc.stdin.write(chunk)

            generation_ms = (time.perf_counter() - t_start) * 1000

            if proc:
                proc.stdin.close()
                proc.wait()

            total_ms = (time.perf_counter() - t_start) * 1000

            print(f"   📊 First byte: {first_byte_ms:.0f}ms  |  Gen: {generation_ms:.0f}ms  |  Total: {total_ms:.0f}ms  |  Size: {total_bytes / 1024:.1f}KB")

            results.append({
                "phrase_index": i + 1,
                "text": phrase,
                "char_count": len(phrase),
                "word_count": len(phrase.split()),
                "first_byte_ms": round(first_byte_ms, 1) if first_byte_ms else None,
                "generation_ms": round(generation_ms, 1),
                "total_ms": round(total_ms, 1),
                "audio_bytes": total_bytes,
            })

        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            results.append({"phrase_index": i + 1, "text": phrase, "error": str(e)})

        print()

    return results


# ── Summary + Save ─────────────────────────────────────────────

def print_and_save(stt_results, tts_results):
    """Print summary and save to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 60)
    print("📊 DEEPGRAM BENCHMARK RESULTS")
    print("=" * 60)

    report = {
        "engine": "deepgram",
        "timestamp": datetime.now().isoformat(),
        "stt_model": Config.DEEPGRAM_STT_MODEL,
        "tts_voice": Config.DEEPGRAM_TTS_VOICE,
        "tts_speed": Config.DEEPGRAM_TTS_SPEED,
    }

    if stt_results:
        valid = [r for r in stt_results if "wer" in r]
        if valid:
            wers = [r["wer"] for r in valid]
            api_lats = [r["api_latency_ms"] for r in valid]
            total_lats = [r["total_latency_ms"] for r in valid]
            confs = [r["confidence"] for r in valid]

            stt_summary = {
                "phrases_tested": len(stt_results),
                "phrases_successful": len(valid),
                "wer_avg": round(sum(wers) / len(wers), 4),
                "wer_min": round(min(wers), 4),
                "wer_max": round(max(wers), 4),
                "wer_zero_count": sum(1 for w in wers if w == 0),
                "accuracy_avg": round(1 - sum(wers) / len(wers), 4),
                "confidence_avg": round(sum(confs) / len(confs), 4),
                "api_latency_avg_ms": round(sum(api_lats) / len(api_lats), 1),
                "api_latency_min_ms": round(min(api_lats), 1),
                "api_latency_max_ms": round(max(api_lats), 1),
                "total_latency_avg_ms": round(sum(total_lats) / len(total_lats), 1),
            }
            report["stt_summary"] = stt_summary
            report["stt_results"] = stt_results

            print(f"\n  STT (Deepgram Nova-3):")
            print(f"    Phrases: {stt_summary['phrases_successful']}/{stt_summary['phrases_tested']}")
            print(f"    WER:   avg={stt_summary['wer_avg']:.1%}  min={stt_summary['wer_min']:.1%}  max={stt_summary['wer_max']:.1%}")
            print(f"    Accuracy: {stt_summary['accuracy_avg']:.1%}  ({stt_summary['wer_zero_count']} perfect)")
            print(f"    Confidence: {stt_summary['confidence_avg']:.3f}")
            print(f"    API Latency: avg={stt_summary['api_latency_avg_ms']:.0f}ms  min={stt_summary['api_latency_min_ms']:.0f}ms  max={stt_summary['api_latency_max_ms']:.0f}ms")

    if tts_results:
        valid = [r for r in tts_results if "first_byte_ms" in r and r.get("first_byte_ms")]
        if valid:
            fbs = [r["first_byte_ms"] for r in valid]
            gens = [r["generation_ms"] for r in valid]
            tots = [r["total_ms"] for r in valid]

            tts_summary = {
                "phrases_tested": len(tts_results),
                "phrases_successful": len(valid),
                "first_byte_avg_ms": round(sum(fbs) / len(fbs), 1),
                "first_byte_min_ms": round(min(fbs), 1),
                "first_byte_max_ms": round(max(fbs), 1),
                "generation_avg_ms": round(sum(gens) / len(gens), 1),
                "total_avg_ms": round(sum(tots) / len(tots), 1),
            }
            report["tts_summary"] = tts_summary
            report["tts_results"] = tts_results

            print(f"\n  TTS (Deepgram Aura-2 @ {Config.DEEPGRAM_TTS_SPEED}x):")
            print(f"    First byte: avg={tts_summary['first_byte_avg_ms']:.0f}ms  min={tts_summary['first_byte_min_ms']:.0f}ms  max={tts_summary['first_byte_max_ms']:.0f}ms")
            print(f"    Generation: avg={tts_summary['generation_avg_ms']:.0f}ms")
            print(f"    Total (incl playback): avg={tts_summary['total_avg_ms']:.0f}ms")

    # Save
    log_dir = Path(__file__).resolve().parent / "test_logs"
    log_dir.mkdir(exist_ok=True)
    filepath = log_dir / f"deepgram_benchmark_{timestamp}.json"
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
