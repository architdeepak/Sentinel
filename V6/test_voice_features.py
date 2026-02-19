#!/usr/bin/env python3
"""
Voice Feature Extraction Test
===============================
Tests the VoiceFeatureExtractor in isolation.

For each round:
  1. Listens to your mic via SpeechRecognition
  2. Transcribes with Deepgram (so you see what it heard)
  3. Runs VoiceFeatureExtractor on the raw audio
  4. Prints ALL raw metrics + the LLM-formatted assessment
  5. Shows thresholds so you can see exactly why it flags you

This helps diagnose why the system always says "mildly drowsy."

Usage:
    python test_voice_features.py            # 5 rounds (default)
    python test_voice_features.py 10         # 10 rounds
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import speech_recognition as sr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from voice_features import VoiceFeatureExtractor


def transcribe_deepgram(audio):
    """Transcribe audio via Deepgram (same as stt_engine)."""
    wav_data = audio.get_wav_data(convert_rate=16000, convert_width=2)
    url = (
        f"https://api.deepgram.com/v1/listen"
        f"?model={Config.DEEPGRAM_STT_MODEL}"
        f"&smart_format=true&punctuate=true"
    )
    headers = {
        "Authorization": f"Token {Config.DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }
    resp = requests.post(url, headers=headers, data=wav_data, timeout=15)
    resp.raise_for_status()
    result = resp.json()
    transcript = (
        result.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
    ).strip()
    return transcript if transcript else None


def print_features(features, llm_text):
    """Print features with threshold comparison."""
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              RAW VOICE METRICS                      │")
    print("  ├─────────────────────┬───────────┬───────────────────┤")
    print("  │ Metric              │ Value     │ Threshold / Note  │")
    print("  ├─────────────────────┼───────────┼───────────────────┤")

    rms = features.get('energy_rms', 0)
    rms_flag = "⚠️ VERY QUIET" if rms < 0.015 else "⚠️ quiet" if rms < 0.03 else "✓ normal"
    print(f"  │ Energy (RMS)        │ {rms:<9.4f} │ <0.015=very quiet │ {rms_flag}")

    peak = features.get('peak_amplitude', 0)
    print(f"  │ Peak Amplitude      │ {peak:<9.4f} │                   │")

    dur = features.get('duration_s', 0)
    print(f"  │ Duration            │ {dur:<7.2f}s  │                   │")

    rate = features.get('speech_rate_wpm')
    if rate is not None:
        rate_flag = "⚠️ VERY SLOW" if rate < 80 else "⚠️ slow" if rate < 110 else "✓ normal"
        print(f"  │ Speech Rate         │ {rate:<7.1f}wpm│ <80=drowsy        │ {rate_flag}")
    else:
        print(f"  │ Speech Rate         │ {'N/A':<9s} │ <80=drowsy        │")

    wc = features.get('word_count', 0)
    print(f"  │ Word Count          │ {wc:<9d} │                   │")

    latency = features.get('response_latency_s')
    if latency is not None:
        lat_flag = "⚠️ VERY SLOW" if latency > 8.0 else "⚠️ slow" if latency > 5.0 else "✓ normal"
        print(f"  │ Response Latency    │ {latency:<7.2f}s  │ >8s=very drowsy   │ {lat_flag}")
    else:
        print(f"  │ Response Latency    │ {'N/A':<9s} │ >8s=very drowsy   │ (no prompt ref)")

    pause = features.get('pause_ratio', 0)
    pause_flag = "⚠️ MANY PAUSES" if pause > 0.5 else "⚠️ some pauses" if pause > 0.35 else "✓ normal"
    print(f"  │ Pause Ratio         │ {pause:<9.3f} │ >0.5=fragmented   │ {pause_flag}")

    print("  └─────────────────────┴───────────┴───────────────────┘")

    # Count flags
    flags = 0
    if rms < 0.03:
        flags += 1
    if rate is not None and rate < 110:
        flags += 1
    if latency is not None and latency > 5.0:
        flags += 1
    if pause > 0.35:
        flags += 1

    print()
    print(f"  🏷️  Drowsy indicators triggered: {flags}/4")
    print()
    print(f"  💬 LLM would see: \"{llm_text}\"")
    print()


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print("\n" + "=" * 60)
    print("🎙️  VOICE FEATURE EXTRACTION TEST")
    print("=" * 60)
    print(f"   Rounds: {rounds}")
    print(f"   This test captures your mic, transcribes it, and shows")
    print(f"   exactly what the voice analyzer detects + why.")
    print()
    print("   TIP: Try speaking normally first, then try whispering,")
    print("   speaking slowly, or pausing a lot to see how it reacts.")
    print("=" * 60)

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.5
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 1.5

    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print("✓ Microphone calibrated\n")

    extractor = VoiceFeatureExtractor()

    all_results = []

    for i in range(rounds):
        print(f"━━━ Round {i + 1}/{rounds} ━━━")
        print("   Say anything (or try different speaking styles)...")

        # Simulate prompt end so response latency is measured
        extractor.mark_prompt_end()

        print("   🎤 Listening...")
        t_start = time.perf_counter()

        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=15, phrase_time_limit=15)

            listen_ms = (time.perf_counter() - t_start) * 1000
            print(f"   ✓ Captured audio ({listen_ms:.0f}ms)")

            # Transcribe
            print("   🔄 Transcribing (Deepgram)...")
            transcript = transcribe_deepgram(audio)
            if transcript:
                print(f"   📝 You said: \"{transcript}\"")
            else:
                print("   ❓ Could not transcribe (empty result)")
                transcript = ""

            # Extract features
            features = extractor.extract_features(audio, transcript)
            if features:
                llm_text = extractor.format_for_llm(features)
                print_features(features, llm_text)

                all_results.append({
                    "round": i + 1,
                    "transcript": transcript,
                    "features": features,
                    "llm_assessment": llm_text,
                })
            else:
                print("   ⚠️ Feature extraction failed")

        except sr.WaitTimeoutError:
            print("   ⏱️ Timeout — no speech detected")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

        print()

    # Summary
    if all_results:
        print("=" * 60)
        print("📊 SUMMARY ACROSS ALL ROUNDS")
        print("=" * 60)

        rms_vals = [r["features"]["energy_rms"] for r in all_results]
        rate_vals = [r["features"]["speech_rate_wpm"] for r in all_results if r["features"].get("speech_rate_wpm")]
        pause_vals = [r["features"]["pause_ratio"] for r in all_results]
        latency_vals = [r["features"]["response_latency_s"] for r in all_results if r["features"].get("response_latency_s") is not None]

        def _stats(name, vals, thresh_low=None, thresh_label=""):
            if not vals:
                print(f"  {name:>20s}: no data")
                return
            avg = sum(vals) / len(vals)
            below = sum(1 for v in vals if thresh_low is not None and v < thresh_low)
            flag = f"  ({below}/{len(vals)} below {thresh_low} {thresh_label})" if thresh_low else ""
            print(f"  {name:>20s}: avg={avg:.4f}  min={min(vals):.4f}  max={max(vals):.4f}{flag}")

        _stats("Energy RMS", rms_vals, 0.03, "= quiet")
        _stats("Speech Rate (wpm)", rate_vals, 110, "= slow")
        _stats("Pause Ratio", pause_vals)
        _stats("Response Latency (s)", latency_vals, 5.0, "= slow")

        # Verdict
        avg_rms = sum(rms_vals) / len(rms_vals)
        avg_pause = sum(pause_vals) / len(pause_vals)

        print()
        if avg_rms < 0.015:
            print("  🔴 Your average RMS is very low — the mic gain might be too low,")
            print("     or it's picking up your voice very quietly. This alone would")
            print("     make the system think you're drowsy even if you're not.")
            print("     → Try speaking louder, or check your mic input level in OS settings.")
        elif avg_rms < 0.03:
            print("  🟡 Your average RMS is below the 'quiet' threshold (0.03).")
            print("     This triggers the 'somewhat quiet' flag every time.")
            print("     → The threshold in voice_features.py may need raising,")
            print("       or your mic gain is low.")
        else:
            print("  🟢 Energy levels look normal — not triggering quiet flags.")

        if avg_pause > 0.35:
            print(f"  🟡 Average pause ratio is {avg_pause:.3f} (>{0.35}).")
            print("     This could be the silence threshold (0.015 RMS) being")
            print("     too aggressive — quiet speech gets counted as 'pause'.")

        print()

        # Save
        log_dir = Path(__file__).resolve().parent / "test_logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = log_dir / f"voice_features_test_{timestamp}.json"
        with open(filepath, "w") as f:
            json.dump({"results": all_results}, f, indent=2)
        print(f"  💾 Saved → {filepath}")
        print("=" * 60)


if __name__ == "__main__":
    main()
