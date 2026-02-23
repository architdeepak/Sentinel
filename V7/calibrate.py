#!/usr/bin/env python3
"""
Voice Baseline Calibration for Driver Drowsiness Detection System V7.

Standalone script to record the driver's baseline voice metrics.
Run this before first use (or any time to recalibrate).

Records several sentences in the driver's normal speaking voice,
extracts voice features, and stores the averages in the SQLite
baselines table. These baselines are used during conversation so
the LLM can reason about deviation from normal ("their RMS dropped
30% from baseline") instead of relying on absolute thresholds.

Usage:
    python calibrate.py           # Standard calibration (5 sentences)
    python calibrate.py 8         # Custom number of sentences
    python calibrate.py --reset   # Clear existing baselines and recalibrate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config
from memory import MemoryManager
from stt_engine import STTEngine
from tts_engine import TTSEngine
from voice_features import VoiceFeatureExtractor


PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "I'm driving to my destination and everything looks clear ahead.",
    "Today has been a really productive day at work, I got a lot done.",
    "My favorite thing to do on weekends is relax with friends and family.",
    "The weather forecast says it will be sunny and warm tomorrow afternoon.",
    "I just picked up some groceries and I'm heading home for dinner now.",
    "There's a great restaurant downtown that serves amazing Italian food.",
    "The highway was pretty empty so I made really good time getting here.",
    "I need to remember to call my friend about our plans this weekend.",
    "The sunset over the mountains was absolutely beautiful this evening.",
]


def run_calibration(num_sentences=5, reset=False):
    memory = MemoryManager()

    if reset:
        print("🔄 Clearing existing baselines...")
        with memory._connect() as conn:
            conn.execute("DELETE FROM baselines")
        print("✓ Baselines cleared\n")

    # Show current baselines if they exist
    existing = memory.get_baselines()
    if existing and not reset:
        print("📊 Current baselines:")
        for name, stats in existing.items():
            print(f"   {name}: avg={stats['avg']:.4f} "
                  f"(range {stats['min']:.4f}–{stats['max']:.4f}, "
                  f"n={stats['sample_count']})")
        print()

    print("=" * 60)
    print("🎙️  VOICE BASELINE CALIBRATION")
    print("=" * 60)
    print(f"   Recording {num_sentences} sentences in your normal voice.")
    print("   Speak at your usual volume, pace, and tone — as if you")
    print("   were having a normal conversation while driving.")
    print()
    print("   TIP: Sit in the same position you'd be in while driving.")
    print("=" * 60)

    tts = TTSEngine()
    stt = STTEngine()
    extractor = VoiceFeatureExtractor()

    # Spoken intro
    tts.speak(
        "Welcome to voice calibration. "
        "I'll read each phrase out loud — just repeat it back to me "
        "in your normal, natural speaking voice. "
        "Ready? Let's begin."
    )
    tts.wait_until_done()

    samples = []

    for i in range(num_sentences):
        prompt = PROMPTS[i % len(PROMPTS)]
        print(f"\n  [{i+1}/{num_sentences}] Phrase:")
        print(f'  >>> "{prompt}"')

        # Speak the phrase so the driver knows exactly what to say
        tts.speak(prompt)
        tts.wait_until_done()

        print("  🎤 Your turn — repeat that phrase now...")
        extractor.mark_prompt_end()
        text, audio = stt.listen(timeout=15, show_diagnostics=False)

        if audio is not None:
            features = extractor.extract_features(audio, text)
            if features:
                samples.append(features)
                print(f"  ✓ Got it!")
                print(f"     RMS: {features['energy_rms']:.4f}  |  "
                      f"Rate: {features.get('speech_rate_wpm', 'N/A')} wpm  |  "
                      f"Pauses: {features['pause_ratio']:.3f}  |  "
                      f"Peak: {features['peak_amplitude']:.4f}")
                if text:
                    print(f"     Heard: \"{text}\"")
            else:
                print("  ⚠️ Couldn't extract features — skipping this one")
                tts.speak("Didn't catch that one, moving on.")
                tts.wait_until_done()
        else:
            print("  ⚠️ No audio captured — skipping")
            tts.speak("No audio detected, moving on.")
            tts.wait_until_done()

        if i < num_sentences - 1:
            print()

    # Store results
    print("\n" + "=" * 60)

    if len(samples) < 2:
        print("⚠️ Not enough valid samples (got {}, need at least 2)".format(len(samples)))
        print("   Try again with: python calibrate.py")
        tts.speak("Not enough samples recorded. Please run calibration again.")
        tts.wait_until_done()
        tts.shutdown()
        stt.cleanup()
        memory.close()
        return

    memory.store_calibration_baselines(samples)

    # Show final baselines
    baselines = memory.get_baselines()
    print("\n📊 Your personal voice baselines:")
    print("   ─────────────────────────────────────")
    for name, stats in baselines.items():
        label = name.replace("_", " ").title()
        print(f"   {label:>22s}: avg={stats['avg']:.4f}  "
              f"(range {stats['min']:.4f} – {stats['max']:.4f})")
    print("   ─────────────────────────────────────")
    print(f"\n✓ Calibration complete ({len(samples)} samples)")
    print("  These baselines will be used to detect deviation from your")
    print("  normal voice patterns during drowsiness conversations.")
    print("=" * 60)

    tts.speak(
        f"Calibration complete. I recorded {len(samples)} samples of your voice. "
        "I'll use these to recognize when you sound different from your normal self. "
        "You're all set."
    )
    tts.wait_until_done()
    tts.shutdown()
    stt.cleanup()
    memory.close()


if __name__ == "__main__":
    n = Config.CALIBRATION_SENTENCES
    reset = False

    for arg in sys.argv[1:]:
        if arg == "--reset":
            reset = True
        elif arg.isdigit():
            n = int(arg)
        else:
            print(f"Usage: python calibrate.py [num_sentences] [--reset]")
            sys.exit(1)

    run_calibration(num_sentences=n, reset=reset)
