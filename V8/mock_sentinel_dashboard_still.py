#!/usr/bin/env python3
"""Static Windows demo of the Sentinel dashboard popup.

Creates a still-frame CV2 window that matches the V8 "Sentinel Dashboard"
look, using realistic mock voice + detection metrics.
"""

import cv2

from dashboard import DashboardRenderer


class FakeReasonerResult:
    def __init__(self, level, confidence, reasoning):
        self.level = level
        self.confidence = confidence
        self.reasoning = reasoning


def clamp01(x):
    return max(0.0, min(1.0, x))


def calc_voice_score(features, baselines):
    energy_base = baselines["energy_rms"]["avg"]
    rate_base = baselines["speech_rate_wpm"]["avg"]
    pause_base = baselines["pause_ratio"]["avg"]
    lat_base = baselines["response_latency_s"]["avg"]

    energy_drop = clamp01((energy_base - features["energy_rms"]) / energy_base)
    rate_drop = clamp01((rate_base - features["speech_rate_wpm"]) / rate_base)
    pause_rise = clamp01((features["pause_ratio"] - pause_base) / pause_base)
    latency_rise = clamp01((features["response_latency_s"] - lat_base) / 4.0)

    return round(
        0.35 * energy_drop
        + 0.25 * rate_drop
        + 0.20 * pause_rise
        + 0.20 * latency_rise,
        3,
    )


def main():
    baselines = {
        "energy_rms": {"avg": 0.041},
        "speech_rate_wpm": {"avg": 142.0},
        "pause_ratio": {"avg": 0.19},
        "response_latency_s": {"avg": 1.4},
    }

    # Previous + current so trend arrows are visible.
    prev_voice = {
        "energy_rms": 0.029,
        "speech_rate_wpm": 112.0,
        "pause_ratio": 0.31,
        "response_latency_s": 2.8,
        "word_count": 18,
        "duration_s": 9.2,
    }
    cur_voice = {
        "energy_rms": 0.026,
        "speech_rate_wpm": 104.0,
        "pause_ratio": 0.34,
        "response_latency_s": 3.2,
        "word_count": 21,
        "duration_s": 11.4,
    }

    voice_score = calc_voice_score(cur_voice, baselines)

    det = {
        "drowsy_score": 0.62,
        "perclos": 0.24,
        "blink_rate": 13,
        "slow_blinks": 4,
        "ear_std": 0.0832,
        "pitch_var": 0.00481,
        "microsleep": False,
        "head_down": True,
        "head_roll": False,
        "alert_duration": 0,
    }

    renderer = DashboardRenderer(baselines=baselines)
    renderer.update_voice(prev_voice)
    renderer.update_voice(cur_voice)

    rr = FakeReasonerResult(
        level="DROWSY",
        confidence=0.86,
        reasoning=(
            f"Visual metrics are elevated and voice score is {voice_score:.3f}. "
            "Energy and speech rate are below baseline while pause ratio and latency are higher. "
            "Pattern is consistent with moderate-to-high drowsiness."
        ),
    )
    renderer.update_reasoner(rr)

    img = renderer.render(det)

    cv2.putText(
        img,
        f"Voice Drowsiness Score: {voice_score:.3f}",
        (10, 520),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 255),
        1,
    )

    cv2.imshow("Sentinel Dashboard (Static Mock)", img)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()