"""
MetricReasoner — local threshold-based drowsiness gate for V7-Local.

Replaces the Groq 8B API with pure Python thresholds that encode the same
logic as the 8B system prompt. All public methods are identical to V7 so
the rest of the codebase needs no changes.

Signal scoring:
  - Each metric contributes 0–3 signal points
  - Total signals → ALERT / MILD / DROWSY / CRITICAL
  - Trend adjustment (history): worsening +1, improving -1
  - Voice supplementary: 2+ voice deviations AND 1+ visual signal → +1

Decision thresholds:
  signals >= 5 → CRITICAL  (0.92 confidence)
  signals >= 3 → DROWSY    (0.78 confidence)
  signals >= 1 → MILD      (0.55 confidence)
  signals == 0 → ALERT     (0.85 confidence)
"""

import time
from collections import deque
from datetime import datetime

from config import Config


# ── Structured result (identical to V7) ──
class ReasonerResult:
    """Result of a single evaluation."""

    __slots__ = ('level', 'confidence', 'reasoning', 'timestamp')

    def __init__(self, level="ALERT", confidence=0.0, reasoning="", timestamp=None):
        self.level = level            # ALERT | MILD | DROWSY | CRITICAL
        self.confidence = confidence  # 0.0 – 1.0
        self.reasoning = reasoning    # Free-text explanation
        self.timestamp = timestamp or time.time()

    def is_drowsy(self):
        return self.level in ("DROWSY", "CRITICAL")

    def __repr__(self):
        return f"ReasonerResult({self.level}, conf={self.confidence:.2f})"


class MetricReasoner:
    """Local threshold-based drowsiness gate (replaces Groq 8B in V7-Local).

    Maintains a rolling history of metric snapshots for trend analysis.
    Called on a throttled basis (~every 3s) only when the pre-filter trips.
    All public methods are identical to V7's MetricReasoner.
    """

    def __init__(self):
        self._history = deque(maxlen=Config.REASONER_HISTORY_SIZE)
        self._last_call_time = 0.0
        self._confirm_count = 0
        self._last_result = ReasonerResult()
        self._evaluation_log = []
        self._driver_patterns = ""
        self._voice_baselines = None
        self._initialize()

    def _initialize(self):
        print("✓ MetricReasoner (local thresholds) ready")

    # ── Public API (identical to V7) ──

    def set_driver_patterns(self, patterns_str):
        """Inject learned driver-specific patterns (stored for reference)."""
        self._driver_patterns = patterns_str or ""
        if patterns_str:
            print(f"✓ MetricReasoner loaded {patterns_str.count(chr(10)) + 1} driver patterns")

    def set_voice_baselines(self, baselines):
        """Set voice baselines for deviation calculation."""
        self._voice_baselines = baselines

    def should_call(self):
        """Check if enough time has passed since the last evaluation."""
        return (time.time() - self._last_call_time) >= Config.REASONER_INTERVAL_S

    def evaluate(self, metrics, microsleep=False, head_down=False, head_roll=False,
                 voice_features=None):
        """Evaluate driver alertness from metrics using local thresholds.

        Args:
            metrics: dict from calculate_metrics() — perclos, blink_rate,
                     slow_blinks, ear_std, pitch_var, drowsy_score
            microsleep: bool — eyes closed > 1.5s
            head_down: bool — head tilted down for extended period
            head_roll: bool — head tilted sideways for extended period
            voice_features: dict from VoiceFeatureExtractor or None

        Returns:
            ReasonerResult with level, confidence, reasoning
        """
        now = time.time()
        self._last_call_time = now

        snapshot = {
            'time': now,
            'perclos': round(metrics.get('perclos', 0), 4),
            'blink_rate': metrics.get('blink_rate', 0),
            'slow_blinks': metrics.get('slow_blinks', 0),
            'ear_std': round(metrics.get('ear_std', 0), 4),
            'pitch_var': round(metrics.get('pitch_var', 0), 5),
            'microsleep': microsleep,
            'head_down': head_down,
            'head_roll': head_roll,
        }

        if voice_features:
            snapshot['energy_rms'] = voice_features.get('energy_rms')
            snapshot['speech_rate_wpm'] = voice_features.get('speech_rate_wpm')
            snapshot['pause_ratio'] = voice_features.get('pause_ratio')
            snapshot['response_latency_s'] = voice_features.get('response_latency_s')

        self._history.append(snapshot)

        result = self._evaluate_local(snapshot, microsleep, voice_features)

        if result.is_drowsy():
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        self._last_result = result

        self._evaluation_log.append({
            'timestamp': datetime.fromtimestamp(now).isoformat(),
            'level': result.level,
            'confidence': result.confidence,
            'reasoning': result.reasoning,
            'perclos': snapshot.get('perclos'),
            'blink_rate': snapshot.get('blink_rate'),
            'slow_blinks': snapshot.get('slow_blinks'),
            'ear_std': snapshot.get('ear_std'),
            'pitch_var': snapshot.get('pitch_var'),
            'microsleep': snapshot.get('microsleep', False),
            'head_down': snapshot.get('head_down', False),
            'head_roll': snapshot.get('head_roll', False),
            'energy_rms': snapshot.get('energy_rms'),
            'speech_rate_wpm': snapshot.get('speech_rate_wpm'),
            'pause_ratio': snapshot.get('pause_ratio'),
            'response_latency_s': snapshot.get('response_latency_s'),
        })

        return result

    def is_confirmed_drowsy(self):
        """True if we've seen DROWSY/CRITICAL N consecutive times."""
        return self._confirm_count >= Config.REASONER_CONFIRM_COUNT

    def get_confirmation_count(self):
        """Current consecutive drowsy confirmation count."""
        return self._confirm_count

    def get_last_result(self):
        """Last ReasonerResult (for display/logging)."""
        return self._last_result

    def get_reasoning_for_llm(self):
        """Format the reasoner's analysis for the conversation model.

        Returns a string like:
        '8B ANALYSIS: DROWSY (conf=0.78) — PERCLOS 0.28 with 3 slow blinks.
        Confirmed 3/3 consecutive times.'
        """
        r = self._last_result
        if r.level == "ALERT":
            return ""
        return (
            f"8B ANALYSIS: {r.level} (confidence={r.confidence:.2f}) — "
            f"{r.reasoning} "
            f"[confirmed {self._confirm_count}/{Config.REASONER_CONFIRM_COUNT}]"
        )

    def reset(self):
        """Reset confirmation counter (after conversation ends)."""
        self._confirm_count = 0
        self._last_result = ReasonerResult()
        # Don't clear _evaluation_log — drained by get_evaluation_log()
        # Don't clear _driver_patterns — persistent across sessions

    def get_evaluation_log(self):
        """Return and clear all logged evaluations for post-session storage."""
        log = self._evaluation_log
        self._evaluation_log = []
        return log

    # ── Internal ──

    def _evaluate_local(self, snapshot, microsleep, voice_features):
        """Score metrics using threshold rules and return a ReasonerResult."""

        # Instant CRITICAL: microsleep (eyes closed > 1.5s)
        if microsleep:
            return ReasonerResult(
                "CRITICAL", 0.97,
                "Microsleep detected — eyes closed continuously"
            )

        perclos = snapshot['perclos']
        slow_blinks = snapshot['slow_blinks']
        ear_std = snapshot['ear_std']
        pitch_var = snapshot['pitch_var']

        signals = 0
        reasons = []

        # ── PERCLOS (most reliable visual signal) ──
        if perclos > 0.40:
            signals += 3
            reasons.append(f"PERCLOS {perclos:.3f} (severe)")
        elif perclos > 0.20:
            signals += 2
            reasons.append(f"PERCLOS {perclos:.3f} (elevated)")
        elif perclos > 0.10:
            signals += 1
            reasons.append(f"PERCLOS {perclos:.3f} (slightly elevated)")

        # ── Slow blinks ──
        if slow_blinks >= 3:
            signals += 2
            reasons.append(f"{slow_blinks} slow blinks")
        elif slow_blinks >= 1:
            signals += 1
            reasons.append(f"{slow_blinks} slow blink(s)")

        # ── Pitch variance (head nodding) ──
        if pitch_var > 0.010:
            signals += 1
            reasons.append(f"pitch variance {pitch_var:.4f} (head nodding)")

        # ── EAR std + PERCLOS combo (eyes stuck droopy) ──
        if ear_std < 0.010 and perclos > 0.15:
            signals += 1
            reasons.append("low EAR variability with elevated PERCLOS")

        # ── Voice supplementary (needs 2+ voice deviations AND 1+ visual) ──
        voice_signals = self._count_voice_signals(voice_features)
        if voice_signals >= 2 and signals >= 1:
            signals += 1
            reasons.append(f"{voice_signals} voice deviation(s)")

        # ── Trend adjustment from rolling history ──
        trend_adj = self._trend_adjustment()
        if trend_adj > 0:
            signals += 1
            reasons.append("worsening trend")
        elif trend_adj < 0 and signals > 0:
            signals -= 1
            reasons.append("improving trend")

        # ── Map signal count to level ──
        if signals >= 5:
            level = "CRITICAL"
            confidence = 0.92
        elif signals >= 3:
            level = "DROWSY"
            confidence = 0.78
        elif signals >= 1:
            level = "MILD"
            confidence = 0.55
        else:
            level = "ALERT"
            confidence = 0.85
            reasons.append("metrics within normal range")

        reasoning = "; ".join(reasons) if reasons else "all metrics normal"
        return ReasonerResult(level, confidence, reasoning)

    def _count_voice_signals(self, voice_features):
        """Count voice metrics that show significant drowsiness deviation."""
        if not voice_features:
            return 0

        bl = self._voice_baselines
        count = 0

        energy = voice_features.get('energy_rms')
        if energy is not None and bl:
            rms_bl = bl.get('energy_rms')
            if rms_bl and rms_bl.get('avg', 0) > 0:
                if energy < rms_bl['avg'] * 0.55:   # 45%+ drop from baseline
                    count += 1

        rate = voice_features.get('speech_rate_wpm')
        if rate is not None and bl:
            rate_bl = bl.get('speech_rate_wpm')
            if rate_bl and rate_bl.get('avg', 0) > 0:
                if rate < rate_bl['avg'] * 0.60:    # 40%+ drop from baseline
                    count += 1

        pause = voice_features.get('pause_ratio')
        if pause is not None and bl:
            pause_bl = bl.get('pause_ratio')
            if pause_bl and pause_bl.get('avg') is not None:
                if pause > pause_bl['avg'] * 1.75:  # 75%+ above baseline
                    count += 1

        latency = voice_features.get('response_latency_s')
        if latency is not None and bl:
            lat_bl = bl.get('response_latency_s')
            if lat_bl and lat_bl.get('avg') is not None:
                if latency > lat_bl['avg'] + 5.0:   # 5s+ above baseline
                    count += 1

        return count

    def _trend_adjustment(self):
        """Return +1 if worsening, -1 if improving, 0 if stable/insufficient."""
        if len(self._history) < 3:
            return 0

        recent = list(self._history)[-5:]
        first_perclos = recent[0]['perclos']
        last_perclos = recent[-1]['perclos']
        first_slow = recent[0]['slow_blinks']
        last_slow = recent[-1]['slow_blinks']

        worsening = (last_perclos > first_perclos + 0.05) or (last_slow > first_slow + 1)
        improving = (last_perclos < first_perclos - 0.05) and (last_slow <= first_slow)

        if worsening:
            return 1
        if improving:
            return -1
        return 0

    # ── Kept for API compatibility (used by nothing in local mode) ──

    def _build_prompt(self, current):
        """Not used in local mode — kept for interface compatibility."""
        return ""
