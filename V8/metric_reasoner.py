"""
MetricReasoner — 8B LLM drowsiness gate for Driver Drowsiness Detection V7.

Replaces the hardcoded weighted drowsiness score formula with an 8B model
(llama-3.1-8b-instant on Groq) that reasons about whether the driver is
actually drowsy from raw metric snapshots + trend history.

Architecture:
  - Local metrics (PERCLOS, EAR, blinks, pitch, roll) still run every frame
  - A cheap pre-filter (lowered threshold ~0.30) gates API calls
  - When pre-filter trips → MetricReasoner evaluates every ~3 seconds
  - Returns: level (ALERT/MILD/DROWSY/CRITICAL), confidence, reasoning
  - Must confirm DROWSY/CRITICAL N consecutive times before triggering
  - Microsleep stays as instant local alert (too critical for API latency)
  - Reasoning text is passed to the 70B conversation model as context  - Voice metrics (energy, speech rate, pauses, latency) are included when
    available (during conversation) for combined audio-visual reasoning
  - Post-session: all evaluations stored in SQLite; 8B learns driver-specific
    patterns over time (what metric combos mean drowsiness for THIS driver)"""

import time
import json
from collections import deque
from datetime import datetime

from groq import Groq
from config import Config


# ── Structured result from the 8B reasoner ──
class ReasonerResult:
    """Result of a single 8B evaluation."""

    __slots__ = ('level', 'confidence', 'reasoning', 'timestamp')

    def __init__(self, level="ALERT", confidence=0.0, reasoning="", timestamp=None):
        self.level = level          # ALERT | MILD | DROWSY | CRITICAL
        self.confidence = confidence  # 0.0 – 1.0
        self.reasoning = reasoning    # Free-text explanation
        self.timestamp = timestamp or time.time()

    def is_drowsy(self):
        return self.level in ("DROWSY", "CRITICAL")

    def __repr__(self):
        return f"ReasonerResult({self.level}, conf={self.confidence:.2f})"


# ── System prompt for the 8B reasoning model ──
_REASONER_SYSTEM_PROMPT = """You are a drowsiness detection system. Classify the driver as ALERT, MILD, DROWSY, or CRITICAL from raw sensor metrics.

## Metric Guide
- **PERCLOS** (0-1): Fraction of time eyes closed in 10s. <0.10 normal, 0.10-0.20 slightly elevated, >0.20 concerning, >0.40 severe.
- **Blink Rate** (count/10s): Normal 2-4. Very low (0-1) or high (5+) can indicate fatigue.
- **Slow Blinks** (count): Blinks >0.4s. 0=normal, 1-2=early fatigue, 3+=strong drowsiness.
- **EAR Std**: Eye Aspect Ratio variability. Near zero+high PERCLOS=bad. 0.02-0.06=normal blinking. >0.08=erratic.
- **Pitch Variance**: Head nodding. <0.002=stable, >0.010=head bobbing (strong sign).
- **Microsleep** (bool): Eyes closed >1.5s. If True → always CRITICAL.
- **Head Down/Roll** (bool): UNRELIABLE — many false positives from normal driving. IGNORE unless PERCLOS>0.20 AND slow_blinks>=2.

## Voice Metrics (when present)
Compare to personal baseline. Natural variation of ±30% is NORMAL.
- **Energy RMS**: 45%+ drop from baseline = drowsy. 20-45% drop = possible, needs corroboration.
- **Speech Rate**: 40%+ drop = drowsy. 20-40% = possible.
- **Pause Ratio**: 75%+ above baseline = drowsy.
- **Response Latency**: 5s+ above baseline = drowsy.

Voice is SUPPLEMENTARY — strengthens visual evidence but NEVER the primary reason for DROWSY. If eyes/head are fine, driver is alert regardless of voice.

## Rules
1. No single metric is conclusive — look for CONVERGENCE of multiple signals.
2. Microsleep=True → always CRITICAL.
3. PERCLOS and slow_blinks are most reliable. EAR std and pitch_var are supplementary.
4. Trends matter: if metrics are IMPROVING, downgrade assessment. Current metrics > past ones.
5. Every evaluation is INDEPENDENT — don't anchor to previous results.
6. ALERT is the default unless clear evidence says otherwise.
7. Head_down and head_roll alone mean NOTHING.

Respond with ONLY this JSON:
{"level": "ALERT|MILD|DROWSY|CRITICAL", "confidence": 0.0-1.0, "reasoning": "brief explanation"}"""


class MetricReasoner:
    """8B LLM-based drowsiness gate replacing the hardcoded weighted formula.

    Maintains a rolling history of metric snapshots so the 8B model can
    reason about trends (worsening/improving). Called on a throttled basis
    (~every 3s) only when the cheap pre-filter trips.
    """

    def __init__(self):
        self.client = None
        self._history = deque(maxlen=Config.REASONER_HISTORY_SIZE)
        self._last_call_time = 0.0
        self._confirm_count = 0      # Consecutive DROWSY/CRITICAL results
        self._last_result = ReasonerResult()
        self._evaluation_log = []     # All evaluations this session (for post-session storage)
        self._driver_patterns = ""   # Learned patterns injected into system prompt
        self._voice_baselines = None  # Voice baselines for deviation calc
        self._initialize()

    def _initialize(self):
        """Initialize Groq client for 8B reasoning calls."""
        try:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            print("✓ MetricReasoner (8B) ready")
        except Exception as e:
            print(f"⚠️ MetricReasoner init failed: {e}")
            self.client = None

    # ── Public API ──

    def set_driver_patterns(self, patterns_str):
        """Inject learned driver-specific patterns into the reasoner.

        Called at startup after loading from SQLite. The patterns are
        appended to the system prompt so the 8B model knows what metric
        combinations are significant for THIS driver.
        """
        self._driver_patterns = patterns_str or ""
        if patterns_str:
            print(f"✓ MetricReasoner loaded {patterns_str.count(chr(10)) + 1} driver patterns")

    def set_voice_baselines(self, baselines):
        """Set voice baselines for deviation calculation.

        Args:
            baselines: dict from memory_manager.get_baselines()
                       e.g. {'energy_rms': {'avg': 0.05, ...}, ...}
        """
        self._voice_baselines = baselines

    def should_call(self):
        """Check if enough time has passed since the last API call."""
        return (time.time() - self._last_call_time) >= Config.REASONER_INTERVAL_S

    def evaluate(self, metrics, microsleep=False, head_down=False, head_roll=False,
                 voice_features=None):
        """Send metrics to the 8B model for drowsiness reasoning.

        Args:
            metrics: dict from calculate_metrics() — perclos, blink_rate,
                     slow_blinks, ear_std, pitch_var, drowsy_score
            microsleep: bool — eyes closed > 1.5s
            head_down: bool — head tilted down for extended period
            head_roll: bool — head tilted sideways for extended period
            voice_features: dict from VoiceFeatureExtractor.extract_features()
                            or None if not in conversation

        Returns:
            ReasonerResult with level, confidence, reasoning
        """
        now = time.time()
        self._last_call_time = now

        # Store snapshot in history
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

        # Add voice features if available
        if voice_features:
            snapshot['energy_rms'] = voice_features.get('energy_rms')
            snapshot['speech_rate_wpm'] = voice_features.get('speech_rate_wpm')
            snapshot['pause_ratio'] = voice_features.get('pause_ratio')
            snapshot['response_latency_s'] = voice_features.get('response_latency_s')

        self._history.append(snapshot)

        # Build prompt with current + history
        prompt = self._build_prompt(snapshot)

        try:
            result = self._call_8b(prompt)
        except Exception as e:
            print(f"⚠️ MetricReasoner API error: {e}")
            # Fallback: use local pre-filter score as rough guide
            result = self._fallback_result(metrics, microsleep, head_down)

        # Update confirmation counter
        if result.is_drowsy():
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        self._last_result = result

        # Log evaluation for post-session storage
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
        """Format the reasoner's analysis for the 70B conversation model.

        Returns a string like:
        '8B ANALYSIS: DROWSY (conf=0.81) — PERCLOS 0.28 with 3 slow blinks
        and increasing pitch variance. Confirmed 3/3 consecutive times.'
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
        # Don't clear _evaluation_log — it's drained by get_evaluation_log()
        # Don't clear _driver_patterns — persistent across sessions

    def get_evaluation_log(self):
        """Return and clear all logged evaluations for post-session storage."""
        log = self._evaluation_log
        self._evaluation_log = []
        return log

    # ── Internal ──

    def _build_prompt(self, current):
        """Build the user message with current metrics + voice + trend history."""
        parts = []

        # Current visual snapshot
        parts.append("## Current Visual Metrics")
        parts.append(f"- PERCLOS: {current['perclos']:.4f}")
        parts.append(f"- Blink Rate (10s window): {current['blink_rate']}")
        parts.append(f"- Slow Blinks: {current['slow_blinks']}")
        parts.append(f"- EAR Std: {current['ear_std']:.4f}")
        parts.append(f"- Pitch Variance: {current['pitch_var']:.5f}")
        parts.append(f"- Microsleep: {current['microsleep']}")
        parts.append(f"- Head Down: {current['head_down']}")
        parts.append(f"- Head Roll: {current['head_roll']}")

        # Current voice metrics (if available)
        has_voice = current.get('energy_rms') is not None
        if has_voice:
            parts.append("")
            parts.append("## Current Voice Metrics")
            parts.append(f"- Energy RMS: {current['energy_rms']:.4f}")
            if current.get('speech_rate_wpm') is not None:
                parts.append(f"- Speech Rate: {current['speech_rate_wpm']:.1f} wpm")
            if current.get('pause_ratio') is not None:
                parts.append(f"- Pause Ratio: {current['pause_ratio']:.3f}")
            if current.get('response_latency_s') is not None:
                parts.append(f"- Response Latency: {current['response_latency_s']:.1f}s")

            # Baseline deviation (if baselines available)
            bl = self._voice_baselines
            if bl:
                deviations = []
                rms_bl = bl.get('energy_rms')
                if rms_bl and rms_bl['avg'] > 0 and current.get('energy_rms') is not None:
                    ratio = current['energy_rms'] / rms_bl['avg']
                    deviations.append(f"Energy = {ratio:.0%} of baseline ({rms_bl['avg']:.4f})")
                rate_bl = bl.get('speech_rate_wpm')
                if rate_bl and rate_bl['avg'] > 0 and current.get('speech_rate_wpm') is not None:
                    ratio = current['speech_rate_wpm'] / rate_bl['avg']
                    deviations.append(f"Speech rate = {ratio:.0%} of baseline ({rate_bl['avg']:.1f} wpm)")
                pause_bl = bl.get('pause_ratio')
                if pause_bl and current.get('pause_ratio') is not None:
                    diff = current['pause_ratio'] - pause_bl['avg']
                    sign = "+" if diff >= 0 else ""
                    deviations.append(f"Pause ratio {sign}{diff:.3f} vs baseline ({pause_bl['avg']:.3f})")
                lat_bl = bl.get('response_latency_s')
                if lat_bl and current.get('response_latency_s') is not None:
                    diff = current['response_latency_s'] - lat_bl['avg']
                    sign = "+" if diff >= 0 else ""
                    deviations.append(f"Latency {sign}{diff:.1f}s vs baseline ({lat_bl['avg']:.1f}s)")
                if deviations:
                    parts.append("### Voice vs Personal Baseline")
                    for d in deviations:
                        parts.append(f"  - {d}")

        # Trend history (if available) — cap at last 5 to stay within context window
        if len(self._history) > 1:
            parts.append("")
            recent = list(self._history)[-5:]  # Only last 5 snapshots
            parts.append(f"## Recent History ({len(recent)} snapshots, ~{Config.REASONER_INTERVAL_S}s apart)")
            t0 = recent[0]['time']
            for snap in recent:
                age = snap['time'] - t0
                flags = []
                if snap['microsleep']:
                    flags.append("MICROSLEEP")
                if snap['head_down']:
                    flags.append("HEAD_DOWN")
                if snap['head_roll']:
                    flags.append("HEAD_ROLL")
                flag_str = f" [{', '.join(flags)}]" if flags else ""

                # Include voice info in history line if available
                voice_str = ""
                if snap.get('energy_rms') is not None:
                    voice_parts = [f"rms={snap['energy_rms']:.4f}"]
                    if snap.get('speech_rate_wpm') is not None:
                        voice_parts.append(f"rate={snap['speech_rate_wpm']:.0f}wpm")
                    if snap.get('pause_ratio') is not None:
                        voice_parts.append(f"pause={snap['pause_ratio']:.3f}")
                    if voice_parts:
                        voice_str = f" | {', '.join(voice_parts)}"

                parts.append(
                    f"  t+{age:5.1f}s: perclos={snap['perclos']:.3f}, "
                    f"blinks={snap['blink_rate']}, slow={snap['slow_blinks']}, "
                    f"ear_std={snap['ear_std']:.4f}, pitch_var={snap['pitch_var']:.5f}"
                    f"{voice_str}{flag_str}"
                )

            # Compute simple trends
            first_perclos = recent[0]['perclos']
            last_perclos = recent[-1]['perclos']
            if len(recent) >= 3:
                trend_dir = "rising" if last_perclos > first_perclos + 0.03 else (
                    "falling" if last_perclos < first_perclos - 0.03 else "stable"
                )
                parts.append(f"\n  PERCLOS trend: {trend_dir} ({first_perclos:.3f} → {last_perclos:.3f})")

                first_slow = recent[0]['slow_blinks']
                last_slow = recent[-1]['slow_blinks']
                slow_trend = "rising" if last_slow > first_slow + 1 else (
                    "falling" if last_slow < first_slow - 1 else "stable"
                )
                parts.append(f"  Slow blinks trend: {slow_trend} ({first_slow} → {last_slow})")

                # Voice trends (if available in history)
                rms_vals = [s.get('energy_rms') for s in recent if s.get('energy_rms') is not None]
                if len(rms_vals) >= 3:
                    rms_trend = "falling" if rms_vals[-1] < rms_vals[0] * 0.85 else (
                        "rising" if rms_vals[-1] > rms_vals[0] * 1.15 else "stable"
                    )
                    parts.append(f"  Voice energy trend: {rms_trend} ({rms_vals[0]:.4f} → {rms_vals[-1]:.4f})")

                rate_vals = [s.get('speech_rate_wpm') for s in recent if s.get('speech_rate_wpm') is not None]
                if len(rate_vals) >= 3:
                    rate_trend = "falling" if rate_vals[-1] < rate_vals[0] * 0.85 else (
                        "rising" if rate_vals[-1] > rate_vals[0] * 1.15 else "stable"
                    )
                    parts.append(f"  Speech rate trend: {rate_trend} ({rate_vals[0]:.0f} → {rate_vals[-1]:.0f} wpm)")

        # Driver patterns (learned from history)
        if self._driver_patterns:
            parts.append("")
            parts.append("## Known Patterns for This Driver (learned from past sessions)")
            parts.append(self._driver_patterns)
            parts.append("Use these patterns to calibrate your assessment — they describe what drowsiness looks like for THIS specific person.")

        parts.append("")
        parts.append("Analyze these metrics and determine the driver's alertness level.")

        return "\n".join(parts)

    def _call_8b(self, prompt):
        """Call the 8B model and parse the JSON response."""
        response = self.client.chat.completions.create(
            model=Config.GROQ_REASONER_MODEL,
            messages=[
                {"role": "system", "content": _REASONER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,       # Slightly above minimum to avoid stuck assessments
            max_tokens=256,        # Room for JSON + reasoning text
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        return self._parse_response(raw)

    def _parse_response(self, raw):
        """Parse JSON response from the 8B model."""
        try:
            data = json.loads(raw)
            level = data.get("level", "ALERT").upper()
            if level not in ("ALERT", "MILD", "DROWSY", "CRITICAL"):
                level = "ALERT"
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
            reasoning = str(data.get("reasoning", ""))
            return ReasonerResult(level, confidence, reasoning)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"⚠️ MetricReasoner parse error: {e} — raw: {raw[:200]}")
            return ReasonerResult("ALERT", 0.5, f"Parse error: {e}")

    def _fallback_result(self, metrics, microsleep, head_down):
        """Fallback when API call fails — use local heuristics."""
        if microsleep:
            return ReasonerResult("CRITICAL", 0.95,
                                  "Fallback: microsleep detected (API unavailable)")
        score = metrics.get('drowsy_score', 0)
        if score > 0.6:
            return ReasonerResult("DROWSY", 0.7,
                                  f"Fallback: high local score {score:.2f} (API unavailable)")
        if score > 0.3:
            return ReasonerResult("MILD", 0.5,
                                  f"Fallback: moderate local score {score:.2f} (API unavailable)")
        return ReasonerResult("ALERT", 0.8,
                              f"Fallback: low local score {score:.2f} (API unavailable)")
