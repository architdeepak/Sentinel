"""
Live Metrics Dashboard Renderer for V7 Conversation Mode.

Renders a metrics panel image showing:
  - Detection metrics: drowsy score, perclos, blinks, microsleep, head_down, etc.
  - Voice metrics (updated per turn): energy, speech rate, pause ratio, latency
  - Visual bar indicators for key metrics
  - Trend arrows (improving / worsening)

NOT threaded — returns an image to be shown by the caller (DetectionThread).
This avoids OpenCV threading issues on Windows/RPi where cv2.imshow must be
called from a single thread.
"""

import cv2
import threading
import numpy as np


# ── Layout constants ──
_W, _H = 480, 550
_BG = (20, 20, 20)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (220, 220, 220)
_GRAY = (100, 100, 100)
_GREEN = (0, 200, 0)
_YELLOW = (0, 200, 255)
_RED = (0, 0, 220)
_BLUE = (220, 150, 0)
_CYAN = (200, 200, 0)


def _score_color(score):
    """Return color based on drowsy score severity."""
    if score < 0.35:
        return _GREEN
    if score < 0.55:
        return _YELLOW
    return _RED


def _bar(img, x, y, w, h, value, max_val=1.0, color=_GREEN):
    """Draw a horizontal bar with background."""
    cv2.rectangle(img, (x, y), (x + w, y + h), _GRAY, 1)
    fill_w = int((min(value, max_val) / max_val) * (w - 2))
    if fill_w > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill_w, y + h - 1), color, -1)


def _trend_arrow(current, previous):
    """Return trend indicator string."""
    if previous is None:
        return ""
    diff = current - previous
    if abs(diff) < 0.01:
        return " ="
    return " ^" if diff > 0 else " v"


class DashboardRenderer:
    """Stateful renderer — tracks previous values for trend arrows.

    Call update_voice() from the main thread when new voice data arrives.
    Call render(detection_metrics) from the display thread to get an image.
    """

    def __init__(self, baselines=None):
        self._baselines = baselines or {}

        # Voice metrics (updated from main thread, read from render thread)
        self._voice_lock = threading.Lock()
        self._voice = {}
        self._prev_voice = {}

        # 8B Reasoner state (updated from conversation thread)
        self._reasoner_lock = threading.Lock()
        self._reasoner_level = "ALERT"
        self._reasoner_conf = 0.0
        self._reasoner_reasoning = ""

        # Previous detection for trend
        self._prev_det = {}

        # Turn counter
        self._turn = 0

    def update_voice(self, features):
        """Thread-safe: called from main thread after each voice extraction."""
        with self._voice_lock:
            self._prev_voice = self._voice.copy()
            self._voice = features.copy() if features else {}
            self._turn += 1

    def update_reasoner(self, result):
        """Thread-safe: called after 8B reasoner evaluates.

        Args:
            result: ReasonerResult with .level, .confidence, .reasoning
        """
        with self._reasoner_lock:
            self._reasoner_level = result.level
            self._reasoner_conf = result.confidence
            self._reasoner_reasoning = result.reasoning

    def render(self, det):
        """Render dashboard image from detection metrics dict.

        Args:
            det: dict from DetectionThread.get_full_state()

        Returns:
            numpy image (H x W x 3, uint8) ready for cv2.imshow
        """
        img = np.full((_H, _W, 3), _BG, dtype=np.uint8)

        # ── Header ──
        cv2.putText(img, "SENTINEL LIVE DASHBOARD", (10, 28),
                    _FONT, 0.65, _CYAN, 2)
        cv2.line(img, (10, 38), (_W - 10, 38), _GRAY, 1)

        # ── Detection metrics ──
        score = det.get('drowsy_score', 0)
        sc = _score_color(score)

        y = 65
        cv2.putText(img, "DETECTION (Camera)", (10, y), _FONT, 0.5, _CYAN, 1)
        y += 28

        # Drowsy score with bar
        cv2.putText(img, f"Drowsy Score: {score:.3f}", (10, y),
                    _FONT, 0.48, sc, 1)
        _bar(img, 220, y - 12, 200, 14, score, 1.0, sc)
        trend = _trend_arrow(score, self._prev_det.get('drowsy_score'))
        cv2.putText(img, trend, (425, y), _FONT, 0.4, _WHITE, 1)
        y += 22

        # PERCLOS with bar
        perclos = det.get('perclos', 0)
        cv2.putText(img, f"PERCLOS: {perclos:.3f}", (10, y),
                    _FONT, 0.45, _WHITE, 1)
        _bar(img, 220, y - 10, 200, 12, perclos, 1.0,
             _RED if perclos > 0.20 else _GREEN)
        y += 20

        # Blinks + slow blinks
        blinks = det.get('blink_rate', 0)
        slow = det.get('slow_blinks', 0)
        cv2.putText(img, f"Blinks: {blinks}  Slow: {slow}", (10, y),
                    _FONT, 0.45, _WHITE, 1)
        y += 20

        # EAR std + Pitch var
        cv2.putText(img,
                    f"EAR std: {det.get('ear_std', 0):.4f}  "
                    f"Pitch var: {det.get('pitch_var', 0):.5f}",
                    (10, y), _FONT, 0.4, _GRAY, 1)
        y += 20

        # Critical flags
        ms = det.get('microsleep', False)
        hd = det.get('head_down', False)
        ms_color = _RED if ms else _GREEN
        hd_color = _RED if hd else _GREEN
        cv2.putText(img, f"Microsleep: {ms}", (10, y), _FONT, 0.45, ms_color, 1)
        cv2.putText(img, f"Head Down: {hd}", (240, y), _FONT, 0.45, hd_color, 1)
        y += 24

        # Alert duration (sustained recovery timer)
        alert_dur = det.get('alert_duration', 0)
        if alert_dur > 0:
            mins = int(alert_dur // 60)
            secs = int(alert_dur % 60)
            dur_str = f"{mins}:{secs:02d}" if mins else f"{secs}s"
            cv2.putText(img, f"Alert for: {dur_str}", (10, y),
                        _FONT, 0.5, _GREEN, 1)
            # Progress bar toward auto-recovery threshold
            from config import Config
            progress = min(alert_dur / Config.ALERT_RECOVERY_SECS, 1.0)
            _bar(img, 220, y - 12, 200, 14, progress, 1.0, _GREEN)
            if progress >= 1.0:
                cv2.putText(img, "RECOVERED", (430, y), _FONT, 0.4, _GREEN, 1)
        else:
            cv2.putText(img, "Alert for: --", (10, y),
                        _FONT, 0.5, _GRAY, 1)
        y += 24

        self._prev_det = det

        # ── Divider ──
        cv2.line(img, (10, y), (_W - 10, y), _GRAY, 1)
        y += 22

        # ── Voice metrics ──
        cv2.putText(img, f"VOICE (Turn {self._turn})", (10, y),
                    _FONT, 0.5, _BLUE, 1)
        y += 26

        with self._voice_lock:
            v = self._voice.copy()
            pv = self._prev_voice.copy()

        if v:
            # Energy RMS with bar + baseline comparison
            rms = v.get('energy_rms', 0)
            cv2.putText(img, f"Energy RMS: {rms:.4f}", (10, y),
                        _FONT, 0.45, _WHITE, 1)
            rms_bl = self._baselines.get('energy_rms', {})
            bar_color = _GREEN
            if rms_bl and rms_bl.get('avg', 0) > 0:
                ratio = rms / rms_bl['avg']
                if ratio < 0.7:
                    bar_color = _RED
                elif ratio < 0.85:
                    bar_color = _YELLOW
                cv2.putText(img, f"({ratio:.0%} of baseline)", (280, y),
                            _FONT, 0.38, _GRAY, 1)
            _bar(img, 220, y - 10, 50, 12, rms, 0.15, bar_color)
            y += 20

            # Speech rate
            rate = v.get('speech_rate_wpm')
            if rate is not None:
                cv2.putText(img, f"Speech Rate: {rate:.0f} wpm", (10, y),
                            _FONT, 0.45, _WHITE, 1)
                rate_bl = self._baselines.get('speech_rate_wpm', {})
                if rate_bl and rate_bl.get('avg', 0) > 0:
                    ratio = rate / rate_bl['avg']
                    cv2.putText(img, f"({ratio:.0%} of baseline)", (280, y),
                                _FONT, 0.38, _GRAY, 1)
                trend = _trend_arrow(rate, pv.get('speech_rate_wpm'))
                cv2.putText(img, trend, (430, y), _FONT, 0.4, _WHITE, 1)
                y += 20

            # Response latency
            lat = v.get('response_latency_s')
            if lat is not None:
                lat_color = _RED if lat > 4.0 else (_YELLOW if lat > 2.5 else _WHITE)
                cv2.putText(img, f"Response Latency: {lat:.1f}s", (10, y),
                            _FONT, 0.45, lat_color, 1)
                trend = _trend_arrow(lat, pv.get('response_latency_s'))
                cv2.putText(img, trend, (280, y), _FONT, 0.4, _WHITE, 1)
                y += 20

            # Pause ratio with bar
            pr = v.get('pause_ratio', 0)
            cv2.putText(img, f"Pause Ratio: {pr:.3f}", (10, y),
                        _FONT, 0.45, _WHITE, 1)
            _bar(img, 220, y - 10, 100, 12, pr, 1.0,
                 _RED if pr > 0.5 else (_YELLOW if pr > 0.3 else _GREEN))
            y += 20

            # Word count + duration
            wc = v.get('word_count', 0)
            dur = v.get('duration_s', 0)
            if wc > 0:
                cv2.putText(img, f"Words: {wc}  Duration: {dur:.1f}s", (10, y),
                            _FONT, 0.4, _GRAY, 1)
                y += 18
        else:
            cv2.putText(img, "Waiting for first voice sample...", (10, y),
                        _FONT, 0.4, _GRAY, 1)
            y += 20

        # ── Divider ──
        cv2.line(img, (10, y + 4), (_W - 10, y + 4), _GRAY, 1)
        y += 22

        # ── 8B Reasoner ──
        with self._reasoner_lock:
            r_level = self._reasoner_level
            r_conf = self._reasoner_conf
            r_reasoning = self._reasoner_reasoning

        level_colors = {
            "ALERT": _GREEN, "MILD": _YELLOW,
            "DROWSY": _RED, "CRITICAL": _RED,
        }
        lc = level_colors.get(r_level, _WHITE)

        cv2.putText(img, "8B REASONER", (10, y), _FONT, 0.5, _CYAN, 1)
        cv2.putText(img, f"{r_level} ({r_conf:.0%})", (160, y),
                    _FONT, 0.5, lc, 1)
        y += 22

        # Word-wrap reasoning text into ~55-char lines
        if r_reasoning:
            words = r_reasoning.split()
            line = ""
            for w in words:
                if len(line) + len(w) + 1 > 58:
                    cv2.putText(img, line.strip(), (10, y),
                                _FONT, 0.35, _GRAY, 1)
                    y += 16
                    line = w + " "
                else:
                    line += w + " "
            if line.strip():
                cv2.putText(img, line.strip(), (10, y),
                            _FONT, 0.35, _GRAY, 1)
                y += 16

        # ── Footer ──
        cv2.putText(img, "Live metrics | Updates every frame + per voice turn",
                    (10, _H - 12), _FONT, 0.3, _GRAY, 1)

        return img
