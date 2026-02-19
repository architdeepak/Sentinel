"""
Voice Feature Extraction for Driver Drowsiness Detection System V7.

Extracts vocal biomarkers from raw audio:
  - Speech energy (RMS) — drowsy people speak more quietly
  - Speech rate (words/minute) — drowsy people speak slower
  - Response latency — drowsy people take longer to start talking
  - Pause ratio — drowsy people pause more within speech

V7 changes:
  - format_for_llm() returns RAW numbers + baseline deviation (not qualitative labels)
  - No hardcoded thresholds — the LLM reasons about severity from raw data
  - Baseline comparison shows % of normal and absolute deviation

Optimized for RPi: uses only numpy (no librosa/scipy), works on
raw PCM audio frames from SpeechRecognition's AudioData.
"""

import time
import numpy as np


class VoiceFeatureExtractor:
    """Extracts drowsiness-relevant voice metrics from raw audio."""

    __slots__ = (
        '_sample_rate', '_last_response_latency', '_prompt_time',
        '_cached_features',
    )

    def __init__(self, sample_rate=16000):
        self._sample_rate = sample_rate
        self._last_response_latency = None
        self._prompt_time = None
        self._cached_features = None

    def mark_prompt_end(self):
        """Call when TTS finishes — marks the start of the driver's response window."""
        self._prompt_time = time.perf_counter()

    def extract_features(self, audio_data, transcript=None):
        """Extract voice features from raw audio.

        Args:
            audio_data: SpeechRecognition AudioData object, or raw PCM bytes
            transcript: The recognized text (for word count / speech rate)

        Returns:
            dict with voice features, or None if extraction fails
        """
        try:
            # Convert AudioData to numpy array
            if hasattr(audio_data, 'get_raw_data'):
                raw = audio_data.get_raw_data(convert_rate=self._sample_rate,
                                               convert_width=2)
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, (bytes, bytearray)):
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                return None

            if len(samples) == 0:
                return None

            # ── Speech energy (RMS) ──
            rms = float(np.sqrt(np.mean(samples ** 2)))

            # ── Peak amplitude ──
            peak = float(np.max(np.abs(samples)))

            # ── Audio duration ──
            duration_s = len(samples) / self._sample_rate

            # ── Speech rate (words per minute) ──
            speech_rate_wpm = None
            word_count = 0
            if transcript and duration_s > 0:
                word_count = len(transcript.split())
                speech_rate_wpm = (word_count / duration_s) * 60.0

            # ── Response latency ──
            response_latency_s = None
            if self._prompt_time is not None:
                speech_start_offset = self._find_speech_onset(samples)
                elapsed = time.perf_counter() - self._prompt_time
                raw_latency = elapsed - duration_s + speech_start_offset
                self._last_response_latency = max(0.0, raw_latency)
                response_latency_s = self._last_response_latency
                self._prompt_time = None  # Reset

            # ── Pause ratio (fraction of audio that is silence) ──
            frame_size = int(0.025 * self._sample_rate)  # 25ms frames
            n_complete = (len(samples) // frame_size) * frame_size
            if n_complete > 0:
                frames = samples[:n_complete].reshape(-1, frame_size)
                frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
                pause_ratio = float(np.mean(frame_rms < 0.015))
            else:
                pause_ratio = 0.0

            features = {
                'energy_rms': round(rms, 4),
                'peak_amplitude': round(peak, 4),
                'duration_s': round(duration_s, 2),
                'speech_rate_wpm': round(speech_rate_wpm, 1) if speech_rate_wpm else None,
                'word_count': word_count,
                'response_latency_s': round(response_latency_s, 2) if response_latency_s is not None else None,
                'pause_ratio': round(pause_ratio, 3),
            }

            self._cached_features = features
            return features

        except Exception as e:
            print(f"⚠️ Voice feature extraction error: {e}")
            return None

    def _find_speech_onset(self, samples):
        """Find approximate time (seconds) when speech starts in the audio."""
        frame_size = int(0.03 * self._sample_rate)  # 30ms frames
        threshold = 0.02

        n_complete = (len(samples) // frame_size) * frame_size
        if n_complete == 0:
            return 0.0

        frames = samples[:n_complete].reshape(-1, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        loud = np.where(frame_rms > threshold)[0]
        if len(loud) > 0:
            return int(loud[0]) * frame_size / self._sample_rate
        return 0.0

    def format_for_llm(self, features=None, baselines=None):
        """Format voice features as raw numbers + baseline deviation.

        V7: No hardcoded qualitative labels. Returns raw values and
        baseline comparisons so the LLM can reason about severity itself.

        Args:
            features: dict from extract_features(), or uses cached
            baselines: dict from MemoryManager.get_baselines(), or None
                       Format: {metric_name: {avg, min, max, sample_count}}

        Returns:
            Compact multi-line string for injection into the LLM context.
        """
        f = features or self._cached_features
        if not f:
            return ""

        parts = []

        # Current raw values (always present)
        parts.append(f"energy_rms={f.get('energy_rms', 0):.4f}")

        if f.get('speech_rate_wpm') is not None:
            parts.append(
                f"speech_rate={f['speech_rate_wpm']:.1f}wpm "
                f"({f.get('word_count', 0)}w/{f.get('duration_s', 0):.1f}s)"
            )

        if f.get('response_latency_s') is not None:
            parts.append(f"response_latency={f['response_latency_s']:.1f}s")

        parts.append(f"pause_ratio={f.get('pause_ratio', 0):.3f}")
        parts.append(f"peak_amp={f.get('peak_amplitude', 0):.4f}")

        current_line = "VOICE: " + ", ".join(parts)

        # Baseline comparison (only if baselines exist)
        if baselines:
            deviations = []

            rms_bl = baselines.get('energy_rms')
            if rms_bl and rms_bl['avg'] > 0:
                ratio = f.get('energy_rms', 0) / rms_bl['avg']
                deviations.append(
                    f"energy={ratio:.0%} of normal (baseline={rms_bl['avg']:.4f})"
                )

            rate_bl = baselines.get('speech_rate_wpm')
            if rate_bl and f.get('speech_rate_wpm') is not None and rate_bl['avg'] > 0:
                ratio = f['speech_rate_wpm'] / rate_bl['avg']
                deviations.append(
                    f"speech_rate={ratio:.0%} of normal (baseline={rate_bl['avg']:.1f}wpm)"
                )

            pause_bl = baselines.get('pause_ratio')
            if pause_bl:
                diff = f.get('pause_ratio', 0) - pause_bl['avg']
                sign = "+" if diff >= 0 else ""
                deviations.append(
                    f"pause_ratio {sign}{diff:.3f} vs baseline ({pause_bl['avg']:.3f})"
                )

            lat_bl = baselines.get('response_latency_s')
            if lat_bl and f.get('response_latency_s') is not None:
                diff = f['response_latency_s'] - lat_bl['avg']
                sign = "+" if diff >= 0 else ""
                deviations.append(
                    f"latency {sign}{diff:.1f}s vs baseline ({lat_bl['avg']:.1f}s)"
                )

            if deviations:
                baseline_line = "VS BASELINE: " + ", ".join(deviations)
                return f"{current_line}\n{baseline_line}"

        return current_line
