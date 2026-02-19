"""
Voice Feature Extraction for Driver Drowsiness Detection System V5.

Extracts vocal biomarkers from raw audio that indicate drowsiness:
  - Speech energy (RMS) — drowsy people speak quietly
  - Speech rate (words/minute) — drowsy people speak slowly
  - Response latency — drowsy people take longer to start talking
  - Pause ratio — drowsy people pause more within speech

These features are injected into the LLM context so it can gauge
alertness from HOW they're talking, not just WHAT they're saying.

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
        self._prompt_time = None          # When TTS finished (driver's turn to speak)
        self._cached_features = None      # Cache last extracted features

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
            # = time from prompt end to when the driver first spoke.
            # We're called post-recording, so: total elapsed - recording duration
            # gives us the wait before recording started, then add the in-audio
            # silence before first speech.
            response_latency_s = None
            if self._prompt_time is not None:
                speech_start_offset = self._find_speech_onset(samples)
                elapsed = time.perf_counter() - self._prompt_time
                # elapsed = pre-listen setup + recording duration
                # latency = elapsed - recording duration + silence-before-speech
                raw_latency = elapsed - duration_s + speech_start_offset
                self._last_response_latency = max(0.0, raw_latency)
                response_latency_s = self._last_response_latency
                self._prompt_time = None  # Reset

            # ── Pause ratio (fraction of audio that is silence) ──
            # Simple energy-based VAD using vectorized numpy reshape
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
        """Find approximate time (seconds) when speech starts in the audio.
        Uses vectorized energy scan across 30ms frames."""
        frame_size = int(0.03 * self._sample_rate)  # 30ms frames
        threshold = 0.02  # RMS threshold for "speech started"

        n_complete = (len(samples) // frame_size) * frame_size
        if n_complete == 0:
            return 0.0

        frames = samples[:n_complete].reshape(-1, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        loud = np.where(frame_rms > threshold)[0]
        if len(loud) > 0:
            return int(loud[0]) * frame_size / self._sample_rate
        return 0.0

    def get_last_features(self):
        """Return the last extracted features (cached)."""
        return self._cached_features

    def format_for_llm(self, features=None):
        """Format voice features as a compact string for LLM context injection.

        Returns a human-readable assessment, not raw numbers.
        The LLM gets qualitative descriptions it can act on.
        """
        f = features or self._cached_features
        if not f:
            return ""

        assessments = []

        # Energy assessment
        rms = f.get('energy_rms', 0)
        if rms < 0.015:
            assessments.append("Voice is very quiet/mumbled (sign of heavy drowsiness)")
        elif rms < 0.03:
            assessments.append("Voice is somewhat quiet (moderate drowsiness indicator)")
        # else: normal volume, don't mention

        # Speech rate assessment
        rate = f.get('speech_rate_wpm')
        if rate is not None:
            if rate < 80:
                assessments.append(f"Speaking very slowly ({rate:.0f} wpm — typically drowsy)")
            elif rate < 110:
                assessments.append(f"Speaking somewhat slowly ({rate:.0f} wpm)")
            # else: normal rate, don't mention

        # Response latency
        latency = f.get('response_latency_s')
        if latency is not None:
            if latency > 8.0:
                assessments.append(f"Very slow to respond ({latency:.1f}s delay — significant drowsiness)")
            elif latency > 5.0:
                assessments.append(f"Slow to respond ({latency:.1f}s delay)")
            # else: normal, don't mention

        # Pause ratio
        pauses = f.get('pause_ratio', 0)
        if pauses > 0.5:
            assessments.append("Speech has many long pauses (fragmented, drowsy pattern)")
        elif pauses > 0.35:
            assessments.append("Speech has noticeable pauses")

        if assessments:
            return "VOICE ANALYSIS: " + "; ".join(assessments)
        return "VOICE ANALYSIS: Speech sounds alert and normal"
