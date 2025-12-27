import wave
import numpy as np
import pyaudio
from pathlib import Path
from piper import PiperVoice

# =========================
# Load Piper model
# =========================
HOME = Path.home()
model_path = HOME / "piper_voices" / "en_US-amy-medium.onnx"

voice = PiperVoice.load(model_path)

text = "This is a test of Piper text to speech on my system."

# =========================
# Synthesize audio
# =========================
audio = voice.synthesize(text)

# Normalize output: handle generator, bytes, or numpy array
import types
if isinstance(audio, types.GeneratorType):
    chunks = []
    for chunk in audio:
        if isinstance(chunk, (bytes, bytearray)):
            chunks.append(chunk)
        elif hasattr(chunk, 'tobytes'):
            chunks.append(chunk.tobytes())
        elif isinstance(chunk, str):
            chunks.append(chunk.encode())
        else:
            try:
                chunks.append(bytes(chunk))
            except Exception:
                pass
    audio = b''.join(chunks)

if isinstance(audio, np.ndarray):
    audio_np = audio.astype(np.int16)
elif isinstance(audio, (bytes, bytearray)):
    audio_np = np.frombuffer(audio, dtype=np.int16)
else:
    raise TypeError(f"Unsupported audio return type: {type(audio)}")

# =========================
# Write WAV file (CORRECT)
# =========================
WAV_FILE = "output.wav"
SAMPLE_RATE = voice.config.sample_rate
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16 = 2 bytes

with wave.open(WAV_FILE, "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(SAMPLE_WIDTH)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_np.tobytes())

print("✅ WAV file written:", WAV_FILE)

# =========================
# Playback with PyAudio
# =========================
p = pyaudio.PyAudio()

stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    output=True
)

print("🔊 Playing audio...")
stream.write(audio_np.tobytes())

stream.stop_stream()
stream.close()
p.terminate()

print("✅ Playback finished")
