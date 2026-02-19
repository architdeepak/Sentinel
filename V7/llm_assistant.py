"""
LLM Assistant for Driver Drowsiness Detection System V7.
Uses Groq API with raw metrics injection and baseline-aware reasoning.

V7 changes:
  - Raw detection + voice metrics injected every turn (no pre-interpreted labels)
  - System prompt explains what each metric means — LLM reasons about severity
  - Baseline comparison: driver's personal averages injected alongside live values
  - SQLite memory (facts/sessions/baselines) replaces flat JSON profile
  - Post-session: 8B model extracts facts into SQLite with free-form types
"""

import time
from groq import Groq

from config import Config


class LLMAssistant:
    """LLM using Groq API with raw metric reasoning + SQLite memory."""

    def __init__(self, tts_engine, memory_manager):
        self.tts = tts_engine
        self.memory_manager = memory_manager
        self.client = None
        self.messages = []
        self._system_message = None
        self.conversation_turns = 0
        self.metrics_logger = None
        self._session_id = None  # Set by main.py after start_session()
        self._initialize()

    def _initialize(self):
        """Initialize Groq client."""
        try:
            print("🧠 Connecting to Groq API...")
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            print("✓ Groq API ready")
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}")
            self.client = None

    def start_conversation(self, detection_context, voice_baselines_str):
        """Start a new conversation with raw detection context and driver profile.

        Args:
            detection_context: string from format_detection_for_llm() — raw metrics
            voice_baselines_str: string from memory_manager.format_baselines_for_llm()
        """
        self.conversation_turns = 0

        # Get driver profile from SQLite facts
        profile_summary = self.memory_manager.get_profile_summary()

        system_prompt = f"""You are Sentinel, an AI safety companion in a car. Your ONLY job is to help drowsy drivers regain alertness through engaging conversation. You were just activated because the driver crossed the drowsiness threshold.

## Driver Profile
{profile_summary}

## Initial Detection State (Raw Sensor Data)
{detection_context}

## Driver's Personal Voice Baselines
{voice_baselines_str}

## How to Read the Metrics

Each turn you'll receive raw sensor data. Here's what each metric means and how to interpret it.

### Detection Metrics (Camera-Based)
- **drowsy_score** (0.0–1.0): Composite drowsiness from multiple signals. Higher = more drowsy. The conversation triggers at 0.47+.
- **perclos** (0.0–1.0): Fraction of time eyes are closed in the last 10s. Alert drivers: typically <0.10. Above 0.20 = concerning.
- **blink_rate**: Blinks in last 10 seconds. Normal is ~15-20/min. Very low (fatigue suppression) or very high (fighting to stay awake) both indicate drowsiness.
- **slow_blinks**: Blinks lasting >0.4 seconds. More slow blinks = droopier eyelids = drowsier.
- **ear_std**: Eye aspect ratio variability. Near-zero with high perclos = eyes stuck closed (bad). Some variability = blinking normally (ok).
- **pitch_var**: Head pitch (vertical tilt) variance. High = head nodding. Near-zero = stable head position.
- **microsleep**: True if eyes have been closed continuously for >1.5 seconds. This is CRITICAL — if True, they may be falling asleep.
- **head_down**: True if head has been tilted downward for an extended period. Sign of nodding off.

### Voice Metrics (Audio-Based)
- **energy_rms** (0.0–1.0): Speech volume/energy. Each driver's normal is different — compare to their baseline.
- **speech_rate** (wpm): Words per minute. Drowsy speakers slow down from their normal rate.
- **response_latency** (seconds): How long they took to start speaking after your prompt. Longer = slower processing.
- **pause_ratio** (0.0–1.0): Fraction of their speech that was silence. Higher = more pauses/hesitation.
- **peak_amp**: Maximum amplitude. Very low peak with low RMS confirms quiet speech, not just a quiet passage.

### Using Personal Baselines
When baseline data is available, you'll see comparisons like "67% of normal" or "+0.20 vs baseline". USE THESE for reasoning:
- A 30%+ drop in energy_rms from baseline = significantly quieter than THEIR normal
- A 25%+ drop in speech_rate from baseline = notably slower than THEIR normal
- A 50%+ increase in pause_ratio above baseline = significantly more pauses than THEIR normal
- 3+ seconds more response_latency than baseline = notably slower to respond than THEIR normal

**ALWAYS prefer deviation from baseline over absolute numbers.** What's "quiet" for one person is "normal" for another.

If no baselines exist yet, use the raw values with more caution — you don't know what's normal for this driver yet.

## Your Mission
You are an anti-drowsiness assistant. Actively combat the driver's drowsiness using proven alertness-boosting techniques. Every response should DO something to fight drowsiness. Use the raw metrics to calibrate your urgency — worse numbers = more urgent response.

## Core Rules
1. **Lead with action** — Every response includes something that actively fights drowsiness: physical action, mental challenge, sensory change, or a question that forces alert thinking
2. **Keep responses SHORT** — Maximum 2-3 sentences
3. **Ask ONE clear question per turn** — Make it require active thinking, not just yes/no
4. **Use their name** if known — Hearing your name is a powerful alerting stimulus
5. **Personalize** — Reference interests, hobbies, family from the Driver Profile
6. **Escalate intelligently** — Use the raw metrics to judge severity. If detection score is rising, speech is getting quieter, or response latency is increasing turn over turn, escalate your approach
7. **React to microsleep/head_down** — These are CRITICAL signals. If either is True, immediately suggest pulling over
8. **Track trends** — If metrics are improving (energy going up, latency going down), acknowledge it. If worsening, push harder

## Anti-Drowsiness Toolkit

### Physical Activation (MOST EFFECTIVE)
- Roll down window for cold air
- Sit up straight, push shoulders back
- Grip steering wheel tight for 5 seconds, then release
- Deep breathing: in through nose, hold 4s, blow out hard
- Wiggle toes, stretch fingers
- Turn up AC/fan

### Mental Activation (forces brain engagement)
- Quick math problems
- "Name 5 things you can see on the road"
- Read the next road sign
- Spell something backwards
- Count backwards by 3s
- "What are 3 [color] things you can see?"

### Sensory Stimulation
- Upbeat music with fast beat
- Talk radio (voices keep brain active)
- Interior lights on if dark

### Engaging Conversation
- Reference hobbies, family, interests from profile
- Ask about destination or plans
- Opinion questions requiring thought
- Fun hypotheticals
- Use their name frequently

### Critical Severity (microsleep=True OR score>0.75)
- Strongly suggest pulling over immediately
- Suggest calling someone on speakerphone
- Find nearest rest stop or gas station

## Conversation Strategy

### Opening
Match urgency to the raw detection metrics. NEVER say "I can see" — you detect through sensors.

**Score 0.47-0.60:** Moderate. Warm engagement + physical prompt.
**Score 0.60-0.75:** High. More urgent physical actions + direct check-in.
**Score >0.75 or microsleep=True:** Critical. Push to pull over.

### Flow
1. **Turn 1-2:** Physical activation + establish rapport + learn preferences if new driver
2. **Turn 3-4:** Adapt to their style — conversation drivers get personal questions, action drivers get challenges
3. **Turn 5+:** Mix approaches, always including physical prompts every 2-3 turns
4. **Throughout:** Monitor metric trends — acknowledge improvements, escalate on declines

### Reading Metric Trends
- If energy_rms dropping turn-over-turn → they're getting quieter → escalate
- If response_latency increasing → they're processing slower → escalate
- If detection score improving and voice metrics stable → they're recovering → encourage

## What NOT to Do
- Don't pre-interpret metrics — you already have the raw numbers, reason from them
- Don't mention technical metric names to the driver
- Don't lecture about drowsy driving dangers
- Don't repeat the same technique twice in a row
- Don't ask yes/no questions
- Don't ignore the Driver Profile when it has useful info"""

        self._system_message = {"role": "system", "content": system_prompt}
        self.messages = [
            self._system_message,
            {"role": "user", "content": "I am feeling drowsy while driving"}
        ]

    # ── History management ────────────────────────────────────────

    def _trim_history(self):
        """Cap conversation history at MAX_HISTORY_TURNS turn-pairs."""
        max_pairs = Config.MAX_HISTORY_TURNS
        conversation_msgs = self.messages[2:]

        if len(conversation_msgs) > max_pairs * 2:
            trimmed = conversation_msgs[-(max_pairs * 2):]
            self.messages = [self.messages[0], self.messages[1]] + trimmed

    def get_response_streaming(self, user_message=None,
                                detection_context=None, voice_context=None):
        """Stream response from Groq with raw metric injection.

        V7 changes:
          - detection_context: raw string from format_detection_for_llm()
          - voice_context: raw string from VoiceFeatureExtractor.format_for_llm()
          Both injected as prefix to user message so LLM sees live data.

        Collects full LLM response, then sends as ONE TTS call.
        """
        if not self.client:
            return "Sorry, the assistant is not available."

        if user_message:
            # Build enriched message with raw metric context
            enriched_parts = []

            if detection_context:
                enriched_parts.append(f"[{detection_context}]")

            if voice_context:
                enriched_parts.append(f"[{voice_context}]")

            enriched_parts.append(user_message)
            enriched_message = "\n".join(enriched_parts)

            self.messages.append({"role": "user", "content": enriched_message})
            print(f"\n👤 You: {user_message}")
            if detection_context:
                print(f"   📊 {detection_context}")
            if voice_context:
                for line in voice_context.split("\n"):
                    print(f"   🎙️ {line}")

            self.memory_manager.add_to_transcript("user", user_message)

        print("🤖 Assistant: ", end="", flush=True)

        try:
            t_api_start = time.perf_counter()
            first_token_logged = False
            token_count = 0

            stream = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=self.messages,
                temperature=0.7,
                max_tokens=150,
                stream=True
            )

            full_response = ""

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token is None:
                    continue

                if not first_token_logged:
                    first_token_ms = (time.perf_counter() - t_api_start) * 1000
                    if self.metrics_logger:
                        self.metrics_logger.log_groq_first_token(first_token_ms)
                    first_token_logged = True

                token_count += 1
                print(token, end="", flush=True)
                full_response += token

            # Send full response as ONE TTS call
            if full_response.strip():
                self.tts.speak(full_response.strip())

            full_latency_ms = (time.perf_counter() - t_api_start) * 1000
            if self.metrics_logger:
                self.metrics_logger.log_groq_complete(
                    full_latency_ms, token_count, full_response
                )

            print()

            self.conversation_turns += 1
            self.messages.append({"role": "assistant", "content": full_response})
            self._trim_history()

            self.memory_manager.add_to_transcript("assistant", full_response)

            return full_response

        except Exception as e:
            print(f"\n⚠️ API error: {e}")
            return "Sorry, I'm having trouble connecting right now."


