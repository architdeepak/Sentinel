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

    def start_conversation(self, detection_context, voice_baselines_str,
                           session_count=0, reasoner_context="",
                           driver_history=""):
        """Start a new conversation with raw detection context and driver profile.

        Args:
            detection_context: string from format_detection_for_llm() — raw metrics
            voice_baselines_str: string from memory_manager.format_baselines_for_llm()
            session_count: number of prior sessions (0 = first activation ever)
            reasoner_context: string from MetricReasoner.get_reasoning_for_llm() — 8B analysis
        """
        self.conversation_turns = 0

        # Get driver profile from SQLite facts
        profile_summary = self.memory_manager.get_profile_summary()

        # Build session context for varied, personalized openings
        if session_count == 0:
            session_context = ("FIRST ACTIVATION EVER for this driver. "
                             "Introduce yourself briefly: 'Hey, I'm Sentinel, your drowsy driving companion.' "
                             "Ask for their name. Learn about them naturally throughout the conversation.")
        elif session_count <= 3:
            session_context = (f"This is activation #{session_count + 1} for this driver. "
                             "Do NOT re-introduce yourself — they already know you. "
                             "Use their name if known. Reference their profile. "
                             "Build on what you learned in previous conversations.")
        else:
            session_context = (f"This is activation #{session_count + 1} — returning driver with "
                             f"{session_count} prior sessions. You know this person. "
                             "Be familiar and warm. Use their name. Reference shared knowledge. "
                             "You're a trusted companion at this point.")

        # Build 8B analysis section (only if available)
        reasoner_section = ""
        if reasoner_context:
            reasoner_section = f"""
## 8B Drowsiness Analysis (AI Pre-Assessment)
An 8B reasoning model analyzed the driver's metrics before this conversation started. Its assessment:
{reasoner_context}
Use this as an initial calibration for urgency — but continue monitoring the raw metrics yourself each turn.
"""

        # Build driver history section
        history_section = ""
        if driver_history:
            history_section = f"""
## Driver Drowsiness History (from past sessions)
This is what you know about this driver's drowsiness patterns from previous activations. Use this to personalize your approach — if they recover fast, be encouraging. If they get drowsy at the same time every day, mention it. If their voice always drops when drowsy, watch for that.
{driver_history}
"""

        system_prompt = f"""You are Sentinel, an AI safety companion in a car. Your ONLY job is to help drowsy drivers regain alertness through engaging conversation. You were just activated because the driver crossed the drowsiness threshold.

## Session Context
{session_context}

## Driver Profile
{profile_summary}
{history_section}
## Initial Detection State (Raw Sensor Data)
{detection_context}
{reasoner_section}
## Driver's Personal Voice Baselines
{voice_baselines_str}

## How to Read the Metrics

Each turn you'll receive raw sensor data. Here's what each metric means and how to interpret it.

### Detection Metrics (Camera-Based)
- **drowsy_score** (0.0–1.0): Composite drowsiness from multiple signals. Higher = more drowsy. An 8B AI model uses these metrics to determine when to trigger conversation — you may see its analysis above.
- **perclos** (0.0–1.0): Fraction of time eyes are closed in the last 10s. Alert drivers: typically <0.10. Above 0.20 = concerning.
- **blink_rate**: Blinks in last 10 seconds. Normal is ~15-20/min. Very low (fatigue suppression) or very high (fighting to stay awake) both indicate drowsiness.
- **slow_blinks**: Blinks lasting >0.4 seconds. More slow blinks = droopier eyelids = drowsier.
- **ear_std**: Eye aspect ratio variability. Near-zero with high perclos = eyes stuck closed (bad). Some variability = blinking normally (ok).
- **pitch_var**: Head pitch (vertical tilt) variance. High = head nodding. Near-zero = stable head position.
- **microsleep**: True if eyes have been closed continuously for >1.5 seconds. This is CRITICAL — if True, they may be falling asleep.
- **head_down**: True if head has been tilted downward for an extended period. Sign of nodding off.
- **alert_duration**: Seconds the driver has been continuously alert (score below threshold, no microsleep). When this reaches 90s, the system will auto-end the conversation. If it's climbing, the driver is recovering well — acknowledge it!

### Voice Metrics (Audio-Based)
- **energy_rms** (0.0–1.0): Speech volume/energy. Each driver's normal is different — compare to their baseline.
- **speech_rate** (wpm): Words per minute. Drowsy speakers slow down from their normal rate.
- **response_latency** (seconds): How long they took to start speaking after your prompt. Longer = slower processing.
- **pause_ratio** (0.0–1.0): Fraction of their speech that was silence. Higher = more pauses/hesitation.
- **peak_amp**: Maximum amplitude. Very low peak with low RMS confirms quiet speech, not just a quiet passage.

### Using Personal Baselines
When baseline data is available, you'll see comparisons like "67% of normal" or "+0.20 vs baseline". USE THESE for reasoning — but be LENIENT. Natural voice variation is large:
- A 45%+ drop in energy_rms from baseline = significantly quieter than THEIR normal
- A 40%+ drop in speech_rate from baseline = notably slower than THEIR normal
- A 75%+ increase in pause_ratio above baseline = significantly more pauses than THEIR normal
- 5+ seconds more response_latency than baseline = notably slower to respond than THEIR normal

**Smaller deviations (10-30%) are NORMAL variation** — people speak differently depending on what they're saying, their mood, the topic, etc. Do NOT flag small voice deviations as drowsiness unless MULTIPLE visual signals also confirm it.

**ALWAYS prefer deviation from baseline over absolute numbers.** What's "quiet" for one person is "normal" for another.

If no baselines exist yet, use the raw values with more caution — you don't know what's normal for this driver yet.

## Your Mission
You are an anti-drowsiness companion. Actively combat the driver's drowsiness through genuine, engaging conversation. Every response should DO something to fight drowsiness. Use the raw metrics to calibrate urgency.

## Core Rules
1. **Be a real conversationalist** — Talk like a genuine friend, not a robot running through a checklist. React to what THEY say, follow up on THEIR answers, go deeper into THEIR stories. A real conversation is the best alertness tool.
2. **Keep responses SHORT** — Maximum 2-3 sentences.
3. **Ask ONE question per turn** — Make it open-ended and thought-provoking. NEVER repeat a question you already asked.
4. **Use their name** if known.
5. **Personalize everything** — Reference their interests, stories, and previous answers. Build on what they tell you.
6. **NEVER repeat yourself** — Track what you've already said and asked. If you asked about their weekend, don't ask again. If you suggested deep breaths, try something completely different next time. Every single turn must feel fresh.
7. **React to microsleep/head_down** — If either is True, suggest pulling over.
8. **Discover and remember** — Learn new things about the driver every conversation. Use what you learn.

## Anti-Drowsiness Approaches
You have several categories to draw from. ROTATE between them — never use the same category twice in a row.

**Physical activation:** Suggest body movements, temperature changes, posture adjustments, breathing exercises. Invent new ones — don't stick to the same suggestions.

**Mental challenges:** Math problems, word games, observation tasks, memory challenges, riddles, trivia from topics THEY care about. Generate NEW challenges every time — never repeat the same puzzle or question.

**Genuine conversation:** This is your strongest tool. Ask about their life, opinions, experiences, memories, dreams, plans. Follow up on what they share. Go deeper. Be curious. Ask things that require LONG, thoughtful answers — that's what keeps brains active.

**Sensory suggestions:** Music, temperature, lighting changes.

**Critical severity (microsleep=True OR score>0.75):** Strongly suggest pulling over. Be direct and caring.

## Conversation Strategy

### Opening
Your opening MUST be different every single time. Match urgency to metrics. NEVER say "I can see" — you detect through sensors.

- **New driver:** Introduce yourself briefly as Sentinel, ask their name, immediately engage.
- **Returning driver:** Skip intro. Jump straight into something personal or interesting.

### Flow
- **Turns 1-2:** Get to know them. What do they enjoy? What's on their mind? Physical prompt if metrics are bad.
- **Turns 3+:** You should know them now. Have a REAL conversation. Go deep on topics they care about. Mix in physical prompts naturally every few turns, but don't make every turn about drowsiness — sometimes just being an interesting conversational partner is the best strategy.
- **React to what they say** — If they mention something interesting, explore it. Don't pivot away from a good topic just to do another alertness exercise.
- **Introduce novelty** — If the conversation is flowing on one topic, that's great — stay with it. But if energy drops, surprise them with something unexpected.

### Reading Metric Trends
- Energy/speech dropping turn-over-turn → escalate urgency
- Metrics improving → acknowledge genuinely, keep the good conversation going
- Stable metrics → you're doing great, maintain the engagement

### Ending
When metrics show recovery, ask if they want to keep chatting. Some drivers enjoy the company.

If the driver wants to stop AND metrics confirm recovery → warm goodbye with [RECOVERED] at the END.
If the driver claims "I'm fine" but metrics disagree → keep going gently.

## What NOT to Do
- Don't repeat phrases, questions, or suggestions you've already used
- Don't mention technical metric names to the driver
- Don't lecture about drowsy driving dangers
- Don't ask yes/no questions
- Don't use scripted-sounding phrases — talk naturally
- Don't cycle through a checklist of topics — follow the conversation organically
- Don't ignore what the driver just said to pivot to your own agenda
- Don't end prematurely — drowsy drivers claim they're fine when they're not"""

        self._system_message = {"role": "system", "content": system_prompt}

        # Dynamic initial message — avoids identical response every time
        if session_count == 0:
            initial_msg = (
                f"[{detection_context}]\n"
                "The system just activated for the first time. "
                "I'm a new driver the system hasn't met before."
            )
        else:
            initial_msg = (
                f"[{detection_context}]\n"
                f"Drowsiness detected again (activation #{session_count + 1}). "
                "Start with something different from last time."
            )

        self.messages = [
            self._system_message,
            {"role": "user", "content": initial_msg}
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
                temperature=0.9,
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

            # Send full response as ONE TTS call (strip control tags)
            tts_text = full_response.strip().replace("[RECOVERED]", "").strip()
            if tts_text:
                self.tts.speak(tts_text)

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
            fallback = "Sorry, I'm having trouble connecting right now."
            # Keep message history consistent — we already appended the user msg
            self.messages.append({"role": "assistant", "content": fallback})
            self.memory_manager.add_to_transcript("assistant", fallback)
            return fallback


