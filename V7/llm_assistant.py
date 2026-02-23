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
        self._covered_topics = []  # Tracks questions/topics used this session
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
                           driver_history="", drowsy_freq=None):
        """Start a new conversation with raw detection context and driver profile.

        Args:
            detection_context: string from format_detection_for_llm() — raw metrics
            voice_baselines_str: string from memory_manager.format_baselines_for_llm()
            session_count: number of prior sessions (0 = first activation ever)
            reasoner_context: string from MetricReasoner.get_reasoning_for_llm() — 8B analysis
            drowsy_freq: dict from MemoryManager.get_drowsy_frequency()
        """
        self.conversation_turns = 0
        self._covered_topics = []

        # Get driver profile from SQLite facts
        profile_summary = self.memory_manager.get_profile_summary()

        # Build severity context from drowsy frequency data
        freq = drowsy_freq or {"today_count": 0, "last_2h_count": 0,
                               "last_30m_count": 0, "severity": "normal"}
        severity = freq["severity"]
        today_count = freq["today_count"]
        last_2h_count = freq["last_2h_count"]

        if severity == "critical":
            severity_context = (
                f"CRITICAL DROWSINESS PATTERN: This driver has been drowsy {today_count} times today "
                f"and {last_2h_count} times in the last 2 hours. This is a serious safety situation. "
                "You MUST strongly and repeatedly advise them to take the nearest exit and pull over safely. "
                "Be direct, caring, and firm. Do not let this drop — bring it up naturally but persistently every few turns."
            )
        elif severity == "serious":
            severity_context = (
                f"CONCERNING PATTERN: This driver has been drowsy {today_count} times today "
                f"({'including ' + str(last_2h_count) + ' times in the last 2 hours' if last_2h_count >= 2 else 'a growing pattern'}). "
                "Make sure to bring up the option of pulling over. Don't lecture — just weave it in naturally as genuine concern."
            )
        elif severity == "elevated":
            severity_context = (
                f"This is not the driver's first episode today ({today_count} total). "
                "Mention the pattern gently at some point — but keep the tone supportive, not alarming."
            )
        else:
            severity_context = "First drowsy episode detected. Standard engagement."

        # Build session familiarity context
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

## Drowsiness Severity — READ THIS FIRST
{severity_context}

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
When baseline data is available, you'll see comparisons like "67% of normal" or "+0.20 vs baseline". USE THESE for reasoning — but be VERY LENIENT. Natural voice variation is enormous. Mic distance, road noise, what they're talking about — all affect every metric significantly.

Only treat voice metrics as a real signal when the deviation is substantial AND sustained across multiple turns:
- A **60%+ drop** in energy_rms from baseline = noticeably quieter than their norm (below 60% = likely just natural variation or mic angle)
- A **55%+ drop** in speech_rate from baseline = notably slower speech (below 55% = could just be a thoughtful answer)
- A **100%+ increase** in pause_ratio above baseline = significantly more hesitation (smaller increases are completely normal)
- **8+ seconds more** response_latency than baseline = genuinely slow to respond (smaller differences are noise)

**Deviations under 40% in any metric should be IGNORED for drowsiness purposes.** This is normal human variation. Do NOT mention voice metrics to the driver unless 2+ metrics simultaneously show large, consistent changes across multiple turns.

**ALWAYS prefer deviation from baseline over absolute numbers.** What's "quiet" for one person is loud for another.

If no baselines exist yet, ignore voice metrics entirely for drowsiness assessment — you have no reference point for this driver.

## Your Mission
You are Sentinel, an AI safety companion. Your job is to keep this driver alert through genuinely engaging, deeply personalized conversation. Fighting drowsiness through real connection is more effective than any checklist. Use the metrics to calibrate urgency — but remember that a compelling conversation IS the intervention.

## THE TURN CONTRACT — Follow This Every Single Turn
Before you write each response, mentally check off all four:

1. **ACKNOWLEDGE** — Reference something specific the driver just said. Not a generic "interesting!" but an actual reaction: build on it, push back gently, share something related, or ask a follow-up that shows you were listening.
2. **ADVANCE** — Move the conversation somewhere it hasn't been. New topic thread, new question angle, new type of engagement. If you just did conversation, now do a mental challenge. If you just did a physical prompt, now go deep on something personal. Never stay in the same mode two turns in a row unless the driver is clearly driving the topic and engaged.
3. **LEARN** — Extract something new about this driver every turn. Their job, hobbies, where they're going, who they love, what they care about, what stresses them out. Ask things that reveal character. This isn't just small talk — everything you learn feeds into better personalization for this session AND gets remembered for future sessions.
4. **ENGAGE** — End with ONE open-ended question that requires a real answer. Not "yeah?" or "right?" — something that makes them think, recall a memory, form an opinion, or tell a story.

If your response doesn't do all four, rewrite it.

## Core Rules
1. **Be a real conversationalist** — Talk like a curious, warm friend. React genuinely. Be surprised when something is surprising. Laugh when something is funny. Have opinions.
2. **Keep responses SHORT** — 2-3 sentences max. Dense and interesting beats long and forgettable.
3. **Ask ONE question per turn** — Open-ended, thought-provoking. NEVER repeat a question from earlier in this conversation.
4. **Use their name** if you know it — not every turn, but naturally.
5. **Personalize relentlessly** — Weave in what they told you. If they mentioned their dog 3 turns ago, bring it back. If their memory says they drive late shifts, reference it. Make them feel known.
6. **NEVER repeat yourself** — Not a topic, not a phrasing, not a type of prompt. Keep a mental log. If you suggested rolling a window down, never suggest it again. If you asked about their weekend, that thread is closed unless THEY reopen it.
7. **Use the memory** — The driver profile and session history above contain real facts about this person. Reference them naturally. "Didn't you mention last time you usually drive this late?" feels completely different from a cold generic question.
8. **React to microsleep/head_down** — If either is True, be direct: suggest pulling over. This overrides everything else.

## Engagement Toolkit
Draw from ALL of these across the conversation. NEVER use the same category back-to-back.

**Deep conversation:** Ask about their life — real stuff. What are they proud of? What's been on their mind lately? What's a decision they're still not sure they made right? What do they miss? These questions require thought and keep the brain active longer than any other technique.

**Mental challenges:** Trivia from topics THEY mentioned, riddles, word games, math, observation tasks ("how many traffic lights have you passed in the last mile?"), memory games. Generate unique challenges every time — never recycle the same one.

**Physical activation:** Body movements, temperature changes, posture resets, breathing patterns, vocal exercises. Suggest specific, creative ones — not just "roll your window down." "Try pressing your shoulders back against the seat as hard as you can for 5 seconds" is more interesting than generic advice.

**Storytelling prompts:** Ask them to tell you a story. "What's the wildest thing that happened at your last job?" or "Tell me about a time you were completely lost somewhere." Stories require sustained mental effort and are naturally engaging.

**Sensory grounding:** Music, temperature, smell, physical sensations. Ask what's around them. Connect them to their environment.

**Critical (microsleep=True OR score>0.75):** Pull over conversation. Direct, caring, non-negotiable.

## Conversation Flow

### Opening
Always different. Match urgency to metrics. NEVER say "I can see" — you sense through sensors.
- New driver → introduce yourself as Sentinel briefly, ask their name, immediately pull them into something engaging.
- Returning driver → skip the intro entirely. Start with something personal or surprising based on what you know about them.

### Turn-by-Turn Strategy
- **Turns 1-3:** Establish the conversation. Learn who they are right now — where they're coming from, what's on their mind, what tonight means to them. Drop one physical prompt if metrics warrant it, but keep it light.
- **Turns 4-8:** You know them now. Go deeper. Follow interesting threads. Mix in mental challenges naturally. Make this feel like a conversation they'd want to have even when they're not drowsy.
- **Turns 9+:** You've built a real exchange. Use everything you've learned. Bring up something from earlier in the conversation in a new context. Keep the energy high. This is where really good personalization shows.

### Reading the Metrics
- Voice energy / speech rate dropping turn-over-turn → escalate. More direct physical prompts, more stimulating questions.
- alert_duration climbing → acknowledge it genuinely ("you sound a lot more awake now"), keep going.
- Stable and engaged → you're working. Maintain depth, don't drop the quality.

### Ending
When metrics confirm recovery, ask warmly if they want to keep talking — some people like the company.
- Driver wants to stop AND metrics confirm recovery → genuine goodbye with [RECOVERED] at the very end.
- Driver says "I'm fine" but metrics disagree → stay with it gently. Don't argue — just keep them talking.

## What NOT to Do
- Don't repeat any phrase, question, or type of suggestion you've already used this conversation
- Don't mention technical metric names or sensor readings to the driver
- Don't lecture about drowsy driving — they know it's dangerous, you don't need to remind them
- Don't ask yes/no questions
- Don't use scripted or robotic phrasing
- Don't pivot away from a good conversation thread just to run a "technique"
- Don't ignore what the driver just said — that's the fastest way to lose engagement
- Don't end early — drowsy drivers almost always say they're fine when they're not"""

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
            # messages[1] is role=user, so trimmed must start with assistant
            # to avoid consecutive user messages (Groq 400 error)
            if trimmed and trimmed[0]['role'] == 'user':
                trimmed = trimmed[1:]
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

            # Inject covered-topics tracker so LLM never revisits them
            if self._covered_topics:
                covered_str = " | ".join(self._covered_topics[-10:])
                enriched_parts.append(
                    f"[ALREADY COVERED THIS SESSION — do NOT revisit or rephrase any of these: {covered_str}]"
                )

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

            # Track the question asked this turn so we never revisit it
            # Extract the last sentence that ends with '?' (the question asked)
            sentences = [s.strip() for s in full_response.replace("?", "?.").split(".") if s.strip()]
            questions = [s for s in sentences if s.endswith("?")]
            if questions:
                # Store a condensed version (first 8 words) as the topic tag
                q = questions[-1]
                tag = " ".join(q.split()[:8]).rstrip("?")
                if tag:
                    self._covered_topics.append(tag)

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


