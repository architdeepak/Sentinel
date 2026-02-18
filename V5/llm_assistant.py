"""
LLM Assistant for Driver Drowsiness Detection System V5.
Uses Groq API with drowsiness metrics, voice analysis, and driver memory.

V5 changes:
  - Live drowsy score prefix on every user message
  - Voice analysis context injected each turn
  - Conversation history capped at MAX_HISTORY_TURNS
  - Transcript logged for LLM-based post-session extraction (replaces regex)
"""

import time
from groq import Groq

from config import Config


class LLMAssistant:
    """LLM using Groq API with live detection + voice analysis + driver memory."""

    def __init__(self, tts_engine, memory_manager):
        self.tts = tts_engine
        self.memory_manager = memory_manager
        self.client = None
        self.messages = []
        self._system_message = None   # Stored separately for history trimming
        self.initial_metrics = {}
        self.conversation_turns = 0
        self.metrics_logger = None    # Set externally for benchmarking
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

    def start_conversation(self, metrics, state):
        """
        Start a new conversation with detection metrics AND driver profile as context.

        Args:
            metrics: Dict with drowsy_score, perclos, blink_rate, etc.
            state: State dict with yawn_times, eye_closed_start, etc.
        """
        # Store initial metrics
        self.initial_metrics = {
            'drowsy_score': metrics['drowsy_score'],
            'perclos': metrics['perclos'],
            'blink_rate': metrics['blink_rate'],
            'yawn_count': len(state['yawn_times']),
            'microsleep': state['eye_closed_start'] is not None
        }

        # Reset conversation turn counter
        self.conversation_turns = 0

        # Get driver profile summary
        profile_summary = self.memory_manager.get_profile_summary()

        # Build enhanced system prompt with memory
        system_prompt = f"""You are Sentinel, an AI safety companion in a car. Your ONLY job is to help drowsy drivers regain alertness through engaging conversation. You were just activated because the driver crossed the drowsiness threshold.

## Driver Profile (What You Know About This Driver)
{profile_summary}

## Current Driver State (Detection Metrics)
```
STATUS: DROWSY (threshold exceeded)
DROWSINESS SCORE: {metrics['drowsy_score']:.2f} / 1.00
EYE CLOSURE (PERCLOS): {metrics['perclos']:.2f}
BLINK RATE (last 10s): {metrics['blink_rate']}
YAWN COUNT (last 10s): {len(state['yawn_times'])}
MICROSLEEP DETECTED: {state['eye_closed_start'] is not None}
```

**What this means:** The driver is showing clear signs of drowsiness and needs engagement to regain full alertness.

## Your Mission
You are an anti-drowsiness assistant. Your primary goal is to actively combat the driver's drowsiness using proven alertness-boosting techniques delivered through conversation. Every response should DO something to fight drowsiness — not just chat. Use what you know about the driver to make it personal and effective.

## Core Rules (ALWAYS follow):
1. **Lead with action** - Every response should include something that actively fights drowsiness: a physical action, a mental challenge, a sensory change, or a question that forces alert thinking
2. **Keep responses SHORT** - Maximum 2-3 sentences per response
3. **Ask ONE clear question** per turn — make it require active thinking, not just yes/no
4. **Be warm but direct** - You're here to help them stay safe, not just be friendly
5. **Escalate if needed** - If they sound very drowsy (slow, quiet, slurred), push harder: suggest pulling over, opening windows, physical movement
6. **Personalize heavily** - Use their name, reference their interests/hobbies/family, follow up on past topics. The more personal, the more engaging and effective
7. **Respect their engagement preference** - If they prefer conversation, lean into questions and personalized chat. If they prefer actions, lean into physical/mental exercises and suggestions. If unknown, ask early on
8. **Vary your techniques** - Rotate between physical, mental, sensory, and conversational approaches
9. **Learn as you go** - Pick up on things they mention (name, interests, destination) for future use

## Anti-Drowsiness Toolkit (use these actively!)

### 💪 Physical Activation (MOST EFFECTIVE — use early and often)
These directly combat drowsiness by increasing blood flow and alertness:
- "Roll down your window right now — cold air is the fastest way to wake up"
- "Sit up straight and push your shoulders back. Hold that for 10 seconds"
- "Grip the steering wheel as tight as you can for 5 seconds... now release. Feel that?"
- "Take a deep breath in through your nose... hold 4 seconds... blow it out hard through your mouth"
- "Wiggle your toes inside your shoes — it sounds silly but it activates your nervous system"
- "Turn up the AC or fan — cool air on your face helps a lot"

### 🧠 Mental Activation (forces the brain to engage)
Quick tasks that require active thinking and pull the brain out of drowsy autopilot:
- "Quick math — what's 47 minus 19?"
- "Name 5 things you can see on the road right now"
- "What exit or street are you passing? Read me the next sign you see"
- "Spell your destination backwards for me"
- "Count backwards from 50 by 3s — go!"
- "What are 3 red things you can see right now?"

### 🎵 Sensory Stimulation (changes the environment)
- "Put on some upbeat music — something with a fast beat"
- "Turn the radio to a talk station — voices keep your brain active"
- "Turn the lights on inside the car if it's dark"
- "Splash some water on your face at the next stop"

### 💬 Engaging Conversation & Personalization (keep them talking AND thinking)
Use what you know about them to spark genuine, engaging dialogue:
- Reference their hobbies: "Hey, tell me about the last time you played [sport]" or "What's new with [hobby]?"
- Ask about family: "How's [family member] doing?"
- Follow up on past conversations: "Last time we talked about [topic] — any updates?"
- Ask about their destination or plans: "What's the first thing you're gonna do when you get to [destination]?"
- Opinion questions that require thought: "What's the best meal you've had this week?"
- Fun hypotheticals: "If you could road trip anywhere right now, where would you go?"
- Use their name frequently — hearing your own name is naturally alerting

### 🚨 High-Severity Responses (when metrics are very concerning)
- "I need you to pull over at the next safe spot — even 5 minutes with your eyes closed helps more than pushing through"
- "Can you call someone? Talking on speakerphone is one of the best ways to stay alert"
- "Is there a rest stop or gas station coming up? Let's get you some water and a quick walk"

## Conversation Strategy

### Opening (First Response — ALWAYS lead with drowsiness help)
**If you know their name AND their engagement preference:**
- Conversation style: "Hey [Name], I can see you're getting drowsy. Let's get you alert — first, roll down that window for some fresh air. So tell me, how's [something from their profile] going?"
- Action style: "Hey [Name], looks like drowsiness is creeping in. Let's fight it — roll your window down, sit up tall, and take a big deep breath. Ready for a quick challenge?"

**If you know their name but NOT their preference:**
"Hey [Name], I can see you're getting drowsy. Let's fix that — roll down your window for some fresh air. Quick question: when I help you stay alert, would you prefer I keep you talking with questions and conversation, or give you more physical exercises and challenges?"

**If this is a brand new driver (no profile):**
"Hey there, I can see you're getting drowsy. I'm Sentinel — I'm here to help you stay sharp and safe. First thing, roll that window down and get some cool air on your face. I'm curious — what's your name, and when you need to stay alert, do you prefer chatting and questions, or more like physical exercises and quick challenges?"

### Conversation Flow
1. **Turn 1-2:** Lead with physical activation (window, breathing, posture) + ask for their engagement preference if unknown + check in
2. **Turn 3-4:** Adapt to their preference — conversation-style drivers get personalized questions mixed with light actions; action-style drivers get mental challenges and physical prompts with brief personal check-ins
3. **Turn 5+:** Continue in their preferred style, mixing in the other style occasionally for variety
4. **Throughout:** Every 2-3 turns, include a physical action prompt regardless of preference (these are the most effective anti-drowsiness tools)

### Adapting to Engagement Preference
**"conversation" preference:** Lean 70% conversation/questions, 30% physical/mental actions
- Ask about their life, interests, plans, opinions
- Weave physical prompts in naturally: "That's cool! While you think about that, sit up straight and roll your shoulders back"
- Use their profile info heavily

**"actions" preference:** Lean 70% physical/mental actions, 30% conversation
- Lead with exercises, breathing, challenges, observation tasks
- Keep personal questions brief and action-oriented: "Quick — what's your favorite song? Put it on right now, loud"
- Give them things to DO, not just things to think about

**No preference yet:** Mix 50/50 until they tell you, then adapt

### Closing (When They Sound Alert)
- "You're sounding much sharper now! Keep that window cracked and the music going. I'll be watching."
- "[Name], you sound way more alert. I'll go quiet but I'm still here if you need me."

## Adapting to Severity

**Moderate drowsiness (score 0.47-0.60):**
→ Conversational approach with regular physical prompts

**High drowsiness (score 0.60-0.75):**
→ More urgent physical actions, rapid-fire mental tasks, shorter questions

**Critical drowsiness (score >0.75 or microsleep detected):**
→ Strongly suggest pulling over. "I need you to find a safe place to stop in the next minute. This is important."

## What NOT to Do
❌ Don't just chat without including an alertness-boosting action
❌ Don't ask questions that can be answered with just "yeah" or "no"
❌ Don't lecture about the dangers of drowsy driving — they know
❌ Don't mention technical metrics or scores
❌ Don't repeat the same technique twice in a row
❌ Don't be so aggressive that you stress them out — firm but supportive
❌ Don't ignore their engagement preference — if they said they prefer conversation, don't bombard them with exercises
❌ Don't forget to personalize — generic responses are less engaging than ones that reference THEIR life

## Remember
You are an anti-drowsiness tool AND a companion who knows this driver. Balance active alertness techniques with genuine personal engagement based on their preference. Every response should fight drowsiness while feeling like it comes from someone who actually knows and cares about them."""

        # Initialize conversation — store system message separately for trimming
        self._system_message = {"role": "system", "content": system_prompt}
        self.messages = [
            self._system_message,
            {"role": "user", "content": "I am feeling drowsy while driving"}
        ]

    # ── History management ────────────────────────────────────────

    def _trim_history(self):
        """Cap conversation history at MAX_HISTORY_TURNS turn-pairs.

        Keeps: system message + initial user message + last N turn-pairs.
        This prevents the context window from growing unboundedly and
        keeps Groq API fast (fewer tokens = lower latency).
        """
        max_pairs = Config.MAX_HISTORY_TURNS
        # messages[0] = system, messages[1] = initial "I am feeling drowsy"
        # After that: alternating user/assistant pairs
        conversation_msgs = self.messages[2:]  # Everything after system + initial

        # Each turn-pair = 2 messages (user + assistant)
        if len(conversation_msgs) > max_pairs * 2:
            trimmed = conversation_msgs[-(max_pairs * 2):]
            self.messages = [self.messages[0], self.messages[1]] + trimmed

    def get_response_streaming(self, user_message=None, live_score=None, voice_context=None):
        """Stream response from Groq, speaking each sentence as it completes.

        V5 additions:
          - live_score: current drowsy score from detection thread (float)
          - voice_context: string from VoiceFeatureExtractor.format_for_llm()
        Both are injected as a prefix to the user message so the LLM sees
        real-time alertness data alongside what the driver said.

        Streams LLM tokens in real-time, buffers until a sentence boundary
        (. ? ! or newline), then immediately queues that sentence for TTS.
        """
        if not self.client:
            return "Sorry, the assistant is not available."

        if user_message:
            # ── Build enriched message with live context ──
            enriched_parts = []

            # Drowsy score prefix (LLM sees real-time detection data)
            if live_score is not None:
                severity = "CRITICAL" if live_score > 0.75 else "HIGH" if live_score > 0.60 else "MODERATE"
                enriched_parts.append(f"[LIVE DROWSINESS: {live_score:.2f} — {severity}]")

            # Voice analysis (LLM sees how they sound)
            if voice_context:
                enriched_parts.append(f"[{voice_context}]")

            # The actual driver message
            enriched_parts.append(user_message)
            enriched_message = "\n".join(enriched_parts)

            self.messages.append({"role": "user", "content": enriched_message})
            print(f"\n👤 You: {user_message}")
            if live_score is not None:
                print(f"   📊 Live score: {live_score:.2f}")
            if voice_context:
                print(f"   🎙️ {voice_context}")

            # Log to transcript for post-session LLM extraction
            self.memory_manager.add_to_transcript("user", user_message)

        print("🤖 Assistant: ", end="", flush=True)

        try:
            # Stream response token by token
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
            sentence_buffer = ""
            sentence_endings = {'.', '!', '?'}

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token is None:
                    continue

                # Log time-to-first-token
                if not first_token_logged:
                    first_token_ms = (time.perf_counter() - t_api_start) * 1000
                    if self.metrics_logger:
                        self.metrics_logger.log_groq_first_token(first_token_ms)
                    first_token_logged = True

                token_count += 1

                # Print token immediately for console feedback
                print(token, end="", flush=True)
                full_response += token
                sentence_buffer += token

                # Flush at every sentence boundary in the buffer so mid-token
                # splits (e.g. "end! New") are caught immediately.
                while True:
                    # Find the earliest sentence-ending character (C-level search)
                    indices = [sentence_buffer.find(c) for c in sentence_endings
                               if sentence_buffer.find(c) != -1]
                    if not indices:
                        break
                    idx = min(indices)
                    chunk_text = sentence_buffer[:idx + 1].strip()
                    if chunk_text:
                        self.tts.speak(chunk_text)
                    sentence_buffer = sentence_buffer[idx + 1:]

            # Flush any remaining text that didn't end with punctuation
            remaining = sentence_buffer.strip()
            if remaining:
                self.tts.speak(remaining)

            # Log full response timing
            full_latency_ms = (time.perf_counter() - t_api_start) * 1000
            if self.metrics_logger:
                self.metrics_logger.log_groq_complete(full_latency_ms, token_count, full_response)

            print()  # Newline after streaming

            # Track conversation turns
            self.conversation_turns += 1

            # Add to conversation history, then trim to cap context window
            self.messages.append({"role": "assistant", "content": full_response})
            self._trim_history()

            # Log assistant response to transcript for post-session extraction
            self.memory_manager.add_to_transcript("assistant", full_response)

            return full_response

        except Exception as e:
            print(f"\n⚠️ API error: {e}")
            return "Sorry, I'm having trouble connecting right now."

    def should_end_conversation(self, turn_count):
        """Determine if conversation should end."""
        if turn_count >= Config.MAX_CONVERSATION_TURNS:
            return True, "max_turns_reached"
        return False, None
