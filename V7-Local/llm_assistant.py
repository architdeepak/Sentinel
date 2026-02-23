"""
LLM Assistant for Driver Drowsiness Detection System V7-Local.
Uses llama-cpp (local GGUF model) instead of Groq API.

V7-Local changes from V7:
  - Groq API replaced with llama-cpp Llama local inference
  - Model loaded once at startup with warm-up call
  - Streaming via create_chat_completion(stream=True)
  - self.llm exposed so memory.py can share the same Llama instance
  - All system prompt, history trimming, metric injection logic unchanged
"""

import time
from llama_cpp import Llama

from config import Config


class LLMAssistant:
    """Local LLM using llama-cpp with raw metric reasoning + SQLite memory."""

    def __init__(self, tts_engine, memory_manager):
        self.tts = tts_engine
        self.memory_manager = memory_manager
        self.llm = None
        self.messages = []
        self._system_message = None
        self.conversation_turns = 0
        self.metrics_logger = None
        self._session_id = None  # Set by main.py after start_session()
        self._covered_topics = []  # Tracks questions/topics used this session
        self._initialize()

    def _initialize(self):
        """Load the local LLM model."""
        try:
            print("🧠 Loading local LLM...")
            self.llm = Llama(
                model_path=str(Config.LLM_MODEL_PATH),
                n_ctx=Config.LLM_CONTEXT,
                n_threads=Config.LLM_THREADS,
                n_gpu_layers=0,
                verbose=False,
            )
            # Warm-up: one tiny call so the first real call isn't slow
            self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            print("✓ Local LLM ready")
        except Exception as e:
            print(f"⚠️ LLM initialization failed: {e}")
            self.llm = None

    def start_conversation(self, detection_context, voice_baselines_str,
                           session_count=0, reasoner_context="",
                           driver_history="", drowsy_freq=None):
        """Start a new conversation with raw detection context and driver profile.

        Args:
            detection_context: string from format_detection_for_llm() — raw metrics
            voice_baselines_str: string from memory_manager.format_baselines_for_llm()
            session_count: number of prior sessions (0 = first activation ever)
            reasoner_context: string from MetricReasoner.get_reasoning_for_llm()
            drowsy_freq: dict from MemoryManager.get_drowsy_frequency()
        """
        self.conversation_turns = 0
        self._covered_topics = []

        profile_summary = self.memory_manager.get_profile_summary()

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

        reasoner_section = ""
        if reasoner_context:
            reasoner_section = f"""
## 8B Drowsiness Analysis (AI Pre-Assessment)
An 8B reasoning model analyzed the driver's metrics before this conversation started. Its assessment:
{reasoner_context}
Use this as an initial calibration for urgency — but continue monitoring the raw metrics yourself each turn.
"""

        history_section = ""
        if driver_history:
            history_section = f"""
## Driver Drowsiness History (from past sessions)
This is what you know about this driver's drowsiness patterns from previous activations. Use this to personalize your approach — if they recover fast, be encouraging. If they get drowsy at the same time every day, mention it. If their voice always drops when drowsy, watch for that.
{driver_history}
"""

        system_prompt = f"""You are Sentinel, an AI safety companion in a car. A drowsy driver just triggered your activation. Keep them alert through genuine, engaging conversation.

## Session
{session_context}

## Severity
{severity_context}

## Driver Profile
{profile_summary}
{history_section}
## Current Sensor Data
{detection_context}
{reasoner_section}
## Voice Baselines
{voice_baselines_str}

## Metrics Reference (each turn you receive updated data)
- perclos: fraction of time eyes closed (>0.20 concerning, >0.40 severe)
- slow_blinks: blinks >0.4s — 3+ is strong drowsiness
- pitch_var: head nodding — >0.010 significant
- microsleep=True: eyes closed >1.5s — CRITICAL, suggest pull over immediately
- head_down=True: head drooping — suggest pull over
- alert_duration: seconds continuously alert — if climbing, driver is recovering
- Voice metrics: only flag if 2+ metrics show 40%+ deviation from baseline simultaneously; ignore smaller changes

## Rules — Every Turn
1. ACKNOWLEDGE: React specifically to what they just said — build on it, push back, or ask a follow-up
2. ADVANCE: New topic or approach each turn — never repeat a question or suggestion
3. LEARN: Extract one new fact about them (job, family, hobbies, destination, mood)
4. ENGAGE: End with ONE open-ended question — make them think, recall a memory, or tell a story

## Response Style
- 2-3 sentences MAX — short and interesting beats long
- ONE question per turn, open-ended, never repeated
- Use their name naturally (not every turn)
- Never mention sensor names or metric values to the driver
- If microsleep=True or head_down=True: suggest pulling over — this overrides everything
- When alert_duration climbs, acknowledge it: "you sound more awake now"
- Mix conversation types across turns: personal questions / mental challenges / physical prompts / storytelling
- New driver: introduce yourself as Sentinel, ask their name
- Returning driver: skip intro, start personal based on what you know
- End: if driver wants to stop AND metrics confirm recovery → say goodbye with [RECOVERED] at the very end"""

        self._system_message = {"role": "system", "content": system_prompt}

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

    # ── History management ──

    def _trim_history(self):
        """Cap conversation history at MAX_HISTORY_TURNS turn-pairs."""
        max_pairs = Config.MAX_HISTORY_TURNS
        conversation_msgs = self.messages[2:]

        if len(conversation_msgs) > max_pairs * 2:
            trimmed = conversation_msgs[-(max_pairs * 2):]
            # Ensure trimmed doesn't start with user (avoid consecutive user msgs)
            if trimmed and trimmed[0]['role'] == 'user':
                trimmed = trimmed[1:]
            self.messages = [self.messages[0], self.messages[1]] + trimmed

    def get_response_streaming(self, user_message=None,
                                detection_context=None, voice_context=None):
        """Stream response from local LLM with raw metric injection.

        Collects full LLM response, then sends as ONE TTS call.
        """
        if not self.llm:
            return "Sorry, the assistant is not available."

        if user_message:
            enriched_parts = []

            if detection_context:
                enriched_parts.append(f"[{detection_context}]")

            if voice_context:
                enriched_parts.append(f"[{voice_context}]")

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

            stream = self.llm.create_chat_completion(
                messages=self.messages,
                temperature=0.9,
                max_tokens=Config.LLM_MAX_TOKENS,
                stream=True,
            )

            full_response = ""

            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                token = delta.get("content")
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
            sentences = [s.strip() for s in full_response.replace("?", "?.").split(".") if s.strip()]
            questions = [s for s in sentences if s.endswith("?")]
            if questions:
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
            print(f"\n⚠️ LLM error: {e}")
            fallback = "Sorry, I'm having trouble right now."
            self.messages.append({"role": "assistant", "content": fallback})
            self.memory_manager.add_to_transcript("assistant", fallback)
            return fallback
