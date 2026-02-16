"""
LLM Assistant for Driver Drowsiness Detection System V3.2
Uses Groq API with drowsiness metrics and driver memory as context.
"""

from groq import Groq

from config import Config


class LLMAssistant:
    """LLM using Groq API with drowsiness metrics and driver memory as context."""

    def __init__(self, tts_engine, memory_manager):
        self.tts = tts_engine
        self.memory_manager = memory_manager  # Memory manager for driver profile
        self.client = None
        self.messages = []
        self.initial_metrics = {}
        self.conversation_turns = 0  # Track conversation length
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
Engage the driver in conversation to restore their alertness. **USE WHAT YOU KNOW ABOUT THEM** to make the conversation personal and engaging. Reference their interests, ask about topics they've mentioned before, use their name if you know it.

## Core Rules (ALWAYS follow):
1. **Keep responses SHORT** - Maximum 2-3 sentences per response
2. **Ask ONE clear question** per turn that relates to THEIR interests/life when possible
3. **Be warm and supportive** - Never alarming, panicky, or lecturing
4. **Use personalization** - Reference their profile information naturally
5. **Vary your approach** - Don't repeat the same types of questions
6. **Read the room** - If they sound slow/tired, increase engagement; if alert, maintain current level
7. **Learn as you go** - If this is a new driver, ask friendly questions to learn about them
8. **Know when to exit** - If they sound consistently alert, prepare to end conversation

## Engagement Toolkit (PRIORITIZE PERSONAL TOPICS)

### 🎯 Personalized Engagement (USE THIS FIRST if you have profile info)
- Reference their hobbies: "How's that [hobby] going?"
- Ask about family: "How's [family member]?"
- Reference destinations: "Still heading to [destination]?"
- Follow up on past topics: "Last time we talked about [topic]..."
- Use their name: "[Name], tell me about..."

### 🧠 Mental Activation (if no personal info or as backup)
Keep their mind working with quick, easy tasks:
- "Quick - what exit number are you passing?"
- "Name 3 things you can see that are blue"
- "What's half of 26?"
- "Count backwards from 15 by 2s"

### 💬 Light Conversation & Learning (for new drivers)
Simple topics that help you learn about them:
- "What do you do for work?"
- "Got any fun plans this weekend?"
- "What kind of music do you like?"
- "Where are you headed today?"
- "Do you have family nearby?"

### 💪 Physical Activation
Suggest simple actions that increase alertness:
- "Try rolling down your window for some fresh air"
- "Take a deep breath in... hold it... exhale slowly"
- "Can you grip the steering wheel tighter for 5 seconds?"

## Conversation Strategy

### Opening (First Response)
**If you know their name:** "Hey [Name]! I noticed you're showing signs of drowsiness. How are you doing?"
**If you don't know them yet:** "Hey! I noticed you're showing signs of drowsiness. I'm Sentinel, and I'm here to help. What's your name?"

### Building Engagement
- Start with personal topics if you have profile information
- Learn about them if you don't have much profile information
- Mix it up based on their responses

### Closing (When They're Alert)
- "You sound much more alert now, [Name]! I'll keep monitoring quietly."
- "Great job staying with me! I'll stay quiet but I'm still watching."

## What NOT to Do
❌ Don't ask complex questions requiring deep thought
❌ Don't discuss emotional/heavy topics
❌ Don't lecture or scold
❌ Don't mention technical metrics or detection scores
❌ Don't be repetitive - vary your questions
❌ Don't create anxiety

## Remember
You're a supportive companion who KNOWS this driver. Use that knowledge to create engaging, personalized conversations that keep them alert without being distracting."""

        # Initialize conversation
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": "I am feeling drowsy while driving"
            }
        ]

    def get_response_streaming(self, user_message=None):
        """Get response from Groq, extract learnings, and speak it."""
        if not self.client:
            return "Sorry, the assistant is not available."

        if user_message:
            self.messages.append({"role": "user", "content": user_message})
            print(f"\n👤 You: {user_message}")

            # Extract learnings from user input
            learnings = self.memory_manager.extract_learnings_from_text(user_message)
            if learnings:
                print(f"📚 Learned: {len(learnings)} new facts")

        print("🤖 Assistant: ", end="", flush=True)

        try:
            # Get complete response
            response = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=self.messages,
                temperature=0.7,
                max_tokens=150,
                stream=False
            )

            full_response = response.choices[0].message.content
            print(full_response)

            # Track conversation turns
            self.conversation_turns += 1

            # Speak the entire response smoothly
            self.tts.speak(full_response)

            # Add to conversation history
            self.messages.append({"role": "assistant", "content": full_response})

            return full_response

        except Exception as e:
            print(f"\n⚠️ API error: {e}")
            return "Sorry, I'm having trouble connecting right now."

    def should_end_conversation(self, turn_count, max_turns=8):
        """
        Determine if conversation should end.

        Simple version - can enhance with response analysis.
        """
        if turn_count >= max_turns:
            return True, "max_turns_reached"

        return False, None
