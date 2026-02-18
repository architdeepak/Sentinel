"""
Memory Manager for Driver Drowsiness Detection System V5.
Manages persistent driver profile and conversation history.

V5: Replaces regex-based extraction with LLM-based post-session extraction.
The LLM reads the full conversation transcript and extracts structured
personal facts — far more accurate than pattern matching.
"""

import json
import time
from pathlib import Path

from groq import Groq
from config import Config


class MemoryManager:
    """Manages persistent driver profile with LLM-powered extraction."""

    def __init__(self, profile_path="sentinel_driver_profile.json"):
        self.profile_path = Path.home() / profile_path
        self.profile = self.load_profile()
        self.conversation_transcript = []  # (role, text) pairs for LLM extraction
        self._groq_client = None

    def load_profile(self):
        """Load driver profile from disk."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading profile: {e}")
                return self.create_default_profile()
        else:
            print("📝 Creating new driver profile...")
            return self.create_default_profile()

    def save_profile(self):
        """Save profile to disk."""
        try:
            with open(self.profile_path, 'w') as f:
                json.dump(self.profile, f, indent=2)
            print(f"✓ Profile saved to {self.profile_path}")
        except Exception as e:
            print(f"⚠️ Error saving profile: {e}")

    def create_default_profile(self):
        """Initialize empty profile structure."""
        return {
            "personal": {
                "name": None,
                "occupation": None,
                "family": [],
                "location": None
            },
            "preferences": {
                "engagement_style": None
            },
            "interests": {
                "hobbies": [],
                "sports_teams": [],
                "music_preferences": [],
                "topics_engaged_with": []
            },
            "driving_patterns": {
                "common_routes": [],
                "typical_times": [],
                "usual_destinations": []
            },
            "conversation_history": {
                "last_topics": [],
                "successful_engagement_types": [],
                "things_to_avoid": []
            },
            "system_metadata": {
                "total_conversations": 0,
                "last_conversation": None,
                "profile_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_drowsy_episodes": 0
            }
        }

    def update_field(self, category, key, value):
        """Update a specific field in the profile (caller must call save_profile)."""
        if category not in self.profile:
            self.profile[category] = {}
        self.profile[category][key] = value

    def append_to_list(self, category, key, value, max_items=10):
        """Append to a list field, maintaining max size (caller must call save_profile)."""
        if category not in self.profile:
            self.profile[category] = {}
        if key not in self.profile[category]:
            self.profile[category][key] = []

        # Avoid duplicates
        if value not in self.profile[category][key]:
            self.profile[category][key].append(value)
            # Keep only most recent items
            if len(self.profile[category][key]) > max_items:
                self.profile[category][key] = self.profile[category][key][-max_items:]

    def get_profile_summary(self):
        """Get formatted profile summary for LLM context."""
        summary_lines = []

        # Personal information
        personal = self.profile.get("personal", {})
        if personal.get("name"):
            summary_lines.append(f"Driver's name: {personal['name']}")
        if personal.get("occupation"):
            summary_lines.append(f"Occupation: {personal['occupation']}")
        if personal.get("family") and len(personal['family']) > 0:
            summary_lines.append(f"Family: {', '.join(personal['family'])}")
        if personal.get("location"):
            summary_lines.append(f"Location: {personal['location']}")

        # Preferences
        preferences = self.profile.get("preferences", {})
        if preferences.get("engagement_style"):
            summary_lines.append(f"Preferred engagement style: {preferences['engagement_style']}")

        # Interests
        interests = self.profile.get("interests", {})
        if interests.get("hobbies") and len(interests['hobbies']) > 0:
            summary_lines.append(f"Hobbies/Interests: {', '.join(interests['hobbies'])}")
        if interests.get("sports_teams") and len(interests['sports_teams']) > 0:
            summary_lines.append(f"Favorite sports teams: {', '.join(interests['sports_teams'])}")
        if interests.get("music_preferences") and len(interests['music_preferences']) > 0:
            summary_lines.append(f"Music preferences: {', '.join(interests['music_preferences'])}")

        # Driving patterns
        driving = self.profile.get("driving_patterns", {})
        if driving.get("common_routes") and len(driving['common_routes']) > 0:
            summary_lines.append(f"Common routes: {', '.join(driving['common_routes'][:3])}")
        if driving.get("usual_destinations") and len(driving['usual_destinations']) > 0:
            summary_lines.append(f"Frequent destinations: {', '.join(driving['usual_destinations'][:3])}")

        # Conversation history
        conv_history = self.profile.get("conversation_history", {})
        if conv_history.get("last_topics") and len(conv_history['last_topics']) > 0:
            recent_topics = conv_history['last_topics'][-3:]
            summary_lines.append(f"Recent conversation topics: {', '.join(recent_topics)}")
        if conv_history.get("successful_engagement_types") and len(conv_history['successful_engagement_types']) > 0:
            summary_lines.append(f"Engagement types that work well: {', '.join(conv_history['successful_engagement_types'][:3])}")

        # Metadata
        metadata = self.profile.get("system_metadata", {})
        if metadata.get("total_conversations", 0) > 0:
            summary_lines.append(f"Total conversations: {metadata['total_conversations']}")
        if metadata.get("last_conversation"):
            summary_lines.append(f"Last conversation: {metadata['last_conversation']}")

        if len(summary_lines) == 0:
            return "**NEW DRIVER** - No profile information yet. This is your first conversation! Learn about the driver by asking friendly questions."
        else:
            return "\n".join(summary_lines)

    # ── Conversation transcript tracking ──────────────────────────

    def add_to_transcript(self, role, text):
        """Add a message to the conversation transcript for post-session extraction."""
        self.conversation_transcript.append((role, text))

    # ── LLM-based post-session extraction ─────────────────────────

    def _get_groq_client(self):
        """Lazy-init Groq client (only needed at end of conversation)."""
        if self._groq_client is None:
            self._groq_client = Groq(api_key=Config.GROQ_API_KEY)
        return self._groq_client

    def extract_and_apply_learnings(self):
        """Use an LLM to extract personal facts from the full conversation transcript.

        Sends the transcript to a small/fast model (8B) with a structured extraction
        prompt. Returns JSON with categorized facts, then merges into the profile.
        Much more accurate than regex — catches indirect mentions, context, nuance.
        """
        if not self.conversation_transcript:
            print("ℹ️ No conversation transcript to extract from")
            return 0

        # Build the transcript string
        transcript = "\n".join(
            f"{'DRIVER' if role == 'user' else 'SENTINEL'}: {text}"
            for role, text in self.conversation_transcript
        )

        extraction_prompt = f"""Analyze this conversation between a drowsy driver and an AI assistant called Sentinel. Extract any personal facts the DRIVER revealed about themselves.

CONVERSATION:
{transcript}

Return ONLY valid JSON with these fields (use null for unknown, [] for empty lists):
{{
  "name": "driver's name or null",
  "occupation": "their job/profession or null",
  "family": ["family members mentioned"],
  "location": "where they live or null",
  "engagement_preference": "conversation or actions or null",
  "hobbies": ["hobbies/interests mentioned"],
  "sports_teams": ["sports teams mentioned"],
  "music_preferences": ["music preferences mentioned"],
  "destinations": ["places they're going to or frequently visit"],
  "topics_discussed": ["2-3 word summary of main topics"],
  "engagement_that_worked": ["what types of engagement seemed to work well"]
}}

Rules:
- Only include facts the DRIVER explicitly stated or clearly implied
- Do NOT include things Sentinel said or assumed
- Keep values short and clear
- Return ONLY the JSON, no other text"""

        try:
            client = self._get_groq_client()
            response = client.chat.completions.create(
                model=Config.GROQ_EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": "You extract structured personal information from conversations. Return only valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,  # Low temp for consistent structured output
                max_tokens=500,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences (handles ```json ... ``` and ``` ... ```)
            if raw.startswith("```"):
                # Remove opening fence and optional language tag
                raw = raw[3:]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
                # Remove closing fence
                if raw.endswith("```"):
                    raw = raw[:-3]
            raw = raw.strip()

            extracted = json.loads(raw)
            return self._apply_extracted_facts(extracted)

        except json.JSONDecodeError as e:
            print(f"⚠️ LLM extraction returned invalid JSON: {e}")
            return 0
        except Exception as e:
            print(f"⚠️ LLM extraction failed: {e}")
            return 0

    def _apply_extracted_facts(self, extracted):
        """Merge extracted facts into the profile."""
        count = 0

        # Personal fields (overwrite)
        for field in ["name", "occupation", "location"]:
            val = extracted.get(field)
            if val and val != "null" and isinstance(val, str):
                self.profile["personal"][field] = val.strip().title() if field == "name" else val.strip()
                count += 1

        # Family (append)
        for member in extracted.get("family", []):
            if member and isinstance(member, str) and member not in self.profile["personal"]["family"]:
                self.profile["personal"]["family"].append(member)
                count += 1

        # Engagement preference
        pref = extracted.get("engagement_preference")
        if pref and pref != "null" and isinstance(pref, str):
            pref_lower = pref.lower().strip()
            if "conversation" in pref_lower or "chat" in pref_lower or "question" in pref_lower:
                self.profile["preferences"]["engagement_style"] = "conversation"
                count += 1
            elif "action" in pref_lower or "exercise" in pref_lower or "physical" in pref_lower:
                self.profile["preferences"]["engagement_style"] = "actions"
                count += 1

        # List fields (append, deduplicate, cap at 10)
        list_mappings = {
            "hobbies": ("interests", "hobbies"),
            "sports_teams": ("interests", "sports_teams"),
            "music_preferences": ("interests", "music_preferences"),
            "destinations": ("driving_patterns", "usual_destinations"),
            "topics_discussed": ("conversation_history", "last_topics"),
            "engagement_that_worked": ("conversation_history", "successful_engagement_types"),
        }

        for json_key, (category, profile_key) in list_mappings.items():
            values = extracted.get(json_key, [])
            if not isinstance(values, list):
                continue
            for val in values:
                if val and isinstance(val, str):
                    val = val.strip()
                    if val and val not in self.profile[category][profile_key]:
                        self.profile[category][profile_key].append(val)
                        # Cap list size
                        if len(self.profile[category][profile_key]) > 10:
                            self.profile[category][profile_key] = self.profile[category][profile_key][-10:]
                        count += 1

        if count > 0:
            self.save_profile()
            print(f"✓ LLM extraction: applied {count} facts to profile")
        else:
            print("ℹ️ LLM extraction: no new facts found")

        # Clear transcript
        self.conversation_transcript = []
        return count

    def log_conversation_metadata(self, metrics, conversation_length):
        """Log metadata about this conversation."""
        metadata = self.profile.get("system_metadata", {})
        metadata["total_conversations"] = metadata.get("total_conversations", 0) + 1
        metadata["last_conversation"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata["total_drowsy_episodes"] = metadata.get("total_drowsy_episodes", 0) + 1
        self.profile["system_metadata"] = metadata
        self.save_profile()
