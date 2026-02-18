"""
Memory Manager for Driver Drowsiness Detection System V3.2
Manages persistent driver profile and conversation history.
"""

import json
import time
from pathlib import Path


class MemoryManager:
    """Manages persistent driver profile and conversation history."""

    def __init__(self, profile_path="sentinel_driver_profile.json"):
        self.profile_path = Path.home() / profile_path
        self.profile = self.load_profile()
        self.current_session_learnings = []  # Track learnings during this conversation

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
        """Update a specific field in the profile."""
        if category not in self.profile:
            self.profile[category] = {}
        self.profile[category][key] = value
        self.save_profile()

    def append_to_list(self, category, key, value, max_items=10):
        """Append to a list field, maintaining max size."""
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
        self.save_profile()

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

    def extract_learnings_from_text(self, user_text, assistant_text=None):
        """Simple pattern-based extraction of information from conversation."""
        learnings = []
        text_lower = user_text.lower()

        # Extract name
        if "my name is" in text_lower or "i'm" in text_lower or "call me" in text_lower:
            if "my name is " in text_lower:
                potential_name = user_text.split("my name is ", 1)[1].split()[0].strip(".,!?")
                if len(potential_name) < 20 and potential_name.isalpha():
                    learnings.append(("personal", "name", potential_name.title()))

        # Extract occupation
        if "i work as" in text_lower or "i'm a " in text_lower or "i am a " in text_lower:
            if "i work as " in text_lower:
                occupation = user_text.lower().split("i work as ", 1)[1].split()[0].strip(".,!?")
                learnings.append(("personal", "occupation", occupation))

        # Extract interests/hobbies
        hobby_phrases = ["i like", "i love", "i enjoy", "i'm into", "my hobby"]
        for phrase in hobby_phrases:
            if phrase in text_lower:
                interest = user_text.lower().split(phrase, 1)[1].strip().split('.')[0].strip()
                if len(interest) < 50:
                    learnings.append(("interests", "hobbies", interest))

        # Extract destination/route info
        if "going to" in text_lower or "heading to" in text_lower or "driving to" in text_lower:
            for phrase in ["going to", "heading to", "driving to"]:
                if phrase in text_lower:
                    destination = user_text.lower().split(phrase, 1)[1].strip().split('.')[0].split(',')[0].strip()
                    if len(destination) < 30:
                        learnings.append(("driving_patterns", "usual_destinations", destination))

        # Extract engagement style preference
        conversation_keywords = ["conversation", "talk", "chat", "questions", "ask me"]
        action_keywords = ["actions", "exercises", "physical", "suggestions", "tips", "stories", "story", "tell me"]
        if any(kw in text_lower for kw in conversation_keywords):
            if any(word in text_lower for word in ["prefer", "like", "want", "love", "enjoy", "rather", "go with", "let's do", "choose"]):
                learnings.append(("preferences", "engagement_style", "conversation"))
        if any(kw in text_lower for kw in action_keywords):
            if any(word in text_lower for word in ["prefer", "like", "want", "love", "enjoy", "rather", "go with", "let's do", "choose"]):
                learnings.append(("preferences", "engagement_style", "actions"))

        # Store learnings for this session
        self.current_session_learnings.extend(learnings)

        return learnings

    def apply_session_learnings(self):
        """Apply all learnings from this conversation session."""
        count = len(self.current_session_learnings)
        for category, key, value in self.current_session_learnings:
            if key in ["hobbies", "sports_teams", "music_preferences", "topics_engaged_with",
                       "common_routes", "usual_destinations", "last_topics", "successful_engagement_types"]:
                self.append_to_list(category, key, value)
            elif key in ["family"]:
                if value not in self.profile[category][key]:
                    self.profile[category][key].append(value)
            else:
                self.update_field(category, key, value)

        # Clear session learnings
        self.current_session_learnings = []
        print(f"✓ Applied {count} learnings to profile")

    def log_conversation_metadata(self, metrics, conversation_length):
        """Log metadata about this conversation."""
        metadata = self.profile.get("system_metadata", {})
        metadata["total_conversations"] = metadata.get("total_conversations", 0) + 1
        metadata["last_conversation"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata["total_drowsy_episodes"] = metadata.get("total_drowsy_episodes", 0) + 1
        self.profile["system_metadata"] = metadata
        self.save_profile()
