import os
import json
from typing import Dict, Any, List

class ProfileTool:
    def __init__(self, db, data_dir: str = None):
        self.db = db
        if data_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(project_root, "data")
        else:
            self.data_dir = data_dir
        self.file_path = os.path.join(self.data_dir, "profile.json")
        self.profile: Dict[str, str] = {}
        self._load_profile()

    def _load_profile(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.profile = json.load(f)
            except:
                self.profile = {}
        else:
            self.profile = {}
            self._save_profile()

    def _save_profile(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.profile, f, indent=4)

    def store_fact(self, key: str, value: str) -> Dict[str, Any]:
        """Store a personal fact with key normalization."""
        key_map = {
            "favorite food": "favourite food",
            "favourite food": "favourite food",
            "pet": "pet name",
            "dog": "pet name",
            "cat": "pet name",
            "college": "college",
            "university": "college",
            "school": "college",
            "name": "name",
            "city": "city",
            "hometown": "hometown",
            "birthday": "birthday",
            "profession": "profession",
            "job": "profession",
            "mother name": "mother name",
            "father name": "father name",
            "hobbies": "hobbies",
            "hobby": "hobbies"
        }
        clean_key = key_map.get(key.strip().lower(), key.strip().lower())
        clean_val = value.strip()
        self.profile[clean_key] = clean_val
        self._save_profile()
        # Index in vector DB for semantic search
        text = f"User Profile Fact: my {clean_key} is {clean_val}"
        self.db.add_texts([text], [{"type": "profile", "key": clean_key, "value": clean_val}])
        return {"success": True, "key": clean_key, "value": clean_val}

    def retrieve_fact(self, key: str) -> str:
        """Retrieve a fact value by key."""
        return self.profile.get(key.strip().lower(), "")

    def get_all_facts(self) -> Dict[str, str]:
        """Return all non-empty profile facts."""
        return {k: v for k, v in self.profile.items() if v and isinstance(v, str)}

    def get_formatted_context(self) -> str:
        """Format facts for system context."""
        facts = self.get_all_facts()
        if not facts:
            return ""
        lines = [f"- User's {k}: {v}" for k, v in facts.items()]
        return "User Profile Facts:\n" + "\n".join(lines)

    def clear(self):
        self.profile = {}
        self._save_profile()