import json
from typing import List, Dict, Any

class HybridMemory:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.messages: List[Dict[str, str]] = []
        self.summary: str = ""

    def add_message(self, role: str, content: str, api_key: str = None):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.window_size:
            # Keep only last window_size messages
            self.messages = self.messages[-self.window_size:]

    def get_formatted_context(self) -> str:
        if not self.messages:
            return ""
        context = "Recent conversation:\n"
        for msg in self.messages:
            context += f"{msg['role'].capitalize()}: {msg['content']}\n"
        return context

    def clear(self):
        self.messages = []
        self.summary = ""