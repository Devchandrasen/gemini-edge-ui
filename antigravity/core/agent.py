"""Antigravity Core Agent Engine.

Coordinates reasoning, conversational LLM handling, and routes actionable
system/UI requests to the ComputerUseSkill engine.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

# Ensure package paths resolve cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_root = os.path.abspath(os.path.join(current_dir, ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from skills.computer_use_skill import ComputerUseSkill


class AntigravityAgent:
    """Main Agent Engine coordinating LLM reasoning and physical PC control skills."""

    def __init__(
        self,
        local_vlm_url: str = "http://localhost:8000/v1",
        model_name: str = "qwen2.5-vl:7b",
    ) -> None:
        self.vlm_url = local_vlm_url
        self.model_name = model_name
        # Initialize the low-latency computer control skill
        self.computer_skill = ComputerUseSkill(local_vlm_url=local_vlm_url)

    def handle_user_command(self, user_query: str) -> None:
        """Process an incoming user request and route to appropriate skills or conversational replies."""
        print(f"\n[Antigravity Agent] Received Query: '{user_query}'")

        # Agar user command system ya screen action se related hai
        action_keywords = ["open", "click", "type", "control", "pc", "show me", "search"]
        if any(keyword in user_query.lower() for keyword in action_keywords):
            print("System Trigger: Activating Antigravity PC Control Engine...")
            # Local GUI tool execute karein
            self.computer_skill.execute_skill(objective=user_query)
        else:
            # Normal text generation logic
            print("Conversation mode: Standard chat reply here.")
            reply = self.generate_conversational_reply(user_query)
            print(f"[Agent Reply]: {reply}")

    def generate_conversational_reply(self, prompt: str) -> str:
        """Query local LLM endpoint for non-GUI conversational queries."""
        import requests

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
        }
        try:
            res = requests.post(f"{self.vlm_url}/chat/completions", json=payload, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass
        return f"Acknowledged: '{prompt}'. (Local LLM offline or responded via standard pipeline)"


if __name__ == "__main__":
    agent = AntigravityAgent()
    agent.handle_user_command("Click on the Google Chrome search bar on my desktop")
