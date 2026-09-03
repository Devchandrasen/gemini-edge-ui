"""Antigravity Main Entry Point.

Registers and coordinates the ComputerUseSkill and routes incoming user
commands through the local offline agent pipeline.
"""

from __future__ import annotations

import os
import sys

# Ensure module path resolution for local skills and core packages
pkg_root = os.path.dirname(os.path.abspath(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from skills.computer_use_skill import ComputerUseSkill


class AntigravityAgent:
    """Main coordinator registering and activating the PC Control Engine."""

    def __init__(self, local_vlm_url: str = "http://localhost:8000/v1") -> None:
        # Local model endpoint define karein
        self.computer_skill = ComputerUseSkill(local_vlm_url=local_vlm_url)

    def handle_user_command(self, user_query: str) -> None:
        """Handle incoming user objective."""
        print(f"\n[Antigravity] Processing command: '{user_query}'")

        # Agar user command system ya screen action se related hai
        action_keywords = ["open", "click", "type", "control", "pc", "show me", "search"]
        if any(keyword in user_query.lower() for keyword in action_keywords):
            print("System Trigger: Activating Antigravity PC Control Engine...")

            # Local GUI tool execute karein
            self.computer_skill.execute_skill(objective=user_query)
        else:
            # Normal text generation logic
            print("Conversation mode: Standard chat reply here.")


if __name__ == "__main__":
    # Agent instantiate karein
    agent = AntigravityAgent()

    # Agar CLI argument diya hai to usko use karein, warna default query
    query = sys.argv[1] if len(sys.argv) > 1 else "Click on the Google Chrome search bar on my desktop"
    agent.handle_user_command(query)
