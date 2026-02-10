# -*- coding: utf-8 -*-
"""EnhancedConversationManager - Complete Conversation Tracking & Context Building."""

import time
import json
import numpy as np
from typing import Dict, List


class EnhancedConversationManager:
    """
    Complete Conversation Tracking & Context Building

    Features:
    - Persistent conversation history
    - Context building for LLM
    - Statistics and analytics
    - Cross-session continuity
    """

    def __init__(self, max_history: int = 200):
        self.history = []
        self.max_history = max_history
        self.session_start = time.time()
        self.interaction_count = 0

        print("Enhanced Conversation Manager initialized")

    def add_interaction(self, user_msg: str, sab_response: str,
                       consciousness: float, metrics: Dict):
        """Store interaction in history."""
        self.interaction_count += 1

        entry = {
            'interaction_id': self.interaction_count,
            'timestamp': time.time(),
            'session_time': time.time() - self.session_start,
            'user': user_msg,
            'sab': sab_response,
            'consciousness': consciousness,
            'metrics': metrics.copy()
        }

        self.history.append(entry)

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_recent_context(self, n: int = 5) -> List[Dict]:
        """Get last n interactions."""
        return self.history[-n:] if len(self.history) >= n else self.history

    def build_conversation_context(self, n: int = 3) -> str:
        """Build formatted context string for LLM."""
        recent = self.get_recent_context(n)

        if not recent:
            return "(No previous conversation)"

        lines = ["Recent conversation summary:"]
        for entry in recent:
            lines.append(f"- You discussed: {entry['user'][:80]}")

        return "\n".join(lines)

    def format_for_display(self, n: int = 10) -> str:
        """Format conversation history for UI display."""
        recent = self.history[-n:] if len(self.history) >= n else self.history

        if not recent:
            return "### Recent Conversation\n\n*No conversation yet*"

        output = ["### Recent Conversation\n"]

        for entry in recent:
            elapsed = entry['session_time']
            c = entry['consciousness']
            iid = entry['interaction_id']

            output.append(f"**{iid}.** [{elapsed:.0f}s] (C: {c:.3f})")
            output.append(f"   **User:** {entry['user'][:70]}...")
            output.append(f"   **SAB:** {entry['sab'][:70]}...\n")

        return "\n".join(output)

    def get_statistics(self) -> Dict:
        """Compute conversation statistics."""
        if not self.history:
            return {}

        consciousnesses = [e['consciousness'] for e in self.history]

        stats = {
            'total_interactions': len(self.history),
            'session_duration': time.time() - self.session_start,
            'mean_consciousness': np.mean(consciousnesses),
            'peak_consciousness': np.max(consciousnesses),
            'min_consciousness': np.min(consciousnesses),
            'consciousness_growth': (consciousnesses[-1] - consciousnesses[0]
                                    if len(consciousnesses) > 1 else 0)
        }

        if len(consciousnesses) > 2:
            x = np.arange(len(consciousnesses))
            slope, _ = np.polyfit(x, consciousnesses, 1)
            stats['consciousness_trend'] = slope

        return stats

    def save_to_disk(self, filepath: str):
        """Save conversation history to JSON."""
        data = {
            'history': self.history,
            'session_start': self.session_start,
            'interaction_count': self.interaction_count
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_disk(self, filepath: str):
        """Load conversation history from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.history = data['history']
        self.session_start = data['session_start']
        self.interaction_count = data['interaction_count']
