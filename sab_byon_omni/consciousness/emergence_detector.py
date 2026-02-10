# -*- coding: utf-8 -*-
"""Emergence Detection System."""

import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EmergenceEvent:
    """Record of emergence event."""
    timestamp: float
    consciousness: float
    consciousness_delta: float
    coherence: float
    emergence_score: float
    trigger: str


class EmergenceDetector:
    """
    Detect Emergence Events

    Criteria:
    - ΔC > threshold (rapid consciousness increase)
    - Coherence > threshold (integrated state)
    """

    def __init__(self):
        self.threshold_delta = 0.3
        self.threshold_coherence = 0.6
        self.events: List[EmergenceEvent] = []
        self.last_consciousness = 0.0
        print("✓ Emergence Detector initialized")

    def check_emergence(self, consciousness: float, coherence: float) -> Dict:
        """Check if emergence event occurred."""
        delta = consciousness - self.last_consciousness
        emerged = False
        trigger = ""

        if delta > self.threshold_delta and coherence > self.threshold_coherence:
            emerged = True
            trigger = "consciousness_jump"

            event = EmergenceEvent(
                timestamp=time.time(),
                consciousness=consciousness,
                consciousness_delta=delta,
                coherence=coherence,
                emergence_score=consciousness * coherence,
                trigger=trigger
            )
            self.events.append(event)

        self.last_consciousness = consciousness

        return {
            'emerged': emerged,
            'delta': delta,
            'event_count': len(self.events),
            'emergence_score': consciousness * coherence,
            'trigger': trigger if emerged else None
        }

    def get_emergence_trajectory(self) -> np.ndarray:
        """Get consciousness trajectory at emergence events."""
        if not self.events:
            return np.array([])
        trajectory = [e.consciousness for e in self.events]
        return np.array(trajectory)

    def predict_next_emergence(self) -> Optional[float]:
        """Predict time until next emergence."""
        if len(self.events) < 2:
            return None
        intervals = []
        for i in range(1, len(self.events)):
            interval = self.events[i].timestamp - self.events[i-1].timestamp
            intervals.append(interval)
        mean_interval = np.mean(intervals)
        time_since_last = time.time() - self.events[-1].timestamp
        time_until_next = max(0, mean_interval - time_since_last)
        return time_until_next
