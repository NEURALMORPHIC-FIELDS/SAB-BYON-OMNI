# -*- coding: utf-8 -*-
"""Fragmergent Cycle Detection & Management."""

import numpy as np
from collections import deque
from typing import Dict, List


class FragmergentEngine:
    """
    Fragmergent Cycle Detection & Management

    Tracks system oscillation between:
    - Fragmentation: Low consciousness, high entropy, distributed
    - Emergence: High consciousness, low entropy, integrated
    """

    def __init__(self):
        self.phase = "emergence"
        self.cycle_count = 0
        self.coherence_history = deque(maxlen=100)
        self.phase_history = deque(maxlen=100)

        self.emergence_consciousness_threshold = 0.4
        self.emergence_coherence_threshold = 0.6
        self.fragmentation_consciousness_threshold = 0.3
        self.fragmentation_coherence_threshold = 0.4

        print("✓ Fragmergent Engine initialized")

    def detect_phase(self, consciousness: float, emergence_score: float) -> str:
        """Detect current fragmergent phase."""
        if (consciousness > self.emergence_consciousness_threshold and
            emergence_score > self.emergence_coherence_threshold):
            new_phase = "emergence"
        elif (consciousness < self.fragmentation_consciousness_threshold or
              emergence_score < self.fragmentation_coherence_threshold):
            new_phase = "fragmentation"
        else:
            new_phase = "transition"

        if new_phase != self.phase:
            self.cycle_count += 1

        self.phase = new_phase
        self.phase_history.append(new_phase)
        return new_phase

    def compute_emergence_score(self, virtue_states: Dict[str, float]) -> float:
        """Emergence score from virtue harmony."""
        values = list(virtue_states.values())
        if not values:
            return 0.0
        mean_val = np.mean(values)
        std_val = np.std(values)
        score = (1.0 - std_val / (mean_val + 1e-8)) if mean_val > 0 else 0.0
        return float(np.clip(score, 0, 1))

    def compute_coherence(self, triadic_resonances: List[float]) -> float:
        """System-wide coherence from triadic resonances."""
        if not triadic_resonances:
            return 0.0
        coherence = np.mean(triadic_resonances)
        self.coherence_history.append(coherence)
        return coherence

    def get_cycle_statistics(self) -> Dict:
        """Fragmergent cycle statistics."""
        if not self.phase_history:
            return {}
        emergence_count = sum(1 for p in self.phase_history if p == "emergence")
        fragmentation_count = sum(1 for p in self.phase_history if p == "fragmentation")
        return {
            'current_phase': self.phase,
            'total_cycles': self.cycle_count,
            'emergence_ratio': emergence_count / len(self.phase_history),
            'fragmentation_ratio': fragmentation_count / len(self.phase_history),
            'mean_coherence': np.mean(self.coherence_history) if self.coherence_history else 0
        }
