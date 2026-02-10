# -*- coding: utf-8 -*-
"""10-Dimensional Personality Evolution System."""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class PersonalitySystem:
    """
    10-Dimensional Personality Evolution

    Traits: Big Five + Cognitive Style
    Evolution: trait(t+1) = α·trait(t) + (1-α)·virtue_influence
    """

    def __init__(self):
        self.traits = {
            'conscientiousness': 0.5, 'openness': 0.5, 'extraversion': 0.5,
            'agreeableness': 0.5, 'neuroticism': 0.5,
            'analytical': 0.5, 'creative': 0.5, 'empathetic': 0.5,
            'philosophical': 0.5, 'reflective': 0.5
        }

        self.evolution_history = deque(maxlen=200)

        self.virtue_trait_map = {
            'stoicism': 'conscientiousness', 'discernment': 'analytical',
            'philosophy': 'philosophical', 'empathy': 'empathetic',
            'curiosity': 'openness', 'humility': 'agreeableness',
            'creativity': 'creative', 'reflexivity': 'reflective',
            'truthlove': 'conscientiousness', 'holographic': 'openness'
        }

        print("✓ Personality System initialized (10 traits)")

    def evolve_traits(self, virtue_states: Dict[str, float], consciousness: float):
        """Update personality traits based on virtue activations."""
        alpha = 0.7 + 0.2 * (1 - consciousness)

        for virtue, trait in self.virtue_trait_map.items():
            if virtue in virtue_states and trait in self.traits:
                virtue_activation = virtue_states[virtue]
                self.traits[trait] = (alpha * self.traits[trait] +
                                     (1 - alpha) * virtue_activation)

        total = sum(self.traits.values())
        if total > 0:
            self.traits = {k: v/total for k, v in self.traits.items()}

        self.evolution_history.append(self.traits.copy())

    def get_dominant_traits(self, k: int = 3) -> List[Tuple[str, float]]:
        """Get top k dominant traits."""
        return sorted(self.traits.items(), key=lambda x: x[1], reverse=True)[:k]

    def get_personality_vector(self) -> np.ndarray:
        """10D personality state vector."""
        return np.array(list(self.traits.values()))

    def compute_personality_stability(self) -> float:
        """Measure personality stability over time."""
        if len(self.evolution_history) < 10:
            return 1.0

        recent = list(self.evolution_history)[-10:]
        trait_names = list(self.traits.keys())
        matrix = np.array([[snapshot[t] for t in trait_names] for snapshot in recent])
        variance = np.var(matrix, axis=0).mean()
        stability = 1.0 / (1.0 + variance)
        return stability

    def get_trait_trajectory(self, trait: str) -> np.ndarray:
        """Get evolution trajectory for specific trait."""
        if trait not in self.traits:
            return np.array([])
        trajectory = [snapshot[trait] for snapshot in self.evolution_history if trait in snapshot]
        return np.array(trajectory)
