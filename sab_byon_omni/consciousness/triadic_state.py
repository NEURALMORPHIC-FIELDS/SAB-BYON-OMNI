# -*- coding: utf-8 -*-
"""Triadic State System - Ontological-Semantic-Resonance Triad."""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class TriadicState:
    """
    Ontological-Semantic-Resonance Triad

    Dynamics:
    dO/dt = (φ - O) × κ
    dS/dt = (O - S) × λ
    Φ = exp(-|O - S|)
    """
    ontological: float = 0.5
    semantic: float = 0.5
    resonance: float = 1.0

    def evolve(self, field_mean: float, curvature: float, dt: float = 0.01):
        """Update triadic state based on field dynamics and geometry."""
        self.ontological += dt * (field_mean - self.ontological) * (curvature + 0.1)
        self.semantic += dt * (self.ontological - self.semantic) * 0.5
        self.resonance = np.exp(-abs(self.ontological - self.semantic))
        self.ontological = np.clip(self.ontological, 0, 1)
        self.semantic = np.clip(self.semantic, 0, 1)

    def consciousness_contribution(self) -> float:
        """C_local = (O + S) × Φ / 2"""
        return (self.ontological + self.semantic) * self.resonance * 0.5

    def to_dict(self) -> Dict:
        return {
            'O': self.ontological,
            'S': self.semantic,
            'R': self.resonance
        }
