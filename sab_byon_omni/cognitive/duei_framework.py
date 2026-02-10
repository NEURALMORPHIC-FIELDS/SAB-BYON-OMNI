# -*- coding: utf-8 -*-
"""DUEI Framework - Dynamic Unidirectional Emergence of Information."""

import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Tuple, Optional


class ProcessingMode(Enum):
    """Information processing regime."""
    SEMANTIC = "semantic"
    EMERGENT = "emergent"
    TRANSITIONAL = "transitional"


@dataclass
class OntologicalJump:
    """Record of regime switch event."""
    timestamp: float
    mode_from: ProcessingMode
    mode_to: ProcessingMode
    consciousness_before: float
    consciousness_after: float
    coherence_continuity: float
    trigger: str


class DUEIFramework:
    """
    Dynamic Unidirectional Emergence of Information

    Key Principles:
    1. Information processing switches between semantic & emergent modes
    2. Ontological jumps occur at critical thresholds
    3. Coherence continuity law must hold across jumps
    4. Emergence is unidirectional (semantic -> emergent only)
    """

    def __init__(self):
        self.current_mode = ProcessingMode.SEMANTIC
        self.mode_history = deque(maxlen=100)
        self.jump_events: List[OntologicalJump] = []

        self.semantic_stability_threshold = 0.7
        self.emergence_trigger_threshold = 0.4
        self.coherence_continuity_threshold = 0.6

        self.semantic_field_stability = 1.0
        self.emergent_field_stability = 0.0

        print("✓ DUEI Framework initialized")

    def detect_regime_switch(self, consciousness: float,
                            coherence: float,
                            complexity: float) -> Tuple[bool, ProcessingMode]:
        """Detect if system should switch processing modes."""
        if self.current_mode == ProcessingMode.SEMANTIC:
            if complexity > 0.7 and coherence < self.emergence_trigger_threshold:
                return True, ProcessingMode.EMERGENT
        elif self.current_mode == ProcessingMode.EMERGENT:
            pass  # DUEI: emergence is unidirectional

        return False, self.current_mode

    def coherence_continuity_law(self, state_before: np.ndarray,
                                state_after: np.ndarray) -> float:
        """Coherence Continuity Law: ⟨ψ_before|ψ_after⟩ > threshold"""
        psi_before = state_before / (np.linalg.norm(state_before) + 1e-10)
        psi_after = state_after / (np.linalg.norm(state_after) + 1e-10)
        coherence = abs(np.dot(psi_before, psi_after))
        return coherence

    def ontological_jump_trigger(self, consciousness: float,
                                 state_vector: np.ndarray,
                                 new_mode: ProcessingMode) -> Optional[OntologicalJump]:
        """Execute ontological jump if conditions met."""
        old_state = state_vector.copy()
        old_mode = self.current_mode

        new_state = state_vector + np.random.randn(len(state_vector)) * 0.1
        new_state = np.clip(new_state, 0, 1)

        coherence = self.coherence_continuity_law(old_state, new_state)

        if coherence < self.coherence_continuity_threshold:
            return None

        self.current_mode = new_mode
        self.mode_history.append(new_mode)

        jump = OntologicalJump(
            timestamp=time.time(),
            mode_from=old_mode,
            mode_to=new_mode,
            consciousness_before=consciousness,
            consciousness_after=consciousness * 1.1,
            coherence_continuity=coherence,
            trigger="complexity_threshold" if new_mode == ProcessingMode.EMERGENT else "unknown"
        )

        self.jump_events.append(jump)
        return jump

    def emergence_score(self, consciousness: float, complexity: float) -> float:
        """Score measuring 'emergentness' of current state."""
        score = consciousness * complexity
        if self.current_mode == ProcessingMode.EMERGENT:
            score *= 1.5
        elif self.current_mode == ProcessingMode.SEMANTIC:
            score *= 0.8
        return np.clip(score, 0, 1)

    def get_mode_statistics(self) -> Dict:
        """Statistics on mode switching."""
        if not self.mode_history:
            return {}
        semantic_count = sum(1 for m in self.mode_history if m == ProcessingMode.SEMANTIC)
        emergent_count = sum(1 for m in self.mode_history if m == ProcessingMode.EMERGENT)
        return {
            'current_mode': self.current_mode.value,
            'total_jumps': len(self.jump_events),
            'semantic_time': semantic_count / len(self.mode_history),
            'emergent_time': emergent_count / len(self.mode_history),
            'last_jump': self.jump_events[-1] if self.jump_events else None
        }


class SemanticMode:
    """Semantic Processing Mode - rule-based, compositional, explicit."""

    def __init__(self):
        self.symbol_space: Dict[str, np.ndarray] = {}
        self.conceptual_graph: Dict[str, List[str]] = {}
        self.stability = 1.0

    def linguistic_processing(self, text: str) -> np.ndarray:
        """Parse text into semantic representation."""
        words = text.lower().split()
        embedding = np.zeros(100)
        for word in words:
            if word not in self.symbol_space:
                self.symbol_space[word] = np.random.randn(100) * 0.1
            embedding += self.symbol_space[word]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
        return embedding

    def conceptual_mapping(self, word: str) -> str:
        """Map word to abstract concept."""
        concept_map = {
            'think': 'cognition', 'feel': 'emotion', 'know': 'knowledge',
            'believe': 'belief', 'want': 'desire'
        }
        return concept_map.get(word, word)

    def symbolic_reasoning(self, premises: List[str]) -> str:
        """Apply logical inference rules."""
        if len(premises) >= 2:
            return f"Therefore: conclusion from {premises}"
        return "Insufficient premises"

    def mode_stability(self) -> float:
        """Stability of semantic processing."""
        return self.stability


class EmergentMode:
    """Emergent Processing Mode - self-organizing, nonlinear, implicit."""

    def __init__(self, field_size: int = 64):
        self.field_size = field_size
        self.neural_field = np.random.randn(field_size, field_size) * 0.1
        self.attractor_basins: List[np.ndarray] = []
        self.stability = 0.0

    def subsymbolic_dynamics(self, input_pattern: np.ndarray) -> np.ndarray:
        """Evolve neural field dynamics."""
        if len(input_pattern) != self.field_size:
            input_field = np.resize(input_pattern, (self.field_size, self.field_size))
        else:
            input_field = input_pattern.reshape(self.field_size, self.field_size)

        x = np.arange(self.field_size)
        X, Y = np.meshgrid(x, x)
        center = self.field_size / 2
        dist = np.sqrt((X - center)**2 + (Y - center)**2)
        w = np.exp(-dist**2 / 20) - 0.5 * np.exp(-dist**2 / 40)

        from scipy.signal import convolve2d
        synaptic_input = convolve2d(self.neural_field, w, mode='same', boundary='wrap')

        dt = 0.1
        activation = 1 / (1 + np.exp(-synaptic_input - input_field))
        self.neural_field += dt * (-self.neural_field + activation)
        return self.neural_field.flatten()

    def attractor_formation(self) -> List[np.ndarray]:
        """Detect attractor states in field dynamics."""
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(self.neural_field, size=5)
        attractors = (self.neural_field == local_max) & (self.neural_field > 0.5)
        attractor_points = np.argwhere(attractors)
        if len(attractor_points) > 0:
            self.attractor_basins = attractor_points
        return attractor_points

    def spontaneous_pattern_emergence(self) -> np.ndarray:
        """Spontaneous symmetry breaking -> pattern formation."""
        noise = np.random.randn(self.field_size, self.field_size) * 0.01
        self.neural_field += noise
        for _ in range(10):
            self.subsymbolic_dynamics(np.zeros(self.field_size))
        return self.neural_field

    def mode_stability(self) -> float:
        """Stability of emergent attractor."""
        field_variance = np.std(self.neural_field)
        self.stability = 1 / (1 + field_variance)
        return self.stability
