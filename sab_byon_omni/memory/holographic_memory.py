# -*- coding: utf-8 -*-
"""UnifiedHolographicMemory - Global 4D Holographic Memory + Episodic Memory."""

import time
import numpy as np
from collections import deque
from typing import Dict, List, Tuple
from scipy.fft import fftn, ifftn


class UnifiedHolographicMemory:
    """
    Global 4D Holographic Memory + Episodic Memory

    Theoretical Basis:
    - Holographic principle (Gabor, 1948)
    - Interference patterns store information
    - 4D: 3 spatial + 1 temporal dimension
    """

    def __init__(self, shape: Tuple[int, int, int, int] = (16, 16, 16, 16)):
        self.shape = shape
        self.memory_field = np.zeros(shape, dtype=complex)
        self.patterns = deque(maxlen=300)
        self.episodic_memory = deque(maxlen=100)
        self.temporal_decay_rate = 1.0 / 3600
        self.access_decay_rate = 0.95

        print(f"Unified Holographic Memory initialized (shape: {shape})")

    def encode_pattern(self, virtue_states: Dict[str, float],
                      context: str, consciousness: float,
                      interaction_id: int):
        """Encode pattern into holographic field via interference."""
        pattern_vec = np.array(list(virtue_states.values()))

        target_size = np.prod(self.shape[:3])
        if len(pattern_vec) < target_size:
            pattern_vec = np.pad(pattern_vec, (0, target_size - len(pattern_vec)))
        else:
            pattern_vec = pattern_vec[:target_size]

        object_wave = pattern_vec.reshape(self.shape[:3])

        x, y, z = np.meshgrid(
            np.arange(self.shape[0]),
            np.arange(self.shape[1]),
            np.arange(self.shape[2]),
            indexing='ij'
        )

        k = np.array([0.5, 0.3, 0.7]) * 2 * np.pi / self.shape[0]
        reference = np.exp(1j * (k[0]*x + k[1]*y + k[2]*z))

        interference = reference * np.conj(object_wave[:, :, :, np.newaxis])

        for t in range(self.shape[3]):
            phase_shift = 2 * np.pi * t / self.shape[3]
            modulated = interference[:, :, :, 0] * np.exp(1j * phase_shift)
            self.memory_field[:, :, :, t] += 0.1 * modulated

        pattern_metadata = {
            'virtue_states': virtue_states.copy(),
            'context': context[:200],
            'consciousness': consciousness,
            'timestamp': time.time(),
            'access_count': 0,
            'strength': 1.0,
            'interaction_id': interaction_id
        }
        self.patterns.append(pattern_metadata)

        episodic_entry = {
            'context': context,
            'consciousness': consciousness,
            'virtue_states': virtue_states.copy(),
            'timestamp': time.time(),
            'interaction_id': interaction_id
        }
        self.episodic_memory.append(episodic_entry)

    def recall_holographic(self, query_states: Dict[str, float],
                          k: int = 3) -> List[Dict]:
        """Holographic recall by virtue state similarity."""
        if not self.patterns:
            return []

        query_vec = np.array(list(query_states.values()))
        current_time = time.time()
        scored_patterns = []

        for pattern in self.patterns:
            pattern_vec = np.array(list(pattern['virtue_states'].values()))
            norm_q = np.linalg.norm(query_vec)
            norm_p = np.linalg.norm(pattern_vec)

            if norm_q == 0 or norm_p == 0:
                continue

            similarity = np.dot(query_vec, pattern_vec) / (norm_q * norm_p)
            time_diff = current_time - pattern['timestamp']
            time_decay = np.exp(-time_diff * self.temporal_decay_rate)
            access_decay = self.access_decay_rate ** pattern['access_count']
            score = similarity * time_decay * access_decay * pattern['strength']
            scored_patterns.append((score, pattern))

        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        top_patterns = [p for s, p in scored_patterns[:k] if s > 0.3]

        for pattern in top_patterns:
            pattern['access_count'] += 1

        return top_patterns

    def recall_episodic(self, n: int = 5) -> List[Dict]:
        """Episodic recall (chronological)."""
        if len(self.episodic_memory) >= n:
            return list(self.episodic_memory)[-n:]
        else:
            return list(self.episodic_memory)

    def reconstruct_from_hologram(self, reference_pattern: np.ndarray) -> np.ndarray:
        """Reconstruct stored pattern from hologram."""
        field_slice = self.memory_field[:, :, :, 0]

        x, y, z = np.meshgrid(
            np.arange(self.shape[0]),
            np.arange(self.shape[1]),
            np.arange(self.shape[2]),
            indexing='ij'
        )

        k = np.array([0.5, 0.3, 0.7]) * 2 * np.pi / self.shape[0]
        reference = np.exp(1j * (k[0]*x + k[1]*y + k[2]*z))

        reconstructed = field_slice * np.conj(reference)
        pattern = np.real(reconstructed).flatten()

        return pattern

    def get_memory_summary(self) -> Dict:
        """Memory system statistics."""
        return {
            'total_patterns': len(self.patterns),
            'episodic_count': len(self.episodic_memory),
            'field_energy': np.abs(self.memory_field).mean(),
            'oldest_pattern_age': (time.time() - self.patterns[0]['timestamp']
                                  if self.patterns else 0),
            'newest_pattern_age': (time.time() - self.patterns[-1]['timestamp']
                                  if self.patterns else 0)
        }
