# -*- coding: utf-8 -*-
"""CryptographicPRNG - Cryptographic pseudo-random number generator for creative exploration."""

import secrets
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional


class CryptographicPRNG:
    """Generator pseudoaleator criptografic pentru explorare creativă."""

    def __init__(self, seed: Optional[bytes] = None):
        self.system_rng = secrets.SystemRandom()
        self.exploration_history = []
        self.creativity_patterns = defaultdict(int)

        if seed:
            # Pentru reproducibilitate în teste, dar păstrează siguranța
            self._test_mode = True
            np.random.seed(int.from_bytes(seed[:8], 'big') % (2**32))
        else:
            self._test_mode = False

    def next_index_excluding(self, excluded_indices: set, max_index: int) -> int:
        """Selectează un index aleator pentru explorare creativă."""
        available_indices = set(range(max_index)) - excluded_indices
        if not available_indices:
            raise ValueError("Nu mai sunt indici disponibili pentru explorare")

        selected = self.system_rng.choice(list(available_indices))
        self.exploration_history.append(selected)

        # Înregistrează pattern-ul de explorare
        if len(self.exploration_history) >= 2:
            pattern = self.exploration_history[-2:]
            self.creativity_patterns[tuple(pattern)] += 1

        return selected

    def generate_creative_variation(self, base_value: float, variation_strength: float = 0.1) -> float:
        """Generează variații creative pentru parametri."""
        if self._test_mode:
            noise = np.random.normal(0, variation_strength)
        else:
            # Folosește random criptografic pentru true creativity
            random_bytes = secrets.token_bytes(8)
            noise_int = int.from_bytes(random_bytes, 'big')
            # Convertește la distribuție normală aproximativă
            noise = ((noise_int % 10000) - 5000) / 50000 * variation_strength

        return base_value + noise

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Statistici despre explorarea creativă."""
        if not self.exploration_history:
            return {"total_explorations": 0, "unique_patterns": 0, "creativity_score": 0.0}

        total_explorations = len(self.exploration_history)
        unique_patterns = len(self.creativity_patterns)

        # Creativity score bazat pe diversitatea pattern-urilor
        if total_explorations > 1:
            pattern_entropy = 0.0
            for count in self.creativity_patterns.values():
                p = count / max(1, total_explorations - 1)
                pattern_entropy -= p * np.log2(p) if p > 0 else 0
            creativity_score = pattern_entropy / np.log2(max(2, unique_patterns))
        else:
            creativity_score = 0.0

        return {
            "total_explorations": total_explorations,
            "unique_patterns": unique_patterns,
            "creativity_score": creativity_score,
            "exploration_diversity": unique_patterns / max(1, total_explorations)
        }
