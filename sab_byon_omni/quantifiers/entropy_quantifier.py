# -*- coding: utf-8 -*-
"""EntropyQuantifier - Shannon entropy based complexity measurement."""

import numpy as np
from collections import defaultdict
from typing import Dict, List

from sab_byon_omni.quantifiers.base_quantifier import BaseQuantifier


class EntropyQuantifier(BaseQuantifier):
    """Cuantificare pentru măsurarea complexității gândirii prin entropie Shannon."""

    def __init__(self, normalize_by_length: bool = True):
        self.token_counts = defaultdict(int)
        self.total_tokens = 0
        self.normalize_by_length = normalize_by_length
        self.entropy_history = []

    def initial_score(self) -> float:
        return 0.0

    def update_score(self, current_score: float, new_element: str,
                    current_subset: List[str]) -> float:
        """Calculează entropia Shannon pentru diversitatea tokens."""
        # Tokenizare simplă
        tokens = new_element.lower().split() if isinstance(new_element, str) else [str(new_element)]

        for token in tokens:
            self.token_counts[token] += 1
            self.total_tokens += 1

        if self.total_tokens == 0:
            return 0.0

        # Calculează entropia Shannon
        entropy = 0.0
        for count in self.token_counts.values():
            if count > 0:
                p = count / self.total_tokens
                entropy -= p * np.log2(p)

        # Normalizează prin lungime dacă e necesar
        if self.normalize_by_length:
            max_entropy = np.log2(len(self.token_counts)) if len(self.token_counts) > 1 else 1.0
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        self.entropy_history.append(entropy)
        return entropy

    def meets_threshold(self, score: float, threshold: float) -> bool:
        return score >= threshold  # Want high entropy (complexity)

    def get_diversity_metrics(self) -> Dict[str, float]:
        """Returnează metrici suplimentare de diversitate."""
        if not self.token_counts:
            return {"unique_tokens": 0, "repetition_rate": 1.0, "vocabulary_richness": 0.0}

        unique_tokens = len(self.token_counts)
        repetition_rate = sum(1 for count in self.token_counts.values() if count > 1) / unique_tokens
        vocabulary_richness = unique_tokens / self.total_tokens if self.total_tokens > 0 else 0.0

        return {
            "unique_tokens": unique_tokens,
            "repetition_rate": repetition_rate,
            "vocabulary_richness": vocabulary_richness
        }
