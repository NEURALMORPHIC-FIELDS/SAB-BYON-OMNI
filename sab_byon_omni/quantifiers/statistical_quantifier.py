# -*- coding: utf-8 -*-
"""StatisticalQuantifier - Welford's algorithm based confidence quantification."""

import numpy as np
from typing import List, Tuple

from sab_byon_omni.quantifiers.base_quantifier import BaseQuantifier


class StatisticalQuantifier(BaseQuantifier):
    """Cuantificare pentru încrederea în răspunsuri cu Welford's algorithm."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99% = 2.576
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.count = 0
        self.variance_history = []

    def initial_score(self) -> float:
        return float('inf')  # Start with infinite uncertainty

    def update_score(self, current_score: float, new_element: float,
                    current_subset: List[float]) -> float:
        """Welford's algorithm pentru actualizare incrementală."""
        self.count += 1
        self.sum_x += new_element
        self.sum_x2 += new_element ** 2

        if self.count < 2:
            return float('inf')

        # Calculează intervalul de încredere
        mean = self.sum_x / self.count
        variance = max(0, (self.sum_x2 - self.sum_x ** 2 / self.count) / (self.count - 1))
        self.variance_history.append(variance)

        std_error = np.sqrt(variance / self.count)
        margin_of_error = self.z_score * std_error

        return margin_of_error  # Confidence interval width

    def meets_threshold(self, score: float, threshold: float) -> bool:
        return score <= threshold  # Want narrow confidence interval

    def get_confidence_interval(self) -> Tuple[float, float]:
        """Returnează intervalul de încredere curent."""
        if self.count < 2:
            return (float('-inf'), float('inf'))

        mean = self.sum_x / self.count
        margin = self.z_score * np.sqrt(max(0, (self.sum_x2 - self.sum_x ** 2 / self.count) / (self.count - 1)) / self.count)
        return (mean - margin, mean + margin)
