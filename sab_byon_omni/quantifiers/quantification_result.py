# -*- coding: utf-8 -*-
"""QuantificationResult dataclass."""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class QuantificationResult:
    """Rezultatul procesului de cuantificare cu metadata extinsă."""
    subset: List[Any]
    steps: int
    final_score: float
    execution_time: float
    convergence_history: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
