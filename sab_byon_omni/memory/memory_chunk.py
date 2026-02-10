# -*- coding: utf-8 -*-
"""EvolutionaryMemoryChunk - Enhanced memory chunk with quantification metadata."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvolutionaryMemoryChunk:
    """Enhanced memory chunk cu quantification metadata."""
    content: str
    timestamp: float
    context_id: str
    frequency_signature: float
    importance_score: float
    fragmergent_params: Dict
    access_count: int = 0
    compression_ratio: float = 1.0
    pathway_evolution: float = 0.0

    # Quantification metadata
    relevance_history: List[float] = field(default_factory=list)
    entropy_score: float = 0.0
    confidence_level: float = 0.0
    reasoning_quality: float = 0.0
