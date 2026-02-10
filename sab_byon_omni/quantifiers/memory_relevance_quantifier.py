# -*- coding: utf-8 -*-
"""MemoryRelevanceQuantifier - Intelligent memory retrieval quantification."""

import time
import numpy as np
from typing import Dict, Any, List

from sab_byon_omni.quantifiers.base_quantifier import BaseQuantifier


class MemoryRelevanceQuantifier(BaseQuantifier):
    """Cuantificare pentru retrieval inteligent de memorie."""

    def __init__(self, decay_factor: float = 3600.0):  # 1 hour decay
        self.semantic_similarities = []
        self.temporal_relevances = []
        self.access_boosts = []
        self.decay_factor = decay_factor
        self.query_embedding = None

    def initial_score(self) -> float:
        return 0.0

    def update_score(self, current_score: float, memory_chunk: Dict,
                    current_context: List[Dict]) -> float:
        """Calculează relevanța memoriei pentru contextul curent."""

        # 1. Relevanță semantică (simplificată)
        semantic_relevance = self._compute_semantic_similarity(
            memory_chunk.get('content', ''),
            [ctx.get('content', '') for ctx in current_context]
        )
        self.semantic_similarities.append(semantic_relevance)

        # 2. Relevanță temporală cu decay exponențial
        current_time = time.time()
        memory_time = memory_chunk.get('timestamp', current_time)
        time_diff = current_time - memory_time
        temporal_relevance = np.exp(-time_diff / self.decay_factor)
        self.temporal_relevances.append(temporal_relevance)

        # 3. Boost din frecvența de acces
        access_count = memory_chunk.get('access_count', 0)
        access_boost = np.log1p(access_count)  # Log(1 + access_count)
        self.access_boosts.append(access_boost)

        # 4. Factorul de importanță
        importance = memory_chunk.get('importance_score', 0.5)

        # Scorul combinat
        relevance_score = (
            semantic_relevance * 0.4 +
            temporal_relevance * 0.3 +
            (access_boost / 10.0) * 0.2 +  # Normalized access boost
            importance * 0.1
        )

        return relevance_score

    def _compute_semantic_similarity(self, memory_content: str, context_contents: List[str]) -> float:
        """Calculează similaritatea semantică simplă."""
        if not memory_content or not context_contents:
            return 0.0

        memory_words = set(memory_content.lower().split())
        if not memory_words:
            return 0.0

        similarities = []
        for context_content in context_contents:
            if not context_content:
                continue

            context_words = set(context_content.lower().split())
            if not context_words:
                continue

            # Jaccard similarity
            intersection = len(memory_words & context_words)
            union = len(memory_words | context_words)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)

        return np.mean(similarities) if similarities else 0.0

    def meets_threshold(self, score: float, threshold: float) -> bool:
        return score >= threshold

    def get_retrieval_analytics(self) -> Dict[str, Any]:
        """Analiză detaliată a procesului de retrieval."""
        if not self.semantic_similarities:
            return {"avg_relevance": 0.0, "temporal_decay": 0.0, "access_patterns": {}}

        return {
            "avg_semantic_similarity": np.mean(self.semantic_similarities),
            "avg_temporal_relevance": np.mean(self.temporal_relevances),
            "avg_access_boost": np.mean(self.access_boosts),
            "relevance_distribution": {
                "high": sum(1 for s in self.semantic_similarities if s > 0.7),
                "medium": sum(1 for s in self.semantic_similarities if 0.3 <= s <= 0.7),
                "low": sum(1 for s in self.semantic_similarities if s < 0.3)
            },
            "total_memories_evaluated": len(self.semantic_similarities)
        }
