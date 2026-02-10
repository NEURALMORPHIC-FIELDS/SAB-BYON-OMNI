# -*- coding: utf-8 -*-
"""ReasoningQuantifier - Incremental reasoning quality assessment."""

import numpy as np
from typing import Dict, Any, List

from sab_byon_omni.quantifiers.base_quantifier import BaseQuantifier


class ReasoningQuantifier(BaseQuantifier):
    """Cuantificare pentru raționament incremental cu analiză logică."""

    def __init__(self):
        self.coherence_scores = []
        self.logical_consistency = []
        self.evidence_strengths = []
        self.reasoning_chain = []

    def initial_score(self) -> float:
        return 0.0

    def update_score(self, current_score: float, new_premise: str,
                    current_reasoning: List[str]) -> float:
        """Evaluează calitatea raționamentului incremental."""

        # 1. Analizează coerența semantică
        coherence = self._analyze_semantic_coherence(new_premise, current_reasoning)
        self.coherence_scores.append(coherence)

        # 2. Verifică consistența logică
        consistency = self._check_logical_consistency(new_premise, current_reasoning)
        self.logical_consistency.append(consistency)

        # 3. Evaluează forța evidențelor
        evidence = self._evaluate_evidence_strength(new_premise)
        self.evidence_strengths.append(evidence)

        # 4. Construiește chain-ul de raționament
        self.reasoning_chain.append({
            "premise": new_premise,
            "coherence": coherence,
            "consistency": consistency,
            "evidence": evidence,
            "step": len(current_reasoning) + 1
        })

        # Scorul combinat cu ponderare adaptivă
        reasoning_score = (
            coherence * 0.4 +
            consistency * 0.4 +
            evidence * 0.2
        )

        return reasoning_score

    def _analyze_semantic_coherence(self, new_premise: str, current_reasoning: List[str]) -> float:
        """Analizează coerența semantică simplă."""
        if not current_reasoning:
            return 0.8  # Prima propoziție are coerență de bază

        # Analiză simplă bazată pe cuvinte comune și lungime
        new_words = set(new_premise.lower().split())

        coherence_scores = []
        for existing in current_reasoning[-3:]:  # Ultimele 3 propoziții
            existing_words = set(existing.lower().split())

            if not new_words or not existing_words:
                coherence_scores.append(0.3)
                continue

            # Jaccard similarity
            intersection = len(new_words & existing_words)
            union = len(new_words | existing_words)
            jaccard = intersection / union if union > 0 else 0.0

            # Ajustare pentru lungime
            length_factor = min(len(new_premise), len(existing)) / max(len(new_premise), len(existing), 1)

            coherence = (jaccard * 0.7 + length_factor * 0.3)
            coherence_scores.append(coherence)

        return np.mean(coherence_scores) if coherence_scores else 0.5

    def _check_logical_consistency(self, new_premise: str, current_reasoning: List[str]) -> float:
        """Verifică consistența logică cu heuristici simple."""
        if not current_reasoning:
            return 0.9

        # Detectează contradicții simple
        contradiction_indicators = ["not", "no", "never", "cannot", "impossible", "false"]
        affirmation_indicators = ["yes", "true", "always", "certainly", "definitely"]

        new_lower = new_premise.lower()
        has_negation = any(indicator in new_lower for indicator in contradiction_indicators)
        has_affirmation = any(indicator in new_lower for indicator in affirmation_indicators)

        consistency_scores = []
        for existing in current_reasoning:
            existing_lower = existing.lower()
            existing_negation = any(indicator in existing_lower for indicator in contradiction_indicators)
            existing_affirmation = any(indicator in existing_lower for indicator in affirmation_indicators)

            # Simpla verificare de contradicție
            if has_negation and existing_affirmation:
                consistency_scores.append(0.3)  # Potențială contradicție
            elif has_affirmation and existing_negation:
                consistency_scores.append(0.3)  # Potențială contradicție
            else:
                consistency_scores.append(0.8)  # Consistent

        return np.mean(consistency_scores) if consistency_scores else 0.8

    def _evaluate_evidence_strength(self, premise: str) -> float:
        """Evaluează forța evidenței bazată pe indicatori linguistici."""
        strength_indicators = {
            "proven": 0.9, "demonstrated": 0.8, "shown": 0.7, "research": 0.8,
            "study": 0.7, "data": 0.8, "evidence": 0.8, "fact": 0.7,
            "probably": 0.6, "likely": 0.6, "suggests": 0.5, "indicates": 0.6,
            "maybe": 0.3, "possibly": 0.4, "might": 0.3, "could": 0.4,
            "believe": 0.3, "think": 0.3, "feel": 0.2, "opinion": 0.3
        }

        premise_lower = premise.lower()
        strengths = []

        for indicator, strength in strength_indicators.items():
            if indicator in premise_lower:
                strengths.append(strength)

        if not strengths:
            return 0.5  # Neutral evidence strength

        return np.mean(strengths)

    def meets_threshold(self, score: float, threshold: float) -> bool:
        return score >= threshold

    def get_reasoning_analytics(self) -> Dict[str, Any]:
        """Returnează analiză detaliată a procesului de raționament."""
        if not self.reasoning_chain:
            return {"reasoning_quality": 0.0, "chain_length": 0, "consistency_trend": []}

        avg_coherence = np.mean(self.coherence_scores)
        avg_consistency = np.mean(self.logical_consistency)
        avg_evidence = np.mean(self.evidence_strengths)

        # Trend analysis
        if len(self.coherence_scores) > 1:
            coherence_trend = np.polyfit(range(len(self.coherence_scores)), self.coherence_scores, 1)[0]
        else:
            coherence_trend = 0.0

        return {
            "reasoning_quality": (avg_coherence + avg_consistency + avg_evidence) / 3,
            "chain_length": len(self.reasoning_chain),
            "avg_coherence": avg_coherence,
            "avg_consistency": avg_consistency,
            "avg_evidence_strength": avg_evidence,
            "coherence_trend": coherence_trend,
            "reasoning_chain": self.reasoning_chain[-5:]  # Last 5 steps
        }
