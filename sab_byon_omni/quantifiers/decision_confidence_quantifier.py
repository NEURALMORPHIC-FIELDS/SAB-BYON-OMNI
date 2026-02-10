# -*- coding: utf-8 -*-
"""DecisionConfidenceQuantifier - Self-calibrated decision confidence."""

import time
import numpy as np
from typing import Dict, Any, List

from sab_byon_omni.quantifiers.base_quantifier import BaseQuantifier


class DecisionConfidenceQuantifier(BaseQuantifier):
    """Cuantificare pentru autocalibrarea încrederii în decizii."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.evidence_weights = []
        self.support_scores = []
        self.weighted_sum = 0.0
        self.weight_sum = 0.0
        self.decision_history = []

    def initial_score(self) -> float:
        return 0.0  # No confidence initially

    def update_score(self, current_score: float, new_evidence: Dict,
                    current_evidence: List[Dict]) -> float:
        """Actualizează încrederea bazată pe noi evidențe."""

        # Extrage parametrii evidenței
        reliability = new_evidence.get('reliability', 1.0)  # 0-1
        support_score = new_evidence.get('support_score', 0.5)  # 0-1
        evidence_type = new_evidence.get('type', 'general')
        source_credibility = new_evidence.get('source_credibility', 0.5)

        # Ajustează greutatea bazată pe credibilitate și tip
        type_multipliers = {
            'experimental': 1.2,
            'statistical': 1.1,
            'expert_opinion': 0.9,
            'anecdotal': 0.6,
            'general': 1.0
        }

        adjusted_weight = reliability * type_multipliers.get(evidence_type, 1.0) * source_credibility

        # Actualizare incrementală
        self.evidence_weights.append(adjusted_weight)
        self.support_scores.append(support_score)

        self.weighted_sum += support_score * adjusted_weight
        self.weight_sum += adjusted_weight

        # Calculează încrederea ponderată
        if self.weight_sum == 0:
            confidence = 0.0
        else:
            weighted_average = self.weighted_sum / self.weight_sum

            # Ajustează pentru numărul de evidențe (mai multe evidențe = mai multă încredere)
            evidence_count_factor = min(1.0, len(self.evidence_weights) / 5.0)

            # Calculează variabilitatea pentru incertitudine
            if len(self.support_scores) > 1:
                variance = np.var(self.support_scores)
                uncertainty_penalty = min(0.3, variance)
            else:
                uncertainty_penalty = 0.2  # High uncertainty with little data

            confidence = weighted_average * evidence_count_factor * (1 - uncertainty_penalty)

        # Înregistrează decizia
        self.decision_history.append({
            'evidence': new_evidence,
            'confidence': confidence,
            'evidence_count': len(self.evidence_weights),
            'timestamp': time.time()
        })

        return confidence

    def meets_threshold(self, score: float, threshold: float) -> bool:
        return score >= threshold

    def get_decision_analytics(self) -> Dict[str, Any]:
        """Analiză detaliată a procesului decizional."""
        if not self.decision_history:
            return {"decision_confidence": 0.0, "evidence_quality": 0.0, "decision_stability": 0.0}

        # Calculează stabilitatea deciziei
        recent_confidences = [d['confidence'] for d in self.decision_history[-5:]]
        decision_stability = 1.0 - np.std(recent_confidences) if len(recent_confidences) > 1 else 0.5

        # Calitatea evidenței
        avg_weight = np.mean(self.evidence_weights)
        evidence_quality = min(1.0, avg_weight)

        # Trend-ul încrederii
        if len(recent_confidences) > 1:
            confidence_trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
        else:
            confidence_trend = 0.0

        return {
            "decision_confidence": self.decision_history[-1]['confidence'] if self.decision_history else 0.0,
            "evidence_quality": evidence_quality,
            "decision_stability": decision_stability,
            "confidence_trend": confidence_trend,
            "total_evidence_count": len(self.evidence_weights),
            "weighted_support": self.weighted_sum / self.weight_sum if self.weight_sum > 0 else 0.0,
            "recent_decisions": self.decision_history[-3:]
        }
