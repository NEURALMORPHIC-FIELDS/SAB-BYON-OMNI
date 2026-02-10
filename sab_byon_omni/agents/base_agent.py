# -*- coding: utf-8 -*-
"""EvolutionaryBaseAgent - Enhanced base class with quantification capabilities."""

import time
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List

from sab_byon_omni.quantifiers import (
    QuantificationResult,
    StatisticalQuantifier,
    EntropyQuantifier,
    ReasoningQuantifier,
    DecisionConfidenceQuantifier,
    CryptographicPRNG,
)
from sab_byon_omni.memory.fragmergent_memory import EvolutionaryFragmergentMemory


class EvolutionaryBaseAgent:
    """Enhanced base class cu quantification capabilities."""

    def __init__(self):
        self.memory_system = None
        self.context_id = None

        # Integrated quantifiers
        self.statistical_quantifier = StatisticalQuantifier()
        self.entropy_quantifier = EntropyQuantifier()
        self.reasoning_quantifier = ReasoningQuantifier()
        self.confidence_quantifier = DecisionConfidenceQuantifier()
        self.creativity_prng = CryptographicPRNG()

        # Agent analytics
        self.decision_history = []
        self.performance_metrics = defaultdict(list)
        self.learning_curve = []

    def set_memory_system(self, memory_system: EvolutionaryFragmergentMemory, context_id: str):
        """Connect agent to evolved memory system."""
        self.memory_system = memory_system
        self.context_id = context_id

    def quantify_decision_confidence(self, evidence_list: List[Dict]) -> QuantificationResult:
        """Cuantifică încrederea în decizie folosind toate evidențele."""
        start_time = time.time()

        convergence_history = []
        for evidence in evidence_list:
            confidence_score = self.confidence_quantifier.update_score(0.0, evidence, evidence_list)
            convergence_history.append(confidence_score)

        result = QuantificationResult(
            subset=evidence_list,
            steps=len(evidence_list),
            final_score=convergence_history[-1] if convergence_history else 0.0,
            execution_time=time.time() - start_time,
            convergence_history=convergence_history,
            metadata=self.confidence_quantifier.get_decision_analytics()
        )

        return result

    def analyze_reasoning_quality(self, reasoning_chain: List[str]) -> QuantificationResult:
        """Analizează calitatea unui chain de raționament."""
        start_time = time.time()

        reasoning_quantifier = ReasoningQuantifier()
        convergence_history = []

        for premise in reasoning_chain:
            reasoning_score = reasoning_quantifier.update_score(0.0, premise, reasoning_chain)
            convergence_history.append(reasoning_score)

        result = QuantificationResult(
            subset=reasoning_chain,
            steps=len(reasoning_chain),
            final_score=convergence_history[-1] if convergence_history else 0.0,
            execution_time=time.time() - start_time,
            convergence_history=convergence_history,
            metadata=reasoning_quantifier.get_reasoning_analytics()
        )

        return result

    def get_agent_analytics(self) -> Dict[str, Any]:
        """Comprehensive agent analytics."""
        return {
            "decision_history_length": len(self.decision_history),
            "statistical_confidence": self.statistical_quantifier.get_confidence_interval(),
            "entropy_diversity": self.entropy_quantifier.get_diversity_metrics(),
            "reasoning_analytics": self.reasoning_quantifier.get_reasoning_analytics() if self.reasoning_quantifier.reasoning_chain else {},
            "decision_confidence": self.confidence_quantifier.get_decision_analytics(),
            "creativity_stats": self.creativity_prng.get_exploration_stats(),
            "performance_trends": {
                metric: {"avg": np.mean(values), "trend": np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0}
                for metric, values in self.performance_metrics.items()
            },
            "learning_curve": self.learning_curve[-20:] if self.learning_curve else []
        }
