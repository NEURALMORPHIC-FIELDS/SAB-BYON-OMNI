# -*- coding: utf-8 -*-
"""EvolutionaryFragParam - Evolved fragmergent parameters with integrated quantification."""

import math
import time
import numpy as np
from typing import Dict, Any, List

from sab_byon_omni.quantifiers import (
    QuantificationResult,
    StatisticalQuantifier,
    EntropyQuantifier,
    CryptographicPRNG,
)
from sab_byon_omni.evolution.metrics_module import metrics


class EvolutionaryFragParam:
    """Evolved fragmergent parameters cu cuantificare integrată."""

    def __init__(self, name: str = "EvolvedFrag", **kwargs):
        self.name = name
        self.vars = kwargs
        self.memory_influence = kwargs.get("memory_influence", 0.1)

        # Quantification engines integration
        self.statistical_quantifier = StatisticalQuantifier()
        self.entropy_quantifier = EntropyQuantifier()
        self.creativity_prng = CryptographicPRNG()

        # Evolution tracking
        self.phi_evolution_history = []
        self.quantification_results = {}

    @metrics.track("EvolutionaryFragParam", "phi_frag_evolved")
    def phi_frag_evolved(self, t: float, memory_factor: float = 0.0, creativity_boost: bool = False) -> float:
        """Enhanced phi_frag cu cuantificare și creativitate integrată."""
        try:
            alpha = self.vars.get("alpha", 0.02)
            lam = self.vars.get("lambda", 0.2)
            omega = self.vars.get("omega", 2.0)

            # Base fragmergent value
            base_value = lam * math.exp(-alpha * t) * math.sin(omega * t)

            # Memory influence modulation
            memory_modulation = 1 + (memory_factor * self.memory_influence)

            # Creative variation if requested
            if creativity_boost:
                creative_variation = self.creativity_prng.generate_creative_variation(base_value, 0.1)
                base_value = creative_variation

            # Apply memory modulation
            evolved_value = base_value * memory_modulation

            # Track evolution
            self.phi_evolution_history.append({
                "timestamp": time.time(),
                "t": t,
                "base_value": base_value,
                "memory_factor": memory_factor,
                "evolved_value": evolved_value,
                "creativity_boost": creativity_boost
            })

            # Update entropy quantifier with the evolved value
            self.entropy_quantifier.update_score(0.0, str(evolved_value), [])

            return evolved_value
        except Exception as e:
            print(f"phi_frag_evolved error: {e}")
            return 0.0

    def quantify_parameter_stability(self, threshold: float = 0.1) -> QuantificationResult:
        """Cuantifică stabilitatea parametrilor folosind StatisticalQuantifier."""
        if len(self.phi_evolution_history) < 2:
            return QuantificationResult([], 0, 0.0, 0.0, [])

        # Extract evolved values for statistical analysis
        values = [entry["evolved_value"] for entry in self.phi_evolution_history]

        # Use statistical quantifier to assess stability
        start_time = time.time()

        # Reset quantifier for fresh analysis
        stability_quantifier = StatisticalQuantifier()
        convergence_history = []

        for i, value in enumerate(values):
            score = stability_quantifier.update_score(0.0, value, values[:i])
            convergence_history.append(score)

            if stability_quantifier.meets_threshold(score, threshold):
                break

        result = QuantificationResult(
            subset=values[:len(convergence_history)],
            steps=len(convergence_history),
            final_score=convergence_history[-1] if convergence_history else float('inf'),
            execution_time=time.time() - start_time,
            convergence_history=convergence_history,
            metadata={
                "confidence_interval": stability_quantifier.get_confidence_interval(),
                "parameter_name": self.name,
                "analysis_type": "stability"
            }
        )

        self.quantification_results["stability"] = result
        metrics.track_quantification_event("statistical", result, {"parameter": self.name})

        return result

    def get_parameter_analytics(self) -> Dict[str, Any]:
        """Analiză completă a parametrilor evoluați."""
        if not self.phi_evolution_history:
            return {"status": "no_data"}

        # Basic statistics
        values = [e["evolved_value"] for e in self.phi_evolution_history]
        memory_factors = [e["memory_factor"] for e in self.phi_evolution_history]
        creativity_usage = sum(1 for e in self.phi_evolution_history if e["creativity_boost"])

        # Entropy analysis
        entropy_metrics = self.entropy_quantifier.get_diversity_metrics()

        # Creativity analysis
        creativity_stats = self.creativity_prng.get_exploration_stats()

        return {
            "parameter_statistics": {
                "mean_value": np.mean(values),
                "std_value": np.std(values),
                "value_range": (np.min(values), np.max(values)),
                "evolution_length": len(self.phi_evolution_history)
            },
            "memory_influence": {
                "avg_memory_factor": np.mean(memory_factors),
                "memory_correlation": np.corrcoef(values, memory_factors)[0, 1] if len(values) > 1 else 0.0
            },
            "creativity_usage": {
                "creative_operations": creativity_usage,
                "creativity_ratio": creativity_usage / len(self.phi_evolution_history),
                "creativity_stats": creativity_stats
            },
            "entropy_analysis": entropy_metrics,
            "quantification_results": {k: {"final_score": v.final_score, "steps": v.steps}
                                     for k, v in self.quantification_results.items()}
        }
