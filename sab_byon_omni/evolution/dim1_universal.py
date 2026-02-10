# -*- coding: utf-8 -*-
"""EvolutionaryDim1_UniversalFragmergence - Enhanced universal fragmergence with quantified theorems."""

import numpy as np
from collections import defaultdict
from typing import Dict, Any, List

from sab_byon_omni.quantifiers import (
    StatisticalQuantifier,
    EntropyQuantifier,
)
from sab_byon_omni.evolution.metrics_module import metrics
from sab_byon_omni.evolution.frag_param import EvolutionaryFragParam
from sab_byon_omni.evolution.pathway_evolution import evolved_pathway_evolution


class EvolutionaryDim1_UniversalFragmergence:
    """Enhanced universal fragmergence cu quantified theorems."""

    def __init__(self):
        self.theorem_history = defaultdict(list)
        self.memory_system = None
        self.theorem_analytics = defaultdict(lambda: defaultdict(list))
        self.convergence_patterns = defaultdict(list)

    def set_memory_system(self, memory_system):
        self.memory_system = memory_system

    @metrics.track("EvolutionaryDim1", "evolved_teorema_1_1")
    def evolved_teorema_1_1(self, Pn: float, t: float, param: EvolutionaryFragParam,
                          context: str = "", confidence_threshold: float = 0.1) -> Dict[str, Any]:
        """Enhanced pathway stabilization cu confidence quantification."""
        # Calculate base result
        result = evolved_pathway_evolution(Pn, t, param, context)

        # Quantify stabilization confidence
        if len(self.theorem_history["teorema_1_1"]) >= 2:
            recent_results = [entry["result"] for entry in self.theorem_history["teorema_1_1"][-10:]]
            statistical_quantifier = StatisticalQuantifier()

            for value in recent_results:
                confidence_score = statistical_quantifier.update_score(0.0, value, recent_results)

            confidence_interval = statistical_quantifier.get_confidence_interval()
            stabilization_confidence = 1.0 / (confidence_score + 0.001) if confidence_score != float('inf') else 0.0
        else:
            confidence_interval = (float('-inf'), float('inf'))
            stabilization_confidence = 0.0

        # Enhanced theorem record
        theorem_record = {
            "result": result,
            "timestamp": __import__('time').time(),
            "context_length": len(context),
            "confidence_interval": confidence_interval,
            "stabilization_confidence": stabilization_confidence,
            "convergence_achieved": stabilization_confidence > confidence_threshold
        }

        self.theorem_history["teorema_1_1"].append(theorem_record)
        self.theorem_analytics["teorema_1_1"]["stabilization_confidence"].append(stabilization_confidence)

        # Track convergence patterns
        if theorem_record["convergence_achieved"]:
            self.convergence_patterns["teorema_1_1"].append({
                "timestamp": __import__('time').time(),
                "result": result,
                "confidence": stabilization_confidence
            })

        return {
            "result": result,
            "analytics": theorem_record,
            "theorem_id": "1_1"
        }

    @metrics.track("EvolutionaryDim1", "evolved_teorema_1_2")
    def evolved_teorema_1_2(self, Pn: float, t: float, param: EvolutionaryFragParam,
                          context: str = "", entropy_threshold: float = 0.5) -> Dict[str, Any]:
        """Enhanced pathway oscillation cu entropy analysis."""
        base_result = evolved_pathway_evolution(Pn, t, param, context)
        phi_modulation = param.phi_frag_evolved(t, len(context)/1000.0, True)
        result = base_result * phi_modulation

        # Entropy analysis pentru oscillation complexity
        entropy_quantifier = EntropyQuantifier()
        oscillation_sequence = str(result) + str(phi_modulation) + str(base_result)
        entropy_score = entropy_quantifier.update_score(0.0, oscillation_sequence, [])

        # Oscillation pattern analysis
        if len(self.theorem_history["teorema_1_2"]) >= 3:
            recent_results = [entry["result"] for entry in self.theorem_history["teorema_1_2"][-5:]]
            oscillation_amplitude = np.std(recent_results)
            oscillation_frequency = self._calculate_oscillation_frequency(recent_results)
        else:
            oscillation_amplitude = 0.0
            oscillation_frequency = 0.0

        theorem_record = {
            "result": result,
            "base_result": base_result,
            "phi_modulation": phi_modulation,
            "timestamp": __import__('time').time(),
            "entropy_score": entropy_score,
            "oscillation_amplitude": oscillation_amplitude,
            "oscillation_frequency": oscillation_frequency,
            "complexity_achieved": entropy_score > entropy_threshold
        }

        self.theorem_history["teorema_1_2"].append(theorem_record)
        self.theorem_analytics["teorema_1_2"]["entropy_score"].append(entropy_score)

        return {
            "result": result,
            "analytics": theorem_record,
            "theorem_id": "1_2"
        }

    def _calculate_oscillation_frequency(self, values: List[float]) -> float:
        """Calculate oscillation frequency from value sequence."""
        if len(values) < 3:
            return 0.0

        # Simple frequency estimation bazat pe zero-crossings
        mean_value = np.mean(values)
        crossings = 0

        for i in range(1, len(values)):
            if (values[i-1] - mean_value) * (values[i] - mean_value) < 0:
                crossings += 1

        # Frequency aproximate (crossings per 2 time units)
        return crossings / (2 * len(values))

    @metrics.track("EvolutionaryDim1", "evolved_teorema_1_3")
    def evolved_teorema_1_3(self, Pn: float, t: float, param: EvolutionaryFragParam,
                          context: str = "") -> Dict[str, Any]:
        """Enhanced pathway convergence cu advanced analysis."""
        evolution_result = evolved_pathway_evolution(Pn, t, param, context)
        convergence_delta = abs(evolution_result - Pn)

        # Convergence trend analysis
        if len(self.theorem_history["teorema_1_3"]) >= 5:
            recent_deltas = [entry["convergence_delta"] for entry in self.theorem_history["teorema_1_3"][-10:]]

            # Trend analysis
            convergence_trend = np.polyfit(range(len(recent_deltas)), recent_deltas, 1)[0] if len(recent_deltas) > 1 else 0.0

            # Stability analysis
            delta_stability = 1.0 - (np.std(recent_deltas) / (np.mean(recent_deltas) + 1e-6))

            # Prediction pentru next convergence
            predicted_next_delta = recent_deltas[-1] + convergence_trend if recent_deltas else convergence_delta
        else:
            convergence_trend = 0.0
            delta_stability = 0.5
            predicted_next_delta = convergence_delta

        theorem_record = {
            "result": convergence_delta,
            "convergence_delta": convergence_delta,
            "evolution_result": evolution_result,
            "original_value": Pn,
            "timestamp": __import__('time').time(),
            "convergence_trend": convergence_trend,
            "delta_stability": delta_stability,
            "predicted_next_delta": predicted_next_delta,
            "is_converging": convergence_trend < 0,
            "is_stable": delta_stability > 0.7
        }

        self.theorem_history["teorema_1_3"].append(theorem_record)
        self.theorem_analytics["teorema_1_3"]["convergence_trend"].append(convergence_trend)

        return {
            "result": convergence_delta,
            "analytics": theorem_record,
            "theorem_id": "1_3"
        }

    def get_dimension_analytics(self) -> Dict[str, Any]:
        """Comprehensive analytics pentru dimension 1."""
        if not any(self.theorem_history.values()):
            return {"status": "no_theorem_data"}

        dimension_analytics = {}

        for theorem_id, history in self.theorem_history.items():
            if history:
                results = [entry.get("result", 0) for entry in history]

                dimension_analytics[theorem_id] = {
                    "execution_count": len(history),
                    "result_statistics": {
                        "mean": np.mean(results),
                        "std": np.std(results),
                        "min": np.min(results),
                        "max": np.max(results),
                        "trend": np.polyfit(range(len(results)), results, 1)[0] if len(results) > 1 else 0.0
                    },
                    "recent_performance": {
                        "last_5_avg": np.mean(results[-5:]) if len(results) >= 5 else np.mean(results),
                        "stability": 1.0 - (np.std(results[-10:]) / (np.mean(results[-10:]) + 1e-6)) if len(results) >= 2 else 1.0
                    }
                }

                # Theorem-specific analytics
                if theorem_id == "teorema_1_1":
                    convergences = [entry.get("convergence_achieved", False) for entry in history]
                    dimension_analytics[theorem_id]["convergence_rate"] = np.mean(convergences)
                elif theorem_id == "teorema_1_2":
                    entropies = [entry.get("entropy_score", 0) for entry in history]
                    dimension_analytics[theorem_id]["avg_entropy"] = np.mean(entropies)
                elif theorem_id == "teorema_1_3":
                    trends = [entry.get("convergence_trend", 0) for entry in history]
                    dimension_analytics[theorem_id]["overall_convergence_trend"] = np.mean(trends)

        # Cross-theorem analysis
        all_results = []
        for history in self.theorem_history.values():
            all_results.extend([entry.get("result", 0) for entry in history])

        dimension_summary = {
            "total_executions": len(all_results),
            "dimension_stability": 1.0 - (np.std(all_results) / (np.mean(all_results) + 1e-6)) if len(all_results) > 1 else 1.0,
            "dimension_activity": len(all_results) / max(1, len(self.theorem_history)),
            "convergence_events": len(self.convergence_patterns.get("teorema_1_1", []))
        }

        return {
            "theorem_analytics": dimension_analytics,
            "dimension_summary": dimension_summary,
            "convergence_patterns": {k: len(v) for k, v in self.convergence_patterns.items()}
        }
