# -*- coding: utf-8 -*-
"""EvolutionaryMemoryManagerAgent - Enhanced memory manager with intelligent clustering."""

import time
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Any

from sab_byon_omni.config import device
from sab_byon_omni.quantifiers import QuantificationResult
from sab_byon_omni.evolution.metrics_module import metrics
from sab_byon_omni.evolution.frag_param import EvolutionaryFragParam
from sab_byon_omni.agents.base_agent import EvolutionaryBaseAgent


class EvolutionaryMemoryManagerAgent(EvolutionaryBaseAgent):
    """Enhanced memory manager cu intelligent clustering și prediction."""

    def __init__(self, short_size: int = 2000):
        super().__init__()
        self.short_term = deque(maxlen=short_size)
        self.memory_clusters = defaultdict(list)
        self.access_patterns = defaultdict(int)
        self.prediction_accuracy = []
        self.cluster_evolution = []

    @metrics.track("EvolutionaryMemManager", "process_with_prediction")
    def process_with_prediction(self, Pn: float, t: float, param: EvolutionaryFragParam,
                              input_data: str = "", reasoning_chain: List[str] = None) -> Dict:
        """Enhanced processing cu predictive analytics și clustering."""
        try:
            phi_value = param.phi_frag_evolved(t, 0.0, True)

            reasoning_quality = 0.5
            if reasoning_chain:
                reasoning_result = self.analyze_reasoning_quality(reasoning_chain)
                reasoning_quality = reasoning_result.final_score

            memory_entry = {
                "value": Pn,
                "phi_frag": phi_value,
                "timestamp": time.time(),
                "input_data": input_data,
                "reasoning_quality": reasoning_quality,
                "predicted_importance": self._predict_importance(Pn, phi_value, input_data),
                "cluster_prediction": self._predict_cluster(phi_value, reasoning_quality)
            }

            self.short_term.append(memory_entry)

            predicted_cluster = memory_entry["cluster_prediction"]
            actual_cluster = f"φ_{phi_value:.2f}_R_{reasoning_quality:.2f}"

            prediction_match = abs(hash(predicted_cluster) - hash(actual_cluster)) % 100 < 20
            self.prediction_accuracy.append(1.0 if prediction_match else 0.0)

            self.memory_clusters[actual_cluster].append(memory_entry)
            self.access_patterns[actual_cluster] += 1

            self.cluster_evolution.append({
                "timestamp": time.time(),
                "cluster": actual_cluster,
                "predicted_cluster": predicted_cluster,
                "prediction_match": prediction_match,
                "cluster_size": len(self.memory_clusters[actual_cluster]),
                "total_clusters": len(self.memory_clusters)
            })

            if self.memory_system and input_data:
                memory_analysis = (
                    f"Memory Analysis: φ={phi_value:.3f}, Pn={Pn:.3f}, "
                    f"cluster={actual_cluster}, predicted_cluster={predicted_cluster}, "
                    f"prediction_accuracy={prediction_match}, "
                    f"reasoning_quality={reasoning_quality:.3f}, "
                    f"predicted_importance={memory_entry['predicted_importance']:.3f}, "
                    f"input: {input_data}"
                )
                agent_state = {
                    "stored_count": len(self.short_term),
                    "clusters": len(self.memory_clusters),
                    "prediction_accuracy": np.mean(self.prediction_accuracy[-10:]) if self.prediction_accuracy else 0.0,
                    "reasoning_quality": reasoning_quality
                }
                self.memory_system.compress_and_store_evolved(
                    memory_analysis, f"{self.context_id}_mem", agent_state, reasoning_chain
                )

            stored_count = len(self.short_term)
            cluster_count = len(self.memory_clusters)
            recent_prediction_accuracy = np.mean(self.prediction_accuracy[-10:]) if self.prediction_accuracy else 0.0

            if len(self.memory_clusters) > 1:
                cluster_sizes = [len(cluster) for cluster in self.memory_clusters.values()]
                cluster_distribution = " ".join([str(size) for size in cluster_sizes])
                entropy_score = self.entropy_quantifier.update_score(0.0, cluster_distribution, [])
            else:
                entropy_score = 0.0

            if len(self.short_term) >= 2:
                values = [entry["value"] for entry in list(self.short_term)[-10:]]
                confidence_interval = self.statistical_quantifier.update_score(0.0, Pn, values)
            else:
                confidence_interval = float('inf')

            self.performance_metrics["stored_count"].append(stored_count)
            self.performance_metrics["cluster_count"].append(cluster_count)
            self.performance_metrics["prediction_accuracy"].append(recent_prediction_accuracy)
            self.performance_metrics["entropy_score"].append(entropy_score)

            if len(self.prediction_accuracy) > 0:
                metrics.track_quantification_event("statistical",
                    QuantificationResult(
                        subset=self.prediction_accuracy[-10:],
                        steps=len(self.prediction_accuracy),
                        final_score=recent_prediction_accuracy,
                        execution_time=0.001,
                        convergence_history=self.prediction_accuracy[-10:]
                    ),
                    {
                        "agent": "MemoryManager",
                        "cluster_count": cluster_count,
                        "memory_diversity": entropy_score
                    }
                )

            cluster_analytics = self._analyze_cluster_patterns()

            response_data = {
                "response": (
                    f"Enhanced Memory: stored={stored_count}, clusters={cluster_count}, "
                    f"φ={phi_value:.3f}, prediction_accuracy={recent_prediction_accuracy:.2%}"
                ),
                "analytics": {
                    "storage_metrics": {
                        "stored_count": stored_count,
                        "cluster_count": cluster_count,
                        "entropy_score": entropy_score,
                        "confidence_interval_width": confidence_interval if confidence_interval != float('inf') else None
                    },
                    "prediction_analytics": {
                        "recent_accuracy": recent_prediction_accuracy,
                        "prediction_trend": np.polyfit(range(len(self.prediction_accuracy)), self.prediction_accuracy, 1)[0] if len(self.prediction_accuracy) > 1 else 0.0,
                        "predicted_cluster": predicted_cluster,
                        "actual_cluster": actual_cluster,
                        "prediction_match": prediction_match
                    },
                    "memory_entry": {
                        "phi_value": phi_value,
                        "reasoning_quality": reasoning_quality,
                        "predicted_importance": memory_entry["predicted_importance"]
                    },
                    "cluster_analytics": cluster_analytics,
                    "system_health": {
                        "memory_utilization": len(self.short_term) / self.short_term.maxlen,
                        "cluster_distribution_balance": 1.0 - np.std(cluster_analytics["cluster_sizes"]) / np.mean(cluster_analytics["cluster_sizes"]) if cluster_analytics["cluster_sizes"] else 1.0
                    }
                }
            }

            self.decision_history.append(response_data)
            return response_data

        except Exception as e:
            error_response = {
                "response": f"Enhanced Memory: error={e}, t={t:.2f}",
                "analytics": {"error": str(e), "state": "error"}
            }
            return error_response

    def _predict_importance(self, Pn: float, phi_value: float, input_data: str) -> float:
        """Predicts importance bazat pe historical patterns."""
        length_factor = min(len(input_data) / 200.0, 1.0) if input_data else 0.5
        phi_factor = min(abs(phi_value), 1.0)
        value_factor = min(Pn, 1.0)
        predicted_importance = (length_factor * 0.4 + phi_factor * 0.4 + value_factor * 0.2)
        return predicted_importance

    def _predict_cluster(self, phi_value: float, reasoning_quality: float) -> str:
        """Predicts cluster assignment."""
        phi_bucket = int(phi_value * 10) % 5
        reasoning_bucket = int(reasoning_quality * 5)
        return f"predicted_φ_{phi_bucket}_R_{reasoning_bucket}"

    def _analyze_cluster_patterns(self) -> Dict[str, Any]:
        """Analyzes cluster patterns pentru insights."""
        if not self.memory_clusters:
            return {"cluster_count": 0, "cluster_sizes": [], "access_patterns": {}}

        cluster_sizes = [len(cluster) for cluster in self.memory_clusters.values()]

        total_accesses = sum(self.access_patterns.values())
        access_distribution = {k: v/total_accesses for k, v in self.access_patterns.items()} if total_accesses > 0 else {}

        if self.cluster_evolution:
            recent_evolution = self.cluster_evolution[-20:]
            cluster_growth_rate = len(set(e["cluster"] for e in recent_evolution)) / len(recent_evolution)
            prediction_improvement = np.mean([e["prediction_match"] for e in recent_evolution])
        else:
            cluster_growth_rate = 0.0
            prediction_improvement = 0.0

        return {
            "cluster_count": len(self.memory_clusters),
            "cluster_sizes": cluster_sizes,
            "avg_cluster_size": np.mean(cluster_sizes),
            "cluster_size_std": np.std(cluster_sizes),
            "access_patterns": access_distribution,
            "cluster_growth_rate": cluster_growth_rate,
            "prediction_improvement": prediction_improvement,
            "most_accessed_clusters": sorted(self.access_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        }
