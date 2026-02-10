# -*- coding: utf-8 -*-
"""EvolutionaryFragmergentAIAgent - Enhanced fragmergent AI with pattern recognition."""

import random
import time
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List

from sab_byon_omni.quantifiers import QuantificationResult
from sab_byon_omni.evolution.metrics_module import metrics
from sab_byon_omni.evolution.frag_param import EvolutionaryFragParam
from sab_byon_omni.evolution.pathway_evolution import evolved_pathway_evolution
from sab_byon_omni.agents.base_agent import EvolutionaryBaseAgent


class EvolutionaryFragmergentAIAgent(EvolutionaryBaseAgent):
    """Enhanced fragmergent AI cu pattern recognition și creativity."""

    def __init__(self, pathways: tuple = (), synergy_level: float = 0.3):
        super().__init__()
        self.pathways = list(pathways)
        self.synergy_level = synergy_level
        self.pathway_patterns = defaultdict(list)
        self.evolution_history = deque(maxlen=100)
        self.creativity_patterns = defaultdict(int)
        self.emergence_events = []

    @metrics.track("EvolutionaryFragAI", "process_with_emergence")
    def process_with_emergence(self, Pn: float, t: float, param: EvolutionaryFragParam,
                             input_data: str = "", reasoning_chain: List[str] = None) -> Dict:
        """Enhanced processing cu emergence detection și creativity."""
        try:
            memory_context = ""
            memory_stats = {}
            if self.memory_system and input_data:
                memory_context, memory_stats = self.memory_system.retrieve_with_quantification(
                    f"pathway emergence {input_data}", self.context_id, max_chunks=4, t=t
                )

            memory_influence = len(memory_context) / 500.0

            reasoning_quality = 0.5
            reasoning_coherence = 0.5
            if reasoning_chain:
                reasoning_result = self.analyze_reasoning_quality(reasoning_chain)
                reasoning_quality = reasoning_result.final_score
                reasoning_coherence = reasoning_result.metadata.get("avg_coherence", 0.5)

            adjusted_synergy = self.synergy_level * (1 + memory_influence + reasoning_quality * 0.3)
            creativity_stats = self.creativity_prng.get_exploration_stats()
            creativity_factor = 1.0 + creativity_stats.get("creativity_score", 0.0) * 0.4

            pathway_created = False
            emergence_detected = False

            if random.random() > adjusted_synergy:
                pathway_evolution_value = evolved_pathway_evolution(Pn, t, param, memory_context, reasoning_chain)
                phi_value = param.phi_frag_evolved(t, memory_influence, True)
                creative_phi = self.creativity_prng.generate_creative_variation(phi_value, 0.15)

                if len(self.pathways) > 0:
                    last_pathway_value = float(self.pathways[-1].split(":")[-1]) if ":" in self.pathways[-1] else 0.0
                    value_delta = abs(pathway_evolution_value - last_pathway_value)
                    if value_delta > 0.5 and reasoning_quality > 0.7:
                        emergence_detected = True
                        self.emergence_events.append({
                            "timestamp": time.time(),
                            "value_delta": value_delta,
                            "reasoning_quality": reasoning_quality,
                            "phi_value": creative_phi,
                            "memory_influence": memory_influence
                        })

                new_pathway = (
                    f"pathway_{len(self.pathways)+1}:Pn={Pn:.3f}:phi={creative_phi:.3f}:"
                    f"psi={pathway_evolution_value:.3f}:R={reasoning_quality:.3f}:"
                    f"E={'1' if emergence_detected else '0'}:C={creativity_factor:.3f}"
                )
                self.pathways.append(new_pathway)
                pathway_created = True

                pattern_key = f"phi_{creative_phi:.2f}_R_{reasoning_quality:.2f}"
                self.pathway_patterns[pattern_key].append({
                    "value": pathway_evolution_value,
                    "emergence": emergence_detected,
                    "creativity": creativity_factor,
                    "timestamp": time.time()
                })

                creativity_pattern = (
                    int(creative_phi * 10) % 10,
                    int(reasoning_quality * 10) % 10,
                    int(memory_influence * 10) % 10
                )
                self.creativity_patterns[creativity_pattern] += 1

                if self.memory_system:
                    pathway_description = (
                        f"Created {'emergent' if emergence_detected else 'standard'} pathway: "
                        f"phi={creative_phi:.3f}, psi={pathway_evolution_value:.3f}, "
                        f"reasoning_quality={reasoning_quality:.3f}, "
                        f"creativity_factor={creativity_factor:.3f}, "
                        f"memory_influence={memory_influence:.3f}, "
                        f"context: {input_data}"
                    )
                    agent_state = {
                        "paths": len(self.pathways),
                        "phi_value": creative_phi,
                        "evolution": pathway_evolution_value,
                        "emergence": emergence_detected,
                        "reasoning_quality": reasoning_quality,
                        "creativity_factor": creativity_factor
                    }
                    self.memory_system.compress_and_store_evolved(
                        pathway_description, f"{self.context_id}_frag", agent_state, reasoning_chain
                    )

            current_evolution = {
                "timestamp": time.time(),
                "pathway_count": len(self.pathways),
                "memory_influence": memory_influence,
                "reasoning_quality": reasoning_quality,
                "creativity_factor": creativity_factor,
                "emergence_detected": emergence_detected,
                "phi_frag": param.phi_frag_evolved(t, memory_influence),
                "pathway_created": pathway_created
            }
            self.evolution_history.append(current_evolution)

            path_count = len(self.pathways)
            pattern_count = len(self.pathway_patterns)
            emergence_count = len(self.emergence_events)
            creativity_diversity = len(self.creativity_patterns)

            if self.pathways:
                pathway_values = [float(p.split(":")[-1]) for p in self.pathways if ":" in p]
                if pathway_values:
                    quantized_values = [str(int(v * 10) % 10) for v in pathway_values]
                    entropy_score = self.entropy_quantifier.update_score(0.0, " ".join(quantized_values), [])
                else:
                    entropy_score = 0.0
            else:
                entropy_score = 0.0

            self.performance_metrics["pathway_count"].append(path_count)
            self.performance_metrics["pattern_diversity"].append(pattern_count)
            self.performance_metrics["emergence_rate"].append(emergence_count)
            self.performance_metrics["creativity_diversity"].append(creativity_diversity)
            self.performance_metrics["entropy_score"].append(entropy_score)

            if pathway_created:
                metrics.track_quantification_event("entropy",
                    QuantificationResult(
                        subset=self.pathways[-5:],
                        steps=len(self.pathways),
                        final_score=entropy_score,
                        execution_time=0.001,
                        convergence_history=[entropy_score]
                    ),
                    {"agent": "FragmergentAI", "emergence": emergence_detected, "creativity_factor": creativity_factor}
                )

            response_data = {
                "response": (
                    f"Enhanced Fragmergent AI: paths={path_count}, patterns={pattern_count}, "
                    f"emergence={'YES' if emergence_detected else 'no'}, "
                    f"creativity={creativity_factor:.2f}, entropy={entropy_score:.3f}"
                ),
                "analytics": {
                    "pathway_metrics": {
                        "total_paths": path_count,
                        "pattern_count": pattern_count,
                        "emergence_events": emergence_count,
                        "creativity_diversity": creativity_diversity,
                        "entropy_score": entropy_score
                    },
                    "current_operation": {
                        "pathway_created": pathway_created,
                        "emergence_detected": emergence_detected,
                        "memory_influence": memory_influence,
                        "reasoning_quality": reasoning_quality,
                        "creativity_factor": creativity_factor
                    },
                    "pattern_analysis": {
                        "top_patterns": dict(sorted(self.pathway_patterns.items(),
                                                  key=lambda x: len(x[1]), reverse=True)[:3]),
                        "creativity_patterns": dict(sorted(self.creativity_patterns.items(),
                                                         key=lambda x: x[1], reverse=True)[:3])
                    },
                    "emergence_analysis": {
                        "total_emergence_events": emergence_count,
                        "recent_emergences": self.emergence_events[-3:],
                        "emergence_rate": emergence_count / max(1, len(self.evolution_history))
                    },
                    "memory_integration": {
                        "memory_stats": memory_stats,
                        "retrieved_content_length": len(memory_context)
                    }
                }
            }

            self.decision_history.append(response_data)
            return response_data

        except Exception as e:
            return {
                "response": f"Enhanced Fragmergent AI: error={e}, t={t:.2f}",
                "analytics": {"error": str(e), "state": "error"}
            }
