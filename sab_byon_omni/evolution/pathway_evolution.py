# -*- coding: utf-8 -*-
"""evolved_pathway_evolution - Enhanced pathway evolution with reasoning integration."""

import math
import numpy as np
from typing import List

from sab_byon_omni.quantifiers import (
    QuantificationResult,
    ReasoningQuantifier,
)
from sab_byon_omni.evolution.metrics_module import metrics
from sab_byon_omni.evolution.frag_param import EvolutionaryFragParam


@metrics.track("Global", "evolved_pathway_evolution")
def evolved_pathway_evolution(Pn: float, t: float, param: EvolutionaryFragParam,
                            memory_context: str = "", reasoning_chain: List[str] = None) -> float:
    """Enhanced pathway evolution cu reasoning quantifier integrat."""
    try:
        alpha = param.vars.get("p_alpha", 0.05)
        beta = param.vars.get("p_beta", 0.02)

        # Base evolution
        base_evolution = Pn + alpha * math.sin(t) - beta * math.cos(t)

        # Memory context influence
        memory_boost = 0.0
        if memory_context:
            context_factor = len(memory_context) / 1000.0
            memory_boost = context_factor * 0.1

        # Reasoning chain influence
        reasoning_boost = 0.0
        if reasoning_chain:
            # Use reasoning quantifier to evaluate the chain quality
            reasoning_quantifier = ReasoningQuantifier()
            for premise in reasoning_chain:
                reasoning_score = reasoning_quantifier.update_score(0.0, premise, reasoning_chain)

            if reasoning_quantifier.reasoning_chain:
                avg_reasoning_quality = np.mean([step["coherence"] for step in reasoning_quantifier.reasoning_chain])
                reasoning_boost = avg_reasoning_quality * 0.15

                # Track reasoning metrics
                reasoning_analytics = reasoning_quantifier.get_reasoning_analytics()
                metrics.track_quantification_event("reasoning",
                    QuantificationResult(
                        subset=reasoning_chain,
                        steps=len(reasoning_chain),
                        final_score=reasoning_analytics["reasoning_quality"],
                        execution_time=0.001,  # Minimal for tracking
                        convergence_history=[step["coherence"] for step in reasoning_quantifier.reasoning_chain]
                    ),
                    reasoning_analytics
                )

        # Combined evolution
        evolved_pathway = base_evolution * (1 + memory_boost + reasoning_boost)

        return evolved_pathway
    except Exception as e:
        print(f"evolved_pathway_evolution error: {e}")
        return Pn
