# -*- coding: utf-8 -*-
"""EvolutionaryReinforcementLearningAgent - Enhanced RL agent with all quantifiers."""

import random
import time
import torch
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List

from sab_byon_omni.config import device
from sab_byon_omni.quantifiers import QuantificationResult
from sab_byon_omni.evolution.metrics_module import metrics
from sab_byon_omni.evolution.frag_param import EvolutionaryFragParam
from sab_byon_omni.agents.base_agent import EvolutionaryBaseAgent


class EvolutionaryReinforcementLearningAgent(EvolutionaryBaseAgent):
    """Enhanced RL agent cu toate cuantificatorii integrați."""

    def __init__(self, state_size: int = 100, action_size: int = 5):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = torch.zeros((state_size, action_size), device=device, dtype=torch.float32)
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1
        self.run_count = 0
        self.experience_buffer = deque(maxlen=1000)

        # Enhanced learning tracking
        self.reward_history = []
        self.action_distribution = defaultdict(int)
        self.exploration_efficiency = []

    @metrics.track("EvolutionaryRL", "process_with_quantification")
    def process_with_quantification(self, Pn: float, t: float, param: EvolutionaryFragParam,
                                  input_data: str = "", reasoning_chain: List[str] = None) -> Dict:
        """Enhanced processing cu toate cuantificatorii activi."""
        try:
            # Memory retrieval cu confidence filtering
            memory_context = ""
            memory_stats = {}
            if self.memory_system and input_data:
                memory_context, memory_stats = self.memory_system.retrieve_with_quantification(
                    input_data, self.context_id, max_chunks=3, t=t, confidence_threshold=0.6
                )

            memory_factor = len(memory_context) / 1000.0

            # Reasoning analysis
            reasoning_quality = 0.5
            if reasoning_chain:
                reasoning_result = self.analyze_reasoning_quality(reasoning_chain)
                reasoning_quality = reasoning_result.final_score
                metrics.track_quantification_event("reasoning", reasoning_result,
                    {"agent": "RL", "context_id": self.context_id})

            # Enhanced epsilon cu creativity
            creativity_boost = self.creativity_prng.generate_creative_variation(0.0, 0.02)
            memory_adjusted_epsilon = max(0.01, self.epsilon - memory_factor * 0.05 + creativity_boost)

            self.run_count += 1
            state = int(Pn * self.state_size) % self.state_size

            # Action selection
            if random.uniform(0, 1) < memory_adjusted_epsilon:
                action = random.randint(0, self.action_size - 1)
                action_confidence = 0.3
            else:
                q_values = self.q_table[state]
                action = torch.argmax(q_values, dim=-1).item()
                q_softmax = torch.softmax(q_values * 5, dim=0)
                action_confidence = q_softmax[action].item()

            self.action_distribution[action] += 1

            # Enhanced reward
            base_reward = param.phi_frag_evolved(t, memory_factor, True)
            memory_bonus = memory_factor * 0.5
            reasoning_bonus = reasoning_quality * 0.3
            confidence_bonus = action_confidence * 0.2

            total_reward = base_reward * (1 + 0.5 * Pn) + memory_bonus + reasoning_bonus + confidence_bonus
            self.reward_history.append(total_reward)

            # Decision confidence analysis
            decision_evidence = [
                {
                    'reliability': action_confidence,
                    'support_score': min(1.0, total_reward + 0.5),
                    'type': 'rl_decision',
                    'source_credibility': memory_stats.get('avg_confidence', 0.5)
                }
            ]
            confidence_result = self.quantify_decision_confidence(decision_evidence)
            decision_confidence = confidence_result.final_score

            # Enhanced experience
            experience = {
                "state": state,
                "action": action,
                "reward": total_reward,
                "memory_context": memory_context[:100],
                "reasoning_quality": reasoning_quality,
                "action_confidence": action_confidence,
                "decision_confidence": decision_confidence,
                "timestamp": time.time()
            }
            self.experience_buffer.append(experience)

            # Store in memory
            if self.memory_system:
                experience_text = (
                    f"RL Experience: State={state}, Action={action}, Reward={total_reward:.3f}, "
                    f"Confidence={decision_confidence:.3f}, Reasoning={reasoning_quality:.3f}, "
                    f"Context='{input_data[:50]}...'"
                )
                agent_state = {
                    "reward": total_reward,
                    "action": action,
                    "state": state,
                    "confidence": decision_confidence,
                    "reasoning_quality": reasoning_quality
                }
                self.memory_system.compress_and_store_evolved(
                    experience_text, f"{self.context_id}_rl", agent_state, reasoning_chain
                )

            # Q-learning update
            next_state = state
            best_next_action = torch.argmax(self.q_table[next_state], dim=-1)
            td_target = torch.tensor(total_reward, device=device) + self.gamma * self.q_table[next_state, best_next_action]
            adaptive_alpha = self.alpha * (0.5 + 0.5 * action_confidence)

            with torch.no_grad():
                self.q_table[state, action] += adaptive_alpha * (td_target - self.q_table[state, action])

            # Track performance
            self.performance_metrics["reward"].append(total_reward)
            self.performance_metrics["confidence"].append(decision_confidence)
            self.performance_metrics["exploration_rate"].append(memory_adjusted_epsilon)

            if len(self.reward_history) >= 10:
                recent_avg_reward = np.mean(self.reward_history[-10:])
                self.learning_curve.append(recent_avg_reward)

            action_entropy = -sum(p * np.log2(p + 1e-8) for p in
                                [count / sum(self.action_distribution.values())
                                 for count in self.action_distribution.values()])
            self.exploration_efficiency.append(action_entropy)

            metrics.track_quantification_event("statistical",
                QuantificationResult(
                    subset=self.reward_history[-10:],
                    steps=len(self.reward_history),
                    final_score=total_reward,
                    execution_time=0.001,
                    convergence_history=self.reward_history[-10:]
                ),
                {"agent": "RL", "state": state, "action": action}
            )

            response_data = {
                "response": f"Enhanced RL: action={action}, reward={total_reward:.3f}, confidence={decision_confidence:.3f}",
                "analytics": {
                    "action": action,
                    "state": state,
                    "total_reward": total_reward,
                    "components": {
                        "base_reward": base_reward,
                        "memory_bonus": memory_bonus,
                        "reasoning_bonus": reasoning_bonus,
                        "confidence_bonus": confidence_bonus
                    },
                    "confidence_metrics": {
                        "action_confidence": action_confidence,
                        "decision_confidence": decision_confidence,
                        "epsilon": memory_adjusted_epsilon
                    },
                    "memory_influence": {
                        "memory_factor": memory_factor,
                        "memory_stats": memory_stats
                    },
                    "reasoning_analysis": {
                        "reasoning_quality": reasoning_quality,
                        "chain_length": len(reasoning_chain) if reasoning_chain else 0
                    },
                    "exploration_stats": {
                        "action_distribution": dict(self.action_distribution),
                        "exploration_efficiency": self.exploration_efficiency[-1] if self.exploration_efficiency else 0
                    }
                }
            }

            self.decision_history.append(response_data)
            return response_data

        except Exception as e:
            return {
                "response": f"Enhanced RL: error={e}, t={t:.2f}",
                "analytics": {"error": str(e), "state": "error"}
            }
