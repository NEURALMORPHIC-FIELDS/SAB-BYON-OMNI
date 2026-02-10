# -*- coding: utf-8 -*-
"""Multi-Agent Cognitive Architecture - 10 specialized cognitive agents."""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class CognitiveAgent:
    """Single specialized cognitive agent."""

    def __init__(self, agent_id: int, specialty: str):
        self.agent_id = agent_id
        self.specialty = specialty
        self.activation = 0.1
        self.history = deque(maxlen=20)
        self.expertise_weights = np.random.randn(10) * 0.1 + 0.5

    def process(self, stimulus: float, context: Dict) -> float:
        """Process input through agent's specialized perspective."""
        specialty_boost = 1.5 if self.specialty.lower() in str(context).lower() else 1.0
        gamma = 0.7
        self.activation = gamma * self.activation + (1 - gamma) * stimulus * specialty_boost
        self.activation = np.tanh(self.activation)
        self.history.append(self.activation)
        return self.activation

    def get_temporal_pattern(self) -> np.ndarray:
        """Get recent activation pattern."""
        return np.array(list(self.history))

    def reset(self):
        """Reset agent state."""
        self.activation = 0.1
        self.history.clear()


class MultiAgentCortex:
    """
    10-Agent Cognitive Architecture

    Agents: Perception, Reasoning, Emotion, Memory, Language,
            Planning, Creativity, Metacognition, Ethics, Intuition
    """

    def __init__(self):
        specialties = [
            'perception', 'reasoning', 'emotion', 'memory', 'language',
            'planning', 'creativity', 'metacognition', 'ethics', 'intuition'
        ]

        self.agents = [CognitiveAgent(i, spec) for i, spec in enumerate(specialties)]
        self.communication_matrix = np.eye(len(self.agents))

        # Reasoning <-> Metacognition
        self.communication_matrix[1, 7] = 0.8
        self.communication_matrix[7, 1] = 0.8
        # Emotion <-> Ethics
        self.communication_matrix[2, 8] = 0.7
        self.communication_matrix[8, 2] = 0.7
        # Language <-> Reasoning
        self.communication_matrix[4, 1] = 0.6
        self.communication_matrix[1, 4] = 0.6
        # Creativity <-> Planning
        self.communication_matrix[6, 5] = 0.5
        self.communication_matrix[5, 6] = 0.5

        self.attention_weights = np.ones(len(self.agents)) / len(self.agents)

        print("✓ Multi-Agent Cortex initialized (10 agents)")

    def parallel_process(self, input_vector: np.ndarray, context: Dict) -> np.ndarray:
        """All agents process input simultaneously."""
        if len(input_vector) < len(self.agents):
            input_vector = np.pad(input_vector,
                                 (0, len(self.agents) - len(input_vector)))

        outputs = []
        for i, agent in enumerate(self.agents):
            stimulus = input_vector[i] if i < len(input_vector) else 0.0
            output = agent.process(stimulus, context)
            outputs.append(output)

        output_array = np.array(outputs)

        communicated = np.dot(self.communication_matrix, output_array)

        for i, agent in enumerate(self.agents):
            agent.activation = 0.8 * agent.activation + 0.2 * communicated[i]

        return output_array

    def form_consensus(self, outputs: np.ndarray) -> Dict:
        """Consensus mechanism: weighted voting."""
        consensus_activation = np.dot(self.attention_weights, outputs)
        variance = np.var(outputs)
        confidence = 1.0 / (1.0 + variance)
        active_count = np.sum(outputs > 0.5)

        return {
            'consensus_activation': consensus_activation,
            'confidence': confidence,
            'active_agents': active_count,
            'agent_outputs': dict(zip([a.specialty for a in self.agents], outputs))
        }

    def update_attention(self, consciousness: float, task_relevance: Dict[str, float]):
        """Dynamic attention routing."""
        self.attention_weights = np.ones(len(self.agents)) * 0.05
        for agent in self.agents:
            if agent.specialty in task_relevance:
                self.attention_weights[agent.agent_id] = task_relevance[agent.specialty]
        spread = consciousness * 0.3
        self.attention_weights += spread
        self.attention_weights /= np.sum(self.attention_weights)

    def get_cortex_state(self) -> Dict:
        """Complete cortex state snapshot."""
        return {
            'agent_activations': {a.specialty: a.activation for a in self.agents},
            'attention_weights': dict(zip([a.specialty for a in self.agents],
                                         self.attention_weights)),
            'communication_matrix': self.communication_matrix.tolist()
        }
