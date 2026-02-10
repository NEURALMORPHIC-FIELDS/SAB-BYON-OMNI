# -*- coding: utf-8 -*-
"""Tests for agent systems."""
import unittest
import numpy as np

from sab_byon_omni.agents.multi_agent_cortex import CognitiveAgent, MultiAgentCortex


class TestCognitiveAgent(unittest.TestCase):
    def test_init(self):
        agent = CognitiveAgent(0, "reasoning")
        self.assertEqual(agent.specialty, "reasoning")
        self.assertAlmostEqual(agent.activation, 0.1)

    def test_process(self):
        agent = CognitiveAgent(0, "reasoning")
        output = agent.process(0.5, {"topic": "reasoning about logic"})
        self.assertIsInstance(output, float)


class TestMultiAgentCortex(unittest.TestCase):
    def test_init(self):
        cortex = MultiAgentCortex()
        self.assertEqual(len(cortex.agents), 10)

    def test_parallel_process(self):
        cortex = MultiAgentCortex()
        input_vec = np.random.rand(10)
        outputs = cortex.parallel_process(input_vec, {"text": "test"})
        self.assertEqual(len(outputs), 10)

    def test_consensus(self):
        cortex = MultiAgentCortex()
        outputs = np.random.rand(10)
        result = cortex.form_consensus(outputs)
        self.assertIn('consensus_activation', result)
        self.assertIn('confidence', result)


if __name__ == "__main__":
    unittest.main()
