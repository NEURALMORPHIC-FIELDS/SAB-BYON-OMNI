# -*- coding: utf-8 -*-
"""Tests for quantification engines."""
import unittest

from sab_byon_omni.quantifiers import (
    QuantificationResult,
    StatisticalQuantifier,
    EntropyQuantifier,
    CryptographicPRNG,
    ReasoningQuantifier,
    MemoryRelevanceQuantifier,
    DecisionConfidenceQuantifier,
)


class TestQuantificationResult(unittest.TestCase):
    def test_creation(self):
        r = QuantificationResult(
            subset=[1, 2, 3], steps=10, final_score=0.85,
            execution_time=0.01, convergence_history=[0.5, 0.7, 0.85]
        )
        self.assertEqual(r.steps, 10)
        self.assertAlmostEqual(r.final_score, 0.85)


class TestStatisticalQuantifier(unittest.TestCase):
    def test_init(self):
        sq = StatisticalQuantifier()
        self.assertIsNotNone(sq)

    def test_update_score(self):
        sq = StatisticalQuantifier()
        score = sq.update_score(0.5, 0.6, [0.4, 0.5, 0.6, 0.7])
        self.assertIsInstance(score, float)


class TestEntropyQuantifier(unittest.TestCase):
    def test_init(self):
        eq = EntropyQuantifier()
        self.assertIsNotNone(eq)

    def test_update_score(self):
        eq = EntropyQuantifier()
        score = eq.update_score(0.0, "test entropy data", [])
        self.assertIsInstance(score, float)


class TestCryptographicPRNG(unittest.TestCase):
    def test_init(self):
        prng = CryptographicPRNG(seed=None)
        self.assertIsNotNone(prng)

    def test_generate(self):
        prng = CryptographicPRNG(seed=b"test_seed")
        val = prng.generate()
        self.assertIsInstance(val, float)
        self.assertTrue(0 <= val <= 1)


class TestReasoningQuantifier(unittest.TestCase):
    def test_init(self):
        rq = ReasoningQuantifier()
        self.assertIsNotNone(rq)


if __name__ == "__main__":
    unittest.main()
