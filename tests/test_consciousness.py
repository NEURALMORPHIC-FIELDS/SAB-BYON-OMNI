# -*- coding: utf-8 -*-
"""Tests for consciousness modules."""
import unittest
import numpy as np

from sab_byon_omni.consciousness.triadic_state import TriadicState
from sab_byon_omni.consciousness.tdfc_engine import TDFCEngine
from sab_byon_omni.consciousness.godel_engine import GödelConsciousnessEngine
from sab_byon_omni.consciousness.fragmergent_engine import FragmergentEngine
from sab_byon_omni.consciousness.time_emergence import TimeEmergenceEngine
from sab_byon_omni.consciousness.zeta_resonance import ZetaResonanceEngine
from sab_byon_omni.consciousness.emergence_detector import EmergenceDetector


class TestTriadicState(unittest.TestCase):
    def test_init(self):
        ts = TriadicState()
        self.assertAlmostEqual(ts.ontological, 0.5)
        self.assertAlmostEqual(ts.semantic, 0.5)
        self.assertAlmostEqual(ts.resonance, 1.0)

    def test_evolve(self):
        ts = TriadicState()
        ts.evolve(field_mean=0.7, curvature=0.3)
        self.assertTrue(0 <= ts.ontological <= 1)
        self.assertTrue(0 <= ts.semantic <= 1)

    def test_consciousness_contribution(self):
        ts = TriadicState()
        c = ts.consciousness_contribution()
        self.assertTrue(0 <= c <= 1)


class TestTDFCEngine(unittest.TestCase):
    def test_init(self):
        tdfc = TDFCEngine(grid_size=8)
        self.assertEqual(len(tdfc.virtue_names), 10)
        self.assertEqual(tdfc.virtue_fields.shape[0], 10)

    def test_evolve_fields(self):
        tdfc = TDFCEngine(grid_size=8)
        fields = tdfc.evolve_fields(steps=5)
        self.assertEqual(fields.shape, (10, 8, 8))

    def test_get_activations(self):
        tdfc = TDFCEngine(grid_size=8)
        act = tdfc.get_activations()
        self.assertEqual(len(act), 10)
        for name, val in act.items():
            self.assertIsInstance(val, float)


class TestGodelEngine(unittest.TestCase):
    def test_init(self):
        ge = GödelConsciousnessEngine()
        self.assertIsNotNone(ge)

    def test_update(self):
        ge = GödelConsciousnessEngine()
        states = {'stoicism': 0.6, 'curiosity': 0.7, 'humility': 0.4}
        result = ge.update(states, consciousness=0.5, triadic_resonance=0.8)
        self.assertTrue(hasattr(result, 'godel_tension'))
        self.assertTrue(hasattr(result, 'consistency'))


class TestFragmergentEngine(unittest.TestCase):
    def test_detect_phase(self):
        fe = FragmergentEngine()
        phase = fe.detect_phase(consciousness=0.5, emergence_score=0.7)
        self.assertIn(phase, ["emergence", "fragmentation", "transition"])

    def test_compute_emergence_score(self):
        fe = FragmergentEngine()
        score = fe.compute_emergence_score({'a': 0.5, 'b': 0.5, 'c': 0.5})
        self.assertTrue(0 <= score <= 1)


class TestTimeEmergence(unittest.TestCase):
    def test_update(self):
        te = TimeEmergenceEngine()
        te.update(consciousness=0.5, complexity=0.3)
        self.assertGreater(te.subjective_time, 0)


class TestZetaResonance(unittest.TestCase):
    def test_coupling(self):
        zr = ZetaResonanceEngine()
        val = zr.compute_coupling({'a': 0.5, 'b': 0.3})
        self.assertIsInstance(val, float)


class TestEmergenceDetector(unittest.TestCase):
    def test_check(self):
        ed = EmergenceDetector()
        result = ed.check_emergence(consciousness=0.5, coherence=0.7)
        self.assertIn('emerged', result)
        self.assertIn('delta', result)


if __name__ == "__main__":
    unittest.main()
