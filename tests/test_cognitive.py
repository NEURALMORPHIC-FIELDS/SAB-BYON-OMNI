# -*- coding: utf-8 -*-
"""Tests for cognitive modules."""
import unittest
import numpy as np

from sab_byon_omni.cognitive.fisher_geometry import FisherGeometryEngine
from sab_byon_omni.cognitive.info_density_field import InformationDensityField
from sab_byon_omni.cognitive.semantic_photon import SemanticPhotonTheory
from sab_byon_omni.cognitive.duei_framework import DUEIFramework, SemanticMode, EmergentMode
from sab_byon_omni.cognitive.personality import PersonalitySystem


class TestFisherGeometry(unittest.TestCase):
    def test_fisher_metric(self):
        fe = FisherGeometryEngine(dim=5)
        state = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mass = fe.compute_fisher_metric(state)
        self.assertIsInstance(mass, float)
        self.assertGreater(mass, 0)

    def test_quantum_fisher(self):
        fe = FisherGeometryEngine(dim=5)
        states = {'a': 0.3, 'b': 0.4, 'c': 0.3}
        qfi = fe.compute_quantum_fisher_info(states)
        self.assertIsInstance(qfi, float)


class TestInfoDensityField(unittest.TestCase):
    def test_fragmergent_density(self):
        idf = InformationDensityField(grid_size=16)
        r = np.zeros((16, 16))
        density = idf.fragmergent_density(r, t=1.0)
        self.assertEqual(density.shape, (16, 16))


class TestSemanticPhoton(unittest.TestCase):
    def test_create_photon(self):
        spt = SemanticPhotonTheory()
        photon = spt.create_photon("Hello world")
        self.assertIsNotNone(photon)
        self.assertGreater(photon.frequency, 0)

    def test_interaction(self):
        spt = SemanticPhotonTheory()
        p1 = spt.create_photon("consciousness")
        p2 = spt.create_photon("awareness")
        strength = spt.semantic_interaction(p1, p2)
        self.assertIsInstance(strength, float)


class TestDUEI(unittest.TestCase):
    def test_detect_regime(self):
        duei = DUEIFramework()
        switch, mode = duei.detect_regime_switch(
            consciousness=0.5, coherence=0.3, complexity=0.8
        )
        self.assertIsInstance(switch, bool)

    def test_emergence_score(self):
        duei = DUEIFramework()
        score = duei.emergence_score(consciousness=0.6, complexity=0.4)
        self.assertTrue(0 <= score <= 1)


class TestPersonality(unittest.TestCase):
    def test_evolve_traits(self):
        ps = PersonalitySystem()
        virtue_states = {'stoicism': 0.6, 'empathy': 0.7, 'creativity': 0.5}
        ps.evolve_traits(virtue_states, consciousness=0.5)
        self.assertEqual(len(ps.traits), 10)

    def test_dominant_traits(self):
        ps = PersonalitySystem()
        dominant = ps.get_dominant_traits(k=3)
        self.assertEqual(len(dominant), 3)


if __name__ == "__main__":
    unittest.main()
