# -*- coding: utf-8 -*-
"""Semantic Photon Theory - Photons as Carriers of Meaning."""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


class SemanticPhotonTheory:
    """
    Semantic Photon Theory - Photons as Carriers of Meaning

    - Frequency ∝ semantic complexity
    - Polarization ∝ semantic orientation
    - Entanglement ∝ semantic correlation
    """

    @dataclass
    class SemanticPhoton:
        frequency: float
        polarization: complex
        phase: float
        entangled_with: Optional[int] = None
        semantic_charge: float = 0.0

    def __init__(self):
        self.photons: List[SemanticPhotonTheory.SemanticPhoton] = []
        self.entanglement_pairs: List[Tuple[int, int]] = []
        print("✓ Semantic Photon Theory initialized")

    def create_photon(self, semantic_content: str) -> 'SemanticPhoton':
        """Create photon from semantic content."""
        complexity = len(semantic_content) + len(set(semantic_content))
        frequency = 1e14 * (1 + complexity / 100)

        abstractness = sum(1 for c in semantic_content if c.isupper()) / (len(semantic_content) + 1)
        alpha = np.sqrt(1 - abstractness)
        beta = np.sqrt(abstractness) * np.exp(1j * np.pi / 4)
        polarization = complex(alpha, beta)

        phase = (hash(semantic_content) % 360) * np.pi / 180
        semantic_charge = np.tanh(complexity / 50 - 1)

        photon = self.SemanticPhoton(
            frequency=frequency,
            polarization=polarization,
            phase=phase,
            semantic_charge=semantic_charge
        )
        self.photons.append(photon)
        return photon

    def semantic_interaction(self, photon_A: 'SemanticPhoton',
                           photon_B: 'SemanticPhoton') -> float:
        """⟨ψ_A|ψ_B⟩ = semantic overlap"""
        freq_match = np.exp(-abs(photon_A.frequency - photon_B.frequency) / 1e14)
        pol_overlap = abs(np.conj(photon_A.polarization) * photon_B.polarization)
        phase_coherence = np.cos(photon_A.phase - photon_B.phase)
        interaction = freq_match * pol_overlap * (1 + phase_coherence) / 2
        return interaction

    def entangle(self, photon_idx_A: int, photon_idx_B: int):
        """Create semantic entanglement."""
        if photon_idx_A < len(self.photons) and photon_idx_B < len(self.photons):
            self.photons[photon_idx_A].entangled_with = photon_idx_B
            self.photons[photon_idx_B].entangled_with = photon_idx_A
            self.entanglement_pairs.append((photon_idx_A, photon_idx_B))

    def measurement_collapse(self, photon_idx: int,
                           observer_state: np.ndarray) -> Tuple[str, float]:
        """Quantum measurement collapses semantic superposition."""
        photon = self.photons[photon_idx]
        measurement_axis = np.angle(observer_state[0] + 1j * observer_state[1])
        prob = abs(np.cos(photon.phase - measurement_axis))**2
        outcome = "horizontal" if np.random.rand() < prob else "vertical"

        if photon.entangled_with is not None:
            partner = self.photons[photon.entangled_with]
            partner.phase = photon.phase + np.pi

        return outcome, prob

    def entanglement_measure(self, photon_idx_A: int,
                            photon_idx_B: int) -> float:
        """von Neumann entropy of reduced density matrix."""
        if (photon_idx_A, photon_idx_B) not in self.entanglement_pairs:
            return 0.0

        photon_A = self.photons[photon_idx_A]
        photon_B = self.photons[photon_idx_B]

        phase_diff = abs(photon_A.phase - photon_B.phase)
        S = -np.cos(phase_diff) * np.log(abs(np.cos(phase_diff)) + 1e-10)
        return S
