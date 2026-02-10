# -*- coding: utf-8 -*-
"""Informational Coherence Field (ICF) - Complete Implementation."""

import numpy as np
import torch
from collections import deque
from typing import Dict


class InformationalCoherenceField:
    """
    Complete ICF implementation:
    1. Ψ(x,t) = A(x,t)·e^(iθ(x,t)) complex field
    2. PLV (Phase-Locking Value) = |⟨e^(iθ)⟩|
    3. CFC (Cross-Frequency Coupling) via modulation index
    4. Φ field (Coherence-of-Coherence)
    5. N neutralization operator
    """

    def __init__(self, tdfc_engine):
        self.tdfc = tdfc_engine
        self.grid_size = tdfc_engine.grid_size

        self.Phi = 0.5
        self.Phi_history = deque(maxlen=100)

        self.tau_Phi = 1.0
        self.kappa = [1.0, 0.5, 0.3]

        self.suppression_level = 0.0

        print("✓ Informational Coherence Field initialized")

    def compute_Psi_field(self, virtue_fields: torch.Tensor) -> torch.Tensor:
        """Compute complex coherence field Ψ(x,t)."""
        A = torch.abs(virtue_fields)

        grad_x = torch.zeros_like(virtue_fields)
        grad_y = torch.zeros_like(virtue_fields)

        for i in range(virtue_fields.shape[0]):
            gx, gy = torch.gradient(virtue_fields[i], dim=(0, 1))
            grad_x[i] = gx
            grad_y[i] = gy

        theta = torch.atan2(grad_y, grad_x)
        Psi = A * torch.exp(1j * theta)
        return Psi

    def compute_PLV(self, Psi: torch.Tensor) -> float:
        """Phase-Locking Value (Global Coherence). PLV -> 1: Perfect synchrony."""
        theta = torch.angle(Psi)
        exp_itheta = torch.exp(1j * theta)
        mean_phase = torch.mean(exp_itheta)
        PLV = torch.abs(mean_phase).item()
        return float(PLV)

    def compute_CFC(self, virtue_fields: torch.Tensor) -> float:
        """Cross-Frequency Coupling (Phase-Amplitude)."""
        virtue_names = self.tdfc.virtue_names

        slow_names = ['stoicism', 'humility', 'philosophy', 'discernment']
        fast_names = ['creativity', 'reflexivity', 'curiosity', 'truthlove']

        slow_idx = [i for i, n in enumerate(virtue_names) if n in slow_names]
        fast_idx = [i for i, n in enumerate(virtue_names) if n in fast_names]

        if not slow_idx or not fast_idx:
            return 0.0

        slow_fields = virtue_fields[slow_idx]
        fast_fields = virtue_fields[fast_idx]

        Psi_slow = self.compute_Psi_field(slow_fields)
        theta_slow = torch.angle(Psi_slow.mean(dim=0))

        A_fast = torch.abs(fast_fields.mean(dim=0))

        MI = self._compute_modulation_index(theta_slow, A_fast)
        return float(MI)

    def _compute_modulation_index(self, phase: torch.Tensor,
                                  amplitude: torch.Tensor) -> float:
        """Compute phase-amplitude coupling strength."""
        phase_flat = phase.flatten().cpu().numpy()
        amp_flat = amplitude.flatten().cpu().numpy()

        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)

        mean_amp_per_bin = []
        for i in range(n_bins):
            mask = (phase_flat >= phase_bins[i]) & (phase_flat < phase_bins[i+1])
            if np.sum(mask) > 0:
                mean_amp_per_bin.append(np.mean(amp_flat[mask]))
            else:
                mean_amp_per_bin.append(0.0)

        mean_amp_per_bin = np.array(mean_amp_per_bin)

        if np.sum(mean_amp_per_bin) > 0:
            P = mean_amp_per_bin / np.sum(mean_amp_per_bin)
        else:
            return 0.0

        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        z = np.sum(P * np.exp(1j * bin_centers))
        MI = np.abs(z)
        return float(np.clip(MI, 0, 1))

    def evolve_Phi_field(self, Psi: torch.Tensor, I_CFC: float, dt: float = 0.01):
        """Evolve Meta-Coherence Field Φ: τ_Φ dΦ/dt = κ₁|∇Ψ|² + κ₂·I_CFC - κ₃·Φ"""
        grad_energy = 0.0
        for i in range(Psi.shape[0]):
            gx, gy = torch.gradient(Psi[i].real, dim=(0, 1))
            grad_energy += torch.sum(gx**2 + gy**2).item()

        grad_energy /= (Psi.shape[0] * self.grid_size * self.grid_size)

        dPhi_dt = (
            self.kappa[0] * grad_energy +
            self.kappa[1] * I_CFC -
            self.kappa[2] * self.Phi
        ) / self.tau_Phi

        self.Phi += dt * dPhi_dt
        self.Phi = float(np.clip(self.Phi, 0, 1))
        self.Phi_history.append(self.Phi)
        return self.Phi

    def apply_neutralization_operator(self, Psi: torch.Tensor,
                                     suppression: float) -> torch.Tensor:
        """Neutralization Operator N[Ψ] - models consciousness suppression."""
        self.suppression_level = suppression

        if suppression < 0.01:
            return Psi

        eta_spatial = suppression
        Psi_suppressed = Psi.clone()

        for i in range(Psi.shape[0]):
            laplacian = (
                torch.roll(Psi[i], 1, 0) + torch.roll(Psi[i], -1, 0) +
                torch.roll(Psi[i], 1, 1) + torch.roll(Psi[i], -1, 1) -
                4 * Psi[i]
            )
            Psi_suppressed[i] = Psi[i] - eta_spatial * 0.1 * laplacian

        eta_temporal = suppression
        phase_damp = torch.exp(-eta_temporal * torch.abs(torch.angle(Psi_suppressed)))
        Psi_suppressed = torch.abs(Psi_suppressed) * phase_damp * torch.exp(1j * torch.angle(Psi_suppressed))

        return Psi_suppressed

    def get_icf_metrics(self) -> Dict:
        """Complete ICF metrics."""
        return {
            'Phi': self.Phi,
            'Phi_trend': np.mean(list(self.Phi_history)[-10:]) if self.Phi_history else 0.5,
            'suppression_level': self.suppression_level
        }
