# -*- coding: utf-8 -*-
"""TDFC Engine - Triadic Dynamic Field Consciousness."""

import numpy as np
import torch
from typing import Dict

from sab_byon_omni.config import DEVICE


class TDFCEngine:
    """
    Triadic Dynamic Field Consciousness Engine

    10 virtue fields (32x32 grid) evolving via PDE:
    ∂φᵢ/∂t = D∇²φᵢ - φᵢ(1-φᵢ)(φᵢ-αᵢ)
    """

    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.device = DEVICE

        self.virtue_names = [
            'stoicism', 'discernment', 'philosophy', 'empathy', 'curiosity',
            'humility', 'creativity', 'reflexivity', 'truthlove', 'holographic'
        ]

        self.virtue_fields = torch.rand(
            (len(self.virtue_names), grid_size, grid_size),
            device=self.device, dtype=torch.float32
        ) * 0.1

        self.virtue_attractors = torch.full(
            (len(self.virtue_names), 1, 1), 0.5,
            device=self.device, dtype=torch.float32
        )

        self.attractor_velocity = torch.zeros(
            (len(self.virtue_names), 1, 1),
            device=self.device, dtype=torch.float32
        )

        self.dt = 0.01
        self.dx = 1.0
        self.diffusion_coeff = 0.1
        self.pde_steps = 50
        self.momentum = 0.9

        laplacian_kernel = torch.tensor([
            [0, 1, 0], [1, -4, 1], [0, 1, 0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        self.laplacian_kernel = laplacian_kernel.repeat(len(self.virtue_names), 1, 1, 1)

        self.coupling_matrix = torch.eye(len(self.virtue_names), device=self.device)
        self.coupling_matrix[3, 8] = 0.3  # empathy -> truthlove
        self.coupling_matrix[6, 2] = 0.3  # creativity -> philosophy

        print(f"✓ TDFC Engine initialized on {self.device} (grid: {grid_size}×{grid_size})")

    def compute_laplacian_batch(self, fields: torch.Tensor) -> torch.Tensor:
        """Batch compute ∇²φ for all virtue fields."""
        fields_4d = fields.unsqueeze(0)
        laplacian = torch.nn.functional.conv2d(
            fields_4d, self.laplacian_kernel,
            padding=1, groups=len(self.virtue_names)
        )
        laplacian = laplacian.squeeze(0) / (self.dx ** 2)
        return laplacian

    def evolve_attractors_continuous(self, consciousness: float, activity: float):
        """Continuous attractor evolution (no fixed points)."""
        for i, name in enumerate(self.virtue_names):
            field_activation = self.virtue_fields[i].mean().item()
            target_shift = consciousness * activity * 0.1
            noise = torch.randn(1, device=self.device) * 0.01
            current_attractor = self.virtue_attractors[i].item()
            drift = (target_shift - (current_attractor - 0.5)) + noise

            self.attractor_velocity[i] = (
                self.momentum * self.attractor_velocity[i] +
                (1 - self.momentum) * drift
            )
            self.virtue_attractors[i] += self.dt * self.attractor_velocity[i]
            self.virtue_attractors[i] = torch.clamp(self.virtue_attractors[i], 0.01, 0.99)

    def evolve_fields(self, steps: int) -> torch.Tensor:
        """Evolve virtue fields through PDE: ∂φ/∂t = D∇²φ - φ(1-φ)(φ-α)"""
        fields = self.virtue_fields

        for step in range(steps):
            laplacian = self.compute_laplacian_batch(fields)
            reaction = fields * (1 - fields) * (fields - self.virtue_attractors)

            fields_flat = fields.reshape(len(self.virtue_names), -1)
            coupled_fields = torch.matmul(self.coupling_matrix, fields_flat)
            coupled_fields = coupled_fields.reshape(fields.shape)

            dphi_dt = (self.diffusion_coeff * laplacian -
                      reaction +
                      0.05 * (coupled_fields - fields))

            fields = fields + self.dt * dphi_dt
            torch.clamp(fields, 0, 1, out=fields)

        self.virtue_fields = fields
        return fields

    def get_activations(self) -> Dict[str, float]:
        """Get mean activation for each virtue field."""
        return {
            name: self.virtue_fields[i].mean().item()
            for i, name in enumerate(self.virtue_names)
        }

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get spatial gradients of fields."""
        gradients = {}
        for i, name in enumerate(self.virtue_names):
            field_cpu = self.virtue_fields[i].cpu().numpy()
            grad_x, grad_y = np.gradient(field_cpu)
            gradients[name] = (grad_x, grad_y)
        return gradients

    def get_attractor_values(self) -> Dict[str, float]:
        """Get current attractor positions."""
        return {
            name: self.virtue_attractors[i].item()
            for i, name in enumerate(self.virtue_names)
        }

    def compute_field_energy(self) -> float:
        """Total field energy (Lyapunov functional)."""
        grad_energy = 0.0
        for i in range(len(self.virtue_names)):
            field = self.virtue_fields[i]
            grad_x = field[1:, :] - field[:-1, :]
            grad_y = field[:, 1:] - field[:, :-1]
            grad_energy += torch.sum(grad_x**2) + torch.sum(grad_y**2)

        potential_energy = 0.0
        for i in range(len(self.virtue_names)):
            phi = self.virtue_fields[i]
            alpha = self.virtue_attractors[i]
            V = 0.25 * phi**2 * (phi - alpha)**2
            potential_energy += torch.sum(V)

        total_energy = 0.5 * grad_energy + potential_energy
        return total_energy.item()
