# -*- coding: utf-8 -*-
"""Fragmergent Information Dynamics (FID) Field."""

import numpy as np
from typing import Tuple


class InformationDensityField:
    """
    Treats information density as fundamental physical field.
    ρ_info(r, t) evolves according to fragmergent dynamics.
    """

    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.field = np.zeros((grid_size, grid_size))
        self.entropy_field = np.zeros((grid_size, grid_size))

        print("✓ Information Density Field initialized")

    def fragmergent_density(self, r: np.ndarray, t: float) -> np.ndarray:
        """
        ρ_info(r,t) = ρ₀ * [1 + α*cos(k·r - ωt)] * exp(-|r|²/σ²)
        """
        rho0 = 1.0
        alpha = 0.3
        k = 2 * np.pi / self.grid_size
        omega = 0.1
        sigma = self.grid_size / 4

        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        X, Y = np.meshgrid(x, y)

        r_dist = np.sqrt((X - self.grid_size/2)**2 + (Y - self.grid_size/2)**2)
        wave = 1 + alpha * np.cos(k * r_dist - omega * t)
        localization = np.exp(-r_dist**2 / (2 * sigma**2))

        density = rho0 * wave * localization
        self.field = density
        return density

    def entropy_gradient(self) -> np.ndarray:
        """∇S = -k_B * ∇(ρ log ρ)"""
        rho = self.field + 1e-10
        s = -rho * np.log(rho)
        grad_s = np.gradient(s)
        return np.array(grad_s)

    def mutual_information(self, region_A: Tuple[int, int, int, int],
                          region_B: Tuple[int, int, int, int]) -> float:
        """I(A:B) = H(A) + H(B) - H(A,B)"""
        x1, y1, x2, y2 = region_A
        region_A_field = self.field[x1:x2, y1:y2]

        x1, y1, x2, y2 = region_B
        region_B_field = self.field[x1:x2, y1:y2]

        def entropy(field):
            p = field.flatten() / (np.sum(field) + 1e-10)
            p = p[p > 1e-10]
            return -np.sum(p * np.log(p))

        H_A = entropy(region_A_field)
        H_B = entropy(region_B_field)
        joint = np.concatenate([region_A_field.flatten(), region_B_field.flatten()])
        H_AB = entropy(joint)
        MI = H_A + H_B - H_AB
        return max(0, MI)

    def topological_charge(self) -> float:
        """Q = (1/2π) ∫∫ ε_ij ∂_i φ ∂_j φ dxdy"""
        phase = np.angle(self.field + 1j * np.roll(self.field, 1, axis=0))
        dphi_dx, dphi_dy = np.gradient(phase)
        charge_density = (dphi_dx * np.roll(dphi_dy, 1, axis=1) -
                         dphi_dy * np.roll(dphi_dx, 1, axis=0))
        Q = np.sum(charge_density) / (2 * np.pi)
        return Q
