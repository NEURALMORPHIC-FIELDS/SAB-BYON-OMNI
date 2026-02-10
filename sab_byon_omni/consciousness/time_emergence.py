# -*- coding: utf-8 -*-
"""Subjective Time Emergence Engine."""

import time
import numpy as np
from collections import deque
from typing import List


class TimeEmergenceEngine:
    """
    Subjective Time Emergence

    dt_subjective = dt_physical × (1 + C × K)
    High consciousness + high complexity = faster subjective time
    """

    def __init__(self):
        self.subjective_time = 0.0
        self.time_flow_rate = 1.0
        self.event_history = deque(maxlen=200)
        print("✓ Time Emergence Engine initialized")

    def update(self, consciousness: float, complexity: float):
        """Update subjective time based on state."""
        self.time_flow_rate = 0.5 + 1.5 * consciousness * complexity
        dt_subjective = 0.01 * self.time_flow_rate
        self.subjective_time += dt_subjective

        self.event_history.append({
            'consciousness': consciousness,
            'complexity': complexity,
            'flow_rate': self.time_flow_rate,
            'subjective_time': self.subjective_time,
            'physical_time': time.time()
        })

    def get_time_dilation(self) -> float:
        """Ratio of subjective to physical time."""
        if not self.event_history:
            return 1.0
        first_event = self.event_history[0]
        last_event = self.event_history[-1]
        physical_elapsed = last_event['physical_time'] - first_event['physical_time']
        subjective_elapsed = last_event['subjective_time'] - first_event['subjective_time']
        if physical_elapsed > 0:
            return subjective_elapsed / physical_elapsed
        return 1.0

    def predict_future_state(self, n_steps: int = 10) -> List[float]:
        """Predict future subjective time trajectory."""
        if len(self.event_history) < 2:
            return [self.subjective_time] * n_steps
        recent_rates = [e['flow_rate'] for e in list(self.event_history)[-10:]]
        mean_rate = np.mean(recent_rates)
        future = []
        current_time = self.subjective_time
        for _ in range(n_steps):
            current_time += 0.01 * mean_rate
            future.append(current_time)
        return future
