# -*- coding: utf-8 -*-
"""EvolutionaryMetricsModule - Advanced metrics tracking for system evolution."""

import time
import psutil
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Callable
from functools import wraps

from sab_byon_omni.quantifiers import QuantificationResult


class EvolutionaryMetricsModule:
    """Sistem avansat de metrici pentru urmărirea evoluției algoritmului."""

    def __init__(self):
        self.metrics = defaultdict(lambda: {'exec_times': [], 'memory_usages': [], 'cpu_usages': []})
        self.agent_metrics = defaultdict(list)
        self.quantification_metrics = defaultdict(list)
        self.evolution_timeline = []
        self.process = psutil.Process()

        # Metrics pentru fiecare cuantificator
        self.statistical_metrics = []
        self.entropy_metrics = []
        self.creativity_metrics = []
        self.reasoning_metrics = []
        self.memory_relevance_metrics = []
        self.decision_confidence_metrics = []

    def track(self, module: str, func_name: str) -> Callable:
        """Enhanced decorator cu tracking evolutional."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self.process.memory_info().rss / 1024 / 1024
                start_cpu = psutil.cpu_percent(interval=None)

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in {module}.{func_name}: {e}")
                    raise

                end_time = time.time()
                end_memory = self.process.memory_info().rss / 1024 / 1024
                end_cpu = psutil.cpu_percent(interval=None)

                exec_time = end_time - start_time
                memory_usage = end_memory - start_memory
                cpu_usage = (start_cpu + end_cpu) / 2

                self.metrics[f"{module}.{func_name}"]['exec_times'].append(exec_time)
                self.metrics[f"{module}.{func_name}"]['memory_usages'].append(memory_usage)
                self.metrics[f"{module}.{func_name}"]['cpu_usages'].append(cpu_usage)

                # Evolution timeline tracking
                self.evolution_timeline.append({
                    "timestamp": time.time(),
                    "module": module,
                    "function": func_name,
                    "execution_time": exec_time,
                    "memory_delta": memory_usage,
                    "cpu_usage": cpu_usage
                })

                return result
            return wrapper
        return decorator

    def track_quantification_event(self, quantifier_type: str, result: QuantificationResult, metadata: Dict = None):
        """Urmărește evenimente de cuantificare."""
        event = {
            "timestamp": time.time(),
            "quantifier_type": quantifier_type,
            "steps": result.steps,
            "final_score": result.final_score,
            "execution_time": result.execution_time,
            "convergence_efficiency": result.final_score / result.steps if result.steps > 0 else 0.0,
            "metadata": metadata or {}
        }

        self.quantification_metrics[quantifier_type].append(event)

        # Store in specific metric lists
        if quantifier_type == "statistical":
            self.statistical_metrics.append(event)
        elif quantifier_type == "entropy":
            self.entropy_metrics.append(event)
        elif quantifier_type == "creativity":
            self.creativity_metrics.append(event)
        elif quantifier_type == "reasoning":
            self.reasoning_metrics.append(event)
        elif quantifier_type == "memory_relevance":
            self.memory_relevance_metrics.append(event)
        elif quantifier_type == "decision_confidence":
            self.decision_confidence_metrics.append(event)

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Generează sumar complet al evoluției sistemului."""
        if not self.evolution_timeline:
            return {"status": "No evolution data available"}

        # Analysis by time windows
        recent_events = [e for e in self.evolution_timeline if time.time() - e["timestamp"] < 3600]  # Last hour

        # Performance trends
        execution_times = [e["execution_time"] for e in recent_events]
        memory_usage = [e["memory_delta"] for e in recent_events]
        cpu_usage = [e["cpu_usage"] for e in recent_events]

        # Quantification effectiveness
        quantification_summary = {}
        for q_type, events in self.quantification_metrics.items():
            if events:
                recent_q_events = [e for e in events if time.time() - e["timestamp"] < 3600]
                if recent_q_events:
                    avg_efficiency = np.mean([e["convergence_efficiency"] for e in recent_q_events])
                    avg_steps = np.mean([e["steps"] for e in recent_q_events])
                    avg_score = np.mean([e["final_score"] for e in recent_q_events])

                    quantification_summary[q_type] = {
                        "avg_efficiency": avg_efficiency,
                        "avg_steps": avg_steps,
                        "avg_score": avg_score,
                        "usage_count": len(recent_q_events)
                    }

        return {
            "evolution_timeline_length": len(self.evolution_timeline),
            "recent_activity": len(recent_events),
            "performance_trends": {
                "avg_execution_time": np.mean(execution_times) if execution_times else 0.0,
                "avg_memory_usage": np.mean(memory_usage) if memory_usage else 0.0,
                "avg_cpu_usage": np.mean(cpu_usage) if cpu_usage else 0.0,
                "performance_stability": 1.0 - np.std(execution_times) / np.mean(execution_times) if execution_times and np.mean(execution_times) > 0 else 0.0
            },
            "quantification_effectiveness": quantification_summary,
            "system_health": self._assess_system_health()
        }

    def _assess_system_health(self) -> Dict[str, Any]:
        """Evaluează sănătatea sistemului."""
        if not self.evolution_timeline:
            return {"status": "insufficient_data", "score": 0.0}

        recent_events = [e for e in self.evolution_timeline if time.time() - e["timestamp"] < 1800]  # Last 30 min

        if not recent_events:
            return {"status": "inactive", "score": 0.3}

        execution_times = [e["execution_time"] for e in recent_events]
        memory_usage = [e["memory_delta"] for e in recent_events]

        # Health indicators
        avg_exec_time = np.mean(execution_times)
        memory_stability = 1.0 - (np.std(memory_usage) / (np.mean(np.abs(memory_usage)) + 1e-6))
        activity_level = len(recent_events) / 30.0

        # Combined health score
        health_score = (
            min(1.0, 1.0 / (avg_exec_time + 0.001)) * 0.3 +
            memory_stability * 0.3 +
            min(1.0, activity_level / 2.0) * 0.4
        )

        status = "excellent" if health_score > 0.8 else "good" if health_score > 0.6 else "fair" if health_score > 0.4 else "poor"

        return {
            "status": status,
            "score": health_score,
            "avg_execution_time": avg_exec_time,
            "memory_stability": memory_stability,
            "activity_level": activity_level
        }


# Module-level singleton metrics instance
metrics = EvolutionaryMetricsModule()
