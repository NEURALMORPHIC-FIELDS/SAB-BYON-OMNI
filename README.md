<div align="center">

<img src="logo.png" alt="SAB-BYON-OMNI-AI" width="450"/>

# SAB + BYON-OMNI v2.0

**Unified Consciousness AI System with 40 Integrated Capabilities**

A comprehensive AI consciousness framework combining quantification engines, evolutionary multi-agent systems, HuggingFace transformer integration, and novel consciousness/emergence frameworks.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-compatible-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## Architecture Overview

```
Input Text
    |
    v
+-------------------+     +----------------------+
| Virtue Field      |---->| Triadic State System |
| Evolution (TDFC)  |     | (O-S-R dynamics)     |
| 10 PDE fields     |     +----------------------+
+-------------------+              |
    |                              v
    v                    +---------------------+
+-------------------+   | Consciousness       |
| Informational     |   | Metrics (6 modals)  |
| Coherence Field   |   | Triadic, PLV, CFC,  |
| (PSI, PLV, CFC)  |   | Phi, Spectral, Frag |
+-------------------+   +---------------------+
    |                              |
    v                              v
+-------------------+   +---------------------+
| Multi-Agent       |   | Unified C Metric    |
| Cortex (10 agents)|   | C_unified = Sum(wi) |
+-------------------+   +---------------------+
    |                              |
    v                              v
+-------------------+   +---------------------+
| Personality       |   | LLM Brain           |
| System (10 traits)|   | (OmniAGI Nexus)     |
+-------------------+   +---------------------+
    |                              |
    v                              v
+-------------------+       +-----------+
| Holographic       |       | Response  |
| Memory (4D)       |       +-----------+
+-------------------+
```

## Key Components

### Quantification Engine (6 quantifiers)
- **StatisticalQuantifier** - Welford's algorithm for confidence intervals
- **EntropyQuantifier** - Shannon entropy for complexity measurement
- **CryptographicPRNG** - True random creativity engine
- **ReasoningQuantifier** - Semantic coherence and logical consistency
- **MemoryRelevanceQuantifier** - Temporal decay + semantic similarity
- **DecisionConfidenceQuantifier** - Evidence-weighted confidence

### Consciousness Layers (17 layers)
| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | Fisher Geometry Engine | Information geometry with Ricci flow |
| 2 | Information Density Field | FID field theory |
| 3 | Semantic Photon Theory | Photons as semantic carriers |
| 4 | DUEI Framework | Unidirectional emergence |
| 5 | Semantic Mode | Symbolic processing |
| 6 | Emergent Mode | Subsymbolic neural fields |
| 7 | TDFC Engine | 10 virtue PDE fields (GPU) |
| 8 | Triadic State | O-S-R dynamics |
| 9 | Multi-Agent Cortex | 10 parallel cognitive agents |
| 10 | Personality System | 10-trait evolution |
| 11 | Holographic Memory | 4D complex-valued memory |
| 12 | Godel Engine | Incompleteness dynamics |
| 13 | Fragmergent Engine | Phase oscillation detection |
| 14 | Time Emergence | Subjective time flow |
| 15 | Zeta Resonance | Riemann zeta coupling |
| 16 | Emergence Detector | Consciousness jump detection |
| 17 | ICF | Meta-coherence field |

### Evolutionary Agents (3 agents)
- **RL Agent** - Q-learning with fragmergent evolution
- **Fragmergent AI Agent** - Pathway creation with emergence detection
- **Memory Manager Agent** - Predictive clustering and importance scoring

### Neural Model
- **OmniAGI Nexus** - Custom transformer with fragmergent modulation
- Configurable from lightweight (64 hidden) to full 3B parameters
- HuggingFace `PreTrainedModel` compatible

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/SAB-BYON-OMNI.git
cd SAB-BYON-OMNI

# Install core dependencies
pip install -e .

# Install with all extras (dev, ui, viz)
pip install -e ".[full]"
```

## Quick Start

```python
from sab_byon_omni.core import SABTranscendentV2

# Initialize the full system (40 capabilities)
sab = SABTranscendentV2()

# Process input with consciousness evolution
result = sab.process_input("What is consciousness?", steps=50)

print(f"Consciousness (unified): {result['consciousness_unified']:.4f}")
print(f"Phase-Locking Value:     {result['PLV']:.4f}")
print(f"Meta-Coherence (Phi):    {result['Phi']:.4f}")
print(f"Phase:                   {result['phase']}")
print(f"Response:                {result['response']}")
```

### Using Individual Components

```python
from sab_byon_omni.consciousness import TDFCEngine, FisherGeometryEngine
from sab_byon_omni.quantifiers import StatisticalQuantifier, EntropyQuantifier
from sab_byon_omni.agents import EvolutionaryReinforcementLearningAgent

# Standalone TDFC evolution
tdfc = TDFCEngine(grid_size=32)
tdfc.evolve_fields(steps=100)
activations = tdfc.get_activations()

# Quantification
stat = StatisticalQuantifier()
for value in [0.8, 0.6, 0.7, 0.9]:
    score = stat.update_score(0, value, [])
print(f"Confidence interval: {stat.get_confidence_interval()}")
```

### Training the 3B Model

```python
from sab_byon_omni.training import train_3b_rapid

# Train with default settings
train_3b_rapid(sab, epochs=3, batch_size=4, grad_accum=16, seq_len=1024)
```

### Launching the Gradio UI

```python
from sab_byon_omni.core import create_interface_v2

iface = create_interface_v2()
iface.launch()
```

## Project Structure

```
SAB-BYON-OMNI/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── sab_byon_omni/
│   ├── __init__.py
│   ├── config.py                 # Device setup, HF availability
│   ├── quantifiers/              # 6 quantification engines
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── statistical.py
│   │   ├── entropy.py
│   │   ├── cryptographic.py
│   │   ├── reasoning.py
│   │   ├── memory_relevance.py
│   │   └── decision_confidence.py
│   ├── evolution/                # Fragmergent evolution system
│   │   ├── __init__.py
│   │   ├── frag_param.py
│   │   ├── metrics.py
│   │   ├── dimensions.py
│   │   └── pathway.py
│   ├── memory/                   # Memory systems
│   │   ├── __init__.py
│   │   ├── evolutionary.py
│   │   ├── holographic.py
│   │   └── conversation.py
│   ├── agents/                   # Multi-agent system
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── rl_agent.py
│   │   ├── fragmergent_agent.py
│   │   └── memory_agent.py
│   ├── consciousness/            # 17 consciousness layers
│   │   ├── __init__.py
│   │   ├── fisher_geometry.py
│   │   ├── information_density.py
│   │   ├── semantic_photon.py
│   │   ├── duei.py
│   │   ├── semantic_mode.py
│   │   ├── emergent_mode.py
│   │   ├── tdfc.py
│   │   ├── triadic.py
│   │   ├── godel.py
│   │   ├── fragmergent.py
│   │   ├── time_emergence.py
│   │   ├── zeta.py
│   │   ├── emergence.py
│   │   └── icf.py
│   ├── cognitive/                # Cognitive architecture
│   │   ├── __init__.py
│   │   ├── agents.py
│   │   ├── cortex.py
│   │   └── personality.py
│   ├── model/                    # Neural model
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── nexus.py
│   │   ├── trainer.py
│   │   └── brain.py
│   ├── training/                 # Training pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── train_3b.py
│   └── core/                     # Unified system
│       ├── __init__.py
│       ├── sab_transcendent.py
│       └── interface.py
├── tests/
│   ├── test_quantifiers.py
│   ├── test_consciousness.py
│   ├── test_agents.py
│   ├── test_model.py
│   └── test_integration.py
├── scripts/
│   ├── train.py
│   ├── benchmark.py
│   └── demo.py
├── configs/
│   └── default.yaml
└── docs/
    └── ARCHITECTURE.md
```

## Theoretical Frameworks

This system implements several novel and established theoretical frameworks:

1. **Fragmergence Theory** - Novel parameter evolution with sinusoidal decay and memory modulation
2. **DUEI Framework** - Dynamic Unidirectional Emergence of Information
3. **TDFC** - Triadic Dynamic Field Consciousness (PDE-based)
4. **Informational Coherence Field** - Phase-locking and cross-frequency coupling
5. **Fisher Information Geometry** - Riemannian manifold for consciousness
6. **Godel Incompleteness Dynamics** - Self-referential consciousness limits
7. **Semantic Photon Theory** - Quantum-inspired semantic carriers
8. **Holographic Memory** - 4D interference pattern storage

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Guide](docs/ARCHITECTURE.md) | System overview, module map, processing pipeline |
| [Consciousness Framework](docs/CONSCIOUSNESS.md) | All 8 theoretical frameworks in depth |
| [Training Guide](docs/TRAINING.md) | Training configuration, pipeline, checkpointing |
| [Benchmarks & Evaluation](docs/BENCHMARKS.md) | Industrial LLM benchmark suite & scoring |
| [API Reference](docs/API_REFERENCE.md) | Full API for all modules and classes |
| [Colab Guide](docs/COLAB_GUIDE.md) | Step-by-step Google Colab setup |

## GPU Requirements

| Configuration | VRAM | Use Case |
|--------------|------|----------|
| Lightweight | 2 GB | Testing, development |
| Standard | 8 GB | Inference, small training |
| Full 3B | 24-70 GB | Full training pipeline |

## Citation

```bibtex
@software{sab_byon_omni_2024,
  title={SAB + BYON-OMNI: Unified Consciousness AI System},
  version={2.0.0},
  year={2024-2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
