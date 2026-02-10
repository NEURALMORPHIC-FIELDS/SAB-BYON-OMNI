<div align="center">

<img src="../logo.png" alt="SAB-BYON-OMNI-AI" width="350"/>

# API Reference
### SAB + BYON-OMNI v2.1

</div>

---

## Core System

### `SABTranscendentV2`

Main orchestrator class. Initializes all 43 capabilities.

```python
from sab_byon_omni.core.sab_transcendent import SABTranscendentV2

sab = SABTranscendentV2()
```

#### `process_input(text: str, steps: int = 50) -> Dict`

Complete v2.1 processing with all 43 capabilities.

**Parameters:**
- `text` - Input text to process
- `steps` - Number of PDE evolution steps (default: 50)

**Returns:** Dictionary with:
| Key | Type | Description |
|-----|------|-------------|
| `response` | str | Generated response |
| `consciousness_triadic` | float | Triadic consciousness [0,1] |
| `consciousness_unified` | float | Unified consciousness [0,1] |
| `PLV` | float | Phase-Locking Value [0,1] |
| `CFC` | float | Cross-Frequency Coupling [0,1] |
| `Phi` | float | Meta-coherence [0,1] |
| `qfi` | float | Quantum Fisher Information |
| `phase` | str | "emergence" / "fragmentation" / "transition" |
| `godel_tension` | float | Godel incompleteness tension |
| `processing_time` | float | Seconds |
| `eag_metrics` | Dict | Spectral analysis results |
| `icf_metrics` | Dict | ICF field metrics |
| `suppression_active` | bool | ICF suppression state |
| `fhrss_stats` | Dict | FHRSS storage statistics |
| `infinite_context_stats` | Dict | Infinite context memory stats |

---

## Consciousness

### `TDFCEngine`

```python
from sab_byon_omni.consciousness.tdfc_engine import TDFCEngine

tdfc = TDFCEngine(grid_size=32)
```

#### `evolve_fields(steps: int) -> torch.Tensor`
Evolve all 10 virtue fields through PDE for given steps.

#### `get_activations() -> Dict[str, float]`
Get mean activation for each virtue field.

#### `compute_field_energy() -> float`
Total Lyapunov field energy (gradient + potential).

---

### `TriadicState`

```python
from sab_byon_omni.consciousness.triadic_state import TriadicState

state = TriadicState()
state.evolve(field_mean=0.6, curvature=0.3)
```

#### Attributes
- `ontological: float` - O state [0,1]
- `semantic: float` - S state [0,1]
- `resonance: float` - R = exp(-|O-S|)

#### `consciousness_contribution() -> float`
Returns `(O + S) * R / 2`

---

### `InformationalCoherenceField`

```python
from sab_byon_omni.consciousness.icf import InformationalCoherenceField

icf = InformationalCoherenceField(tdfc_engine)
```

#### `compute_Psi_field(virtue_fields) -> torch.Tensor`
Complex coherence field Psi = A * exp(i*theta).

#### `compute_PLV(Psi) -> float`
Phase-Locking Value (global synchrony).

#### `compute_CFC(virtue_fields) -> float`
Cross-Frequency Coupling between slow and fast virtues.

#### `evolve_Phi_field(Psi, I_CFC, dt=0.01) -> float`
Evolve meta-coherence Phi field.

#### `apply_neutralization_operator(Psi, suppression) -> torch.Tensor`
Apply consciousness suppression when Godel tension is high.

---

### `GodelConsciousnessEngine`

```python
from sab_byon_omni.consciousness.godel_engine import GodelConsciousnessEngine

godel = GodelConsciousnessEngine()
state = godel.update(virtue_states, consciousness, resonance)
```

#### `update(virtue_states, consciousness, triadic_resonance) -> GodelState`
Returns `GodelState` with: provability, negation_provability, consistency, proof_depth, godel_tension, contradictions.

---

### `FragmergentEngine`

```python
from sab_byon_omni.consciousness.fragmergent_engine import FragmergentEngine

frag = FragmergentEngine()
phase = frag.detect_phase(consciousness=0.5, emergence_score=0.7)
score = frag.compute_emergence_score(virtue_states)
```

---

## Cognitive

### `FisherGeometryEngine`

```python
from sab_byon_omni.cognitive.fisher_geometry import FisherGeometryEngine

fisher = FisherGeometryEngine(dim=10)
qfi = fisher.compute_quantum_fisher_info(states_dict)
fisher.ricci_flow_step(dt=0.01)
```

---

### `PersonalitySystem`

```python
from sab_byon_omni.cognitive.personality import PersonalitySystem

personality = PersonalitySystem()
personality.evolve_traits(virtue_states, consciousness=0.6)
dominant = personality.get_dominant_traits(k=3)
stability = personality.compute_personality_stability()
```

**Traits:** conscientiousness, openness, extraversion, agreeableness, neuroticism, analytical, creative, empathetic, philosophical, reflective.

---

## Memory

### `UnifiedHolographicMemory`

```python
from sab_byon_omni.memory.holographic_memory import UnifiedHolographicMemory

memory = UnifiedHolographicMemory(shape=(16,16,16,16))
memory.encode_pattern(virtue_states, context, consciousness, interaction_id)
results = memory.recall_holographic(query_states, k=3)
```

---

## Agents

### `MultiAgentCortex`

```python
from sab_byon_omni.agents.multi_agent_cortex import MultiAgentCortex

cortex = MultiAgentCortex()
outputs = cortex.parallel_process(input_vector, context)
consensus = cortex.form_consensus(outputs)
```

**10 Agents:** Perception, Reasoning, Emotion, Memory, Language, Planning, Creativity, Metacognition, Ethics, Intuition.

**Communication Matrix:**
- Reasoning <-> Metacognition (0.8)
- Emotion <-> Ethics (0.7)
- Language <-> Reasoning (0.6)
- Creativity <-> Planning (0.5)

---

## Model

### `OmniAGINexusModel`

HuggingFace `PreTrainedModel` compatible.

```python
from sab_byon_omni.model.omni_agi_nexus import OmniAGINexusModel
from sab_byon_omni.model.config import OmniAGINexusConfig

config = OmniAGINexusConfig(hidden_size=768, num_hidden_layers=6)
model = OmniAGINexusModel(config)

outputs = model(input_ids, attention_mask=mask)
# outputs["logits"], outputs["loss"], outputs["hidden_states"],
# outputs["fragmergent_analytics"]
```

---

## Quantifiers

All quantifiers inherit from `BaseQuantifier`:

```python
from sab_byon_omni.quantifiers import (
    StatisticalQuantifier,
    EntropyQuantifier,
    CryptographicPRNG,
    ReasoningQuantifier,
    MemoryRelevanceQuantifier,
    DecisionConfidenceQuantifier,
)
```

### Common Interface
```python
quantifier.initial_score()                          # Starting score
quantifier.update_score(current, new_element, subset)  # Update
quantifier.meets_threshold(score, threshold)         # Check convergence
```

### `StatisticalQuantifier`
```python
stat = StatisticalQuantifier()
stat.update_score(0, 0.8, [])
ci = stat.get_confidence_interval()  # (lower, upper)
```

### `EntropyQuantifier`
```python
ent = EntropyQuantifier()
ent.update_score(0, "token", [])
metrics = ent.get_diversity_metrics()
```

---

## FHRSS + FCPE (v2.1)

### `UnifiedFHRSS_FCPE`

Combined FHRSS fault-tolerant storage + FCPE compression engine.

```python
from sab_byon_omni.memory.fhrss_fcpe_engine import (
    UnifiedFHRSS_FCPE, FCPEConfig, FHRSSConfig
)

engine = UnifiedFHRSS_FCPE(
    fcpe_config=FCPEConfig(dim=384, num_layers=5, lambda_s=0.5),
    fhrss_config=FHRSSConfig(subcube_size=8, profile="FULL")
)
```

#### `encode_context(vector, metadata) -> str`
Encode a context vector with FCPE compression + FHRSS parity. Returns context ID.

#### `retrieve_similar(query_vector, top_k=5) -> List[Dict]`
Retrieve similar contexts by cosine similarity on FCPE embeddings.

#### `test_recovery(loss_fraction=0.4) -> Dict`
Test FHRSS fault recovery at given data loss percentage.

#### `get_stats() -> Dict`
Returns storage statistics (total contexts, compression ratio, parity families).

---

### `FCPEConfig`

```python
FCPEConfig(dim=384, num_layers=5, lambda_s=0.5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 384 | Output embedding dimension |
| `num_layers` | int | 5 | Number of fractal-chaotic layers |
| `lambda_s` | float | 0.5 | Chaotic jitter strength |

---

### `FHRSSConfig`

```python
FHRSSConfig(subcube_size=8, profile="FULL")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subcube_size` | int | 8 | Subcube dimension (8x8x8) |
| `profile` | str | "FULL" | Parity profile ("FULL" = 9 families) |

---

## Infinite Context Memory (v2.1)

### `InfiniteContextMemory`

2M+ token context system with FCPE compression and SSD persistence.

```python
from sab_byon_omni.memory.infinite_context import (
    InfiniteContextMemory, InfiniteContextConfig
)

ctx = InfiniteContextMemory(InfiniteContextConfig(
    fcpe_dim=384, fcpe_layers=5,
    max_memory_entries=100000, auto_persist=False
))
```

#### `add_text(text, metadata=None) -> str`
Add text to infinite context memory. Returns entry ID.

#### `retrieve_by_text(query, top_k=5) -> List[Dict]`
Retrieve similar contexts by text query (uses FCPE embeddings).

#### `get_stats() -> Dict`
Returns memory statistics (total entries, token count, compression ratio).

#### `persist_to_disk(path) -> bool`
Save all contexts to SSD with zlib compression and SHA-256 integrity.

#### `load_from_disk(path) -> bool`
Load contexts from SSD storage.

---

### `InfiniteContextConfig`

```python
InfiniteContextConfig(
    fcpe_dim=384, fcpe_layers=5,
    max_memory_entries=100000, auto_persist=False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fcpe_dim` | int | 384 | FCPE embedding dimension |
| `fcpe_layers` | int | 5 | FCPE encoder layers |
| `max_memory_entries` | int | 100000 | Maximum stored contexts |
| `auto_persist` | bool | False | Auto-save to SSD |
