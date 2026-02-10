<div align="center">

<img src="../logo.png" alt="SAB-BYON-OMNI-AI" width="350"/>

# Benchmarks & Evaluation
### SAB + BYON-OMNI v2.1

</div>

---

## Industrial LLM Benchmark Suite

SAB-BYON-OMNI is evaluated using the same protocols as GPT-4, LLaMA, and Mistral.

## Benchmark Descriptions

### 1. Perplexity

**What**: Language modeling quality measured as exp(average negative log-likelihood).

**Method**: 20-sentence scientific corpus, causal LM shift, token-level NLL.

**Scoring**: `score = 100 * (1 - log(PPL) / log(vocab_size))`

**Reference**: Lower PPL = better. GPT-4 achieves ~10-15 PPL on standard corpora.

---

### 2. MMLU (Massive Multitask Language Understanding)

**What**: Multiple-choice knowledge and reasoning across subjects.

**Method**: 20 questions across computer science, machine learning, deep learning, neural networks, statistics, information theory, calculus, generative models.

**Scoring**: Accuracy % (model picks answer with lowest loss).

**Reference**:
| Model | Score |
|-------|-------|
| GPT-4 | ~86.4% |
| LLaMA-70B | ~69.8% |
| Mistral-7B | ~60.1% |
| Random | ~25.0% |

---

### 3. HellaSwag (Commonsense NLI)

**What**: Commonsense reasoning - predict the most plausible continuation.

**Method**: 10 scenarios with 1 plausible + 3 absurd endings. Score only the continuation portion (not the context).

**Reference**:
| Model | Score |
|-------|-------|
| GPT-4 | ~95.3% |
| LLaMA-70B | ~87.3% |
| Mistral-7B | ~81.3% |

---

### 4. ARC (AI2 Reasoning Challenge)

**What**: Science reasoning with grade-school level questions.

**Method**: 10 multiple-choice questions about ML/AI concepts.

**Reference**:
| Model | Score |
|-------|-------|
| GPT-4 | ~96.3% |
| LLaMA-70B | ~85.3% |
| Mistral-7B | ~78.5% |

---

### 5. TruthfulQA

**What**: Truthfulness - does the model prefer factual over misleading answers?

**Method**: 10 paired items (truthful vs false statement). Model should assign lower loss to the truthful answer.

**Reference**:
| Model | Score |
|-------|-------|
| GPT-4 | ~59% |
| LLaMA-70B | ~50% |
| Mistral-7B | ~42% |

---

### 6. Coherence & Generation Quality

**What**: Output consistency, confidence, and diversity.

**Components**:
- **Top-1 Confidence** (40%): How confident is the model in predictions
- **Decisiveness** (30%): Low entropy = more focused predictions
- **Diversity** (30%): Unique token ratio in predictions

---

### 7. Speed & Efficiency

**What**: Inference throughput and memory usage.

**Metrics**:
- Tokens per second at seq_len 64, 128, 256, 512
- Parameter memory (GB)
- Peak VRAM usage
- Parameters per MB ratio

---

## Composite Scoring

```
Composite = 0.20 * Perplexity
          + 0.20 * MMLU
          + 0.15 * HellaSwag
          + 0.15 * ARC
          + 0.10 * TruthfulQA
          + 0.10 * Coherence
          + 0.10 * Speed
```

### Grading Scale

| Grade | Score | Level |
|-------|-------|-------|
| A+ | >= 90% | State-of-the-art |
| A | >= 80% | Excellent |
| B | >= 70% | Good |
| C | >= 60% | Fair |
| D | >= 50% | Below average |
| F | < 50% | Needs improvement |

## Running Benchmarks

### Via Colab
Use `colab/SAB_BYON_OMNI_v2.ipynb` Section 5.

### Via Script
```python
# After training, model is in eval mode
model.eval()

# Run individual benchmarks
from benchmark_utils import evaluate_perplexity, evaluate_mmlu
ppl_score = evaluate_perplexity(model, corpus)
mmlu_score = evaluate_mmlu(model, questions)
```

## Output Files

After benchmark completion:
- `results/benchmark_results.json` - All scores and details
- `results/benchmark_results.png` - Radar chart + bar chart visualization
- `results/training_curves.png` - Training loss curves
