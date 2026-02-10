<div align="center">

<img src="../logo.png" alt="SAB-BYON-OMNI-AI" width="350"/>

# Google Colab Guide
### SAB + BYON-OMNI v2.1

</div>

---

## Overview

The Colab notebook (`colab/SAB_BYON_OMNI_v2.ipynb`) provides a complete pipeline for training and evaluating SAB-BYON-OMNI on Google Colab with GPU acceleration.

## Setup Instructions

### Step 1: Open in Colab

Upload `colab/SAB_BYON_OMNI_v2.ipynb` to Google Colab, or open it directly from GitHub.

### Step 2: Select GPU Runtime

1. Go to **Runtime** > **Change runtime type**
2. Select **GPU** (T4 recommended, V100/A100 for full 3B)
3. Click **Save**

### Step 3: Upload Logo

Upload `logo.png` to Colab's `/content/` directory (drag & drop in the Files panel). The first cell will display it automatically.

### Step 4: Run Sections in Order

## Notebook Sections

### Section 1: File System

Creates the complete project directory tree:

```
/content/SAB-BYON-OMNI/
  sab_byon_omni/
    config.py
    quantifiers/    (9 files)
    evolution/      (5 files)
    memory/         (7 files, incl. FHRSS/FCPE + InfiniteContext)
    agents/         (6 files)
    cognitive/      (6 files)
    consciousness/  (9 files)
    model/          (3 files)
    training/       (2 files)
    core/           (2 files)
  configs/default.yaml
  checkpoints/
  results/
```

### Section 2: Dependencies

Installs all packages:
- PyTorch 2.0+ with CUDA 12.1
- transformers, datasets, accelerate
- scipy, matplotlib, seaborn, pandas
- lm-eval (EleutherAI benchmark harness)
- rouge-score, nltk, sacrebleu, scikit-learn

### Section 3: Source Code (Monolithic Deployment)

**v2.1**: All 51 source files are deployed automatically in a single monolithic cell. No manual code pasting required.

**What happens:**
1. A single cell contains all 51 source files (~6000 lines) embedded as `write_source()` calls
2. Running this cell writes the entire codebase to disk automatically
3. All modules are available immediately after execution

**Files deployed (51 total):**
- `config.py` + `__init__.py` (2 files)
- `quantifiers/` (9 files)
- `evolution/` (5 files)
- `memory/` (7 files, including FHRSS/FCPE and InfiniteContext)
- `agents/` (6 files)
- `cognitive/` (6 files)
- `consciousness/` (9 files)
- `model/` (3 files)
- `training/` (2 files)
- `core/` (2 files)

### Section 4: Training

Runs the full training pipeline:

| Setting | Value |
|---------|-------|
| Epochs | 3 |
| Batch size | 4 |
| Grad accumulation | 16 |
| Effective batch | 64 |
| Learning rate | 2e-5 |
| Mixed precision | FP16 |
| Sequence length | 1024 |
| Dataset | 5000 samples |

**Outputs:**
- Live training logs with loss, VRAM, grad norm
- Checkpoints in `/content/SAB-BYON-OMNI/checkpoints/`
- Training curves plot saved to `results/training_curves.png`

### Section 5: Benchmarks

Runs 7 industrial LLM benchmarks:

1. **Perplexity** - Language modeling quality
2. **MMLU** - Knowledge & reasoning (20 questions)
3. **HellaSwag** - Commonsense reasoning (10 items)
4. **ARC** - Science reasoning (10 questions)
5. **TruthfulQA** - Truthfulness (10 items)
6. **Coherence** - Output quality metrics
7. **Speed** - Tokens/sec throughput

**Outputs:**
- Final scorecard with composite grade
- Radar chart + bar chart visualization
- `results/benchmark_results.json` with all data

## GPU Memory Guide

| Colab GPU | VRAM | Supports |
|-----------|------|----------|
| T4 | 15 GB | Lightweight model training |
| V100 | 16 GB | Lightweight model training |
| A100 | 40 GB | Full 3B inference + limited training |
| A100 (80GB) | 80 GB | Full 3B training |

## Troubleshooting

### "CUDA out of memory"
- Reduce `batch_size` to 2 or 1
- Reduce `seq_len` to 512 or 256
- Use the lightweight model config (default)

### "Import failed"
- Ensure the monolithic Section 3 cell ran successfully
- Check output for any write errors

### "Module not found"
- Ensure Section 1 (File System) was run first
- Check that PROJECT_ROOT is in sys.path

### Session disconnected
- Checkpoints are saved per epoch in `checkpoints/`
- Download checkpoints to Google Drive before session ends
