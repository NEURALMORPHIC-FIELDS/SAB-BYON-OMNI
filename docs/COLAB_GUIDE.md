<div align="center">

<img src="../logo.png" alt="SAB-BYON-OMNI-AI" width="350"/>

# Google Colab Guide
### SAB + BYON-OMNI v2.0

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
    memory/         (5 files)
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

### Section 3: Source Code

**This is where you paste your code.** Each cell has a `write_source()` call.

**How to paste a file:**
1. Open the cell for the module (e.g., "3.1: sab_byon_omni/config.py")
2. Replace the placeholder text between `r'''` and `'''` with your code
3. Run the cell - it writes the file to disk

**Order matters!** Paste files in this order:
1. `config.py` (device setup)
2. `__init__.py` (main package)
3. `quantifiers/` (all 9 files)
4. `evolution/` (all 5 files)
5. `memory/` (all 5 files)
6. `agents/` (all 6 files)
7. `cognitive/` (all 6 files)
8. `consciousness/` (all 9 files)
9. `model/` (all 3 files)
10. `training/` (all 2 files)
11. `core/` (all 2 files)

After pasting all files, run cell **3.12: VERIFY ALL IMPORTS** to confirm everything works.

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
- Check that all source files are pasted in Section 3
- Run cells in order (config.py first)
- Run cell 3.12 to see the specific import error

### "Module not found"
- Ensure Section 1 (File System) was run first
- Check that PROJECT_ROOT is in sys.path

### Session disconnected
- Checkpoints are saved per epoch in `checkpoints/`
- Download checkpoints to Google Drive before session ends
