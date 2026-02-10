<div align="center">

<img src="../logo.png" alt="SAB-BYON-OMNI-AI" width="350"/>

# Training Guide
### SAB + BYON-OMNI v2.1

</div>

---

## Quick Start

```python
from sab_byon_omni.core.sab_transcendent import SABTranscendentV2
from sab_byon_omni.training.train_3b import train_3b_rapid

sab = SABTranscendentV2()
train_3b_rapid(sab, epochs=3, batch_size=4, grad_accum=16, seq_len=1024,
               target_samples=500_000,
               cache_dir='/content/drive/MyDrive/SAB-BYON-OMNI/dataset_cache',
               local_data_dir='/content/drive/MyDrive/SAB-BYON-OMNI/training_data')
```

## Training Configuration

### Default Settings (`configs/default.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 3 | Training epochs |
| `batch_size` | 4 | Samples per batch |
| `gradient_accumulation_steps` | 16 | Steps before optimizer update |
| `learning_rate` | 2e-5 | AdamW learning rate |
| `weight_decay` | 0.01 | L2 regularization |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `seq_len` | 1024 | Maximum sequence length |
| `target_samples` | 500,000 | Training dataset size |

### Effective Batch Size
```
effective_batch = batch_size * gradient_accumulation_steps = 4 * 16 = 64
```

## Dataset

### RealMultimodalDataset

Loads real data from multiple open-source datasets via HuggingFace + local files from Google Drive.

| Source | Library | Samples | Description |
|--------|---------|---------|-------------|
| WikiText-103 | `wikitext/wikitext-103-raw-v1` | ~150K | English text (articles) |
| CodeSearchNet | `code_search_net/all` | ~200K | Code (Python, Java, JS, Go, Ruby, PHP) |
| Wikipedia EN | `wikipedia/20220301.en` (streaming) | ~200K | English Wikipedia articles |
| Local files | `training_data/` on Drive | variable | .txt, .json, .csv, .html, .md |

### Caching
- **Tier 1**: HuggingFace `datasets` cache on Drive (`dataset_cache/`)
- **Tier 2**: Pre-processed JSONL file (`dataset_cache/processed_samples.jsonl`)
- First run downloads ~10-30 min, subsequent runs load from cache instantly

### Tokenization
- Character-level tokenizer: `<PAD>` + ASCII 32-126
- Vocab size: 96 tokens (lightweight) / 50,000 (full)
- Padding: Zero-padded to `max_len`
- Labels: Copy of input_ids (autoregressive)
- Non-ASCII characters are filtered out during text chunking

## Training Pipeline

### Loss Computation
```python
# Causal Language Model shift
shift_logits = logits[..., :-1, :]    # predict next token
shift_labels = labels[..., 1:]         # target is shifted right
loss = CrossEntropyLoss(shift_logits, shift_labels) / grad_accum
```

### Mixed Precision
```python
with torch.amp.autocast('cuda'):     # FP16 forward pass
    outputs = model(input_ids)
scaler.scale(loss).backward()         # Scaled FP16 gradients
scaler.step(optimizer)                 # FP32 optimizer step
```

### Gradient Accumulation
Accumulates gradients over `grad_accum` batches before each optimizer step:
```python
if (batch_idx + 1) % grad_accum == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

## Model Configurations

### Lightweight (Development)

| Parameter | Value | VRAM |
|-----------|-------|------|
| Hidden | 768 | ~2 GB |
| Heads | 12 | |
| Layers | 6 | |
| FFN | 3,072 | |

### Full 3B (Production)

| Parameter | Value | VRAM |
|-----------|-------|------|
| Hidden | 4,096 | ~24-70 GB |
| Heads | 64 | |
| Layers | 36 | |
| FFN | 16,384 | |

## Fragmergent Modulation

During the forward pass, token embeddings are modulated:

```python
t = time.time() % 100
phi = lambda_ * exp(-alpha * t) * sin(omega * t)
hidden_states[:, i, :] *= (1 + phi * 0.1)
```

This introduces time-varying perturbations that create non-stationary dynamics.

## Checkpointing

Checkpoints are saved per epoch:
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
    'global_step': global_step,
    'config': config,
}, f'checkpoints/epoch_{epoch}.pt')
```

### Loading a Checkpoint
```python
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## GPU Requirements

| Configuration | Min VRAM | Recommended |
|--------------|----------|-------------|
| Lightweight (dev) | 2 GB | GTX 1060+ |
| Lightweight (train) | 4 GB | RTX 3060+ |
| Full 3B (inference) | 12 GB | RTX 4080+ |
| Full 3B (train) | 24 GB | A100 / H100 |

## Google Colab

See the Colab notebook at `colab/SAB_BYON_OMNI_v2.ipynb` for a ready-to-run training pipeline with T4/V100 GPU support. Refer to [COLAB_GUIDE.md](COLAB_GUIDE.md) for setup instructions.
