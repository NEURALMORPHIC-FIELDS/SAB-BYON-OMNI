# -*- coding: utf-8 -*-
"""Training pipeline for SAB + BYON-OMNI 3B model."""

import os
import glob
import time as _time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sab_byon_omni.model.config import RealMultimodalDataset


def _save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer: optim.Optimizer,
                     loss: float, best_loss: float, global_step: int, config: dict) -> bool:
    """Save checkpoint with error handling and verification."""
    try:
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss,
            'best_loss': best_loss,
            'global_step': global_step,
            'config': config,
        }, path)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [SAVED] {path} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  [ERROR] Save appeared to succeed but file not found: {path}")
            return False
    except Exception as e:
        print(f"  [ERROR] Failed to save checkpoint: {e}")
        return False


def train_3b_rapid(sab, epochs=3, batch_size=4, grad_accum=16, seq_len=1024,
                   log_every_batches=5, num_workers=0,
                   target_samples=500_000, cache_dir=None, local_data_dir=None,
                   checkpoint_dir=None):
    """
    Train 3B model with visible progress, correct causal LM loss, and checkpoint saving.

    Loads real training data from HuggingFace (WikiText-103, CodeSearchNet, Wikipedia)
    and local files from Google Drive. Saves checkpoints after each epoch and supports
    intelligent checkpoint recovery.

    Args:
        target_samples: Number of samples to load (default 500K).
        cache_dir: Path to cache processed dataset on Drive.
        local_data_dir: Path to training_data/ folder on Drive.
        checkpoint_dir: Path to save/load checkpoints. If None, no checkpoints are saved.
    """
    print("\nTRAINING 3B MODEL (optimized)")
    print(f"  Config: epochs={epochs}, batch_size={batch_size}, grad_accum={grad_accum}, seq_len={seq_len}")
    model = sab.llm.model
    device = sab.llm.device
    model.to(device)

    dataset = RealMultimodalDataset(
        max_len=seq_len,
        target_samples=target_samples,
        cache_dir=cache_dir,
        local_data_dir=local_data_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    total_batches = len(dataloader)
    total_steps_est = (total_batches // grad_accum) * epochs
    print(f"  Dataset: {len(dataset)} samples, {total_batches} batches/epoch, ~{total_batches // grad_accum} steps/epoch")
    print(f"  Estimated total optimizer steps: ~{total_steps_est}")

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    global_steps = 0
    best_loss = float('inf')
    start_epoch = 0

    # ---- Checkpoint recovery ----
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"  Checkpoint dir: {checkpoint_dir}")

        existing = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')))
        if existing:
            latest = existing[-1]
            try:
                ckpt = torch.load(latest, map_location=device)
                model.load_state_dict(ckpt['model_state'])
                optimizer.load_state_dict(ckpt['optimizer_state'])
                start_epoch = ckpt['epoch'] + 1
                global_steps = ckpt.get('global_step', 0)
                best_loss = ckpt.get('best_loss', float('inf'))
                print(f"  [RECOVERY] Loaded {latest}")
                print(f"  [RECOVERY] Resuming from epoch {start_epoch}, step {global_steps}, best_loss={best_loss:.4f}")
            except Exception as e:
                print(f"  [WARN] Checkpoint corrupt: {e} - starting fresh")
                start_epoch = 0
        else:
            print("  [INFO] No existing checkpoints found")
    else:
        print("  [INFO] No checkpoint_dir - checkpoints will NOT be saved")

    if start_epoch >= epochs:
        print(f"  [INFO] Training already complete ({start_epoch}/{epochs} epochs)")
        return True

    # ---- Training loop ----
    training_start = _time.perf_counter()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_start = _time.perf_counter()
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")

        for idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                # CAUSAL LM SHIFT
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss = criterion(shift_logits, shift_labels) / grad_accum

            scaler.scale(loss).backward()

            if (idx + 1) % grad_accum == 0 or (idx + 1) == total_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_steps += 1
                step_elapsed = _time.perf_counter() - training_start
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"  STEP {global_steps:04d}/{total_steps_est} | batch {idx+1}/{total_batches} | "
                      f"loss={loss.item()*grad_accum:.4f} | "
                      f"VRAM: {allocated:.1f}/{reserved:.1f}GB | {step_elapsed:.0f}s")

            epoch_loss += loss.item() * grad_accum

            if (idx + 1) % log_every_batches == 0 and (idx + 1) % grad_accum != 0:
                elapsed = _time.perf_counter() - epoch_start
                pct = (idx + 1) / total_batches * 100
                print(f"    batch {idx+1}/{total_batches} ({pct:.0f}%) | "
                      f"loss={loss.item()*grad_accum:.4f} | {elapsed:.0f}s")

        avg_epoch_loss = epoch_loss / total_batches
        epoch_elapsed = _time.perf_counter() - epoch_start
        print(f"  -> Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f} | time: {epoch_elapsed:.0f}s")

        # Save checkpoint
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pt')
            _save_checkpoint(ckpt_path, epoch, model, optimizer,
                             avg_epoch_loss, best_loss, global_steps,
                             {'epochs': epochs, 'batch_size': batch_size,
                              'grad_accum': grad_accum, 'seq_len': seq_len})

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = os.path.join(checkpoint_dir, 'best_model.pt')
                _save_checkpoint(best_path, epoch, model, optimizer,
                                 avg_epoch_loss, best_loss, global_steps,
                                 {'epochs': epochs, 'batch_size': batch_size,
                                  'grad_accum': grad_accum, 'seq_len': seq_len})
                print(f"  [BEST] New best loss: {best_loss:.6f}")

    total_time = _time.perf_counter() - training_start
    print(f"\n3B MODEL TRAINING DONE - {global_steps} steps in {total_time:.0f}s")
    print(f"  Best loss: {best_loss:.6f}")
    if checkpoint_dir:
        print(f"  Checkpoints: {checkpoint_dir}")
    return True
