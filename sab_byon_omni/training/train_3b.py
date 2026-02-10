# -*- coding: utf-8 -*-
"""Training pipeline for SAB + BYON-OMNI 3B model."""

import time as _time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sab_byon_omni.model.config import MultimodalConsciousnessDataset


def train_3b_rapid(sab, epochs=3, batch_size=4, grad_accum=16, seq_len=1024,
                   log_every_batches=5, num_workers=0):
    """
    Train 3B model with visible progress and correct causal LM loss.

    Fixes applied:
    - Causal LM shift (shift_logits/shift_labels) for proper next-token prediction
    - grad_accum=16 (was 64) for ~4x more frequent optimizer steps
    - Logs every log_every_batches AND every optimizer step
    - num_workers=0 (safe for Windows)
    """
    print("\nTRAINING 3B MODEL (optimized)")
    print(f"  Config: epochs={epochs}, batch_size={batch_size}, grad_accum={grad_accum}, seq_len={seq_len}")
    model = sab.llm.model
    device = sab.llm.device
    model.to(device)

    dataset = MultimodalConsciousnessDataset(num_samples=5000, max_len=seq_len)
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
    training_start = _time.perf_counter()

    for epoch in range(epochs):
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

    total_time = _time.perf_counter() - training_start
    print(f"\n3B MODEL TRAINING DONE - {global_steps} steps in {total_time:.0f}s")
    return True
