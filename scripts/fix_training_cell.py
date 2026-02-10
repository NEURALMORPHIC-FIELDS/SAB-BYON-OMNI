#!/usr/bin/env python3
"""Rewrite training loop cell with intelligent checkpoint recovery."""

import json

NOTEBOOK_PATH = 'colab/SAB_BYON_OMNI_v2.ipynb'

with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cell10 = '''# ---- TRAINING LOOP WITH INTELLIGENT CHECKPOINT RECOVERY ----
import glob

optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg['learning_rate'],
    weight_decay=cfg['weight_decay']
)
criterion = nn.CrossEntropyLoss(ignore_index=0)
scaler = torch.amp.GradScaler('cuda')

# Training history
history = {
    'batch_loss': [],
    'step_loss': [],
    'epoch_loss': [],
    'learning_rates': [],
    'vram_usage': [],
    'step_times': [],
}

model.train()
global_step = 0
best_loss = float('inf')

# ============================================================================
# INTELLIGENT CHECKPOINT RECOVERY
# ============================================================================
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_*.pt')))
START_EPOCH = 0

if checkpoints:
    latest_checkpoint = checkpoints[-1]
    try:
        checkpoint_data = torch.load(latest_checkpoint, map_location='cuda:0')

        model.load_state_dict(checkpoint_data['model_state'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state'])
        START_EPOCH = checkpoint_data['epoch'] + 1
        global_step = checkpoint_data.get('global_step', 0)
        best_loss = checkpoint_data.get('best_loss', float('inf'))
        if 'history' in checkpoint_data:
            history = checkpoint_data['history']

        print('\\n' + '='*70)
        print('[RECOVERY] CHECKPOINT LOADED')
        print('='*70)
        print(f'  File: {latest_checkpoint}')
        print(f'  Epoch completed: {checkpoint_data["epoch"]}')
        print(f'  Resuming from epoch: {START_EPOCH}')
        print(f'  Global step: {global_step}')
        print(f'  Best loss: {best_loss:.6f}')
        print('='*70 + '\\n')

    except Exception as e:
        print(f'[WARN] Checkpoint corrupt: {e}')
        print('[INFO] Starting fresh training from epoch 0')
        START_EPOCH = 0
else:
    print('[INFO] No checkpoint found - starting fresh training')
    print(f'  Checkpoints will save to: {CHECKPOINT_DIR}')

# ============================================================================
# TRAINING LOOP - STARTS FROM START_EPOCH
# ============================================================================
t_start = time.perf_counter()
num_epochs = cfg['epochs']

if START_EPOCH >= num_epochs:
    print(f'\\n[INFO] Training already complete ({START_EPOCH}/{num_epochs} epochs done)')
    print('[INFO] Increase epochs in TRAIN_CONFIG to continue training')
else:
    print('\\n' + '='*70)
    print(f'STARTING TRAINING (epoch {START_EPOCH+1} to {num_epochs})')
    print('='*70)

    for epoch in range(START_EPOCH, num_epochs):
        epoch_loss = 0.0
        epoch_start = time.perf_counter()
        optimizer.zero_grad()

        print(f'\\n{"="*70}')
        print(f'EPOCH {epoch+1}/{num_epochs}')
        print(f'{"="*70}')

        for idx, batch in enumerate(dataloader):
            step_t0 = time.perf_counter()

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[..., 1:].contiguous().view(-1)
                loss = criterion(shift_logits, shift_labels) / cfg['grad_accum']

            scaler.scale(loss).backward()
            batch_loss_val = loss.item() * cfg['grad_accum']
            epoch_loss += batch_loss_val
            history['batch_loss'].append(batch_loss_val)

            if (idx + 1) % cfg['grad_accum'] == 0 or (idx + 1) == total_batches:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                step_time = time.perf_counter() - step_t0
                history['step_loss'].append(batch_loss_val)
                history['step_times'].append(step_time)
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])

                if torch.cuda.is_available():
                    vram = torch.cuda.memory_allocated() / 1e9
                    history['vram_usage'].append(vram)
                else:
                    vram = 0

                elapsed = time.perf_counter() - t_start
                print(f'  Step {global_step:04d}/{total_steps} | '
                      f'batch {idx+1}/{total_batches} | '
                      f'loss={batch_loss_val:.4f} | '
                      f'grad_norm={grad_norm:.3f} | '
                      f'VRAM={vram:.1f}GB | '
                      f'{elapsed:.0f}s')

            elif (idx + 1) % cfg['log_every'] == 0:
                pct = (idx + 1) / total_batches * 100
                print(f'    batch {idx+1}/{total_batches} ({pct:.0f}%) | loss={batch_loss_val:.4f}')

        # Epoch summary
        avg_loss = epoch_loss / total_batches
        epoch_time = time.perf_counter() - epoch_start
        history['epoch_loss'].append(avg_loss)

        print(f'\\n  Epoch {epoch+1} Summary:')
        print(f'    Avg loss: {avg_loss:.4f}')
        print(f'    Time: {epoch_time:.0f}s')
        print(f'    Samples/sec: {len(dataset)/epoch_time:.1f}')

        # Save checkpoint to Google Drive (persistent across restarts)
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch:03d}.pt')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'global_step': global_step,
            'history': history,
            'config': cfg,
        }, ckpt_path)
        print(f'    [SAVED] {ckpt_path}')

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'loss': avg_loss,
                'config': cfg,
            }, best_path)
            print(f'    [BEST] New best loss: {best_loss:.6f} -> {best_path}')

    total_time = time.perf_counter() - t_start
    print(f'\\n{"="*70}')
    print('TRAINING COMPLETE')
    print(f'  Epochs: {START_EPOCH+1} to {num_epochs}')
    print(f'  Total steps: {global_step}')
    print(f'  Total time: {total_time:.0f}s ({total_time/60:.1f}min)')
    print(f'  Best loss: {best_loss:.6f}')
    print(f'  Checkpoints saved in: {CHECKPOINT_DIR}')
    print(f'{"="*70}')
'''

nb['cells'][10]['source'] = [new_cell10]

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Cell 10 rewritten: {len(new_cell10)} chars")
print("OK")
