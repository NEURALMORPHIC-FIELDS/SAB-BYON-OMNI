#!/usr/bin/env python
"""Training script for SAB + BYON-OMNI 3B model."""
import argparse
import sys
sys.path.insert(0, '.')

from sab_byon_omni.core import SABTranscendentV2
from sab_byon_omni.training.train_3b import train_3b_rapid


def main():
    parser = argparse.ArgumentParser(description='Train SAB + BYON-OMNI 3B Model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--log-every', type=int, default=5)
    args = parser.parse_args()

    print("Initializing SAB Transcendent v2.0...")
    sab = SABTranscendentV2()

    print("Starting training...")
    train_3b_rapid(
        sab,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        log_every_batches=args.log_every,
    )

    print("Training complete!")


if __name__ == '__main__':
    main()
