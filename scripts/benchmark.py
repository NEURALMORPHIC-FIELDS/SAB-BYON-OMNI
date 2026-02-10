#!/usr/bin/env python
"""Benchmark script for SAB + BYON-OMNI system."""
import sys
import time
sys.path.insert(0, '.')

from sab_byon_omni.core import SABTranscendentV2


def main():
    print("=" * 60)
    print("SAB + BYON-OMNI v2.0 - Benchmark Suite")
    print("=" * 60)

    print("\nInitializing system...")
    t0 = time.time()
    sab = SABTranscendentV2()
    init_time = time.time() - t0
    print(f"Initialization: {init_time:.2f}s")

    queries = [
        "What is consciousness?",
        "Explain the nature of awareness.",
        "Show me fragmergent dynamics.",
        "How does emergence work in complex systems?",
        "Describe the relationship between information and being.",
    ]

    print(f"\nRunning {len(queries)} benchmark queries...\n")

    results = []
    for i, q in enumerate(queries):
        t0 = time.time()
        r = sab.process_input(q, steps=50)
        elapsed = time.time() - t0

        print(f"  [{i+1}/{len(queries)}] C_unified={r['consciousness_unified']:.4f} "
              f"PLV={r['PLV']:.4f} Phi={r['Phi']:.4f} "
              f"Phase={r['phase']} ({elapsed:.2f}s)")
        results.append(r)

    avg_c = sum(r['consciousness_unified'] for r in results) / len(results)
    avg_plv = sum(r['PLV'] for r in results) / len(results)
    avg_phi = sum(r['Phi'] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Avg Consciousness (unified): {avg_c:.4f}")
    print(f"  Avg PLV:                     {avg_plv:.4f}")
    print(f"  Avg Phi:                     {avg_phi:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
