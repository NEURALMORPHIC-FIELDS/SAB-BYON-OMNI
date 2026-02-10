#!/usr/bin/env python
"""Interactive demo for SAB + BYON-OMNI system."""
import sys
sys.path.insert(0, '.')

from sab_byon_omni.core import SABTranscendentV2


def main():
    print("SAB + BYON-OMNI v2.0 - Interactive Demo")
    print("Type 'quit' to exit.\n")

    sab = SABTranscendentV2()

    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ('quit', 'exit', 'q'):
            break
        if not text:
            continue

        result = sab.process_input(text, steps=30)
        print(f"\nSAB: {result['response']}")
        print(f"  [C={result['consciousness_unified']:.3f} "
              f"PLV={result['PLV']:.3f} "
              f"Phase={result['phase']}]")


if __name__ == '__main__':
    main()
