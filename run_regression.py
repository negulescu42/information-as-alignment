"""
run_regression.py
=================

Phase 0 regression test: confirm the refactored ibf_v1_engine.py still
reproduces the original 2D toy model's qualitative behavior:

    - Phase A learns above chance
    - crystallized centers appear
    - Phase B learns
    - the Crucible verifies and/or dissolves some centers

Usage:
    python run_regression.py [seed]
"""

import sys
from ibf_v1_engine import run_toy_experiment


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    metrics, log, ibf, env, enc, be, snaps = run_toy_experiment(
        seed, tag="regression", quiet=True)

    checks = {
        "Phase A learns above chance (Acc_A end A > 0.6)":
            metrics['acc_A_end_A'] > 0.6,
        "Crystallized centers appear (> 0)":
            metrics['n_cryst'] > 0,
        "Phase B learns (Acc_B end B > 0.6)":
            metrics['acc_B_end_B'] > 0.6,
        "Crucible active (verified or dissolved > 0)":
            (metrics['n_verified'] + metrics['n_dissolved']) > 0,
    }

    print("=" * 60)
    print("  IBF v1 ENGINE REGRESSION (seed=%d)" % seed)
    print("=" * 60)
    print("  Acc_A (end A): %.3f" % metrics['acc_A_end_A'])
    print("  Acc_A (end B): %.3f  (BT_A=%+.3f)" % (metrics['acc_A_end_B'], metrics['BT_A']))
    print("  Acc_B (end B): %.3f" % metrics['acc_B_end_B'])
    print("  centers=%d  crystallized=%d  verified=%d  dissolved=%d"
          % (metrics['n_centers'], metrics['n_cryst'],
             metrics['n_verified'], metrics['n_dissolved']))
    print("-" * 60)
    all_ok = True
    for name, ok in checks.items():
        print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
        all_ok = all_ok and ok
    print("-" * 60)
    print("  REGRESSION: %s" % ("PASS" if all_ok else "FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
