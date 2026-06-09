"""Run all nine AGI-roadmap upgrade validations (and the multi-scale sigma* base).

Each module's ``main()`` prints its result and asserts its measured guarantee, so a
clean exit means every mechanism is functional with its honest, measured advantage.
See ``MSCN_AGI_ROADMAP.md`` and ``ARCHITECTURE.md`` §8 for the logged results.

Run: ``python -m mscn.agi_all``
"""

from __future__ import annotations

import importlib
import time

# (module, upgrade label) in roadmap priority order
UPGRADES = [
    ("mscn.agi_simulation", "U1  Learnable Simulation Homomorphism"),
    ("mscn.agi_exploration", "U3  Coherence-Gradient Exploration"),
    ("mscn.agi_planning", "U7  Coherence-Based Planning"),
    ("mscn.agi_hierarchical_sim", "U2  Hierarchical Simulation"),
    ("mscn.agi_directed", "U4  Directed Exploration (info gain)"),
    ("mscn.agi_transfer", "U5  Cross-Domain Coherence Morphisms"),
    ("mscn.agi_temporal", "U8  Temporal Abstraction (macro-actions)"),
    ("mscn.agi_compositional", "U6  Compositional Coherence Algebras"),
    ("mscn.agi_selfimprove", "U9  Active Self-Improvement (reflexive)"),
    ("mscn.hierarchical_sigma", "R2.1/3.1  Hierarchical operating resolution"),
]


def main() -> None:
    results = []
    for mod_name, label in UPGRADES:
        print("\n" + "=" * 74)
        print(f"  RUNNING  {label}   ({mod_name})")
        print("=" * 74)
        t0 = time.time()
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
            results.append((label, "PASS", time.time() - t0))
        except AssertionError as e:               # a measured guarantee failed
            results.append((label, f"FAIL: {e}", time.time() - t0))
        except Exception as e:                     # pragma: no cover
            results.append((label, f"ERROR: {type(e).__name__}: {e}", time.time() - t0))

    print("\n" + "#" * 74)
    print("#  AGI-ROADMAP UPGRADE SUMMARY")
    print("#" * 74)
    n_pass = sum(s == "PASS" for _, s, _ in results)
    for label, status, dt in results:
        mark = "PASS" if status == "PASS" else status
        print(f"  [{'OK ' if status == 'PASS' else '!! '}] {label:<46} {mark:<8} ({dt:.1f}s)")
    print(f"\n  {n_pass}/{len(results)} upgrade validations pass.")
    assert n_pass == len(results), "some upgrade validations failed"
    print("  All AGI-roadmap mechanisms are functional with their measured advantages.\n")


if __name__ == "__main__":
    main()
