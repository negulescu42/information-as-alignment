"""One-command verification gate: runs the fast layers of the test pyramid and
exits non-zero on the first failure.

Layers covered (~12-15 min total):
  0  the 24 behavioural guarantee checks            (mscn.tests)
  1  the 9 AGI-upgrade validations + sigma*         (mscn.agi_all)
     statistical state-merging (8.11)               (mscn.agi_alergia)
     the paper engine lifecycle decomposition (10)  (mscn.ibf_engine)
     operating-bandwidth manuscript checks          (mscn.opband_check)
     the IBF-ASI suite, reduced seeds               (mscn.ibf_asi --quick)
     invariant fuzzing, reduced                     (mscn.ibf_asi_fuzz --n 60)

The heavy layers (regime matrix, gauntlets, KRK runs, benchmark) are run
separately; see ARCHITECTURE.md sections 9-10 for their entry points.

Run: ``python -m mscn.testall``
"""

from __future__ import annotations

import subprocess
import sys
import time

GATE = [
    ("guarantee checks", [sys.executable, "-m", "mscn.tests"]),
    ("AGI upgrades", [sys.executable, "-m", "mscn.agi_all"]),
    ("statistical merging", [sys.executable, "-m", "mscn.agi_alergia"]),
    ("paper engine", [sys.executable, "-m", "mscn.ibf_engine"]),
    ("operating-bandwidth checks", [sys.executable, "-m", "mscn.opband_check"]),
    ("IBF-ASI suite (quick)", [sys.executable, "-m", "mscn.ibf_asi", "--quick"]),
    ("invariant fuzz (60)", [sys.executable, "-m", "mscn.ibf_asi_fuzz", "--n", "60"]),
]


def main() -> int:
    t0 = time.time()
    for name, cmd in GATE:
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t
        status = "PASS" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] {name:<28} {dt:6.1f}s")
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            print(f"\nGATE FAILED at: {name}")
            return 1
    print(f"\nGATE GREEN in {time.time() - t0:.0f}s "
          f"({len(GATE)} suites, every assert passed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
