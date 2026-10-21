"""Write tests/slow_tests.txt: the tests a plain ``pytest`` skips (``AIME_TESTS=all`` runs
them, as CI does). Feed it the output of a full run with durations:

    AIME_TESTS=all pytest --durations=0 --durations-min=1 > durations.log
    python scripts/find_slow_tests.py durations.log [--threshold 10]

A test is slow when its call, or its setup (a fixture that trains a model), takes at least
the threshold in seconds.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"^\s*([0-9.]+)s (call|setup|teardown)\s+(\S+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()
    slow: dict[str, float] = {}
    for line in args.log.read_text().splitlines():
        m = LINE.match(line)
        if m and m.group(2) != "teardown" and float(m.group(1)) >= args.threshold:
            slow[m.group(3)] = max(slow.get(m.group(3), 0.0), float(m.group(1)))
    out = ROOT / "tests" / "slow_tests.txt"
    out.write_text(
        "# Tests a plain `pytest` skips: AIME_TESTS=all (or --run-slow) runs them, as CI\n"
        f"# does. Written by scripts/find_slow_tests.py (>= {args.threshold:g} s).\n"
        + "".join(f"{nodeid}\n" for nodeid in sorted(slow)))
    print(f"wrote {out.relative_to(ROOT)}: {len(slow)} slow tests "
          f"({sum(slow.values()) / 60:.0f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
