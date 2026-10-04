"""Run the complete C-test with a local memory fixture; never reads a key."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.bench.ctest_runner import CTestRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args(argv)
    runner = CTestRunner(args.output, experiment_id=args.experiment_id)
    summary = runner.run()
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
