"""Run named local software fixtures only; no credential or network options."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.bench.e1b_cases import load_e1b_cases
from aria.bench.e1b_runner import E1bRunner, local_fixture_factory
from aria.engine.perception.external_emotion import SyntheticEmotionExperiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--phase", choices=("main", "prescreen"), default="main")
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args(argv)
    suite = load_e1b_cases()
    selected = args.case_ids or [case.id for case in suite.items]
    if len(set(selected)) != len(selected) or set(selected) - {case.id for case in suite.items}:
        parser.error("case ids must be distinct known draft cases")
    token = SyntheticEmotionExperiment.create(args.output, args.experiment_id)
    recordings = []
    runner = E1bRunner(suite, root=token.root / "run", seed=args.seed,
                       app_factory=local_fixture_factory(token, recordings=recordings))
    replies = 0
    for case_id in selected:
        report = runner.run_case(case_id, phase=args.phase)
        replies += report["scored_reply_count"]
    summary = {"evidence_kind": "software_fixture", "case_count": len(selected),
               "phase": args.phase, "scoreable_fixture_replies": replies,
               "online_fixture_calls": sum(len(clients.online.calls) for _, clients in recordings),
               "memory_fixture_calls": sum(len(clients.memory.calls) for _, clients in recordings),
               "reflection_fixture_calls": sum(len(clients.reflection.calls) for _, clients in recordings),
               "real_model_calls": 0, "new_model_cost_usd": 0,
               "frozen": False, "main_gate": "not_evaluated", "formation_gate": "not_evaluated",
               "source_sha256": suite.source_sha256, "seed": args.seed}
    path = token.root / "rehearsal-summary.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
