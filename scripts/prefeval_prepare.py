"""E3 public-source preparation and optional zero-network software rehearsal."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aria.bench.prefeval_cases import call_inventory, distraction_plan, load_suite, qualification_audit
from aria.bench.prefeval_judge import export_materials
from aria.bench.prefeval_runner import PrefEvalClients, PrefEvalFixtureClient, PrefEvalRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rehearse", action="store_true", help="Run local fixtures, not models")
    args = parser.parse_args()
    local = Path(__file__).resolve().parents[1] / "artifacts" / "local"
    output = args.output.resolve()
    if local.resolve() not in output.parents:
        parser.error("source-derived materials must stay inside ignored artifacts/local")
    suite = load_suite(args.source)
    output.mkdir(parents=True, exist_ok=False)
    data = {"selection": suite.frozen_selection(), "qualification": qualification_audit(suite),
            "call_inventory": call_inventory(suite), "distraction_question_plan": distraction_plan(suite),
            "sample": [asdict(case) for case in suite.cases], "effect_status": "not_run"}
    (output / "preparation.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.rehearse:
        clients = PrefEvalClients(PrefEvalFixtureClient("online"), PrefEvalFixtureClient("memory"))
        generated = PrefEvalRunner(suite, root=output / "software-rehearsal", clients=clients).run()
        materials = export_materials(suite, generated)
        (output / "software-judge-materials.json").write_text(json.dumps(materials, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_topics": 20, "source_rows": len(suite.pool), "selected": len(suite.cases),
                      "structurally_excluded": data["qualification"]["structurally_excluded"],
                      "real_model_effect": "not_run", "software_rehearsal": args.rehearse}))


if __name__ == "__main__":
    main()
