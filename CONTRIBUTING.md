# Contributing to Aria

Start with [PROGRESS](docs/PROGRESS.md) and [the product scope](docs/PRODUCT.md).
Aria has one product: elder companionship with a bounded care workflow. Keep changes
small enough to explain through that experience.

## Local checks

```powershell
python -m pip install -e ".[dev,secure]" build twine
python -m ruff check aria examples scripts tests
python -m pytest -q
python -m pip check
python scripts/build_minimal_release.py --check
python -m build
```

Dependency installation may use the network; tests must use local fixtures and
must not call paid models or send real notifications. Validate the built wheel and
source distribution outside the source directory, as described in the
[release guide](docs/MINIMAL_RELEASE_CHECKLIST.md). Report the environment and what
was actually checked; reused local dependencies are not a clean online install.

## Change boundaries

- Preserve unrelated working-tree changes and historical research originals.
- Keep facts, confirmed preferences, inferred hypotheses, and current emotion distinct.
  Hypotheses need sources, scope, confidence, counter-evidence, and correction history.
- Do not lower person-evidence thresholds to make a short story appear to learn.
- Keep emergency handling and permission rules independent of personality preferences.
- Do not infer permission to spend, activate services, contact people, or publish from
  an API key or an item in a plan.
- Do not commit credentials, runtime profiles, conversations, recordings, external
  datasets, or per-person evaluation outputs.
- Saved playback must not call models, create reminders, or send notifications.
- Report model output and local safety rewrites separately. Tests and synthetic
  demonstrations are not evidence of real companionship benefit.

Describe the problem, resulting behavior, relevant verification, and limitations in
changes. Run checks appropriate to the change rather than adding duplicate tests.
Maintain current status only in `docs/PROGRESS.md`; the coordinating agent owns that
file during parallel work. Do not create competing dated progress reports.

Report vulnerabilities privately as described in [SECURITY](SECURITY.md).
