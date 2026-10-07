# Aria

[简体中文](README.zh.md) · [Current progress](docs/PROGRESS.md)

Aria is a local elder-companionship prototype: understand how someone wants to be
approached, keep the evidence behind that understanding, and respect their choices.
It combines continuous conversation, revisable person hypotheses, and a small care
workflow. Personality frameworks guide communication; they are not diagnoses.

Model effectiveness has **not passed acceptance**. In the completed E1b-R run,
all 14 proposed communication preferences were rejected before confirmation, so
the intended preference-use effect was not measured. See the
[fixed result and diagnosis](docs/E1BR_RESULTS_2026_10_05.md). Later repairs need
their own model evidence; software tests do not establish a benefit to elders.

The first E1c run dispatched 124 of 1,056 planned positions and completed one of
12 cases before a response violated the output schema. It remains **not evaluable**;
the short-plan prompt repair has software checks, not proven model reliability.
See the [evidence boundaries](docs/EVIDENCE.md) for what these results can support.

The [design audit repairs](docs/DESIGN_REPAIR_2026_10_06.md) cover preference admission,
cognitive privacy, same-call safety checks, call recovery and pre-send reservations.
The subsequent E1c-v2 run completed generation and first model scoring, and failed
its effect gates. The later 24-case check also failed its task and reply gates.
Three real-model story repeats are complete, but supported person hypotheses were
not injected, so they do not establish a personality-understanding benefit. See the
[versioned validation results](docs/VALIDATION_RESULTS_2026_10_07.md) for failures,
exposure, subsequent repairs, and the separate repair check.

## Run locally

Requires Python 3.10 or newer. From this directory, on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\aria-web.exe --minimal --port 8081
```

On Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/aria-web --minimal --port 8081
```

Open the loopback address printed in the terminal. The default preview uses clearly
labelled placeholder replies, reads no `.env`, and makes no model or notification
calls. Use fictional inputs. Installing dependencies may require internet access.

For a caregiver account, run the same command with `--care-setup-admin caregiver`.
Set its password in the private terminal prompts; there is no default password.

Real text-model use additionally requires `--enable-model-calls`,
`--model-max-requests`, `--model-stop-cost-usd`, and
`--model-reservation-policy`. Prepare the price/endpoint policy offline from saved
public metadata with `scripts/prepare_model_policy.py`; the [local guide](docs/MINIMAL_DEMO_LOCAL.md#显式启用文字模型)
lists its three inputs and commands. Neither a saved key nor a policy file enables
calls by itself. The guard reserves estimated costs before sending and retains
unknown results; this is not a provider billing guarantee.

## First-version scope

- Several conversations with the same fictional person, visible fact/person
  distinctions, sources, corrections, and explicitly started post-session processing.
- Owner-confirmed communication preferences, kept separate from inferred hypotheses,
  with withdrawal and a saved account of the inputs used for each reply.
- A lightweight stage, verbatim captions, editable input, and saved playback.
- Caregiver login and elder scope, dated one-off/daily reminders, responses,
  revocable consent, a simulated notification ledger, and follow-up tasks.
- Optional, explicitly enabled text-model calls with persistent request/cost stops.
- Optional per-turn external emotion readings, labelled as fallible hints. No
  reading means no extra emotion prompt; unknown never means calm.
- Memory and daily-change observations, off by default and enabled by the fictional
  owner. Original quotes can be inspected/deleted; this does not diagnose illness.

The minimal live client uses `v3` with `mode_v1`: the same online call includes a
safety signal, including on simple turns. Earlier `v1/v2` experiments retain their
original contracts; `profile_v1` remains experimental. Keyword safety checks stay
in place. A model signal grants no notification or medical authority, and the new
semantic path has been checked with real models on frozen versions, with failed
effect gates. Later layout and receipt-state repairs have software checks only.
The proposed consented
message experience and real-model short film have not been completed. See the
progress page for version-specific implementation and acceptance results.

The first version uses a simulated notification channel. Camera, acoustic emotion,
ASR/TTS, Omni, full-duplex voice, hardware, other product scenes, and the old large
research platform are outside this version. This is not a production care service.
The external emotion interface does not start a sensor or emotion model. Synthetic
readings require a separate Python experiment namespace and are visibly labelled.
The local demo token lets its holder act as the fictional owner; this is not identity
verification for a real elder. Caregiver permissions do not grant those owner controls.

## Documentation

- [Product, architecture, and boundaries](docs/PRODUCT.md)
- [Local use, care, data, and model activation](docs/MINIMAL_DEMO_LOCAL.md)
- [What the evidence can support](docs/EVIDENCE.md)
- [Focused validation design](docs/VALIDATION.md)
- [Build and release checks](docs/MINIMAL_RELEASE_CHECKLIST.md)
- [Contributing](CONTRIBUTING.md) · [Security and privacy](SECURITY.md)

Software tests, simulated workflows, real-model quality, and release acceptance are
separate claims. The only current status and next-work list is
[PROGRESS](docs/PROGRESS.md). Historical research remains preserved in a private
archive; it is not a prerequisite for running this product.

E1 generated 204 responses to authored fictional scenarios. Uncalibrated Codex
judgments failed the core improvement gate, and measured experimental overhead
failed the latency gates. These results do not establish benefits for real elders
or personality learning. Optional human review
could calibrate these judgments; running or developing the prototype does not
require recruiting elders. CPED preparation stopped at a local feasibility gate;
no real E2 was run. Later product changes do not inherit E1's model conclusions.

## Licence

First-party code is [MIT](LICENSE), subject to file-specific notices. See
[third-party notices](THIRD_PARTY_NOTICES.md). External datasets and personal data
are not included. Model services and external datasets retain their own terms.
