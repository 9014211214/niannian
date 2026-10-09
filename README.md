# Aria

[简体中文](README.zh.md) · [Current progress](docs/PROGRESS.md)

Aria is a local elder-companionship prototype: understand how someone wants to be
approached, keep the evidence behind that understanding, and respect their choices.
It combines conversation, revisable person hypotheses, and a small local care
workflow. **Model effectiveness has not passed acceptance.** Known model errors
include false safety alarms and incomplete task replies. This is not a production
care service; notifications are simulated. See the [evidence boundaries](docs/EVIDENCE.md).

Three design choices guide the prototype:

- Keep facts, owner-confirmed communication preferences, and inferred hypotheses separate.
- Preserve sources, scope, corrections and counterevidence; personality frameworks
  guide communication and are not diagnoses or permanent labels.
- Require actual capability, permission and receipts for actions. A conversation,
  preference or model risk label cannot grant care authority.

## What you can try

Use a fictional person and explore the local workflow. These features are
implemented; the table does not claim real-world effectiveness.

| Experience | Available locally | What remains unproven or limited |
| --- | --- | --- |
| Conversation | Captions, editable input, saved playback, optional real text-model replies | Task and reply quality have failed checks on frozen versions. Default replies are placeholders. |
| Facts and communication preferences | Sources, corrections, owner confirmation and withdrawal; explicitly started post-session processing | Preference use has been observed after repairs; benefit and consistent quality remain unproven. |
| Person hypotheses | Revisable candidates, evidence and scope, with gates before use | Earlier E1c-v2 and three-session checks did not observe their actual injection or benefit. Long-term formation and benefit are unverified. |
| Safety | Keyword fallback plus a safety signal in the same online model call | False alarms have replaced ordinary replies; broader risk recognition has not passed dedicated acceptance. |
| Care | Caregiver scope, reminders, revocable consent, receipt states and follow-up | The notification channel is local simulation; no real recipient delivery or care outcome is established. |
| Daily-change observations | Off by default; owner can enable, inspect, delete or disable quoted records | No diagnosis; dedicated real-model effectiveness checks are incomplete. |

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
labelled placeholder replies, reads no `.env`, makes no model calls, and sends no
real notifications. Local simulated care records can still advance while the
service runs. Use fictional inputs. Installing dependencies may require internet
access.

For a caregiver account, run the same command with `--care-setup-admin caregiver`.
Set its password in the private terminal prompts; there is no default password.

Real text-model use additionally requires `--enable-model-calls`,
`--model-max-requests`, `--model-stop-cost-usd`, and
`--model-reservation-policy`. Prepare the price/endpoint policy offline from saved
public metadata with `scripts/prepare_model_policy.py`; the [local guide](docs/MINIMAL_DEMO_LOCAL.md#显式启用文字模型)
lists its three inputs and commands. Neither a saved key nor a policy file enables
calls by itself. The guard reserves estimated costs before sending and retains
unknown results; this is not a provider billing guarantee.

## Model mode and boundaries

The minimal live client uses `v3` with `mode_v1`: the same online call includes a
safety signal, including on simple turns. Earlier `v1/v2` experiments retain their
original contracts; `profile_v1` remains experimental. Keyword safety checks stay
in place. A model signal grants no notification or medical authority. Software
checks, real-model checks and acceptance belong to their tested versions; later
repairs and isolated candidates do not inherit earlier model conclusions.
Post-session processing defaults to manual. The owner can explicitly enable automatic processing for newly sealed sessions, with a daily session limit and the same cost guard. Model-reviewed preference proposals can be restated on the owner page and answered in the next conversation turn; code checks the delivered question and saves before confirming. An eligible owner can also choose to discuss change records for one next request without a fabricated emotion reading. These are local engineering paths; model accuracy and timing remain unverified. The owner can separately select one complete saved utterance from the active session, review its recipient and local channel, approve it, and then send it to a local simulated inbox. The preview expires after five minutes; refresh does not resend, and uncertain attempts stop further sharing. Real external delivery and the real-model short film remain incomplete.

The first version uses a simulated notification channel. Camera, acoustic emotion,
ASR/TTS, Omni, full-duplex voice, hardware, other product scenes, and the old large
research platform are outside this version. This is not a production care service.
The external emotion interface does not start a sensor or emotion model. Synthetic
readings require a separate Python experiment namespace and are visibly labelled.
Optional readings are fallible hints; missing readings add no emotion prompt,
and unknown never means calm. Each reply keeps a record of its actual input basis.
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

Versioned negative results remain available:

- [E1b-R](docs/E1BR_RESULTS_2026_10_05.md): all 14 communication-preference
  candidates were rejected before confirmation; the intended use effect was not measured.
- [E1c-v2](docs/E1CV2_RESULTS_2026_10_07.md) and the
  [subsequent validation checks](docs/VALIDATION_RESULTS_2026_10_07.md): effect gates
  failed or were not evaluable. Earlier three-session checks recorded no person-hypothesis
  injection; this does not establish a fixed number of conversations needed for formation.
- [The 24-case two-arm check](docs/VALIDATION_RESULTS_2026_10_07.md#8-2026-10-08-另版双臂检查):
  30 replies received, one rate-limited request, 17 positions unissued. All received
  replies met the output schema; low exposure left both effect gates not evaluable,
  and separate protection and fixed-denominator format gates failed. The 1.8 prompt
  and reflection-format candidates remain isolated from the product.
- [Evidence and earlier research](docs/EVIDENCE.md): incomplete runs and the failed
  original E1 improvement/latency gates are preserved. CPED stopped at local preparation;
  no real E2 was run. Optional human review is not required to run or develop the prototype.

Current repairs and new isolated candidates are tracked only in the progress page.
Software tests and simulated workflows do not establish benefits for real elders.

## Licence

First-party code is [MIT](LICENSE), subject to file-specific notices. See
[third-party notices](THIRD_PARTY_NOTICES.md). External datasets and personal data
are not included. Model services and external datasets retain their own terms.
