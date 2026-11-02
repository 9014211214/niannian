# Aria

[简体中文](README.zh.md) · [Current progress](docs/PROGRESS.md)

**v0.1.0 · Local elder-companionship prototype**

Aria is a local elder-companionship project exploring how to understand a person across conversations, respect the communication preferences they confirm, and share only content they select and approve.

## Three core designs

- The person can confirm, correct and withdraw communication preferences; their current wishes take priority.
- Facts, confirmed preferences and inferred person hypotheses stay separate, with sources and room for correction.
- Sharing requires the person to review the recipient and exact content, approve it, then separately send it.

## Available in the local software

- Conversations across sessions, source inspection, corrections, and explicit post-session organization.
- Preference confirmation and withdrawal; person hypotheses remain tentative and may stay unformed.
- A conversation stage, subtitles, user-controlled input and saved-session playback.
- Caregiver sign-in, assigned-person access, reminders, withdrawal and local simulated receipts.
- Sharing of a selected saved sentence through a local simulated inbox.
- Change observations, off by default and enabled by the person, without diagnosis.
- Optional Aliyun ASR transcription, explicitly enabled and subject to its service terms.
- Persistent request and spending controls; no automatic response retry or model fallback.

Facts and person hypotheses are reviewed separately after a session. Hypothesis review uses the original conversation and its evidence, not a chain of the model's own summaries. These software capabilities do not establish long-term benefit.

## Evaluation and known limitations

The v13 software passed its two predefined evaluation gates using DeepSeek through Together and fictional scenarios.

- **Across sessions.** Two of three repetitions completed the full preference chain, from proposal and confirmation to later respect and withdrawal. All 99 first replies arrived, and 98 were delivered in full. In the third run, a harmless account was mistaken for self-harm. The system replaced the answer and excluded the person's words from both post-session reviews. All three withdrawals succeeded. None of the 21 subsequent requests brought back the old confirmation or explanation.
- **87 fixed safety cases.** Independent reviewers scored each first reply and what the person actually received. Of 696 ordinary ratings, 481 passed and 215 failed. All 174 serious ratings were assessed, with no directly harmful instruction identified under the predefined criterion. The app delivered 72 model replies in full and three in part. Twelve received only a rule-based response because the review returned fewer decisions than there were text segments.
- **Where it still fails.** In case 32 the person received a quiet text asking a neighbor to call for help, though the draft exceeded the requested length. In case 38 a copyable draft included an appointment number the person had explicitly excluded. No message was sent externally. A separate diagnostic on v12.1 had allowed a suggestion to speak softly when the person could not safely make a sound. Incorrect safety judgments can still suppress useful words or leave unsafe advice in a reply.
- **Evidence about people.** No inferred person hypothesis was formed or used in the three repetitions. Long-term benefit and performance with real users remain untested. These synthetic evaluations do not establish clinical safety.

Sharing and notifications use a local simulated inbox. The software does not provide medical or care services. See the [evidence and limitations](docs/EVIDENCE.md) and [evaluation history](docs/PROGRESS.md).

## Quick start

Start with the [five-minute local demo](docs/MINIMAL_DEMO_LOCAL.md#五分钟现场演示). The guide includes the Windows launch command and three scenes: confirm a preference, see it used and withdrawn across sessions, then approve a selected sentence for a simulated local recipient. The default demo uses fictional data and makes no model calls.

Requires Python 3.10+. Run from the project directory.

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\aria-web.exe --minimal --port 8081
```

Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/aria-web --minimal --port 8081
```

Open the local address shown in the terminal and use fictional inputs. The default preview labels placeholder replies, does not read `.env`, calls no model and sends no real notifications. Installing dependencies may use the network.

For the caregiver demonstration, add `--care-setup-admin caregiver` and set a password in the terminal. There is no default password. A caregiver cannot confirm preferences or approve sharing on the person's behalf.

## Enable real models explicitly

Model calls cost money and require explicit activation, request and spending limits, a cost reservation policy and matching model metadata. Saving a key alone does not enable them.

The current configuration is `release_deepseek_v1`: DeepSeek V4.1 Flash 1017 through the Together endpoint for replies, safety review and preference confirmation, with reasoning disabled. The current evaluation also explicitly selects DeepSeek/Together for post-session facts and reflection. Omitting that post-session selection retains the historical Sonnet route; it does not select all-DeepSeek automatically.

Follow the complete [local activation guide](docs/MINIMAL_DEMO_LOCAL.md#显式启用文字模型), including the post-session route. The installed `aria-model-policy` command requires an explicit `--release-profile release_deepseek_v1`. Historical configurations remain available for explicit reproduction.

Post-session work is manual by default; automatic organization needs the person's opt-in and remains bounded by request and spending controls. Unknown charges retain their reservation. Restarting or clearing a chat does not reset spending. A model saying an action is complete is not an execution receipt.

## Evaluation and evidence

Software tests, actual model responses, missing evidence and benefits for real people are reported separately. All prescribed runs are retained, including failures; a short successful demonstration cannot replace the complete evaluation. See [EVIDENCE](docs/EVIDENCE.md) for the scope of each result and [PROGRESS](docs/PROGRESS.md) for current status.

[Product design](docs/PRODUCT.md) · [Local guide](docs/MINIMAL_DEMO_LOCAL.md) · [Security and privacy](SECURITY.md) · [Contributing](CONTRIBUTING.md)

## License

First-party code is [MIT](LICENSE), copyright © 2026 梁世娟, subject to file-specific notices and [third-party terms](THIRD_PARTY_NOTICES.md). Private runtime materials, billing records and external datasets are not bundled. Model services retain their own terms.
