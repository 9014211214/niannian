# Security and privacy

Report credentials, private conversations, or deployment vulnerabilities privately to
the repository owner. Do not include them in public issues. If the repository later
supports private security advisories, use that channel. Current work is tracked only
in [PROGRESS](docs/PROGRESS.md).

## Local service and identity

The first-version Web entry is a loopback-only, single-instance prototype for
fictional data. Its page token, Host checks, and Origin checks reduce cross-site
requests; the page token does not authenticate an elder or caregiver.

Caregiver setup uses private password prompts, explicit elder assignments, salted
PBKDF2 password hashes, hashed session tokens, bounded failed logins, and HttpOnly,
SameSite=Strict cookies. The care API checks the authenticated caregiver's elder
scope and rejects caregiver confirmation, refusal, or postponement of an elder's
reminder response. Assigned caregivers can manage reminders and follow-up without
recording those actions as the elder's response. These controls do not provide
institutional identity, TLS, or production operations. The companion page remains
bound to one fictional elder.

## Stored data

The default root is `data/minimal_demo`, with separate `local_preview` and `live`
subdirectories. Profile and replay records contain fictional conversation text.
The original-conversation queue is bounded to eight jobs and 120 turns / 60,000
characters per session, with a 24-hour retention policy. Replay records remain until
explicit deletion; queue expiry does not promise deletion of every transcript.

Active recall of a saved episode or relationship fact requires its complete,
verified owner-source context. Older records with incomplete context remain stored
and readable, but cannot authorize active reference. Reference permission fields
accept only booleans; an invalid legacy field fails profile loading and preserves
the original file for repair rather than silently deleting it.

With the `secure` extra and `ARIA_PROFILE_KEY`, the profile envelope and minimal
replay file can be encrypted. Without them, these files are plaintext.
**`care.sqlite3` and `model_calls.sqlite3` remain plaintext.** Care records include
account hashes, assignments, reminders, consent, contacts, receipts, and follow-up.
Protect filesystem access and keep the encryption key separately.

Care-related chat may attach a bounded read-only snapshot of this elder's recent
records. It excludes contact names/addresses, notification text, follow-up notes,
and login data. Labels, opaque record IDs, times, and statuses remain. In explicitly
enabled model mode this snapshot goes to the model with that turn. Keyword selection
is limited; a missing snapshot does not mean no records exist. Saved replay uses the
original snapshot. It supplies neither personality evidence nor action permission.

Optional external emotion readings are per-turn, bounded hints whose source and
modality are supplied by the caller. Aria does not verify a sensor observation or
start an emotion service. Explicit model mode sends the accepted hint with the
current turn; it is not a person's own words or personality evidence. Synthetic
emotion experiments use a marked, separate Python-only namespace.

Daily-change observation is off by default and enabled only through the fictional
owner surface. Caregiver access cannot enable it. Its sensitive original quotes
use a separate bounded post-session queue for memory processing; reflection does
not receive that channel. Disabling observation stops collection and clears pending
quotes; saved observations can be retained or deleted. Deleting an observation
does not erase the original chat or provider-held information. Enabling observation
does not authorize sharing it; the first-version notification scope excludes these
observations. See the [daily-change design](docs/COGNITIVE_CHANGE.md).

“Clear this demo” deletes the current fictional profile, conversations, queue, and
this elder's care records. It retains caregiver accounts, assignments, and the live
model-call ledger. It does not erase other modes, backups, or provider-held data, and
is not secure disk erasure.

## Model calls and failures

Default preview reads no `.env` and activates no model or media service. Explicit
`--enable-model-calls` requires request and observed-cost limits before credentials
are loaded. Online, memory, and reflection calls share a persistent ledger: intent
is saved before sending. Product output failures retain the failed result and use
a labelled fallback. The current release isolates a fully settled format failure
only after verifying that no business state was partially committed; it preserves
that failure and permits a separately submitted new input. Eligible unreceived
online failures require the owner's explicit choice to retain their reservation
before a new input. The historical ProductCallGuard's three-failure pause/resume
is a separate option, not the default release rule. No continuation resends an
old request or restores budget.
Unknown cost/result, failed persistence, or a budget, routing or privacy violation
remains a hard stop that the recovery button cannot clear. The current release
has zero automatic paid retries or model fallbacks. Historical retry protocols
remain explicit archival options, not current defaults. Physical requests reject HTTP redirects rather
than send credentials or create an unrecorded request at another address. The
[local guide](docs/MINIMAL_DEMO_LOCAL.md#显式启用文字模型) describes the complete conditions.
Refresh, restart, and clearing chat do not reset that ledger.
Existing live conversations without a ledger cannot silently start accounting at zero.

The ledger stores bounded IDs, role/configuration bindings, request digests, and
usage metadata, not prompts or hidden model reasoning. Its limit is per directory,
not account-wide. A request already sent may exceed the observed-cost threshold;
this is a stop-before-next-call mechanism, not a supplier invoice cap. Do not evade
an agreed budget by switching directories or deleting records. Review provider
retention and processing terms before sending real information.

## Care, consent, and replay

The first version supplies a local simulated notification channel and simulated
consent. A simulated receipt is not real delivery or family acknowledgement.
Reminder sharing requires a trigger, scope, expiry and revocation. A separate
owner-controlled flow can preview and approve one complete saved utterance for
one chosen contact, then explicitly send through the local simulated channel.
Caregivers cannot approve that flow for the owner; reminder consent does not
authorize it. Inferred psychological hypotheses and change observations are excluded.
An unconfirmed reminder is not proof of missed medication or danger. Revocation
blocks pending sends but cannot retract delivered information. Unknown outcomes
require review rather than automatic resending.

Chat does not create care actions: the structured form records an explicit choice.
Personality preferences cannot change medical rules, permissions, or escalation
thresholds. Crisis and acute-warning handling remains a deterministic safeguard,
not a validated diagnosis or an emergency response service.

Closing/idle seals a conversation. Post-session model work is manual by default;
explicit owner opt-in may process newly sealed sessions within daily and shared
call/cost limits. A foreground input may wait for an already sent physical request
to settle, then takes priority before the next background call; it does not wait
for the entire review. Saved playback performs no model or care writes. Stage animation is never
emotion evidence. Enabled care scheduling continues while the service runs,
including during playback; stopping the service pauses scheduling.

## Source releases

The [archive tool](scripts/build_minimal_release.py) scans selected current files
and records hashes. It does not review Git history, runtime data, all personal
information, upstream rights, or every possible secret. A passing scan is neither
release approval nor a guarantee of safety. Follow the
[release guide](docs/MINIMAL_RELEASE_CHECKLIST.md).
