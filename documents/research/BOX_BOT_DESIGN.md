# box-bot: Autonomous Multi-Platform Strategy Commentary Bot (design)

**Status: research design, forward plan. Design only, no code, no commitments.**
**Date: 2026-07-07.**

This document designs ecosystem initiative 2, codenamed `box-bot` (product name
"F1StratLab Live"): an autonomous bot that publishes F1 strategy insight to **multiple
platforms** (X/Twitter and Discord at launch; Bluesky, Mastodon, or Threads later as
plug-in adapters) while real sessions run. It consumes the core's live stream (the
real-time OpenF1 consumer fanned out over the backend SSE/WS relay), phrases posts
with the `gridmind` LoRA (or an OpenAI mini model until the LoRA exists), and posts
fully automatically, no human in the loop per post, under a hard, product-defining
guardrail: **it never invents a number**. Fase 5 of the ecosystem roadmap, deliberately
last: it depends on the live consumer, benefits from gridmind, and must never run on
drifting models (Fase 4 before Fase 5).

Hard constraints honored throughout: design only, no code; box-bot lives in its own
independent public repo and the core gains zero references to it; the LLM provider is
OpenAI or LM Studio (serving the gridmind LoRA), never Anthropic;
`scripts/run_simulation_cli.py`, `src/agents/` internals, and `notebooks/**` are
untouchable (box-bot touches nothing in the core anyway).

Documents this builds on (read, not re-planned):
`documents/research/REALTIME_OPENF1_CONSUMER_DESIGN.md` (the live producer and its
fan-out), `documents/research/GRIDMIND_LORA_DESIGN.md` (sections 6.2 and 6.5, the
numeric grounding checker and the serving-time verifier this design reuses),
`documents/research/ECOSYSTEM_REPO_INTEGRATION.md` (the downstream-service mechanism
and dependency invariant), `documents/research/PITWALL_REALISM_AND_TELEMETRY_SURFACE.md`
(the O1 relay, observability tier labels, and the ECharts layer the chart media reuses).

---

## 1. Framing and invariants

Four invariants define the whole design; everything else is detail:

1. **Downstream service, one-way dependency.** box-bot is an independent public repo
   that subscribes to the core's published live stream and pins a core release on ITS
   side. It is never a submodule; the core never imports, tests, or references it. A
   bot outage can never block a core release, and a core refactor breaks box-bot's
   contract test, not the core's CI (ecosystem integration doc, sections 1, 2, 5).
2. **One pipeline, many publishers.** Content generation and the numeric guardrail are
   platform-independent and run exactly once per event; platforms are thin `Publisher`
   adapters behind one interface. Adding a platform is adding an adapter, never
   touching the pipeline (section 3).
3. **Never invent a number.** Every numeric token in a published post, on any platform,
   must be traceable verbatim (or by depth-1 arithmetic) to the live stream payload the
   post was generated from. The LLM phrases; it does not recall. Enforcement is
   mechanical (section 5): the gridmind-published numeric grounding checker is the
   blocking gate, and per-platform rendering is deterministic so verification done once
   covers every platform.
4. **Silence over speculation.** No stream, no posts. Stale data, no posts. Models in
   the 2026 drift window, no recommendation posts. The bot's failure mode is always
   quiet, never guessy, on every platform.

What box-bot inherits from the live reality: the stream is timing-tier data for all
cars (the live consumer's section 3). The bot therefore only ever talks about lap and
sector times, gaps, positions, compounds, tyre ages, pit events, track status, and the
core's own model outputs. It never claims rival fuel states, tyre temperatures, or
anything a public feed cannot see, because those numbers never enter its input.

---

## 2. What box-bot consumes (the upstream contract)

- **The stream.** The backend's live-session mode publishes the same event vocabulary
  the simulate endpoint already froze: `start`, `lap`, `error`, `summary` (live
  consumer, section 5). Each `lap` event carries the lap-cadence payload derived from
  `lap_state` plus the orchestrator's `StrategyRecommendation` (the frozen
  `LapDecision`/`RunSummary` shapes; changes upstream are additive by rule). box-bot
  prefers the SSE form (it is a one-directional consumer and never needs to send); the
  client treats transport as a detail and the event schema as the contract.
- **Session identity comes from the stream.** `session_meta` names the GP, session
  type, and "our" driver. box-bot carries no race configuration of its own; it narrates
  whatever session the core is running.
- **Staleness metadata is honored.** The `session_meta.live` block (feed latency,
  per-rival staleness) gates posting: a card is never built from a payload flagged
  stale, and estimate-tier fields (for example `fuel_load`, a model estimate by
  construction) are either omitted or labeled "est." in rendered text, matching the
  pit-wall doc's observed/derived/estimate labeling discipline.
- **Pinning.** box-bot pins a core release tag plus a stream schema version, and its CI
  runs a contract test against the exported schema (the ecosystem doc's Q5
  recommendation: export the existing Pydantic models; box-bot consumes that export).
  Opting into a new core release is an explicit, reviewable bump in box-bot's repo.
- **No direct OpenF1 access.** box-bot reads only the core's stream. One source of
  truth; if the core is not live, box-bot has nothing to say by design.

---

## 3. Architecture: one pipeline, many publishers

The pipeline is a straight line with a fan-out at the very end:

```
core live stream (SSE/WS)
      |
      v
trigger engine (section 4.2)        one instance, platform-independent
      |
      v
event card builder (3.1)            numeric slots filled from the frozen payload
      |
      v
LLM color (gridmind LoRA / OpenAI)  phrasing only, no numbers of its own
      |
      v
guardrail gate (section 5)          checker + lint, verified ONCE per card
      |
      v
fan-out ---> X adapter -----------> deterministic render + X rate limiter
        ---> Discord adapter -----> deterministic render + Discord rate limiter
        ---> (future adapters)      Bluesky / Mastodon / Threads
```

### 3.1 The platform-neutral event card

The unit that flows through the pipeline is not a tweet; it is an **event card**, a
structured, platform-neutral object:

- Identity: `card_id` (deterministic from the dedupe key, section 4.3), post type,
  priority class, storyline id (links follow-ups to their originating card), TTL.
- Verified content: the numeric slots (filled by code from the frozen payload, each
  slot tagged with its source field and tier label), `color_short` (one punchy LLM
  sentence, sized for X), `color_long` (two or three LLM sentences for embed-friendly
  platforms), and the numberless fallback line for the type.
- Provenance: the frozen stream payload snapshot the card was built from, the checker
  verdict, and the generation trace (for the audit log).

Both color fields are generated in one LLM call and verified together at the gate, so
a card that clears the gate is publishable on every platform without re-verification.

### 3.2 The `Publisher` interface

Each platform implements one interface (described functionally, no code):

- **Capability descriptor**: max text length, thread/reply support, rich-embed support,
  media support, effective rate limits, and which priority classes and post types the
  platform subscribes to. The budgeter (section 4.4) and renderer read capabilities;
  they never special-case platform names.
- **render(card)**: deterministic assembly of the card's verified pieces (numeric
  slots, one of the color fields, fixed template text) into the platform's native shape
  (a 280-char post, a Discord embed). Rendering may TRUNCATE or OMIT verified pieces;
  it may never add, reformat, or recompute a number. This rule is what lets the gate
  run once (section 5.2).
- **publish(rendered, idempotency_key)**: send, respecting the platform's own rate
  limiter, recording to the shared sent-ledger (section 3.3).
- **health()**: reachable/credentialed/limited; a sick publisher is skipped, never
  blocks the others.

Concrete adapters at launch: **X** (section 6.1) and **Discord** (section 6.2).
Bluesky, Mastodon, and Threads are each one more adapter (section 6.3); nothing
upstream of the fan-out changes.

### 3.3 Fan-out, shared dedupe, idempotency

- **Shared dedupe before the fan-out**: the dedupe keys (section 4.2) are evaluated
  once, on the card, so the same underlying event can never become two cards no matter
  how many platforms exist.
- **Per-platform idempotency after the fan-out**: the sent-ledger records
  `(card_id, platform)` before each send attempt and reconciles after. A retry
  following an ambiguous failure (timeout after send) consults the ledger and the
  platform's recent-posts state before re-sending, so a platform's retries can never
  double-post the same card, and one platform's retry storm cannot affect another.
- **Independence**: publishers run concurrently and fail independently. X being
  rate-limited does not delay Discord; Discord's webhook being revoked does not mute X.
  TTL expiry is evaluated per platform at send time (a card can make Discord's cheap
  limits but miss X's queue; that is correct behavior, not a bug).

---

## 4. Content model

### 4.1 Post taxonomy

Seven post types, each with a priority class and a freshness TTL (a card older than
its TTL at a given platform's send time is dropped there, never posted late):

| Type | What it says | Priority | TTL |
|---|---|---|---|
| `SESSION_OPEN` | Coverage starts: GP, session, our driver, starting compound/position | P1 | none |
| `PIT_CALL` | The orchestrator's pit recommendation: action, target compound, window laps, key probability | P0 | 90 s |
| `UNDERCUT_ALERT` | Undercut/overcut window on a named rival: gap, tyre age delta, N16 probability | P1 | 90 s |
| `SC_REACTION` | SC/VSC deployed or ending: lap, cheap-stop math (pit loss under SC vs green), model stance | P0 | 45 s |
| `TYRE_NOTE` | Stint color: degradation cliff proximity (TCN P10), pace trend on current compound | P2 | 2 laps |
| `RIVAL_MOVE` | Anticipated rival strategy (Rival Agent output), framed as model expectation | P1 | 90 s |
| `SUMMARY` | Post-session wrap: strategy timeline, calls made vs what happened, honest scorecard | P1 | none |

`RIVAL_MOVE` is gated on the Rival Agent existing (the TFM deliverable); the taxonomy
reserves its slot now so the content engine does not need redesign later.

### 4.2 Trigger rules (stream event to card)

Triggers are pure functions of the event payload plus a small dedupe store; no trigger
ever consults anything outside the stream:

- `SESSION_OPEN` fires on the `start` event, once.
- `PIT_CALL` fires when a `lap` event's recommendation carries a pit-type action with
  confidence at or above a configured threshold, and no `PIT_CALL` card exists for the
  same pit window (dedupe key: driver + stint + window). A window that closes without a
  stop may emit one follow-up card in the same storyline (section 4.5).
- `UNDERCUT_ALERT` fires when the undercut probability for a tracked rival crosses the
  threshold upward, with hysteresis: one card at the crossing, re-armed only after the
  probability falls below a lower bound or the window resolves (dedupe key: driver +
  rival + stint).
- `SC_REACTION` fires on a track-status transition to SC/VSC (and on the restart),
  sourced from the stream's `race_control`-derived track status. This is the only
  trigger allowed to bypass lap-spacing (section 4.4), because its value decays in
  seconds.
- `TYRE_NOTE` fires at most once per stint, when the degradation model's P10 cliff
  estimate comes within a configured lap horizon.
- `RIVAL_MOVE` fires when the Rival Agent's predicted move probability crosses its
  threshold (same hysteresis pattern as `UNDERCUT_ALERT`).
- `SUMMARY` fires on the `summary` event.

### 4.3 Card identity

`card_id` derives deterministically from the dedupe key (type + its natural key +
session), which makes idempotency (3.3) and the audit trail (5.2) line up: one real
event, one card, at most one post per platform.

### 4.4 Cadence and budgets, per platform

A busy race produces far more trigger hits than any audience should receive. Budgets
live in each adapter's capability descriptor; the shared pipeline only enforces
trigger-level dedupe and hysteresis:

| Throttle | X | Discord |
|---|---|---|
| Session budget | 12 in-session cards + `SUMMARY` thread (sized to the Free tier daily cap, section 6.1) | 30 in-session cards + `SUMMARY` (reader-spam bound, not API bound) |
| Reserved P0 allocation | 2 (P0 may post past budget) | 4 |
| Spacing | min 1 lap (~90 s) between posts; P0 may break spacing | min 30 s; P0 may break spacing |
| Types subscribed | P0 + P1 only (`TYRE_NOTE` excluded by default) | all types including P2 color |
| Eviction | priority queue; P0 evicts queued P2 | same |

The X budget is a hard product parameter, not a tuning knob: the content model is
deliberately sized to X's free rate limit (section 6.1). Discord's looser budget is a
policy choice for readability, and it is also why Discord carries the fuller feed:
same cards, more of them.

### 4.5 Structure and per-platform content adaptation

The SAME card renders differently per platform, from the same verified pieces:

| Aspect | X render | Discord render |
|---|---|---|
| Shape | Single post, max 280 chars: skeleton line + `color_short` + at most one driver tag and one event hashtag | Rich embed: title (type + lap), field grid (the numeric slots with labels), `color_long` as description, footer with session + "model opinion" note |
| Urgency signal | Leading emoji per priority (kept minimal) | Embed accent color by priority (P0 red, P1 amber, P2 neutral) |
| Storyline | Follow-ups reply to the original post; P0/P1 originals standalone for reach | Follow-ups post in-channel referencing the original message; optionally one thread per storyline |
| `SUMMARY` | Thread of 3 to 4 posts | Single long embed (or two), full timeline + scorecard table |
| Media | Deferred to B5 (tier-dependent, section 6.1) | Chart image attached from B4 (free, section 6.2) |
| Detail level | Terse: 2 or 3 numbers max, the rest cut by the renderer | Full: every verified numeric slot the card carries, labeled |

Renders adapt AMOUNT and PACKAGING, never substance: both platforms show numbers from
the same verified card, and truncation on X is slot omission, never rounding or
recomputation.

### 4.6 Media (charts)

A chart image (gap evolution, stint timeline) is rendered server-side from the same
frozen payload the card used, reusing the pit-wall/ECharts layer headlessly, so chart
numbers satisfy the same traceability rule as text. Discord gets charts from public
launch (attachment upload is free and unlimited in practice); X gets them in B5 once
the tier question (section 6.1) is settled. Charts are attached media on an existing
card type, never a separate post budget.

---

## 5. The never-invent-numbers guardrail (verify-before-post)

This is the product-defining section. The gridmind design (its section 6) specifies
the shared machinery; box-bot wires it in front of the fan-out so no unchecked number
can reach any public timeline.

### 5.1 Generation posture: deterministic skeleton, LLM color

The strongest guardrail is structural: numbers are placed by code, not by the model.

- Every post type has a deterministic skeleton: code fills every numeric slot directly
  from the frozen event payload (lap numbers, gaps, probabilities, compounds, window
  bounds), each slot tagged with its source field. The skeleton alone is always a
  publishable post on every platform.
- The LLM (gridmind LoRA via LM Studio when it exists, an OpenAI mini model until then,
  never Anthropic; low temperature per gridmind 6.5) contributes only the two color
  fields (`color_short`, `color_long`): phrasing, tone, context, generated from a
  prompt containing only the same frozen payload plus the house style rules. The prompt
  template forbids outside statistics explicitly (gridmind threat T6).
- The LoRA's `bot-style` training subset (gridmind section 4) targets exactly this
  register: punchy, present tense, grounded, no invented color.

### 5.2 The gate pipeline (runs once per card)

1. **Freeze the context.** The exact `lap` (or `start`/`summary`) event payload and
   recommendation the card was built from is snapshotted and stored on the card.
   Verification always runs against this snapshot, never against "current" state.
2. **Numeric grounding check.** The gridmind-published numeric grounding checker
   (gridmind 6.2) runs over the card's full publishable text (skeleton slots plus BOTH
   color fields) against the snapshot: every numeric span (integers, decimals,
   percentages, times with units, probability-like decimals) must be grounded (verbatim
   match, formatting-tolerant) or derivable (depth-1 arithmetic over snapshot numbers).
   Any fabrication hit fails the card.
3. **On failure: one regeneration, then the numberless fallback.** Exactly the gridmind
   6.5 contract: a fabrication hit blocks the card and triggers one LLM regeneration of
   the color fields; if the retry also fails, the card ships with the numberless
   fallback line in place of color ("Box for NOR, hards, rejoining in traffic") or, for
   P2 types, is dropped entirely. The bot never publishes an unchecked number and never
   loops regeneration.
4. **Content lint.** A deterministic pass for the non-numeric rules: banned vocabulary
   (betting/tipping terms, guarantees, abuse), estimate-tier labeling ("est." on
   estimate fields if quoted at all), mention/hashtag budget, duplicate-text check
   against the session's already-posted set.
5. **Audit log.** Every card, verdict, regeneration, per-platform send outcome, and
   final action (posted, fallback, dropped, TTL-expired) is logged with its snapshot.
   The log is the shadow-season metric source (verifier block rate, gridmind's bot
   gate) and the postmortem trail if a bad post ever ships.

Because per-platform rendering is deterministic and can only omit or truncate verified
pieces (3.2), the gate runs ONCE per card and its verdict covers every platform. The
only per-platform check after the gate is mechanical lint (length fits, mention budget
fits), which involves no numbers.

The checker is consumed as the gridmind-published artifact so both repos run one
implementation (gridmind 6.5 states this explicitly). box-bot does not fork or
reimplement it; if gridmind has not shipped the checker yet, that phase of box-bot
waits (roadmap, section 10).

### 5.3 Fail-closed rules

- Checker unavailable or erroring: no posts on any platform (the invariant is
  "verified", not "probably fine").
- LLM unavailable: skeleton-only cards are permitted (they contain only code-placed
  numbers and still pass the checker), so an LM Studio outage degrades style, not
  safety.
- Stream stale or down: no posts (section 7).
- Models behind the 2026 drift gate: recommendation-bearing types (`PIT_CALL`,
  `UNDERCUT_ALERT`, `RIVAL_MOVE`) disabled on all platforms (the live consumer's
  section 7 rule applied downstream).

### 5.4 Opinion, not fact

Posture rules baked into the templates and the profiles:

- Every recommendation is attributed to the model, never stated as fact or as knowledge
  of team intentions: "StratLab calls", "our model expects", "the window opens on our
  numbers". `RIVAL_MOVE` posts especially: they predict, they do not report.
- Account/server bios carry the standing disclaimer: unofficial, model-generated
  strategy opinion, part of the F1 StratLab ecosystem, not affiliated with F1, FOM,
  FIA, or any team, and not betting advice. Discord embeds repeat a one-line footer
  note ("model opinion, not team information") because embeds are shared out of
  context.
- No odds framing, no "lock", no tipping language, ever (enforced by the content lint).

---

## 6. Platform adapters

### 6.1 X adapter

- **Tier and cost reality** (as of this design's date; X pricing churns, re-verify at
  implementation, open question Q1): Free tier is write-focused, roughly 500
  posts/month with a daily write cap around 17 posts per 24 h and negligible read
  allowance, at zero cost. Basic is around 200 USD/month for roughly 3,000 posts/month
  and unproblematic media upload.
- **Decision: design to the Free tier and make it sufficient.** The X session budget
  (12 in-session + a 4-post summary thread = 16) fits under the daily cap with the race
  as the only covered session that day; a month of race + sprint coverage stays well
  under the monthly cap. Basic becomes worth it only if coverage expands (quali + FP +
  media); that is a B5 decision, not a launch requirement.
- **Auth and secrets**: the bot posts as its own dedicated account via OAuth
  (user-context token). Keys live in the host's environment/secret store, never in the
  repo; the public repo ships an `.env.example` with names only.
- **Compliance**: the account carries X's automated-account label, managed by a
  personal account (required by X automation policy); no duplicate content (dedupe
  layer); minimal mentions/hashtags; back off on 429s, never retry through limits.
  Suspension is an existential outage for this adapter (risk register).
- **Rate limiter**: serialized sends, per-endpoint limits respected, TTL expiry empties
  a backed-up queue naturally; nothing is posted late to "catch up".

### 6.2 Discord adapter

- **Mechanism, decided**: an **incoming webhook** per target channel for v1. box-bot is
  post-only; a webhook needs no gateway connection, no privileged intents, no bot
  presence, and its rate limits (roughly 5 requests per 2 s per webhook, plus global
  limits) are far above the budget. A full bot application (slash commands, roles,
  subscriptions like "ping me on SC") is a deliberate later upgrade and does not change
  the pipeline, only this adapter.
- **Where it posts**: the F1 StratLab community server (to be created; open question
  Q3), one channel per feed, suggested `#race-live` (P0/P1), `#race-color` (P2), and
  `#session-summaries`. Channel routing is part of the adapter's capability config.
- **Render**: rich embeds (section 4.5): title, labeled numeric fields, urgency accent
  color, `color_long` description, chart image attachment from B4, footer disclaimer.
  No 280-char corset, so Discord is the surface that shows the model's full verified
  output.
- **Secrets**: webhook URLs are credentials; same secret-store discipline as X tokens,
  plus rotation on any suspicion (a leaked webhook lets anyone post to the channel).
- **Cost**: zero. This is why Discord is also the launch-first, lower-stakes surface in
  the roadmap (section 10).

### 6.3 Future adapters (each is just another `Publisher`)

- **Bluesky**: AT Protocol, free API, ~300-char posts, media supported; closest to the
  X render path.
- **Mastodon**: free API per instance, 500-char default, media supported; X-like render
  with more room.
- **Threads**: Meta's API, OAuth app review required; evaluate only if audience data
  justifies the review cost.

None of these change the pipeline, the card, or the gate; each is a capability
descriptor, a renderer, and a rate limiter.

---

## 7. Scheduling and lifecycle (auto-posting)

Posting is fully automatic during live sessions: no human approves individual posts.
The human controls are all upstream and structural: the calendar (when the bot may
run), the budgets and thresholds (config), the guardrail gate (mechanical), and the
kill switch (one config flag that silences all publishers immediately).

box-bot is a well-behaved service that is off far more than it is on. Five states:

| State | Meaning | Entry | Exit |
|---|---|---|---|
| `DORMANT` | Between race weekends; process may not even be running | default | calendar says a covered session is near |
| `ARMED` | T minus ~30 min before a covered session; connecting to the core stream, posting disabled | calendar | `start` event received, stream healthy |
| `LIVE` | Session running, stream healthy, posting enabled | `ARMED` | `summary` event, or stream loss |
| `DEGRADED` | Stream lost mid-session; silent, reconnecting | `LIVE` | stream back (to `LIVE`) or session-end timeout (to `COOLDOWN`) |
| `COOLDOWN` | Session over: publish `SUMMARY` (if the session was actually covered), flush the audit log, disconnect | `LIVE`/`DEGRADED` | done, back to `DORMANT` |

- **Calendar awareness.** Activation windows come from a season calendar file in
  box-bot's own config (session date/times per GP, which sessions are covered),
  refreshed from a public calendar source at the start of each race week. The calendar
  only decides WHEN to arm; whether there is anything to narrate is decided solely by
  the core stream actually publishing a session. If the core is not running, `ARMED`
  times out back to `DORMANT` and posts nothing anywhere.
- **Covered sessions v1**: races and sprint races only. Quali and FP coverage is a B5
  expansion (more sessions is where X's daily cap starts binding).
- **Stream-down behavior.** In `DEGRADED` the bot posts nothing and never extrapolates.
  Exactly one operational post is allowed per platform: if the session had already been
  opened publicly and the outage exceeds a configured window, a single "coverage
  paused" note, so followers are not left mid-story. No data content, no guesses, no
  backfill of missed laps on reconnect.
- **Per-platform health**: a sick publisher (revoked webhook, X rate-lock) is skipped
  while the others continue; health transitions are logged, never posted about.
- **Between weekends** the service is off (a scheduler starts it for the
  armed window and stops it after cooldown). No idle polling, no idle connections.

---

## 8. Repo topology and deployment

- **Repo**: independent public repo (working name `box-bot`) under a personal GitHub
  account, bootstrapped with the standard baseline (branch protection, CI, Dependabot, release
  automation). Its CI runs unit tests over triggers/budgets/renderers, the content lint
  suite, and the stream contract test against the pinned core release. The core repo is
  not touched: no submodule, no workflow, no import, no mention required (ecosystem
  checklist step 6).
- **Branding rule**: the name carries no "f1stratlab", so the README, the X profile,
  and the Discord server description must all state explicitly that box-bot is part of
  the F1 StratLab ecosystem, with links to the core repo and docs site. Same rule as
  gridmind/radiogate/pitlab.
- **What it pins**: core release tag + stream schema version (contract), the
  gridmind-published checker artifact version, and (when adopted) the gridmind LoRA
  revision served in LM Studio. Every pin bump is an explicit commit.
- **Where it runs (v1 decision)**: co-located with the core live stack on the same GPU
  workstation during race weekends. Rationale: the core's live consumer and backend
  already run there during a session, LM Studio needs that GPU to serve the LoRA, and
  the bot only needs to exist while the stream does. The correlated-failure objection
  (same host, same outage) is acceptable because the bot's outage behavior is silence
  (section 7). A small always-on VPS with the OpenAI provider is the documented upgrade
  path if weekend reliability becomes the bottleneck; the provider-agnostic LLM layer
  makes that a config change, not a redesign.

---

## 9. Safety and reputation

- **Disclaimers**: bio/server-level plus the automated-account label on X and the embed
  footer on Discord (section 5.4). No per-post disclaimer spam on X; the framing rules
  carry the posture in the text itself.
- **No betting surface**: no odds, no tipping language, no "guaranteed", no engagement
  with betting accounts or servers. Enforced by the content lint and by policy in the
  README.
- **Being wrong in public, gracefully**: strategy calls will be wrong; the product
  stance is to own it. Wrong calls are never deleted (the timeline is the record); the
  `SUMMARY` scorecard states what the model called and what actually happened, on both
  platforms. Honesty is the differentiator versus hype accounts and costs nothing but
  pride.
- **Corrections**: a post that is wrong because of a BUG (wrong number shipped, wrong
  driver tagged) is deleted/edited (Discord allows edits; X requires delete + repost)
  and followed by a correction note, and the incident goes to the audit log and an
  issue. Fabrication-class incidents (a number that should have been blocked)
  additionally freeze posting on all platforms until the gate failure is understood.
- **Rate-limit and outage behavior**: back off, go quiet, never retry-spam (sections
  3.3, 7). A silent race is a non-event; a spammy or hallucinating race is a
  reputation incident on every platform at once.
- **The 2026 drift gate**: no recommendation posts from models the #189 retraining
  pipeline has not re-validated for the current regulation. The bot can still open
  sessions and post observed-fact content (SC deployed, pit happened) in that window,
  but model-opinion post types stay disabled. This restates the ecosystem rule that
  Fase 4 precedes Fase 5.

---

## 10. Phased roadmap

| Phase | Deliverable | Gate |
|---|---|---|
| **B0: repo + contract** | Repo bootstrapped; stream schema pin + contract test against a core release; season calendar config | Core exports the stream schema (ecosystem Q5); live consumer L4 shape known |
| **B1: content engine, dry-run** | Triggers, dedupe, budgets, priority queue, TTLs, card builder, skeleton templates, both renderers writing to a log; no LLM, no network publishers | B0; recorded streams from the live consumer's shadow-replay harness (L1) |
| **B2: guardrail gate + publishers** | Checker wired as the blocking gate; LLM color path (OpenAI mini first); regeneration + fallback + lint + audit log; X and Discord adapters implemented behind the `Publisher` interface | B1; gridmind G1 (checker artifact published) |
| **B3: shadow season** | Full pipeline over live sessions publishing ONLY to a private Discord channel (the shadow surface; no X posts); measure verifier block rate, end-to-end latency vs TTLs, per-platform budget behavior | B2; live consumer L4 in production; aligns with gridmind G6's shadow-run |
| **B4: public launch** | Public Discord server first (full feed + charts), then the X account (races + sprints, text-only, Free tier); scorecard summaries on both | B3 metrics green; the 2026-model gate satisfied (or a pre-drift-season race); identity confirmed (Q2/Q3) |
| **B5: expansion** | gridmind LoRA as primary phrasing model; X chart media (tier decision); quali/FP coverage; `RIVAL_MOVE` posts; Bluesky/Mastodon adapters if wanted | B4 stable; gridmind LoRA shipped; Rival Agent exists (for `RIVAL_MOVE`) |

Discord launches before X deliberately: it is free, editable, lower-stakes, and its
private channel doubles as the B3 shadow surface, so the pipeline is battle-tested on
a real platform before the first tweet exists.

---

## 11. Risks

- **X pricing/policy churn**: tiers, caps, and automation rules change on short notice.
  Mitigation: X budget sized with margin under the Free cap; the tier decision
  revisited at B5; the adapter isolates the API surface; Discord carries the product
  regardless.
- **Account suspension (X)**: the existential outage for that adapter. Mitigation:
  automation label, dedupe, conservative volume, posting-only (no engagement
  automation); Discord and future adapters keep the product alive.
- **Verifier over-blocking mutes the bot**: a high block rate turns race coverage into
  silence. Mitigation: gridmind's bot gate defines an operability threshold on block
  rate; the skeleton-first posture means most cards can ship without any LLM numbers to
  block; measured in B3 before anyone watches.
- **Latency makes posts stale**: LLM + verify + queue exceeding TTLs on the laps that
  matter most (SC, pit windows). Mitigation: TTL-drop rather than late posting;
  skeleton-only fast path for P0 if the color path is slow; latency measured in B3.
- **Cross-platform double-posting or drift**: retries or renderer divergence showing
  different numbers per platform. Mitigation: one card, one gate, deterministic
  renders that may only omit; the sent-ledger idempotency (3.3).
- **Correlated outage with the core host** (v1 co-location). Mitigation: accepted
  consciously; silence is safe; VPS upgrade path documented.
- **Reputational blast radius of a loud wrong call**: mitigated by opinion framing, the
  scorecard habit, and never deleting honest misses.
- **Trademark exposure**: "F1" and GP names in handles/server names invite platform or
  rights-holder friction. Mitigation: names avoid protected marks (Q2/Q3), profiles
  state unofficial status, content quotes public timing data only.
- **Upstream drift**: the core's stream schema evolves additively by rule, but box-bot
  must not lag forever on an old pin. Mitigation: contract test in CI plus a scheduled
  pin-bump review each season.

---

## 12. Open questions

1. **X tier at launch**: confirm Free tier with the 16-post budget for B4, deferring
   Basic (about 200 USD/month) to B5 if coverage expands. Re-verify the exact caps at
   implementation time; they will have changed.
2. **X account identity**: handle and display name (avoiding protected marks), bio
   wording, which personal account owns the automation label. Suggest deciding at B3 so
   the shadow setup can mirror the final identity.
3. **Discord home**: a new public F1 StratLab community server (recommended: it doubles
   as the ecosystem's community hub), or webhooks into an existing server? Channel
   layout confirmation (`#race-live` / `#race-color` / `#session-summaries`).
4. **Discord upgrade path**: stay webhook-only, or plan the full bot application
   (slash commands, "ping me on SC" role subscriptions) as a B5+ feature?
5. **Language**: English-only posts (recommended for reach and one style model), or
   bilingual ES/EN? Bilingual doubles the per-post budget cost on X or halves coverage;
   on Discord it could be a second channel instead.
6. **Covered sessions v1**: races + sprints only (recommended), or include quali from
   day one at the cost of X's daily write cap on Saturdays?
7. **Scorecard posture**: confirm the honest post-race "called vs happened" summary as
   standing policy (recommended; it is the reputation moat), or keep summaries
   result-only.
8. **`RIVAL_MOVE` comfort check**: posts predicting named rivals' strategies are the
   most shareable and the most wrong-in-public content type. Confirm the framing rules
   suffice, or whether rival predictions stay Discord-only (or shadow-only) for a
   season.
9. **Priority-class split per platform**: confirm X = P0/P1 only and Discord = full
   feed including P2 color, or should X also carry `TYRE_NOTE` at the cost of budget?
