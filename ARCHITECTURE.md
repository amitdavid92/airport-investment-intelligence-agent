# Architecture & Design

Airport Investment Intelligence Agent — an assistant that helps analysts find
US airports where modernization or expansion is most likely to pay off.

---

## 1. The question, restated

The brief asks for airports "where renovations will be most profitable based on
increased flight and passenger capacity." Ranking by size answers a different
question — the biggest airports are already built out. The question worth
answering is:

> **Where is demand pressing hardest against physical supply, and where can
> something actually be built?**

That framing drives the scoring model: pressure signals earn most of the
weight, and feasibility gates them, because an airport with no land and no
funding route is not an investment candidate no matter how strained it is.

---

## 2. System shape

```
   Streamlit chat UI · web terminal UI
                │
                ▼
            FastAPI
    ┌───────────┼────────────────────────────┐
    │           │                            │
 /chat      /rank  /airport  /compare    /long-haul
 (agent)    ────── no LLM in this path ──────
    │           │
    ▼           ▼
  Gemini     Scoring engine  (pandas — deterministic)
 tool runner       │
    │              ▼
    └────────► Query façade (service.py)
                   │
                   ▼
            Data providers
     BTS T-100 · FAA Runways · FAA Facilities · OurAirports
              live HTTP  │  on-disk cache
```

**The central claim: the language model never computes a number.** It resolves
names, chooses a weight profile, calls tools, and explains what comes back.
Every figure originates in `scoring/`, which has no dependency on `google-genai`.

This is verifiable rather than asserted — `/rank` returns the same numbers the
chat quotes, with no model in the call path:

```bash
curl 'localhost:8000/rank?region=new_england&weight_profile=terminal&top_n=5'
```

`tests/test_service.py::test_api_and_service_agree` pins that equivalence.

---

## 3. Data

Four public sources, **all key-free** — no signup, nothing to expire mid-demo.

| Source | What it provides | Coverage |
|---|---|---|
| **BTS T-100** (ArcGIS FeatureServer) | enplanements, passengers, departures, arrivals, freight, mail | 1,279 US airports, CY2024 |
| **FAA Runways** (NTAD) | runway count, length, width, surface, **pavement condition** | 8,779 runways |
| **FAA Aviation Facilities** (NTAD) | acreage, ownership, tower, location | 4,877 public-use airports |
| **OurAirports** | names, cities, IATA/ICAO/FAA identifiers, coordinates | daily-updated CSV |

After joining and filtering to airports with real traffic and known runways:
**995 scored airports.**

### Sources evaluated and rejected

| Source | Why not |
|---|---|
| **OpenSky historical flights** | `403 You cannot access historical flights` anonymously — OAuth2 client credentials required since March 2026. Implemented as an *optional* provider; never on the demo path. |
| **BTS TranStats bulk download** | `PREZIP` paths 404; the export needs a form session. Too fragile to depend on. |
| **FAA ASPM / OPSNET** | Login required. Not public. |

### Three real reconciliation problems

These are the parts worth reading the code for — each was found by checking the
data rather than trusting the join.

**1. IATA ≠ FAA location identifier.** T-100 is keyed on IATA; the FAA layers
are keyed on the FAA LocID. They differ for a few hundred airports — AZA/IWA,
FCA/GPI, SCE/UNV. Joining naively silently dropped **115 airports**, some with
over a million passengers. Fixed by routing FAA joins through OurAirports'
`local_code` ([`ingest.py`](src/airport_iq/ingest.py)).

**2. Identifier drift between vintages.** Palm Beach (4.1M enplanements) appears
in 2024 T-100 as `PBI` but has since been recoded to `DJT`; it exists under
*no* FAA layer as `PBI`. A documented alias table reconciles the handful of such
cases rather than losing them silently.

**3. `EXCELLENT` is better than `GOOD`.** The first pass treated `GOOD` as the
only sound pavement condition, so airports with `EXCELLENT` runways — JFK, EWR,
SEA — scored **maximum renewal need**. The signal was exactly inverted, and the
ranking looked plausible enough to ship. Separately, ~1,700 runways publish a
blank condition, which was being read as "bad" rather than "unknown".

Both are now handled by an explicit severity scale:

| Condition | Renewal severity |
|---|---|
| `EXCELLENT`, `GOOD` | 0.0 |
| `FAIR` | 0.5 |
| `POOR` | 0.9 |
| `FAILED` | 1.0 |
| *blank* | **unknown → NaN** (component dropped, weights renormalized) |

The general rule this taught: **missing data must never be scored as zero.**
It is either unknown — in which case the component drops out and its weight is
redistributed — or it is a real value.

---

## 4. Scoring methodology

Five components, each a 0–100 score. Weights come from a named profile.

| Component | Measures | Raw KPI |
|---|---|---|
| **Demand Pressure** | runway-system saturation | `(departures + arrivals) / runway_count` |
| **Terminal Strain** | gate/terminal saturation via aircraft gauge | `passengers / departures` |
| **Unmet Demand** | pressure × how binding the constraint is | composite, damped by land headroom |
| **Renewal Need** | pavement work implied | severity-weighted share of runway length |
| **Expansion Feasibility** | can a project happen | land per traffic + public ownership + long-haul capability |

```
InvestmentScore = Σ wᵢ · componentᵢ     (weights renormalized over available components)
```

### Why Terminal Strain is the interesting one

`passengers / departures` is average aircraft gauge. When an airport cannot add
frequency — slots, gates, curbside — airlines respond by flying **larger
aircraft** on the same slots. Rising gauge against flat movements is the
classic fingerprint of a *terminal*-side constraint rather than a runway one.
That distinction is what makes the brief's "terminal expansion" question
answerable with something better than "the busiest airports."

### Unmet demand

```
pressure   = 0.5 · demand_pressure + 0.5 · terminal_strain
constraint = 1 − 0.5 · land_headroom_percentile
unmet      = pressure × constraint
```

An airport with abundant land can relieve its own pressure, so the constraint
is less binding and the score is damped. A boxed-in airport carries its
pressure at full weight. The damping is capped at half so that a saturated
airport still registers unmet demand even with room to grow.

### Normalization choices

- **Percentile rank, not min-max.** US traffic is extremely long-tailed;
  min-max would compress every airport except Atlanta into the bottom few
  percent and the ranking would only restate "big airports are big."
  Values are winsorized at p1/p99 first.
- **Renewal Need is the deliberate exception.** It is already a meaningful
  fraction — "34% of runway length is not in sound condition" — and
  percentile-ranking it would destroy that meaning, since most airports sit at
  exactly zero and would be spread arbitrarily across the bottom half.
- **National ranking by default**, so scores are comparable across the whole
  dataset. `within_hub_class=true` ranks against FAA size peers instead.

### Peer groups use the real FAA hub classification

Derived from each airport's share of national enplanements — the FAA's own
thresholds (14 CFR Part 158), not invented buckets:

| Class | Share of national enplanements |
|---|---|
| Large hub | ≥ 1% |
| Medium hub | 0.25 – 1% |
| Small hub | 0.05 – 0.25% |
| Nonhub | < 0.05% |

This is what makes the brief's LAX-vs-SNA question honest: they are different
hub classes, so absolute figures are not like-for-like, and the agent is
required to say so.

### Weight profiles

Set in [`config/weights.yaml`](config/weights.yaml), validated to sum to 1.0 at
load time. The agent selects one from the question and states which it used.

| Profile | Leading weight | For |
|---|---|---|
| `balanced` | DP 0.30 | no stated project type |
| `terminal` | **TS 0.40** | terminal / gate expansion |
| `runway` | **DP 0.40** | airfield capacity |
| `renewal` | **IRN 0.45** | pavement rehabilitation |

### Confidence is a first-class output

Every airport carries `data_completeness` (share of inputs present) and
`weight_coverage` (share of the profile's weight actually applied). Their mean
becomes a `low` / `medium` / `high` band, reported alongside the score, with
notes naming any component that dropped out and why.

---

## 5. AI integration — where it is, and where it deliberately is not

**Where AI is used:** intent understanding, name resolution, profile selection,
tool orchestration, and explanation in prose.

**Where it is not:** any arithmetic, any ranking, any score.

Implemented directly against Google's Gemini API (`client.chats.create` /
`chat.send_message`, model `gemini-3.1-flash-lite`) — no agent framework. For
six tools over a local dataset, a framework would add an abstraction layer to
defend in review without changing behaviour.

Tool calling uses the SDK's *automatic function calling*: the plain typed
Python functions in `agent/tools.py` are passed straight to the SDK, which
derives each schema from the signature and docstring, executes the call, and
feeds the result back — all inside one `send_message`. `Chat` keeps the
transcript internally, so `ConversationSession` holds the `Chat` object and no
transcript of its own, and follow-up questions resolve against what was
already said with no extra bookkeeping.

Two implementation notes worth recording, because both cost real debugging
time:

- **The newer Interactions API was tried first and abandoned.** Requests
  returned successfully but were only actually routed through the model after
  being reissued via the SDK's documented `chats.create` path — which itself
  raised a runtime warning pointing there. The `chats` path is what ships.
- **`from __future__ import annotations` silently breaks tool calling.** The
  SDK's argument converter reads `inspect.signature()` rather than
  `typing.get_type_hints()`, so postponed annotations arrive as strings
  (`"str | None"`) and its `isinstance(value, annotation)` check raises
  `TypeError` for every explicitly-passed argument. The tool call fails, and
  the model papers over the gap in prose instead of reporting data. That
  import is deliberately absent from `agent/tools.py`.

**Provider note.** The assignment sets no LLM budget, so the chat runs on
Gemini's standing free tier (a Google account via AI Studio, no card, no
expiry) rather than a paid API. This was a deliberate late substitution — the
system was designed and built against Claude's tool-use model first, and the
swap validates the separation of concerns the architecture claims: `service.py`
and everything below it is unchanged, and the move was confined to the two
files in `agent/` that touch the provider — `tools.py` (the same functions,
now handed to the SDK for automatic schema derivation rather than wrapped in
an explicit tool-runner) and `session.py` (the `Chat` object in place of a
tool-runner loop).

`agent/prompt.py` carries no provider-specific syntax, and the ranking rules
in it did not change. It did gain tool-economy rules — call each tool once,
don't re-fetch what a ranking already returned — after the free tier's
rate limits made redundant calls expensive. Those are stated in
provider-neutral terms and would apply to any tool-using model.

| Tool | Purpose |
|---|---|
| `resolve_airport` | "LA" → LAX, "Santa Ana" → SNA, with match method + confidence |
| `rank_airports` | ranking under an explicit, reported scope |
| `get_airport_profile` | full breakdown for one airport — components ordered by points contributed, which is also the answer to "why does it score this?" |
| `compare_airports` | side-by-side, flags cross-hub-class comparisons |
| `get_long_haul_share` | long-haul share, or an explicit statement that it is unavailable |
| `list_scoring_options` | available profiles, regions, hub classes |

The system prompt enforces the division of labour: every number must come from
a tool result; raw figures must accompany scores; peer group must be named;
`warnings` and `caveat` fields must be surfaced.

---

## 6. Key tradeoffs

**Single year of traffic.** The public T-100 FeatureServer publishes CY2024
only, so there is no growth trend. This measures *current capacity pressure*,
not a forecast — and the agent is instructed never to present it as one. A
multi-year series would need the TranStats bulk export, which is form-gated.

**`ops_per_runway` ignores runway geometry.** It treats four runways as four
units of capacity. San Francisco's close-parallel runways cannot be used
independently in low visibility, so SFO is materially more constrained than the
figure suggests. Modelling this properly needs runway configuration and
meteorological data; the simplification is documented rather than hidden.

**No route-level data, and no invented substitute.** The brief asks for the
long-haul share out of Anchorage. No key-free source publishes per-route
distance. Rather than approximate, `get_long_haul_share` returns
`long_haul_share_available: false` with a runway-*capability* proxy and an
explicit explanation. Supplying `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET`
switches on a real computation — great-circle distance from live OpenSky
departure records — with no other change. **Refusing to fabricate a statistic is
the intended behaviour, not a gap.**

**Capacity pressure ≠ investment return.** The model has no construction cost,
airline agreements, local politics, or environmental constraints. It narrows a
1,000-airport field to a shortlist worth diligence; it does not underwrite.

**In-memory chat sessions.** Fine for a single-process demo, wrong for
production — a real deployment needs a shared store.

---

## 7. Testing

71 tests. The 65 deterministic ones need no API key; the 6 agent tests skip
cleanly without one.

- **Golden values** — published BTS/FAA figures pinned for SFO, ATL, ORD, DEN
- **Determinism** — identical inputs produce identical scores; row order is irrelevant
- **Monotonicity** — more operations on the same runways never lowers demand pressure
- **Regression** — `EXCELLENT` is never scored as needing renewal; blank condition stays NaN
- **Missing-data renormalization** — a component gap never penalizes an airport
- **Scope honesty** — hub-class mixing raises a warning; unknown regions list valid options
- **Refusal to fabricate** — long-haul share reports unavailable without credentials
- **Layer agreement** — `/rank` over HTTP matches the service layer exactly

---

## 8. What I would do next

1. **Multi-year traffic** from the TranStats bulk export → real growth trends,
   turning a pressure snapshot into a trajectory.
2. **Runway configuration modelling** — independent vs dependent parallels,
   which is the largest single source of error in demand pressure today.
3. **Capital cost and ROI** — pressure identifies candidates; cost per added
   passenger would rank them by return.
4. **Catchment demand** — population and income within the drive-time isochrone,
   to distinguish suppressed demand from genuinely small markets.
