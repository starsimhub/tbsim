# TBsim Resistance — Implementation Decisions

Decisions taken while executing [update-plan.md](update-plan.md). The plan's Phase 0 asks that any open decision be resolved and recorded here; this file is that record. Where the plan already recommended an option, the recommendation is confirmed with the concrete parameterization chosen.

Each decision notes the option chosen, the rationale, and the resulting default (which is always backward-compatible with the pre-update behavior unless stated otherwise, so the existing test suite keeps passing).

---

## D-L1 — DST result expiry / retreatment semantics

**Question (plan [L1]).** Should the semantics be "one test → one treatment attempt" (require a fresh re-test before re-treatment) or "test valid for N months"? Which of result-expiry, a retreat guard, or both does Phase 1 implement?

**Decision.** Implement **both independent levers**, each defaulting to *off* so current behavior is preserved:

1. **Result expiry** — `DSTDelivery(result_validity=None)` (an `ss.dur`; `None` = never expires). Each step, before administering, results older than the window are wiped (`dst_tested→False`, `dst_profile→0`, `ti_dst→nan`), so the agent is re-tested by the default eligibility (`~dst_tested`).
2. **Freshness gating at the eligibility site** — `DSTDelivery.matches(..., max_age=None)` and `observed_resistant(drug, max_age=None)`. When set, an agent matches only if `sim.ti - ti_dst ≤ max_age/dt`.
3. **Retreatment refractory guard** — `TxDeliveryR(retreat_after=None)` (an `ss.dur`). An agent whose most recent course *ended* fewer than `retreat_after` steps ago is excluded from (re-)initiation. This is the most direct fix for "continuously retreated" and works for both rate-based and eligibility-routed deliveries.

**Recommended usage / intended semantics.** The realistic clinical model is **"a DST result is valid for a finite window"**: set `result_validity` to the clinical re-test interval (e.g. `ss.months(12)`) so a stale result cannot drive perpetual re-treatment, and optionally add `retreat_after` as a hard refractory period after any course. Using `retreat_after` alone reproduces the stricter "one attempt per result" reading.

**Rationale.** The two mechanisms answer different questions (when does a *test* go stale vs. how soon may a *patient* be re-treated) and compose cleanly. Making all three opt-in keeps the ODE-validation and existing behavior bit-identical by default.

---

## D-L3 — Latent-treatment divergence from base tbsim

**Question (plan [L3]).** When an eligibility/DST-routed `TxDeliveryR` selects a latent (`INFECTION`) agent, should it run a full treatment course (current) or clear immediately like base `tbsim.TxDelivery` (which sends latent agents straight to `CLEARED`)?

**Decision.** **Option B (plan's recommendation).** Add `TxDeliveryR(treat_latent=False)`.

- `treat_latent=False` (**default**): latent agents selected for treatment are cleared immediately — `state→CLEARED`, `rr_reinfection=rr_reinfection_cleared`, reinfection-wane scheduled, `strain_mask→0`, counts→0, `infected=False`, `susceptible=True` — matching base tbsim, and are **excluded** from the course-based path (they never enter `TREATMENT`, never appear in `n_acquired`).
- `treat_latent=True`: keep the current behavior (latent agents run a course that can fail / acquire resistance), for modeling latent-TB regimens explicitly.

**Rationale.** Defaulting to base-tbsim behavior is the least surprising and keeps single-strain and strain-aware models consistent, while the flag preserves the richer path for those who want it. Note the default rate-based eligibility never selects latent agents, so this only changes behavior for custom/DST-routed eligibilities.

---

## D-L4 — Acquisition strain selection (treatment-failure and TPT)

**Question (plan [L4]).** On an acquisition hit ("one trial per drug per episode"), which carried drug-susceptible strain acquires resistance? Currently the **lowest-id** carried susceptible strain is flipped, which is arbitrary and pan-leaning.

**Decision.** **Option A.** Pick **one carried strain susceptible to the hit drug, uniformly at random by default** (∝ fitness optionally), via a shared helper reused by `TxR.acquire`, `TPTRx._acquire`. Exposed as `acq_select='random'` (values `'random' | 'fitness'`) on `TxR` and `TPTRx`, paralleling the module's `prog_select`.

- Implemented with a CRN-safe per-agent `choice2d` over the strains that the agent both **carries** and is **susceptible to** for that drug; the chosen strain is replaced by its `| drug_bit` counterpart (replacement semantics, unchanged).
- Cardinality is unchanged: still one trial per drug per episode.

**Rationale.** Removes the pan-leaning bias and lets acquisition land on a strain already carrying other resistances (e.g. RIF-R + FQ hit → RIF+FQ), which the lowest-id rule could never produce. In the `n=1` two-strain case there is only ever one carried susceptible strain, so this is **identical** to the old behavior — the ODE-validation and π(m→s) tests are unaffected.

---

## D-COUNTER — Per-strain counter semantics (spec updates §1–§6)

The spec updates add a per-strain multiplicity counter. Design decisions for the parts the spec leaves implicit:

**Representation.** `m = 2**n` scalar `ss.IntArr('strain_count_{j}')` states, held in a list `self.strain_counts`, with `_counts(uids) → (k, m)` / `_set_counts` / `_add_counts` helpers. Chosen over a Python-object per-agent array for Starsim compatibility and vectorized performance (`m` is small: `n≤3 → m≤8`).

**Core invariant.** `strain_count_j > 0` **iff** `strain_mask` bit `j` is set. Every strain-mask mutation maintains this. Concretely, on any operation that changes an agent's carried set:
- a strain **bit newly set** → its count becomes **1** (unless the same operation is an identical re-exposure; see below),
- a strain **bit cleared** → its count becomes **0**,
- a strain **bit unchanged** → its count is unchanged.

**Where multiplicity >1 comes from.** *Only* repeated transmission/seeding of an already-carried strain. On a transmission/seed event where the target already carries the drawn strain, that strain's count is **incremented** (`+= 1`). A brand-new infection of a susceptible agent starts the founding strain at count **1**, regardless of how many copies the source carried (spec §1).

**Within-host strain-id changes** (de-novo mutation, treatment-failure acquisition, TPT acquisition, bottleneck collapse) follow the invariant rule above: newly-present strains get count 1, newly-absent strains get count 0. That is, an *emergent* resistant lineage is a fresh count-1 lineage; mutation does **not** carry over source multiplicity. This is consistent with "a newly-infected agent always starts at 1" and with the spec's rule that **count does not affect acquisition** — the counter is not read by any acquisition/treatment/DST code path.

**Consumers (only two, per spec §1 "scope").**
1. **Transmission multinomial** — P(pass strain `j`) ∝ `count[j] × fitness[j]`. The *overall* per-contact transmission probability (`rel_trans = max carried fitness`) is **unchanged** and does not depend on count.
2. **Progression bottleneck** (`p_multi < 1`) — when one strain survives, P(survivor = `j`) ∝ `count[j]` (× `fitness[j]` if `prog_select='fitness'`).

Clearance/resolution (natural), successful treatment cure, and death reset the relevant counts to 0. DST, treatment efficacy, and acquisition probability ignore the counter entirely (coupling of identical strains is automatic under the one-bit-per-id mask).

**Diagnostic result rename.** The obsolete `new_blocked_superinf` result (which counted *blocked* identical-strain events) is renamed to **`new_identical_superinf`**, now counting identical-strain **count-increment** events (the same physical event, no longer blocked). Internal counter `_n_blocked → _n_identical_superinf`.

**Consequence for ODE validation.** The reference two-strain ODE (`ode.r`) has no multiplicity counter. So wherever `count > 1` arises (σ>0 superinfection + a multi-strain agent + `p_multi<1` bottleneck), the ABM legitimately diverges from the ODE: count-weighted transmission *and* the count-weighted bottleneck amplify competitive exclusion (a more-transmitted strain accumulates higher multiplicity and wins bottlenecks and transmission draws). In the strong-bottleneck regime of devtest `test_q2b_fitness_selection_lowers_resistance` this drives the less-fit strain fully extinct by end-of-run in *both* random and fitness selection modes (the ABM endpoint ties at 0 where the count-less ODE keeps it positive). The Q2b directional signal (fitness selection lowers resistance faster than count-only selection) is preserved and is now asserted on the **time-averaged** resistant fraction rather than the endpoint. This is expected, spec-correct behavior, not a regression; the survivor of a bottleneck **retains** its per-strain count (the "instances persist" reading), which is what produces the amplification.

---

## D-DEFER — Confirmed deferred (needs modeling-team spec input)

Per plan Phase 6, these remain **out of scope** for this pass and are not implemented:

- **[L6] TPT `resist_penalty`** for partial (per-drug) TPT efficacy — explicitly deferred by TR; schedule with multi-drug TPT work.
- **[L10] Time-varying progression hazard** and the remaining §13 backlog (adherence distribution, DST-indeterminate outcome, failure-vs-new-case classifier, LAI_TPT burden table, LTFU) — needs the progression-hazard *shape* from the modeling team; do not implement blind. The clock-reset hook (`ti_infected` reset on every exposure, incl. same-strain) is left in place and ready.
