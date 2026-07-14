# TBsim Resistance — Update Plan

A single, start-to-finish plan that merges two workstreams into one dependency-ordered sequence:

1. **Spec updates** ([tbsim-resistance-tech-spec-updates.md](tbsim-resistance-tech-spec-updates.md)) — new specification behavior since the implemented spec. Every item stems from one reversal: identical-strain superinfection, previously **blocked**, is now **allowed** via a per-strain **counter** that then threads through progression, clearance, random acquisition, treatment, and DST.
2. **Loop fixes** ([loop_planned_fixes.md](loop_planned_fixes.md)) — 10 review-driven gaps in the current `resistance-updates` implementation (bugs, decisions, features, docs).

References use `file:line` from the two source docs; line numbers drift, so re-check against the working tree before editing. Loop-fix items are cited as **[L#]**, spec-update sections as **[S#]**.

## Key architectural realization (read first)

The counter is the backbone of the spec updates, and it is **additive**, not a rewrite of the strain representation:

- Today each agent carries a single `strain_mask` (`ss.IntArr`, bit *j* = carries strain *j*; `tb_resistant.py:90`). This is pure presence/absence and drives everything: transmission choice (`transmit_probs`, `tb_resistant.py:159`), progression bottleneck (`_select_probs`, `:275`), clearance (`:213,235`), acquisition (`:301-319`), DST, treatment.
- Because the mask has **one bit per strain id**, operations keyed on strain id — DST, treatment cure, resistance acquisition — **already couple identical strains for free**. There is no way to carry "two copies" of a strain under the mask, so "same strains are coupled" [S4, S5, S6] is satisfied by construction once we simply *don't* track those operations per-copy.
- The counter is therefore a **separate multiplicity array** (per-agent, per-strain-id count) that only **two** consumers read: the transmission multinomial [S1] and the progression bottleneck [S2]. Clearance and successful treatment reset it; DST/treatment-efficacy/acquisition-probability ignore it.

This is what makes the spec updates tractable: keep `strain_mask` as-is, add `strain_count`, wire two consumers, reset in the right places, and delete the identical-strain block.

---

## Phase 0 — Lock decisions before coding

Resolve these so execution never stalls midway. The spec-update design is already decided; these are the loop-fix open questions plus confirmation of deferrals.

- **[L1] DST retreatment semantics.** "One test → one treatment attempt" (require a fresh re-test before re-treatment) **vs** "test valid for N months." Determines whether Phase 1 implements result-expiry, a retreat guard, or both. Confirm with TR / the intended `issue-dst-retreatment.md`.
- **[L3] Latent-treatment divergence.** Recommend **Option B**: default to base-tbsim behavior (clear latent agents immediately, no course) but expose a `treat_latent=False` flag to opt into the course-based path. Confirm with TR.
- **[L4] Acquisition strain selection.** Recommend **Option A**: on an acquisition hit, pick the susceptible carried strain to mutate at random / ∝ fitness (parallel to `_select_probs`) rather than lowest-id. This policy is also what the coupled acquisition in Phase 3 uses. Confirm with KG.
- **[L6, L10] Confirm deferred.** TPT `resist_penalty` [L6] and time-varying progression [L10] both need modeling-team spec input and are explicitly out of scope for this pass. Confirm they stay in Phase 6.

---

## Phase 1 — Correctness bugs (independent of the counter)

Land these first: they are isolated, low-risk, and give a clean baseline before the counter refactor touches the same files. Neither depends on counter work.

### 1.1 — DST result never expires → continuous retreatment [L1]

- **Gap.** `DSTDelivery` stores `dst_profile`/`dst_tested` permanently; `ti_dst` is recorded (`dst.py:99-103,161`) but never used to expire a result, so an agent that fails and returns to ASY/SYM is re-selected forever.
- **Changes (per the Phase 0 decision).**
  - Add `result_validity=None` (`ss.dur`, `None` = never expire) to `DSTDelivery.__init__`; in `step`, expire stale results (`self.ti - ti_dst >= result_validity/dt` → `dst_tested=False`, `dst_profile=0`, `ti_dst=nan`) so eligible agents get re-tested.
  - And/or add `max_age=None` to `matches()`/`observed_resistant()` requiring `sim.ti - ti_dst[sel] <= max_age/dt`.
  - Consider a `retreat_after` guard on `TxDeliveryR` eligibility (record `ti_treatment_end`; exclude agents whose last course ended < N steps ago) — the most direct fix for "continuously retreated."
- **Test.** `devtests/test_diagnostics_tpt.py`: low `base_efficacy` to force failures; assert an agent is not treated on consecutive resolutions without an intervening fresh DST, and `n_treated` plateaus once the eligible pool is exhausted.

### 1.2 — TPT-driven acquisition invisible to the origin decomposition [L2]

- **Gap.** `TPTRx._acquire` mutates `strain_mask` (`tpt.py:80-110`) but increments no counter and exposes no result; `ResistanceStats` (`analyzers.py:44-53`) sums only de-novo, transmitted, and `TxDeliveryR.n_acquired`, so TPT-created resistance is counted in no flux channel.
- **Changes.**
  - `TPTRx._acquire`: snapshot `before = strain_mask[uids].copy()`, count changed masks (`n = int(np.count_nonzero(surv != before))`), accumulate into `self._n_tpt_acquired`; reset it per step; define `ss.Result('n_acquired', …)` and write it in `update_results` (surface on the delivery if the product has no results hook).
  - `ResistanceStats`: add `flux_tptacq`, collect TPT interventions in `init_pre`, populate in `step`, add to `to_df`; optionally assert channels sum to total new resistance.
- **Test.** `devtests/test_acquisition.py`: `p_tpt_acq > 0` with `q_acq`/`p_rand` = 0; assert `flux_tptacq > 0` and the other channels ~0.

---

## Phase 2 — Per-strain counter: data model [S1, S3, S5]

The foundational spec change. Purely additive to `strain_mask`.

### 2.1 — Add the counter state and the "starts at 1" rule [S1]

- Add per-agent, per-strain-id multiplicity. Given `m = 2**n` strains, the simplest Starsim-friendly representation is `m` scalar `ss.IntArr('strain_count_j')` (or a single object holding an `(n_agents, m)` view); avoid a Python-object array for performance. Count is 0 exactly when the corresponding `strain_mask` bit is 0.
- On a **new** infection of a susceptible agent (`set_prognoses`, `tb_resistant.py:149-153`), set the founding strain's count to **1**, regardless of how many copies the source carries.

### 2.2 — Allow identical-strain superinfection; delete the block [S1]

- Remove the identical-strain block in `set_prognoses` (`tb_resistant.py:163-174`: the `already`/`blocked`/`acq = ~already` logic). On re-exposure to a **carried** strain, **increment that strain's count** instead of dropping the event.
- Keep the clock reset: `ti_infected` must still be set on *every* successful exposure, now including same-strain re-exposure (previously only on the `blocked`/`acq` paths at `:171,182`). This preserves the [S1] rule that reinfection with any strain resets the progression clock (consumed later by time-varying progression, Phase 6).
- Remove or repurpose the now-obsolete `new_blocked_superinf` result (`tb_resistant.py:390`) and its per-step accumulation. The spec reframes the old "count blocked events" analyzer as a past request, no longer active; if a superinfection-rate diagnostic is still wanted, replace it with a `new_identical_superinf` counter of count-increment events.

### 2.3 — Reset counters on clearance and successful treatment [S3, S5]

- **Natural clearance/resolution** (`step_transitions`, `tb_resistant.py:213,235`, INFECTION→CLEARED and NON_INFECTIOUS→CLEARED): already zeroes `strain_mask`; additionally **reset all counts to 0** [S3].
- **Successful treatment** of a strain (`TxR`/`TxDeliveryR` cure path, `treatments.py`): when a strain is cured, **reset that strain's count to 0** regardless of starting count (treatment clears all copies) [S5]. This lands with the treatment-coupling work in Phase 3.4.

---

## Phase 3 — Per-strain counter: consumers [S1, S2, S4, S5, S6]

Wire the two real consumers, then verify the coupling constraints (which are largely free under the mask).

### 3.1 — Transmission multinomial weighting [S1]

- The probability a given carried strain is transmitted becomes ∝ **count × fitness** (both are multipliers). Update `Strains.transmit_probs` / its caller (`tb_resistant.py:159`) to multiply each carried strain's fitness by its count.
- The **overall** per-contact transmission probability (`rel_trans = max fitness`, unchanged) must **not** depend on count. Confirm `step_bookkeeping` still uses max-fitness only.
- **Test.** Superinfected source with counts {susceptible: 2, resistant: 1}, no fitness cost → 2/3 vs 1/3 split of which strain is passed; total infectiousness unchanged from count 1/1.

### 3.2 — Progression bottleneck selection [S2]

- When `p_multi < 1` and the bottleneck fires (agent does not retain all strains at →ASYMPTOMATIC), weight the single surviving strain by **count** (× fitness if `prog_select='fitness'`). Update `_select_probs`/`_progress` (`tb_resistant.py:275`).
- Default `p_multi = 1` means no bottleneck, so this is exercised only in the `p_multi < 1` path — implementable now, independent of time-varying progression.
- **Test.** `p_multi < 1`, agent with counts {sus: 2, res: 1}, force a bottleneck → 2/3 vs 1/3 chance the survivor is the susceptible vs resistant strain.

### 3.3 — Random-acquisition coupling + strain selection [S4, L4]

- **Coupling [S4].** Because de-novo acquisition operates on `strain_mask` ids (`_denovo`, `tb_resistant.py:301-319`), all copies of a strain acquire together automatically; count does **not** change acquisition probability. Add a test asserting this rather than new logic.
- **Selection [L4].** Per the Phase 0 decision (Option A), when a drug-acquisition trial hits, pick the susceptible carried strain to mutate at random / ∝ fitness via a shared helper (reuse the `_select_probs` / `choice2d` pattern) instead of lowest-id (`treatments.py:120-129`, `tpt.py:101-108`, and the `_denovo` iteration). Keep "one trial per drug per episode."
- **Test.** `devtests/test_acquisition.py`: 2-drug `['RIF','FQ']`, agent co-carrying strain 0 (pan) and strain 1 (RIF-R); force an FQ hit → resistant target distribution includes strain 3 (RIF+FQ), not always strain 2.

### 3.4 — Treatment coupling + counter reset [S5]

- **Coupling [S5].** Treatment cure and acquisition-on-failure operate on `strain_mask` ids, so identical strains are cured-together / acquire-together for free; count does **not** change efficacy or acquisition probability. Verify and test.
- **Counter reset [S5].** On a successful cure of a strain, reset that strain's count to 0 (Phase 2.3). On partial cure, the agent stays in-state with the surviving strains and their counts intact.
- **Test.** Multi-copy agent, successful cure of the susceptible strain → its count is 0 afterward; surviving resistant strain's count unchanged; efficacy independent of starting count.

### 3.5 — DST unaffected by count [S6]

- DST aggregation reads `strain_mask` + `p_strain_obs`; ensure it ignores count entirely (sensitivity/specificity independent of copies). Likely no code change — add an assertion test.
- **Test.** Two agents identical except strain counts → identical DST profile distributions.

---

## Phase 4 — Decision-gated behavior change

### 4.1 — Latent-treatment divergence from base tbsim [L3]

- **Gap.** `TxDeliveryR._initiate` puts any selected latent agent into a full `TREATMENT` course (`treatments.py:217-250`), whereas base `tbsim.TxDelivery` clears latent agents immediately (`interventions/treatments.py:210-228`). Default rate eligibility spares latent agents; DST/eligibility-routed deliveries do not.
- **Changes (Option B from Phase 0).** In `_initiate`, split `latent = start[tb.latent[start]]`; when `self.treat_latent` is False (default), clear them via the base-tbsim block (→ CLEARED, `rr_reinfection`, wane, `infected=False`, `susceptible=True`) and exclude from the course-based `start`; when True, keep the current course path.
- **Test.** `eligibility` selecting latent agents → under the default, latent agents end CLEARED without entering `TREATMENT` and never appear in `n_acquired`.

---

## Phase 5 — Features, usability, and docs

### 5.1 — Treatment-monitoring eligibility combinator [L5]

- Add `eligibility_all(*callables)` / `eligibility_any(*callables)` (intersect/union of `sim -> uids`) next to `treatment_monitoring_eligibility` (`treatments.py:292-320`).
- Add `treatment_monitoring_eligibility(..., require=None)` sugar AND-ing in another callable (e.g. `require=dst.matches(RIF=True)`).
- Add a `will_fail(tx_name)` factory selecting on-treatment agents with `pending_surv != 0` (documented oracle for scenario construction).
- **Test.** Compose monitoring + `dst.matches(...)`; assert only agents meeting both switch regimens.

### 5.2 — Strain-reduced / agnostic convenience mode [L7]

- Add a `TBResistant.agnostic(...)` factory (or document the recipe) setting the "effectively single-strain" defaults (`init_strains=[1,0]`, `p_rand=None`, no resistant transmission). Do **not** attempt a true `m=1` space — the bitmask assumes `m = 2**n`.
- **Test.** `agnostic()`-configured `TBResistant` reproduces `tbsim.TB` results within CRN noise on a matched scenario.

### 5.3 — Drug-name validation helper [L8]

- Add `Strains.validate_drugs(names, where='')` raising on any name not in `drug_idx`; call it from `TxR`, `TPTRx`, `DST.__init__` for `regimen_drugs` and the keys of `q_acq`/`p_tpt_acq`/`sens`/`spec` (currently silent `.get()`).
- **Test.** `TxR(regimen_drugs=['RIFF'])` raises a clear error naming the bad drug.

### 5.4 — User-guide block 8 clarity [L9] (docs only)

- Fix `tbsim-resistance-user-manual.md` §8 (lines 437-446): the RIF-resistant second-line delivery uses `regimen_drugs=['RIF']` with no `resist_penalty`, so it cures RIF-resistant TB at full efficacy. Rename to a distinct drug (e.g. `['BDQ']` in a `drugs=['RIF','BDQ']` example) or add explanation + `resist_penalty`. Do when the guide merges toward `main`.

---

## Phase 6 — Deferred (needs modeling-team spec input)

### 6.1 — TPT `resist_penalty` for partial efficacy [L6]

- TPT sterilization is all-or-nothing (`_covered_mask`, `tpt.py:67-71,112-133`). Add `resist_penalty` to `TPTRx`, compute per-strain sterilization probability (mirror `TxR.eff_by_id`), replace the mask-clear with per-strain Bernoulli clearance (+ CRN streams), and penalty-weight `apply_protection`. Explicitly deferred by TR; schedule with multi-drug TPT work.

### 6.2 — Time-varying progression hazard + remaining Minerva gaps [L10]

- Make INFECTION→(NON_INFECTIOUS/ASYMPTOMATIC) rates in `step_transitions` (`tb_resistant.py:204-208`) functions of `sim.ti - ti_infected[u]`. The clock-reset hook is ready (and Phase 2.2 extends it to same-strain re-exposure); the missing piece is the hazard *shape* and where it lives (`tbsim.TB` vs `TBResistant`).
- **Do not implement blind** — request the progression-hazard spec, then validate against UAT-07 (new infection resets the clock; number of prior infecting strains does not change the temporal pattern).
- Remaining §13 backlog (adherence distribution, DST-indeterminate outcome, failure-vs-new-case classifier, LAI_TPT burden table, LTFU) is separate and lower priority.

---

## Consolidated sequence

| Order | Phase | Items | Gate |
|-------|-------|-------|------|
| 1 | 0 | Lock decisions | — |
| 2 | 1 | L1 DST retreatment, L2 TPT flux | needs L1 decision |
| 3 | 2 | S1 counter model, remove block, S3/S5 counter resets | — |
| 4 | 3 | S1 transmission, S2 progression, S4/L4 acquisition, S5 treatment, S6 DST | depends on Phase 2 |
| 5 | 4 | L3 latent treatment | needs L3 decision |
| 6 | 5 | L5 combinator, L7 agnostic mode, L8 validation, L9 docs | — |
| 7 | 6 | L6 TPT penalty, L10 time-varying progression | needs spec input |

Run tests after each phase (`cd tests && bash run_tests`, plus the per-item `devtests`). Phases 1 and 2 are independent and may proceed in parallel; Phase 3 depends on Phase 2; Phases 4–5 depend on Phase 3 only where they touch acquisition/treatment (L4 is folded into 3.3).
