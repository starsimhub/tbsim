# Resistance Loop — planned fixes

Implementation plans for the gaps identified in [TBsim_Resistance_Loop.md](TBsim_Resistance_Loop.md), i.e. the reviewer comments that are **not** currently reflected in the code on `resistance-updates`. Each item states the gap, the desired behavior, the concrete code changes, and a test. Items are ordered high→low priority. Struck-through comments in the loop doc and questions already answered by the code (KHG "diagnostics", TR item 4) require no change and are omitted here.

Cross-references use `file:line` from the review; verify against the working tree before editing since line numbers drift.

---

## P1 — Bugs (correctness)

### 1. DST result never expires → continuous retreatment (loop doc, TR #1)

**Gap.** `DSTDelivery` stores `dst_profile` / `dst_tested` permanently and never clears them; `ti_dst` is recorded (`dst.py:99-103,161`) but never used to expire a result. `matches()` / `observed_resistant()` keep an agent eligible as long as the stored profile matches, and `matches(exclude_on_treatment=True)` excludes only agents *currently* in `TREATMENT`. So an agent that fails a course and returns to ASY/SYM is immediately re-selected and re-treated forever.

**Desired behavior.** A DST result should have a finite useful life: an agent should not be perpetually re-treated off a single stale test. Two independent levers, ideally both:

1. **Result expiry.** Treat a stored profile as valid only for a configurable window after `ti_dst`; once expired, `dst_tested` reverts to `False` so the agent must be re-tested before it is eligible again.
2. **Re-test gating in eligibility.** `matches()` / `observed_resistant()` should optionally require the result to be *fresh* (within the window) and/or exclude agents treated since the test.

**Changes.**
- `DSTDelivery.__init__`: add `result_validity=None` (an `ss.dur`, default `None` = never expires, preserving current behavior). Store it.
- `DSTDelivery.step`: before administering, expire stale results — for agents where `self.ti - self.ti_dst >= result_validity/dt`, set `dst_tested=False`, `dst_profile=0`, `ti_dst=nan`. This naturally re-tests them if still eligible under the default eligibility (`~dst_tested`).
- Alternatively (or additionally) gate in `matches()`/`observed_resistant()`: add `max_age=None` param; when set, require `sim.ti - dst.ti_dst[sel] <= max_age/dt`.
- Consider a `retreat_after` guard on `TxDeliveryR` eligibility so a just-failed agent isn't re-treated on the very next step (record `ti_treatment_end` history; exclude agents whose last course ended < N steps ago). This is the more direct fix for "continuously retreated" and may be preferable to result-expiry alone.

**Decision needed.** Whether the intended semantics are "one test → one treatment attempt" (needs re-test before re-treatment) or "test stays valid for N months." Confirm with TR / the (missing) `issue-dst-retreatment.md` intent.

**Test.** `devtests/test_diagnostics_tpt.py`: DST-routed treatment with `base_efficacy` low enough to guarantee failures; assert an individual agent is not treated on consecutive resolutions without an intervening fresh DST, and that `n_treated` stops growing once the eligible pool is exhausted.

---

### 2. TPT-driven acquisition invisible to the origin decomposition (loop doc, TR #2)

**Gap.** `TPTRx._acquire` mutates `tb.strain_mask` (`tpt.py:80-110`) but increments no counter and exposes no result. `ResistanceStats` sums only de-novo, transmitted, and `TxDeliveryR.n_acquired` (`analyzers.py:44-53`), so TPT-created resistance is counted in none of the flux channels — the three-way decomposition undercounts total new resistance and misattributes origins whenever `p_tpt_acq` is active.

**Desired behavior.** TPT-acquired resistance is a fourth origin channel (or folded into "treatment-acquired" with a clear definition). The decomposition must sum to total new resistance.

**Changes.**
- `TPTRx._acquire`: count agents whose mask actually changed, e.g. `n = int(np.count_nonzero(surv != before))` (snapshot `before = tb.strain_mask[uids].copy()` before the loop). Accumulate into a per-step attribute `self._n_tpt_acquired`.
- `TPTRx`: reset `self._n_tpt_acquired = 0` at the start of each step (override `step`/`apply` entry point, mirroring `TBResistant.step`), define an `ss.Result('n_acquired', …)` in `init_results`, and write it in `update_results`. (Verify `TPTTx`/its delivery already has a results hook; if the product isn't an `ss.Module` with results, surface the counter on the delivery instead.)
- `ResistanceStats`: add a `flux_tptacq` result and, in `init_pre`, collect TPT interventions (`isinstance(iv, TPTRx)` or the delivery wrapping it). In `step`, set `res.flux_tptacq[ti] = sum(...)`. Add it to `to_df`. Optionally add a `flux_total`/assertion that channels sum to `frac_resist`-implied new cases.

**Test.** `devtests/test_acquisition.py`: run with `p_tpt_acq` > 0 and `q_acq`/`p_rand` = 0; assert `flux_tptacq` > 0 and that de-novo/txacq/transmitted are ~0, i.e. the new resistance is attributed to TPT and not lost.

---

## P2 — Behavior changes needing a decision

### 3. Latent-treatment divergence from base tbsim (loop doc, TR #3)

**Gap.** Base `tbsim.TxDelivery.step_start_treatment` clears latent (`TBS.INFECTION`) agents **immediately** to CLEARED with no course (`interventions/treatments.py:210-228`). `TxDeliveryR._initiate` puts any selected agent — latent included, if a custom `eligibility` selects them — into a full `TREATMENT` course that can fail/acquire resistance (`treatments.py:217-250`). Default rate-based eligibility spares latent agents, but DST/eligibility-routed deliveries do not.

**Desired behavior (unrecorded — `issue-latent-treatment-divergence.md` absent).** Options:
- **(A) Match base tbsim:** short-circuit latent agents in `_initiate` — clear them immediately (→ CLEARED, reinfection protection), no course, no acquisition. Simplest, consistent with single-strain tbsim; but loses the ability to model latent-TB treatment failure/resistance in the resistant model.
- **(B) Keep the course but gate it:** treat latent agents through a course only if explicitly opted in (e.g. a `treat_latent=False` flag on `TxDeliveryR`); otherwise clear immediately. Preserves both behaviors.
- **(C) Document the divergence as intentional** if resistant-model latent treatment (e.g. TPT-adjacent regimens) is a wanted capability.

**Changes (if A or B).** In `TxDeliveryR._initiate`, after computing `start`, split `latent = start[tb.latent[start]]`; clear them via the same block as base tbsim (`state=CLEARED`, `rr_reinfection`, wane, `infected=False`, `susceptible=True`) and exclude from the course-based `start`. Under (B), gate on `self.treat_latent`.

**Recommendation.** Pursue (B) — it defaults to base-tbsim-consistent behavior while leaving the richer path available. Confirm with TR before implementing.

**Test.** Feed `eligibility` selecting latent agents; under the chosen default, assert latent agents end in CLEARED without entering `TREATMENT` and without appearing in `n_acquired`.

---

### 4. Acquisition-on-failure strain selection is lowest-id, capped at one strain (loop doc, KG notes)

**Gap.** `TxR.acquire` (`treatments.py:120-129`) and `TPTRx._acquire` (`tpt.py:101-108`) draw one trial per agent per regimen drug (intended — "once per episode"), but on a hit they flip only the **first (lowest-id)** carried susceptible strain (`done |= has_j`). The choice of lowest-id is arbitrary and pan-leaning, and never lands on a strain already carrying other resistances. `TBResistant._denovo` instead iterates every carried strain. Only bites with ≥2 drugs + an agent superinfected with multiple strains susceptible to the drug — unexercised by current devtests.

**Desired behavior (decision needed).** The "one trial per drug per episode" cardinality should stay (spec). The open question is *which* susceptible carried strain acquires resistance on a hit:
- **(A) Random / fitness-weighted** among the agent's carried drug-susceptible strains (parallels `TBResistant._select_probs`). Most defensible; removes the pan-leaning bias.
- **(B) Keep lowest-id** but document it explicitly as a simplifying assumption.

**Changes (if A).** Factor a small helper (shared by `TxR.acquire` and `TPTRx._acquire`) that, for the hit agents and a given drug, samples one carried susceptible strain per agent (uniform or ∝ fitness) via a CRN `choice2d`, then replaces it with its `| bit` counterpart. Reuse the `_select_probs` pattern from `tb_resistant.py:281-286`.

**Test.** `devtests/test_acquisition.py`: 2-drug (`['RIF','FQ']`) setup, seed agents co-carrying strain 0 (pan) and strain 1 (RIF-R, FQ-susceptible), force an FQ-acquisition hit; assert the resistant target distribution is not always strain 2 (pan+FQ) but includes strain 3 (RIF+FQ), matching the chosen policy.

---

## P3 — Feature requests

### 5. Treatment-monitoring eligibility contingent on DST or failure (loop doc, TR #5)

**Gap.** `treatment_monitoring_eligibility` is purely time-based (`treatments.py:292-320`); there is no first-class way to make it contingent on an observed DST profile or on predicted failure, and no combinator to AND two eligibility callables. Manual composition works today (both callables return `ss.uids`; intersect them) but is undocumented.

**Changes.**
- Add a small combinator helper, e.g. `eligibility_all(*callables)` / `eligibility_any(*callables)` returning a `sim -> uids` that intersects/unions results. Put it next to `treatment_monitoring_eligibility`.
- Optionally extend `treatment_monitoring_eligibility(..., require=None)` where `require` is another callable AND-ed in (sugar over the combinator), so `require=dst.matches(RIF=True)` gives "on-treatment ≥ N steps AND observed RIF-resistant."
- Failure-contingent monitoring: since the course outcome is frozen at initiation in `TxDeliveryR.pending_surv`, add an eligibility factory `will_fail(tx_name)` selecting on-treatment agents whose `pending_surv != 0`. Document it as an oracle (uses pre-rolled outcome) for scenario construction.

**Test.** Compose monitoring + `dst.matches(...)`; assert only agents meeting both switch regimens, unlike the current "everyone after N steps."

---

### 6. TPT `resist_penalty` for partial efficacy (loop doc, TR #6, future)

**Gap.** TPT sterilization is all-or-nothing: `_covered_mask` clears only strains susceptible to *every* regimen drug, so any regimen-resistant strain has 0% clearance (`tpt.py:67-71,112-133`); progression-protection uses the same binary coverage weight. No `resist_penalty` analog exists as it does in `TxR`.

**Desired behavior.** A per-drug multiplicative penalty giving a strain resistant to some (but not all) regimen drugs a reduced-but-nonzero clearance/protection — needed for 2-drug TPT regimens partially efficacious against single-drug-resistant strains.

**Changes.**
- `TPTRx.__init__`: add `resist_penalty=None`; compute a per-strain-id sterilization probability `p_ster_by_id = p_sterilize × ∏ penalty over regimen drugs the strain resists` (mirror `TxR.eff_by_id`, `treatments.py:64-70`), instead of the binary `_covered_mask`.
- `_apply_sterilization`: replace the mask-clear with a per-strain Bernoulli clearance using `p_ster_by_id` (add per-strain CRN streams like `TxR._cure_rngs`). An agent goes to CLEARED once all strains are cleared.
- `apply_protection`: replace the binary coverage weight `w` with a penalty-weighted effectiveness per carried strain so partial protection is possible.

**Priority.** Explicitly deferred by TR ("not as important for the current project"). Schedule with the multi-drug TPT work.

**Test.** 2-drug TPT regimen vs. a strain resistant to 1 of the 2 drugs; assert clearance probability is strictly between 0 and the pan-susceptible value.

---

## P4 — Usability / validation / docs

### 7. Strain-reduced convenience mode (loop doc, KHG notes)

**Gap.** No one-line switch between agnostic `tbsim.TB` and strain-aware `TBResistant`, and no genuine single-strain mode (`Strains` requires ≥1 drug → `m ≥ 2`). Complicates LAI-style strain-aware-vs-agnostic comparisons.

**Changes (lightweight).** Document the "effectively single-strain" recipe (`init_strains=[1,0]`, `p_rand=None`, no resistant transmission) in the user manual, and/or add a `TBResistant` classmethod/factory `TBResistant.agnostic(...)` that sets those defaults so a comparison run needs no manual parameterization. Do **not** attempt a true `m=1` strain space; the bitmask machinery assumes `m = 2**n`.

**Test.** Assert an `agnostic()`-configured `TBResistant` reproduces `tbsim.TB` results (within CRN noise) on a matched scenario.

---

### 8. Drug-name validation helper (loop doc, KHG struck item — partial)

**Gap.** `resist_penalty` keys resolve through `strains.drug_idx` (raise on typo) but `regimen_drugs`, `q_acq`/`p_tpt_acq`, and `DST.sens`/`spec` use silent `.get()`, so a mis-typed drug name is silently ignored. No single place validates that all intervention drug names ⊆ `strains.drugs`.

**Changes.** Add a `Strains.validate_drugs(names, where='')` helper that raises on any name not in `drug_idx`, and call it from `TxR`, `TPTRx`, and `DST.__init__` for `regimen_drugs` and the keys of `q_acq`/`p_tpt_acq`/`sens`/`spec`. Low severity; improves fail-fast ergonomics.

**Test.** Constructing `TxR(regimen_drugs=['RIFF'])` (typo) raises a clear error naming the bad drug.

---

### 9. User-guide block 8 clarity (loop doc, TR #7 — docs only)

**Gap.** `tbsim-resistance-user-manual.md` §8 (lines 437-446) gives the RIF-resistant second-line delivery `regimen_drugs=['RIF']` with no `resist_penalty`, so it cures RIF-resistant TB at full `base_efficacy=0.8` — reads as self-contradictory. Not a code bug.

**Changes (docs).** Rename the second-line regimen to a distinct drug (e.g. `regimen_drugs=['BDQ']` in a 2-drug `drugs=['RIF','BDQ']` example), or keep RIF but add a sentence explaining that `regimen_drugs` names the *drugs the regimen acts on* and that a realistic second-line would use a different drug (add `resist_penalty` if RIF is retained). Do when/if the guide is merged toward `main`.

---

### 10. Time-varying progression hazard + remaining Minerva gaps (loop doc, TR #8)

**Gap.** Progression is constant-hazard state transitions with no dependence on time-since-infection. `ti_infected` is reset on successful/blocked exposures (hook present, `tb_resistant.py:171,182`) but nothing consumes it; the code comment at `tb_resistant.py:166-168` calls the reset "a no-op until time-varying progression exists." UAT-07 is NOT IMPLEMENTED (`tbsim-resistance-uat.md:268-288`). Other §13 limitations (adherence distribution, DST-indeterminate, failure-vs-new-case classifier, LAI_TPT burden table, count-based carriage, LTFU) are acknowledged and lower priority.

**Flagged priority: time-varying progression.** TR notes "we owe software additional info on" this — so **do not implement blind**. Scope pending spec input:
- The INFECTION→(NON_INFECTIOUS/ASYMPTOMATIC) rates in `step_transitions` (`tb_resistant.py:204-208`) would become functions of `sim.ti - ti_infected[u]` (e.g. elevated early-progression risk decaying to a lower reactivation rate), applied via `rr_activation` or a rate-lookup rather than constant `p.inf_non`/`p.inf_asy`.
- With the clock already reset on every exposure, the hook is ready; the missing piece is the hazard *shape* and whether it lives in base `tbsim.TB` or `TBResistant`.

**Action.** Request the progression-hazard specification from the modeling team, then implement + validate against UAT-07's acceptance criteria (new infection resets the clock; number of prior infecting strains does not change the temporal pattern). Treat the other §13 items as a separate backlog.

---

## Suggested sequencing

1. **Now (bugs):** items 1 (DST retreatment) and 2 (TPT flux) — both correctness, both have clear fixes.
2. **Next (decisions):** items 3 (latent divergence) and 4 (acquisition strain selection) — confirm desired behavior with TR/KG first, then implement.
3. **Then (features/usability):** items 5 (monitoring combinator), 7 (agnostic mode), 8 (drug validation), 9 (docs).
4. **Deferred (needs spec / future project):** items 6 (TPT resist_penalty) and 10 (time-varying progression + remaining Minerva gaps).
