# Update v2 plan

This is the implementation plan for the six items raised in [update-v2-notes.md](update-v2-notes.md). For each item it records the current behavior (with code references), the desired behavior, the proposed change, and any open decisions. Nothing here is implemented yet.

Ordering / dependency note: items **2** and **3** share one implementation (per-strain count transfer on acquisition) and touch the same three call sites as item **1 (TR)**, so they should be done together. Item 1 (Reinfection) and item 4 (DST) are independent. Item 5 (latent treatment) is independent but touches `TxDeliveryR._initiate`.

---

## 1. Reinfection of INFECTED agents in base TB

**Current behavior.** In base `TB`, `step_bookkeeping` sets `susceptible = st.isin((TBS.SUSCEPTIBLE, TBS.CLEARED))` ([tb.py:462](../../tb.py)), so latent (`INFECTION`) agents are never eligible for reinfection and never get their `ti_infected` clock reset. Base `TB.pars` has `rr_reinfection_rec` / `_treat` / `_cleared` but **no** `rr_reinfection_inf` (or `_non`) ([tb.py:152-154](../../tb.py)). `TBResistant` already added this: its `step_bookkeeping` includes `INFECTION`/`NON_INFECTIOUS` in the susceptible set with per-state `rel_sus = rr_reinfection_inf` / `rr_reinfection_non` ([tb_resistant.py:443,470-471](../tb_resistant.py)), and its `set_prognoses` resets `ti_infected` for reinfected agents while keeping the current state for superinfection.

**Desired behavior.** Mirror `TBResistant` in base `TB`: latent agents are eligible for reinfection, controlled by a new `rr_reinfection_inf` (σ_L) parameter that defaults to `rr_reinfection_rec`; a reinfected `INFECTION` agent gets `ti_infected` reset to the current step (raising its progression hazard under front-loading), with no strain/count tracking. The notes call out `INFECTION` specifically; `TBResistant` also handles `NON_INFECTIOUS` via `rr_reinfection_non`.

**Proposed change.**

1. Add `rr_reinfection_inf=None` (and, per the decision below, `rr_reinfection_non=None`) to base `TB.define_pars` ([tb.py:147](../../tb.py)), documented as σ_L / σ_N. After `update_pars`, apply the same defaulting `TBResistant` uses: `rr_reinfection_inf ← rr_reinfection_rec` when `None`, `rr_reinfection_non ← rr_reinfection_inf` when `None`.
2. In `TB.step_bookkeeping` ([tb.py:462,480-487](../../tb.py)): add `INFECTION` (and `NON_INFECTIOUS`) to the `susceptible` mask, and set `rel_sus` per state — `rel_sus[INFECTION] = rr_reinfection_inf`, `rel_sus[NON_INFECTIOUS] = rr_reinfection_non` — instead of the flat `rel_sus[:] = 1`. Keep the existing CLEARED waning branch.
3. `set_prognoses` resets `ti_infected` for all targeted `uids` ([tb.py:274](../../tb.py)) but unconditionally does `state[uids] = TBS.INFECTION` ([tb.py:275](../../tb.py)). Per **D1b** (clock-reset-only), reinfection must **not** change the state of an already-infected agent — for a latent agent `→ INFECTION` is a harmless no-op, but a reinfected `NON_INFECTIOUS` agent would be wrongly knocked back to `INFECTION`. So guard the state write to fire only for agents entering from a non-infected state (`SUSCEPTIBLE`/`CLEARED`), mirroring `TBResistant.set_prognoses`'s `was_uninfected` handling ([tb_resistant.py:270-272](../tb_resistant.py)); everyone still gets `ti_infected` reset. This is the one non-trivial edit on the base `set_prognoses`.

**Coordination with `TBResistant`.** `TBResistant.__init__` re-declares `rr_reinfection_inf`/`_non` in its own `define_pars` and does the defaulting ([tb_resistant.py:68-94](../tb_resistant.py)). Once these live in base `TB`, remove them from `TBResistant.define_pars` (a duplicate `define_pars` key will collide) and keep only the shared defaulting in the base class. `TBResistant` fully overrides `step_bookkeeping`/`set_prognoses`, so the base changes will not otherwise perturb it.

**Decisions (resolved).**
- **D1a — resolved: include both.** Add both `rr_reinfection_inf` (σ_L) and `rr_reinfection_non` (σ_N) to base `TB`, fully mirroring `TBResistant`'s reinfection surface.
- **D1b — resolved: clock reset only.** Reinfection of a latent or non-infectious agent resets `ti_infected` and does **not** change the agent's state (implemented via the `set_prognoses` state-write guard in step 3 above).

**Tests.** Add a base-TB test that a latent agent exposed to force of infection has `ti_infected` reset and elevated progression; verify default `rr_reinfection_inf == rr_reinfection_rec`; confirm `TBResistant` still reproduces the ODE null (`rr_reinfection_inf=1.0`) and `test_resistance.py` passes.

---

## 2. Preserve strain count through resistance acquisition

**Current behavior.** All three acquisition mechanisms edit `strain_mask` and then call `_sync_counts_to_mask(uids, before)` ([tb_resistant.py:167-184](../tb_resistant.py)), which sets any newly-set strain bit to count **1** and any cleared bit to count **0**. So when a strain carried at count 2 mutates (e.g. `A`→`B` on treatment failure), the source bit is cleared (count → 0, correct) but the emergent resistant strain gets count reset to **1** rather than inheriting the source's count of 2. Call sites: de-novo `_denovo` ([tb_resistant.py:422](../tb_resistant.py)), treatment `_resolve`/`acquire` ([treatments.py:376,152](../treatments.py) via `mutate_one_susceptible` in [strains.py:194](../strains.py)), and TPT `_acquire` ([tpt.py:127](../tpt.py)).

**Desired behavior.** The now-resistant strain keeps the count of the strain it mutated from (count stays 2 in the example), for all three mechanisms.

**Proposed change.** Replace the "emergent bit → count 1" convention with an explicit **count transfer** at each acquisition site: when source strain `j` (count `c`) mutates to target `k`, do `count[k] += c` and, for replacement, `count[j] = 0`. This must be threaded through the operators that currently rely on the crude sync:
- Add a helper on `TBResistant`, e.g. `_transfer_counts(uids, src_ids, dst_ids, replace=True)`, that moves per-agent source counts onto the target strain (accumulating; see item 3). It must handle the vectorized "different `src`/`dst` per agent" shape the operators produce.
- `_denovo` ([tb_resistant.py:389-423](../tb_resistant.py)): it already computes `targets = j | acquired[got]` per source strain `j`; transfer `count[j]` onto `count[target]` instead of relying on `_sync_counts_to_mask`.
- Treatment `acquire` ([treatments.py:130-154](../treatments.py)) and TPT `_acquire` ([tpt.py:106-129](../tpt.py)) move to the shared per-strain acquisition loop (see TR item 1 / D-TR1), which — like `_denovo` — knows each source strain `j` and its target `k` directly, so it can call `_transfer_counts` per source. This removes the dependence on `mutate_one_susceptible` (which is deleted) and its blind `_sync_counts_to_mask` follow-up. The bottleneck path (`_bottleneck`) still legitimately uses `_sync_counts_to_mask` (survivor keeps its count, dropped strains → 0) and should be left as-is.

**Decision — D2 (mixed mode) — resolved.** De-novo defaults to `prog_resist_mode='mixed'`, which *keeps* the source strain bit set. Points 2/3 describe replacement-style semantics ("removes that strain … (correct)"). Resolved behavior: in **replacement** the full source count transfers to the target (source → 0); in **mixed** the source **keeps** its count and the target additionally receives a copy of the source count (so total per-agent load rises when a mixed superinfection lineage emerges). Tx-acquired and TPT-acquired are always replacement, so they are unambiguous. **This asymmetry must be documented clearly** where it lives: (a) a precise docstring on the `_transfer_counts` helper stating the replacement-vs-mixed count rule, (b) an inline comment at the `_denovo` call site noting mixed keeps the source count while the target gets a copy, and (c) a line in the resistance tech-spec / D-COUNTER note in implementation-decisions.md so the total-load consequence is on record.

**Tests.** Unit test each mechanism: seed an agent with a single strain at count 2, force acquisition, assert the resistant strain's count is 2 (not 1) and the source is 0 (replacement).

---

## 3. Accumulate counts when the resistant target already exists (and across multiple mutating sources)

**Current behavior.** An extension of item 2. If the agent *already* carries the resistant target strain (count ≥ 1), `_sync_counts_to_mask` leaves its count unchanged because the target bit was already set (not "newly set") — equivalent to just clearing the mutating source. Likewise, if several susceptible strains mutate to the same target in one round, only the final bit-state is honored, not the sum. Desired: the target's count increases by the sum of the mutating sources' counts (e.g. target had 2, two count-1 sources mutate in → target becomes 4).

**Desired behavior.** Target count `= existing target count + Σ (count of each source strain that mutates into it)`, for all three mechanisms.

**Proposed change.** Implement item 2's `_transfer_counts` as **additive accumulation** onto the target (`count[k] += c`, never overwrite), and make it correctly sum when multiple sources map to the same target within one step. In `_denovo`, the per-source-strain loop already iterates `j` separately, so accumulating onto the target across iterations gives the sum naturally — but the current single trailing `_sync_counts_to_mask` call must be replaced by in-loop transfers so earlier iterations' target counts are not clobbered. For treatment/TPT, the move to the per-strain acquisition loop (TR item 1 / D-TR1) is what lets multiple sources mutate into the same target in one round, so the accumulation helper must be in place before/with that change.

**Interaction with TR-notes item 1 (multiple strains acquiring resistance).** Note the update-v2-notes "TR Notes" item 1 asks that treatment-acquired resistance allow *multiple* strains to mutate per round (like de-novo), rather than the current single `mutate_one_susceptible` pick. That change (tracked below) is what makes the "two susceptible strains acquire resistance → count 4" scenario reachable on the treatment path; the count-accumulation logic here must be built to handle it.

**Tests.** Seed an agent carrying resistant target at count 2 plus two susceptible strains at count 1 each; force both to acquire; assert target count == 4 and both sources == 0. Repeat per mechanism.

---

## TR item 1. Allow multiple strains to acquire treatment-acquired resistance

**Current behavior.** `TxR.acquire` runs one trial per agent per regimen drug and, on a hit, mutates exactly **one** carried drug-susceptible strain chosen via `mutate_one_susceptible` ([treatments.py:144-153](../treatments.py), [strains.py:155-195](../strains.py)). So at most one strain acquires resistance to a given drug per treatment round. (This is already an improvement over the old "first/most-susceptible strain" bias.)

**Desired behavior.** Allow multiple carried strains to acquire resistance independently in one round, mirroring de-novo's per-strain independent mechanism (`_denovo`, [tb_resistant.py:389-423](../tb_resistant.py)).

**Proposed change.** Refactor `TxR.acquire` to loop over carried strains (as `_denovo` does), giving each carried drug-susceptible strain an independent per-drug acquisition trial (`q_acq[drug] × acq_state_rr`), OR-ing acquired drug bits per source strain and mutating each hit strain to its resistant counterpart (replacement). Reuse the per-drug CRN streams already allocated in `TxR` (`_acq_rngs`). This retires the single-pick `mutate_one_susceptible` / `acq_select` path for treatment. Since backwards compatibility is not a concern on this dev branch, **remove `acq_select` from `TxR`** rather than keeping a no-op shim (it only chose *which single* strain mutates, which no longer applies). Couple this with the count-transfer work in items 2/3 so each mutated strain carries its source count.

**Decision — D-TR1 — resolved (backcompat n/a).** Drop `acq_select` from `TxR` and switch `TxR.acquire` to the per-strain loop. **TPT symmetry — resolved: yes.** `TPTRx._acquire` ([tpt.py:106-129](../tpt.py)) likewise moves from the single-pick `mutate_one_susceptible` to the same per-strain independent mechanism, so all three acquisition mechanisms (de-novo, treatment, TPT) share one model: each carried drug-susceptible strain gets an independent per-drug trial and mutates to its resistant counterpart. Drop `acq_select` from `TPTRx` too. Consequence: with de-novo, treatment, and TPT all per-strain, `Strains.mutate_one_susceptible` ([strains.py:155-195](../strains.py)) has no remaining callers and should be **removed** (dead code), along with the per-drug `_acq_select` choice streams in `TxR`/`TPTRx` that only fed it. Factor the shared per-strain acquisition loop into one helper (see Cross-cutting) so `_denovo`, `TxR.acquire`, and `TPTRx._acquire` don't reimplement it three times.

**Tests.** Seed an agent with two distinct drug-susceptible strains; set `q_acq` high; assert both can become resistant in one round (previously impossible).

---

## TR item 4. Independent DST errors across drugs within a strain

**Current behavior.** `DST.administer` draws **one** call random number per strain (`_call_rngs[j]`) and reuses it across all drugs for that strain ([dst.py:52-53,80-86](../dst.py)). The per-drug marginals (sensitivity/specificity) are exact, but the errors are perfectly correlated across drugs within a strain: a genuinely RIF-resistant/BDQ-susceptible strain reads correctly or gets "swapped," but essentially never over-called resistant to both. This only matters when treatment is routed on a joint multi-drug DST profile under imperfect `sens`/`spec`.

**Desired behavior.** Per-drug DST errors are independent within a strain, so a multi-drug DST product behaves like independent per-drug tests. (Workaround today: model a separate single-drug `DST` product per drug.)

**Proposed change.** Give the call draw an independent stream per **(strain, drug)** rather than per strain. Concretely, replace `self._call_rngs = [ss.random(name=f'dst_call_{j}') …]` with a 2D set of streams `dst_call_{j}_{drug}` (or an equivalent `(m, n)` layout), and in the per-drug loop ([dst.py:81-86](../dst.py)) draw from the `(j, di)` stream instead of the shared per-strain `call`. Keep `_obs_rngs` per strain (the culture-bottleneck observation is genuinely a single per-strain event). Verify CRN reproducibility and that the multi-strain detection boost (independence *across* strains) is unaffected.

**Tests.** Set `sens`/`spec` < 1 for a two-drug strain, run many agents, assert the joint distribution of (drug0 call, drug1 call) matches the product of the marginals (independence), and that each marginal still matches `sens`/`spec`. Confirm `test_resistance.py` DST tests still pass.

---

## TR item 5. Treatment of latent (`INFECTION`) agents

**Current behavior.** `TxDeliveryR` has a binary `treat_latent` flag ([treatments.py:184-189,318-329](../treatments.py)): `False` (default) clears every selected latent agent to `CLEARED` immediately (all strains removed, no course, no acquisition); `True` runs them through a full course that can fail and select for resistance exactly like active disease. There is no mode that clears latent agents of treatment-susceptible strains while leaving prior resistance intact.

**Desired behavior.** Latent agents treated for active disease should have all **treatment-susceptible** strains cleared, **prior resistant strains persist**, and **no new resistance is acquired**. Per feedback this becomes the new **`treat_latent=False`** behavior (replacing the current clear-everything-to-CLEARED semantics); backwards compatibility is not a concern on this unmerged dev branch.

**Proposed change.** Rewrite the `treat_latent=False` branch in `_initiate` ([treatments.py:318-329](../treatments.py)) so selected latent agents go through a strain-aware sterilization step that:

- clears strains susceptible to the regimen drugs (a strain is cleared iff susceptible to *every* regimen drug — the same "covered" definition as `TPTRx._apply_sterilization`, [tpt.py:131-153](../tpt.py); factor that per-strain sterilization into a shared `TBResistant` helper and call it from both),
- with certain clearance of covered strains (the notes' "all INFECTED agents are cleared unless they have prior resistance" implies deterministic clearance, not `base_efficacy`),
- leaves regimen-resistant strains in place (agent stays latent/infected with the surviving strains; goes to `CLEARED` only if no strain remains, with post-clearance reinfection protection and counts reset),
- performs **no** acquisition and **no** course (never enters `TREATMENT`).

For a pan-susceptible latent agent (the only case in single-strain / `agnostic` usage) the single strain is covered and cleared, so the agent still goes to `CLEARED` — i.e. **behavior is unchanged for non-resistant/single-strain models**; the only difference is that latent agents carrying a regimen-resistant strain now retain it instead of being fully sterilized. Keep `treat_latent=True` (full failable course) as the alternative option. Ensure `strain_mask`/counts and `infected`/`susceptible` flags are updated consistently (reuse the existing latent branch and `_reset_counts`).

**Response to feedback (issues with making this the `treat_latent=False` behavior).** No blocking issues. Concretely:
- *Single-strain / base-parity is preserved* — a pan-susceptible latent agent is still cleared to `CLEARED`, so runs without resistant strains behave exactly as before. The default rate-based eligibility never selects latent agents anyway (only ASYMPTOMATIC/SYMPTOMATIC), so this only affects custom/DST-routed eligibilities — the same audience the flag already served.
- *Minor accounting decision:* partially-sterilized latent agents run no course, so (consistent with today) they should **not** be counted in `n_treated`. Flag: if you want a separate tally of "latent agents acted on," add a distinct result rather than overloading `n_treated`.
- *"Covered" definition:* uses susceptibility to every regimen drug (TPT's rule), so a strain resistant to *any* regimen drug persists — matching "unless they have prior resistance."
- *Keeping `treat_latent=True`:* still useful for scenarios that deliberately model latent agents on a failable course; recommend keeping it. Say the word if you'd rather drop it entirely and make latent handling unconditional.

**Tests.** Seed a latent agent carrying one pan-susceptible + one regimen-resistant strain; treat with `treat_latent=False`; assert the susceptible strain is cleared, the resistant strain persists (agent stays latent), no acquisition occurred, and the agent did not enter `TREATMENT`. Verify a fully-susceptible latent agent goes to `CLEARED`. Verify a single-strain run is unchanged vs the pre-rewrite `treat_latent=False`.

---

## Cross-cutting

- **Shared helpers.** Items 2/3/TR-1 all need (a) one **per-strain acquisition loop** — each carried drug-susceptible strain gets an independent per-drug trial and mutates to its resistant counterpart — shared by `_denovo`, `TxR.acquire`, and `TPTRx._acquire` (each supplies its own per-drug probabilities, state RR, and CRN streams; mode is replacement except mixed de-novo), and (b) a robust per-agent additive **count-transfer** primitive (`_transfer_counts`) that loop calls per source→target. Building these retires `Strains.mutate_one_susceptible` and the `_acq_select` choice streams (delete them). Item TR-5 wants the per-strain regimen-susceptible sterilization currently inside `TPTRx._apply_sterilization` factored onto `TBResistant` (e.g. `TBResistant.sterilize_covered(uids, regimen_drugs)`) so both TPT and latent treatment call it.
- **De-duplication between `TB` and `TBResistant` (per feedback).** Concrete targets found while tracing the code — fold these in while touching the relevant methods:
  - *Shape-parameter validation.* The `k_asy`/`k_non` `>= 0` checks are copy-pasted in `TB.__init__` ([tb.py:181-184](../../tb.py)) and `TBResistant.__init__` ([tb_resistant.py:84-87](../tb_resistant.py)) (the duplicate exists only because `TBResistant` applies pars after `super().__init__`). Extract a `TB._validate_pars()` method and call it from both after their respective `update_pars`, so the rule lives in one place.
  - *Reinfection-parameter defaulting.* The σ_L/σ_N `None`→default coupling ([tb_resistant.py:91-94](../tb_resistant.py)) moves to base `TB` as part of item 1, removing the `TBResistant` copy entirely.
  - *Reinfection-wane scheduling.* Base `TB` inlines the `ti_rr_reinfection_wane` scheduling twice in `step_transitions` ([tb.py:422-423,433-434](../../tb.py)) and again in `step_bookkeeping`, while `TBResistant` already has the `_set_reinfection_wane` helper ([tb_resistant.py:425-429](../tb_resistant.py)). Lift `_set_reinfection_wane` to base `TB` and replace all base inline copies with calls to it; `TBResistant` then just inherits it.
  - *CLEARED-entry bookkeeping.* Both classes repeat the "set `rr_reinfection`, schedule wane" pattern on every CLEARED entry (base per-pathway inline; `TBResistant` via `_reset_counts` + `_set_reinfection_wane`). Consider a small base helper `_enter_cleared(uids, rr)` that sets `rr_reinfection` and schedules waning, which `TBResistant` overrides/extends to also zero strain counts — collapsing several near-identical blocks.
  - Do these as small, separately-reviewable refactors (ideally landing before or alongside the behavioral changes) so the diffs for the behavioral items stay focused.
- **CRN / reproducibility.** Items 4 and TR-1 change the number/layout of RNG streams; expect existing seeded-run baselines to shift. Update any golden-value tests accordingly and note the break in the resistance docs/changelog.
- **Docs.** After implementation, update the resistance user manual / tech spec under `tbsim/resistance/docs/` to reflect: base-TB reinfection of latent agents, per-strain count accumulation on acquisition, multi-strain treatment acquisition, independent per-drug DST, and the new latent-treatment mode.
- **Test suites.** Extend `tests/test_resistance.py` (CI) with the targeted unit tests above; use `tbsim/resistance/devtests/` for heavier scenario checks. Run with the base conda Python per project convention.
