# TBsim Resistance

**Technical specification:** Loading — edits made to request a "same-strain" counter (see tracked changes)

> **Claude review (2026-07-14, `resistance-updates` branch):** Annotations below (`> …`) assess each non-struck-through comment against the current code in `tbsim/resistance/`. Struck-through items were verified as covered (one-line confirmations inline). Implementation plans for all confirmed gaps are in [loop_planned_fixes.md](loop_planned_fixes.md). Note: the three referenced issue files (`issue-dst-retreatment.md`, `issue-latent-treatment-divergence.md`, `issue-tpt-resist-penalty.md`) do **not** exist on this branch — they were on `resistance-uat`/other branches and were not carried over.

## KHG notes on mme_resistance branch

- ~~checks to ensure that StrainSpec phenotypes are consistent (e.g., INH vs inh — is this done in the StrainCatalog step?)~~
    - ~~maybe should have to define the set of phenotypes elsewhere to then be checked against all strainspec/strain specific interventions to make sure you're not creating mismatch, e.g. between infecting strain and tx phenotypes~~
    - KHG — think this is done in 'Confirm Tx/DST/TPT dict keys resolve through `Strains.drug_idx`'

> **Claude — partially covered.** `Strains.__init__` validates the drug set (rejects duplicate names, requires ≥1 drug) and `rel_fitness` keys (raises on unknown drug / out-of-range cost). `TxR.resist_penalty` keys *do* resolve through `strains.drug_idx[d]`, so a typo raises `KeyError` (`treatments.py:67`). **But** `regimen_drugs` (`treatments.py:56,69`), `TxR.q_acq`/`TPTRx.p_tpt_acq` (`.get(drug, 0.0)`), and `DST.sens`/`spec` (`_perdrug` → `.get(drug, default)`) all use silent `.get()`: a mis-typed drug name there is silently ignored rather than flagged. There is no single place that checks all intervention drug names are a subset of `strains.drugs`. Low-severity; see loop_planned_fixes.md (validation helper).
- ~~unclear how treatment timing works?~~
    - ~~seems like it assumes constant hazard of cure (and failure?) during duration of treatment? but will it continue treatment (e.g., for costing purposes) until end of duration?~~
    - ~~other doc says "TxDelivery pre-rolls treatment outcomes and resolves them at treatment completion"~~
    - ~~but also mentioned "StrainAwareTxDelivery.step_start_treatment clears susceptible strains"~~
    - ~~more accurate that cure could happen at any time point during treatment (although probably not constant hazard) while failure should be defined at end of regimen and resistance decided on then? or allowed to develop over course of treatment duration (in which case could hypothetically get resistance developed + cured in same treatment course?)~~
    - KHG — leaving for posterity but don't think any of the above applies to new branch

> **Claude — struck items verified covered.** `TxDeliveryR` pre-rolls the whole-course outcome once at initiation (`roll_survivors`, adherence then per-strain cure) and resolves it at completion when `ti >= ti_treatment_end` (`treatments.py:244-281`). There is no constant per-step cure hazard; the outcome is frozen at start and applied at the end of the fixed `dur_treatment`, so the old concerns don't apply.
- ~~how does `RegimenRouter` come into play for StrainAwareTxDelivery?~~
    - ~~e.g. could it be used to model product mix if needed in absence of DST results, or only applies to individuals with resolved observed resistance phenotypes?~~
    - ~~can it be used to consume prior treatment status (e.g., immediately go to 2nd line treatment following failure and/or DST results — or should we be building this into eligibility statements for each treatment regimen?)~~
    - KHG — leaving for posterity but no longer in new branch

> **Claude — struck item verified covered.** `RegimenRouter` does not exist on this branch (confirmed by grep; only referenced in `resistance_comparison.md` as the PR #430 / `mme_resistance` alternative that was deliberately *not* ported). Its role is filled by `DSTDelivery.matches(DRUG=bool, …)` (composable observed-profile routing, `dst.py:124`) plus `TxDeliveryR(supersedes=[…])` for mid-course regimen switching (`treatments.py:197,223`).

- diagnostics
    - confirming `treatment_monitoring_eligibility(tx_name, after_steps=N)` would be usable for standard dx eligibility statement too (i.e. not just DST)?

> **Claude — yes, covered.** `treatment_monitoring_eligibility` returns a generic `sim -> uids` callable (`treatments.py:292-320`) that reads only TB state + `ti_treatment_start`; it has no DST dependency. Its own docstring states it can be fed as `eligibility=` to a monitoring `DxDelivery` to flag still-bacteriologically-positive agents. So it is usable for a standard Dx eligibility statement, not just DST-routing. No code change needed.

- should there be an easier switch/interface between strain aware vs agnostic models?
    - e.g., thinking about LAI work — should there be easier way to create 'strain reduced' model with just 1 strain but otherwise using all resistance mechanisms/classes for comparison?

> **Claude — GAP (minor / future).** There is no single-strain "reduced" mode and no one-line toggle between agnostic (`tbsim.TB`) and strain-aware (`TBResistant`). `Strains` requires ≥1 drug, so the smallest strain space is `m = 2` (pan + resistant); you cannot instantiate a genuine 1-strain `TBResistant`. You *can* approximate an agnostic run with `TBResistant` by setting `init_strains=[1, 0]` (all pan-susceptible seeds), `p_rand=None` (no de-novo), and no resistant transmission — but this is undocumented and not a first-class switch. See loop_planned_fixes.md (strain-reduced convenience mode).

## TR notes on resistance_uat branch/user guide

In approximate order of importance:

1. **Bug:** with DST-linked treatment eligibility, DST result never expires and agents are continuous retreated (see `issue-dst-retreatment.md` for details)

> **Claude — CONFIRMED (bug still present).** `DSTDelivery` stores `dst_profile` / `dst_tested` permanently and never resets them; `ti_dst` is recorded but never consumed for expiry (`dst.py:99-103,153-162`). `matches()` / `observed_resistant()` therefore keep an agent eligible for as long as its stored profile matches. `matches()` excludes only *currently* on-treatment agents (`exclude_on_treatment`), so as soon as a failed course returns the agent to ASY/SYM it is re-selected and re-treated indefinitely. `issue-dst-retreatment.md` is not on this branch. Implementation plan in loop_planned_fixes.md (DST result expiry / re-test gating).

2. **Bug:** TPT-driven acquisition is invisible to the origin decomposition. `TPTRx._acquire` mutates strain_mask but increments **no** resistance-origin counter, and ResistanceStats sums only de-novo, transmitted, and `TxDeliveryR.n_acquired` (`analyzers.py:50-52`). So when TPT selection (`p_tpt_acq`) is active, new resistance it creates is counted in **none** of the three flux channels — the decomposition undercounts total new resistance and misattributes the epidemic's resistance origins.

> **Claude — CONFIRMED.** `TPTRx._acquire` writes `tb.strain_mask[uids] = surv` with no counter (`tpt.py:80-110`); it exposes no result. `ResistanceStats.step` populates `flux_denovo`/`flux_transmitted` from `TBResistant` results and `flux_txacq` from `TxDeliveryR.results.n_acquired` only (`analyzers.py:44-53`). TPT-acquired resistance is counted in none of the three channels. Implementation plan in loop_planned_fixes.md (add a TPT-acquired counter + a fourth `flux_tptacq` channel).

3. **Change:** Inconsistency between what happens to `INFECTED` agents who are treated between base/single-strain `tbsim` and resistant/multi-strain version (see `issue-latent-treatment-divergence.md` for description and notes on desired behavior)

> **Claude — CONFIRMED divergence (desired-behavior decision needed).** Base `tbsim.TxDelivery.step_start_treatment` clears latent (`TBS.INFECTION`) agents **immediately** to CLEARED, with no treatment course (`interventions/treatments.py:210-228`). `TxDeliveryR._initiate` has no latent special-case: any selected agent — including latent, if a custom `eligibility` selects them — is put into a full `TREATMENT` course, pre-rolled and resolved after `dur_treatment`, and can fail / acquire resistance (`treatments.py:217-250`). By default the rate-based path only targets ASY/SYM so latent agents aren't treated, but a DST/eligibility-routed delivery can, producing behavior that diverges from base tbsim. `issue-latent-treatment-divergence.md` is not on this branch, so the exact desired behavior is unrecorded. Implementation plan (with options) in loop_planned_fixes.md.
4. ~~**Change:** We probably only want `p_strain_obs` to apply for multistrain infections — because otherwise this should already be wrapped into DST sensitivity and specificity estimates.~~
    a. **Decision:** We are okay with how this is handled for now and will probably just override `p_strain_obs` to turn this feature off entirely.

> **Claude — struck; decision achievable.** `p_strain_obs` can be disabled by passing a scalar `1.0` (or a per-id dict of 1.0) to `DST(...)`, which makes every carried strain always observed (`dst.py:39-44,73`). No code change needed for the stated decision.

5. **Question:** Is it possible to make the `treatment_monitoring_eligibility` mechanism contingent on a DST result or on eventual failure?
    a. Right now it seems like in the example in block 9 of the user guide, everyone switches after 2 timesteps — just want to confirm this capability is available? If not, can we please add this capability.

> **Claude — PARTIAL (not first-class).** `treatment_monitoring_eligibility` is purely time-based: it selects everyone on `tx_name`'s course past `after_steps` (`treatments.py:292-320`), which is why "everyone switches after 2 timesteps." It is **not** contingent on DST or on predicted failure, and there is no provided combinator to AND it with `DSTDelivery.matches(...)`. You *can* write a custom `eligibility=lambda sim: monitoring(sim).intersect(dst.matches(RIF=True)(sim))` today (both return `ss.uids`), and the failure outcome is already known at initiation (`pending_surv`), so a "will fail" predicate is feasible — but neither is exposed. Implementation plan in loop_planned_fixes.md (eligibility combinator + DST-/failure-contingent monitoring).

6. **Lower priority:** Right now resistant strains have a 0% chance of being cleared by TPT. This is fine as default behavior, but we eventually want the ability to add a similar `resist_penalty` as in TB treatment to TPT
    a. Not as important for the current project, but desired in the future for any 2-drug TPT regimens, for example, which may be partially efficacious against strains with resistance to only 1 of the drugs; see `issue-tpt-resist-penalty.md` for details

> **Claude — CONFIRMED (future feature).** TPT sterilization is all-or-nothing: `_covered_mask` clears only strains susceptible to *every* regimen drug, so a strain resistant to any regimen drug has a 0% clearance probability (`tpt.py:67-71,112-133`). Progression-protection (`apply_protection`) uses the same binary per-strain coverage weight `w`, so there is no partial efficacy against a strain resistant to only 1 of 2 regimen drugs. `TxR` has `resist_penalty` (reduced-but-nonzero efficacy); `TPTRx` has no equivalent. `issue-tpt-resist-penalty.md` is not on this branch. Implementation plan (future) in loop_planned_fixes.md.

7. **Note on user guide:** in block 8 (DST and treatment), it's a bit confusing to have the second-line drug (for whom only RIF-resistant are eligible) seemingly be a RIF-based regimen, as indicated by `regimen_drugs=['RIF']`.
    a. I don't think this is a bug in the code base or anything — just something confusing in the user guide that may be nice to clear up if the guide eventually gets merged in the `resistance` and/or `main` branches (which would be nice since the guide is very helpful!)

> **Claude — CONFIRMED (docs clarity, not a code bug).** In `tbsim-resistance-user-manual.md` block 8 (§8, lines 437-446) the `second` delivery targets observed-RIF-resistant agents but sets `regimen_drugs=['RIF']` with `base_efficacy=0.8` and **no** `resist_penalty`. Because `resist_penalty` is empty, the RIF-resistant strain's `eff_by_id` = `0.8 × 1.0` (`treatments.py:64-70`) — i.e. the "RIF second-line" cures RIF-resistant TB at full efficacy, which reads as contradictory. It runs correctly but models a second-line *drug* mislabeled as RIF. Doc-clarification plan in loop_planned_fixes.md (rename to a distinct second-line drug, e.g. `['BDQ']`, or add explanatory prose).

8. I've noted the remaining gaps identified by Minerva at the end of the user guide and agree they are not currently included. For me, these are important to note and address in the future, but not high priority to fix immediately (except the time-varying progression hazard, which we owe software additional info on).

> **Claude — CONFIRMED (matches the guide's "Known limitations").** `tbsim-resistance-user-manual.md` §13 lists: adherence distribution (single Bernoulli), DST-indeterminate (binary profile only), failure-vs-new-case classifier (no durable "time since last treatment"), time-varying progression hazard, LAI_TPT burden table, count-based strain carriage, and LTFU outcome — all confirmed absent in code. **Time-varying progression hazard** is the flagged priority: `ti_infected` is reset on successful/blocked exposures (hook present, `tb_resistant.py:171,182`) but progression rates are constant-hazard state transitions with no dependence on time-since-infection (`step_transitions`), and the code comment at `tb_resistant.py:166-168` explicitly calls the reset "a no-op until time-varying progression exists." UAT-07 is marked TODO/NOT IMPLEMENTED (`tbsim-resistance-uat.md:268-288,693`). Implementation plan (scoped, pending spec input) in loop_planned_fixes.md.

## KG Notes (Claude assisted)

- Resistance following treatment failure capped at one strain
    - `TxR.acquire()` and `TPTRx._acquire()` both behave differently to `_denovo` resistance
    - **What happens:** for each regimen drug, the code draws **one** Bernoulli per agent, then walks `sus_ids` (strains susceptible to that drug) in ascending strain-id order and flips only the *first* one the agent carries, marking that agent `done` so no other co-carried susceptible strain can acquire resistance that episode:
    - **Why it matters:** the tech spec says resistance acquisition applies "among baseline *i*-susceptible strains" of an agent — de-novo acquisition (`_denovo` in `tb_resistant.py`) correctly honors this by looping over *every* carried strain independently. But treatment/TPT acquisition only ever mutates one strain per agent per drug, and it's always the lowest-id (pan-leaning) candidate strain, never a strain that happens to carry other resistances already. This only bites with ≥2 drugs and an agent superinfected with multiple strains that are each susceptible to the drug in question — a scenario none of the current devtests exercise (they're all single-drug, 2-strain A/B setups where at most one candidate ever exists).

> **Claude — CONFIRMED (behavior/decision, subtle).** Verified: `TxR.acquire` (`treatments.py:120-129`) and `TPTRx._acquire` (`tpt.py:101-108`) both draw one trial per agent per regimen drug, then walk `sus_ids` in ascending strain-id order and flip only the first carried susceptible strain (`done |= has_j`), whereas `TBResistant._denovo` (`tb_resistant.py:302-321`) loops over every carried strain `j` independently. Two distinct facts here: (a) **one trial per drug** for treatment/TPT is *intended* per the spec ("once per treatment episode"), unlike per-strain de-novo — this is a deliberate difference, not a bug; (b) but **which** strain the hit lands on is arbitrary (lowest-id, pan-leaning), which is a genuine modeling choice that is currently undocumented and only matters with ≥2 drugs + multi-susceptible-strain superinfection. Needs a decision (random/fitness-weighted selection among carried susceptible strains vs. keep lowest-id + document). Implementation plan in loop_planned_fixes.md.
