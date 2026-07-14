# TBsim Resistance

**Technical specification:** Loading — edits made to request a "same-strain" counter (see tracked changes)

## KHG notes on mme_resistance branch

- ~~checks to ensure that StrainSpec phenotypes are consistent (e.g., INH vs inh — is this done in the StrainCatalog step?)~~
    - ~~maybe should have to define the set of phenotypes elsewhere to then be checked against all strainspec/strain specific interventions to make sure you're not creating mismatch, e.g. between infecting strain and tx phenotypes~~
    - KHG — think this is done in 'Confirm Tx/DST/TPT dict keys resolve through `Strains.drug_idx`'
- ~~unclear how treatment timing works?~~
    - ~~seems like it assumes constant hazard of cure (and failure?) during duration of treatment? but will it continue treatment (e.g., for costing purposes) until end of duration?~~
    - ~~other doc says "TxDelivery pre-rolls treatment outcomes and resolves them at treatment completion"~~
    - ~~but also mentioned "StrainAwareTxDelivery.step_start_treatment clears susceptible strains"~~
    - ~~more accurate that cure could happen at any time point during treatment (although probably not constant hazard) while failure should be defined at end of regimen and resistance decided on then? or allowed to develop over course of treatment duration (in which case could hypothetically get resistance developed + cured in same treatment course?)~~
    - KHG — leaving for posterity but don't think any of the above applies to new branch
- ~~how does `RegimenRouter` come into play for StrainAwareTxDelivery?~~
    - ~~e.g. could it be used to model product mix if needed in absence of DST results, or only applies to individuals with resolved observed resistance phenotypes?~~
    - ~~can it be used to consume prior treatment status (e.g., immediately go to 2nd line treatment following failure and/or DST results — or should we be building this into eligibility statements for each treatment regimen?)~~
    - KHG — leaving for posterity but no longer in new branch
- diagnostics
    - confirming `treatment_monitoring_eligibility(tx_name, after_steps=N)` would be usable for standard dx eligibility statement too (i.e. not just DST)?
- should there be an easier switch/interface between strain aware vs agnostic models?
    - e.g., thinking about LAI work — should there be easier way to create 'strain reduced' model with just 1 strain but otherwise using all resistance mechanisms/classes for comparison?

## TR notes on resistance_uat branch/user guide

In approximate order of importance:

1. **Bug:** with DST-linked treatment eligibility, DST result never expires and agents are continuous retreated (see `issue-dst-retreatment.md` for details)
2. **Bug:** TPT-driven acquisition is invisible to the origin decomposition. `TPTRx._acquire` mutates strain_mask but increments **no** resistance-origin counter, and ResistanceStats sums only de-novo, transmitted, and `TxDeliveryR.n_acquired` (`analyzers.py:50-52`). So when TPT selection (`p_tpt_acq`) is active, new resistance it creates is counted in **none** of the three flux channels — the decomposition undercounts total new resistance and misattributes the epidemic's resistance origins.
3. **Change:** Inconsistency between what happens to `INFECTED` agents who are treated between base/single-strain `tbsim` and resistant/multi-strain version (see `issue-latent-treatment-divergence.md` for description and notes on desired behavior)
4. ~~**Change:** We probably only want `p_strain_obs` to apply for multistrain infections — because otherwise this should already be wrapped into DST sensitivity and specificity estimates.~~
    a. **Decision:** We are okay with how this is handled for now and will probably just override `p_strain_obs` to turn this feature off entirely.
5. **Question:** Is it possible to make the `treatment_monitoring_eligibility` mechanism contingent on a DST result or on eventual failure?
    a. Right now it seems like in the example in block 9 of the user guide, everyone switches after 2 timesteps — just want to confirm this capability is available? If not, can we please add this capability.
6. **Lower priority:** Right now resistant strains have a 0% chance of being cleared by TPT. This is fine as default behavior, but we eventually want the ability to add a similar `resist_penalty` as in TB treatment to TPT
    a. Not as important for the current project, but desired in the future for any 2-drug TPT regimens, for example, which may be partially efficacious against strains with resistance to only 1 of the drugs; see `issue-tpt-resist-penalty.md` for details
7. **Note on user guide:** in block 8 (DST and treatment), it's a bit confusing to have the second-line drug (for whom only RIF-resistant are eligible) seemingly be a RIF-based regimen, as indicated by `regimen_drugs=['RIF']`.
    a. I don't think this is a bug in the code base or anything — just something confusing in the user guide that may be nice to clear up if the guide eventually gets merged in the `resistance` and/or `main` branches (which would be nice since the guide is very helpful!)
8. I've noted the remaining gaps identified by Minerva at the end of the user guide and agree they are not currently included. For me, these are important to note and address in the future, but not high priority to fix immediately (except the time-varying progression hazard, which we owe software additional info on).

## KG Notes (Claude assisted)

- Resistance following treatment failure capped at one strain
    - `TxR.acquire()` and `TPTRx._acquire()` both behave differently to `_denovo` resistance
    - **What happens:** for each regimen drug, the code draws **one** Bernoulli per agent, then walks `sus_ids` (strains susceptible to that drug) in ascending strain-id order and flips only the *first* one the agent carries, marking that agent `done` so no other co-carried susceptible strain can acquire resistance that episode:
    - **Why it matters:** the tech spec says resistance acquisition applies "among baseline *i*-susceptible strains" of an agent — de-novo acquisition (`_denovo` in `tb_resistant.py`) correctly honors this by looping over *every* carried strain independently. But treatment/TPT acquisition only ever mutates one strain per agent per drug, and it's always the lowest-id (pan-leaning) candidate strain, never a strain that happens to carry other resistances already. This only bites with ≥2 drugs and an agent superinfected with multiple strains that are each susceptible to the drug in question — a scenario none of the current devtests exercise (they're all single-drug, 2-strain A/B setups where at most one candidate ever exists).
