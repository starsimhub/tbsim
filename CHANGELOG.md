# What's new

All notable changes to the codebase are documented in this file.

## Version 1.0.0 (in development)
- Moving to Starsim v4, where a model declared as states and transitions runs as an ODE, as a stochastic compartmental model, or as agents. This is a breaking release: most of the API has changed.
- **`tbsim.TB` is declared** (`tbsim/tb.py`): the LSHTM model as states and transitions, which as an ODE is the reference ODE (`compartmental/lshtm_ode.R`) to within 1e-4, and as agents runs with networks and demographics. People who are no longer infected are in `cleared`, `recovered`, or `treated`, which differ in their risk of reinfection (in place of a per-agent relative risk). Optional: front-loaded progression (`k_latent`, with a remote latent stage, in place of `k_asy`/`k_non`), relapse after treatment (`p_relapse`), and waning protection (`dur_reinfection_protection`). Parameters are `ss.par`s, so they can be fitted with `sim.calibrate()`. `TBS`, `BaseTB`, `choice2d`, and the per-agent `rr_*` arrays are gone: interventions and comorbidities act on the model's transitions instead. `tbsim.state_codes(tb)` gives each agent's state as a number.
- **Interventions are Starsim v4 products and deliveries**, so they run under every solver: `tbsim.care(tb)` (seeking care, testing, and starting treatment) in place of `HealthSeekingBehavior`, `DxDelivery`, and `TxDelivery`; `tbsim.Xpert()`, `OralSwab()`, `FujiLAM()`, `CAD()`, and `tbsim.test()` (an `ss.Test` with a sensitivity for each state, by age); `tbsim.regimen('dots')` (the model's treatment parameters for a regimen) in place of `Tx` and its subclasses; `tbsim.TPT()` and `tbsim.BCG()` (products acting on progression, recovery, and TB mortality) in place of `TPTTx`, `BCGVx`, and their deliveries; `ss.change()` in place of `BetaByYear`.
- **Comorbidities** act on the model through modifiers: `tbsim.hiv_effects(tb, hiv)` and `tbsim.nutrition_effects(tb, ...)` in place of `TB_HIV_Connector` and `TB_Nutrition_Connector`. `supplementation_rr()` now gives the rate ratio (it gave 0 for 0.5, from an integer array).
- **`tbsim.Sim`** takes the TB model (`tb=`) and TB parameters directly; it has no default network or demographics, so it runs as an ODE unless given a network or `solver='abm'`.
- **Drug resistance is a strain dimension of `tbsim.TB`** (`tbsim/resistance.py`): `tbsim.TB(drugs=['RIF', 'INH'])` has a strain for each set of drugs it is resistant to, and people can carry several at once (up to `max_strains`), with superinfection, the progression bottleneck, resistance acquired on progression and during treatment, and treatment that cures each strain with its own chance. It runs under every solver, every intervention works with it, and results are read by strain (`n_symptomatic.by('resistance')`, `tbsim.resistant_share(sim)`). A test for resistance is `ss.Test(detects=tb.resistance != 'DS')`. This replaces `TBResistant`, `Strains` (now the strains and sets of them for a list of drugs), `TxR`/`TxDeliveryR`, `DST`/`DSTDelivery`, `TPTRx`, and the resistance analyzers.
- `tbsim.LSHTMResistant` (`tbsim/models.py`): the structure of `two_strain_ode.R` (treatment from asymptomatic and symptomatic TB), with resistance added the same way; as an ODE with one drug it is `two_strain_ode.R` to within 1e-4 in every compartment.
- Removed `tbsim.compartmental.TB_SS`, which `tbsim.TB` replaces. `TB_ODE` and `TwoStrainODE` stay as the references.
- **Migration** is two parameters of `tbsim.TB`: `immigration` (people arriving per year, into the states given by `immigrants=`, at ages like the population's) and `emigration` (per person per year), in place of `Migration`. It runs under every solver, and the parameters can be fitted.
- **`tbsim.HIV` is declared**: people at risk are infected at a given `incidence` (which can vary in time, and be fitted) and progress through acute and chronic HIV to AIDS; `art` is a dimension of the infected states that slows progression, and people start ART at the rate `art_start` or by a delivery (`ss.transfer(art=True, coverage=...)`). It runs under every solver: as an ODE, TB (with or without resistance) and HIV are tracked jointly, so `hiv_effects()` works as compartments too. This replaces `HIVState`, `on_ART`, and `HivInterventions` (which set prevalence and ART coverage to targets).
- `tbsim.TB` takes `pars=` (a dict of parameters) as its first argument, as Starsim modules do.
- `tbsim.Sim` also passes `init`, `drugs`, `max_strains`, `strains`, and `immigrants` to the TB model. `tbsim.care(tb)` tests by age only if the model is stratified by age at 15 (`tb.stratify(age=[0, 15])`); `tbsim.test()` no longer says that `tbsim.TB` is by default.
- Fixed: `tbsim.HIV` never seeded its initial infections (`init_prev`, `init_onart`); its shares (`hiv_prevalence`, `atrisk`, `acute`, `latent`, `aids`) and `Malnutrition`'s `people_alive` were multiplied by the population scale; `DwellTime` labeled leaving the model (death or emigration) as `-1.0.nan`.
- **Second-line treatment, strain-aware TPT, and contact tracing**: `tbsim.TB(drugs=..., second_line=True)` has a second-line regimen (its own rates, cure of resistant strains, and relapse), which `tbsim.care(tb, dst=tbsim.DST(tb))` starts for people found resistant by a test for resistance; `add_resistance()` takes a cure, `rr_cure`, and `q_treat` for each regimen. `tbsim.TPT(tb=tb, drugs=['INH'])` protects only against the strains its drugs act on (and with `tb=`, acts on every transition from latent infection to active TB). `tbsim.contact_tracing(tb)` tests the household contacts of people starting treatment, treats those who test positive, and gives TPT to those with a latent infection (as agents; with `ss.screen(trace=...)`). `tbsim.monitoring(tb)` tests people on first-line treatment for resistance, and switches those found resistant to second line (`switch_second_line`). `tbsim.TB(tpt=dict(drugs=['INH'], clearance=..., acquisition=...))` declares what TPT does to the infection of the people who hold it: it clears the strains it acts on, and the others can become resistant to its drugs, most often in people with active TB that was not found (these are rates while TPT is held, where `TPTTx` and `TPTRx` had outcomes at the end of a course). As agents, `tbsim.care(tb, dst=..., retreatment=ss.years(2))` tests for resistance only the people treated within that time before (`tbsim.treated_within()`). These replace `TPTHousehold`, `HouseholdContactTracing`, `treatment_monitoring_eligibility`, `failure_case_eligibility`, and the sterilization and acquisition of `TPTTx` and `TPTRx`.
- The docs, tutorials, and examples use the new API.
- `DwellTime`'s Kaplan-Meier plot computes the curve itself, so `lifelines` is no longer a dependency.
- Raised minimum Starsim dependency to `>=4.0`.

## Version 0.10.3 (2026-08-29)
- Updated for Starsim 3.6.0, which makes `float()` on an `ss.TimePar` raise a `TypeError` (it silently discarded the unit) and records `new_infections` inside `ss.Infection.set_prognoses()` rather than inferring them from `ti_infected` afterwards.
- **Fixed `DxDelivery(result_validity=...)`**: `step_expire_results` converted the validity window with `float(self.result_validity)`, which now raises. It divides by the module's own `dt` instead (`self.result_validity / self.t.dt`), matching `DSTDelivery._expire_stale`.
- **Fixed the LSHTM reference ODE** (`tbsim.compartmental.TB_SS`): its Euler step took `float(self.dt)` for a timestep in years, which now raises; it uses `self.dt.years` explicitly.
- **Fixed silently-zero incidence results for `TBResistant`**: `TBResistant.set_prognoses` overrode the base method without calling `super()`, so under Starsim 3.6.0 its `new_infections` and `cum_infections` results stayed zero for the whole run (under 3.5.x the base class recomputed them afterwards, hiding the omission). Added a regression test.
- Raised minimum Starsim dependency to `>=3.6.0`.

## Version 0.10.2 (2026-07-20)
- Reworked several multi-strain / drug-resistance behaviors (`tbsim.resistance`) and lifted latent reinfection into base `TB`.
- **Latent reinfection in base `TB`**: `rr_reinfection_inf` (σ_L) and `rr_reinfection_non` (σ_N) are now base-`TB` parameters, so latent (`INFECTION`) and non-infectious (`NON_INFECTIOUS`) agents are reinfection-eligible in single-strain models too. A re-exposure resets the `ti_infected` clock without otherwise changing the agent's state (clock reset only). σ_L defaults to `rr_reinfection_rec` and σ_N to σ_L; set them to `0` to disable.
- **Per-strain count preserved through acquisition**: when a strain acquires resistance (de-novo, on treatment, or under TPT), the emergent resistant strain now inherits the source strain's per-strain multiplicity count and accumulates onto any copies already present, instead of resetting to `1`. Replacement moves the source count to the target; de-novo `'mixed'` keeps the source and copies the count to the target.
- **Multi-strain treatment/TPT acquisition**: treatment- and TPT-acquired resistance now let *each* carried drug-susceptible strain acquire resistance independently in one round (mirroring de-novo), rather than a single picked strain; the `acq_select` parameter was removed from `TxR`/`TPTRx`.
- **Independent per-drug DST errors**: `DST` draws sensitivity/specificity errors independently per (strain, drug), so a multi-drug DST behaves like independent per-drug tests.
- **Latent-treatment mode**: `TxDeliveryR(treat_latent=False)` (default) now clears a latent agent's regimen-susceptible strains with certainty while keeping any regimen-resistant strain (no course, no acquisition, not counted in `n_treated`), instead of unconditionally clearing the agent to `CLEARED`; pan-susceptible / single-strain behavior is unchanged.
- **Added a `skills/` folder** (symlink to `.claude/skills/`) holding repo-specific [Claude Code](https://claude.com/claude-code) skills, starting with `tbsim.address-issue` for fixing a specified GitHub issue end to end.
- **Fixed cross-delivery treatment resolution (#447)**: with two or more `TxDeliveryR` deliveries treating overlapping agents, one delivery could re-resolve an agent another delivery was actively treating — using its own stale `pending_surv` — zeroing an infectious agent's `strain_mask` and crashing the next transmission draw (`IndexError` in the strain PPF). `TxDeliveryR` now tracks a delivery-scoped `on_course` ownership flag, so `_resolve`, `interrupt`, `treatment_monitoring_eligibility`, and `will_fail` only ever act on the courses that delivery itself started. `ti_treatment_end` still persists after a course (so the `retreat_after` refractory guard is unchanged).

## Version 0.10.1 (2026-07-15)
- Added optional **time-varying (front-loaded) TB progression**: the latent `INFECTION`-exit hazards can now decline exponentially with time since infection, `inf_asy(τ) = inf_asy · exp(−k_asy · τ)` (and `inf_non` via `k_non`), where `τ` is years since the agent was infected (reinfection restarts the clock). Controlled by two new `TB` parameters, `k_asy` and `k_non`, both defaulting to `0` (constant hazard), so the default behaviour and existing calibrations are bit-for-bit unchanged. Set `k_asy > 0` for the recommended one-parameter front-loaded progression to active TB, matching the front-loaded shape seen in historical household-contact data. The per-agent effective rates are exposed via `TB.progression_rates()`.

## Version 0.10.0 (2026-07-10)
- Added the multi-strain / drug-resistance extension `tbsim.resistance`.
- `TBResistant` (drop-in `TB` subclass) encoding `2ⁿ` strains over `n` drugs as a per-agent bitmask with per-strain fitness costs, superinfection, de-novo resistance, and a per-strain multiplicity counter (identical-strain re-exposure now increments the count and feeds the transmission multinomial and progression bottleneck).
- Strain-aware product/delivery interventions `TxR`/`TxDeliveryR` (per-strain efficacy incl. explicit `efficacy_by_strain` vectors, float or per-agent-distribution `adherence`, acquisition-on-failure, regimen switching, and `failure_case_eligibility` retreatment-vs-new-case classification), `DST`/`DSTDelivery` diagnostics with observed-profile routing, and strain-aware `TPTRx`.
- `ResistanceStats` and `StrainResults` analyzers, plus a two-strain reference ODE (`tbsim.compartmental.TwoStrainODE`) for ABM↔ODE validation.

## Version 0.9.0 (2026-07-09)
- Added read-only boolean views of TB `state` (`latent`, `non_infectious`, `asymptomatic`, `symptomatic`, `active_tb`, `terminal`), giving the Starsim boolean idiom at call sites while keeping the categorical `state` as the single source of truth.
- Replaced the `TBS` static-method state groups with tuple constants (`TBS.ACTIVE`, `TBS.TERMINAL`, `TBS.CARE_SEEKING`) used via `state.isin(...)`; added `HIVState.INFECTED` similarly.
- Made `TBS` enum values contiguous (0–8) so per-state result counts use a single-pass `np.bincount`, replacing per-state `==`/`np.isin` scans and the every-step re-derivation of `infected`/`susceptible`/`on_treatment`.
- Updated for Starsim 3.5.1: `ss.library.HouseholdNet` (replacing `ss.HouseholdNet`).
- Raised minimum Starsim dependency to `>=3.5.1`.

## Version 0.8.1 (2026-06-10)
- Fixed a UID/position confusion bug (#425) where several interventions identified agents by running `np.where`/`np.flatnonzero` over a starsim `Arr`'s compact, alive-only `.values` view and then treating the resulting positions as UIDs. Once any agents had died (so UIDs no longer matched compact positions), this silently selected the wrong agents. Affected `HouseholdContactTracing.step`, `TPTHousehold.check_eligibility`, `TxDelivery._get_eligible`, `Migration._members_by_household_id`, and `HealthSeekingBehavior.step`. All now use native starsim filtering (`arr.auids[mask]`) so positions map back to real UIDs.
- Added regression tests covering household contact tracing, treatment eligibility, care-seeking, and diagnostic administration after agent deaths.

## Version 0.8.0 (2026-06-02)
- Added `migration.py` with a single `Migration` demographics class providing bidirectional, household-aware population turnover:
    - Immigration (new agents enter) and emigration (existing agents leave), each driven by an annual `ss.freq` rate; setting `emigration_rate=0` gives immigration-only behavior
    - Optional `maintain_population` mode that tops up arrivals each step to hold the active (non-terminal) population near its starting size
    - Configurable immigrant age profiles via `immigration_age_distribution` bins or an `age_data` histogram, plus optional age-weighted emigrant selection via `emigration_age_distribution`
    - TB-state-aware imports: immigrants enter with a TB-state mix from `tb_state_distribution`, or a default derived from the TB module's `init_prev` and progression parameters
    - Household integration with `ss.HouseholdNet`: immigrants are assigned to existing households (size-weighted) and wired into household edges, and emigrants are removed from their households
    - Per-step `n_immigrants`, `n_emigrants`, and `net_migration` results
- Removed the separate `Immigration` class in favor of the unified `Migration` class
- Removed the `TBAcute` model variant and its ACUTE state; only `TB` is supported now

## Version 0.7.0 (2026-03-16)
- Added `tbsim.Sim`, a convenience wrapper around `ss.Sim` that auto-routes flat parameters between the sim and the TB module, provides TB-specific defaults (demographics, networks, disease), and supports a `tb_model` argument for selecting among TB model variants. Includes a `demo()` function for quick setup.
- Refactored all diagnostic and treatment interventions into a product/delivery architecture:
    - Added `Dx` diagnostic product class with DataFrame-based test definitions supporting state, age, and HIV stratification
    - Added built-in diagnostic products: `Xpert`, `OralSwab`, `FujiLAM`, `CAD`
    - Added `DxDelivery` intervention for delivering diagnostics with eligibility filtering, coverage, and false-negative retry logic
    - Added `Tx` treatment product class with drug-type-based efficacy
    - Added built-in treatment products: `DOTS`, `DOTSImproved`, `FirstLine`, `SecondLine`
    - Added `TxDelivery` intervention for delivering treatments with success/failure tracking and retry logic
    - Simplified `drug_types.py` to a single dictionary of drug parameters (previously ~600 lines of class hierarchy)
    - Removed old monolithic `tb_diagnostic.py`, `tb_drug_types.py`, and `tb_treatment.py`
- Removed `TB_EMOD` and renamed `TB_LSHTM` to TB; only `TB` and `TBAcute` are supported now
- Added `HouseholdStats` analyzer for tracking household size distributions, age-mixing matrices, and contact patterns over time when using `ss.HouseholdNet`. Includes visualization methods for household statistics, age-mixing heatmaps, and normalized contact matrices.
- Renamed `tb_health_seeking.py` to `health_seeking.py` and simplified the `HealthSeekingBehavior` class
- Moved `immigration.py` to an archive folder
- Added admin files (`code_of_conduct.md`, `contributing.md`, `CHANGELOG.md`)
- Updated `starsim` dependency to v3.2.1 for `HouseholdNet` support