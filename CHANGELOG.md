# What's new

All notable changes to the codebase are documented in this file.

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