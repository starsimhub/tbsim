# What's new

All notable changes to the codebase are documented in this file.

## Version 0.10.0 (2026-07-10)
- Added the multi-strain / drug-resistance extension `tbsim.resistance`:
    - `TBResistant` (drop-in `TB` subclass) encoding `2ⁿ` strains over `n` drugs as a per-agent bitmask with per-strain fitness costs, superinfection, de-novo resistance, and a per-strain multiplicity counter (identical-strain re-exposure now increments the count and feeds the transmission multinomial and progression bottleneck).
    - Strain-aware product/delivery interventions `TxR`/`TxDeliveryR` (per-strain efficacy incl. explicit `efficacy_by_strain` vectors, float or per-agent-distribution `adherence`, acquisition-on-failure, regimen switching, and `failure_case_eligibility` retreatment-vs-new-case classification), `DST`/`DSTDelivery` diagnostics with observed-profile routing, and strain-aware `TPTRx`.
    - `ResistanceStats` and `StrainResults` analyzers, plus a two-strain reference ODE (`tbsim.compartmental.TwoStrainODE`) for ABM↔ODE validation.
- Added a resistance tutorial, user manual, and updates guide under `docs/` and `tbsim/resistance/docs/`.

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