# TBsim × Drug Resistance — Cross-Feature Mapping

This document lists TBsim features and validates whether each can be combined with or is impacted by the drug resistance implementation (`MultiStrainTB` + `tbsim/resistance/`).

## Table of contents

- [Legend](#legend)
- [1. Core disease models](#1-core-disease-models)
- [2. Transmission & networks](#2-transmission--networks)
- [3. Diagnostics](#3-diagnostics)
- [4. Treatment](#4-treatment)
- [5. TPT (preventive therapy)](#5-tpt-preventive-therapy)
- [6. Vaccination & prevention](#6-vaccination--prevention)
- [7. Comorbidities](#7-comorbidities)
- [8. Demographics & migration](#8-demographics--migration)
- [9. Resistance engine (strain model)](#9-resistance-engine-strain-model)
- [10. Analyzers & plotting](#10-analyzers--plotting)
- [11. Sim wrapper & infrastructure](#11-sim-wrapper--infrastructure)
- [12. Example / scenario scripts](#12-example--scenario-scripts)
- [Summary matrix (by combination type)](#summary-matrix-by-combination-type)
- [Recommended wiring for a full resistance sim](#recommended-wiring-for-a-full-resistance-sim)
- [Bottom line](#bottom-line)
- [Related docs](#related-docs)

## Legend

| Status | Meaning |
|--------|---------|
| **Native** | Resistance-specific module or hook exists |
| **Compatible** | Works together; no strain-specific code needed |
| **Compatible†** | Works, but resistance changes outcomes or needs extra wiring |
| **Requires swap** | Base module ignores strains; use strain-aware counterpart |
| **N/A** | No meaningful combination (different paradigm or no interaction) |
| **Gap** | Known limitation or not yet implemented |

**Validation:** Tested = covered in `tests/test_resistance*.py`; Partial = examples only; Untested = no resistance-specific tests found.

---

## 1. Core disease models

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `TB` (base) | `tb.py` | **Compatible** | Comparator arm only; no strain state | Use as `no_resistance` baseline | Tested |
| `MultiStrainTB` | `resistance/multistrain_tb.py` | **Native** | Strain overlay on same `TBS` states | `strains=`, optional `p_multi`, `p_random_acquisition`, `alpha_*` | Tested |
| `TBS` state machine | `tb.py` | **Compatible†** | Agent-level NH unchanged; strains ride alongside | Progression/clearance hooks wipe or retain strains | Tested |
| `get_tb()` | `tb.py` | **Compatible** | Returns `MultiStrainTB` when configured | — | Tested |
| `TB_ODE` / `TB_SS` | `compartmental/` | **N/A** | Compartmental; no per-agent strain state | Cannot combine without new model | N/A |
| `choice2d` | `tb.py` | **Compatible** | Used by products; strain logic is elsewhere | — | Partial |

---

## 2. Transmission & networks

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `ResistanceConnector` | `resistance/connector.py` | **Native** | Applies max-strain fitness to `rel_trans` | Must add to `sim.connectors` | Tested |
| `ss.RandomNet` (default) | via `Sim` | **Compatible** | Transmission strain assignment in `set_prognoses` | Standard | Tested |
| `ss.HouseholdNet` | Starsim | **Compatible†** | Household mixing unchanged; strain passed like any infection | Same connector + `MultiStrainTB` | Partial (household TPT tested, not strain+HCT) |
| Superinfection (`alpha_super`, `alpha_act`) | `resistance/multistrain_tb.py` | **Native** | Gates re-infection susceptibility by state | Config on `MultiStrainTB` | Tested |
| Fitness-weighted strain pick | `strains.py` (`AgentStrains`) | **Native** | Multinomial transmission of one strain | Auto with connector | Tested |
| Duplicate-strain blocking | `resistance/multistrain_tb.py` | **Native** | Same strain cannot superinfect twice | Auto | Tested |
| `BetaByYear` | `interventions/beta.py` | **Compatible†** | Scales `beta` globally; resistance fitness is multiplicative on top | Independent params | Untested |
| `plot_household` / `plot_household_structure` | `networks.py`, `plots.py` | **Compatible** | Visualization only | — | Untested |

---

## 3. Diagnostics

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `Dx` / `Xpert` / `OralSwab` / `FujiLAM` / `CAD` | `diagnostics.py` | **Compatible†** | Agent-level TB diagnosis; no drug resistance phenotype | Upstream of DST in cascade | Tested (integration) |
| `DxDelivery` | `diagnostics.py` | **Compatible†** | Sets `diagnosed`; gates DST and Tx eligibility | HSB → confirm → DST → Tx | Tested |
| `DSTDx` | `resistance/diagnostics.py` | **Native** | Per-drug observed resistance (sens/spec, `p_strain_obs`, `p_sample`, `p_culture`) | `catalog` + drug list | Tested |
| `DSTDelivery` | `resistance/diagnostics.py` | **Native** | Delivers DST; writes `observed_*_resistant` | After `DxDelivery` | Tested |
| `RegimenRouter` | `resistance/diagnostics.py` | **Native** | Routes Tx by observed phenotype | `StrainAwareTxDelivery` + `copy_inputs=False` if shared refs | Tested |
| `treatment_monitoring_eligibility` | `resistance/diagnostics.py` | **Native** | Gates monitoring Dx by time on Tx | Named `StrainAwareTxDelivery` + monitor `DxDelivery` | Tested |
| FujiLAM + HIV stratification | `diagnostics.py` | **Compatible†** | HIV affects Dx sens/spec, not strain assignment | HIV + `MultiStrainTB` independent | Untested |
| Monitoring → regimen switch | `resistance/tx.py` | **Native** | `cancel_delivery=` on `StrainAwareTxDelivery` cancels superseded in-flight course | Wire switch delivery with `cancel_delivery='first_line'` | Tested |

---

## 4. Treatment

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `Tx` / `TxMulti` / `DOTS` / presets | `treatments.py` | **Requires swap** | Agent-level cure; clears whole infection | Use `StrainAwareTx` + `Regimen` | N/A (by design) |
| `TxDelivery` | `treatments.py` | **Requires swap** | No per-strain failure/acquisition | Use `StrainAwareTxDelivery` | N/A |
| `StrainAwareTx` | `resistance/tx.py` | **Native** | Per-strain ψ, adherence, selective acquisition on failure/relapse | `Regimen` + `catalog` | Tested |
| `StrainAwareTxDelivery` | `resistance/tx.py` | **Native** | Latent partial clear, active pre-roll, relapse strain restore | DST-routed eligibility optional | Tested |
| `Regimen` | `resistance/regimens.py` | **Native** | Min-effective-drug cure model (`max` / `parallel`) | Drug list must match catalog | Tested |
| `Regimen(resistance_penalty=...)` | `resistance/regimens.py` | **Native** | Reduced-but-nonzero per-drug efficacy against resistant strains | Optional per-drug multiplier in [0, 1] | Tested |
| `drug_params` presets | `treatments.py` | **Compatible†** | Reference data only; wire into `Regimen` manually | Map to `StrainAwareTx` | Partial |
| `HealthSeekingBehavior` | `health_seeking.py` | **Compatible** | Care-seeking unchanged; enables Dx cascade | Standard | Tested (integration) |
| Latent Tx selective acquisition | `resistance/tx.py` | **Native** | Latent clears susceptible strains; ω on residual resistant carriers | `p_selective_acquisition` on `StrainAwareTx` | Tested |
| Overlapping Tx courses | `resistance/tx.py` | **Native** | `cancel_delivery=` clears prior pending course before starting new regimen | Set on switch `StrainAwareTxDelivery` | Tested |

---

## 5. TPT (preventive therapy)

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `TPTTx` / `TPTSimple` | `tpt.py` | **Requires swap** | Agent-level sterilize/suppress | Product = `StrainAwareTPTTx` | N/A |
| `StrainAwareTPTTx` | `resistance/tpt.py` | **Native** | Per-strain sterilization + TPT-driven acquisition | `Regimen` + `p_tpt_acquisition` | Tested |
| `TPTHousehold` | `tpt.py` | **Compatible†** | Household tracing unchanged; TPT product must be strain-aware | `StrainAwareTPTTx` inside | Untested |
| `HouseholdContactTracing` | `tpt.py` | **Compatible†** | Flags contacts; downstream TPT/Dx strain-agnostic unless swapped | Strain-aware products downstream | Untested |
| `TPTDelivery` | `tpt.py` | **Compatible†** | Delivery layer OK if product is `StrainAwareTPTTx` | — | Partial |
| `_apply_neither_branch` hook | `tpt.py` | **Native** | Base hook; `StrainAwareTPTTx` runs acquisition on ineffective TPT | Must keep hook in base `TPTTx` | Tested |
| Per-strain TPT suppression (`rr_*`) | `resistance/tpt.py` | **Native** | `rr_*` scaled by fraction of carried strains covered by regimen | `StrainAwareTPTTx.apply_protection` | Tested |

---

## 6. Vaccination & prevention

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `BCGVx` / `BCGRoutine` | `bcg.py` | **Compatible†** | Modifies agent-level `rr_*` / susceptibility; **does not assign or clear strains** | Independent of `AgentStrains` overlay | Untested |
| `BetaByYear` | `beta.py` | **Compatible†** | Global transmission modifier | Stacks with fitness via connector | Untested |

---

## 7. Comorbidities

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `HIV` / `HIVState` | `comorbidities/hiv.py` | **Compatible†** | HIV progression independent; affects Dx (FujiLAM) and TB activation via connector | `TB_HIV_Connector` + `MultiStrainTB` | Untested |
| `HivInterventions` | `comorbidities/hiv.py` | **Compatible†** | Prevalence/ART targets; no strain interaction | — | Untested |
| `TB_HIV_Connector` | `comorbidities/hiv.py` | **Compatible†** | Multiplies TB activation risk; strains unaffected | Both connectors in `sim.connectors` | Untested |
| `Malnutrition` | `comorbidities/malnutrition.py` | **Compatible†** | Anthropometrics; TB RR modifiers at agent level | `TB_Nutrition_Connector` | Untested |
| `TB_Nutrition_Connector` | `comorbidities/malnutrition.py` | **Compatible†** | Activation/clearance/susceptibility modifiers; not strain-specific | — | Untested |

---

## 8. Demographics & migration

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `ss.Births` / `ss.Deaths` (default) | via `Sim` | **Compatible†** | Death clears strains (`step_die`); births start susceptible | Auto on `MultiStrainTB` | Partial |
| `Migration` | `migration.py` | **Compatible†** | Assigns TB state to immigrants; seeds strains via `MultiStrainTB.seed_strains` | Auto when TB is `MultiStrainTB` | Tested |
| `HouseholdNet` + `Migration` | `migration.py` | **Compatible†** | Household assignment for migrants; strain-agnostic | — | Untested |

---

## 9. Resistance engine (strain model)

All three strain data-model classes live in `tbsim/resistance/strains.py`
(see [resistance_architecture.md §1.1](resistance_architecture.md#11-strainspec-vs-straincatalog-vs-agentstrains)).
The classes are intentionally TB-focused in this branch and use one naming
path (`resistance`, `drugs`) for lower API complexity.

| Class | Role |
|-------|------|
| `StrainSpec` | One strain's blueprint (resistance bits, fitness, `init_prev`) — pure config. |
| `StrainCatalog` | Catalog of all specs as indexed numpy tables for fast lookup (`drugs`, `uids`, `resistance`, `fitness`, `init_prev`). |
| `AgentStrains` | Per-agent runtime state (`carries_<uid>` `ss.BoolArr` on `MultiStrainTB`). Disease-agnostic. |

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `StrainSpec` / `StrainCatalog` | `resistance/strains.py` | **Native** | TB strain catalog (`resistance`/`drugs`) | Required | Tested |
| `AgentStrains` | `resistance/strains.py` | **Native** | Per-agent `carries_*` flags | Auto on `MultiStrainTB` | Tested |
| Bitmask prototype | Historical prototype | **N/A** | Reference/prototype representation only; not runtime state in this branch | Use named `AgentStrains` BoolArrs instead | Tested by design guard |
| `ProgressionResolver` | `resistance/resolvers.py` | **Native** | Bottleneck at activation (`p_multi`) | `progression_mode='bottleneck'` | Tested |
| `AcquisitionResolver` | `resistance/resolvers.py` | **Native** | Random (add) + selective (replace) acquisition | `p_random_acquisition` / Tx/TPT ω | Tested |
| `StrainResults` | `resistance/analyzers.py` | **Native** | Per-strain prevalence/incidence | Analyzer on sim | Tested |
| `DuplicateStrainAnalyzer` | `resistance/analyzers.py` | **Native** | Blocked duplicate superinfection count | Optional analyzer | Tested |
| Per-step acquisition counters | `resistance/multistrain_tb.py` | **Native** | `_n_denovo_*`, `_n_txacq_*`, `_n_transmitted_*` counters incremented by resolvers, Tx, TPT, and transmission | Auto on `MultiStrainTB`; consumed by `ResistanceStats` | Tested |
| `ResistanceStats` ODE summary | `resistance/analyzers.py` | **Native** | ODE-facing `frac_resist`, `frac_super`, and origin fluxes (de-novo / tx-acquired / transmitted) | Optional analyzer on `MultiStrainTB` sims | Tested |
| Zero-fitness transmission | `connector.py` | **Native** | Carriers with only `fitness=0` strains get multiplier 0; legacy no-strain agents unchanged | `ResistanceConnector` + `carries_any` | Tested |
| `init_prev` fallback | `resistance/multistrain_tb.py` | **Native** | Unresolved source + zero weights → no strain assigned + warning | `seed_strains` / transmission fallback | Tested |
| Missing `ResistanceConnector` | `resistance/multistrain_tb.py` | **Native** | `MultiStrainTB.init_post` warns if connector absent | Add `ResistanceConnector()` | Tested |

---

## 10. Analyzers & plotting

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `DwellTime` / aliases | `analyzers.py` | **Compatible** | Tracks `TBS` dwell times; strain-agnostic | Add `StrainResults` for strain metrics | Untested |
| `HouseholdStats` | `analyzers.py` | **Compatible** | Household mixing stats; no strain dimension | — | Untested |
| `tbsim.plot()` / `Sim.plot()` | `plots.py`, `sim.py` | **Compatible†** | Default panels are agent-level TB; strain plots need custom/`StrainResults` | Example scripts plot strain series | Partial |
| `StrainResults` time series | `resistance/analyzers.py` | **Native** | Per-strain carriers/active/new | Required for strain dashboards | Tested |
| `TwoStrainODE` reference | `compartmental/two_strain_ode.py` | **Native** | Deterministic two-strain ABM-vs-ODE validation target | Translate old helper/scenario scripts as needed | Tested |

---

## 11. Sim wrapper & infrastructure

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `tbsim.Sim` | `sim.py` | **Compatible†** | Routes params; `get_dx()`, `get_tb()` for cascades | `tb_model=MultiStrainTB(...)`, `connectors`, `copy_inputs=False` for DST router | Tested |
| `ResistanceSim` | `resistance/sim.py` | **Native** | One-liner wrapper: strains + connector + optional cascade/analyzers; spec matrix via `build_spec_sim()` | `from tbsim.resistance import ResistanceSim, build_spec_sim` | Tested |
| `ss.Sim` (raw) | Starsim | **Compatible†** | Works but loses `get_dx()` helpers | Prefer `tbsim.Sim` | Partial |
| `ss.parallel()` / `MultiSim` | Starsim | **Compatible†** | Example scripts use parallel scenarios; analyzer lookup needs name-based `_find` | See `run_resistance_program_demo.py` | Tested |
| `demo()` | `sim.py` | **Compatible** | Plain `TB` only; no resistance | — | N/A |
| `ProductMulti` / `TBProductRoutine` | `interventions/` | **Compatible†** | Generic delivery; strain logic in products | — | Partial |
| Resistance namespace | `tbsim.resistance` | **Native** | Not re-exported on `tbsim`; import explicitly from subpackage. Acyclic: `resistance/*` uses `from ..tb import …`, never `import tbsim`. | `from tbsim.resistance import MultiStrainTB` | Tested |

---

## 12. Example / scenario scripts

| Script | Combine? | Role vs resistance | Validation |
|--------|----------|-------------------|------------|
| `run_resistance.py` | **Native** | Minimal disease-stack demo (strains→MultiStrainTB→connector→Tx/TPT/DST→analyzers) | Smoke tested |
| `run_resistance.py` | **Native** | INH/BDQ TPT trade-off (5 strains) | Smoke tested |
| `run_resistance2.py` | **Native** | Spec sensitivity matrix + DST routing | Tested (directional) |
| `run_resistance_program_demo.py` | **Native** | Full RIF/BDQ/FQ program arms | Manual smoke |
| `run_resistance_critical_paths.py` | **Native** | 23 path scenarios vs baseline | Manual smoke |
| Other `tbsim_examples/*` | **Requires swap** | Use base `TB` + base Tx unless adapted | Not validated with resistance |

---

## Summary matrix (by combination type)

| Category | Count | Features |
|----------|-------|----------|
| **Native / strain-aware** | 25 | `MultiStrainTB`, connector, `strains.py` (Spec/Registry/Profile with generic phenotype), resolvers, regimen (+`resistance_penalty`), strain Tx/TPT/DST (incl. cancel/switch, latent ω, per-strain suppress, lab dropout), router, monitoring helper, per-step acquisition counters, strain analyzers |
| **Compatible (no code change)** | 12 | Plain `TB` comparator, `get_tb`, HSB, base plotting, `demo`, explicit `tbsim.resistance` imports, `choice2d`, etc. |
| **Compatible† (works, behavior changes)** | 22 | All base Dx products, networks, comorbidities, BCG, beta, migration, births/deaths, `tbsim.Sim`, parallel runs |
| **Requires swap to strain-aware** | 6 | `Tx`/`TxDelivery`, `TPTTx` alone, `DOTS`/presets, compartmental N/A |

---

## Recommended wiring for a full resistance sim

```text
Disease:     MultiStrainTB(strains=[...], p_multi=..., p_random_acquisition=...)
Connector:   ResistanceConnector()
Cascade:     HealthSeekingBehavior → DxDelivery(Xpert) → DSTDelivery(DSTDx)
             → StrainAwareTxDelivery (× tiers via RegimenRouter)
             → TPTSimple(StrainAwareTPTTx)   [optional]
Analyzers:   StrainResults(), DuplicateStrainAnalyzer()   [optional]
Comparator:  TB() on parallel arm (no connector, no strain analyzers)
```

---

## Bottom line

- **Most TBsim features can be combined with resistance** at the agent/TB-state level without new code.
- **Treatment, TPT, and DST must use strain-aware variants** (`StrainAwareTx`, `StrainAwareTPTTx`, `DSTDx`/`DSTDelivery`) for resistance to matter clinically.
- **Comorbidities, BCG, migration, household networks, and base diagnostics** are compatible but **largely untested** with the overlay; they modify agent-level risk or care pathways, not strain identity.
- **Compartmental models (`TB_ODE`/`TB_SS`) cannot combine** with the current resistance implementation.
- **Previously documented engine gaps are now closed**: in-flight Tx cancel via `cancel_delivery`, per-strain TPT suppression weighting, latent Tx selective acquisition, and multi-stage DST dropout (`p_sample` / `p_culture` / `p_strain_obs`).

---

## Related docs

- [resistance_architecture.md](resistance_architecture.md) — implementation architecture (§1.1: StrainSpec / StrainCatalog / AgentStrains)
- [resistance_step_by_step_guide.md](resistance_step_by_step_guide.md) — step-by-step behavior guide and coverage status
