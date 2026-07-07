# TBsim × Drug Resistance — Cross-Feature Mapping

This document lists TBsim features and validates whether each can be combined with or is impacted by the drug resistance implementation (`MultiStrainTB` + `tbsim/resistance/`).

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
| `DSTDx` | `resistance/diagnostics.py` | **Native** | Per-drug observed resistance (sens/spec, `p_strain_obs`) | `catalog` + drug list | Tested |
| `DSTDelivery` | `resistance/diagnostics.py` | **Native** | Delivers DST; writes `observed_*_resistant` | After `DxDelivery` | Tested |
| `RegimenRouter` | `resistance/diagnostics.py` | **Native** | Routes Tx by observed phenotype | `StrainAwareTxDelivery` + `copy_inputs=False` if shared refs | Tested |
| `treatment_monitoring_eligibility` | `resistance/diagnostics.py` | **Native** | Gates monitoring Dx by time on Tx | Named `StrainAwareTxDelivery` + monitor `DxDelivery` | Tested |
| FujiLAM + HIV stratification | `diagnostics.py` | **Compatible†** | HIV affects Dx sens/spec, not strain assignment | HIV + `MultiStrainTB` independent | Untested |
| Monitoring → regimen switch | — | **Gap** | Monitoring selects agents; **cannot cancel in-flight Tx** | Second `StrainAwareTxDelivery` only | Partial |

---

## 4. Treatment

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `Tx` / `TxMulti` / `DOTS` / presets | `treatments.py` | **Requires swap** | Agent-level cure; clears whole infection | Use `StrainAwareTx` + `Regimen` | N/A (by design) |
| `TxDelivery` | `treatments.py` | **Requires swap** | No per-strain failure/acquisition | Use `StrainAwareTxDelivery` | N/A |
| `StrainAwareTx` | `resistance/tx.py` | **Native** | Per-strain ψ, adherence, selective acquisition on failure/relapse | `Regimen` + `catalog` | Tested |
| `StrainAwareTxDelivery` | `resistance/tx.py` | **Native** | Latent partial clear, active pre-roll, relapse strain restore | DST-routed eligibility optional | Tested |
| `Regimen` | `resistance/regimens.py` | **Native** | Min-effective-drug cure model (`max` / `parallel`) | Drug list must match catalog | Tested |
| `drug_params` presets | `treatments.py` | **Compatible†** | Reference data only; wire into `Regimen` manually | Map to `StrainAwareTx` | Partial |
| `HealthSeekingBehavior` | `health_seeking.py` | **Compatible** | Care-seeking unchanged; enables Dx cascade | Standard | Tested (integration) |
| Latent Tx selective acquisition | `resistance/tx.py` | **Gap** | Latent clears susceptible strains only; ω on latent failure deferred | — | Documented |
| Overlapping Tx courses | — | **Gap** | New regimen can start while prior course still scheduled | — | Documented |

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
| Per-strain TPT suppression (`rr_*`) | `resistance/tpt.py` | **Gap** | Only agent-level suppression implemented | — | Documented |

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
(see [resistance_architecture.md §1.1](resistance_architecture.md#11-strainspec-vs-straincatalog-vs-agentstrains)):

| Class | Role |
|-------|------|
| `StrainSpec` | One strain's blueprint (resistance bits, fitness, `init_prev`) — pure config |
| `StrainCatalog` | Catalog of all specs as indexed numpy tables for fast lookup |
| `AgentStrains` | Per-agent runtime state (`carries_<uid>` `ss.BoolArr` on `MultiStrainTB`) |

| Feature | File | Combine? | Resistance impact | Required wiring | Validation |
|---------|------|----------|-------------------|-----------------|------------|
| `StrainSpec` / `StrainCatalog` | `resistance/strains.py` | **Native** | Strain catalog | Required | Tested |
| `AgentStrains` | `resistance/strains.py` | **Native** | Per-agent `carries_*` flags | Auto on `MultiStrainTB` | Tested |
| Bitmask prototype | Historical prototype | **N/A** | Reference/prototype representation only; not runtime state in this branch | Use named `AgentStrains` BoolArrs instead | Tested by design guard |
| `ProgressionResolver` | `resistance/resolvers.py` | **Native** | Bottleneck at activation (`p_multi`) | `progression_mode='bottleneck'` | Tested |
| `AcquisitionResolver` | `resistance/resolvers.py` | **Native** | Random (add) + selective (replace) acquisition | `p_random_acquisition` / Tx/TPT ω | Tested |
| `StrainResults` | `resistance/analyzers.py` | **Native** | Per-strain prevalence/incidence | Analyzer on sim | Tested |
| `DuplicateStrainAnalyzer` | `resistance/analyzers.py` | **Native** | Blocked duplicate superinfection count | Optional analyzer | Tested |
| `ResistanceStats` ODE summary | `resistance/analyzers.py` | **Native** | ODE-facing `frac_resist`, `frac_super`, and origin fluxes | Optional analyzer on `MultiStrainTB` sims | Tested |
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
| `ss.Sim` (raw) | Starsim | **Compatible†** | Works but loses `get_dx()` helpers | Prefer `tbsim.Sim` | Partial |
| `ss.parallel()` / `MultiSim` | Starsim | **Compatible†** | Example scripts use parallel scenarios; analyzer lookup needs name-based `_find` | See `run_resistance_demo.py` | Tested |
| `demo()` | `sim.py` | **Compatible** | Plain `TB` only; no resistance | — | N/A |
| `ProductMulti` / `TBProductRoutine` | `interventions/` | **Compatible†** | Generic delivery; strain logic in products | — | Partial |
| Resistance namespace | `tbsim.resistance` | **Native** | Not re-exported on `tbsim`; import explicitly from subpackage. Acyclic: `resistance/*` uses `from ..tb import …`, never `import tbsim`. | `from tbsim.resistance import MultiStrainTB` | Tested |

---

## 12. Example / scenario scripts

| Script | Combine? | Role vs resistance | Validation |
|--------|----------|-------------------|------------|
| `run_resistance.py` | **Native** | INH/BDQ TPT trade-off (5 strains) | Smoke tested |
| `run_resistance2.py` | **Native** | Spec sensitivity matrix + DST routing | Tested (directional) |
| `run_resistance_demo.py` | **Native** | Full RIF/BDQ/FQ program arms | Manual smoke |
| `run_resistance_critical_paths.py` | **Native** | 23 path scenarios vs baseline | Manual smoke |
| Other `tbsim_examples/*` | **Requires swap** | Use base `TB` + base Tx unless adapted | Not validated with resistance |

---

## Summary matrix (by combination type)

| Category | Count | Features |
|----------|-------|----------|
| **Native / strain-aware** | 18 | `MultiStrainTB`, connector, `strains.py` (Spec/Registry/Profile), resolvers, regimen, strain Tx/TPT/DST, router, monitoring helper, strain analyzers |
| **Compatible (no code change)** | 12 | Plain `TB` comparator, `get_tb`, HSB, base plotting, `demo`, explicit `tbsim.resistance` imports, `choice2d`, etc. |
| **Compatible† (works, behavior changes)** | 22 | All base Dx products, networks, comorbidities, BCG, beta, migration, births/deaths, `tbsim.Sim`, parallel runs |
| **Requires swap to strain-aware** | 6 | `Tx`/`TxDelivery`, `TPTTx` alone, `DOTS`/presets, compartmental N/A |
| **Known gaps** | 4 | In-flight Tx cancel, per-strain TPT suppress, latent Tx ω, DST lab pipeline |

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
- **Four documented gaps** remain where spec intent and engine behavior may diverge (in-flight Tx cancel, per-strain TPT suppression, latent Tx selective acquisition, richer DST dropout).

---

## Related docs

- [resistance_architecture.md](resistance_architecture.md) — implementation architecture (§1.1: StrainSpec / StrainCatalog / AgentStrains)
- [resistance_spec_pseudocode.md](resistance_spec_pseudocode.md) — spec-aligned pseudocode and coverage status
