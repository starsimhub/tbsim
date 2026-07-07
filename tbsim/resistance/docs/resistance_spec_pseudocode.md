# TBsim Resistance Spec Plain-English Pseudocode

This document translates the resistance technical specification into English-only pseudocode. It is intended to answer two questions:

- What needs to happen in the model at each critical path?
- Where should that behavior be covered: engine, example scenario, analyzer, or test?

The model keeps one TB natural-history state per person. Drug resistance is an overlay beside that state. A person can carry zero, one, or multiple resistance profiles. The word "strain" below means a modeled resistance profile, such as pan-susceptible, RIF-resistant, BDQ-resistant, or RIF plus FQ resistant.

## Table of contents

- [Strain data model (`tbsim/resistance/strains.py`)](#strain-data-model-tbsimresistancestrainspy)
- [Coverage Summary](#coverage-summary)
- [Critical Path 1: Build The Resistance Catalog](#critical-path-1-build-the-resistance-catalog)
- [Critical Path 2: Initialize Person-Level Strain State](#critical-path-2-initialize-person-level-strain-state)
- [Critical Path 3: Transmission From A Single-Strain Infector](#critical-path-3-transmission-from-a-single-strain-infector)
- [Critical Path 4: Transmission From A Superinfected Infector](#critical-path-4-transmission-from-a-superinfected-infector)
- [Critical Path 5: Recipient Is Fully Susceptible](#critical-path-5-recipient-is-fully-susceptible)
- [Critical Path 6: Recipient Is Already Infected And Eligible For Superinfection](#critical-path-6-recipient-is-already-infected-and-eligible-for-superinfection)
- [Critical Path 7: Duplicate-Strain Superinfection Attempt](#critical-path-7-duplicate-strain-superinfection-attempt)
- [Critical Path 8: Progression From Early Infection To Non-Infectious TB](#critical-path-8-progression-from-early-infection-to-non-infectious-tb)
- [Critical Path 9: Progression From Early Or Non-Infectious TB To Active TB](#critical-path-9-progression-from-early-or-non-infectious-tb-to-active-tb)
- [Critical Path 10: Natural Clearance And Death](#critical-path-10-natural-clearance-and-death)
- [Critical Path 11: Random Endogenous Acquisition](#critical-path-11-random-endogenous-acquisition)
- [Critical Path 12: Treatment Eligibility](#critical-path-12-treatment-eligibility)
- [Critical Path 13: Treatment Product Effect](#critical-path-13-treatment-product-effect)
- [Critical Path 14: Treatment Success Resolution](#critical-path-14-treatment-success-resolution)
- [Critical Path 15: Treatment Failure Resolution And Selective Acquisition](#critical-path-15-treatment-failure-resolution-and-selective-acquisition)
- [Critical Path 16: TPT Eligibility](#critical-path-16-tpt-eligibility)
- [Critical Path 17: TPT Sterilization And Partial Clearance](#critical-path-17-tpt-sterilization-and-partial-clearance)
- [Critical Path 18: TPT Suppression Or Ineffective TPT](#critical-path-18-tpt-suppression-or-ineffective-tpt)
- [Critical Path 19: DST Sampling](#critical-path-19-dst-sampling)
- [Critical Path 20: DST-Based Regimen Routing](#critical-path-20-dst-based-regimen-routing)
- [Critical Path 21: Treatment Monitoring](#critical-path-21-treatment-monitoring)
- [Critical Path 22: Results And Analyzers](#critical-path-22-results-and-analyzers)
- [Critical Path 23: Baseline And Intervention Scenarios](#critical-path-23-baseline-and-intervention-scenarios)
- [Critical Path 24: Validation And Regression Testing](#critical-path-24-validation-and-regression-testing)
- [Critical Path 25: Known Residual Items](#critical-path-25-known-residual-items)
- [Recommended Next Example Additions](#recommended-next-example-additions)
- [Bottom Line](#bottom-line)

## Strain data model (`tbsim/resistance/strains.py`)

Three classes sit at different layers. See
[resistance_architecture.md §1.1](resistance_architecture.md#11-strainspec-vs-straincatalog-vs-agentstrains)
for the full write-up.

| | **StrainSpec** | **StrainCatalog** | **AgentStrains** |
|---|---|---|---|
| **What it is** | One strain's definition | The full strain catalog | Per-agent "who carries which strain" state |
| **Scope** | Single strain | All strains in the model | All agents in the simulation |
| **Has sim state?** | No — pure config | No — lookup tables | Yes — `ss.BoolArr` on `MultiStrainTB` |
| **Typical use** | `StrainSpec('pan', {'INH': 0, 'RIF': 0})` | `StrainCatalog([pan, inh_r, …])` | `agent_strains.add_strain(uids, 'pan')` |

```
StrainSpec (×N)  →  StrainCatalog  →  AgentStrains (on MultiStrainTB)
   "what strains        "catalog for        "agent 42 carries
    exist?"              fast lookup"         pan + inh_r"
```

Critical-path mapping:

| Critical path (below) | Primary class(es) |
|-----------------------|-------------------|
| 1 — Build the resistance catalog | `StrainSpec`, `StrainCatalog` |
| 2 — Initialize person-level strain state | `AgentStrains` (+ `MultiStrainTB`) |
| 3–6 — Transmission | `AgentStrains`, `ResistanceConnector` |
| 7–9 — Progression / clearance / random acquisition | `AgentStrains`, resolvers |
| 10–14 — Treatment / TPT / DST | `Regimen`, products, `AgentStrains` |

## Coverage Summary

The full spec is larger than one example script. Complete coverage should be split across three layers:

- Engine coverage: the reusable model behavior in the resistance overlay.
- Example coverage: scenario scripts that demonstrate how to configure and compare baseline and intervention arms.
- Test coverage: deterministic checks for the spec's quantitative rules and regression checks for scenario-level behavior.

Current engine and test coverage is broad. The critical-path example now includes baseline, intervention, sensitivity, no-resistance comparator, and optional multi-seed uncertainty workflows. Treatment monitoring is represented as a scheduled diagnostic pathway; full cancellation of an in-flight treatment course remains a future engine feature.

## Critical Path 1: Build The Resistance Catalog

Plain-English pseudocode:

1. Choose the drug or drug-class list for the model.
2. For every strain that will be represented, give it a short name.
3. For each strain, record whether it is susceptible or resistant to each drug.
4. For each strain, record its transmission fitness.
5. For each strain, record how common it is at initialization.
6. Validate that every profile has the same drug list.
7. Validate that strain names are unique.
8. Store the profiles in a catalog so all later logic can ask:
   - Which profiles exist?
   - Which drugs are modeled?
   - Is this profile resistant to this drug?
   - What is this profile's fitness?
   - If this profile gains resistance to a drug, which profile does it become?

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered for RIF, BDQ, and FQ in the South Africa-style example.
- Covered for alternative drug sets in tests and other resistance examples.
- Engine: `StrainSpec` + `StrainCatalog` in `tbsim/resistance/strains.py`.

## Critical Path 2: Initialize Person-Level Strain State

Plain-English pseudocode:

1. Give every person a yes-or-no flag for every resistance profile.
2. When the simulation starts, identify people who begin with TB infection.
3. For each initially infected person, assign one or more resistance profiles according to the configured initial prevalences.
4. If a person has no TB infection, make sure they carry no resistance profile.
5. Keep these profile flags attached to `MultiStrainTB` so births, deaths, and state changes stay synchronized with the rest of Starsim.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered by the resistance overlay and by the example strain catalog.
- Engine: `AgentStrains` in `tbsim/resistance/strains.py` (one `ss.BoolArr`
  per strain on `MultiStrainTB`).

## Critical Path 3: Transmission From A Single-Strain Infector

Plain-English pseudocode:

1. When a person can transmit TB, look at the profile they carry.
2. Start from the normal TB transmission probability.
3. Multiply that probability by the carried profile's fitness.
4. If transmission occurs, pass that same profile to the recipient.
5. Do not create new resistance during transmission.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered by the resistance connector and `AgentStrains` transmission logic.
- Demonstrated indirectly in examples through fitness-cost scenarios.

## Critical Path 4: Transmission From A Superinfected Infector

Plain-English pseudocode:

1. When a person carries more than one strain, look up the fitness of every carried profile.
2. Use the fittest carried profile to set the person's total transmission probability.
3. If no transmission occurs, stop.
4. If transmission occurs, choose exactly one carried profile to pass onward.
5. Choose the transmitted profile with probabilities proportional to the fitness of the carried profiles.
6. Pass only the selected profile to the recipient.
7. Do not pass the entire mixed infection in one transmission event.
8. Do not create new resistance during transmission.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered by engine logic and spec example tests.
- Demonstrated by superinfection tracking in the scenario example, but the current example may not produce many superinfected agents in one seed.

## Critical Path 5: Recipient Is Fully Susceptible

Plain-English pseudocode:

1. If the recipient has no current TB infection and receives a transmitted profile, mark the person as infected.
2. Set the person's TB state to early infection.
3. Record the infection time in the usual TB state machine.
4. Add the transmitted profile to the person's strain flags.
5. Leave all other resistance profiles absent.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered.

## Critical Path 6: Recipient Is Already Infected And Eligible For Superinfection

Plain-English pseudocode:

1. If the recipient already has TB infection, decide whether their current TB state allows superinfection.
2. If the recipient is in early infection, apply the early-infection superinfection risk multiplier.
3. If the recipient is in non-infectious TB, apply the non-infectious superinfection risk multiplier.
4. If the recipient is asymptomatic active TB, apply the asymptomatic superinfection risk multiplier.
5. If the recipient is symptomatic active TB, apply the symptomatic superinfection risk multiplier.
6. If the recipient is on treatment or cleared, do not treat them as eligible for superinfection through this path.
7. If the multiplier is zero, block the superinfection.
8. If the multiplier is greater than zero and the transmission draw succeeds, continue to duplicate-profile checking.
9. If the transmitted profile is new to the recipient, add it to the person's carried profiles.
10. Preserve the person's current TB state and timers.
11. Do not reset their natural-history state just because they acquired a second profile.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered in engine and tests.
- Example coverage exists through superinfection tracking, but more explicit high-superinfection scenarios would make this easier to inspect.

## Critical Path 7: Duplicate-Strain Superinfection Attempt

Plain-English pseudocode:

1. After a transmission event selects a profile to pass, check whether the recipient already carries that exact profile.
2. If the recipient does not carry it, add the profile.
3. If the recipient already carries it, do not add a second copy.
4. Count the blocked duplicate attempt.
5. Store both per-step and cumulative duplicate-block counts.
6. Use those counts to judge whether blocking repeated same-profile exposures may bias the model toward rare resistant strains.

Coverage target:

- Engine: required.
- Analyzer: required.
- Examples: required.
- Tests: required.

Current status:

- Covered by the duplicate-profile analyzer and shown in scenario summaries.

## Critical Path 8: Progression From Early Infection To Non-Infectious TB

Plain-English pseudocode:

1. When the base TB model moves a person from early infection to non-infectious TB, keep their current carried profiles.
2. Do not choose a single dominant profile at this transition.
3. Do not change the person's progression risk based on the number of carried profiles.
4. After the transition is chosen, run random acquisition checks if configured.

Coverage target:

- Engine: required.
- Tests: required.

Current status:

- Covered.

## Critical Path 9: Progression From Early Or Non-Infectious TB To Active TB

Plain-English pseudocode:

1. When the base TB model moves a person into active TB, first identify all profiles they carry.
2. If the person carries zero profiles, leave profile state unchanged and rely on tests to catch the inconsistency.
3. If the person carries one profile, that profile progresses.
4. If the person carries more than one profile, draw whether multi-profile active TB occurs.
5. If the multi-profile draw succeeds, keep all carried profiles.
6. If the multi-profile draw fails, choose one carried profile with equal probability.
7. Keep only the chosen profile and remove the others.
8. Do not use transmission fitness to choose the progressing profile unless a sensitivity analysis explicitly asks for it.
9. Do not change the underlying TB progression rate based on number of profiles.
10. If the person is leaving early infection, run random acquisition checks if configured.
11. If the person is leaving non-infectious TB for active TB, do not run random acquisition a second time unless the spec changes.

Coverage target:

- Engine: required.
- Examples: required for sensitivity.
- Tests: required.

Current status:

- Covered in engine and tests.
- Covered in the expanded example by the progression bottleneck sensitivity arm.

## Critical Path 10: Natural Clearance And Death

Plain-English pseudocode:

1. When a person naturally clears early infection, remove all carried profiles.
2. When a person spontaneously resolves non-infectious TB, remove all carried profiles.
3. When a person dies or is removed from the population, remove all carried profiles.
4. Do not make natural clearance selective by drug or profile in the default implementation.
5. Preserve the usual TB reinfection protection behavior after natural clearance.

Coverage target:

- Engine: required.
- Tests: required.

Current status:

- Covered.

## Critical Path 11: Random Endogenous Acquisition

Plain-English pseudocode:

1. Trigger this path only when a person leaves early infection for non-infectious TB or active TB.
2. For each carried profile, examine every modeled drug.
3. If the strain is already resistant to a drug, skip that drug.
4. If the strain is susceptible to a drug, draw whether random acquisition occurs for that drug.
5. If acquisition does not occur, leave the strain as is.
6. If acquisition occurs, find the profile that matches the original profile plus the new resistance bit.
7. If that target profile exists in the catalog, add it to the person.
8. Keep the original profile.
9. Allow more than one strain in the same person to acquire resistance during the same transition.
10. Allow more than one drug acquisition if the configured draws produce it.
11. Do not run this as a per-timestep process.

Coverage target:

- Engine: required.
- Examples: required for sensitivity.
- Tests: required.

Current status:

- Covered in engine and tests.
- Demonstrated in the high-acquisition-pressure example scenario.

## Critical Path 12: Treatment Eligibility

Plain-English pseudocode:

1. Use the normal health-seeking and diagnostic pathway to identify people eligible for treatment.
2. Require that the person is alive.
3. Require that the person has a diagnosis state appropriate for treatment.
4. Exclude people already on treatment unless a regimen-switch path explicitly allows them.
5. If a DST result is required, require that the person has been DST tested.
6. Route the person to the treatment regimen that matches the observed resistance phenotype.
7. If no resistant phenotype is observed, route to the default first-line regimen.
8. If a resistant phenotype is observed, route to the matching second-line or alternative regimen.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered in the scenario example through RIF-based first-line versus BPaL-like routing.
- Regimen router helper exists for more general observed-phenotype routing.

## Critical Path 13: Treatment Product Effect

Plain-English pseudocode:

1. When treatment starts, read every carried profile for each treated person.
2. Draw adherence once for each person and treatment episode.
3. Apply that same adherence result to all carried profiles for that person.
4. For each carried profile, calculate whether the regimen can cure it.
5. A strain susceptible to all regimen drugs receives full regimen efficacy.
6. A strain resistant to regimen drugs receives reduced or zero efficacy according to regimen settings.
7. Draw cure outcome separately for each carried profile.
8. If every carried profile is cured, mark the treatment episode as successful.
9. If at least one carried profile survives, mark the episode as unsuccessful.
10. Store which profiles are scheduled to be cleared when the treatment episode resolves.
11. Store relapse information for successfully treated people if relapse is configured.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered in engine and tests.
- Demonstrated in first-line and BPaL-like example regimens.

## Critical Path 14: Treatment Success Resolution

Plain-English pseudocode:

1. At treatment completion, identify people whose episode was successful.
2. Clear all remaining carried profiles for those people.
3. Move the person into the usual post-treatment TB state used by the base model.
4. Schedule relapse if the treatment product configured relapse.
5. If relapse later occurs, restore the saved carried profiles from the treatment start or success point as configured.

Coverage target:

- Engine: required.
- Tests: required.

Current status:

- Covered.

## Critical Path 15: Treatment Failure Resolution And Selective Acquisition

Plain-English pseudocode:

1. At treatment completion, identify people whose episode was unsuccessful.
2. Remove any carried profiles that were cured during the episode.
3. Keep profiles that were not cured.
4. Return the person to the TB state they had before treatment, unless the base model specifies another failure state.
5. For each regimen drug, check whether selective acquisition is configured.
6. For each surviving strain that is susceptible to that drug, draw whether selective acquisition occurs.
7. Adjust the acquisition probability by the person's TB state if state modifiers are configured.
8. If acquisition does not occur, leave the strain unchanged.
9. If acquisition occurs, find the profile that matches the original profile plus the new resistance bit.
10. Replace the original profile with the newly resistant profile.
11. Do not keep the original strain during selective acquisition.
12. Allow acquisition after relapse, because relapse is an unsuccessful treatment outcome.

Coverage target:

- Engine: required.
- Examples: required for sensitivity.
- Tests: required.

Current status:

- Covered in engine and tests.
- Demonstrated in high-acquisition-pressure scenarios.

## Critical Path 16: TPT Eligibility

Plain-English pseudocode:

1. Identify people eligible for preventive treatment using the configured TPT delivery rule.
2. Apply TPT coverage.
3. Start TPT for selected people at the configured time or campaign schedule.
4. Do not require the person to have active TB diagnosis unless the scenario deliberately models that.
5. Pass selected people to the strain-aware TPT product.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered in TPT scale-up examples.

## Critical Path 17: TPT Sterilization And Partial Clearance

Plain-English pseudocode:

1. For every person receiving TPT, read all carried profiles.
2. For each carried profile, decide whether the TPT regimen covers it.
3. A strain is covered if it is susceptible to the drugs in the regimen.
4. A strain is not covered if it is resistant to the drugs in the regimen.
5. If the TPT episode is on the sterilization path, first apply TPT-driven acquisition if configured.
6. Then remove covered profiles from people who are still in early infection.
7. Leave uncovered resistant strains in place.
8. If no profiles remain after removal, move the person to cleared.
9. If one or more profiles remain, keep the person infected and keep their current TB state.
10. For people not in early infection, apply acquisition logic as configured but do not force a cleared state through the latent sterilization pathway.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered in engine and demonstrated by TPT scenarios.

## Critical Path 18: TPT Suppression Or Ineffective TPT

Plain-English pseudocode:

1. If TPT suppresses rather than sterilizes, apply the usual agent-level TPT protection effects.
2. Keep carried profiles unless the regimen explicitly clears them.
3. If TPT is ineffective, leave carried profiles in place.
4. For ineffective TPT, run TPT-driven acquisition if configured.
5. Use state-specific acquisition modifiers:
   - very low for early infection,
   - intermediate for non-infectious TB,
   - high for asymptomatic active TB,
   - high for symptomatic active TB,
   - zero for treatment and cleared states.

Coverage target:

- Engine: required.
- Examples: required for sensitivity.
- Tests: required.

Current status:

- Covered in engine.
- Current example forces sterilization to make the resistance contrast visible. A non-sterilizing TPT sensitivity scenario should be added for complete example coverage.

## Critical Path 19: DST Sampling

Plain-English pseudocode:

1. Identify people eligible for DST.
2. Eligibility can be immediate after diagnosis, after treatment failure, or after treatment monitoring, depending on the scenario.
3. Apply DST coverage.
4. For each selected person, read all carried profiles.
5. For each carried profile, draw whether that strain is observed by the sample.
6. If strain observability is not explicitly configured, use profile fitness as the default observation probability.
7. For every observed strain, apply sensitivity and specificity for each drug.
8. Convert strain-level observations into one person-level observed phenotype.
9. If any observed strain is reported resistant to a drug, mark the person as observed resistant to that drug.
10. Do not expose which specific profile caused the resistant result.
11. Store the observed drug-level phenotype on the DST delivery state.

Coverage target:

- Engine: required.
- Examples: required.
- Tests: required.

Current status:

- Covered in engine and tests.
- Covered in the expanded example by the DST strain dropout sensitivity arm.

## Critical Path 20: DST-Based Regimen Routing

Plain-English pseudocode:

1. After DST results are available, build a list of people eligible for treatment.
2. For each candidate regimen, define the observed resistance pattern that qualifies a person for it.
3. For first-line treatment, select people whose observed phenotype does not show resistance to the key first-line marker.
4. For second-line treatment, select people whose observed phenotype shows the configured resistant marker.
5. Exclude people already on treatment unless the route is a regimen switch.
6. Start the matching treatment delivery.
7. If multiple regimens could match, apply them in a documented priority order.

Coverage target:

- Engine/helper: required.
- Examples: required.
- Tests: required.

Current status:

- Covered by direct RIF routing in the example.
- General helper exists for multi-drug phenotype routing.

## Critical Path 21: Treatment Monitoring

Plain-English pseudocode:

1. For every person currently on treatment, track when their treatment episode started.
2. At the configured monitoring time, identify people who have been on treatment long enough.
3. Optionally repeat monitoring at a configured interval.
4. Apply a diagnostic test to monitored people.
5. If the monitoring diagnostic says the person remains bacteriologically positive, mark them eligible for a follow-up action.
6. The follow-up action may be DST, regimen extension, or regimen change.
7. A regimen extension should be represented as a new treatment product.
8. A regimen change should be represented as another treatment delivery gated by the monitoring result.
9. If the new treatment is intended to replace an active treatment course, the model needs a cancellation or superseding rule for the old pending outcome.
10. If no cancellation rule is implemented, document that the new treatment starts while the previous scheduled outcome still exists.

Coverage target:

- Engine/helper: required.
- Examples: required for full spec demonstration.
- Tests: required.

Current status:

- Time-on-treatment eligibility helper is covered in engine and tests.
- Full premature in-flight regimen cancellation is not yet a built-in mechanism.
- Covered in the expanded example by the treatment monitoring pathway arm. In the default seed, monitoring is configured and counted, but produces zero positives.

## Critical Path 22: Results And Analyzers

Plain-English pseudocode:

1. At every timestep, count how many people carry each profile.
2. At every timestep, count how many active TB cases carry each profile.
3. At every timestep, count how many people newly acquired each profile.
4. At every timestep, count how many duplicate-profile superinfection attempts were blocked.
5. At every timestep, count how many people carry two or more profiles.
6. Summarize final active TB counts by profile.
7. Summarize the percentage of active TB that carries resistance to each drug.
8. Summarize cumulative duplicate blocks.
9. Summarize treatment starts, successes, failures, and relapse where available.
10. Summarize DST tests and observed resistant results where relevant.

Coverage target:

- Engine/analyzers: required.
- Examples: required.
- Tests: required.

Current status:

- Covered for per-profile counts, duplicate blocks, and superinfection in examples.
- Treatment and DST summaries can be expanded in the example output.

## Critical Path 23: Baseline And Intervention Scenarios

Plain-English pseudocode:

1. Define one baseline scenario that represents the current program.
2. Define at least one TPT scale-up scenario.
3. Define at least one DST scale-up or DST routing scenario.
4. Define at least one second-line regimen scenario, such as a BPaL-like arm.
5. Define at least one combined intervention scenario.
6. Define one high-acquisition-pressure sensitivity scenario.
7. Define one lower-fitness-cost sensitivity scenario.
8. Define one progression bottleneck sensitivity scenario.
9. Define one DST strain-dropout sensitivity scenario.
10. Define one treatment-monitoring scenario.
11. Define one no-resistance comparator scenario.
12. Run all scenarios with common seed and common baseline parameters for a simple example.
13. For decision-quality analysis, rerun scenarios across multiple seeds and summarize uncertainty.
14. Plot time series for disease burden and resistance share.
15. Plot final strain composition by scenario.
16. Save the plots to a results folder.

Coverage target:

- Examples: required.

Current status:

- Current example covers baseline, TPT scale-up, DST plus BPaL, combined BPaL plus TPT, acquisition pressure, fitness cost, progression bottleneck, DST dropout, and treatment monitoring.
- Current example also includes a no-resistance comparator and optional replicate uncertainty workflow.

## Critical Path 24: Validation And Regression Testing

Plain-English pseudocode:

1. Run a default TB simulation without resistance and confirm it still works.
2. Run a matched simulation with resistance enabled but minimal resistance pressure.
3. Compare core burden metrics between the two runs:
   - active TB prevalence,
   - new active TB incidence,
   - TB mortality.
4. Confirm that adding the resistance overlay does not unintentionally dominate overall TB dynamics.
5. Increase random acquisition and confirm resistant strains increase.
6. Reduce resistant-profile fitness and confirm resistant strains become less competitive.
7. Increase treatment pressure against susceptible profiles and confirm resistant share can increase.
8. Increase TPT pressure and confirm susceptible profiles are selectively reduced when the regimen targets them.
9. Check that DST observed resistance behaves correctly under sensitivity, specificity, and strain dropout.
10. Check that superinfection obeys state-specific gates.
11. Check that duplicate-profile attempts are blocked and counted.
12. Check that random acquisition adds profiles.
13. Check that selective acquisition replaces profiles.
14. Check that treatment monitoring selects only people who have been on treatment long enough.
15. Run the examples and confirm plots are generated.

Coverage target:

- Tests: required.
- Examples: required for smoke validation.

Current status:

- Most deterministic and scenario-level tests exist.
- Example smoke validation exists.
- Multi-seed uncertainty plots are available through the critical-path example's uncertainty mode.

## Critical Path 25: Known Residual Items

These are not hidden requirements. They are explicit implementation or example coverage items that should be addressed or documented.

1. Treatment monitoring can identify people ready for follow-up diagnostics, but the model does not yet have a full built-in cancellation mechanism for prematurely stopping an already scheduled treatment course.
2. DST strain dropout uses a single strain-observation probability. It does not yet model separate bottlenecks for sample collection, culture growth, sequencing, or other lab processes.
3. Non-infectious superinfection defaults may need a final scientific default. The engine allows configuration.
4. The current monitoring scenario demonstrates eligibility and diagnostic scheduling, but the default seed may produce zero monitoring positives.
5. The uncertainty workflow is example-grade; decision-grade analysis should use larger populations and more seeds.

## Recommended Next Example Additions

The critical-path scenario runner now includes these arms:

1. No-resistance comparator.
2. Baseline resistance program.
3. TPT scale-up.
4. DST plus second-line regimen.
5. Combined TPT plus second-line regimen.
6. High acquisition pressure.
7. Lower fitness cost.
8. Progression bottleneck sensitivity.
9. DST dropout sensitivity.
10. Treatment monitoring with follow-up DST or regimen switch.
11. Multi-seed baseline and intervention comparison with uncertainty bands.

## Bottom Line

The spec can be implemented as an opt-in resistance overlay with clear critical paths. The engine should own transmission, superinfection, progression, clearance, acquisition, treatment effects, TPT effects, DST, routing helpers, monitoring helpers, and analyzers. The example suite now exercises every major configurable path with baseline, intervention, sensitivity, no-resistance, and uncertainty scenarios. The main residual model feature is full cancellation or superseding of in-flight treatment when a monitoring diagnostic triggers an early regimen change.
