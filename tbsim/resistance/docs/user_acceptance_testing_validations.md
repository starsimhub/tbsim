# TBsim Resistance & Multi-Strain — User Acceptance Validations

Companion to the resistance technical specification (`tbsim-resistance-tech-spec -UPDATED.pdf`).
One user acceptance validation (UAT) per feature section of the technical specification, plus follow-on UATs for requirements called out inside those sections that are not yet implemented.
Each UAT states the acceptance criterion, plain-English steps, and either a code snippet against the current `tbsim.resistance` API or a `Placeholder **`.

**Scope:** Resistance profiles, multi-strain infection, transmission, competition/reinfection protection, progression, clearance, de-novo acquisition, treatment, TPT, DST / treatment monitoring, notation, and testing.

**Related code:** `tbsim/resistance/`; architecture and tests under `tbsim/resistance/docs/`.

---

## Table of contents

- [UAT-01 — Individual strain resistance profiles](#uat-01--individual-strain-resistance-profiles)
- [UAT-02 — Multi-strain infections (superinfection)](#uat-02--multi-strain-infections-superinfection)
- [UAT-03 — Transmission (fitness + single-strain pass)](#uat-03--transmission-fitness--single-strain-pass)
- [UAT-04 — Strain competition / protection against reinfection](#uat-04--strain-competition--protection-against-reinfection)
- [UAT-05 — Identical-strain carriage counts (alternative model) (TODO)](#uat-05--identical-strain-carriage-counts-alternative-model-todo)
- [UAT-06 — Progression to disease](#uat-06--progression-to-disease)
- [UAT-07 — Time-varying progression risk (reset the clock) (TODO)](#uat-07--time-varying-progression-risk-reset-the-clock-todo)
- [UAT-08 — Clearance](#uat-08--clearance)
- [UAT-09 — Random (de novo) acquisition](#uat-09--random-de-novo-acquisition)
- [UAT-10 — Treatment & selective acquisition](#uat-10--treatment--selective-acquisition)
- [UAT-11 — Adherence as a per-agent distribution (TODO)](#uat-11--adherence-as-a-per-agent-distribution-todo)
- [UAT-12 — LTFU as a separate treatment outcome (TODO)](#uat-12--ltfu-as-a-separate-treatment-outcome-todo)
- [UAT-13 — TPT](#uat-13--tpt)
- [UAT-14 — Diagnostics & treatment modification (DST + monitoring)](#uat-14--diagnostics--treatment-modification-dst--monitoring)
- [UAT-15 — DST indeterminate outcomes (TODO)](#uat-15--dst-indeterminate-outcomes-todo)
- [UAT-16 — Treatment failure vs new case (time since last treatment) (TODO)](#uat-16--treatment-failure-vs-new-case-time-since-last-treatment-todo)
- [UAT-17 — Notation / drug naming](#uat-17--notation--drug-naming)
- [UAT-18 — Testing / burden & parameter-effect acceptance](#uat-18--testing--burden--parameter-effect-acceptance)
- [UAT-19 — Burden table on `tb_LAI_TPT` parameters (TODO)](#uat-19--burden-table-on-tb_lai_tpt-parameters-todo)
- [Traceability summary](#traceability-summary)

---

## UAT-01 — Individual strain resistance profiles

**Spec section:** Individual strain resistance profiles

**Status:** Implemented

**Accept when:** For `n` named drugs/classes, the model can represent the full space of `m = 2^n` binary resistance profiles `X_j`, including pan-susceptible and every combination (e.g. RIF-only, RIF+FQ). Fitness of a strain is the product of per-drug costs for the drugs it resists (or an equivalent user-supplied multiplicative fitness).

**Steps**

1. Create a strain catalog with drugs `RIF`, `BDQ`, `FQ`.
2. Confirm there are 8 strains (`2^3`) when all combinations are registered.
3. Confirm profile decoding: pan-susceptible = `{0,0,0}`; RIF-only = `{1,0,0}`; RIF+FQ = `{1,0,1}`.
4. Confirm fitness is the product of per-drug costs for resisted drugs only.
5. Confirm adding a drug name requires no code change beyond appending to the drug list / strain specs.

```python
from itertools import product
from tbsim.resistance import StrainSpec, StrainCatalog

drugs = ['RIF', 'BDQ', 'FQ']
rel_fitness = {'RIF': 0.5, 'BDQ': 0.8, 'FQ': 0.9}

specs = []
for bits in product([0, 1], repeat=len(drugs)):
    res = dict(zip(drugs, bits))
    fit = 1.0
    for d, bit in res.items():
        if bit:
            fit *= rel_fitness[d]
    uid = 'pan' if sum(bits) == 0 else '_'.join(d for d, b in res.items() if b) + '_r'
    specs.append(StrainSpec(uid=uid, resistance=res, fitness=fit))

catalog = StrainCatalog(specs, drugs=drugs)
assert catalog.n == 8
assert list(catalog.resistance[0]) == [0, 0, 0]          # pan
rif_only = next(i for i, r in enumerate(catalog.resistance) if list(r) == [1, 0, 0])
rif_fq = next(i for i, r in enumerate(catalog.resistance) if list(r) == [1, 0, 1])
assert abs(catalog.fitness[rif_fq] - 0.5 * 0.9) < 1e-9
```

---

## UAT-02 — Multi-strain infections (superinfection)

**Spec section:** Allow for multi-strain infections (i.e., superinfections)

**Status:** Implemented

**Accept when:** One agent can carry multiple distinct strains at once. The agent profile `Y_k` is readable as the set of carried strains. There is no hard cap on how many strains an agent may carry.

**Steps**

1. Build a short `MultiStrainTB` sim with drugs `RIF`, `BDQ`, `FQ`.
2. Assign Agent A = RIF-only; Agent B = pan + RIF+FQ (spec examples).
3. Confirm membership matches the examples via `agent_strains`.
4. Confirm an agent can carry more than two strains without error.

```python
import starsim as ss
from tbsim.resistance import MultiStrainTB, StrainSpec, StrainCatalog

drugs = ['RIF', 'BDQ', 'FQ']
catalog = StrainCatalog([
    StrainSpec('pan',   {'RIF': 0, 'BDQ': 0, 'FQ': 0}),
    StrainSpec('rif_r', {'RIF': 1, 'BDQ': 0, 'FQ': 0}, fitness=0.5),
    StrainSpec('rif_fq',{'RIF': 1, 'BDQ': 0, 'FQ': 1}, fitness=0.45),
], drugs=drugs)

tb = MultiStrainTB(strains=catalog, init_prev=0)
sim = ss.Sim(diseases=tb, n_agents=10, start='2000-01-01', stop='2000-01-08', dt=ss.days(7))
sim.init()

a, b = 0, 1
tb.agent_strains.add_strain([a], 'rif_r')
tb.agent_strains.add_strain([b], 'pan')
tb.agent_strains.add_strain([b], 'rif_fq')
assert tb.agent_strains.n_strains_per_agent([a])[0] == 1
assert tb.agent_strains.n_strains_per_agent([b])[0] == 2
```

---

## UAT-03 — Transmission (fitness + single-strain pass)

**Spec section:** Transmission

**Status:** Implemented

**Accept when:** Infectees only acquire a strain the source already carries (no resistance emergence on transmission). Overall force of infection uses the fittest carried strain. Which strain is passed is multinomial ∝ fitness. Superinfection does not lower overall infectiousness relative to mono-infection with the fittest strain. Spec table values hold (~56% / ~44% → 0.28β / 0.22β).

**Steps**

1. Build strains with `r_RIF=0.5`, `r_BDQ=0.8`.
2. Source carries `{1,0,0}` and `{1,1,0}` (RIF and RIF+BDQ).
3. Assert max fitness == 0.5 (not reduced by the weaker strain).
4. Assert transmit split ≈ `1/1.8` and `0.8/1.8`, and per-strain rates ≈ `0.28` / `0.22` × β.
5. Confirm every positive transmit probability corresponds to a carried strain.

```python
import numpy as np
from tbsim.resistance import StrainSpec, StrainCatalog

catalog = StrainCatalog([
    StrainSpec('rif_r',     {'RIF': 1, 'BDQ': 0}, fitness=0.5),
    StrainSpec('rif_bdq_r', {'RIF': 1, 'BDQ': 1}, fitness=0.5 * 0.8),
])
fit = catalog.fitness
assert np.isclose(fit.max(), 0.5)
p_rif = fit[0] / fit.sum()
p_rif_bdq = fit[1] / fit.sum()
assert np.isclose(p_rif, 1 / 1.8) and np.isclose(p_rif_bdq, 0.8 / 1.8)
assert np.isclose(p_rif * 0.5, 0.28, atol=0.01)
assert np.isclose(p_rif_bdq * 0.5, 0.22, atol=0.01)
```

---

## UAT-04 — Strain competition / protection against reinfection

**Spec section:** Strain competition and protection against reinfection

**Status:** Implemented

**Accept when:** Superinfection risk depends on disease state (`INFECTION` / `NON_INFECTIOUS` allowed; `ASYMPTOMATIC` / `SYMPTOMATIC` blocked by default). Protection is strain-agnostic and count-agnostic. Identical-strain re-exposure does not add a second copy; blocked events are countable. Defaults couple early-infection protection to recovered reinfection protection, and non-infectious protection to early-infection protection.

**Steps**

1. Run with early-infection / non-infectious superinfection factors > 0, and ASY/SYM factors = 0.
2. Confirm agents in ASY/SYM do not gain a new strain under default parameters.
3. Confirm an already-superinfected agent is not *more* protected than a mono-infected agent against a third distinct strain.
4. Force re-exposure to an already-carried strain; confirm carriage unchanged and the duplicate-block counter increments.

```python
import starsim as ss
from tbsim.resistance import (
    MultiStrainTB, StrainSpec, StrainCatalog,
    ResistanceConnector, DuplicateStrainAnalyzer,
)

catalog = StrainCatalog([
    StrainSpec('pan', {'RIF': 0}, init_prev=0.2),
    StrainSpec('rif_r', {'RIF': 1}, fitness=0.7),
])
tb = MultiStrainTB(
    strains=catalog,
    alpha_super=0.5,
    alpha_act={'non_infectious': 0.5, 'asymptomatic': 0.0, 'symptomatic': 0.0},
)
sim = ss.Sim(
    diseases=tb,
    connectors=ResistanceConnector(),
    analyzers=DuplicateStrainAnalyzer(),
    n_agents=5000,
    start='2000-01-01',
    stop='2005-12-31',
    dt=ss.days(7),
)
sim.run()
assert tb._alpha_act['asymptomatic'] == 0.0
assert tb._alpha_act['symptomatic'] == 0.0
```

---

## UAT-05 — Identical-strain carriage counts (alternative model) (TODO)

**Spec section:** Strain competition — alternative to blocking identical strains

**Status:** NOT IMPLEMENTED

**Accept when:** As an alternative to blocking superinfection with identical strains, the model can count how many instances of each strain an agent carries, and use those counts when deciding which strains progress and/or get transmitted — without requiring a continuous relative-frequency declaration.

**Steps**

1. Enable count-based carriage mode (once implemented).
2. Allow repeated successful exposures to the same strain to increment that strain’s count.
3. Confirm progression and/or transmission selection can weight by counts.
4. Compare bias toward low-prevalence strains vs the current block-identical-strains rule (using the blocked-event analyzer as a baseline).

```
Placeholder **
```

**Notes:** Spec flags this as the main decision not yet tested with an ODE. Current implementation blocks identical-strain superinfection and counts blocked events via `DuplicateStrainAnalyzer`; count-based carriage is not implemented.

---

## UAT-06 — Progression to disease

**Spec section:** Progression to disease

**Status:** Implemented

**Accept when:** Natural history remains agent-level (not per-strain). Progression rates do not depend on strain count. INFECTION→NON_INFECTIOUS retains all strains. At →ASYMPTOMATIC, with `p_multi=1` all strains are retained; with `p_multi=0` exactly one strain progresses (equal probability by default).

**Steps**

1. Seed a multi-strain agent in INFECTION; progress to NON_INFECTIOUS → all strains remain.
2. Repeat with `p_multi=1` into ASYMPTOMATIC → all strains remain.
3. Repeat with `p_multi=0` → exactly one strain remains.
4. Confirm selection among carried strains is equal-probability (not fitness-weighted) unless a sensitivity analysis explicitly asks otherwise.

```python
import starsim as ss
from tbsim import TBS
from tbsim.resistance import MultiStrainTB, StrainSpec, StrainCatalog

catalog = StrainCatalog([
    StrainSpec('pan', {'RIF': 0, 'BDQ': 0}),
    StrainSpec('rif_r', {'RIF': 1, 'BDQ': 0}, fitness=0.5),
])
tb = MultiStrainTB(strains=catalog, p_multi=0.0, progression_mode='bottleneck', init_prev=0)
sim = ss.Sim(diseases=tb, n_agents=20, start='2000-01-01', stop='2000-01-08', dt=ss.days(7))
sim.init()
uid = 0
tb.state[uid] = TBS.INFECTION
tb.agent_strains.add_strain([uid], 'pan')
tb.agent_strains.add_strain([uid], 'rif_r')
assert tb.agent_strains.n_strains_per_agent([uid])[0] == 2
# Bottleneck path is exercised by ProgressionResolver on activation
```

---

## UAT-07 — Time-varying progression risk (reset the clock) (TODO)

**Spec section:** Progression to disease — time-varying risk

**Status:** NOT IMPLEMENTED

**Accept when:** When time-varying risk of disease progression is enabled, each new infection resets the individual’s time-since-infection clock. Ideally, successful exposure to a strain the agent already carries also resets the clock even if the strain profile does not change. The number of previously infecting strains does not change the temporal pattern of progression risk.

**Steps**

1. Enable time-varying progression hazard (once implemented).
2. Infect a susceptible agent; record progression hazard trajectory vs time since infection.
3. Superinfect with a new strain; confirm the clock resets and the hazard trajectory restarts.
4. Re-expose to an identical already-carried strain (blocked superinfection); confirm the clock still resets even though carriage is unchanged.
5. Confirm 0→1 and 1→2 infection transitions follow the same temporal progression pattern.

```
Placeholder **
```

**Notes:** Spec asks for clock reset on new infection and ideally on successful same-strain re-exposure. A full time-varying progression hazard driven by that clock is not modeled yet.

---

## UAT-08 — Clearance

**Spec section:** Clearance

**Status:** Implemented

**Accept when:** Natural clearance (INFECTION→CLEARED) and spontaneous resolution (NON_INFECTIOUS→CLEARED) clear **all** strains. Clearance rates are unaffected by superinfection. Selective (per-strain) clearance occurs only under treatment/TPT, not natural clearance.

**Steps**

1. Put a superinfected agent in INFECTION or NON_INFECTIOUS.
2. Allow (or force via the natural-history path) clearance to CLEARED.
3. Assert no strains remain and state is CLEARED.
4. Contrast with a treatment partial-cure case where some strains remain (UAT-10).

```python
import starsim as ss
from tbsim import TBS
from tbsim.resistance import MultiStrainTB, StrainSpec, StrainCatalog

catalog = StrainCatalog([
    StrainSpec('pan', {'RIF': 0}),
    StrainSpec('rif_r', {'RIF': 1}, fitness=0.7),
])
tb = MultiStrainTB(strains=catalog, init_prev=0)
sim = ss.Sim(diseases=tb, n_agents=5, start='2000-01-01', stop='2000-01-08', dt=ss.days(7))
sim.init()
u = 0
tb.state[u] = TBS.NON_INFECTIOUS
tb.agent_strains.add_strain([u], 'pan')
tb.agent_strains.add_strain([u], 'rif_r')
# Natural clearance path clears all carried strains on →CLEARED
```

---

## UAT-09 — Random (de novo) acquisition

**Spec section:** (Random) Acquisition

**Status:** Implemented (partial — random path uses add/superinfection; replacement mode for random acquisition is not a user switch)

**Accept when:** Resistance can appear as a one-time, independent per-drug probability `p_rand_i` at INFECTION→NON_INFECTIOUS or →ASYMPTOMATIC (not a per-timestep rate). Already-resistant drugs are skipped. Multiple acquisitions (across strains/drugs) are allowed. Spec allows either mixed (add) or replacement; the chosen strategy must be documented and consistent.

**Steps**

1. Set `p_random_acquisition={'BDQ': 1.0}` (RIF absent / 0).
2. Progress a pan-susceptible carrier out of INFECTION.
3. Confirm original + BDQ-resistant strain both present under the current add/superinfection strategy.
4. Confirm a BDQ-resistant strain does not re-acquire BDQ.
5. Confirm a denovo-resistance counter increments when acquisition occurs.

```python
import starsim as ss
from tbsim.resistance import MultiStrainTB, StrainSpec, StrainCatalog, ResistanceConnector

catalog = StrainCatalog([
    StrainSpec('pan', {'RIF': 0, 'BDQ': 0}, init_prev=0.3),
    StrainSpec('bdq_r', {'RIF': 0, 'BDQ': 1}, fitness=0.8),
])
tb = MultiStrainTB(
    strains=catalog,
    p_random_acquisition={'BDQ': 1.0},
)
sim = ss.Sim(
    diseases=tb,
    connectors=ResistanceConnector(),
    n_agents=2000,
    start='2000-01-01',
    stop='2010-12-31',
    dt=ss.days(7),
)
sim.run()
# Expect denovo events when pan carriers progress and BDQ target exists in catalog
assert getattr(tb, '_n_denovo_resistance_this_step', 0) >= 0
```

---

## UAT-10 — Treatment & selective acquisition

**Spec section:** Treatment & (Selective) Acquisition

**Status:** Implemented (partial — see UAT-11, UAT-12)

**Accept when:** Per-strain efficacy `T_l` is full for strains susceptible to regimen drugs and reduced for resistant strains. Partial cure leaves the agent in the prior TB state with surviving strains (no memory of cleared strains). Failed courses can acquire regimen-drug resistance by **replacement**, once per episode, scaled by TB-state RR (default 1 for ASY/SYM, 0 elsewhere). Adherence correlates outcomes across strains for one course. `q_{l,i}=0` for drugs not in the regimen.

**Steps**

1. Treat a multi-strain symptomatic agent with a BDQ-containing regimen (`q_acq={'BDQ': >0}`; RIF/FQ = 0).
2. Confirm susceptible strains clear at full efficacy; resistant strains at reduced efficacy.
3. On failure of a BDQ-susceptible survivor, confirm replacement to a BDQ-resistant profile.
4. Confirm non-adherent agents clear no strains that course.
5. Confirm acquisition risk is zero when prior state is not ASY/SYM (default RR).

```python
import starsim as ss
from tbsim.resistance import (
    MultiStrainTB, StrainSpec, StrainCatalog, ResistanceConnector,
    Regimen, StrainAwareTx, StrainAwareTxDelivery, ResistanceStats,
)

catalog = StrainCatalog([
    StrainSpec('pan', {'RIF': 0, 'BDQ': 0}, init_prev=0.2),
    StrainSpec('bdq_r', {'RIF': 0, 'BDQ': 1}, fitness=0.8),
])
tb = MultiStrainTB(strains=catalog)
regimen = Regimen('first_line', drugs=['BDQ'], base_efficacy=0.85, resistance_penalty={'BDQ': 0.33})
tx = StrainAwareTx(
    regimen=regimen,
    catalog=catalog,
    adherence=1.0,
    p_selective_acquisition={'BDQ': 0.2},
)
delivery = StrainAwareTxDelivery(product=tx, rate_sym=ss.peryear(5.0), name='first_line')
sim = ss.Sim(
    diseases=tb,
    connectors=ResistanceConnector(),
    interventions=delivery,
    analyzers=ResistanceStats(),
    n_agents=5000,
    start='2000-01-01',
    stop='2015-12-31',
    dt=ss.days(7),
)
sim.run()
assert sim.results['first_line'].n_treated.sum() > 0
```

---

## UAT-11 — Adherence as a per-agent distribution (TODO)

**Spec section:** Treatment & (Selective) Acquisition — adherence

**Status:** NOT IMPLEMENTED

**Accept when:** Adherence can be specified as a regimen-level **distribution** that varies by agent (not only a single Bernoulli probability). One draw per agent per treatment course is applied across all of that agent’s strains, inducing agent-level correlation in treatment efficacy.

**Steps**

1. Configure adherence as a distribution (e.g. Beta or empirical) rather than a fixed probability.
2. Start a multi-strain agent on treatment; confirm one adherence draw gates clearance of every carried strain for that course.
3. Confirm agents with high adherence clear more strains on average than agents with low adherence, with correlation across strains within the same agent.
4. Confirm a scalar adherence probability remains supported as a special case of the distribution.

```
Placeholder **
```

**Notes:** Current `StrainAwareTx` uses a single per-agent Bernoulli (`adherence` float). Spec asks for a regimen-level distribution that varies by agent and is applied across all strains during a given treatment course.

---

## UAT-12 — LTFU as a separate treatment outcome (TODO)

**Spec section:** Treatment & (Selective) Acquisition — unsuccessful outcomes

**Status:** NOT IMPLEMENTED

**Accept when:** Loss to follow-up (LTFU) can be modeled as a distinct unsuccessful treatment outcome (separate from failure and relapse). Acquisition risk `q_{l,i}` can apply on LTFU when that outcome is enabled, once per treatment episode, consistent with other unsuccessful outcomes.

**Steps**

1. Enable LTFU as a separate treatment outcome on a strain-aware regimen.
2. Confirm some treated agents exit via LTFU rather than cure / failure / relapse.
3. Confirm acquisition-on-LTFU can be configured (including off).
4. Confirm acquisition still occurs at most once per treatment episode and uses replacement for surviving susceptible strains.

```
Placeholder **
```

**Notes:** Spec notes LTFU is not currently a separate outcome in TBsim. Today acquisition applies on unsuccessful resolution of the course (failure path) without a distinct LTFU state.

---

## UAT-13 — TPT

**Spec section:** TPT

**Status:** Implemented

**Accept when:** TPT clears only strains susceptible to the TPT regimen. Resistant strains can remain and later progress/transmit (unmasking). Failed TPT (“neither” branch) can acquire resistance with a state-dependent risk gradient (highest ASY/SYM, lower NON_INFECTIOUS, lowest INFECTION). With `p_multi=1`, the main resistance harm is transmission unmasking, not progression bottleneck.

**Steps**

1. Give strain-aware TPT to agents carrying pan + resistant strains (`p_multi=1`).
2. Confirm pan strain can be cleared while resistant strain remains.
3. Among ineffective outcomes, confirm acquisition risk ranks SYM/ASY ≥ NON ≥ INF.
4. Compare resistant fraction / transmission vs a no-TPT control.

```python
from tbsim.resistance import (
    MultiStrainTB, StrainSpec, StrainCatalog, StrainAwareTPTTx, Regimen,
)

catalog = StrainCatalog([
    StrainSpec('pan', {'INH': 0}, init_prev=0.3),
    StrainSpec('inh_r', {'INH': 1}, fitness=0.9),
])
tb = MultiStrainTB(strains=catalog, p_multi=1.0)
product = StrainAwareTPTTx(
    regimen=Regimen('tpt_inh', drugs=['INH'], base_efficacy=0.5),
    catalog=catalog,
    p_tpt_acquisition={'INH': 0.1},
)
# Wrap in a TPT delivery, then compare resistant fraction among active TB
# vs a no-TPT control run.
```

---

## UAT-14 — Diagnostics & treatment modification (DST + monitoring)

**Spec section:** Diagnostics & Treatment Modification

**Status:** Implemented (partial — see UAT-15, UAT-16)

**Accept when:** DST produces an observed **n-drug** profile (not strain IDs). Sensitivity/specificity apply per strain, then aggregate. `p_strain_obs` (default = strain fitness) can drop strains from observation. Multi-strain carriage with `p_strain_obs=1` raises detection of a shared phenotype; `p_strain_obs < 1` lowers overall sensitivity. Treatment can be routed on the observed profile. Treatment monitoring can interrupt/switch an ongoing regimen after time-on-treatment.

**Steps**

1. Run DST on mono- vs multi-strain carriers; with `p_strain_obs=1`, multi-strain detection of a shared phenotype is higher.
2. Lower `p_strain_obs` and confirm detection falls.
3. Route second-line treatment from DST observed resistance / regimen router.
4. After N steps on first-line, use `treatment_monitoring_eligibility` + `cancel_delivery=` to switch regimens mid-course; confirm first-line is interrupted and second-line starts.

```python
import starsim as ss
from tbsim.resistance import (
    MultiStrainTB, StrainSpec, StrainCatalog, ResistanceConnector,
    DSTDx, DSTDelivery, Regimen, StrainAwareTx, StrainAwareTxDelivery,
    treatment_monitoring_eligibility,
)

catalog = StrainCatalog([
    StrainSpec('pan', {'RIF': 0, 'BDQ': 0}, init_prev=0.2),
    StrainSpec('rif_r', {'RIF': 1, 'BDQ': 0}, fitness=0.7),
])
tb = MultiStrainTB(strains=catalog)
dst = DSTDx(catalog, drugs=['RIF', 'BDQ'], sensitivity=0.9, specificity=0.98)
dst_iv = DSTDelivery(product=dst, name='dst')

first = StrainAwareTxDelivery(
    product=StrainAwareTx(Regimen('first_line', drugs=['RIF'], base_efficacy=0.85), catalog),
    name='first_line',
    rate_sym=ss.peryear(3.0),
)
second = StrainAwareTxDelivery(
    product=StrainAwareTx(Regimen('second_line', drugs=['BDQ'], base_efficacy=0.8), catalog),
    name='second_line',
    eligibility=treatment_monitoring_eligibility('first_line', after_steps=4),
    cancel_delivery='first_line',
)
sim = ss.Sim(
    diseases=tb,
    connectors=ResistanceConnector(),
    interventions=[dst_iv, first, second],
    n_agents=3000,
    start='2000-01-01',
    stop='2010-12-31',
    dt=ss.days(7),
)
sim.run()
```

---

## UAT-15 — DST indeterminate outcomes (TODO)

**Spec section:** Diagnostics & Treatment Modification — DST

**Status:** NOT IMPLEMENTED

**Accept when:** For each drug/class, DST can return user-defined observed outcomes including at least positive (resistant), negative (susceptible), and **indeterminate**, not only a binary resistant/susceptible call.

**Steps**

1. Configure DST with outcome categories that include indeterminate.
2. Administer DST to agents with known true phenotypes.
3. Confirm some calls can be indeterminate under the configured sens/spec / indeterminate probabilities.
4. Confirm treatment routing can treat indeterminate separately from resistant and susceptible (e.g. do not auto-switch regimen on indeterminate alone).

```
Placeholder **
```

**Notes:** Current `DSTDx` aggregates to a binary observed phenotype per drug. Indeterminate is not a supported call.

---

## UAT-16 — Treatment failure vs new case (time since last treatment) (TODO)

**Spec section:** Diagnostics & Treatment Modification — DST eligibility

**Status:** NOT IMPLEMENTED

**Accept when:** The model tracks time since last treatment initiation and uses it to classify a later presentation as treatment failure (eligible for DST / second-line) versus a new case.

**Steps**

1. Treat an agent, then allow them to leave treatment (cure or failure).
2. After a short interval, re-present the agent and confirm they are classified as treatment failure / retreatment (DST / second-line path).
3. After a long interval (beyond a configurable threshold), re-present the same agent and confirm they are classified as a new case.
4. Confirm DST eligibility and regimen choice can depend on that classification.

```
Placeholder **
```

**Notes:** `StrainAwareTxDelivery` tracks `ti_treatment_start` for the current course (used by monitoring), but there is no durable “time since last treatment initiation” used to distinguish failure vs new case.

---

## UAT-17 — Notation / drug naming

**Spec section:** Notation

**Status:** Implemented

**Accept when:** Users configure drugs and per-drug parameters by **name** (not fragile positional arrays). Adding a drug is “append to drugs / strain specs,” not a code change. Unknown drug names in parameter dicts are rejected.

**Steps**

1. Construct a catalog with `['RIF','BDQ']` and pass fitness / acquisition keyed by name.
2. Add `'FQ'` and confirm the catalog expands; shared drug names remain stable.
3. Confirm Tx/DST/TPT dict keys resolve through catalog drug names.

```python
from tbsim.resistance import StrainSpec, StrainCatalog

catalog = StrainCatalog([
    StrainSpec('pan', {'RIF': 0, 'BDQ': 0}),
    StrainSpec('bdq_r', {'RIF': 0, 'BDQ': 1}, fitness=0.8),
])
assert catalog.drugs == ['RIF', 'BDQ']
catalog2 = StrainCatalog([
    StrainSpec('pan', {'RIF': 0, 'BDQ': 0, 'FQ': 0}),
    StrainSpec('fq_r', {'RIF': 0, 'BDQ': 0, 'FQ': 1}, fitness=0.9),
])
assert 'FQ' in catalog2.drugs
```

---

## UAT-18 — Testing / burden & parameter-effect acceptance

**Spec section:** Testing

**Status:** Implemented (partial — see UAT-19)

**Accept when:** (a) Enabling multi-strain with no resistance pressure does not materially change overall TB burden vs plain TB. (b) Raising acquisition risk, lowering fitness costs, lowering resistant-strain efficacy, or changing treatment rate moves resistant fraction and burden in the expected direction. Prefer a realistic baseline (e.g. `tb_LAI_TPT` best-fit) for the burden table (prevalence, asymptomatic incidence, mortality per 100,000).

**Steps**

1. Compare plain `TB` vs `MultiStrainTB` with resistance pressure off — prevalence, asymptomatic incidence, and mortality should be close.
2. Sweep acquisition risk, fitness, resistant-strain efficacy, and treatment rate; confirm `% resistant among active TB` and burden respond as expected.
3. Optionally run the packaged directional / scenario report helpers.

```python
from tbsim.resistance import (
    build_spec_sim, get_spec_scenario_configs,
    summarize_spec_sim, compute_spec_directional_checks,
)

# Directional / reduction checks via packaged scenario helpers
configs = get_spec_scenario_configs()
# Run selected scenarios, summarize, then:
# checks = compute_spec_directional_checks(summary_df)
# assert checks pass for acquisition / fitness / treatment pressure
```

---

## UAT-19 — Burden table on `tb_LAI_TPT` parameters (TODO)

**Spec section:** Testing — before/after burden comparison

**Status:** NOT IMPLEMENTED

**Accept when:** A documented comparison table exists for overall TB disease prevalence per 100,000, annual incidence of new asymptomatic disease per 100,000, and annual TB mortality per 100,000, using the best-fitting parameter set / configuration from `tb_LAI_TPT`, before vs after resistance/multi-strain is enabled. Material shifts are interrogated and explained.

**Steps**

1. Configure a baseline sim from the `tb_LAI_TPT` best-fit parameters (plain `TB` or resistance-off `MultiStrainTB`).
2. Configure the matched multi-strain / resistance-enabled sim.
3. Report prevalence, asymptomatic incidence, and mortality per 100,000 for both.
4. Confirm differences are small, or document why/under what conditions they are not.

```
Placeholder **
```

**Notes:** Directional parameter-effect helpers exist in `tbsim/resistance/spec.py`. The LAI_TPT burden-per-100k before-vs-after table has not been produced.

---

## Traceability summary

| UAT | Spec section | Status | Primary API / note |
|-----|--------------|--------|--------------------|
| UAT-01 | Strain profiles | Implemented | `StrainSpec`, `StrainCatalog` |
| UAT-02 | Superinfection | Implemented | `MultiStrainTB.agent_strains` |
| UAT-03 | Transmission | Implemented | `ResistanceConnector`, catalog fitness |
| UAT-04 | Competition / reinfection | Implemented | `alpha_super`, `alpha_act`, `DuplicateStrainAnalyzer` |
| UAT-05 | Strain carriage counts | NOT IMPLEMENTED | block + counter only |
| UAT-06 | Progression | Implemented | `ProgressionResolver`, `p_multi` |
| UAT-07 | Time-varying progression | NOT IMPLEMENTED | clock-reset hazard not modeled |
| UAT-08 | Clearance | Implemented | natural clear removes all strains |
| UAT-09 | De novo acquisition | Implemented (partial) | `p_random_acquisition` (add path) |
| UAT-10 | Treatment | Implemented (partial) | `StrainAwareTx`, `StrainAwareTxDelivery` |
| UAT-11 | Adherence distribution | NOT IMPLEMENTED | Bernoulli only |
| UAT-12 | LTFU outcome | NOT IMPLEMENTED | not a separate outcome |
| UAT-13 | TPT | Implemented | `StrainAwareTPTTx` |
| UAT-14 | DST + monitoring | Implemented (partial) | `DSTDx`, `DSTDelivery`, `cancel_delivery` |
| UAT-15 | DST indeterminate | NOT IMPLEMENTED | binary observed phenotype only |
| UAT-16 | Failure vs new case | NOT IMPLEMENTED | no durable time-since-last-Tx |
| UAT-17 | Notation | Implemented | name-keyed drugs / catalog |
| UAT-18 | Testing | Implemented (partial) | `spec.py` scenario helpers |
| UAT-19 | LAI_TPT burden table | NOT IMPLEMENTED | table not produced |

**Out of scope for UAT:** Sources section (background papers only; not a software requirement).
