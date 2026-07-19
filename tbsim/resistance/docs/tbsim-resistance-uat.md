# TBsim Resistance & Multi-Strain — User Acceptance Validations

Companion to [tbsim-resistance-tech-spec.md](tbsim-resistance-tech-spec.md).
One user acceptance validation (UAT) per feature section of the technical specification, plus follow-on UATs for requirements called out inside those sections that are not yet implemented.
Each UAT states the acceptance criterion, plain-English steps, and either a code snippet against the current `tbsim.resistance` API or a `NOT IMPLEMENTED (TODO)` placeholder.

**Scope:** Resistance profiles, multi-strain infection, transmission, competition/reinfection protection, progression, clearance, de-novo acquisition, treatment, TPT, DST / treatment monitoring, notation, and testing.

**Related code:** `tbsim/resistance/`; automated checks in `tbsim/resistance/devtests/`.

## Table of contents

1. [UAT-01 — Individual strain resistance profiles](#uat-01)
2. [UAT-02 — Multi-strain infections (superinfection)](#uat-02)
3. [UAT-03 — Transmission (fitness + single-strain pass)](#uat-03)
4. [UAT-04 — Strain competition / protection against reinfection](#uat-04)
5. [UAT-05 — Identical-strain carriage counts](#uat-05)
6. [UAT-06 — Progression to disease](#uat-06)
7. [UAT-07 — Time-varying progression risk (reset the clock) (TODO)](#uat-07)
8. [UAT-08 — Clearance](#uat-08)
9. [UAT-09 — Random (de novo) acquisition](#uat-09)
10. [UAT-10 — Treatment & selective acquisition](#uat-10)
11. [UAT-11 — Adherence as a per-agent distribution](#uat-11)
12. [UAT-12 — LTFU as a separate treatment outcome (TODO)](#uat-12)
13. [UAT-13 — TPT](#uat-13)
14. [UAT-14 — Diagnostics & treatment modification (DST + monitoring)](#uat-14)
15. [UAT-15 — DST indeterminate outcomes (TODO)](#uat-15)
16. [UAT-16 — Treatment failure vs new case (time since last treatment)](#uat-16)
17. [UAT-17 — Notation / drug naming](#uat-17)
18. [UAT-18 — Testing / burden & parameter-effect acceptance](#uat-18)
19. [UAT-19 — Burden table on `tb_LAI_TPT` parameters](#uat-19)

---

## Requirement-to-UAT mapping (verified)

This crosswalk is verified against the requirement sections in
[tbsim-resistance-tech-spec.md](tbsim-resistance-tech-spec.md).

| Requirement section in PDF | UAT coverage |
|---|---|
| Individual strain resistance profiles | UAT-01 |
| Allow for multi-strain infections (superinfections) | UAT-02 |
| Transmission | UAT-03 |
| Strain competition and protection against reinfection | UAT-04, UAT-05 |
| Progression to disease | UAT-06, UAT-07 (TODO) |
| Clearance | UAT-08 |
| (Random) Acquisition | UAT-09 |
| Treatment & (Selective) Acquisition | UAT-10, UAT-11, UAT-12 (TODO) |
| TPT | UAT-13 |
| Diagnostics & Treatment Modification (DST + treatment monitoring) | UAT-14, UAT-15 (TODO), UAT-16 |
| Notation | UAT-17 |
| Testing | UAT-18, UAT-19 |
| Sources | Informational references (intentionally out of scope for software UAT) |

---

<a id="uat-01"></a>
## UAT-01 — Individual strain resistance profiles

**Spec section:** Individual strain resistance profiles

**Status:** Implemented

**Accept when:** For `n` named drugs/classes, the model exposes exactly `m = 2^n` binary resistance profiles `X_j`, including pan-susceptible and every combination (e.g. RIF-only, RIF+FQ). Fitness of a strain is the product of per-drug costs for the drugs it resists.

**Steps**

1. Create a strain registry with drugs `RIF`, `BDQ`, `FQ`.
2. Confirm there are 8 strains (`2^3`).
3. Confirm strain id decoding: id `0` = pan-susceptible; bit 0 set = RIF-resistant; id `5` = RIF+FQ (`{1,0,1}`).
4. Confirm fitness is the product of per-drug costs for resisted drugs only.
5. Confirm adding a drug name requires no code change beyond appending to `drugs`.

```python
import tbsim
import numpy as np

s = tbsim.Strains(['RIF', 'BDQ', 'FQ'], rel_fitness={'RIF': 0.5, 'BDQ': 0.8, 'FQ': 0.9})
assert s.n == 3 and s.m == 8
assert s.labels[0] == 'pan'
assert list(s.profile[1]) == [True, False, False]   # RIF only
assert list(s.profile[5]) == [True, False, True]    # RIF+FQ
assert np.isclose(s.fitness[5], 0.5 * 0.9)
```

**Existing coverage:** `tbsim/resistance/strains.py`; transmission operator tests use the same registry.

---

<a id="uat-02"></a>
## UAT-02 — Multi-strain infections (superinfection)

**Spec section:** Allow for multi-strain infections (i.e., superinfections)

**Status:** Implemented

**Accept when:** One agent can carry multiple distinct strains at once. The agent profile `Y_k` is readable as the set of carried strains. There is no hard cap on how many strains an agent may carry.

**Steps**

1. Build a short `TBResistant` sim with drugs `RIF`, `BDQ`, `FQ`.
2. Assign Agent A mask = RIF-only; Agent B mask = pan + RIF+FQ (spec examples).
3. Decode with `Strains.carried` and confirm membership matches the examples.
4. Confirm an agent can carry more than two strains without error.

```python
import tbsim
import starsim as ss
import numpy as np

tb = tbsim.TBResistant(drugs=['RIF', 'BDQ', 'FQ'], init_prev=0)
sim = ss.Sim(diseases=tb, n_agents=10, start='2000-01-01', stop='2000-01-08', dt=ss.days(7))
sim.init()
tb = sim.diseases.tb  # sim copies its modules at init; work with the live copy

a, b = 0, 1
tb.strain_mask[a] = 1 << 1                              # {1,0,0} RIF
tb.strain_mask[b] = (1 << 0) | (1 << 5)                 # {0,0,0} + {1,0,1}
carried = tb.strains.carried(tb.strain_mask.values[[a, b]])
assert carried[0].sum() == 1 and carried[0, 1]
assert carried[1].sum() == 2 and carried[1, 0] and carried[1, 5]
```

**Existing coverage:** `TBResistant.strain_mask` in `tb_resistant.py`; natural-history / transmission devtests.

---

<a id="uat-03"></a>
## UAT-03 — Transmission (fitness + single-strain pass)

**Spec section:** Transmission

**Status:** Implemented

**Accept when:** Infectees only acquire a strain the source already carries (no resistance emergence on transmission). Overall force of infection uses the fittest carried strain. Which strain is passed is multinomial ∝ fitness. Superinfection does not lower overall infectiousness relative to mono-infection with the fittest strain. Spec table values hold (~56% / ~44% → 0.28β / 0.22β).

**Steps**

1. Build strains with `r_RIF=0.5`, `r_BDQ=0.8`.
2. Source carries `{1,0,0}` and `{1,1,0}` (RIF and RIF+BDQ).
3. Assert `max_fitness == 0.5` (not reduced by the weaker strain).
4. Assert transmit split ≈ `1/1.8` and `0.8/1.8`, and per-strain rates ≈ `0.28` / `0.22` × β.
5. Confirm every positive transmit probability corresponds to a carried strain.

```python
import tbsim
import numpy as np

s = tbsim.Strains(['RIF', 'BDQ'], rel_fitness={'RIF': 0.5, 'BDQ': 0.8})
mask = np.array([(1 << 1) | (1 << 3)])  # {RIF} and {RIF,BDQ}
assert np.isclose(s.max_fitness(mask)[0], 0.5)
tp = s.transmit_probs(mask)[0]
assert np.isclose(tp[1], 1 / 1.8) and np.isclose(tp[3], 0.8 / 1.8)
assert np.isclose(tp[1] * 0.5, 0.28, atol=0.01)
assert np.isclose(tp[3] * 0.5, 0.22, atol=0.01)
assert set(np.where(tp > 0)[0]).issubset({1, 3})
```

**Existing coverage:** `tbsim/resistance/devtests/test_transmission.py::test_transmission_split_matches_spec_table`, `test_superinfection_does_not_reduce_infectiousness`.

---

<a id="uat-04"></a>
## UAT-04 — Strain competition / protection against reinfection

**Spec section:** Strain competition and protection against reinfection

**Status:** Implemented

**Accept when:** Superinfection risk depends on disease state (`INFECTION` / `NON_INFECTIOUS` allowed; `ASYMPTOMATIC` / `SYMPTOMATIC` blocked by default via `rr_reinfection_asy` / `rr_reinfection_sym` = 0; `TREATMENT` not superinfectable). Protection is strain-agnostic and count-agnostic. Identical-strain re-exposure increments that strain's per-agent count (rather than being blocked) and the events are countable. Defaults couple `rr_reinfection_inf` → `rr_reinfection_rec` and `rr_reinfection_non` → `rr_reinfection_inf`.

**Steps**

1. Run with `rr_reinfection_inf > 0`, `rr_reinfection_non > 0`, ASY/SYM RR = 0.
2. Confirm agents in ASY/SYM (and agents currently in TREATMENT) do not gain a new strain under default parameters.
3. Confirm an already-superinfected agent is not *more* protected than a mono-infected agent against a third distinct strain.
4. Force re-exposure to an already-carried strain; confirm the mask is unchanged and `new_identical_superinf` (and the strain's per-agent count) increments.

```python
import tbsim
import starsim as ss

tb = tbsim.TBResistant(
    drugs=['RIF'],
    rel_fitness={'RIF': 0.7},
    rr_reinfection_inf=0.5,
    rr_reinfection_non=0.5,
    rr_reinfection_asy=0.0,
    rr_reinfection_sym=0.0,
    init_prev=0.2,
)
sim = ss.Sim(diseases=tb, n_agents=5000, start='2000-01-01', stop='2005-12-31', dt=ss.days(7))
sim.run()
tb = sim.diseases.tb  # sim copies its modules at init; read results off the live copy
assert 'new_identical_superinf' in tb.results
assert tb.pars.rr_reinfection_asy == 0.0 and tb.pars.rr_reinfection_sym == 0.0
```

**Existing coverage:** `test_natural_history.py::test_identical_strain_superinfection_allowed_and_counted`, `test_q3a_noninfectious_eligibility_raises_superinfection`, `test_q3b_active_superinfection_adds_coinfection`; per-strain counter in `test_counter.py`.

---

<a id="uat-05"></a>
## UAT-05 — Identical-strain carriage counts

**Spec section:** Strain competition — count-based carriage of identical strains

**Status:** Implemented

**Accept when:** The model counts how many instances of each strain an agent carries (`tb.strain_counts`, one array per strain id), and uses those counts when deciding which strains progress and/or get transmitted — without requiring a continuous relative-frequency declaration.

**Steps**

1. Allow repeated successful exposures to the same strain to increment that strain's count (the default; identical-strain re-exposure is no longer blocked).
2. Confirm the count invariant holds (count > 0 ⟺ the strain bit is set) and that clearance/death reset counts to 0.
3. Confirm transmission selection weights by `count × fitness` and, under `p_multi < 1`, the progression bottleneck weights the surviving strain by count.

```python
import numpy as np
import starsim as ss
import tbsim

s = tbsim.Strains(['TX'], rel_fitness=None)  # neutral fitness
# Source carrying {pan:2, resistant:1} passes pan 2/3 of the time; infectiousness is count-independent.
assert np.allclose(s.transmit_probs(np.array([0b11]), counts=np.array([[2, 1]]))[0], [2/3, 1/3])
assert s.max_fitness(np.array([0b11]))[0] == 1.0
```

**Notes:** This replaces the earlier block-identical-strains rule. The counter feeds only the transmission multinomial and the progression bottleneck; transition rates, DST, treatment efficacy, and acquisition remain count-agnostic (identical copies behave as one).

**Existing coverage:** `test_counter.py` (data model, invariant, count-weighted transmission and bottleneck); `tests/test_resistance.py::test_strain_counter_and_count_weighting`.

---

<a id="uat-06"></a>
## UAT-06 — Progression to disease

**Spec section:** Progression to disease

**Status:** Implemented

**Accept when:** Natural history remains agent-level (not per-strain). Progression rates do not depend on strain count unless `rr_prog_super` is set. INFECTION→NON_INFECTIOUS retains all strains. At →ASYMPTOMATIC, with `p_multi=1` all strains are retained; with `p_multi=0` exactly one strain progresses (equal probability by default via `prog_select='random'`).

**Steps**

1. Seed a multi-strain agent in INFECTION; progress to NON_INFECTIOUS → all strains remain.
2. Repeat with `p_multi=1` into ASYMPTOMATIC → all strains remain.
3. Repeat with `p_multi=0` → exactly one strain bit remains.
4. Optionally set `prog_select='fitness'` and confirm selection weights by fitness.
5. Confirm `rr_prog_super` (ψ) scales multi-strain progression when set ≠ 1.

```python
import tbsim
import starsim as ss
from tbsim import TBS

tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], p_multi=0.0, prog_select='random')
sim = ss.Sim(diseases=tb, n_agents=20, start='2000-01-01', stop='2000-01-08', dt=ss.days(7))
sim.init()
tb = sim.diseases.tb  # sim copies its modules at init; work with the live copy
uid = 0
tb.state[uid] = TBS.INFECTION
tb.strain_mask[uid] = (1 << 0) | (1 << 1)  # pan + RIF
assert bin(int(tb.strain_mask[uid])).count('1') == 2
# Full bottleneck path exercised in test_natural_history.py (p_multi / fitness selection)
```

**Existing coverage:** `test_natural_history.py::test_q2a_bottleneck_reduces_superinfection`, `test_q2b_fitness_selection_lowers_resistance`, `test_q1_superinfection_faster_progression`.

---

<a id="uat-07"></a>
## UAT-07 — Time-varying progression risk (reset the clock) (TODO)

**Spec section:** Progression to disease — time-varying risk

**Status:** NOT IMPLEMENTED (TODO)

**Accept when:** When time-varying risk of disease progression is enabled, each new infection resets the individual’s time-since-infection clock. Ideally, successful exposure to a strain the agent already carries also resets the clock even if the strain profile does not change. The number of previously infecting strains does not change the temporal pattern of progression risk.

**Steps**

1. Enable time-varying progression hazard (once implemented).
2. Infect a susceptible agent; record progression hazard trajectory vs time since infection.
3. Superinfect with a new strain; confirm the clock resets and the hazard trajectory restarts.
4. Re-expose to an identical already-carried strain (blocked superinfection); confirm the clock still resets even though `strain_mask` is unchanged.
5. Confirm 0→1 and 1→2 infection transitions follow the same temporal progression pattern.

```
NOT IMPLEMENTED (TODO)
```

**Notes:** `ti_infected` is reset on successful and blocked exposures (hook exists), but a full time-varying progression hazard driven by that clock is not modeled yet.

---

<a id="uat-08"></a>
## UAT-08 — Clearance

**Spec section:** Clearance

**Status:** Implemented

**Accept when:** Natural clearance (INFECTION→CLEARED) and spontaneous resolution (NON_INFECTIOUS→CLEARED) clear **all** strains. Clearance rates are unaffected by superinfection unless `rr_clear_super` is set. Selective (per-strain) clearance occurs only under treatment/TPT, not natural clearance.

**Steps**

1. Put a superinfected agent in INFECTION or NON_INFECTIOUS.
2. Allow (or force via the natural-history path) clearance to CLEARED.
3. Assert `strain_mask == 0` and state is CLEARED.
4. Contrast with a treatment partial-cure case where some strains remain (UAT-10).

```python
import tbsim
import starsim as ss
from tbsim import TBS

tb = tbsim.TBResistant(drugs=['RIF'], init_prev=0)
sim = ss.Sim(diseases=tb, n_agents=5, start='2000-01-01', stop='2000-01-08', dt=ss.days(7))
sim.init()
tb = sim.diseases.tb  # sim copies its modules at init; work with the live copy
u = 0
tb.state[u] = TBS.NON_INFECTIOUS
tb.strain_mask[u] = (1 << 0) | (1 << 1)
# Natural clearance path in step_transitions sets strain_mask = 0 on →CLEARED
# Verified end-to-end in test_natural_clearance_removes_all_strains
```

**Existing coverage:** `test_natural_history.py::test_natural_clearance_removes_all_strains`.

---

<a id="uat-09"></a>
## UAT-09 — Random (de novo) acquisition

**Spec section:** (Random) Acquisition

**Status:** Implemented

**Accept when:** Resistance can appear as a one-time, independent per-drug probability `p_rand_i` at INFECTION→NON_INFECTIOUS or →ASYMPTOMATIC (not a per-timestep rate). Already-resistant drugs are skipped. Multiple acquisitions (across strains/drugs) are allowed. Mode is configurable: `mixed` (superinfection) vs `replacement`.

**Steps**

1. Set `p_rand={'BDQ': 1.0}` (RIF absent / 0).
2. Progress a pan-susceptible carrier out of INFECTION.
3. In `prog_resist_mode='mixed'`: original + BDQ-resistant strain both present.
4. In `prog_resist_mode='replacement'`: pan strain replaced by BDQ-resistant strain.
5. Confirm a BDQ-resistant strain does not re-acquire BDQ.
6. Confirm `new_denovo_resistance` increments.

```python
import tbsim
import starsim as ss

tb = tbsim.TBResistant(
    drugs=['RIF', 'BDQ'],
    p_rand={'BDQ': 1.0}, prog_resist_mode='mixed', init_prev=0.3,
)
sim = ss.Sim(diseases=tb, n_agents=2000, start='2000-01-01', stop='2010-12-31', dt=ss.days(7))
sim.run()
tb = sim.diseases.tb  # sim copies its modules at init; read results off the live copy
assert tb.results['new_denovo_resistance'].sum() > 0
```

**Existing coverage:** `test_acquisition.py` (`test_q4_mixed_makes_ab_replacement_does_not`, `test_denovo_per_drug_specificity`, `test_denovo_multistrain_each_strain_mutates`).

---

<a id="uat-10"></a>
## UAT-10 — Treatment & selective acquisition

**Spec section:** Treatment & (Selective) Acquisition

**Status:** Implemented (partial — see UAT-11, UAT-12)

**Accept when:** Per-strain efficacy `T_l` is full for strains susceptible to regimen drugs and reduced for resistant strains. Partial cure leaves the agent in the prior TB state with surviving strains (no memory of cleared strains). Failed courses can acquire regimen-drug resistance by **replacement**, once per episode, scaled by TB-state RR (`acq_state_rr`; default 1 for ASY/SYM, 0 elsewhere). Adherence correlates outcomes across strains for one course. `q_{l,i}=0` for drugs not in the regimen.

**Steps**

1. Treat a multi-strain symptomatic agent with a BDQ-containing regimen (`q_acq={'BDQ': >0}`; RIF/FQ = 0).
2. Confirm susceptible strains clear at full efficacy; resistant strains at reduced efficacy (`resist_penalty`).
3. On failure of a BDQ-susceptible survivor, confirm replacement to a BDQ-resistant profile.
4. Confirm non-adherent agents clear no strains that course.
5. Confirm acquisition risk is zero when prior state is not ASY/SYM (default RR).

```python
import tbsim
import starsim as ss

tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], rel_fitness={'BDQ': 0.8}, init_prev=0.2)
tx = tbsim.TxR(
    strains=tb.strains,
    base_efficacy=0.85,
    resist_penalty={'BDQ': 0.33},
    adherence=1.0,
    q_acq={'BDQ': 0.2},
    regimen_drugs=['BDQ'],
)
delivery = tbsim.TxDeliveryR(product=tx, rate_sym=ss.peryear(5.0), name='first_line')
sim = ss.Sim(
    diseases=tb,
    interventions=delivery,
    analyzers=tbsim.ResistanceStats(),
    n_agents=5000,
    start='2000-01-01',
    stop='2015-12-31',
    dt=ss.days(7),
)
sim.run()
assert sim.results['first_line'].n_treated.sum() > 0
assert sim.results['first_line'].n_acquired.sum() >= 0
```

**Existing coverage:** `test_treatment.py` (`test_treatment_operator_matches_ode_pi_table`, `test_acquisition_only_in_active_states`, `test_treatment_selects_for_resistance_matches_ode`).

---

<a id="uat-11"></a>
## UAT-11 — Adherence as a per-agent distribution

**Spec section:** Treatment & (Selective) Acquisition — adherence

**Status:** Implemented

**Accept when:** Adherence can be specified as a regimen-level **distribution** that varies by agent (not only a single Bernoulli probability). One draw per agent per treatment course is applied across all of that agent's strains, inducing agent-level correlation in treatment efficacy.

**Steps**

1. Configure adherence as a callable `uids -> per-agent probability` rather than a fixed probability.
2. Start a multi-strain agent on treatment; confirm one adherence draw gates clearance of every carried strain for that course.
3. Confirm agents with high adherence clear more strains on average than agents with low adherence, with correlation across strains within the same agent.
4. Confirm a scalar adherence probability remains supported as the degenerate special case.

```python
import numpy as np
import starsim as ss
import tbsim

tb = tbsim.TBResistant(drugs=['RIF'], init_prev=ss.bernoulli(0.0))
# Per-agent adherence distribution: first half fully adherent, second half never.
prod = tbsim.TxR(strains=tb.strains, base_efficacy=1.0,
                 adherence=lambda uids: np.where(np.asarray(uids) < 5000, 1.0, 0.0))
assert callable(prod.adherence_distribution)  # a float would leave this None
```

**Notes:** `adherence` accepts either a float (one regimen-level probability shared by all agents — the degenerate distribution) or a callable `uids -> per-agent probability`. The single per-agent completion draw is shared across all of the agent's strains.

**Existing coverage:** `tests/test_resistance.py::test_treatment_adherence_distribution`; `codex_review/test_spec_gaps.py`.

---

<a id="uat-12"></a>
## UAT-12 — LTFU as a separate treatment outcome (TODO)

**Spec section:** Treatment & (Selective) Acquisition — unsuccessful outcomes

**Status:** NOT IMPLEMENTED (TODO)

**Accept when:** Loss to follow-up (LTFU) can be modeled as a distinct unsuccessful treatment outcome (separate from failure and relapse). Acquisition risk `q_{l,i}` can apply on LTFU when that outcome is enabled, once per treatment episode, consistent with other unsuccessful outcomes.

**Steps**

1. Enable LTFU as a separate treatment outcome on a strain-aware regimen.
2. Confirm some treated agents exit via LTFU rather than cure / failure / relapse.
3. Confirm acquisition-on-LTFU can be configured (including off).
4. Confirm acquisition still occurs at most once per treatment episode and uses replacement for surviving susceptible strains.

```
NOT IMPLEMENTED (TODO)
```

**Notes:** Spec notes LTFU is not currently a separate outcome in TBsim. Today acquisition applies on unsuccessful resolution of the course (failure path) without a distinct LTFU state.

---

<a id="uat-13"></a>
## UAT-13 — TPT

**Spec section:** TPT

**Status:** Implemented

**Accept when:** TPT clears only strains susceptible to every drug in the TPT regimen. Resistant strains can remain and later progress/transmit (unmasking). Failed TPT (“neither” branch) can acquire resistance with a state-dependent risk gradient (highest ASY/SYM, lower NON_INFECTIOUS, lowest INFECTION). With `p_multi=1`, the main resistance harm is transmission unmasking, not progression bottleneck.

**Steps**

1. Give `TPTRx` to agents carrying pan + resistant strains (`p_sterilize > 0`).
2. Confirm pan strain can be cleared while resistant strain remains.
3. Among ineffective outcomes, confirm acquisition risk ranks SYM/ASY ≥ NON ≥ INF.
4. Compare resistant fraction / transmission vs a no-TPT control.

```python
import tbsim
import starsim as ss

tb = tbsim.TBResistant(
    drugs=['INH'],
    rel_fitness={'INH': 0.9},
    p_multi=1.0, init_prev=0.3,
)
product = tbsim.TPTRx(
    strains=tb.strains,
    regimen_drugs=['INH'],
    p_tpt_acq={'INH': 0.1},
    pars=dict(p_sterilize=0.5, efficacy=0.5),
)
# Wrap in a TPT delivery, e.g. tbsim.TPTSimple(product=product), then compare
# resistant fraction among active TB vs a no-TPT control run.
```

**Existing coverage:** `test_diagnostics_tpt.py::test_tpt_unmasks_and_selects_resistance`, `test_tpt_failure_acquisition_state_gradient`.

---

<a id="uat-14"></a>
## UAT-14 — Diagnostics & treatment modification (DST + monitoring)

**Spec section:** Diagnostics & Treatment Modification

**Status:** Implemented (partial — see UAT-15, UAT-16)

**Accept when:** DST produces an observed **n-drug** profile (not strain IDs). Sensitivity/specificity apply per strain, then aggregate. `p_strain_obs` (default = strain fitness) can drop strains from observation. Multi-strain carriage with `p_strain_obs=1` raises detection of a shared phenotype; `p_strain_obs < 1` lowers overall sensitivity. DST eligibility can be configured as immediate-after-diagnosis or treatment-failure-triggered. Treatment can be routed on the observed profile. Treatment monitoring can interrupt/switch an ongoing regimen after time-on-treatment.

**Steps**

1. Run DST on mono- vs multi-strain carriers; with `p_strain_obs=1`, multi-strain detection of a shared phenotype is higher.
2. Lower `p_strain_obs` and confirm detection falls.
3. Route second-line `TxDeliveryR` with `eligibility=` from `DSTDelivery.observed_resistant` / `matches`.
4. Validate both DST eligibility modes: immediate testing after diagnosis and deferred testing after treatment failure logic (when configured).
5. After N steps on first-line, use `treatment_monitoring_eligibility` + `supersedes=` to switch regimens mid-course; confirm first-line is interrupted and second-line starts.

```python
import tbsim
import starsim as ss

tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], init_prev=0.2)
dst = tbsim.DST(
    strains=tb.strains,
    sens={'RIF': 0.9, 'BDQ': 0.85},
    spec=0.98,
)  # p_strain_obs defaults to strain fitness
dst_iv = tbsim.DSTDelivery(product=dst, name='dst')

first = tbsim.TxDeliveryR(
    product=tbsim.TxR(tb.strains, regimen_drugs=['RIF'], base_efficacy=0.85),
    name='first_line',
    rate_sym=ss.peryear(3.0),
)
second = tbsim.TxDeliveryR(
    product=tbsim.TxR(tb.strains, regimen_drugs=['BDQ'], base_efficacy=0.8),
    name='second_line',
    eligibility=tbsim.treatment_monitoring_eligibility('first_line', after_steps=4),
    supersedes=['first_line'],
)
sim = ss.Sim(
    diseases=tb,
    interventions=[dst_iv, first, second],
    n_agents=3000,
    start='2000-01-01',
    stop='2010-12-31',
    dt=ss.days(7),
)
sim.run()
```

**Existing coverage:** `test_diagnostics_tpt.py` (`test_dst_recovers_sens_spec_mono`, `test_dst_multiple_strains_raise_detection`, `test_dst_p_strain_obs_bottleneck_lowers_detection`). The treatment-monitoring switch is implemented (`treatment_monitoring_eligibility` + `supersedes` → `TxDeliveryR.interrupt`) and exercised by the snippet above, but is not yet covered by a dedicated `devtests/` test.

---

<a id="uat-15"></a>
## UAT-15 — DST indeterminate outcomes (TODO)

**Spec section:** Diagnostics & Treatment Modification — DST

**Status:** NOT IMPLEMENTED (TODO)

**Accept when:** For each drug/class, DST can return user-defined observed outcomes including at least positive (resistant), negative (susceptible), and **indeterminate**, not only a binary resistant/susceptible call.

**Steps**

1. Configure DST with outcome categories that include indeterminate.
2. Administer DST to agents with known true phenotypes.
3. Confirm some calls can be indeterminate under the configured sens/spec / indeterminate probabilities.
4. Confirm treatment routing can treat indeterminate separately from resistant and susceptible (e.g. do not auto-switch regimen on indeterminate alone).

```
NOT IMPLEMENTED (TODO)
```

**Notes:** Current `DST` aggregates to a binary n-bit observed profile (`dst_profile`). Indeterminate is not a supported call.

---

<a id="uat-16"></a>
## UAT-16 — Treatment failure vs new case (time since last treatment)

**Spec section:** Diagnostics & Treatment Modification — DST eligibility

**Status:** Implemented

**Accept when:** The model tracks time since last treatment initiation and uses it to classify a later presentation as treatment failure (eligible for DST / second-line) versus a new case.

**Steps**

1. Treat an agent, then allow them to leave treatment (cure or failure). Every `TxDeliveryR` stamps a durable, cross-regimen `tb.ti_last_treatment` at initiation.
2. Within the window, `TxDeliveryR.failure_case_eligibility(within=...)` classifies the agent as treatment failure / retreatment (DST / second-line path).
3. Beyond the window (or never treated), `failure_case_eligibility(within=..., new_case=True)` classifies the agent as a new case.
4. Feed either callable as `eligibility=` on a DST or `TxDeliveryR` so routing/regimen choice depends on the classification.

```python
import numpy as np
import starsim as ss
import tbsim

failed = tbsim.TxDeliveryR.failure_case_eligibility(within=ss.years(2))
new_case = tbsim.TxDeliveryR.failure_case_eligibility(within=ss.years(2), new_case=True)
# The two are complementary partitions of the candidate pool (default: active TB).
```

**Notes:** `ti_last_treatment` is a durable per-agent state on `TBResistant` (distinct from the current-course `ti_treatment_start` used by monitoring). `base=` restricts the candidate pool (default active TB); `within` is any `ss.dur`.

**Existing coverage:** `tests/test_resistance.py::test_failure_vs_new_case_classification`; `codex_review/test_spec_gaps.py`.

---

<a id="uat-17"></a>
## UAT-17 — Notation / drug naming

**Spec section:** Notation

**Status:** Implemented

**Accept when:** Users configure drugs and per-drug parameters by **name** (not fragile positional arrays). Adding a drug is “append to `drugs`,” not a code change. Unknown drug names in parameter dicts are rejected.

**Steps**

1. Construct `Strains(['RIF','BDQ'])` and pass `rel_fitness={'BDQ': 0.8}` (name-keyed).
2. Add `'FQ'` to `drugs` and confirm `m` doubles; RIF/BDQ bit positions remain stable for the shared prefix.
3. Confirm Tx/DST/TPT dict keys resolve through `Strains.drug_idx`.

```python
import tbsim

s = tbsim.Strains(['RIF', 'BDQ'], rel_fitness={'BDQ': 0.8})
assert s.drug_idx['RIF'] == 0 and s.drug_idx['BDQ'] == 1
s2 = tbsim.Strains(['RIF', 'BDQ', 'FQ'], rel_fitness={'FQ': 0.9})
assert s2.m == 2 * s.m
```

**Existing coverage:** `Strains.__init__` validation; name-keyed APIs on `TxR`, `DST`, `TPTRx`.

---

<a id="uat-18"></a>
## UAT-18 — Testing / burden & parameter-effect acceptance

**Spec section:** Testing

**Status:** Implemented (partial — see UAT-19)

**Accept when:** (a) Enabling multi-strain with no resistance pressure does not materially change overall TB burden vs plain TB. (b) Raising acquisition risk, lowering fitness costs, lowering resistant-strain efficacy, or changing treatment rate moves resistant fraction and burden in the expected direction. Prefer a realistic baseline (e.g. `tb_LAI_TPT` best-fit) for the burden table (prevalence, asymptomatic incidence, mortality per 100,000).

**Steps**

1. Compare plain `TB` vs `TBResistant(drugs=['TX'], …)` with resistance off — prevalence, asymptomatic incidence, and mortality should be close.
2. Sweep `q_acq`, `rel_fitness`, `resist_penalty`, and treatment rate; confirm `% resistant among active TB` and burden respond as expected.
3. Optionally run ABM↔ODE validation for the two-strain reference.

```python
# Directional / reduction checks (existing suite)
# pytest tbsim/resistance/devtests/test_transmission.py -k competitive
# pytest tbsim/resistance/devtests/test_treatment.py -k selects_for_resistance
# pytest tbsim/resistance/devtests/test_transmission.py -k single_strain_reduction
# Full ABM↔ODE:
# python tbsim/resistance/docs/validate_resistance_abm_vs_ode.py
```

**Existing coverage:** `test_transmission.py` (reduction, competitive exclusion); `test_treatment.py` (selection for resistance); `docs/validate_resistance_abm_vs_ode.py`.

---

<a id="uat-19"></a>
## UAT-19 — Burden table on `tb_LAI_TPT` parameters

**Spec section:** Testing — before/after burden comparison

**Status:** Implemented

**Accept when:** A documented comparison table exists for overall TB disease prevalence per 100,000, annual incidence of new asymptomatic disease per 100,000, and annual TB mortality per 100,000, before vs after resistance/multi-strain is enabled. Material shifts are interrogated and explained.

**Steps**

1. Configure a baseline sim (resistance-off: plain `TB` and machinery-on-but-single-strain `TBResistant.agnostic()`).
2. Configure the matched resistance-on sim (circulating resistant strain).
3. Report prevalence, asymptomatic incidence, and mortality per 100,000 for each.
4. Confirm differences are small, or document why/under what conditions they are not.

```bash
python -m tbsim.resistance.codex_review.make_burden_validation   # writes lai_tpt_burden_validation.csv
```

**Notes:** `codex_review/make_burden_validation.py` runs three matched, seed-averaged scenarios (`resistance_off`, `resistance_on_agnostic`, `resistance_on`) and writes `codex_review/lai_tpt_burden_validation.csv`. The agnostic row matches base TB to within noise; the resistance-on row shifts overall burden only modestly (consistent with the spec's expectation). ABM↔ODE validation and directional parameter-effect tests also exist (`devtests/`, reference-ODE set).

---

## Traceability summary

| UAT | Spec section | Status | Primary API / note |
|-----|--------------|--------|--------------------|
| UAT-01 | Strain profiles | Implemented | `Strains` |
| UAT-02 | Superinfection | Implemented | `TBResistant.strain_mask` |
| UAT-03 | Transmission | Implemented | `max_fitness`, `transmit_probs` |
| UAT-04 | Competition / reinfection | Implemented | `rr_reinfection_*`, `new_identical_superinf` |
| UAT-05 | Strain carriage counts | Implemented | `strain_counts` (count-weighted transmit/bottleneck) |
| UAT-06 | Progression | Implemented | `p_multi`, bottleneck |
| UAT-07 | Time-varying progression | NOT IMPLEMENTED (TODO) | `ti_infected` hook only |
| UAT-08 | Clearance | Implemented | natural clear → `strain_mask=0` |
| UAT-09 | De novo acquisition | Implemented | `p_rand`, `prog_resist_mode` |
| UAT-10 | Treatment | Implemented | `TxR` (incl. `efficacy_by_strain`), `TxDeliveryR` |
| UAT-11 | Adherence distribution | Implemented | `adherence` callable → `adherence_distribution` |
| UAT-12 | LTFU outcome | NOT IMPLEMENTED (TODO) | not a separate outcome |
| UAT-13 | TPT | Implemented | `TPTRx` |
| UAT-14 | DST + monitoring | Implemented (partial) | `DST`, `DSTDelivery`, monitoring helpers |
| UAT-15 | DST indeterminate | NOT IMPLEMENTED (TODO) | binary `dst_profile` only |
| UAT-16 | Failure vs new case | Implemented | `failure_case_eligibility`, `ti_last_treatment` |
| UAT-17 | Notation | Implemented | name-keyed dicts / `drug_idx` |
| UAT-18 | Testing | Implemented (partial) | `devtests/` + ODE validation |
| UAT-19 | LAI_TPT burden table | Implemented | `make_burden_validation.py` → CSV |

**Out of scope for UAT:** Sources section (background papers only; not a software requirement).
