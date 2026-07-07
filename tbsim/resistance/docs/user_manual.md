# TBsim Resistance User Manual

This manual is a practical guide to using the TB drug-resistance overlay in `tbsim`.

## 1) What this module does

The resistance overlay adds:

- Multiple strains per person (`AgentStrains`)
- Drug-resistance phenotypes per strain (`StrainSpec` / `StrainCatalog`)
- Fitness-weighted transmission (`ResistanceConnector`)
- Strain-aware progression and acquisition (`ProgressionResolver`, `AcquisitionResolver`)
- Strain-aware treatment, TPT, DST, and routing (`StrainAwareTx`, `StrainAwareTPTTx`, `DSTDx`, `RegimenRouter`)

Core natural-history states (`TBS`) remain agent-level.

---

## 2) Core concepts

- `StrainSpec`: one strain definition (`uid`, `resistance`, `fitness`, `init_prev`)
- `StrainCatalog`: indexed collection of strain specs
- `AgentStrains`: per-agent carriage state (`carries_<uid>` arrays on `MultiStrainTB`)
- `MultiStrainTB`: TB disease model with strain overlay
- `ResistanceConnector`: applies strain fitness to transmission

---

## 3) Imports

Use the smallest import set for your workflow.

### Quick start (recommended)

```python
from tbsim.resistance import ResistanceSim
```

### Manual wiring

```python
import tbsim
import starsim as ss

from tbsim.resistance import (
    MultiStrainTB, ResistanceConnector,
    StrainSpec, Regimen,
    StrainAwareTx, StrainAwareTxDelivery,
    DSTDx, DSTDelivery, RegimenRouter,
    StrainResults,
)
```

---

## 4) Quick start (recommended)

Use `ResistanceSim` for a ready-to-run resistance-enabled simulation.

```python
from tbsim.resistance import ResistanceSim

sim = ResistanceSim(
    n_agents=2000,
    strain_preset='standard',  # or 'two_strain'
    cascade='routed',          # optional: 'basic', 'routed', 'uniform_no_dst', 'dst_routed_inh', 'tx_pressure'
)
sim.run()
```

Get key modules:

```python
tb = sim.get_multistrain_tb()
dst = sim.get_dst()
strain_results = sim.get_strain_results()
```

---

## 5) Manual wiring (full control)

```python
strains = [
    StrainSpec('pan',   {'INH': 0, 'RIF': 0, 'BDQ': 0}, fitness=1.00, init_prev=0.04),
    StrainSpec('inh_r', {'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95, init_prev=0.01),
    StrainSpec('rif_r', {'INH': 0, 'RIF': 1, 'BDQ': 0}, fitness=0.90, init_prev=0.005),
]

tb = MultiStrainTB(
    strains=strains,
    pars=dict(init_prev=ss.bernoulli(0.05)),
    progression_mode='bottleneck',   # or 'all'
    p_multi=1.0,
    p_random_acquisition={'INH': 0.01},
)

sim = tbsim.Sim(
    n_agents=2000,
    diseases=tb,
    connectors=ResistanceConnector(),
    analyzers=[StrainResults(), DuplicateStrainAnalyzer()],
)
sim.run()
```

---

## 6) Strains and resistance setup

### Define strains

`resistance` is a per-drug bit mapping: `1 = resistant`, `0 = susceptible`.

```python
StrainSpec('mdr', {'INH': 1, 'RIF': 1, 'BDQ': 0}, fitness=0.85, init_prev=0.002)
```

### Build a catalog explicitly (optional)

```python
catalog = StrainCatalog(strains, drugs=['INH', 'RIF', 'BDQ'])
```

You can pass either a list of `StrainSpec` or a pre-built `StrainCatalog` to `MultiStrainTB`.

---

## 7) Treatment

### Regimen

```python
reg = Regimen(
    'first_line',
    drugs=['INH', 'RIF'],
    per_drug_efficacy={'INH': 0.95, 'RIF': 0.95},
    resistance_penalty={'INH': 0.0, 'RIF': 0.0},  # optional
    combine='max',                                 # or 'parallel'
)
```

### Strain-aware treatment product + delivery

```python
tx_product = StrainAwareTx(
    regimen=reg,
    catalog=tb._strain_catalog,
    p_selective_acquisition={'INH': 0.02, 'RIF': 0.01},
    adherence=0.85,
)

tx_delivery = StrainAwareTxDelivery(
    name='first_line_tx',
    product=tx_product,
)
```

For regimen-switch workflows, set `cancel_delivery='old_tx_name'`.

---

## 8) TPT

```python
tpt_reg = Regimen('tpt_inh', drugs=['INH'], per_drug_efficacy={'INH': 0.9})
tpt_product = StrainAwareTPTTx(
    regimen=tpt_reg,
    catalog=tb._strain_catalog,
    p_tpt_acquisition={'INH': 0.01},
)
```

Behavior:
- Clears only regimen-susceptible carried strains
- Applies acquisition on failure paths
- Supports state-modified acquisition risk

---

## 9) DST and routing

### DST product + delivery

```python
dst_product = DSTDx(
    tb._strain_catalog,
    drugs=['INH', 'RIF'],
    sensitivity=0.95,
    specificity=0.99,
    p_strain_obs=1.0,   # optional; default uses strain fitness
    p_sample=1.0,
    p_culture=1.0,
)
dst_delivery = DSTDelivery(name='dst', product=dst_product, coverage=0.85)
```

### Route treatment by DST phenotype

```python
router = RegimenRouter(dst_delivery, diagnosed_state='diagnosed', require_dst_tested=True)

second_line_tx = StrainAwareTxDelivery(
    name='second_line_tx',
    product=second_line_product,
    eligibility=router.matches(INH=True),  # resistant
)

first_line_tx = StrainAwareTxDelivery(
    name='first_line_tx',
    product=first_line_product,
    eligibility=router.matches(INH=False), # susceptible
)
```

### Treatment monitoring

```python
monitor_elig = treatment_monitoring_eligibility('first_line_tx', after_steps=8)
```

---

## 10) Analyzers and outputs

- `StrainResults`: per-strain carrier/active/new-carrier channels
- `DuplicateStrainAnalyzer`: blocked duplicate superinfection counts
- `ResistanceStats`: aggregate ODE-facing resistance observables

Typical usage:

```python
sim = tbsim.Sim(..., analyzers=[StrainResults(), DuplicateStrainAnalyzer(), ResistanceStats()])
sim.run()
```

---

## 11) Common pitfalls

- Forgetting `ResistanceConnector`:
  - `MultiStrainTB` runs, but strain fitness will not modify `rel_trans`.
- Using base `TB` instead of `MultiStrainTB`:
  - no strain overlay state will exist.
- Drug name mismatch:
  - regimen/DST drug names must exist in `catalog.drugs`.
- Assuming a single-strain state:
  - carriage is multi-strain; logic should use `n_strains_per_agent()`.

---

## 12) Minimal checklist

1. Define strains with `StrainSpec(..., resistance=...)`
2. Use `MultiStrainTB(strains=...)`
3. Add `ResistanceConnector()`
4. Optionally add cascade (`build_care_cascade` or manual Tx/TPT/DST)
5. Add `StrainResults()` analyzer
6. Run sim and inspect resistance outputs

---

## 13) Where to go next

- Architecture details: `tbsim/resistance/docs/resistance_architecture.md`
- Step-by-step behavior guide (plain language): `tbsim/resistance/docs/resistance_step_by_step_guide.md`
- Worked scenarios and scripts: `tbsim_examples/resistance/`
