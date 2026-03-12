---
title: TB diagnostic refactor plan
format:
  pdf:
    colorlinks: false
    linkcolor: black
    urlcolor: black
    toccolor: black
    highlight-style: monochrome
    include-in-header:
      text: |
        \usepackage{xcolor}
        \definecolor{shadecolor}{RGB}{240,240,240}
---

# TB diagnostic refactor plan

## Motivation

The current `TBDiagnostic` and `EnhancedTBDiagnostic` classes in
`tbsim/interventions/tb_diagnostic.py` have two problems:

1. **Mixed concerns**: test characteristics (sensitivity, specificity) and delivery
   logic (who gets tested, when, false-negative retry) are bundled into one class,
   making it hard to swap out diagnostic products independently of delivery logic.

2. **Model lock-in**: both classes are hardcoded to TB_EMOD states
   (`TBS.ACTIVE_SMPOS`, `TBS.ACTIVE_SMNEG`, `TBS.ACTIVE_EXPTB`) and cannot be
   used with the TB_LSHTM natural history model.

The refactor separates the code into two layers — **product** and **delivery** —
following the same pattern used by `ss.Dx` and `ss.Tx` in the core Starsim
`products.py`.

---

## Design overview

```
ss.Product  (starsim)
    └── TBDx  (tbsim)         ← what the test does
            │
            │  used by
            ▼
    TBDiagnosticDelivery       ← when/how/to whom the test is delivered
        (ss.Intervention)
```

Multiple `TBDiagnosticDelivery` steps can be chained to form a cascade
(e.g. screen → confirm → triage), each using a different product.

---

## Layer 1: `TBDx` product

### Location

New file: `tbsim/interventions/tb_dx_products.py`

### Inheritance

`TBDx(ss.Product)`

### Interface

```python
TBDx(df, hierarchy=None, **kwargs)
```

### The test positivity matrix (`df`)

`df` is a `pd.DataFrame` with three columns:

| column | type | description |
|--------|------|-------------|
| `state` | int (enum value) | TB state from `TBSL` or `TBS` |
| `result` | str | test result label (e.g. `'positive'`, `'negative'`) |
| `probability` | float | probability of this result given the state |

Rows must sum to 1.0 per state (i.e. probabilities across all results for a given
state must sum to 1).

### The `hierarchy` parameter

A list of result strings in priority order, e.g. `['positive', 'negative']`.
Agents whose TB state is not listed in `df` are assigned the **last** entry in the
hierarchy (the default). This means:

- Unlisted states → `'negative'` by default (zero false-positive rate).
- To model false positives for a non-active state, add an explicit row for it.

This is identical to the behaviour of `ss.Dx`.

### `administer(uids)` method

```
1. Look up the TB module via get_tb(sim)  — works for both TB_LSHTM and TB_EMOD
2. Pre-fill results Series with default_value (index = last hierarchy entry)
3. For each state_value in df.state.unique():
       in_state_uids = uids where tb.state == state_value
       probs = [df[(state==state_value) & (result==r)].probability for r in hierarchy]
       result_dist.set(p=probs)
       draws = result_dist.rvs(in_state_uids)
       results.loc[in_state_uids] = minimum(draws, results.loc[in_state_uids])
4. Return dict keyed by hierarchy strings:
       {'positive': uids_positive, 'negative': uids_negative}
```

The `minimum(draws, ...)` rule matches `ss.Dx`: the best (lowest-index) result wins
if a uid qualifies under multiple states.

### Difference from `ss.Dx`

`ss.Dx` finds agents in a state via `getattr(disease, state_name).uids`, which
requires named boolean-state attributes. TB models store state as a single integer
enum array (`tb.state`), so `TBDx` uses `tb.state == state_value` instead.
Everything else is structurally identical.

### Example: Xpert for TB_LSHTM

```python
import pandas as pd
import tbsim
from tbsim import TBSL

df = pd.DataFrame([
    dict(state=TBSL.SYMPTOMATIC,  result='positive', probability=0.90),
    dict(state=TBSL.SYMPTOMATIC,  result='negative', probability=0.10),
    dict(state=TBSL.ASYMPTOMATIC, result='positive', probability=0.50),
    dict(state=TBSL.ASYMPTOMATIC, result='negative', probability=0.50),
    # Agents in any other state default to 'negative' (last in hierarchy)
])
xpert = tbsim.TBDx(df=df, hierarchy=['positive', 'negative'])
```

### Example: Xpert for TB_EMOD (state-stratified sensitivity)

```python
from tbsim import TBS

df = pd.DataFrame([
    dict(state=TBS.ACTIVE_SMPOS, result='positive', probability=0.909),
    dict(state=TBS.ACTIVE_SMPOS, result='negative', probability=0.091),
    dict(state=TBS.ACTIVE_SMNEG, result='positive', probability=0.775),
    dict(state=TBS.ACTIVE_SMNEG, result='negative', probability=0.225),
    dict(state=TBS.ACTIVE_EXPTB, result='positive', probability=0.775),
    dict(state=TBS.ACTIVE_EXPTB, result='negative', probability=0.225),
])
xpert = tbsim.TBDx(df=df, hierarchy=['positive', 'negative'])
```

### Pre-built factory functions

To avoid users having to construct DataFrames by hand for common tests, provide
factory functions in `tb_dx_products.py`:

```python
tbsim.xpert_lshtm()       # Xpert MTB/RIF, calibrated for TB_LSHTM states
tbsim.xpert_emod()        # Xpert MTB/RIF, calibrated for TB_EMOD states
tbsim.cxr_lshtm()         # Chest X-ray (includes ASYMPTOMATIC with reduced sensitivity)
tbsim.oral_swab_lshtm()   # Oral swab
```

Each factory returns a `TBDx` instance with evidence-based default probabilities.
Users can still construct `TBDx` directly from a custom DataFrame to override any
parameter.

---

## Layer 2: `TBDiagnosticDelivery` intervention

### Location

Replaces the contents of `tbsim/interventions/tb_diagnostic.py`.

### Inheritance

`TBDiagnosticDelivery(ss.Intervention)`

### Parameters

| parameter | type | default | description |
|-----------|------|---------|-------------|
| `product` | `TBDx` | required | the diagnostic product to administer |
| `coverage` | float or `ss.Dist` | `1.0` | fraction of eligible agents who receive the test |
| `eligibility` | callable `(sim) → uids` | see below | who is eligible for this step |
| `result_state` | str | `'diagnosed'` | people-state to set `True` for positive results |
| `care_seeking_multiplier` | float | `1.0` | multiplier on care-seeking rate for false negatives |

**Default `eligibility`** (first step in a cascade):
```python
lambda sim: (sim.people.sought_care & ~sim.people.diagnosed & sim.people.alive).uids
```

For downstream steps in a cascade, the user supplies a custom eligibility callable
that reads the `result_state` set by the previous step (see cascade example below).

### Auto-registration of `result_state`

During `init_pre`, if `result_state` is not `'diagnosed'` and the named state does
not already exist on `sim.people`, `TBDiagnosticDelivery` registers it as a
`ss.BoolState` with `default=False`. This avoids requiring users to manually declare
intermediate states in `extra_states`.

### `step()` logic

```
1. Compute eligible_uids via self.eligibility(sim)
2. Apply coverage filter
3. Call product.administer(selected_uids)
       → {'positive': pos_uids, 'negative': neg_uids}
4. Set ppl.<result_state>[pos_uids] = True
5. Handle false negatives (TB-positive agents who tested negative):
       - identify: neg_uids who have active TB
       - apply care_seeking_multiplier (once per agent, guarded by a flag)
       - reset sought_care and tested flags to allow retry
6. Store counts for update_results
```

### Results

| result key | description |
|------------|-------------|
| `n_tested` | agents tested this step |
| `n_positive` | positive results this step |
| `n_negative` | negative results this step |
| `cum_positive` | cumulative positives |
| `cum_negative` | cumulative negatives |

---

## Cascade: chaining multiple delivery steps

Because each `TBDiagnosticDelivery` writes its output to a named people-state and
reads its input via an `eligibility` callable, steps can be chained arbitrarily.

### Example: CXR screen → Xpert confirm → treatment triage

```python
import tbsim
import starsim as ss

screen = tbsim.TBDiagnosticDelivery(
    product      = tbsim.cxr_lshtm(),
    coverage     = 0.9,
    # default eligibility: sought_care & ~diagnosed & alive
    result_state = 'screen_positive',
)

confirm = tbsim.TBDiagnosticDelivery(
    product      = tbsim.xpert_lshtm(),
    coverage     = 0.8,
    eligibility  = lambda sim: sim.people.screen_positive.uids,
    result_state = 'test_positive',
)

triage = tbsim.TBDiagnosticDelivery(
    product      = tbsim.dst_lshtm(),          # drug-sensitivity test
    coverage     = 1.0,
    eligibility  = lambda sim: sim.people.test_positive.uids,
    result_state = 'diagnosed',                # terminal step
)

sim = ss.Sim(
    diseases     = tbsim.TB_LSHTM(),
    interventions = [
        tbsim.HealthSeekingBehavior(),
        screen,
        confirm,
        triage,
    ],
    pars = dict(start='2000', stop='2020'),
)
sim.run()
```

The intermediate people-states (`screen_positive`, `test_positive`) are
auto-registered by each delivery intervention, so no `extra_states` declaration
is needed.

### Single-step usage (no cascade)

```python
diag = tbsim.TBDiagnosticDelivery(
    product  = tbsim.xpert_lshtm(),
    coverage = 0.8,
    # result_state defaults to 'diagnosed'
)
```

---

## Migration from existing classes

| Old class | Replacement |
|-----------|-------------|
| `TBDiagnostic` | `TBDiagnosticDelivery(product=tbsim.xpert_lshtm(), ...)` |
| `EnhancedTBDiagnostic` | `TBDiagnosticDelivery(product=tbsim.TBDx(df=...), ...)` with a state-stratified `df` |

The old classes will be kept temporarily with deprecation warnings to avoid breaking
existing scripts, then removed in a future release.

---

## Files changed

| file | change |
|------|--------|
| `tbsim/interventions/tb_dx_products.py` | **new** — `TBDx` class and factory functions |
| `tbsim/interventions/tb_diagnostic.py` | **replace** both old classes with `TBDiagnosticDelivery`; keep old classes with deprecation warnings |
| `tbsim/interventions/__init__.py` | export `TBDx`, `TBDiagnosticDelivery`, and factory functions |
| `tbsim/__init__.py` | export same |

---

## Open questions

1. **False-negative retry**: the current `TBDiagnostic` boosts `care_seeking_multiplier`
   for false negatives and resets `sought_care`. Should this logic live in
   `TBDiagnosticDelivery`, or be moved to `HealthSeekingBehavior`? The latter would
   be cleaner architecturally but requires an interface between the two interventions.

2. **FujiLAM (HIV-stratified sensitivity)**: deferred. When needed, the options are:
   (a) a `TBDxHIV` subclass that overrides `administer()` to check both TB and HIV
   state, or (b) splitting the cohort by HIV status in `TBDiagnosticDelivery` before
   calling `administer()`.

3. **Cumulative results**: currently computed incrementally each step. Could instead
   use `finalize_results()` with `np.cumsum`, consistent with how TB_LSHTM does it.
   Worth standardising.
