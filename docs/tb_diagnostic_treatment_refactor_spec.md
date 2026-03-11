# TB Diagnostic & Treatment Refactor Spec

## Motivation

The current `TBDiagnostic`, `EnhancedTBDiagnostic`, `TBTreatment`, and `EnhancedTBTreatment` classes each bundle two concerns into one class:

1. **Product logic** — test sensitivity/specificity (diagnostics) or drug efficacy (treatment)
2. **Delivery logic** — who gets tested/treated, when, false-negative retry, flag management

This makes it hard to swap diagnostic products independently of delivery logic (e.g. compare Xpert vs oral swab under the same delivery strategy), and it makes the Enhanced variants unwieldy with dozens of parameters.

The refactor separates each into **product** (what the test/drug does) and **delivery** (who/when/how it's administered), following the same pattern already used by `BCGVx`/`BCGRoutine` and `TPTTx`/`TPTSimple` in tbsim.

## Scope

- TB_LSHTM only (referred to simply as "TB" throughout). TB_EMOD support is dropped.
- All diagnostic interventions (`TBDiagnostic`, `EnhancedTBDiagnostic`) are replaced.
- All treatment interventions (`TBTreatment`, `EnhancedTBTreatment`) are replaced.
- `HealthSeekingBehavior`, `TBProductRoutine`, BCG, TPT, `BetaByYear` are unchanged.

---

## Design Overview

```
ss.Product (starsim)
    |
    +-- Dx (tbsim)           <- what the test does
    |       |
    |       |  used by
    |       v
    |   DxDelivery            <- when/how/to whom the test is delivered
    |       (ss.Intervention)
    |
    +-- Tx (tbsim)           <- what the drug does
            |
            |  used by
            v
        TxDelivery            <- when/how/to whom treatment is delivered
            (ss.Intervention)
```

---

## Layer 1a: `Dx` Product

### File

New file: `tbsim/interventions/dx_products.py`

### Inheritance

`Dx(ss.Product)`

### Constructor

```python
Dx(df, hierarchy=None, **kwargs)
```

### The test positivity DataFrame (`df`)

`df` is a `pd.DataFrame` with required columns and optional filter columns:

**Required columns:**

| column | type | description |
|--------|------|-------------|
| `state` | int (TBSL enum value) | TB state |
| `result` | str | test result label (e.g. `'positive'`, `'negative'`) |
| `probability` | float | P(result \| state, filters) |

**Optional filter columns** (only include columns relevant to the specific diagnostic):

| column | type | description |
|--------|------|-------------|
| `age_min` | float | minimum age, inclusive (default 0 if column absent) |
| `age_max` | float | maximum age, exclusive (default inf if column absent) |
| `hiv` | bool | HIV status filter (only for HIV-sensitive tests like FujiLAM) |

Rules:
- Only include optional columns that the diagnostic actually needs. `xpert()` includes `age_min`/`age_max` (child vs adult) but not `hiv`. `fujilam()` includes `hiv`. A simple test with no stratification needs only the three required columns.
- Rows must sum to 1.0 per unique combination of (state + all filter columns present).
- `TBDx.administer()` checks which optional columns are present in the DataFrame and only filters on those.

### The `hierarchy` parameter

A list of result strings in priority order, e.g. `['positive', 'negative']`. Agents whose TB state is not listed in `df` are assigned the **last** entry (the default). This means unlisted states default to `'negative'` (zero false-positive rate).

### `administer(uids)` method

```
1. Look up the TB module via get_tb(sim)
2. Get agent attributes needed by the DataFrame's columns (TB state, age, HIV status)
3. Pre-fill results with default_value (last hierarchy entry)
4. For each unique combination of filter values in df:
       match agents by state + any filter columns present
       probs = probabilities for each result in hierarchy
       draw stochastic results
       assign results to matching agents
5. Return dict keyed by hierarchy strings:
       {'positive': uids_positive, 'negative': uids_negative}
```

### Factory functions

Pre-configured `Dx` instances with evidence-based parameters:

```python
tbsim.xpert()       # Xpert MTB/RIF — age-stratified (child/adult), TB-state-stratified
tbsim.oral_swab()   # Oral swab — age-stratified, TB-state-stratified
tbsim.fujilam()     # FujiLAM — HIV-stratified, age-stratified
tbsim.cad_cxr()     # CAD chest X-ray
```

### Example: `xpert()` factory DataFrame

```python
df = pd.DataFrame([
    # Adults (age >= 15), by TB state
    dict(state=TBSL.SYMPTOMATIC,  result='positive', probability=0.909, age_min=15, age_max=np.inf),
    dict(state=TBSL.SYMPTOMATIC,  result='negative', probability=0.091, age_min=15, age_max=np.inf),
    dict(state=TBSL.ASYMPTOMATIC, result='positive', probability=0.775, age_min=15, age_max=np.inf),
    dict(state=TBSL.ASYMPTOMATIC, result='negative', probability=0.225, age_min=15, age_max=np.inf),
    dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.775, age_min=15, age_max=np.inf),
    dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.225, age_min=15, age_max=np.inf),
    # Children (age < 15)
    dict(state=TBSL.SYMPTOMATIC,  result='positive', probability=0.73, age_min=0, age_max=15),
    dict(state=TBSL.SYMPTOMATIC,  result='negative', probability=0.27, age_min=0, age_max=15),
    dict(state=TBSL.ASYMPTOMATIC, result='positive', probability=0.73, age_min=0, age_max=15),
    dict(state=TBSL.ASYMPTOMATIC, result='negative', probability=0.27, age_min=0, age_max=15),
    dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.73, age_min=0, age_max=15),
    dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.27, age_min=0, age_max=15),
])
```

### Example: `fujilam()` factory DataFrame (HIV-stratified)

```python
df = pd.DataFrame([
    # HIV+, adults
    dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.75, age_min=15, age_max=np.inf, hiv=True),
    dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.25, age_min=15, age_max=np.inf, hiv=True),
    dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.75, age_min=15, age_max=np.inf, hiv=True),
    dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.25, age_min=15, age_max=np.inf, hiv=True),
    # HIV-, adults
    dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.58, age_min=15, age_max=np.inf, hiv=False),
    dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.42, age_min=15, age_max=np.inf, hiv=False),
    # HIV+, children
    dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.579, age_min=0, age_max=15, hiv=True),
    dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.421, age_min=0, age_max=15, hiv=True),
    # HIV-, children
    dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.51, age_min=0, age_max=15, hiv=False),
    dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.49, age_min=0, age_max=15, hiv=False),
    # ... etc
])
```

---

## Layer 1b: `Tx` Product

### File

New file: `tbsim/interventions/tx_products.py`

### Inheritance

`Tx(ss.Product)`

### Constructor

```python
Tx(efficacy=0.85, drug_type=None, **kwargs)
```

- `efficacy` (float): probability of treatment success
- `drug_type` (TBDrugType, optional): if provided, overrides `efficacy` with drug-specific cure probability from `TBDrugTypeParameters`

### `administer(uids)` method

Draws success/failure for each agent using a Bernoulli distribution with `p=efficacy`. Returns `{'success': uids, 'failure': uids}`.

### Factory functions

```python
tbsim.dots()           # Standard DOTS (85% cure)
tbsim.dots_improved()  # Enhanced DOTS
tbsim.first_line()     # First-line combination (95% cure)
tbsim.second_line()    # Second-line for MDR-TB
```

---

## Layer 2a: `DxDelivery` Intervention

### File

Replaces contents of `tbsim/interventions/tb_diagnostic.py` as `tbsim/interventions/diagnostics.py`

### Inheritance

`DxDelivery(ss.Intervention)`

### Parameters

| parameter | type | default | description |
|-----------|------|---------|-------------|
| `product` | `Dx` | required | the diagnostic product to administer |
| `coverage` | float or `ss.Dist` | `1.0` | fraction of eligible agents who receive the test |
| `eligibility` | callable `(sim) -> uids` | see below | who is eligible for this step |
| `result_state` | str | `'diagnosed'` | people-state to set True for positive results |
| `care_seeking_multiplier` | float | `1.0` | multiplier on care-seeking rate for false negatives |

**Default eligibility** (first step in a cascade):
```python
lambda sim: (sim.people.sought_care & ~sim.people.diagnosed & sim.people.alive).uids
```

### Auto-registration of `result_state`

During `init_pre`, if `result_state` is not `'diagnosed'` and the named state does not already exist on `sim.people`, `DxDelivery` registers it as a `ss.BoolState` with `default=False`.

### `step()` logic

```
1. Compute eligible_uids via self.eligibility(sim)
2. Apply coverage filter
3. Call product.administer(selected_uids) -> {'positive': pos_uids, 'negative': neg_uids}
4. Set ppl.<result_state>[pos_uids] = True
5. Handle false negatives (TB-positive agents who tested negative):
       - apply care_seeking_multiplier (once per agent, guarded)
       - reset sought_care and tested to allow retry
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

## Layer 2b: `TxDelivery` Intervention

### File

Replaces contents of `tbsim/interventions/tb_treatment.py` as `tbsim/interventions/treatments.py`

### Inheritance

`TxDelivery(ss.Intervention)`

### Parameters

| parameter | type | default | description |
|-----------|------|---------|-------------|
| `product` | `Tx` | required | the treatment product |
| `eligibility` | callable `(sim) -> uids` | see below | who is eligible |
| `reseek_multiplier` | float | `2.0` | care-seeking multiplier after failure |
| `reset_flags` | bool | `True` | reset diagnosed/tested on failure |

**Default eligibility:**
```python
lambda sim: diagnosed & active_tb_state & alive
```

### `step()` logic

```
1. Get eligible UIDs (diagnosed, in active TB state, alive)
2. Call tb.start_treatment(uids) -- moves to TREATMENT state
3. Call product.administer(treated_uids) -> {'success': ..., 'failure': ...}
4. Success: set state to CLEARED, reset diagnosed, mark tb_treatment_success
5. Failure: optionally reset diagnosed/tested, apply reseek_multiplier,
            reset sought_care for re-engagement
```

### Results

| result key | description |
|------------|-------------|
| `n_treated` | agents treated this step |
| `n_success` | successful treatments this step |
| `n_failure` | failed treatments this step |
| `cum_success` | cumulative successes |
| `cum_failure` | cumulative failures |

---

## Cascade Example

```python
import tbsim

# Screen -> confirm -> treat
screen = tbsim.DxDelivery(
    product      = tbsim.cad_cxr(),
    coverage     = 0.9,
    result_state = 'screen_positive',
)

confirm = tbsim.DxDelivery(
    product      = tbsim.xpert(),
    coverage     = 0.8,
    eligibility  = lambda sim: sim.people.screen_positive.uids,
    result_state = 'diagnosed',
)

treat = tbsim.TxDelivery(
    product = tbsim.dots(),
)

sim = tbsim.Sim(
    diseases     = tbsim.TB(),
    interventions = [
        tbsim.HealthSeekingBehavior(),
        screen,
        confirm,
        treat,
    ],
)
sim.run()
```

### Single-step usage (no cascade)

```python
sim = tbsim.Sim(
    diseases     = tbsim.TB(),
    interventions = [
        tbsim.HealthSeekingBehavior(),
        tbsim.DxDelivery(product=tbsim.xpert()),
        tbsim.TxDelivery(product=tbsim.dots()),
    ],
)
```

---

## Files Changed

| file | change |
|------|--------|
| `tbsim/interventions/dx_products.py` | **new** -- `Dx` class and factory functions |
| `tbsim/interventions/tx_products.py` | **new** -- `Tx` class and factory functions |
| `tbsim/interventions/diagnostics.py` | **replace** -- delete old classes, write `DxDelivery` |
| `tbsim/interventions/treatments.py` | **replace** -- delete old classes, write `TxDelivery` |
| `tbsim/interventions/__init__.py` | update exports |
| `tbsim/__init__.py` | update exports |
| `tests/test_interventions.py` | update to use new classes |
| `tbsim_examples/run_interventions.py` | update examples |

**Unchanged:** `HealthSeekingBehavior`, `TBProductRoutine`, BCG, TPT, `BetaByYear`, `drug_types.py` (except for rename)

---

## Open Questions (Resolved)

1. **TB_EMOD support** -- Dropped. TBSL only.
2. **False-negative retry** -- Lives in `DxDelivery`, not `HealthSeekingBehavior`.
3. **FujiLAM HIV stratification** -- Handled via optional `hiv` column in DataFrame.
4. **Treatment product complexity** -- Minimal (Option A): product just returns success/failure, delivery handles state transitions.
5. **Reuse of TBProductRoutine** -- Not reused for Dx/Tx delivery; kept for BCG/TPT.
6. **Legacy classes** -- Deleted, not deprecated.
