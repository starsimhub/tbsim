# Diagnostic & Treatment Product/Delivery Refactor

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate all diagnostic and treatment interventions into product (what the test/drug does) and delivery (who/when/how) layers.

**Architecture:** Products (`Dx`, `Tx`) inherit from `ss.Product` and encapsulate test/drug characteristics. Delivery classes (`DxDelivery`, `TxDelivery`) inherit from `ss.Intervention` and handle eligibility, coverage, and outcome side effects. Products return result dicts; delivery classes interpret them.

**Tech Stack:** starsim (`ss.Product`, `ss.Intervention`, `ss.bernoulli`, `ss.BoolState`), pandas DataFrames, numpy

**Spec:** `docs/tb_diagnostic_treatment_refactor_spec.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `tbsim/interventions/dx_products.py` | **New.** `Dx` product class + factory functions (`xpert`, `oral_swab`, `fujilam`, `cad_cxr`) |
| `tbsim/interventions/tx_products.py` | **New.** `Tx` product class + factory functions (`dots`, `dots_improved`, `first_line`, `second_line`) |
| `tbsim/interventions/diagnostics.py` | **New.** `DxDelivery` intervention class |
| `tbsim/interventions/treatments.py` | **New.** `TxDelivery` intervention class |
| `tbsim/interventions/__init__.py` | **Modify.** Update imports to use new files, remove old imports |
| `tbsim/__init__.py` | **Modify.** No changes needed (re-exports from `interventions`) |
| `tests/test_interventions.py` | **Modify.** Rewrite dx/tx tests for new classes (was `test_tbinterventions.py`) |
| `tbsim/interventions/tb_diagnostic.py` | **Delete.** Replaced by `diagnostics.py` + `dx_products.py` |
| `tbsim/interventions/tb_treatment.py` | **Delete.** Replaced by `treatments.py` + `tx_products.py` |
| `tbsim/interventions/tb_drug_types.py` | **Rename** to `drug_types.py` |

**Unchanged:** `interventions.py` (TBProductRoutine), `tb_health_seeking.py`, `tpt.py`, `bcg.py`, `beta.py`, `immigration.py`

---

## Chunk 1: Dx Product

### Task 1: Write `Dx` product class with tests

**Files:**
- Create: `tbsim/interventions/dx_products.py`
- Create: `tests/test_dx_product.py`

- [ ] **Step 1: Write failing test for Dx with simple DataFrame (no optional columns)**

Create `tests/test_dx_product.py`:

```python
"""Tests for the Dx diagnostic product."""

import pytest
import numpy as np
import pandas as pd
import starsim as ss
import tbsim
from tbsim import TBSL


def make_dx_sim(n_agents=200):
    """Create a minimal sim with TB for testing Dx products."""
    pop = ss.People(n_agents=n_agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    sim = ss.Sim(people=pop, diseases=tb, networks=net, pars=pars)
    sim.init()
    return sim


def test_dx_simple_dataframe():
    """Dx with only required columns (state, result, probability) returns correct dict."""
    from tbsim.interventions.dx_products import Dx

    df = pd.DataFrame([
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=1.0),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.0),
    ])
    dx = Dx(df=df, hierarchy=['positive', 'negative'])

    sim = make_dx_sim()
    # Run a few steps so some agents become symptomatic
    for _ in range(20):
        sim.run_one_step()

    tb = tbsim.get_tb(sim)
    all_uids = sim.people.alive.uids
    results = dx.administer(sim, all_uids)

    assert 'positive' in results
    assert 'negative' in results

    # With sensitivity=1.0 for SYMPTOMATIC, all symptomatic agents should test positive
    symptomatic_uids = ss.uids(np.where(np.asarray(tb.state) == TBSL.SYMPTOMATIC)[0])
    symptomatic_alive = np.intersect1d(symptomatic_uids, all_uids)
    positive_uids = np.asarray(results['positive'])
    for uid in symptomatic_alive:
        assert uid in positive_uids, f"Symptomatic agent {uid} should test positive with sensitivity=1.0"

    # Agents not in df states should default to 'negative' (last in hierarchy)
    non_symptomatic = all_uids[~np.isin(all_uids, symptomatic_alive)]
    negative_uids = np.asarray(results['negative'])
    for uid in non_symptomatic[:10]:  # spot-check
        assert uid in negative_uids, f"Non-symptomatic agent {uid} should default to negative"


def test_dx_age_stratified():
    """Dx with age_min/age_max columns filters agents by age."""
    from tbsim.interventions.dx_products import Dx

    df = pd.DataFrame([
        # Adults: high sensitivity
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=1.0, age_min=15, age_max=np.inf),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.0, age_min=15, age_max=np.inf),
        # Children: zero sensitivity (always negative)
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.0, age_min=0, age_max=15),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=1.0, age_min=0, age_max=15),
    ])
    dx = Dx(df=df, hierarchy=['positive', 'negative'])

    sim = make_dx_sim(n_agents=500)
    for _ in range(20):
        sim.run_one_step()

    tb = tbsim.get_tb(sim)
    all_uids = sim.people.alive.uids
    results = dx.administer(sim, all_uids)

    # Check that symptomatic children test negative
    symptomatic = np.asarray(tb.state) == TBSL.SYMPTOMATIC
    children = np.asarray(sim.people.age) < 15
    child_symptomatic = ss.uids(np.where(symptomatic & children)[0])
    child_symptomatic_alive = np.intersect1d(child_symptomatic, all_uids)

    negative_uids = np.asarray(results['negative'])
    for uid in child_symptomatic_alive:
        assert uid in negative_uids, f"Child symptomatic agent {uid} should test negative with sensitivity=0.0"


def test_dx_probabilities_sum_to_one_validation():
    """Dx raises an error if probabilities don't sum to 1.0 per group."""
    from tbsim.interventions.dx_products import Dx

    df = pd.DataFrame([
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.5),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.3),  # sums to 0.8, not 1.0
    ])
    with pytest.raises(ValueError, match="sum to 1.0"):
        Dx(df=df, hierarchy=['positive', 'negative'])


def test_dx_default_hierarchy():
    """Dx infers hierarchy from df.result.unique() if not provided."""
    from tbsim.interventions.dx_products import Dx

    df = pd.DataFrame([
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.9),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.1),
    ])
    dx = Dx(df=df)
    assert list(dx.hierarchy) == ['positive', 'negative']
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_dx_product.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tbsim.interventions.dx_products'`

- [ ] **Step 3: Implement `Dx` class**

Create `tbsim/interventions/dx_products.py`:

```python
"""Diagnostic products for TB testing."""

import numpy as np
import pandas as pd
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['Dx', 'xpert', 'oral_swab', 'fujilam', 'cad_cxr']

# Optional filter columns recognized by Dx
_FILTER_COLS = ['age_min', 'age_max', 'hiv']


class Dx(ss.Product):
    """
    TB diagnostic product defined by a DataFrame of state-to-result probabilities.

    Args:
        df: DataFrame with required columns (state, result, probability) and
            optional filter columns (age_min, age_max, hiv). Only include
            filter columns relevant to the specific diagnostic.
        hierarchy: List of result strings in priority order, e.g. ['positive', 'negative'].
            Agents in states not listed in df get the last entry (default negative).
    """

    def __init__(self, df, hierarchy=None, **kwargs):
        super().__init__(**kwargs)
        self.df = df.copy()
        self.hierarchy = hierarchy if hierarchy is not None else list(df.result.unique())
        self._filter_cols = [c for c in _FILTER_COLS if c in df.columns]
        self._validate()

        # Pre-build a multinomial distribution for stochastic draws
        n_results = len(self.hierarchy)
        self.result_dist = ss.rv_discrete(
            values=(np.arange(n_results), np.ones(n_results) / n_results)
        )

    def _validate(self):
        """Check that probabilities sum to 1.0 per group."""
        required = {'state', 'result', 'probability'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        group_cols = ['state'] + self._filter_cols
        for name, group in self.df.groupby(group_cols):
            total = group.probability.sum()
            if not np.isclose(total, 1.0):
                raise ValueError(
                    f"Probabilities for {dict(zip(group_cols, name if isinstance(name, tuple) else [name]))} "
                    f"sum to {total}, not 1.0"
                )

    @property
    def default_value(self):
        return len(self.hierarchy) - 1

    def administer(self, sim, uids):
        """
        Administer diagnostic test to agents.

        Returns:
            dict keyed by hierarchy strings, e.g. {'positive': uids, 'negative': uids}
        """
        tb = tbsim.get_tb(sim)
        ppl = sim.people

        # Pre-fill with default (last hierarchy entry = most benign result)
        results = np.full(len(uids), self.default_value, dtype=int)
        uid_to_idx = {int(u): i for i, u in enumerate(uids)}

        # Get agent attributes needed for filtering
        tb_states = np.asarray(tb.state[uids])
        ages = np.asarray(ppl.age[uids]) if ('age_min' in self._filter_cols or 'age_max' in self._filter_cols) else None

        hiv_states = None
        if 'hiv' in self._filter_cols and hasattr(sim.diseases, 'hiv'):
            hiv_states = np.asarray(sim.diseases.hiv.infected[uids])

        # Group df rows by (state + filter columns)
        group_cols = ['state'] + self._filter_cols
        for group_key, group_df in self.df.groupby(group_cols):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            vals = dict(zip(group_cols, group_key))

            # Build mask for agents matching this group
            mask = tb_states == vals['state']
            if 'age_min' in vals:
                mask = mask & (ages >= vals['age_min'])
            if 'age_max' in vals:
                mask = mask & (ages < vals['age_max'])
            if 'hiv' in vals and hiv_states is not None:
                mask = mask & (hiv_states == vals['hiv'])

            matched_idx = np.where(mask)[0]
            if len(matched_idx) == 0:
                continue

            matched_uids = uids[matched_idx]

            # Get probabilities in hierarchy order
            probs = []
            for r in self.hierarchy:
                row = group_df[group_df.result == r]
                probs.append(row.probability.values[0] if len(row) > 0 else 0.0)
            self.result_dist.pk = probs

            # Draw results and take minimum (best result wins if multiple matches)
            draws = self.result_dist.rvs(matched_uids) - matched_uids
            results[matched_idx] = np.minimum(draws, results[matched_idx])

        # Convert to dict of UIDs per result
        output = {}
        for i, label in enumerate(self.hierarchy):
            output[label] = ss.uids(uids[results == i])
        return output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_dx_product.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tbsim/interventions/dx_products.py tests/test_dx_product.py
git commit -m "feat: add Dx diagnostic product class with DataFrame-based test definitions"
```

---

### Task 2: Write Dx factory functions with tests

**Files:**
- Modify: `tbsim/interventions/dx_products.py`
- Modify: `tests/test_dx_product.py`

- [ ] **Step 1: Write failing tests for factory functions**

Append to `tests/test_dx_product.py`:

```python
def test_xpert_factory():
    """xpert() returns a Dx with age-stratified, state-stratified DataFrame."""
    from tbsim.interventions.dx_products import xpert
    dx = xpert()
    assert isinstance(dx, Dx)
    assert 'age_min' in dx.df.columns
    assert 'age_max' in dx.df.columns
    assert 'hiv' not in dx.df.columns
    assert set(dx.hierarchy) == {'positive', 'negative'}
    # Check adult symptomatic sensitivity
    adult_symp = dx.df[(dx.df.state == TBSL.SYMPTOMATIC) & (dx.df.age_min == 15) & (dx.df.result == 'positive')]
    assert np.isclose(adult_symp.probability.values[0], 0.909)


def test_oral_swab_factory():
    """oral_swab() returns a Dx with age and state stratification."""
    from tbsim.interventions.dx_products import oral_swab
    dx = oral_swab()
    assert isinstance(dx, Dx)
    assert 'hiv' not in dx.df.columns


def test_fujilam_factory():
    """fujilam() returns a Dx with HIV stratification."""
    from tbsim.interventions.dx_products import fujilam
    dx = fujilam()
    assert isinstance(dx, Dx)
    assert 'hiv' in dx.df.columns


def test_cad_cxr_factory():
    """cad_cxr() returns a Dx product."""
    from tbsim.interventions.dx_products import cad_cxr
    dx = cad_cxr()
    assert isinstance(dx, Dx)


def test_xpert_runs_in_sim():
    """xpert() product can be administered in a running sim."""
    from tbsim.interventions.dx_products import xpert
    dx = xpert()
    sim = make_dx_sim(n_agents=200)
    for _ in range(20):
        sim.run_one_step()
    results = dx.administer(sim, sim.people.alive.uids)
    assert len(results['positive']) + len(results['negative']) == len(sim.people.alive.uids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_dx_product.py -v -k "factory or runs_in_sim"`
Expected: FAIL — `ImportError: cannot import name 'xpert'`

- [ ] **Step 3: Implement factory functions**

Add to `tbsim/interventions/dx_products.py`:

```python
def xpert():
    """Xpert MTB/RIF — age-stratified (child/adult), TB-state-stratified."""
    df = pd.DataFrame([
        # Adults (age >= 15)
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.909, age_min=15, age_max=np.inf),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.091, age_min=15, age_max=np.inf),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.775, age_min=15, age_max=np.inf),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.225, age_min=15, age_max=np.inf),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.775, age_min=15, age_max=np.inf),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.225, age_min=15, age_max=np.inf),
        # Children (age < 15)
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.73, age_min=0, age_max=15),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.27, age_min=0, age_max=15),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.73, age_min=0, age_max=15),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.27, age_min=0, age_max=15),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.73, age_min=0, age_max=15),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.27, age_min=0, age_max=15),
    ])
    return Dx(df=df, hierarchy=['positive', 'negative'])


def oral_swab():
    """Oral swab — age-stratified, TB-state-stratified."""
    df = pd.DataFrame([
        # Adults
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.80, age_min=15, age_max=np.inf),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.20, age_min=15, age_max=np.inf),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.30, age_min=15, age_max=np.inf),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.70, age_min=15, age_max=np.inf),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.30, age_min=15, age_max=np.inf),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.70, age_min=15, age_max=np.inf),
        # Children
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.25, age_min=0, age_max=15),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.75, age_min=0, age_max=15),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.25, age_min=0, age_max=15),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.75, age_min=0, age_max=15),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.25, age_min=0, age_max=15),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.75, age_min=0, age_max=15),
    ])
    return Dx(df=df, hierarchy=['positive', 'negative'])


def fujilam():
    """FujiLAM — HIV-stratified, age-stratified."""
    df = pd.DataFrame([
        # HIV+, adults
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.75,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.25,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.75,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.25,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.75,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.25,  age_min=15, age_max=np.inf, hiv=True),
        # HIV-, adults
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.58,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.42,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.58,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.42,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.58,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.42,  age_min=15, age_max=np.inf, hiv=False),
        # HIV+, children
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.579, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.421, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.579, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.421, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.579, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.421, age_min=0, age_max=15, hiv=True),
        # HIV-, children
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.51,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.49,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.51,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.49,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.51,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.49,  age_min=0, age_max=15, hiv=False),
    ])
    return Dx(df=df, hierarchy=['positive', 'negative'])


def cad_cxr():
    """CAD chest X-ray — simple, no age/HIV stratification."""
    df = pd.DataFrame([
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.66),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.34),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.66),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.34),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.66),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.34),
    ])
    return Dx(df=df, hierarchy=['positive', 'negative'])
```

- [ ] **Step 4: Run all Dx tests**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_dx_product.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tbsim/interventions/dx_products.py tests/test_dx_product.py
git commit -m "feat: add Dx factory functions (xpert, oral_swab, fujilam, cad_cxr)"
```

---

## Chunk 2: Tx Product

### Task 3: Write `Tx` product class and factory functions with tests

**Files:**
- Create: `tbsim/interventions/tx_products.py`
- Create: `tests/test_tx_product.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tx_product.py`:

```python
"""Tests for the Tx treatment product."""

import pytest
import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL


def make_tx_sim(n_agents=200):
    """Create a minimal sim with TB for testing Tx products."""
    pop = ss.People(n_agents=n_agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    sim = ss.Sim(people=pop, diseases=tb, networks=net, pars=pars)
    sim.init()
    return sim


def test_tx_basic():
    """Tx with efficacy=1.0 returns all agents as success."""
    from tbsim.interventions.tx_products import Tx
    tx = Tx(efficacy=1.0)
    sim = make_tx_sim()
    for _ in range(10):
        sim.run_one_step()

    uids = sim.people.alive.uids[:20]
    results = tx.administer(sim, uids)
    assert 'success' in results
    assert 'failure' in results
    assert len(results['success']) == len(uids)
    assert len(results['failure']) == 0


def test_tx_zero_efficacy():
    """Tx with efficacy=0.0 returns all agents as failure."""
    from tbsim.interventions.tx_products import Tx
    tx = Tx(efficacy=0.0)
    sim = make_tx_sim()
    for _ in range(10):
        sim.run_one_step()

    uids = sim.people.alive.uids[:20]
    results = tx.administer(sim, uids)
    assert len(results['success']) == 0
    assert len(results['failure']) == len(uids)


def test_tx_drug_type_overrides_efficacy():
    """Tx with drug_type overrides the efficacy parameter."""
    from tbsim.interventions.tx_products import Tx
    from tbsim.interventions.tb_drug_types import TBDrugType
    tx = Tx(efficacy=0.5, drug_type=TBDrugType.FIRST_LINE_COMBO)
    # FIRST_LINE_COMBO has 95% cure rate, which should override 0.5
    assert np.isclose(tx.efficacy, 0.95)


def test_dots_factory():
    """dots() returns a Tx with DOTS cure probability."""
    from tbsim.interventions.tx_products import dots
    tx = dots()
    assert np.isclose(tx.efficacy, 0.85)


def test_first_line_factory():
    """first_line() returns a Tx with first-line cure probability."""
    from tbsim.interventions.tx_products import first_line
    tx = first_line()
    assert np.isclose(tx.efficacy, 0.95)


def test_dots_runs_in_sim():
    """dots() product can be administered in a running sim."""
    from tbsim.interventions.tx_products import dots
    tx = dots()
    sim = make_tx_sim()
    for _ in range(10):
        sim.run_one_step()
    uids = sim.people.alive.uids[:10]
    results = tx.administer(sim, uids)
    assert len(results['success']) + len(results['failure']) == len(uids)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_tx_product.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tbsim.interventions.tx_products'`

- [ ] **Step 3: Implement `Tx` class and factory functions**

Create `tbsim/interventions/tx_products.py`:

```python
"""Treatment products for TB."""

import starsim as ss
from .tb_drug_types import TBDrugType, TBDrugTypeParameters

__all__ = ['Tx', 'dots', 'dots_improved', 'first_line', 'second_line']


class Tx(ss.Product):
    """
    TB treatment product that encapsulates drug efficacy.

    Args:
        efficacy: Probability of treatment success (0-1). Default 0.85.
        drug_type: If provided, overrides efficacy with drug-specific cure probability.
    """

    def __init__(self, efficacy=0.85, drug_type=None, **kwargs):
        super().__init__(**kwargs)
        if drug_type is not None:
            params = TBDrugTypeParameters.create_parameters_for_type(drug_type)
            self.efficacy = params.cure_prob
        else:
            self.efficacy = efficacy
        self.dist_success = ss.bernoulli(p=self.efficacy)

    def administer(self, sim, uids):
        """
        Administer treatment to agents.

        Returns:
            dict with 'success' and 'failure' UIDs.
        """
        success_uids, failure_uids = self.dist_success.filter(uids, both=True)
        return {'success': success_uids, 'failure': failure_uids}


def dots():
    """Standard DOTS (85% cure)."""
    return Tx(drug_type=TBDrugType.DOTS)


def dots_improved():
    """Enhanced DOTS (90% cure)."""
    return Tx(drug_type=TBDrugType.DOTS_IMPROVED)


def first_line():
    """First-line combination therapy (95% cure)."""
    return Tx(drug_type=TBDrugType.FIRST_LINE_COMBO)


def second_line():
    """Second-line therapy for MDR-TB (75% cure)."""
    return Tx(drug_type=TBDrugType.SECOND_LINE_COMBO)
```

- [ ] **Step 4: Run tests**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_tx_product.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tbsim/interventions/tx_products.py tests/test_tx_product.py
git commit -m "feat: add Tx treatment product class with factory functions"
```

---

## Chunk 3: DxDelivery Intervention

### Task 4: Write `DxDelivery` intervention with tests

**Files:**
- Create: `tbsim/interventions/diagnostics.py`
- Create: `tests/test_dx_delivery.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dx_delivery.py`:

```python
"""Tests for the DxDelivery diagnostic intervention."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL


def make_sim(n_agents=200):
    """Create sim with HSB + Dx delivery for testing."""
    pop = ss.People(n_agents=n_agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'), rand_seed=42)
    return pop, tb, net, pars


def test_dx_delivery_runs():
    """DxDelivery completes a full run with HealthSeekingBehavior."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()

    # Should have tested some people
    assert sim.results.dxdelivery.n_tested.values.sum() > 0
    assert sim.results.dxdelivery.n_positive.values.sum() > 0


def test_dx_delivery_diagnoses_agents():
    """DxDelivery sets diagnosed=True for positive results."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim(n_agents=500)
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()

    # Some agents should be diagnosed
    n_diagnosed = np.sum(np.asarray(sim.people.diagnosed))
    assert n_diagnosed > 0


def test_dx_delivery_custom_result_state():
    """DxDelivery with custom result_state auto-registers and sets that state."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert(), result_state='screen_positive')
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()

    # The custom state should exist and have some True values
    assert hasattr(sim.people, 'screen_positive')
    n_screen_pos = np.sum(np.asarray(sim.people.screen_positive))
    assert n_screen_pos > 0


def test_dx_delivery_cascade():
    """Two DxDelivery steps can be chained: screen -> confirm."""
    from tbsim.interventions.dx_products import cad_cxr, xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim(n_agents=500)
    hsb = tbsim.HealthSeekingBehavior()

    screen = DxDelivery(
        product=cad_cxr(),
        coverage=0.9,
        result_state='screen_positive',
    )
    confirm = DxDelivery(
        product=xpert(),
        coverage=0.8,
        eligibility=lambda sim: sim.people.screen_positive.uids,
        result_state='diagnosed',
    )

    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, screen, confirm], pars=pars)
    sim.run()

    # Both steps should have recorded results
    assert sim.results.dxdelivery.n_tested.values.sum() > 0
    assert sim.results.dxdelivery_1.n_tested.values.sum() > 0


def test_dx_delivery_coverage():
    """DxDelivery with coverage < 1.0 tests fewer agents."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim(n_agents=500)
    hsb = tbsim.HealthSeekingBehavior()
    dx_full = DxDelivery(product=xpert(), coverage=1.0)
    sim_full = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx_full], pars=pars)
    sim_full.run()
    n_full = sim_full.results.dxdelivery.n_tested.values.sum()

    pop2, tb2, net2, pars2 = make_sim(n_agents=500)
    hsb2 = tbsim.HealthSeekingBehavior()
    dx_half = DxDelivery(product=xpert(), coverage=0.5)
    sim_half = ss.Sim(people=pop2, diseases=tb2, networks=net2, interventions=[hsb2, dx_half], pars=pars2)
    sim_half.run()
    n_half = sim_half.results.dxdelivery.n_tested.values.sum()

    # Half coverage should test substantially fewer (with noise)
    assert n_half < n_full
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_dx_delivery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tbsim.interventions.diagnostics'`

- [ ] **Step 3: Implement `DxDelivery`**

Create `tbsim/interventions/diagnostics.py`:

```python
"""Diagnostic delivery intervention for TB testing."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['DxDelivery']

_ACTIVE_TB_STATES = [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]


class DxDelivery(ss.Intervention):
    """
    Delivers a diagnostic product to eligible agents.

    Handles eligibility (default: sought care, not yet diagnosed, alive),
    coverage filtering, result-state setting, and false-negative retry logic.

    Args:
        product: A Dx product instance.
        coverage: Fraction of eligible agents tested (float or ss.Dist). Default 1.0.
        eligibility: Callable (sim) -> uids. Default: sought_care & ~diagnosed & alive.
        result_state: People-state name set True on positive result. Default 'diagnosed'.
        care_seeking_multiplier: Multiplier on care-seeking rate for false negatives. Default 1.0.
    """

    def __init__(self, product, coverage=1.0, eligibility=None, result_state='diagnosed',
                 care_seeking_multiplier=1.0, **kwargs):
        super().__init__(**kwargs)
        self.product = product
        self.coverage = coverage
        self._eligibility_fn = eligibility
        self.result_state = result_state
        self.care_seeking_multiplier = care_seeking_multiplier

        # Coverage distribution
        if not isinstance(coverage, ss.Dist):
            self.dist_coverage = ss.bernoulli(p=coverage)
        else:
            self.dist_coverage = coverage

        # Person-level states
        self.define_states(
            ss.BoolState('sought_care', default=False),
            ss.BoolState('diagnosed', default=False),
            ss.BoolArr('tested', default=False),
            ss.IntArr('n_times_tested', default=0),
            ss.BoolState('test_result', default=False),
            ss.FloatArr('care_seeking_multiplier', default=1.0),
            ss.BoolState('multiplier_applied', default=False),
        )

        # Tracking for results
        self._n_tested = 0
        self._n_positive = 0
        self._n_negative = 0

    def init_pre(self, sim):
        super().init_pre(sim)
        # Auto-register custom result_state if needed
        if self.result_state != 'diagnosed' and not hasattr(sim.people, self.result_state):
            sim.people.add_state(ss.BoolState(self.result_state, default=False))

    def _get_eligible(self, sim):
        """Get eligible UIDs using custom or default eligibility."""
        if self._eligibility_fn is not None:
            return ss.uids(self._eligibility_fn(sim))
        return (self.sought_care & (~self.diagnosed) & sim.people.alive).uids

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        eligible = self._get_eligible(sim)
        if len(eligible) == 0:
            self._n_tested = self._n_positive = self._n_negative = 0
            return

        # Coverage filter
        selected = self.dist_coverage.filter(eligible)
        if len(selected) == 0:
            self._n_tested = self._n_positive = self._n_negative = 0
            return

        # Administer diagnostic product
        results = self.product.administer(sim, selected)
        pos_uids = results.get('positive', ss.uids())
        neg_uids = results.get('negative', ss.uids())

        # Set result state
        result_arr = getattr(ppl, self.result_state)
        result_arr[pos_uids] = True

        # Update tracking states
        self.tested[selected] = True
        self.n_times_tested[selected] += 1
        self.test_result[selected] = np.isin(selected, pos_uids)

        # Handle false negatives: TB-positive agents who tested negative
        tb_states = np.asarray(tb.state[neg_uids])
        has_tb = np.isin(tb_states, _ACTIVE_TB_STATES)
        false_neg_uids = neg_uids[has_tb]

        if len(false_neg_uids) > 0 and self.care_seeking_multiplier != 1.0:
            unboosted = false_neg_uids[~self.multiplier_applied[false_neg_uids]]
            if len(unboosted) > 0:
                csm = getattr(ppl, 'care_seeking_multiplier', None)
                if csm is not None:
                    csm[unboosted] *= self.care_seeking_multiplier
                self.multiplier_applied[unboosted] = True

        # Reset flags for false negatives to allow retry
        if len(false_neg_uids) > 0:
            self.sought_care[false_neg_uids] = False
            self.tested[false_neg_uids] = False

        # Store for results
        self._n_tested = len(selected)
        self._n_positive = len(pos_uids)
        self._n_negative = len(neg_uids)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_tested', dtype=int),
            ss.Result('n_positive', dtype=int),
            ss.Result('n_negative', dtype=int),
            ss.Result('cum_positive', dtype=int),
            ss.Result('cum_negative', dtype=int),
        )

    def update_results(self):
        self.results['n_tested'][self.ti] = self._n_tested
        self.results['n_positive'][self.ti] = self._n_positive
        self.results['n_negative'][self.ti] = self._n_negative

        if self.ti > 0:
            self.results['cum_positive'][self.ti] = self.results['cum_positive'][self.ti - 1] + self._n_positive
            self.results['cum_negative'][self.ti] = self.results['cum_negative'][self.ti - 1] + self._n_negative
        else:
            self.results['cum_positive'][self.ti] = self._n_positive
            self.results['cum_negative'][self.ti] = self._n_negative

        self._n_tested = self._n_positive = self._n_negative = 0
```

- [ ] **Step 4: Run tests**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_dx_delivery.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tbsim/interventions/diagnostics.py tests/test_dx_delivery.py
git commit -m "feat: add DxDelivery intervention with cascade support"
```

---

## Chunk 4: TxDelivery Intervention

### Task 5: Write `TxDelivery` intervention with tests

**Files:**
- Create: `tbsim/interventions/treatments.py`
- Create: `tests/test_tx_delivery.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tx_delivery.py`:

```python
"""Tests for the TxDelivery treatment intervention."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL


def make_sim(n_agents=200):
    pop = ss.People(n_agents=n_agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'), rand_seed=42)
    return pop, tb, net, pars


def test_tx_delivery_runs():
    """TxDelivery completes a full run with HSB + DxDelivery upstream."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery
    from tbsim.interventions.tx_products import dots
    from tbsim.interventions.treatments import TxDelivery

    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert())
    tx = TxDelivery(product=dots())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()

    assert sim.results.txdelivery.n_treated.values.sum() > 0


def test_tx_delivery_clears_infection():
    """Successful treatment moves agents to CLEARED state."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery
    from tbsim.interventions.tx_products import Tx
    from tbsim.interventions.treatments import TxDelivery

    pop, tb, net, pars = make_sim(n_agents=500)
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert())
    tx = TxDelivery(product=Tx(efficacy=1.0))  # 100% success
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()

    # With 100% efficacy, all treated should succeed
    n_success = sim.results.txdelivery.cum_success.values[-1]
    n_failure = sim.results.txdelivery.cum_failure.values[-1]
    assert n_success > 0
    assert n_failure == 0


def test_tx_delivery_failure_resets_flags():
    """Treatment failure resets diagnosed/tested flags for re-engagement."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery
    from tbsim.interventions.tx_products import Tx
    from tbsim.interventions.treatments import TxDelivery

    pop, tb, net, pars = make_sim(n_agents=500)
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert())
    tx = TxDelivery(product=Tx(efficacy=0.0), reseek_multiplier=3.0)  # 0% success
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()

    # With 0% efficacy, all treated should fail
    n_success = sim.results.txdelivery.cum_success.values[-1]
    n_failure = sim.results.txdelivery.cum_failure.values[-1]
    assert n_success == 0
    assert n_failure > 0


def test_full_cascade():
    """Full HSB -> screen -> confirm -> treat cascade runs end-to-end."""
    from tbsim.interventions.dx_products import cad_cxr, xpert
    from tbsim.interventions.diagnostics import DxDelivery
    from tbsim.interventions.tx_products import dots
    from tbsim.interventions.treatments import TxDelivery

    pop, tb, net, pars = make_sim(n_agents=500)
    hsb = tbsim.HealthSeekingBehavior()
    screen = DxDelivery(product=cad_cxr(), coverage=0.9, result_state='screen_positive')
    confirm = DxDelivery(product=xpert(), coverage=0.8,
                         eligibility=lambda sim: sim.people.screen_positive.uids,
                         result_state='diagnosed')
    treat = TxDelivery(product=dots())

    sim = ss.Sim(people=pop, diseases=tb, networks=net,
                 interventions=[hsb, screen, confirm, treat], pars=pars)
    sim.run()

    # All three steps should have done work
    assert sim.results.txdelivery.n_treated.values.sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_tx_delivery.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tbsim.interventions.treatments'`

- [ ] **Step 3: Implement `TxDelivery`**

Create `tbsim/interventions/treatments.py`:

```python
"""Treatment delivery intervention for TB."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['TxDelivery']

_ACTIVE_TB_STATES = [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]


class TxDelivery(ss.Intervention):
    """
    Delivers a treatment product to diagnosed agents.

    Handles eligibility (default: diagnosed, active TB, alive), treatment
    initiation via tb.start_treatment(), and success/failure outcomes.

    Args:
        product: A Tx product instance.
        eligibility: Callable (sim) -> uids. Default: diagnosed & active_tb & alive.
        reseek_multiplier: Care-seeking multiplier after failure. Default 2.0.
        reset_flags: Whether to reset diagnosed/tested on failure. Default True.
    """

    def __init__(self, product, eligibility=None, reseek_multiplier=2.0,
                 reset_flags=True, **kwargs):
        super().__init__(**kwargs)
        self.product = product
        self._eligibility_fn = eligibility
        self.reseek_multiplier = reseek_multiplier
        self.reset_flags = reset_flags

        self.define_states(
            ss.BoolState('sought_care', default=False),
            ss.BoolState('diagnosed', default=False),
            ss.BoolArr('tested', default=False),
            ss.IntArr('n_times_treated', default=0),
            ss.BoolState('tb_treatment_success', default=False),
            ss.BoolArr('treatment_failure', default=False),
            ss.FloatArr('care_seeking_multiplier', default=1.0),
            ss.BoolState('multiplier_applied', default=False),
        )

        self._n_treated = 0
        self._n_success = 0
        self._n_failure = 0

    def _get_eligible(self, sim):
        """Get eligible UIDs using custom or default eligibility."""
        if self._eligibility_fn is not None:
            return ss.uids(self._eligibility_fn(sim))
        tb = tbsim.get_tb(sim)
        diagnosed_uids = (self.diagnosed & sim.people.alive).uids
        active_tb_uids = np.where(np.isin(np.asarray(tb.state), _ACTIVE_TB_STATES))[0]
        return ss.uids(np.intersect1d(diagnosed_uids, active_tb_uids))

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        uids = self._get_eligible(sim)
        if len(uids) == 0:
            self._n_treated = self._n_success = self._n_failure = 0
            return

        # Start treatment (moves active -> TREATMENT state in TB model)
        tb.start_treatment(uids)
        tx_uids = uids[tb.on_treatment[uids]]

        if len(tx_uids) == 0:
            self._n_treated = self._n_success = self._n_failure = 0
            return

        self.n_times_treated[tx_uids] += 1

        # Administer treatment product
        results = self.product.administer(sim, tx_uids)
        success_uids = results['success']
        failure_uids = results['failure']

        # Successful treatment clears infection
        if len(success_uids) > 0:
            tb.state[success_uids] = TBSL.CLEARED
            tb.on_treatment[success_uids] = False
            tb.susceptible[success_uids] = True
            tb.infected[success_uids] = False
            self.diagnosed[success_uids] = False
            self.tb_treatment_success[success_uids] = True

        # Handle failures
        if len(failure_uids) > 0:
            self.treatment_failure[failure_uids] = True

            if self.reset_flags:
                self.diagnosed[failure_uids] = False
                self.tested[failure_uids] = False

            self.sought_care[failure_uids] = False
            self.care_seeking_multiplier[failure_uids] *= self.reseek_multiplier
            self.multiplier_applied[failure_uids] = True

        self._n_treated = len(tx_uids)
        self._n_success = len(success_uids)
        self._n_failure = len(failure_uids)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_success', dtype=int),
            ss.Result('n_failure', dtype=int),
            ss.Result('cum_success', dtype=int),
            ss.Result('cum_failure', dtype=int),
        )

    def update_results(self):
        self.results['n_treated'][self.ti] = self._n_treated
        self.results['n_success'][self.ti] = self._n_success
        self.results['n_failure'][self.ti] = self._n_failure

        if self.ti > 0:
            self.results['cum_success'][self.ti] = self.results['cum_success'][self.ti - 1] + self._n_success
            self.results['cum_failure'][self.ti] = self.results['cum_failure'][self.ti - 1] + self._n_failure
        else:
            self.results['cum_success'][self.ti] = self._n_success
            self.results['cum_failure'][self.ti] = self._n_failure

        self._n_treated = self._n_success = self._n_failure = 0
```

- [ ] **Step 4: Run tests**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_tx_delivery.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tbsim/interventions/treatments.py tests/test_tx_delivery.py
git commit -m "feat: add TxDelivery intervention with success/failure handling"
```

---

## Chunk 5: Wiring & Cleanup

### Task 6: Update exports and rename files

**Files:**
- Modify: `tbsim/interventions/__init__.py`
- Rename: `tbsim/interventions/tb_drug_types.py` -> `tbsim/interventions/drug_types.py`
- Delete: `tbsim/interventions/tb_diagnostic.py`
- Delete: `tbsim/interventions/tb_treatment.py`

- [ ] **Step 1: Update `tbsim/interventions/__init__.py`**

Replace contents with:

```python
"""
TBsim Interventions Module
"""

from .interventions import *
from .beta import *
from .tpt import *
from .bcg import *
from .drug_types import *
from .tb_health_seeking import *
from .dx_products import *
from .tx_products import *
from .diagnostics import *
from .treatments import *
```

- [ ] **Step 2: Rename `tb_drug_types.py` to `drug_types.py`**

```bash
git mv tbsim/interventions/tb_drug_types.py tbsim/interventions/drug_types.py
```

Update the import in `tx_products.py`:
```python
# Change:
from .tb_drug_types import TBDrugType, TBDrugTypeParameters
# To:
from .drug_types import TBDrugType, TBDrugTypeParameters
```

- [ ] **Step 3: Delete old files**

```bash
git rm tbsim/interventions/tb_diagnostic.py
git rm tbsim/interventions/tb_treatment.py
```

- [ ] **Step 4: Fix any remaining imports of old module names**

Search for and update any imports of `tb_drug_types`, `tb_diagnostic`, or `tb_treatment` across the codebase. Key files to check:
- `tbsim/interventions/tb_treatment.py` imported `from .tb_drug_types import ...` — now deleted
- `tests/test_tbinterventions.py` — will be updated in next task

- [ ] **Step 5: Run existing tests to verify nothing is broken**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/ -v --ignore=tests/test_tbinterventions.py`
Expected: All tests PASS (the old test file is ignored since it references deleted classes)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: wire up new exports, rename drug_types, delete old diagnostic/treatment classes"
```

---

### Task 7: Update tests and examples

**Files:**
- Modify: `tests/test_tbinterventions.py` (rename to `tests/test_interventions.py`)
- Modify: `tbsim_examples/run_tb_interventions.py` (rename to `tbsim_examples/run_interventions.py`)

- [ ] **Step 1: Rewrite test file**

Rename and update `tests/test_tbinterventions.py` to `tests/test_interventions.py`. Keep the BetaByYear, HealthSeekingBehavior, and Immigration tests unchanged. Replace the diagnostic and treatment tests:

```python
# Replace the TBDiagnostic, EnhancedTBDiagnostic, TBTreatment, EnhancedTBTreatment test sections with:

# ---------------------------------------------------------------------------
# DxDelivery tests
# ---------------------------------------------------------------------------

def test_dx_delivery_runs():
    """DxDelivery completes a full run with TB_LSHTM + HealthSeekingBehavior."""
    pop, tb, net, pars = make_sim(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.DxDelivery(product=tbsim.xpert())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()


def test_dx_delivery_with_acute():
    """DxDelivery completes a full run with TB_LSHTM_Acute."""
    pop, tb, net, pars = make_sim_acute(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.DxDelivery(product=tbsim.xpert())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()


# ---------------------------------------------------------------------------
# TxDelivery tests
# ---------------------------------------------------------------------------

def test_tx_delivery_runs():
    """TxDelivery completes a full run with TB_LSHTM (requires HSB + Dx upstream)."""
    pop, tb, net, pars = make_sim(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.DxDelivery(product=tbsim.xpert())
    tx = tbsim.TxDelivery(product=tbsim.dots())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()


def test_tx_delivery_with_acute():
    """TxDelivery completes a full run with TB_LSHTM_Acute."""
    pop, tb, net, pars = make_sim_acute(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.DxDelivery(product=tbsim.xpert())
    tx = tbsim.TxDelivery(product=tbsim.dots())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()
```

- [ ] **Step 2: Update example file**

Rename and update `tbsim_examples/run_tb_interventions.py` to use the new API:

```python
# Replace old diagnostic/treatment usage with:
interventions.append(tbsim.DxDelivery(product=tbsim.xpert()))
interventions.append(tbsim.TxDelivery(product=tbsim.dots()))
```

- [ ] **Step 3: Run all tests**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/test_interventions.py tests/test_dx_product.py tests/test_tx_product.py tests/test_dx_delivery.py tests/test_tx_delivery.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: update tests and examples for new Dx/Tx product-delivery API"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd /home/cliffk/idm/tbsim && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify imports work at top level**

```bash
cd /home/cliffk/idm/tbsim && python -c "
import tbsim
# Products
print(tbsim.Dx)
print(tbsim.Tx)
# Factories
print(tbsim.xpert())
print(tbsim.dots())
# Delivery
print(tbsim.DxDelivery)
print(tbsim.TxDelivery)
print('All imports OK')
"
```
Expected: All classes print without error

- [ ] **Step 3: Commit any final fixes**

If needed, commit any remaining fixes.
