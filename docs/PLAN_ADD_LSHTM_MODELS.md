# Plan: Add TB_LSHTM and TB_LSHTM_Acute Models to tbsim

**Goal:** Add the two LSHTM disease models (`TB_LSHTM` and `TB_LSHTM_Acute`) to the **tbsim** package so they can be imported and used directly from tbsim without requiring **tb_acf**.

**Scope:** This plan focuses ONLY on adding the two model classes. It does NOT include:
- Changes to analyzers
- Factory functions or model selection helpers
- Migration of TB_LSHTM scripts
- TBSL enum export (unless required by the models internally)

**Status:** Plan only — not yet implemented.

---

## 1. Current State

### 1.1 What Needs to Be Added

| Model | Current Location | What It Is |
|-------|-----------------|------------|
| **TB_LSHTM** | `TB_LSHTM/tb_lshtm.py` | Base LSHTM natural-history model (no acute state) |
| **TB_LSHTM_Acute** | `TB_LSHTM/tb_lshtm.py` | LSHTM model with ACUTE state before INFECTION |

Both models are `ss.Infection` subclasses that implement LSHTM-style TB progression with states: SUSCEPTIBLE, INFECTION, CLEARED, UNCONFIRMED, ASYMPTOMATIC, SYMPTOMATIC, TREATMENT, TREATED, DEAD (and ACUTE for the Acute variant).

### 1.2 Dependencies

- **TB_LSHTM/tb_lshtm.py** imports `from tb_acf import expon_LTV` on line 5.
- **Usage of expon_LTV:** Only in **commented-out** parameter definitions (lines 43, 46, 48). The active code uses `ss.expon()` instead.
- **TBSL enum:** Defined in the same file and used internally by both models. It's not exported separately in this minimal plan, but the models need it internally.

### 1.3 Target Location

- **Add to:** `tbsim/tb_lshtm.py` (new file)
- **Export from:** `tbsim/__init__.py`

---

## 2. Implementation Steps

### Step 1: Create `tbsim/tb_lshtm.py`

1. **Copy the models from `TB_LSHTM/tb_lshtm.py`**
   - Copy the entire file content to `tbsim/tb_lshtm.py`.

2. **Remove tb_acf dependency**
   - Remove line: `from tb_acf import expon_LTV`
   - The commented-out LTV code (lines 43, 46, 48) can remain as-is since it's not active. If you want to clean it up, you can remove those comment lines, but it's not necessary.

3. **Update `__all__`**
   - Keep: `__all__ = ['TB_LSHTM', 'TB_LSHTM_Acute', 'TBSL']`
   - Note: `TBSL` is included because it's used by the models internally, but we're not explicitly exporting it from tbsim in this minimal plan (see Step 2).

4. **Verify imports**
   - Ensure all other imports are satisfied:
     - `numpy`, `starsim`, `matplotlib`, `pandas` — all standard tbsim dependencies
     - `enum.IntEnum` — standard library

**Result:** `tbsim/tb_lshtm.py` contains `TB_LSHTM` and `TB_LSHTM_Acute` with no external dependencies beyond what tbsim already requires.

---

### Step 2: Export from `tbsim/__init__.py`

1. **Add import**
   - Add to `tbsim/__init__.py`:
     ```python
     from .tb_lshtm import TB_LSHTM, TB_LSHTM_Acute
     ```

2. **Optional: Export TBSL**
   - If you want `TBSL` available for external use (e.g., for analyzers or scripts), also add:
     ```python
     from .tb_lshtm import TB_LSHTM, TB_LSHTM_Acute, TBSL
     ```
   - If not, the models will still work internally; `TBSL` just won't be in the public API.

**Result:** Users can do `from tbsim import TB_LSHTM, TB_LSHTM_Acute` (and optionally `TBSL`).

---

### Step 3: Basic Verification

1. **Import test**
   ```python
   import tbsim
   assert hasattr(tbsim, 'TB_LSHTM')
   assert hasattr(tbsim, 'TB_LSHTM_Acute')
   ```

2. **Instantiation test**
   ```python
   from tbsim import TB_LSHTM, TB_LSHTM_Acute
   
   # Should not raise
   tb1 = TB_LSHTM()
   tb2 = TB_LSHTM_Acute()
   ```

3. **Minimal simulation test** (optional but recommended)
   ```python
   import starsim as ss
   from tbsim import TB_LSHTM
   
   pop = ss.People(n_agents=100)
   nets = ss.MixingPool(diseases='tb', beta=1, contacts=ss.poisson(3))
   tb = TB_LSHTM(name='tb')
   sim = ss.Sim(people=pop, networks=nets, diseases=tb, start='2000-01-01', stop='2001-01-01')
   sim.run()  # Should complete without errors
   ```

**Result:** The models are functional and can be used in simulations.

---

## 3. File Changes Summary

| File | Action | Details |
|------|--------|---------|
| `tbsim/tb_lshtm.py` | **CREATE** | Copy from `TB_LSHTM/tb_lshtm.py`, remove `from tb_acf import expon_LTV` |
| `tbsim/__init__.py` | **MODIFY** | Add `from .tb_lshtm import TB_LSHTM, TB_LSHTM_Acute` (and optionally `TBSL`) |

---

## 4. Dependencies

- **No new dependencies required.** The models only need:
  - `numpy` (already in tbsim)
  - `starsim` (already in tbsim)
  - `pandas` (already in tbsim)
  - `matplotlib` (already in tbsim, used for plotting methods)
  - `enum` (standard library)

- **tb_acf is NOT required** after removing the unused `expon_LTV` import.

---

## 5. What This Plan Does NOT Include

This minimal plan explicitly excludes:

- ❌ Changes to `tbsim/analyzers.py` (e.g., using `TBSL` for LSHTM models)
- ❌ Factory functions like `from_inhost()` or `get_tb_model()`
- ❌ Migration of `TB_LSHTM/` scripts (base, run, scenarios, etc.)
- ❌ Creating `tbsim.disease` module
- ❌ Documentation updates (beyond inline code comments)
- ❌ Comprehensive test suite (only basic verification)

These can be addressed in separate plans or follow-up work.

---

## 6. Success Criteria

- [ ] `from tbsim import TB_LSHTM, TB_LSHTM_Acute` works without errors
- [ ] `TB_LSHTM()` and `TB_LSHTM_Acute()` can be instantiated
- [ ] A minimal `ss.Sim` with `TB_LSHTM` or `TB_LSHTM_Acute` runs without errors
- [ ] No `import tb_acf` in `tbsim/tb_lshtm.py`
- [ ] No import errors when importing the models

---

## 7. Optional Enhancements (Out of Scope)

If you want to go beyond the minimal plan, consider:

1. **Export TBSL:** Add `TBSL` to `tbsim/__init__.py` exports if external code needs it.
2. **Clean up commented LTV code:** Remove the commented `expon_LTV` parameter definitions if you don't plan to use them.
3. **Add docstrings:** Ensure both models have clear docstrings explaining their differences.
4. **Add to tests:** Create a test file `tests/test_tb_lshtm.py` with basic instantiation and simulation tests.

---

*Document version: 1.0. Last updated: 2025-01-26.*
