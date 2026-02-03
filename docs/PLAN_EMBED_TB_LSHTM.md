# Plan: Embed TB_LSHTM, TB_LSHTM_Acute, and TBSL into tbsim

**Goal:** Make `TB_LSHTM`, `TB_LSHTM_Acute`, and `TBSL` first-class parts of the **tbsim** package so that all three TB model families (TBsim, LSHTM, LSHTM-Acute) live under tbsim and can be used without depending on **tb_acf** for the core disease models.

**Quick reference:**  
- **TBSL** = LSHTM state enum (SUSCEPTIBLE, INFECTION, CLEARED, …).  
- **TB_LSHTM** = base LSHTM intrahost model (no acute state).  
- **TB_LSHTM_Acute** = LSHTM model with ACUTE state before INFECTION.  
- **TBsim** = existing tbsim model (`TB` / `TBS` in `tbsim.tb`).

**Status:** Plan only — not yet implemented.

---

## 1. Current State

### 1.1 What Exists Where

| Artifact | Location | Notes |
|----------|----------|-------|
| **TBSL** (IntEnum) | `TB_LSHTM/tb_lshtm.py` | LSHTM state enum: SUSCEPTIBLE, INFECTION, CLEARED, UNCONFIRMED, ASYMPTOMATIC, SYMPTOMATIC, TREATMENT, TREATED, DEAD, ACUTE |
| **TB_LSHTM** (ss.Infection) | `TB_LSHTM/tb_lshtm.py` | Base LSHTM natural-history model (no acute state) |
| **TB_LSHTM_Acute** (ss.Infection) | `TB_LSHTM/tb_lshtm.py` | LSHTM model with ACUTE state before INFECTION |
| **TB** (ss.Infection) | `tbsim/tb.py` | TBsim model (TBS states: latent slow/fast, active presymp/smpos/smneg/exptb) |
| **TBS** (IntEnum) | `tbsim/tb.py` | TBsim state enum |

### 1.2 Dependencies

- **TB_LSHTM/tb_lshtm.py**: Imports `from tb_acf import expon_LTV`; usage is **commented out** only (theta, phi, mutb LTV). `TB_LSHTM/utils.py` defines `expon_LTV` locally.
- **TB_LSHTM** (base, run, scenarios, calibrate, plot, disease, utils): Heavily use **tb_acf** for:
  - `CaseFinding`, `AgeInfect`, `sigmoidally_varying_parameter`, `set_beta`, `get_intv`
  - `make_sim`, `apply_calib`, `apply_scen`, `build_arms`, `make_scenarios`, etc.
  - `from_inhost` via `tb_acf.disease` (base expects `tb_acf.disease.from_inhost`).
- **TB_LSHTM/disease.py**: Imports `tbsim.disease` (BaseDisease, TB, LSHTM) — **`tbsim.disease` does not exist**; TB lives in `tbsim.tb`.
- **tbsim/analyzers.py**: `DwtAnalyzer` uses `TBS` by default; when `'LSHTM' in sim.diseases[0].__class__.__name__`, it switches to `tb_acf.TBSL`.

### 1.3 Model Naming Conventions

- **TBsim** / **TBSL**: User request uses “TBSL” for the **enum** (LSHTM states). The **models** are `TB_LSHTM` and `TB_LSHTM_Acute`. Scenarios use `InHost`: `'TBsim'`, `'LSHTM'`, `'LSHTM-Acute'`.

---

## 2. Embedding Strategy

### 2.1 Principles

1. **Core models in tbsim:** `TBSL`, `TB_LSHTM`, and `TB_LSHTM_Acute` live inside the tbsim package and are exposed via `tbsim` public API.
2. **Remove tb_acf for core models:** No `import tb_acf` in the embedded LSHTM disease code. Use local helpers (e.g. `expon_LTV`) or drop unused LTV code.
3. **Unified model selection:** Provide a single factory (e.g. `from_inhost` or `get_tb_model`) in tbsim to choose among `TB`, `TB_LSHTM`, `TB_LSHTM_Acute` by name.
4. **Analyzers use tbsim.TBSL:** When running LSHTM models, analyzers use `tbsim.TBSL` instead of `tb_acf.TBSL`.
5. **TB_LSHTM scripts:** `TB_LSHTM/` scripts (base, run, scenarios, calibrate, plot) can keep using **tb_acf** for interventions and harness **or** be migrated to tbsim-only in a later phase.

---

## 3. Implementation Phases

### Phase 1: Embed Core Models and TBSL (no tb_acf)

**Scope:** Move `TBSL`, `TB_LSHTM`, `TB_LSHTM_Acute` into tbsim and wire them into the public API.

1. **Create `tbsim/tb_lshtm.py`**
   - Copy `TB_LSHTM/tb_lshtm.py` into `tbsim/tb_lshtm.py`.
   - Remove `from tb_acf import expon_LTV`. Keep LTV-related pars as commented optional; if needed later, use a local `expon_LTV` (see below).
   - Keep `TBSL`, `TB_LSHTM`, `TB_LSHTM_Acute` as-is. Dependencies: `numpy`, `starsim`, `pandas`, `matplotlib` only.

2. **Optional: `expon_LTV` in tbsim**
   - Either copy `TB_LSHTM/utils.expon_LTV` into `tbsim/utils/` (e.g. `tbsim/utils/distributions.py`) and use it from `tb_lshtm` if LTV pars are re-enabled, or leave LTV commented and add it in a later change.

3. **Update `tbsim/__init__.py`**
   - Add:
     ```python
     from .tb_lshtm import TB_LSHTM, TB_LSHTM_Acute, TBSL
     ```
   - Export them in `__all__` (or equivalent) so that `tbsim.TB_LSHTM`, `tbsim.TB_LSHTM_Acute`, `tbsim.TBSL` are available.

4. **Update `tbsim/analyzers.py`**
   - Where `DwtAnalyzer` currently does:
     ```python
     if 'LSHTM' in str(self.sim.diseases[0].__class__):
         import tb_acf as tbacf
         self.eSTATES = tbacf.TBSL
     ```
   - Change to use `tbsim.TBSL` (e.g. `import tbsim as mtb` already used; set `self.eSTATES = mtb.TBSL`).
   - Update docstrings/examples that say `from tb_acf import TBSL` to `from tbsim import TBSL`.

5. **Sanity checks**
   - `import tbsim; assert hasattr(tbsim, 'TB_LSHTM') and hasattr(tbsim, 'TB_LSHTM_Acute') and hasattr(tbsim, 'TBSL')`.
   - Run existing tests that use TB; add a minimal test that builds a `ss.Sim` with `TB_LSHTM` or `TB_LSHTM_Acute` and runs a few steps.

**Outcome:** `TB_LSHTM`, `TB_LSHTM_Acute`, and `TBSL` are part of tbsim. No tb_acf dependency for these. Analyzers use `tbsim.TBSL` for LSHTM.

---

### Phase 2: Unified Model Factory in tbsim

**Scope:** Single entry point to resolve “TBsim” vs “LSHTM” vs “LSHTM-Acute” to the right infection class.

1. **Add `tbsim/model_factory.py` (or equivalent)**
   - Implement something like:
     ```python
     def from_inhost(inhost: str, *, name: str = "tb"):
         key = (inhost or "").strip().lower()
         if key == "tbsim":
             return TB(name=name)        # tbsim.tb.TB
         if key == "lshtm":
             return TB_LSHTM(name=name)
         if key == "lshtm-acute":
             return TB_LSHTM_Acute(name=name)
         raise ValueError(f"Unknown inhost: {inhost!r}. Use 'TBsim', 'LSHTM', or 'LSHTM-Acute'.")
     ```
   - Use `TB` from `tbsim.tb`, `TB_LSHTM` / `TB_LSHTM_Acute` from `tbsim.tb_lshtm`.
   - Expose `from_inhost` (or `get_tb_model`) via `tbsim/__init__.py`.

2. **Optional: `tbsim.disease` module**
   - If you want to align with `TB_LSHTM/disease.py` (which expects `tbsim.disease`), introduce `tbsim/disease.py` that re-exports:
     - `TB`, `TBS` from `tbsim.tb`
     - `TB_LSHTM`, `TB_LSHTM_Acute`, `TBSL` from `tbsim.tb_lshtm`
     - `from_inhost` from the factory.
   - Then `TB_LSHTM/disease.py` could eventually import from `tbsim.disease` instead of `tb_acf` for model resolution (see Phase 4).

3. **Documentation**
   - Document the three models and factory in `docs/` (e.g. “TB models” or “LSHTM embedding”).
   - Mention supported `inhost` values: `TBsim`, `LSHTM`, `LSHTM-Acute`.

**Outcome:** Callers can do `from tbsim import from_inhost; tb = from_inhost('LSHTM')` (or similar) without touching tb_acf.

**Note:** Current TB_LSHTM code uses `from_inhost` → wrapper → `make_module()` to get the disease. The factory above returns the infection **instance** directly. When updating TB_LSHTM scripts (Phase 3), either (a) switch them to use the instance directly, or (b) add a thin wrapper in tbsim that exposes `make_module()` and other ACT3-specific helpers for backward compatibility.

---

### Phase 3: TB_LSHTM Scripts Use tbsim Models

**Scope:** Point `TB_LSHTM/` code at tbsim for disease models and, where feasible, reduce tb_acf usage.

1. **`TB_LSHTM/base.py`**
   - Currently uses `from tb_acf.disease import from_inhost` and `acf.make_sim`, etc.
   - **Option A (minimal):** Keep using tb_acf for `make_sim`, `apply_calib`, interventions. Replace `tb_acf.disease.from_inhost` with `tbsim.from_inhost` (or `tbsim.disease.from_inhost` once it exists). Ensure `make_sim` builds sims using tbsim’s `TB` / `TB_LSHTM` / `TB_LSHTM_Acute`.
   - **Option B (deeper):** Migrate `make_sim`, `apply_calib`, etc. into tbsim or a dedicated `tbsim.lshtm` harness; then tb_acf is only needed for shared ACF interventions if they stay in tb_acf.

2. **`TB_LSHTM/disease.py`**
   - Today it imports `tbsim.disease` (nonexistent) and `acf.TB_LSHTM` / `acf.TBSL`.
   - After Phase 1–2: implement `tbsim.disease` as above, then update `TB_LSHTM/disease.py` to import `TB`, `TB_LSHTM`, `TB_LSHTM_Acute`, `TBSL`, `from_inhost` from `tbsim.disease` (or `tbsim`). Drop `acf` for model resolution.

3. **`TB_LSHTM/scenarios.py`**
   - Uses `acf.TBSL`, `acf.get_intv`, etc. For TBSL: use `tbsim.TBSL`. For `get_intv`, either keep using tb_acf or add a local `get_intv` (e.g. from `TB_LSHTM/utils.py`) and use that.

4. **`TB_LSHTM/run.py`, `calibrate.py`, `plot.py`, `run_sensitivity`**
   - Prefer `tbsim` for models and `TBSL`; keep or migrate tb_acf usage (CaseFinding, make_sim, etc.) per Option A/B above.

5. **Calibration data**
   - `TB_LSHTM/calib/{TBsim,LSHTM,LSHTM_Acute}/` stay as-is. Scripts that read them should use tbsim models when building sims.

**Outcome:** TB_LSHTM workflows use tbsim for `TB_LSHTM`, `TB_LSHTM_Acute`, `TBSL`, and optionally for `from_inhost`; tb_acf usage is reduced or confined to non-model code.

---

### Phase 4: Cleanup and Optional Restructure

**Scope:** Clarify long-term layout and de-duplicate.

1. **`TB_LSHTM/` folder**
   - **Option A:** Keep `TB_LSHTM/` as a **script/config** layer (base, run, scenarios, calibrate, plot, ACT3-specific logic) that uses tbsim for all disease models. `TB_LSHTM/tb_lshtm.py` is **deprecated**; everything imports from `tbsim`.
   - **Option B:** Move those scripts under `scripts/lshtm/` or `scripts/act3/` and treat `TB_LSHTM/` as legacy, or remove it once migration is complete.

2. **`tb_acf`**
   - Document when tb_acf is still required (e.g. CaseFinding, make_sim, apply_calib) and when it is not (core models, analyzers using `TBSL`).
   - If you later implement equivalent interventions/harness in tbsim, you can optionally drop tb_acf for LSHTM workflows.

3. **Tests and CI**
   - Add tests for `TB_LSHTM` and `TB_LSHTM_Acute` (single sim, short run).
   - Add a test for `from_inhost` returning the correct class for `TBsim`, `LSHTM`, `LSHTM-Acute`.
   - Run existing test suite (e.g. `test_basics`, `test_tb_people`, analyzer tests) and fix any regressions.

4. **Versioning and docs**
   - Bump tbsim version per your policy. Mention in changelog/README that LSHTM models are now part of tbsim.
   - Update any docs that reference `tb_acf.TB_LSHTM` or `tb_acf.TBSL` to use `tbsim`.

**Outcome:** Clear separation between tbsim (models + factory + analyzers) and TB_LSHTM/tb_acf (scripts and ACF harness). No duplicate model definitions.

---

## 4. File-Level Checklist

| Task | File(s) | Phase |
|------|---------|-------|
| Add `tbsim/tb_lshtm.py` with `TBSL`, `TB_LSHTM`, `TB_LSHTM_Acute` (no tb_acf) | `tbsim/tb_lshtm.py` | 1 |
| Export LSHTM models and TBSL from tbsim | `tbsim/__init__.py` | 1 |
| Use `tbsim.TBSL` in DwtAnalyzer for LSHTM | `tbsim/analyzers.py` | 1 |
| Add `from_inhost` (or `get_tb_model`) in tbsim | `tbsim/model_factory.py` or `tbsim/disease.py` | 2 |
| Optional: add `tbsim.disease` and re-exports | `tbsim/disease.py` | 2 |
| Point `TB_LSHTM/base` at tbsim `from_inhost` | `TB_LSHTM/base.py` | 3 |
| Update `TB_LSHTM/disease` to use tbsim | `TB_LSHTM/disease.py` | 3 |
| Use `tbsim.TBSL` in scenarios | `TB_LSHTM/scenarios.py` | 3 |
| Align run/calibrate/plot with tbsim | `TB_LSHTM/run.py`, etc. | 3 |
| Deprecate or remove `TB_LSHTM/tb_lshtm.py` | `TB_LSHTM/tb_lshtm.py` | 4 |
| Tests for LSHTM models and `from_inhost` | `tests/` | 4 |
| Docs and changelog | `docs/`, `README` | 4 |

---

## 5. Dependencies Summary

- **Phase 1:** None beyond existing tbsim deps (numpy, starsim, pandas, etc.). No tb_acf.
- **Phase 2:** Same.
- **Phase 3–4:** tb_acf remains optional if TB_LSHTM scripts still use it for CaseFinding, `make_sim`, etc. Document clearly.

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| tb_acf and tbsim both define LSHTM models | Phase 1 establishes tbsim as source of truth; TB_LSHTM scripts later switch to tbsim (Phase 3–4). |
| Existing TB_LSHTM/ACT3 workflows break | Phase 3 changes are incremental; keep tb_acf until scripts are migrated and tested. |
| `tbsim.disease` vs `tbsim.tb` confusion | Use `tbsim.disease` only as a thin re-export layer; document that TB lives in `tbsim.tb`, LSHTM in `tbsim.tb_lshtm`. |

---

## 7. Success Criteria

- [ ] `from tbsim import TB_LSHTM, TB_LSHTM_Acute, TBSL` works.
- [ ] `from tbsim import from_inhost` (or equivalent) returns `TB`, `TB_LSHTM`, or `TB_LSHTM_Acute` for `'TBsim'`, `'LSHTM'`, `'LSHTM-Acute'`.
- [ ] `DwtAnalyzer` uses `tbsim.TBSL` when the sim uses an LSHTM disease model.
- [ ] A short `ss.Sim` run with `TB_LSHTM` or `TB_LSHTM_Acute` completes without errors.
- [ ] TB_LSHTM calibration/scenario scripts run using tbsim models (after Phase 3).
- [ ] No `import tb_acf` in `tbsim/tb_lshtm.py` or in analyzer logic that selects TBSL.

---

*Document version: 1.0. Last updated: 2025-01-26.*
