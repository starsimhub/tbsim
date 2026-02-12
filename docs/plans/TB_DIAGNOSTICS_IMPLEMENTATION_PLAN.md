# TB Disease Diagnostics and Testing – Implementation Plan

**Reference:** [GitHub Issue #10](https://github.com/starsimhub/tbsimV2/issues/10), [GitHub Issue #11](https://github.com/starsimhub/tbsimV2/issues/11)  
**Model basis:** LSHTM Reinfection (Undulation) paradigm – Schwalb et al. 2024 (preprint)

---

## Executive Summary

This plan outlines the design and implementation of a flexible TB diagnostics and testing framework that:

- Uses **test classifications** (e.g., `LTBI_TEST`, `ACTIVE_TB_TEST`, `TRIAGE_TEST`) instead of specific test names (Xpert, FujiLAM, etc.)
- Supports **multiple tests within a classification** (e.g., sputum smear + Xpert under `ACTIVE_TB_TEST`)
- Is **ready for reinfection undulation** states (Unconfirmed TB, Asymptomatic TB, Symptomatic TB) aligned with the Schwalb et al. paradigm
- Remains backward compatible with the current TBS state model (LATENT_*, ACTIVE_PRESYMP, ACTIVE_SMPOS, etc.) and extendable when an LSHTM-style undulation model is added

---

## 1. Test Classifications (Not Specific Test Names)

Define test **classifications** that abstract over concrete tests and can hold multiple implementations.

| Classification        | Purpose                                              | Target TB states (current)              | Target TB states (undulation) |
|-----------------------|------------------------------------------------------|-----------------------------------------|--------------------------------|
| `LTBI_TEST`           | Detect latent infection                              | LATENT_SLOW, LATENT_FAST                | Infection                       |
| `ACTIVE_TB_TEST`      | Confirm active disease for treatment                 | ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB| Unconfirmed, Asymptomatic, Symptomatic |
| `TRIAGE_TEST`         | Screen / triage before definitive test               | ACTIVE_PRESYMP, ACTIVE_*                | Unconfirmed, Asymptomatic      |
| `TREATMENT_RESPONSE`  | Monitor response during treatment                    | (on treatment)                          | Treatment                       |

### Design principles

- User-facing configuration uses these classifications.
- Specific tests (Xpert, FujiLAM, smear, etc.) are *implementations* of a classification.
- Results, eligibility, and cascade logic are defined at classification level; implementations plug in via a common interface.

---

## 2. State Mapping for Reinfection Undulation

To prepare for the LSHTM undulation model while supporting the current state model, diagnostics operate on **logical diagnostic groups** that map to both:

### Current TBS states (existing model)

| TBS state        | Diagnostic group          | Typical use in diagnostics          |
|------------------|---------------------------|-------------------------------------|
| LATENT_SLOW/FAST | Infection (LTBI)          | LTBI_TEST                           |
| ACTIVE_PRESYMP   | Unconfirmed / Asymptomatic| TRIAGE_TEST, possibly ACTIVE_TB_TEST|
| ACTIVE_SMPOS     | Symptomatic               | ACTIVE_TB_TEST                      |
| ACTIVE_SMNEG     | Symptomatic               | ACTIVE_TB_TEST                      |
| ACTIVE_EXPTB     | Symptomatic (EPTB)        | ACTIVE_TB_TEST (EPTB-specific sens) |

### Target LSHTM undulation states (future)

| LSHTM state     | Diagnostic group              | Primary tests                         |
|-----------------|-------------------------------|---------------------------------------|
| Infection       | Infection                     | LTBI_TEST                             |
| Unconfirmed TB  | Unconfirmed                   | TRIAGE_TEST, ACTIVE_TB_TEST           |
| Asymptomatic TB | Asymptomatic                  | TRIAGE_TEST, ACTIVE_TB_TEST           |
| Symptomatic TB  | Symptomatic                   | ACTIVE_TB_TEST (φ → Treatment)        |
| Treatment       | Treatment                     | TREATMENT_RESPONSE                    |

Implementation will use a **DiagnosticStateMapper** abstraction that:

1. Maps TBS / LSHTM states → logical diagnostic groups.
2. Is swappable so that when an undulation disease module is added, only the mapper changes, not the diagnostic logic.

---

## 3. Multiple Tests Within One Classification

### Conceptual model

- A **classification** (e.g., `ACTIVE_TB_TEST`) can have multiple **test implementations**.
- Tests within a classification can be:
  - **Parallel**: run together (e.g., smear + Xpert)
  - **Sequential**: triage → confirmatory (e.g., TRIAGE_TEST positive → ACTIVE_TB_TEST)
  - **Composite**: result derived from several tests (e.g., “any positive” or “all positive”).

### Data structures (proposed)

```python
# Test classification (enum or str)
TestClassification = Literal['LTBI_TEST', 'ACTIVE_TB_TEST', 'TRIAGE_TEST', 'TREATMENT_RESPONSE']

# Single test within a classification
@dataclass
class TestSpec:
    classification: TestClassification
    name: str                    # Internal ID, e.g. "xpert", "smear", "fujilam"
    sensitivity: Mapping[DiagnosticGroup, float]   # Per diagnostic group
    specificity: float
    target_groups: Sequence[DiagnosticGroup]      # Which states this test applies to
    order: int = 0               # For sequential ordering (0 = first)

# Classification config: multiple tests
@dataclass
class ClassificationConfig:
    classification: TestClassification
    tests: Sequence[TestSpec]
    combination_rule: Literal['any_positive', 'all_positive', 'first_positive', 'parallel']
```

### Combination rules

- `any_positive`: positive if any test is positive.
- `all_positive`: positive only if all tests positive.
- `first_positive`: run tests in order; stop on first positive.
- `parallel`: run all; combine per `any_positive` or `all_positive`.

### Person-level tracking (extend `TBPeople`)

```python
# Per-classification tracking (supports multiple tests per classification)
n_tests_by_classification: Dict[TestClassification, IntArr]   # e.g. n_tests_by_classification['ACTIVE_TB_TEST']
last_test_result_by_classification: Dict[TestClassification, BoolState]
last_test_ti_by_classification: Dict[TestClassification, FloatArr]
```

---

## 4. Diagnostic Eligibility and Integration Points

### Eligibility by classification

| Classification   | Eligible population (current model)                         | Eligible population (undulation)        |
|------------------|-------------------------------------------------------------|----------------------------------------|
| LTBI_TEST        | Latent (LS/ LF), contacts, high-risk                       | Infection, high-risk                    |
| ACTIVE_TB_TEST   | Sought care, not diagnosed, alive                          | Unconfirmed / Asymptomatic / Symptomatic|
| TRIAGE_TEST      | Screening (e.g., ACTIVE_PRESYMP, contacts)                 | Unconfirmed, Asymptomatic               |
| TREATMENT_RESPONSE | On treatment                                              | Treatment                               |

### Integration with health-seeking

- Keep existing `sought_care`-driven cascade for symptomatic care.
- Add **screening pathways** for:
  - Unconfirmed / Asymptomatic (undulation)
  - ACTIVE_PRESYMP (current model)
  - Contact investigation
  - Programmatic screening (e.g., HIV clinics)

### State transitions influenced by diagnostics

From the undulation paradigm:

- **Unconfirmed → Asymptomatic / Symptomatic / Recovered**: diagnostics clarify disease status.
- **Asymptomatic → Symptomatic**: progression; triage/active tests detect before symptoms.
- **Symptomatic → Treatment (φ)**: ACTIVE_TB_TEST positive triggers treatment.
- **Treatment → Treated (δ)**: TREATMENT_RESPONSE can inform success/failure.

Diagnostics will **modify rates** (φ, uncasy, asyunc, etc.) or provide **flags** used by the disease module for transitions, rather than forcing transitions directly.

---

## 5. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DiagnosticOrchestrator                            │
│  - Resolves eligibility by DiagnosticStateMapper                     │
│  - Selects ClassificationConfig per eligibility                      │
│  - Runs TestSpecs according to combination_rule                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐
│ LTBI_TEST     │   │ ACTIVE_TB_TEST  │   │ TRIAGE_TEST      │
│ - TST/IGRA    │   │ - Xpert/smear   │   │ - CXR/CAD        │
│ (future)      │   │ - FujiLAM       │   │ - Symptom screen │
└───────────────┘   └─────────────────┘   └──────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ DiagnosticState │
                    │ Mapper          │
                    │ TBS ↔ LSHTM     │
                    └─────────────────┘
```

---

## 6. Implementation Phases

### Phase 1: Test Classifications and Data Model (core refactor)

1. Introduce `TestClassification` enum and `DiagnosticGroup` enum.
2. Add `TestSpec` and `ClassificationConfig` dataclasses.
3. Implement `DiagnosticStateMapper` with mappings for current TBS states (and placeholder for LSHTM).
4. Add person-level states for per-classification tracking (`n_tests_by_classification`, `last_test_result_by_classification`, etc.).

**Deliverables:** `tbsim/interventions/diagnostics/test_classifications.py`, `diagnostic_state_mapper.py`, updates to `TBPeople`.

### Phase 2: Classification-Based Diagnostic Engine

1. Implement `TBDiagnosticV2` (or refactor `TBDiagnostic`) to:
   - Accept `ClassificationConfig` instead of raw sensitivity/specificity.
   - Resolve eligibility via `DiagnosticStateMapper`.
   - Apply tests according to `combination_rule`.
2. Support multiple tests within a classification (any_positive, all_positive, sequential).
3. Preserve backward compatibility via a thin wrapper that builds `ClassificationConfig` from legacy `sensitivity`/`specificity`.

**Deliverables:** Refactored `tb_diagnostic.py`, tests.

### Phase 3: Undulation Readiness

1. Define LSHTM state enum (Infection, Unconfirmed, Asymptomatic, Symptomatic, Treatment, Treated, Cleared, Recovered).
2. Extend `DiagnosticStateMapper` to map LSHTM states ↔ diagnostic groups.
3. Ensure eligibility and sensitivity/specificity logic work with both TBS and LSHTM when the undulation disease module exists.
4. Document how diagnostics influence transition rates (φ, uncasy, asyunc, etc.).

**Deliverables:** `lshtm_states.py` (or extension of TBS), mapper extensions, design doc for rate modifiers.

### Phase 4: Multiple Tests and Cascade Logic

1. Implement combination rules (any_positive, all_positive, first_positive, parallel).
2. Add configurable cascades (e.g., TRIAGE → ACTIVE_TB_TEST).
3. Per-classification result tracking and reporting.

**Deliverables:** Cascade engine, extended results, examples.

### Phase 5: Preset Configurations and Migrations

1. Provide presets for common scenarios:
   - `create_xpert_baseline_config()`
   - `create_oral_swab_config()`
   - `create_fujilam_config()`
   - `create_cad_cxr_config()`
   - `create_multi_test_cascade(...)`
2. Migration path from `TBDiagnostic` and `EnhancedTBDiagnostic` to the new framework.
3. Documentation and tutorials.

---

## 7. API Sketch

### User-facing configuration

```python
# Classification-based, multiple tests per classification
from tbsim.interventions.diagnostics import (
    TestClassification,
    ClassificationConfig,
    TestSpec,
    TBDiagnosticV2,
    create_xpert_baseline_config,
)

config = ClassificationConfig(
    classification=TestClassification.ACTIVE_TB_TEST,
    tests=[
        TestSpec(classification=TestClassification.ACTIVE_TB_TEST, name="xpert",
                 sensitivity={DiagnosticGroup.SYMPTOMATIC: 0.91, DiagnosticGroup.ASYMPTOMATIC: 0.78},
                 specificity=0.96, target_groups=[DiagnosticGroup.SYMPTOMATIC, DiagnosticGroup.ASYMPTOMATIC]),
        TestSpec(classification=TestClassification.ACTIVE_TB_TEST, name="smear",
                 sensitivity={DiagnosticGroup.SYMPTOMATIC: 0.7}, specificity=0.98,
                 target_groups=[DiagnosticGroup.SYMPTOMATIC]),
    ],
    combination_rule='any_positive',
)

diag = TBDiagnosticV2(pars={
    'coverage': 1.0,
    'classifications': [config],
    'eligibility': 'sought_care',  # or 'screening', 'contact_investigation'
})
```

### Backward compatibility

```python
# Legacy-style still works
diag = TBDiagnostic(pars={'sensitivity': 0.9, 'specificity': 0.95})
# Internally builds ClassificationConfig for ACTIVE_TB_TEST with one TestSpec
```

---

## 8. Result Metrics (Extended)

| Metric                                   | Description                                   |
|------------------------------------------|-----------------------------------------------|
| `n_tested_by_classification[ACTIVE_TB]`  | Tests performed per classification            |
| `n_positive_by_classification[...]`      | Positive results per classification           |
| `n_tests_per_person_by_classification`   | Distribution of repeat testing                |
| `n_cascade_TRIAGE_to_ACTIVE`             | Triage → confirmatory cascade completions     |
| `n_false_negative_by_classification`     | False negatives for retest logic              |
| Existing: `n_tested`, `n_test_positive`, etc. | Preserved for backward compatibility     |

---

## 9. Files to Create or Modify

| Path                                           | Action                                        |
|------------------------------------------------|-----------------------------------------------|
| `tbsim/interventions/diagnostics/__init__.py`   | New module                                    |
| `tbsim/interventions/diagnostics/test_classifications.py` | TestSpec, ClassificationConfig, enums  |
| `tbsim/interventions/diagnostics/diagnostic_state_mapper.py` | TBS ↔ LSHTM mapping               |
| `tbsim/interventions/diagnostics/engine.py`     | Diagnostic engine / orchestrator              |
| `tbsim/interventions/tb_diagnostic.py`          | Refactor to use diagnostics module; keep API  |
| `tbsim/people.py`                              | Add per-classification tracking states        |
| `tests/test_tb_diagnostic.py`                  | Extend for classifications, multi-test        |
| `docs/`                                        | User guide, migration notes                   |

---

## 10. Risks and Mitigations

| Risk                               | Mitigation                                                |
|------------------------------------|-----------------------------------------------------------|
| Complexity for simple use cases    | Presets and legacy-compatible wrapper                     |
| Performance with many tests        | Vectorized test application; avoid per-person Python loop |
| Undulation model not yet merged    | Mapper abstraction; Phase 3 independent of disease model  |
| Breaking existing analyses         | Keep `TBDiagnostic` behavior via wrapper; deprecate later |

---

## 11. Success Criteria

- [ ] Configurations use test classifications, not test-specific names.
- [ ] Multiple tests can be defined per classification.
- [ ] DiagnosticStateMapper supports both current TBS and future LSHTM undulation states.
- [ ] Diagnostics integrate with health-seeking and can support screening pathways.
- [ ] Backward compatibility with `TBDiagnostic` and `EnhancedTBDiagnostic`.
- [ ] Unit and integration tests for classifications, multi-test, and cascades.
- [ ] Documentation and migration guide for existing users.
