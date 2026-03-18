# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TBsim is an agent-based tuberculosis (TB) model built on the [Starsim](https://github.com/starsimhub/starsim) framework. It simulates TB transmission, disease progression, and treatment outcomes in populations. Currently in alpha (v0.7.0). Python >=3.11.

## Build & Development Commands

```bash
# Install (editable mode)
pip install -e .

# Install with dev dependencies (tests, docs)
pip install -e .[dev]

# Run all tests (from tests/ directory)
cd tests && bash run_tests

# Run a single test file
pytest tests/test_tb.py

# Run a single test function
pytest tests/test_tb.py::test_something -v

# Run tests in parallel
pytest tests/test_*.py -n auto

# Build docs
mkdocs serve  # local preview
mkdocs build  # build static site
```

## Architecture

### Core Disease Model

The TB natural history model lives in `tbsim/`:

- **TB** ([tb.py](tbsim/tb.py)) — LSHTM "spectrum of disease" approach with states: SUSCEPTIBLE → INFECTION → NON_INFECTIOUS → ASYMPTOMATIC → SYMPTOMATIC → TREATMENT (or CLEARED/DEAD). State enum is `TBS`. Variant `TBAcute` adds an ACUTE state post-exposure.

Extends `ss.Disease` from Starsim.

### Sim Wrapper

[tbsim/sim.py](tbsim/sim.py) provides `tbsim.Sim`, a convenience wrapper around `ss.Sim` that auto-routes flat parameters to the sim or TB module, and provides sensible defaults (demographics, random network, TB disease).

### Intervention Architecture (Product/Delivery Pattern)

Interventions in [tbsim/interventions/](tbsim/interventions/) follow a product/delivery separation:

- **Products** define *what* (test sensitivity, drug efficacy): `Dx` (diagnostics), `Tx` (treatments)
- **Delivery** classes define *how* (eligibility, coverage, timing): `DxDelivery`, `TxDelivery`
- Other interventions: `HealthSeekingBehavior`, `BCG`, `TPT`, `BetaModifier`

### Comorbidities

[tbsim/comorbidities/](tbsim/comorbidities/) contains modules for HIV and malnutrition co-infection that modify TB disease parameters.

### Analyzers

[tbsim/analyzers.py](tbsim/analyzers.py) provides `DwellTime` (time spent in each disease state) and `HouseholdStats` (household-level transmission analysis).

## Style Conventions

- Follows the [Starsim style guide](https://github.com/starsimhub/styleguide) (Google Python style with exceptions)
- Use Starsim-AI skills when writing or modifying Python files: https://github.com/starsimhub/starsim_ai/tree/main/plugins/starsim
- Max line length: 200 characters
- Short variable/function names are acceptable (Starsim convention allows 1-15 char names)
- Wildcard imports with `__all__` are standard practice
- Linting configured in [.pylintrc](.pylintrc)
- PRs should target `main`; all tests must pass

## Key Dependencies

- **starsim** (>=3.2.1) — ABM framework; disease models extend `ss.Disease`, interventions extend `ss.Intervention`
- **sciris** (>=3.1.0) — Utility library used throughout (`sc.objdict`, `sc.mergedicts`, etc.)
- **pandas** (>=2.0.0) — Used heavily in diagnostic product definitions (DataFrame-based)
