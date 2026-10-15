# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TBsim is a tuberculosis (TB) model built on [Starsim](https://github.com/starsimhub/starsim) v4. It simulates TB transmission, disease progression, and treatment outcomes in populations. The model is declared as states and transitions, so it runs as an ODE, as a stochastic compartmental model, or as agents. Version 1.0.0 (in development, on the `v1.0.0` branch) is the move to Starsim v4. Python >=3.11.

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

# Build docs (Quarto; run from the docs/ directory)
cd docs && ./preview   # local preview with live reload
cd docs && ./render    # build static site to docs/_site
```

## Architecture

### Core Disease Model

The TB natural history model lives in `tbsim/`:

- **TB** ([tb.py](tbsim/tb.py)) — the LSHTM "spectrum of disease" model as an `ss.Disease` declared from `ss.State`s and `ss.Transition`s: `susceptible` → `infection` → `non_infectious` ↔ `asymptomatic` ↔ `symptomatic` → `treatment` → `treated`, with `cleared` and `recovered`. Parameters (`tb_pars()`) are `ss.par`s, rates per year; optional parts (front-loading, relapse, waning protection, migration) add states and transitions only when used. `tbsim.state_codes(tb)` gives each agent's state as a number.

### Drug resistance

[tbsim/resistance.py](tbsim/resistance.py) adds drug resistance as a strain dimension of `tbsim.TB` (`drugs=`): its levels are the sets of strains people carry (e.g. `DS+RIF`), up to `max_strains`, and `add_resistance()` adds superinfection, the progression bottleneck, de novo resistance, and treatment outcomes by strain. [models.py](tbsim/models.py) has `tbsim.LSHTMResistant`, the structure of the reference two-strain ODE (`tbsim.compartmental.TwoStrainODE`), which it matches to 1e-4. Tests are in [tests/test_resistance.py](tests/test_resistance.py), [tests/test_models.py](tests/test_models.py), and (agents against the ODE) [tests/test_compartmental.py](tests/test_compartmental.py).

### Sim Wrapper

[tbsim/sim.py](tbsim/sim.py) provides `tbsim.Sim`, an `ss.Sim` with a TB model: TB parameters can be given directly (`tbsim.Sim(beta=12)`), or a TB model with `tb=`. It has no default network or demographics, so it runs as an ODE unless given a network or `solver='abm'`.

### Two conventions to follow

- **Getting the TB module:** on a `tbsim.Sim`, call `sim.get_tb()` — do *not* use `tbsim.get_tb(sim)`, and don't pass `which=` (it auto-finds the TB module, including `LSHTMResistant`). The module-level `tbsim.get_tb(sim, ...)` is only for library internals that must accept an arbitrary `ss.Sim`. In tests/docs/examples, build the sim as `tbsim.Sim(...)` so `sim.get_tb()` is available.
- **Passing TB parameters:** pass TB parameters directly as keyword arguments (e.g. `tbsim.TB(beta=12, init=dict(infection=0.1))`) rather than wrapping them in `pars=dict(...)`. Only build a standalone `pars` dict when the *same* dict is reused across multiple module instances.

### Intervention Architecture (Product/Delivery Pattern)

Interventions in [tbsim/interventions/](tbsim/interventions/) are Starsim v4 products and deliveries, so they run under every solver:

- **Products** define *what*: tests (`tbsim.Xpert()`, `OralSwab()`, `FujiLAM()`, `CAD()`, `tbsim.test()`), `tbsim.BCG()`, and `tbsim.TPT()`
- **Deliveries** define *how* (eligibility, coverage, timing): `ss.screen()`, `ss.vaccinate()`, `ss.treat()`, and `tbsim.care(tb)` (the care cascade: seeking care, testing, and starting treatment)
- `tbsim.regimen('dots')` gives the model's treatment parameters for a regimen; `ss.change()` changes a parameter at given times

### Comorbidities

[tbsim/comorbidities/](tbsim/comorbidities/) contains `tbsim.HIV` and `tbsim.Malnutrition`, which act on TB through modifiers (`tbsim.hiv_effects(tb, hiv)`, `tbsim.nutrition_effects(tb)`), passed to the sim as `connectors`.

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

## Claude Code Skills

Repo-specific Claude Code skills live in `.claude/skills/` (also exposed at the top level via the `skills/` symlink). Currently one skill:

- **`tbsim.address-issue`** — fixes a specified TBsim GitHub issue end to end: reviews the issue, writes and confirms a failing test, implements the change in Starsim style, runs the test suite, and updates the changelog/docs. Invoke with `/tbsim.address-issue <number-or-url>` or by asking to "address"/"fix" an issue.

## Key Dependencies

- **starsim** (v4, the `v4` branch) — the modeling framework; use `ss.library.HouseholdNet` for households
- **sciris** (>=3.1.0) — Utility library used throughout (`sc.objdict`, `sc.mergedicts`, etc.)
- **pandas** (>=2.0.0) — Used in the analyzers and malnutrition data
