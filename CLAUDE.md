# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TBsim is an agent-based tuberculosis (TB) model built on the [Starsim](https://github.com/starsimhub/starsim) framework. It simulates TB transmission, disease progression, and treatment outcomes in populations. Currently in alpha (v0.10.0). Python >=3.11.

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

- **TB** ([tb.py](tbsim/tb.py)) — LSHTM "spectrum of disease" approach with states: SUSCEPTIBLE → INFECTION → NON_INFECTIOUS → ASYMPTOMATIC → SYMPTOMATIC → TREATMENT (or CLEARED/DEAD). State enum is `TBS`.

Extends `ss.Disease` from Starsim.

### Multi-strain / drug-resistance extension

[tbsim/resistance/](tbsim/resistance/) adds a multi-strain, drug-resistance overlay. `TBResistant` (in [tb_resistant.py](tbsim/resistance/tb_resistant.py)) is a drop-in `TB` subclass that encodes `2ⁿ` strains over `n` named drugs as a per-agent bitmask (`strain_mask`) plus a per-strain multiplicity counter, with per-strain fitness costs, superinfection, and de-novo resistance. It ships strain-aware product/delivery interventions — `TxR`/`TxDeliveryR` (treatment), `DST`/`DSTDelivery` (diagnostics), `TPTRx` (preventive therapy) — the `ResistanceStats`/`StrainResults` analyzers, and a two-strain reference ODE (`tbsim.compartmental.TwoStrainODE`) for ABM↔ODE validation. All public entry points are re-exported on `tbsim`. CI coverage lives in [tests/test_resistance.py](tests/test_resistance.py) and the ABM↔ODE validation in [tests/test_compartmental.py](tests/test_compartmental.py). The implementation reference is [tbsim/resistance/README.md](tbsim/resistance/README.md); the usage tutorial is `docs/tutorials/resistance_tutorial.qmd`.

### Sim Wrapper

[tbsim/sim.py](tbsim/sim.py) provides `tbsim.Sim`, a convenience wrapper around `ss.Sim` that auto-routes flat parameters to the sim or TB module, and provides sensible defaults (demographics, random network, TB disease). A pre-built TB module may be passed via either `tb_model=` or `diseases=` (both are respected — no default TB is added on top). Pass `demographics=[]` (or `networks=[]`) to suppress the default demographics/network for that slot.

### Two conventions to follow

- **Getting the TB module:** on a `tbsim.Sim`, call `sim.get_tb()` — do *not* use `tbsim.get_tb(sim)`, and don't pass `which=` (it auto-finds the TB module, including `TBResistant`). The module-level `tbsim.get_tb(sim, ...)` is only for library internals that must accept an arbitrary `ss.Sim`. In tests/docs/examples, build the sim as `tbsim.Sim(...)` so `sim.get_tb()` is available.
- **Passing TB parameters:** pass TB/`TBResistant` parameters directly as keyword arguments (e.g. `tbsim.TBResistant(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.1))`) rather than wrapping them in `pars=dict(...)`. Only build a standalone `pars` dict when the *same* dict is reused across multiple module instances.

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

## Claude Code Skills

Repo-specific Claude Code skills live in `.claude/skills/` (also exposed at the top level via the `skills/` symlink). Currently one skill:

- **`tbsim.address-issue`** — fixes a specified TBsim GitHub issue end to end: reviews the issue, writes and confirms a failing test, implements the change in Starsim style, runs the test suite, and updates the changelog/docs. Invoke with `/tbsim.address-issue <number-or-url>` or by asking to "address"/"fix" an issue.

## Key Dependencies

- **starsim** (>=3.5.1) — ABM framework; use `ss.library.HouseholdNet`
- **sciris** (>=3.1.0) — Utility library used throughout (`sc.objdict`, `sc.mergedicts`, etc.)
- **pandas** (>=2.0.0) — Used heavily in diagnostic product definitions (DataFrame-based)
