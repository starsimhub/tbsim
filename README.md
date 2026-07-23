# TBsim

TBsim is an agent-based model for simulating tuberculosis transmission, disease progression, and treatment outcomes in populations. Built on the Starsim agent-based modeling platform, it enables researchers to evaluate intervention strategies, explore disease dynamics under different conditions, and inform TB control policy decisions.

**Note:** TBsim is still under active development; results may change without warning.

## Introduction

Tuberculosis is a major global health problem, and understanding its dynamics can help in developing better strategies for control and treatment. This project uses the Starsim package to simulate TB spread in a population, considering factors like transmission rates, treatment efficacy, and social dynamics.

TBsim uses an agent-based implementation of the LSHTM "spectrum of TB disease" natural history (susceptible → infection/cleared → non-infectious → asymptomatic → symptomatic → treatment/treated/death), designed to support evaluating active case-finding / population-wide screening algorithms (e.g., CXR and NAAT workflows) as in Schwalb et al. 2025 ([PLOS Glob Public Health](https://journals.plos.org/globalpublichealth/article?id=10.1371/journal.pgph.0005050)).

TBsim also includes a **multi-strain / drug-resistance extension** (`tbsim.resistance`) for modeling `2ⁿ` strains over `n` drugs — with fitness costs, superinfection, de-novo and acquired resistance, drug-susceptibility testing, and strain-aware treatment/preventive therapy. See the [`tbsim.resistance` README](tbsim/resistance/README.md) and the resistance tutorial in the docs.

## Getting started

### Installation

TBsim is not yet released on PyPI, so you need to install from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/starsimhub/tbsim.git
   cd tbsim
   ```
2. Install the required packages:
   ```bash
   pip install -e .
   ```

## Project structure

- `tbsim/` -- Core package (disease models, interventions, analyzers, networks, comorbidities)
- `tbsim_examples/` -- Ready-to-run example scripts
- `tests/` -- Test suite
- `docs/` -- Documentation source (Quarto)
- `skills/` -- [Claude Code](https://claude.com/claude-code) skills for working on TBsim (symlink to `.claude/skills/`); currently just `tbsim.address-issue`, which fixes a specified GitHub issue end to end (plan, test, implement, run the suite, update the changelog)

### Running a sample simulation

1. Navigate to the folder `tbsim_examples`
2. Run the script:
   ```bash
   python run_tb.py
   ```
3. Running this script should result in basic charts being displayed.

Or run directly in Python:

```python
import starsim as ss
import tbsim

sim = ss.Sim(diseases=tbsim.TB(), networks='random', demographics=True, start=2000, stop=2010, dt='month')
sim.run()
sim.plot()
```

## Usage
- Usage examples are available in the **[tbsim_examples](https://github.com/starsimhub/tbsim/tree/main/tbsim_examples)** folder.

## Documentation
TBsim documentation is available at [starsim.org/tbsim](https://starsim.org/tbsim).

TBsim is based on Starsim; please refer to [Starsim documentation](https://docs.starsim.org) for additional information.

## Contributing
Contributions to the TBsim project are welcome! Please read [contributing.md](contributing.md) for details on our code of conduct, and the process for submitting pull requests.
