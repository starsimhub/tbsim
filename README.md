# TBsim

**Warning! TBsim is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is not ready to be used for real research or policy questions.**

TBsim is an agent-based model for simulating tuberculosis transmission, disease progression, and treatment outcomes in populations. Built on the Starsim agent-based modeling platform, it enables researchers to evaluate intervention strategies, explore disease dynamics under different conditions, and inform TB control policy decisions.

## Introduction

Tuberculosis is a major global health problem, and understanding its dynamics can help in developing better strategies for control and treatment. This project uses the Starsim package to simulate TB spread in a population, considering factors like transmission rates, treatment efficacy, and social dynamics.

## Features

- **TB Dynamics Simulation:** Models the spread of TB in a given population.
- **Treatment Scenarios:** Evaluates the efficacy of different treatment strategies.
- **Customizable Parameters:** Allows adjustment of various parameters to simulate different scenarios.
- **Visualization Tools:** Includes tools for visualizing the simulation results.
- **IBM and NBM:** Leverages Individual-Based Models and Network-Based Models for more accurate and comprehensive simulations.
- **TB natural history model structures:**
  - **Natural history model** (`TB_EMOD`): Agent-based (stochastic) TB progression with latent infection split into fast/slow pathways, progression through an active pre-symptomatic stage, and active disease stratified into smear-positive, smear-negative, and extra-pulmonary states, with TB mortality. See Abu-Raddad et al. (2009) for background: https://doi.org/10.1073/pnas.0901720106
  - **LSHTM-style spectrum model** (`TB_LSHTM`, `TB_LSHTM_Acute`): Agent-based implementation of the LSHTM “spectrum of TB disease” natural history (susceptible → infection/cleared → non-infectious → asymptomatic → symptomatic → treatment/treated/death; optional acute infectious state), designed to support evaluating active case-finding / population-wide screening algorithms (e.g., CXR and NAAT workflows) as in Schwalb et al. 2025 ([PLOS Glob Public Health](https://journals.plos.org/globalpublichealth/article?id=10.1371/journal.pgph.0005050)).

## Getting Started

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
- `docs/` -- Documentation source (MkDocs)

### Running a sample simulation

1. Navigate to the folder `tbsim_examples`
2. Run the script:
   ```bash
   python run_tb_lshtm.py
   ```
3. Running this script should result in basic charts being displayed.

Or run directly in Python:

```python
import starsim as ss
import tbsim

sim = ss.Sim(diseases=tbsim.TB_LSHTM())
sim.run()
sim.plot()
```

## Usage
- Usage examples are available in the **[tbsim_examples](https://github.com/starsimhub/tbsim/tree/main/tbsim_examples)** folder.

## Documentation
TBsim documentation is available at [starsim.org/tbsim](https://starsim.org/tbsim).

TBsim is based on Starsim; please refer to [Starsim documentation](https://docs.starsim.org) for additional information.

## Contributing
Contributions to the TBsim project are welcome! Please read [CONTRIBUTING.md](https://github.com/starsimhub/starsim/blob/main/contributing.md) for details on our code of conduct, and the process for submitting pull requests.

## Authors
-  **Minerva Enriquez** - [menriquez-idm](https://github.com/MEnriquez-IDM)
-  **Daniel Klein** - [daniel-klein](https://github.com/daniel-klein)
-  **Cliff Kerr** - [cliffckerr](https://github.com/cliffckerr)
-  **Ryan Hull** - [rhull-idm](https://github.com/rhull-idm)
-  **Mike Famulare** - [famulare](https://github.com/famulare)
-  **Deven Gokhale** - [gokhale616](https://github.com/gokhale616)
-  **Stewart Chang** - [stchang-idm](https://github.com/stchang-idm)


## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/amath-idm/tbsim/blob/main/LICENSE) file for details.

## Disclaimer
The code in this repository was developed by IDM and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.

## References

[Epidemiological benefits of more-effective tuberculosis vaccines, drugs, and diagnostics](https://www.pnas.org/doi/abs/10.1073/pnas.0901720106?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed)