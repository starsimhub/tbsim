# Tuberculosis Modeling using Starsim (TBsim)

**Warning! TBsim is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is not ready to be used for real research or policy questions.**

This repository contains tuberculosis (TB) models built on the [Starsim](https://github.com/starsimhub/starsim) package. It simulates TB spread, treatment, and interventions in a population, with support for comorbidities (HIV, malnutrition), interventions (BCG, TPT, DOTS, diagnostics), and analysis tools.

## Introduction

Tuberculosis is a major global health problem, and understanding its dynamics can help in developing better strategies for control and treatment. TBsim uses Starsim to simulate TB in a population, including transmission, progression, treatment, and the impact of interventions and comorbidities.

## Features

- **Two TB model formulations:**
  - **Natural history model** (`TB`): Latent (slow/fast) and active states (pre-symptomatic, smear +/- , extra-pulmonary), with detailed progression and treatment.
  - **LSHTM-style model** (`TB_LSHTM`, `TB_LSHTM_Acute`): Compartmental formulation with infection → unconfirmed/asymptomatic/symptomatic → treatment, and optional acute-infectious state.
- **Comorbidities:** HIV and malnutrition with connectors for TB interaction.
- **Interventions:** BCG vaccination, TPT, DOTS-style treatment, enhanced diagnostics, health-seeking behavior.
- **Networks:** Random and household networks for contact structure.
- **Analysis:** Dwell-time analyzers, visualizations, and post-processing.
- **IBM and network-based:** Individual-based and network-based transmission.

## Getting Started

Available tutorials (in `docs/tutorials/`):
- `tuberculosis_sim.ipynb` - Basic TB simulation
- `lshtm_model_example.ipynb` - LSHTM-style TB model (TB_LSHTM) quick start
- `tb_interventions_tutorial.ipynb` - TB interventions (BCG, TPT, DOTS, etc.)
- `tbhiv_comorbidity.ipynb` - TB-HIV comorbidity analysis
- `run_tbhiv_scens.ipynb` - TB-HIV scenario runs
- `comprehensive_analyzer_plots_example.ipynb` - Dwell-time analysis and plotting

### Local installation (recommended)

#### Prerequisites

- Python 3.11 or higher
- pip
- Dependencies are listed in `requirements.txt` (Starsim and others are installed with TBsim)

#### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/starsimhub/tbsim.git
   cd tbsim
   ```
2. Install TBsim and its dependencies:
   ```bash
   pip install -e .
   ```

#### Optional: latest Starsim (development)

To use the latest unreleased Starsim features, install Starsim in development mode after installing TBsim:

```bash
git clone https://github.com/starsimhub/starsim.git
cd starsim
pip install -e .
```

#### Running a sample simulation

From the repo root:

```bash
# Natural-history TB model
python scripts/basic/run_tb.py

# LSHTM-style TB model
python scripts/basic/run_tb_lshtm.py
```

Or in Python:

```python
import starsim as ss
from tbsim import TB, TB_LSHTM

# Natural-history model
sim = ss.Sim(diseases=TB())
sim.run()

# LSHTM-style model
sim = ss.Sim(diseases=TB_LSHTM())
sim.run()
```

## Usage

- **Tutorials:** See `docs/tutorials/` (and the [online docs](https://starsimhub.github.io/tbsim/) if built).
- **Scripts:** Example runs and demos are in [scripts/](https://github.com/starsimhub/tbsim/tree/main/scripts) (e.g. `scripts/basic/`).

## Documentation

- **TBsim API and tutorials:** Build locally with `cd docs && make html` (see [Building the Documentation](#building-the-documentation) below), or see the deployed site if available.
- **Starsim:** TBsim is built on [Starsim](https://docs.idmod.org/projects/starsim/en/latest/); refer to its docs for the core framework.

## Contributing

Contributions to the TBsim project are welcome! Please read [CONTRIBUTING.md](https://github.com/starsimhub/starsim/blob/main/contributing.rst) for details on our code of conduct, and the process for submitting pull requests.

## Authors

-  **Daniel Klein** - [daniel-klein](https://github.com/daniel-klein)
-  **Minerva Enriquez** - [menriquez-idm](https://github.com/MEnriquez-IDM)
-  **Stewart Chang** - [stchang-idm](https://github.com/stchang-idm)
-  **Deven Gokhale** - [gokhale616](https://github.com/gokhale616)
-  **Mike Famulare** - [famulare](https://github.com/famulare)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/starsimhub/tbsim/blob/main/LICENSE) file for details.

## Disclaimer
The code in this repository was developed by IDM, the Burnet Institute, and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.


## References

[Epidemiological benefits of more-effective tuberculosis vaccines, drugs, and diagnostics](https://www.pnas.org/doi/abs/10.1073/pnas.0901720106?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed)

## Building the Documentation

To build the API and tutorial docs locally:

1. From the repo root, install doc dependencies:
   ```bash
   pip install -r docs/requirements.txt
   ```
2. Build the HTML docs:
   ```bash
   cd docs && make html
   ```
3. Open `docs/_build/html/index.html` in a browser.
