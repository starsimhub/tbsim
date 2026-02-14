# Tuberculosis Modeling with starsim

**Warning! TBsim is still in the early stages of development. It is being shared solely for transparency and to facilitate collaborative development. It is not ready to be used for real research or policy questions.**

TBsim is a computational modeling framework for simulating tuberculosis transmission, disease progression, and treatment outcomes in populations. Built on the Starsim agent-based modeling platform, it enables researchers to evaluate intervention strategies, explore disease dynamics under different conditions, and inform TB control policy decisions.

## Introduction

Tuberculosis is a major global health problem, and understanding its dynamics can help in developing better strategies for control and treatment. This project uses the Starsim package to simulate TB spread in a population, considering factors like transmission rates, treatment efficacy, and social dynamics.

## Features

- **TB Dynamics Simulation:** Models the spread of TB in a given population.
- **Treatment Scenarios:** Evaluates the efficacy of different treatment strategies.
- **Customizable Parameters:** Allows adjustment of various parameters to simulate different scenarios.
- **Visualization Tools:** Includes tools for visualizing the simulation results.
- **IBM and NBM:** Leverages Individual-Based Models and Network-Based Models for more accurate and comprehensive simulations.
- **TB natural history model structures:**
  - **Natural history model** (`TB`): Agent-based (stochastic) TB progression with latent infection split into fast/slow pathways, progression through an active pre-symptomatic stage, and active disease stratified into smear-positive, smear-negative, and extra-pulmonary states, with TB mortality. See Abu-Raddad et al. (2009) for background: https://doi.org/10.1073/pnas.0901720106
  - **LSHTM-style spectrum model** (`TB_LSHTM`, `TB_LSHTM_Acute`): Agent-based implementation of the LSHTM “spectrum of TB disease” natural history (susceptible → infection/cleared → non-infectious → asymptomatic → symptomatic → treatment/treated/death; optional acute infectious state), designed to support evaluating active case-finding / population-wide screening algorithms (e.g., CXR and NAAT workflows) as in Schwalb et al. 2025 ([PLOS Glob Public Health](https://journals.plos.org/globalpublichealth/article?id=10.1371/journal.pgph.0005050)).

## Getting Started

### Option 1: Run Online (Google Colab)

To run tutorials online, open a tutorial notebook in Google Colab (links are included at the top of each tutorial notebook) and run the install cell to install TBsimV2 from GitHub.

Available tutorials:
- `tb_interventions_tutorial.ipynb` - TB interventions modeling
- `tbhiv_comorbidity.ipynb` - TB-HIV comorbidity analysis  
- `tuberculosis_sim.ipynb` - Basic TB simulation

### Option 2: Local Installation

#### Prerequisites

- Python 3.11 or higher
- Starsim package
- Other dependencies (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/starsimhub/tbsimV2.git
   cd tbsimV2
   ```
2. Install the required packages:
   ```bash
   pip install -e .
   ```

### Starsim:
The steps described below will allow you to use the latest unreleased features of starsim, it needs to be run after tbsim has been installed to allow for the package to be updated:
1. Clone and install starsim:
   ```bash
   git clone https://github.com/starsimhub/starsim.git 
   cd starsim
   pip install -e .

   ```

### Running a sample simulation

1. Navigate to the directory **_scripts/basic_**
2. Run the script:
   ```bash
   python run_tb.py
   ```
3. running this script should result in basic charts being displayed.

## Usage 
- Usage examples are available in the **[scripts](https://github.com/starsimhub/tbsim/tree/main/scripts)** folder.

## Documentation: 
_TBsim_ is based on Starsim, please refer to [Starsim documentation](https://docs.idmod.org/projects/starsim/en/latest/) for additional information.

## Contributing

Contributions to the TBsim project are welcome! Please read [CONTRIBUTING.md](https://github.com/starsimhub/starsim/blob/main/contributing.rst) for details on our code of conduct, and the process for submitting pull requests.

## Authors

-  **Daniel Klein** - [daniel-klein](https://github.com/daniel-klein)
-  **Minerva Enriquez** - [menriquez-idm](https://github.com/MEnriquez-IDM)
-  **Stewart Chang** - [stchang-idm](https://github.com/stchang-idm)
-  **Deven Gokhale** - [gokhale616](https://github.com/gokhale616)
-  **Mike Famulare** - [famulare](https://github.com/famulare)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/amath-idm/tbsim/blob/main/LICENSE) file for details.

## Disclaimer
The code in this repository was developed by IDM, the Burnet Institute, and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.


## References

[Epidemiological benefits of more-effective tuberculosis vaccines, drugs, and diagnostics](https://www.pnas.org/doi/abs/10.1073/pnas.0901720106?url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org&rfr_dat=cr_pub++0pubmed)

## Building the Documentation

To build the documentation locally:

1. Install the documentation dependencies:
   ```bash
   pip install -r docs/requirements.txt
   ```
2. Build the docs:
   ```bash
   cd docs
   make html
   ```
3. The generated HTML will be in `docs/_build/html/index.html`.
