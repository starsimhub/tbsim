# Tuberculosis Modeling using starsim (TBSim)

This repository contains the implementation of a new Tuberculosis (TB) Model using the StarSim package. The model aims to simulate the dynamics of TB spread and treatment efficacy under various scenarios.

## Introduction

Tuberculosis is a major global health problem, and understanding its dynamics can help in developing better strategies for control and treatment. This project uses the StarSim package to simulate TB spread in a population, considering factors like transmission rates, treatment efficacy, and social dynamics.

## Features

- **TB Dynamics Simulation:** Models the spread of TB in a given population.
- **Treatment Scenarios:** Evaluates the efficacy of different treatment strategies.
- **Customizable Parameters:** Allows adjustment of various parameters to simulate different scenarios.
- **Visualization Tools:** Includes tools for visualizing the simulation results.
- **IBM and NBM:** Leverages Individual-Based Models and Network-Based Models for more accurate and comprehensive simulations.

## Getting Started

### Prerequisites

- Python 3.x
- StarSim package
- Other dependencies (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/starsimhub/tbsim.git 
   cd tbsim
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

1. Navigate to the directory **_scripts_**
2. Run the script:
   ```bash
   python run_tb.py
   ```
3. running this script should result in basic charts being displayed.

## Usage 
- Usage examples are available in the **[scripts](https://github.com/starsimhub/tbsim/tree/main/scripts)** folder.

## Documentation: 
_TBSim_ is based on Starsim, please refer to [Starsim documentation](https://docs.idmod.org/projects/starsim/en/latest/) for additional information.

## Contributing

Contributions to the TBSim project are welcome! Please read [CONTRIBUTING.md](https://github.com/starsimhub/starsim/blob/main/contributing.rst) for details on our code of conduct, and the process for submitting pull requests.

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
