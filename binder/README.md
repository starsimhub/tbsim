# Binder Configuration for TB Simulation Tutorials

This directory contains the configuration files needed to run the TB simulation tutorials in Binder.

## What is Binder?

Binder allows you to create custom, sharable, interactive, reproducible computing environments from your GitHub repositories. When someone visits your Binder link, they get a live, interactive environment with all the necessary packages and data pre-installed.

## Files in this directory:

- `requirements.txt`: Python packages needed to run the tutorials
- `postBuild`: Script that runs after the environment is built
- `start`: Script that starts the Jupyter server
- `runtime.txt`: Specifies the Python version
- `Dockerfile`: Custom Docker configuration (optional)
- `.binderignore`: Files to exclude from the Binder build

## How to use:

1. **Launch Binder**: Click the Binder badge in the main README or visit the Binder URL
2. **Wait for build**: The environment will take a few minutes to build
3. **Start exploring**: Navigate to the `tutorials` directory and open any notebook

## Available Tutorials:

- `tb_interventions_tutorial.ipynb`: TB interventions modeling
- `tbhiv_comorbidity.ipynb`: TB-HIV comorbidity analysis
- `tuberculosis_sim.ipynb`: Basic TB simulation

## Troubleshooting:

If you encounter issues:
1. Check that all cells are run in order
2. Restart the kernel if needed
3. Check the troubleshooting section in each tutorial
4. Ensure you have a stable internet connection

## Customization:

To modify the environment:
1. Edit `requirements.txt` to add/remove packages
2. Modify `postBuild` to add custom setup steps
3. Update `runtime.txt` to change Python version
4. Customize `Dockerfile` for more complex setups 