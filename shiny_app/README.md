# TBsim Shiny Web Application

A web-based interface for running tuberculosis simulations using the TBsim package. This Shiny application provides an intuitive graphical user interface for configuring and running TB transmission simulations, visualizing results, and exporting data.

## Features

- **Interactive Parameter Configuration**: Adjust simulation parameters through a user-friendly interface
- **Real-time Simulation**: Run TB transmission simulations with customizable parameters
- **Interactive Visualizations**: View results using interactive plots powered by Plotly
- **Data Export**: Export simulation results for further analysis
- **Multiple Views**: Results, detailed plots, and raw data tabs for comprehensive analysis

## Prerequisites

- R (version 4.0 or higher)
- Python (version 3.8 or higher)
- Internet connection for package installation

## Installation

### Option 1: Automated Setup (Recommended)

1. Open R or RStudio
2. Navigate to the `shiny_app` directory
3. Run the setup script:
   ```r
   source("setup.R")
   ```

### Option 2: Manual Setup

1. **Install R packages**:
   ```r
   install.packages(c("shiny", "plotly", "DT", "reticulate"))
   ```

2. **Set up Python environment**:
   ```r
   library(reticulate)
   py_install(c("numpy", "scipy", "pandas", "sciris", "matplotlib", 
                "numba", "starsim", "seaborn", "plotly", "lifelines", 
                "tqdm", "networkx"))
   py_install("tbsim", pip = TRUE)
   ```

## Usage

### Starting the Application

1. Navigate to the `shiny_app` directory
2. Start the Shiny app:
   ```r
   shiny::runApp("app.R")
   ```

3. The application will open in your default web browser

### Using the Interface

#### 1. Configure Parameters

Use the sidebar to adjust simulation parameters:

- **Basic Settings**: Population size, simulation dates, time step, random seed
- **TB Disease Parameters**: Initial prevalence, transmission rate, latent TB probability
- **Demographics**: Birth and death rates
- **Social Network**: Average contacts per person

#### 2. Run Simulation

1. Click "Run Simulation" to start the simulation
2. Monitor the status in the sidebar
3. Results will appear in the main panel once complete

#### 3. View Results

- **Results Tab**: Main time series plot showing infected, latent, and active TB cases
- **Detailed Plots Tab**: Multiple subplots for detailed analysis
- **Raw Data Tab**: Tabular view of all simulation data

#### 4. Export Data

Use the "Raw Data" tab to view and export simulation results in tabular format.

## Parameter Descriptions

### Basic Settings
- **Population Size**: Number of individuals in the simulation (100-10,000)
- **Start/End Date**: Simulation time period
- **Time Step**: Days between simulation steps (1-30)
- **Random Seed**: For reproducible results

### TB Disease Parameters
- **Initial Prevalence**: Fraction of population initially infected (0-1)
- **Transmission Rate**: Annual probability of TB transmission (0-0.1)
- **Probability of Fast Latent TB**: Fraction developing fast vs. slow latent TB (0-1)

### Demographics
- **Birth Rate**: Annual births per 1000 population
- **Death Rate**: Annual deaths per 1000 population

### Social Network
- **Average Contacts**: Mean number of contacts per person (1-50)

## Understanding the Results

### Key Metrics
- **Infected**: Total number of individuals with TB infection
- **Latent**: Number with latent TB (not yet symptomatic)
- **Active**: Number with active, symptomatic TB

### Plot Interpretations
- **Time Series**: Shows how TB spreads through the population over time
- **State Transitions**: Shows new infections and disease progression
- **Detailed Analysis**: Multiple views for comprehensive understanding

## Troubleshooting

### Common Issues

1. **Python not found**: Ensure Python 3.8+ is installed and accessible
2. **Package installation fails**: Check internet connection and try again
3. **Simulation errors**: Verify parameter values are within valid ranges
4. **Memory issues**: Reduce population size for large simulations

### Getting Help

- Check the "About" tab for more information
- Review parameter descriptions in the interface
- Ensure all dependencies are properly installed

## Technical Details

### Architecture
- **Frontend**: Shiny UI with reactive programming
- **Backend**: R server with Python integration via reticulate
- **Simulation Engine**: TBsim package built on Starsim framework
- **Visualization**: Plotly for interactive plots

### Performance
- Simulations with 1000 individuals typically complete in seconds
- Larger populations (5000+) may take several minutes
- Memory usage scales with population size

## Citation

If you use this application in your research, please cite:

- The TBsim package and repository
- The original Starsim framework
- Any relevant publications

## License

This application is provided under the same license as the TBsim package (MIT License).

## Contributing

Contributions are welcome! Please see the main TBsim repository for contribution guidelines.

## Support

For technical support or questions:
- Check the main TBsim repository issues
- Review the documentation in the "About" tab
- Ensure all dependencies are properly installed
