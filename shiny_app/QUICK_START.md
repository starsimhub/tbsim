# TBsim Shiny App - Quick Start Guide

## What is this?

A web-based interface for running tuberculosis simulations using the TBsim package. This application allows you to:

- Configure TB simulation parameters through a user-friendly interface
- Run individual-based TB transmission simulations
- Visualize results with interactive plots
- Export data for further analysis

## Quick Start (3 steps)

### 1. Prerequisites
- R (version 4.0+)
- Python (version 3.8+)
- Internet connection

### 2. Setup
```bash
cd shiny_app
./start.sh
```

### 3. Use the App
- The app will open in your web browser
- Adjust parameters in the sidebar
- Click "Run Simulation"
- View results in the main panel

## Alternative Setup Methods

### Method 1: Automated (Recommended)
```bash
cd shiny_app
./start.sh
```

### Method 2: Manual R Setup
```r
# In R or RStudio
source("setup.R")
shiny::runApp("app.R")
```

### Method 3: Docker
```bash
docker-compose up
```
Then visit http://localhost:3838

## Key Features

### Parameter Controls
- **Population Size**: 100-10,000 individuals
- **Simulation Duration**: Custom date ranges
- **TB Parameters**: Transmission rates, prevalence, etc.
- **Demographics**: Birth/death rates
- **Social Networks**: Contact patterns

### Visualizations
- **Time Series**: TB spread over time
- **State Transitions**: Disease progression
- **Detailed Analysis**: Multiple metric views
- **Interactive Plots**: Zoom, hover, export

### Data Export
- **Summary Statistics**: Key metrics table
- **Raw Data**: Complete simulation results
- **Multiple Formats**: CSV, JSON export options

## Example Workflow

1. **Set Parameters**:
   - Population: 1000
   - Duration: 1940-2010
   - Initial prevalence: 1%
   - Transmission rate: 0.0025/year

2. **Run Simulation**:
   - Click "Run Simulation"
   - Wait for completion (usually seconds)

3. **Analyze Results**:
   - View time series plot
   - Check summary statistics
   - Export data if needed

## Troubleshooting

### Common Issues
- **"Python not found"**: Install Python 3.8+
- **"Package installation fails"**: Check internet connection
- **"Simulation errors"**: Verify parameter ranges
- **"Memory issues"**: Reduce population size

### Getting Help
- Check the "About" tab in the app
- Review parameter descriptions
- Run `Rscript test_setup.R` to diagnose issues

## Advanced Usage

### Custom Parameters
- Modify `app.R` for additional parameters
- Add new visualization types
- Extend the analysis capabilities

### Batch Processing
- Use the R API directly for multiple simulations
- Integrate with other analysis tools
- Export results for further analysis

## Performance Tips

### Optimization
- Use smaller populations for testing (100-500)
- Reduce time steps for faster runs
- Close other applications to free memory

### Scaling
- 1000 individuals: ~5-10 seconds
- 5000 individuals: ~30-60 seconds
- 10000 individuals: ~2-5 minutes

## Next Steps

1. **Explore Parameters**: Try different values to see their effects
2. **Compare Scenarios**: Run multiple simulations with different settings
3. **Export Data**: Save results for further analysis
4. **Customize**: Modify the code for your specific needs

## Support

- **Documentation**: Check the README.md file
- **Issues**: Report problems on the GitHub repository
- **Community**: Join the TBsim community discussions

Happy simulating! 🦠📊
