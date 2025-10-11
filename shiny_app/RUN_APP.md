# Running the TBsim Shiny Web Application

## Quick Start

The TBsim Shiny app has been verified and is ready to run. All tests passed successfully! ✓

### Prerequisites

Make sure you have:
1. ✅ R installed (with required packages: shiny, plotly, DT, reticulate, shinydashboard, shinyBS)
2. ✅ Python 3.8+ with tbsim and starsim packages installed
3. ✅ Virtual environment set up at `/Users/mine/gitweb/tbsim/venv/`

### Running the App

#### Option 1: Using Rscript (Recommended)
```bash
cd /Users/mine/gitweb/tbsim/shiny_app
Rscript -e "shiny::runApp('app.R', port=3838, host='0.0.0.0')"
```

#### Option 2: Using R Console
```bash
cd /Users/mine/gitweb/tbsim/shiny_app
R
```
Then in R:
```r
shiny::runApp('app.R')
```

#### Option 3: Using RStudio
1. Open `app.R` in RStudio
2. Click the "Run App" button in the top-right corner

### Accessing the App

Once running, open your browser and navigate to:
- **Local:** http://localhost:3838
- **Network:** http://your-ip-address:3838

### Features

The app includes:
- 🎛️ **Interactive parameter controls** for TB disease modeling
- 📊 **Real-time visualization** of simulation results
- 💉 **BCG vaccination modeling** with coverage analysis
- 🔬 **Advanced visualizations** including Sankey diagrams
- 📈 **Comparison mode** to evaluate intervention impacts
- 🧩 **Custom components** (interventions, analyzers, networks)
- 🌙 **Dark/Light theme toggle** for comfortable viewing
- 📥 **Export capabilities** for data and plots

### Validation

Run the validation script to ensure everything is working:
```bash
cd /Users/mine/gitweb/tbsim/shiny_app
Rscript test_app_validation.R
```

Expected output:
```
==============================================
✓ ALL TESTS PASSED!
The Shiny app is ready to run.
==============================================
```

### Troubleshooting

#### Python Module Not Found
If you get errors about missing Python modules:
1. Activate the virtual environment: `source /Users/mine/gitweb/tbsim/venv/bin/activate`
2. Install missing packages: `pip install tbsim starsim`
3. Run the app again

#### Port Already in Use
If port 3838 is already in use, change the port:
```bash
Rscript -e "shiny::runApp('app.R', port=8080, host='0.0.0.0')"
```

#### R Package Missing
If you're missing R packages:
```r
install.packages(c("shiny", "plotly", "DT", "reticulate", "shinydashboard", "shinyBS"))
```

### Configuration

The app automatically detects the Python environment in the following order:
1. `VIRTUAL_ENV` environment variable
2. `/Users/mine/gitweb/tbsim/venv/bin/python`
3. Relative paths (`../venv/bin/python`, `venv/bin/python`, etc.)
4. System Python (`python3`)

To use a specific Python installation, set the `VIRTUAL_ENV` environment variable:
```bash
export VIRTUAL_ENV=/path/to/your/venv
Rscript -e "shiny::runApp('app.R')"
```

### Performance Tips

For better performance with large populations:
- Start with smaller population sizes (1000-5000 agents)
- Use longer time steps (7 days) for faster simulations
- Enable fewer custom components initially
- Use the logarithmic scale for better visualization

### Advanced Usage

#### Docker Deployment
See `Dockerfile` and `docker-compose.yml` for containerized deployment.

#### Custom Python Environment
Edit the `configure_python_env()` function in `app.R` (lines 13-48) to add custom Python paths.

#### Adding Custom Components
Use the "Simulation Components" panel in the app to add:
- Custom interventions (BCG, TB treatment, diagnostics)
- Custom analyzers (dwell time analysis)
- Custom networks (household, RATIONS trial)
- Custom connectors (TB-HIV, TB-malnutrition)

## Support

For issues or questions:
1. Check the validation script output
2. Review the troubleshooting section above
3. Check the GitHub repository: https://github.com/starsimhub/tbsim

---

**Status:** ✅ App verified and working (all validation tests passed)
**Last Updated:** 2025-10-11


