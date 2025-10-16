# TB Simulation Web App - Quick Start Guide

## 🚀 Quick Start

### 1. Start the Web App
```bash
cd /Users/mine/anaconda_projects/newgit/newtbsim/scripts/webapp
./start_webapp.sh
```

### 2. Open Your Browser
Navigate to: **http://localhost:5001**

### 3. Explore Scripts
- Browse script categories on the main page
- Use search and filters to find specific scripts
- Click on any script to view details and execute

## 📁 What You'll Find

The web app automatically discovers and categorizes scripts from your TB simulation project:

### Categories Available:
- **Basic** - Core TB simulation scripts
- **Interventions** - TB intervention scenarios and treatments  
- **Calibration** - Model calibration and parameter estimation
- **HIV** - HIV-TB coinfection models
- **Burn-in** - Model burn-in and initialization
- **Optimization** - Parameter optimization and sensitivity analysis
- **Analyzers** - Data analysis and visualization tools
- **How-to** - Tutorials and examples

### Key Features:
- 🔍 **Search & Filter** - Find scripts quickly
- 📊 **Script Analysis** - View dependencies, functions, and parameters
- ▶️ **Live Execution** - Run scripts with real-time output
- 🛑 **Execution Control** - Start, stop, and monitor execution
- 📱 **Responsive Design** - Works on desktop and mobile

## 🎯 Example Workflow

1. **Browse Categories**: Click on "Interventions" to see TB intervention scripts
2. **Search Scripts**: Use the search box to find "calibration" scripts
3. **View Details**: Click on a script to see its description and functions
4. **Execute Script**: Click "Execute Script" to run it with live output
5. **Monitor Progress**: Watch real-time output and stop if needed

## 🛠️ Troubleshooting

### If the app won't start:
```bash
# Check Python version (needs 3.8+)
python3 --version

# Install dependencies manually
pip install -r requirements.txt

# Run directly
python3 app.py
```

### If scripts aren't discovered:
- Check that the scripts directory path is correct in `app.py`
- Ensure you have read permissions on the scripts directory

### If execution fails:
- Make sure TB simulation dependencies (tbsim, starsim) are installed
- Check that the script has a `if __name__ == '__main__':` block

## 📚 More Information

- See `README.md` for detailed documentation
- Check `requirements.txt` for all dependencies
- Use `demo_script.py` to test the execution functionality

## 🎉 Enjoy Exploring!

The web app provides an intuitive way to discover and run your TB simulation scripts. Happy simulating!
