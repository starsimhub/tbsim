# TB Simulation Script Catalog Web App

A modern web application for browsing, exploring, and executing TB simulation scripts. This web app provides an intuitive interface to discover and run various simulation scenarios, interventions, and analysis tools.

## Features

- **Script Discovery**: Automatically discovers and categorizes all Python scripts in the TB simulation project
- **Interactive Browser**: Browse scripts by category with search and filtering capabilities
- **Real-time Execution**: Execute scripts with live output streaming via WebSocket
- **Script Analysis**: View detailed information about scripts including dependencies, functions, and parameters
- **Modern UI**: Responsive design with Bootstrap 5 and custom styling
- **Category Organization**: Scripts organized by type (basic, interventions, calibration, HIV, etc.)

## Installation

### Prerequisites

- Python 3.8 or higher
- TB simulation environment with required dependencies

### Setup

1. **Navigate to the webapp directory:**
   ```bash
   cd /Users/mine/anaconda_projects/newgit/newtbsim/scripts/webapp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure TB simulation modules are available:**
   ```bash
   # Make sure tbsim and starsim are installed in your environment
   pip install -e /Users/mine/anaconda_projects/newgit/newtbsim
   ```

## Running the Application

### Start the Web Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

### Browsing Scripts

1. **Category View**: The main page shows script categories with counts and descriptions
2. **Script List**: Browse all available scripts with search and filtering
3. **Script Details**: Click on any script to view detailed information

### Executing Scripts

1. **Select a Script**: Click on a script to view its details
2. **Execute**: Click the "Execute Script" button to run the script
3. **Monitor Output**: Watch real-time output in the execution panel
4. **Stop Execution**: Use the stop button to halt execution if needed

### Features

- **Search**: Use the search box to find scripts by name or description
- **Filter**: Filter scripts by category
- **Sort**: Sort scripts by name, category, or modification date
- **Real-time Output**: See script output as it executes
- **Execution Control**: Start, stop, and monitor script execution

## Script Categories

- **Basic**: Core TB simulation scripts
- **Interventions**: TB intervention scenarios and treatments
- **Calibration**: Model calibration and parameter estimation
- **HIV**: HIV-TB coinfection models
- **Burn-in**: Model burn-in and initialization
- **Optimization**: Parameter optimization and sensitivity analysis
- **Analyzers**: Data analysis and visualization tools
- **How-to**: Tutorials and examples

## Technical Details

### Architecture

- **Backend**: Flask with Socket.IO for real-time communication
- **Frontend**: Bootstrap 5 with custom JavaScript
- **Script Discovery**: Automatic analysis of Python scripts
- **Execution**: Subprocess-based script execution with output streaming

### File Structure

```
webapp/
├── app.py                 # Main Flask application
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/           # HTML templates
│   ├── base.html       # Base template
│   └── index.html      # Main page template
└── static/             # Static assets
    ├── css/
    │   └── style.css   # Custom styles
    └── js/
        └── app.js      # Frontend JavaScript
```

### API Endpoints

- `GET /` - Main application page
- `GET /api/scripts` - Get all scripts and categories
- `GET /api/script/<path>` - Get detailed script information
- `POST /api/execute` - Execute a script
- `POST /api/stop` - Stop current execution
- `GET /api/status` - Get execution status

### WebSocket Events

- `output` - Real-time script output
- `execution_complete` - Script execution finished
- `execution_error` - Execution error occurred

## Troubleshooting

### Common Issues

1. **Scripts not discovered**: Ensure the scripts directory path is correct in `app.py`
2. **Execution fails**: Check that all required dependencies are installed
3. **WebSocket connection issues**: Ensure the server is running and accessible

### Debug Mode

Run with debug mode for detailed error information:
```bash
export FLASK_DEBUG=1
python app.py
```

## Customization

### Adding New Script Categories

Edit the `script_patterns` dictionary in `app.py` to add new categories:

```python
script_patterns = {
    'new_category': {
        'path': 'new_category',
        'description': 'Description of new category',
        'icon': 'icon-name',
        'color': 'color-name'
    }
}
```

### Modifying Script Analysis

The `analyze_script` method in the `ScriptCatalog` class can be extended to extract additional metadata from scripts.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the TB simulation framework and follows the same licensing terms.

