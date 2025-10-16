#!/usr/bin/env python3
"""
TB Simulation Script Catalog Web App

A web application for browsing, exploring, and executing TB simulation scripts.
Provides a user-friendly interface to discover and run various simulation scenarios.
"""

import os
import sys
import json
import subprocess
import threading
import time
import re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import ast
import inspect

# Add the parent directory to the path to import tbsim modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tb_sim_webapp_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for script execution
current_process = None
execution_output = []

class ScriptCatalog:
    """Manages the discovery and categorization of simulation scripts."""
    
    def __init__(self, scripts_root):
        self.scripts_root = Path(scripts_root)
        self.scripts = {}
        self.categories = {}
        self.discover_scripts()
    
    def discover_scripts(self):
        """Discover and categorize all Python scripts in the scripts directory."""
        script_patterns = {
            'basic': {
                'path': 'basic',
                'description': 'Basic TB simulation scripts',
                'icon': 'play-circle',
                'color': 'blue'
            },
            'interventions': {
                'path': 'interventions',
                'description': 'TB intervention scenarios and treatments',
                'icon': 'medical-kit',
                'color': 'green'
            },
            'calibration': {
                'path': 'calibration',
                'description': 'Model calibration and parameter estimation',
                'icon': 'target',
                'color': 'orange'
            },
            'hiv': {
                'path': 'hiv',
                'description': 'HIV-TB coinfection models',
                'icon': 'heart-pulse',
                'color': 'red'
            },
            'burn_in': {
                'path': 'burn_in',
                'description': 'Model burn-in and initialization',
                'icon': 'fire',
                'color': 'purple'
            },
            'optimization': {
                'path': 'optimization',
                'description': 'Parameter optimization and sensitivity analysis',
                'icon': 'chart-line',
                'color': 'teal'
            },
            'analyzers': {
                'path': 'analyzers',
                'description': 'Data analysis and visualization tools',
                'icon': 'chart-bar',
                'color': 'indigo'
            },
            'howto': {
                'path': 'howto',
                'description': 'Tutorials and examples',
                'icon': 'book',
                'color': 'gray'
            }
        }
        
        for category, info in script_patterns.items():
            category_path = self.scripts_root / info['path']
            if category_path.exists():
                self.categories[category] = info
                self.scripts[category] = []
                
                # Find all Python scripts in this category
                for py_file in category_path.rglob('*.py'):
                    if py_file.name != '__init__.py':
                        script_info = self.analyze_script(py_file, category)
                        if script_info:
                            self.scripts[category].append(script_info)
    
    def analyze_script(self, script_path, category):
        """Analyze a Python script to extract metadata and information."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract basic information
            script_info = {
                'name': script_path.stem,
                'path': str(script_path.relative_to(self.scripts_root)),
                'full_path': str(script_path),
                'category': category,
                'size': script_path.stat().st_size,
                'modified': script_path.stat().st_mtime,
                'description': '',
                'parameters': [],
                'functions': [],
                'dependencies': [],
                'executable': True
            }
            
            # Extract docstring and description
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            if docstring_match:
                script_info['description'] = docstring_match.group(1).strip()
            
            # Extract function definitions
            function_matches = re.findall(r'def\s+(\w+)\s*\([^)]*\):', content)
            script_info['functions'] = function_matches
            
            # Extract imports to identify dependencies
            import_matches = re.findall(r'^(?:from\s+\w+\s+)?import\s+([^\n]+)', content, re.MULTILINE)
            script_info['dependencies'] = [imp.strip() for imp in import_matches]
            
            # Check if script has main execution block
            has_main = '__main__' in content
            script_info['executable'] = has_main
            
            # Try to extract parameter information from function signatures
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name in ['main', 'run', 'execute']:
                        for arg in node.args.args:
                            if arg.arg != 'self':
                                script_info['parameters'].append({
                                    'name': arg.arg,
                                    'type': 'unknown',
                                    'required': True
                                })
            except:
                pass
            
            return script_info
            
        except Exception as e:
            print(f"Error analyzing script {script_path}: {e}")
            return None

# Initialize script catalog
# Get the scripts directory relative to this file
SCRIPTS_ROOT = Path(__file__).parent.parent
catalog = ScriptCatalog(SCRIPTS_ROOT)

@app.route('/')
def index():
    """Main page showing script categories and overview."""
    return render_template('index.html', 
                          categories=catalog.categories,
                          scripts=catalog.scripts)

@app.route('/api/scripts')
def api_scripts():
    """API endpoint to get all scripts."""
    return jsonify({
        'categories': catalog.categories,
        'scripts': catalog.scripts
    })

@app.route('/api/script/<path:script_path>')
def api_script_details(script_path):
    """API endpoint to get detailed information about a specific script."""
    full_path = catalog.scripts_root / script_path
    
    if not full_path.exists():
        return jsonify({'error': 'Script not found'}), 404
    
    # Re-analyze the script for detailed information
    script_info = catalog.analyze_script(full_path, 'unknown')
    return jsonify(script_info)

@app.route('/api/execute', methods=['POST'])
def execute_script():
    """Execute a script with optional parameters."""
    global current_process, execution_output
    
    data = request.get_json()
    script_path = data.get('script_path')
    parameters = data.get('parameters', {})
    
    print(f"[API] Execute request for script: {script_path}")
    print(f"[API] Parameters: {parameters}")
    
    if not script_path:
        print("[API] Error: No script path provided")
        return jsonify({'error': 'No script path provided'}), 400
    
    # Stop any currently running process
    if current_process:
        current_process.terminate()
        current_process = None
    
    # Clear previous output
    execution_output = []
    
    # Start execution in a separate thread
    def run_script():
        global current_process, execution_output
        
        try:
            # Change to the scripts directory
            os.chdir(catalog.scripts_root)
            
            # Build command
            cmd = [sys.executable, script_path]
            
            # Add parameters if any
            for key, value in parameters.items():
                if value:
                    cmd.extend([f'--{key}', str(value)])
            
            print(f"[EXECUTION] Starting script: {script_path}")
            print(f"[EXECUTION] Command: {' '.join(cmd)}")
            print(f"[EXECUTION] Working directory: {os.getcwd()}")
            
            # Start process
            current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output
            for line in iter(current_process.stdout.readline, ''):
                execution_output.append(line.rstrip())
                print(f"[SCRIPT OUTPUT] {line.rstrip()}")
                print(f"[WEBSOCKET] Emitting output event: {line.rstrip()}")
                socketio.emit('output', {'line': line.rstrip()})
            
            current_process.wait()
            
            print(f"[EXECUTION] Script completed with exit code: {current_process.returncode}")
            print(f"[EXECUTION] Total output lines: {len(execution_output)}")
            
            # Emit completion signal
            socketio.emit('execution_complete', {
                'exit_code': current_process.returncode,
                'output': execution_output
            })
            
        except Exception as e:
            print(f"[EXECUTION ERROR] {str(e)}")
            socketio.emit('execution_error', {'error': str(e)})
        finally:
            current_process = None
    
    # Start execution thread
    thread = threading.Thread(target=run_script)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Script execution started'})

@app.route('/api/stop', methods=['POST'])
def stop_execution():
    """Stop the currently running script."""
    global current_process
    
    if current_process:
        current_process.terminate()
        current_process = None
        return jsonify({'status': 'stopped'})
    else:
        return jsonify({'status': 'no_process'})

@app.route('/api/status')
def execution_status():
    """Get the current execution status."""
    global current_process, execution_output
    
    return jsonify({
        'running': current_process is not None,
        'output': execution_output[-50:] if execution_output else []  # Last 50 lines
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'message': 'Connected to TB Simulation Web App'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

if __name__ == '__main__':
    print("TB Simulation Script Catalog Web App")
    print("=" * 50)
    print(f"Scripts directory: {catalog.scripts_root}")
    print(f"Found {sum(len(scripts) for scripts in catalog.scripts.values())} scripts")
    print(f"Categories: {', '.join(catalog.categories.keys())}")
    print("=" * 50)
    print("Starting web server...")
    print("Access the app at: http://localhost:5001")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
