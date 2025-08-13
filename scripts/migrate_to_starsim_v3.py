#!/usr/bin/env python3
"""
Migration script for converting TBSim from Starsim v2 to v3.

This script performs automated migrations for common patterns:
- ss.rate_prob() -> ss.per() 
- ss.beta() -> ss.prob()
- ss.BoolState() -> ss.BoolState()
- unit parameter removal in Sim constructor
- Time parameter changes

Usage:
    python scripts/migrate_to_starsim_v3.py [--dry-run] [--file path/to/file.py]
"""

import os
import re
import argparse
import glob
from pathlib import Path

def migrate_file(filepath, dry_run=False):
    """Migrate a single file from Starsim v2 to v3."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Migration patterns
    migrations = [
        # ss.rate_prob() -> ss.per() (preferred) or ss.prob()
        (r'ss\.rate_prob\(([^)]+)\)', r'ss.per(\1)'),
        
        # ss.beta() -> ss.prob() 
        (r'ss\.beta\(([^)]+)\)', r'ss.prob(\1)'),
        
        # ss.BoolState() -> ss.BoolState()
        (r'ss\.State\(', r'ss.BoolState('),
        
        # Remove unit parameter from Sim constructor and other contexts
        (r'(\s+)unit\s*=\s*[\'"][^\'"]+[\'"],?\s*', r'\1'),
        
        # Update dt parameter to use time units when unit is specified
        (r'dt\s*=\s*(\d+),\s*unit\s*=\s*[\'"]([^\'"]+)[\'"]', r'dt=ss.\2s(\1)'),
        
        # Handle standalone dt parameters (convert to days by default)
        (r'dt\s*=\s*(\d+)(?!\s*,\s*unit)', r'dt=ss.days(\1)'),
        
        # Handle unit parameter in rate_prob context
        (r'unit\s*=\s*[\'"]([^\'"]+)[\'"]', r'dt=ss.\1s()'),
    ]
    
    for pattern, replacement in migrations:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        if dry_run:
            print(f"  Would modify {filepath}")
            # Show some key changes
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'ss.per(' in line or 'ss.prob(' in line or 'ss.BoolState(' in line:
                    print(f"    Line {i+1}: {line.strip()}")
        else:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"  Modified {filepath}")
    else:
        print(f"  No changes needed for {filepath}")

def find_python_files(directory):
    """Find all Python files in the directory tree."""
    patterns = ['*.py', '*.ipynb']
    files = []
    
    for pattern in patterns:
        files.extend(glob.glob(f"{directory}/**/{pattern}", recursive=True))
    
    return files

def main():
    parser = argparse.ArgumentParser(description='Migrate TBSim from Starsim v2 to v3')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--file', help='Migrate specific file only')
    parser.add_argument('--directory', default='.', help='Directory to process (default: current)')
    
    args = parser.parse_args()
    
    if args.file:
        files = [args.file]
    else:
        files = find_python_files(args.directory)
        # Filter out common directories to avoid
        files = [f for f in files if not any(exclude in f for exclude in [
            'venv/', '__pycache__/', '.git/', 'tbsim.egg-info/'
        ])]
    
    print(f"Found {len(files)} Python files to process")
    
    for filepath in files:
        if os.path.exists(filepath):
            migrate_file(filepath, args.dry_run)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main() 