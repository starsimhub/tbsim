#!/usr/bin/env python3
"""
Updated migration script for converting TBSim from Starsim v2 to v3.

This script implements the official migration patterns from the Starsim v3 migration guide:
- Rates migration (ss.beta, ss.rate_prob, ss.time_prob, ss.rate)
- Unit argument removal and dt parameter updates
- State renaming (ss.BoolState -> ss.BoolState)
- Starsim examples migration

Usage:
    python scripts/migrate_to_starsim_v3_updated.py [--dry-run] [--file path/to/file.py] [--directory path]
"""

import re
import sys
import argparse
from pathlib import Path

def is_prob(value: str) -> bool:
    """Determine if a value is probably a probability (<1)."""
    try:
        return float(value) < 1
    except:
        return True  # Default to per() if unsure

def migrate_rates(text):
    """
    Migrate v2 ss.rate() and derived classes to appropriate v3 classes based on time units.
    
    - ss.beta:
        - ss.peryear(x) -> ss.peryear(x)  # TODO: Check automatic migration change for ss.beta
        - ss.perday(x) -> ss.perday(x)  # TODO: Check automatic migration change for ss.beta
        - ss.perweek(x) -> ss.perweek(x)  # TODO: Check automatic migration change for ss.beta
        - ss.permonth(x) -> ss.permonth(x)  # TODO: Check automatic migration change for ss.beta
        - ss.peryear(x) -> ss.peryear(x)  # TODO: Check automatic migration change for ss.beta
    - ss.rate_prob:
        - ss.peryear(x) -> ss.peryear(x)  # TODO: Check automatic migration change for ss.rate_prob
        - ss.perperday(x) -> ss.perday(x)  # TODO: Check automatic migration change for ss.rate_prob
    - ss.time_prob:
        - ss.probperyear(x) -> ss.probperyear(x)  # TODO: Check automatic migration change for ss.time_prob
        - ss.probperday(x) -> ss.probperday(x)   # TODO: Check automatic migration change for ss.time_prob
    - ss.rate:
        - For x<1:
            - ss.peryear(x) -> ss.peryear(x)  # TODO: Check automatic migration change for ss.rate
            - ss.perday(x) -> ss.perday(x)  # TODO: Check automatic migration change for ss.rate
        - For x>=1:
            - ss.peryear(x) -> ss.freqperyear(x)  # TODO: Check automatic migration change for ss.rate
            - ss.perday(x) -> ss.freqperday(x)  # TODO: Check automatic migration change for ss.rate
    """
    lines = text.splitlines()
    new_lines = []

    unit_map = {
        'days': 'perday',
        'weeks': 'perweek',
        'months': 'permonth',
        'years': 'peryear',
    }

    freq_map = {
        'days': 'freqperday',
        'weeks': 'freqperweek',
        'months': 'freqpermonth',
        'years': 'freqperyear',
    }

    # Patterns to match calls with optional time unit
    # Each entry is (pattern, replacement_func, token_name)
    patterns = [
        # Two-arg patterns first (must come first to prevent premature matching)
        (r'ss\.beta\s*\(\s*([^,]+)\s*,\s*[\'"](\w+)[\'"]\s*\)', 
         lambda v, u: f"ss.{unit_map.get(u, 'peryear')}({v})", 'ss.beta'), # ss.peryear(x)  # TODO: Check automatic migration change for ss.beta
        
        (r'ss\.time_prob\s*\(\s*([^,]+)\s*,\s*[\'"](\w+)[\'"]\s*\)', 
         lambda v, u: f"ss.prob{unit_map.get(u, 'peryear')}({v})", 'ss.time_prob'), # ss.probperyear(x)  # TODO: Check automatic migration change for ss.time_prob
        
        (r'ss\.rate_prob\s*\(\s*([^,]+)\s*,\s*[\'"](\w+)[\'"]\s*\)', 
         lambda v, u: f"ss.per{unit_map.get(u, 'year')}({v})", 'ss.rate_prob'), # ss.peryear(x)  # TODO: Check automatic migration change for ss.rate_prob
        
        (r'ss\.rate\s*\(\s*([^,]+)\s*,\s*[\'"](\w+)[\'"]\s*\)', 
         lambda v, u: f"ss.{unit_map[u] if is_prob(v) else freq_map[u]}({v})", 'ss.rate'), # ss.peryear(x, 'unit')  # TODO: Check automatic migration change for ss.rate
                         
        # One-arg patterns second
        (r'ss\.beta\s*\(\s*([^)]+?)\s*\)', 
         lambda v: f"ss.peryear({v})", 'ss.beta'), # ss.peryear(x)  # TODO: Check automatic migration change for ss.beta
        
        (r'ss\.time_prob\s*\(\s*([^)]+?)\s*\)', 
         lambda v: f"ss.probperyear({v})", 'ss.time_prob'), # ss.probperyear(x)  # TODO: Check automatic migration change for ss.time_prob
        
        (r'ss\.rate_prob\s*\(\s*([^)]+?)\s*\)', 
         lambda v: f"ss.peryear({v})", 'ss.rate_prob'), # ss.peryear(x)  # TODO: Check automatic migration change for ss.rate_prob
        
        (r'ss\.rate\s*\(\s*([^)]+?)\s*\)', 
         lambda v: f"ss.peryear({v})" if is_prob(v) else f"ss.freqperyear({v})", 'ss.rate'), # ss.peryear(x)  # TODO: Check automatic migration change for ss.rate
    ]

    for orig_line in lines:
        line = orig_line
        for pattern, repl_func, token in patterns:
            match = re.search(pattern, line)
            if not match:
                continue
            try:
                groups = match.groups()
                new_expr = repl_func(*[g.strip() for g in groups])
                line = re.sub(pattern, new_expr, line, count=1)
                # Only apply first successful match per line
                if line != orig_line:
                    line += f"  # TODO: Check automatic migration change for {token}"
                break
            except Exception:
                continue
        new_lines.append(line)

    return '\n'.join(new_lines)

def migrate_unit_arguments(text):
    """
    Migrate v2 unit arguments to v3 dt arguments.
    
    Rules:
    - If dt is not defined or dt=1: change unit=<x> to dt=<x>
    - If dt=<y>, change unit=<x>, dt=<y> to dt=ss.<x>(<y>)
    
    Examples:
    - ss.SIS(dt=1.0) -> ss.SIS(dt=ss.days(1))
    - ss.Births(dt=ss.days(0.25)) -> ss.Births(dt=ss.years(0.25))
    - ss.RandomNet() -> ss.RandomNet(dt='week')
    - ss.Sim(unit='day', dt=ss.weeks(2)) -> ss.Sim(dt=ss.days(2))
    """
    unit_map = {
        'day': 'days',
        'days': 'days',
        'week': 'weeks',
        'weeks': 'weeks',
        'month': 'months',
        'months': 'months',
        'year': 'years',
        'years': 'years',
    }

    # First, handle multi-line patterns by looking for unit and dt in the same dict/block
    lines = text.splitlines()
    new_lines = []
    
    for i, line in enumerate(lines):
        # Check if this line has unit= and the next line has dt=
        if 'unit=' in line and i + 1 < len(lines) and 'dt=' in lines[i + 1]:
            # Extract unit value
            unit_match = re.search(r'unit\s*=\s*[\'"](\w+)[\'"]', line)
            if unit_match:
                unit = unit_match.group(1)
                ss_unit = unit_map.get(unit, unit + 's')
                # Remove unit from this line
                line = re.sub(r'unit\s*=\s*[\'"](\w+)[\'"],?\s*', '', line)
                # Update next line's dt
                dt_match = re.search(r'dt\s*=\s*([\d\.]+)', lines[i + 1])
                if dt_match:
                    dt = dt_match.group(1)
                    lines[i + 1] = re.sub(r'dt\s*=\s*([\d\.]+)', f'dt=ss.{ss_unit}({dt})', lines[i + 1])
        
        # Handle standalone unit= patterns
        if 'unit=' in line and 'dt=' not in line:
            unit_match = re.search(r'unit\s*=\s*[\'"](\w+)[\'"]', line)
            if unit_match:
                unit = unit_match.group(1)
                ss_unit = unit_map.get(unit, unit + 's')
                line = re.sub(r'unit\s*=\s*[\'"](\w+)[\'"]', f'dt=ss.{ss_unit}(1)', line)
        
        # Handle unit=, dt= on same line
        if 'unit=' in line and 'dt=' in line:
            unit_match = re.search(r'unit\s*=\s*[\'"](\w+)[\'"]\s*,\s*dt\s*=\s*([\d\.]+)', line)
            if unit_match:
                unit, dt = unit_match.groups()
                ss_unit = unit_map.get(unit, unit + 's')
                line = re.sub(r'unit\s*=\s*[\'"](\w+)[\'"]\s*,\s*dt\s*=\s*([\d\.]+)', f'dt=ss.{ss_unit}({dt})', line)
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)

def migrate_states(text):
    """
    Migrate ss.BoolState to ss.BoolState.
    """
    return text.replace('ss.BoolState', 'ss.BoolState')

def migrate_starsim_examples(text):
    """
    Migrate example diseases and networks from starsim to starsim_examples namespace.
    """
    names = [
        'ART', 'CD4_analyzer', 'Cholera', 'DiskNet', 'Ebola', 'EmbeddingNet',
        'ErdosRenyiNet', 'Gonorrhea', 'HIV', 'Measles', 'NullNet',
        'Syphilis', 'syph_screening', 'syph_treatment'
    ]

    pattern = re.compile(r'\bss\.(?:' + '|'.join(map(re.escape, names)) + r')\b')
    text = pattern.sub(lambda m: m.group(0).replace('ss.', 'sse.'), text)

    # Add import if needed
    if 'sse.' in text and 'import starsim_examples as sse' not in text:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == 'import starsim as ss':
                lines.insert(i + 1, 'import starsim_examples as sse')
                break
        text = '\n'.join(lines)

    return text

def migrate_file(filepath, dry_run=False):
    """Migrate a single file from Starsim v2 to v3."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply migrations in order
    content = migrate_rates(content)
    content = migrate_unit_arguments(content)
    content = migrate_states(content)
    content = migrate_starsim_examples(content)
    
    if content != original_content:
        if dry_run:
            print(f"  Would modify {filepath}")
            # Show some key changes
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if any(pattern in line for pattern in ['ss.per', 'ss.prob', 'ss.BoolState', 'sse.', 'dt=ss.']):
                    print(f"    Line {i+1}: {line.strip()}")
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Modified {filepath}")
    else:
        print(f"  No changes needed for {filepath}")

def find_python_files(directory):
    """Find all Python files in the directory tree."""
    files = []
    for path in Path(directory).rglob('*.py'):
        if not any(exclude in str(path) for exclude in ['venv/', '__pycache__/', '.git/', 'tbsim.egg-info/', '.DS_Store', 'migrate_to_starsim_v3']):
            files.append(str(path))
    return files

def main():
    parser = argparse.ArgumentParser(description='Migrate TBSim from Starsim v2 to v3 using official patterns')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--file', help='Migrate specific file only')
    parser.add_argument('--directory', default='.', help='Directory to process (default: current)')
    
    args = parser.parse_args()
    
    if args.file:
        files = [args.file]
    else:
        files = find_python_files(args.directory)
    
    print(f"Found {len(files)} Python files to process")
    
    for filepath in files:
        if Path(filepath).exists():
            migrate_file(filepath, args.dry_run)
        else:
            print(f"File not found: {filepath}")

if __name__ == '__main__':
    main() 