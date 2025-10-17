"""
WHO Global TB Report Data Extraction Script

This script extracts and processes TB case notification data from the WHO Global TB Report
2024 data files. It reads RDA (R Data Archive) files, extracts South Africa TB data,
and calculates case notification rates per 100,000 population over time.

Purpose:
--------
- Extract WHO Global TB Report data for South Africa
- Calculate case notification rates from absolute case counts
- Merge TB case data with population data
- Visualize TB case notification trends over time
- Prepare calibration data for TB simulation models

Data Sources:
-------------
The script requires WHO Global TB Report 2024 data files in RDA format:
- tb.rda: TB case notification data by country and year
- pop.rda: Population data by country and year
- dic.rda: Data dictionary and variable definitions

Expected directory structure:
    scripts/data/gtbreport2024/data/gtb/
        ├── snapshot_2024-07-29/
        │   └── tb.rda
        └── other/
            ├── pop.rda
            └── dic.rda

Output:
-------
- Displays columns available in TB data
- Shows notification variables found in the data
- Plots case notification rate over time for South Africa
- Case notification rate per 100,000 population trend plot

Usage:
------
    python scripts/data/extract_gtb_data.py

Requirements:
-------------
- rdata: For reading R data files
- pandas: For data manipulation
- matplotlib: For visualization
- WHO GTB Report 2024 data files (downloaded separately)

Notes:
------
The script automatically searches for the most appropriate notification variable
from the WHO data (e.g., new_bact_pos, new_labconf, new_notif, new_pos) and uses
the first available one for analysis.
"""

import os
import rdata
import pandas as pd
import matplotlib.pyplot as plt

# Paths to GTB report data
base_dir = os.path.dirname(os.path.abspath(__file__))
gtb_dir = os.path.join(base_dir, 'gtbreport2024', 'data', 'gtb')
snapshot_dir = os.path.join(gtb_dir, 'snapshot_2024-07-29')
other_dir = os.path.join(gtb_dir, 'other')

tb_rda_path = os.path.join(snapshot_dir, 'tb.rda')
pop_rda_path = os.path.join(other_dir, 'pop.rda')
dic_rda_path = os.path.join(other_dir, 'dic.rda')

# Helper to load RDA file and return as pandas DataFrame
def load_rda_df(rda_path):
    """
    Load an R Data Archive (RDA) file and convert to pandas DataFrame.
    
    This helper function reads RDA files from WHO Global TB Reports and
    extracts the first DataFrame object found in the file.
    
    Parameters
    ----------
    rda_path : str
        Path to the RDA file to load
    
    Returns
    -------
    pd.DataFrame
        The first DataFrame found in the RDA file
        
    Raises
    ------
    ValueError
        If no DataFrame is found in the RDA file
        
    Notes
    -----
    The function uses the rdata library to parse and convert R objects to
    pandas DataFrames. It searches through all objects in the RDA file and
    returns the first DataFrame encountered.
    """
    parsed = rdata.parser.parse_file(rda_path)
    converted = rdata.conversion.convert(parsed)
    # Find first DataFrame in the RDA file
    for v in converted.values():
        if isinstance(v, pd.DataFrame):
            return v
    raise ValueError(f"No DataFrame found in {rda_path}")

tb_df = load_rda_df(tb_rda_path)
pop_df = load_rda_df(pop_rda_path)
# Optionally load dictionary for variable lookup
dic_df = load_rda_df(dic_rda_path)

# Inspect columns for TB data
print("TB Data Columns:", tb_df.columns.tolist())

# Find South Africa country code (should be 'ZAF')
if 'iso3' in tb_df.columns:
    sa_code = 'ZAF'
    tb_sa = tb_df[tb_df['iso3'] == sa_code]
    pop_sa = pop_df[pop_df['iso3'] == sa_code]
else:
    # Try alt country code column
    raise ValueError('iso3 column not found in TB data')

# Find notification variable(s) (e.g., new_bact_pos, new_labconf, etc.)
notif_vars = [col for col in tb_sa.columns if 'new' in col and ('bact' in col or 'labconf' in col or 'notif' in col or 'pos' in col)]
print("Notification variables:", notif_vars)

# Use new_bact_pos if available, else try others
notif_var = None
for v in ['new_bact_pos', 'new_labconf', 'new_notif', 'new_pos']:
    if v in tb_sa.columns:
        notif_var = v
        break
if notif_var is None and notif_vars:
    notif_var = notif_vars[0]
if notif_var is None:
    raise ValueError('No notification variable found in TB data')

# Merge with population by year
if 'year' not in tb_sa.columns:
    raise ValueError('No year column in TB data')
if 'year' not in pop_sa.columns:
    raise ValueError('No year column in population data')

# Some pop files may have 'pop' or 'e_pop_num' as population
pop_col = None
for c in ['pop', 'e_pop_num', 'population']:
    if c in pop_sa.columns:
        pop_col = c
        break
if pop_col is None:
    raise ValueError('No population column found in population data')

# Merge on year
merged = pd.merge(tb_sa[['year', notif_var]], pop_sa[['year', pop_col]], on='year', how='inner')
merged = merged.sort_values('year')

# Calculate notification rate per 100,000
merged['notif_rate_per_100k'] = merged[notif_var] / merged[pop_col] * 1e5

# Plot
plt.figure(figsize=(8,5))
plt.plot(merged['year'], merged['notif_rate_per_100k'], marker='o', color='red', label='Case Notification Rate')
plt.xlabel('Year')
plt.ylabel('Case Notification Rate (per 100,000)')
plt.title('South Africa TB Case Notification Rate Over Time')
plt.grid(True)
plt.legend()
# Set x-axis to show discrete years (e.g., every 5 years)
years = merged['year'].unique()
years_to_show = [y for y in range(years.min(), years.max()+1) if y % 5 == 0 or y == years.min() or y == years.max()]
plt.xticks(years_to_show, rotation=45)
plt.tight_layout()
plt.show() 