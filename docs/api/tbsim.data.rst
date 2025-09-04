TBsim Data Modules
==================

This module provides data extraction, processing, and management utilities for TBsim simulations.

Available Data Modules
---------------------

**extract_south_africa**
   Extract South Africa demographic data from UN World Population Prospects (WPP)

**extract_gtb_data**
   Extract Global TB data for calibration and validation

Key Features
-----------

- **Demographic Data**: Population data from authoritative sources
- **TB Epidemiology**: Global TB statistics and trends
- **Data Processing**: Automated extraction and formatting
- **Calibration Support**: Data for model parameter fitting
- **Validation Data**: Real-world data for model validation

Usage Examples
-------------

Extracting South Africa data:

.. code-block:: python

   from tbsim.data.extract_south_africa import (
       extract_south_africa_cbr,
       extract_south_africa_asmr,
       extract_south_africa_asfr
   )
   
   # Extract crude birth rates
   cbr_data = extract_south_africa_cbr()
   
   # Extract age-sex-specific mortality rates
   asmr_data = extract_south_africa_asmr()
   
   # Extract age-specific fertility rates
   asfr_data = extract_south_africa_asfr()

Extracting Global TB data:

.. code_block:: python

   from tbsim.data.extract_gtb_data import extract_gtb_data
   
   # Extract global TB statistics
   gtb_data = extract_gtb_data(
       countries=['South Africa', 'India', 'China'],
       years=[2010, 2020]
   )

Data Sources
------------

**UN World Population Prospects (WPP)**
   - Life tables by age and sex
   - Demographic indicators
   - Fertility rates by age
   - Population projections

**Global TB Report**
   - TB incidence and prevalence
   - Treatment outcomes
   - Mortality statistics
   - Drug resistance data

**Required Files**
   - WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv
   - WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv
   - WPP2024_Demographic_Indicators_Medium.csv
   - WPP2024_Fertility_by_Age1.csv

Output Files
------------

**Demographic Data**
   - South_Africa_CBR.csv: Crude birth rates by year
   - South_Africa_ASMR.csv: Age-sex-specific mortality rates
   - South_Africa_ASFR.csv: Age-specific fertility rates

**TB Data**
   - Country-specific TB statistics
   - Time series data for calibration
   - Validation datasets

These data modules provide the foundation for realistic TBsim simulations with real-world demographic and epidemiological data.
