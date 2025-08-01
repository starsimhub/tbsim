"""
Extract South Africa demographic data from UN WPP sources

This script downloads and processes UN World Population Prospects (WPP) data for South Africa
to create the necessary demographic files for TB simulation modeling.

Required UN WPP files (download from https://population.un.org/wpp/Download/Standard/):
- WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv
- WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv  
- WPP2024_Demographic_Indicators_Medium.csv
- WPP2024_Fertility_by_Age1.csv

Output files:
- South_Africa_CBR.csv: Crude birth rates by year
- South_Africa_ASMR.csv: Age-sex-specific mortality rates
- South_Africa_ASFR.csv: Age-specific fertility rates
"""

import pandas as pd
import numpy as np
import os

def download_wpp_data():
    """
    Download UN WPP data files if they don't exist locally.
    Note: This requires manual download from UN WPP website.
    """
    required_files = [
        'WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv',
        'WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv',
        'WPP2024_Demographic_Indicators_Medium.csv',
        'WPP2024_Fertility_by_Age1.csv'  # updated to new CSV file
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required UN WPP data files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease download these files from:")
        print("https://population.un.org/wpp/Download/Standard/")
        print("\nPlace them in the same directory as this script.")
        return False
    
    return True

def extract_south_africa_cbr():
    """Extract South Africa crude birth rates from UN WPP data."""
    print("Extracting South Africa crude birth rates...")
    
    try:
        df = pd.read_csv('WPP2024_Demographic_Indicators_Medium.csv')
        df = df.rename(columns={'Time': 'Year'})
        df = df.set_index(['Location', 'Year'])
        
        # Extract South Africa data
        sa_cbr = df.loc['South Africa'][['CBR']].reset_index()
        sa_cbr.to_csv('South_Africa_CBR.csv', index=False)
        
        print(f"✓ Created South_Africa_CBR.csv with {len(sa_cbr)} years of data")
        print(f"  Year range: {sa_cbr['Year'].min()} - {sa_cbr['Year'].max()}")
        print(f"  CBR range: {sa_cbr['CBR'].min():.2f} - {sa_cbr['CBR'].max():.2f}")
        
        return sa_cbr
        
    except Exception as e:
        print(f"✗ Error extracting CBR data: {e}")
        return None

def extract_south_africa_asmr():
    """Extract South Africa age-sex-specific mortality rates from UN WPP data."""
    print("Extracting South Africa age-sex-specific mortality rates...")
    
    try:
        # Read female mortality data
        df_female = pd.read_csv('WPP2024_Life_Table_Complete_Medium_Female_1950-2023.csv')
        df_female = df_female.set_index(['Location', 'Time', 'Sex', 'AgeGrpStart'])
        sa_female = df_female.loc['South Africa'][['mx']].reset_index()
        
        # Read male mortality data  
        df_male = pd.read_csv('WPP2024_Life_Table_Complete_Medium_Male_1950-2023.csv')
        df_male = df_male.set_index(['Location', 'Time', 'Sex', 'AgeGrpStart'])
        sa_male = df_male.loc['South Africa'][['mx']].reset_index()
        
        # Combine and format
        sa_asmr = pd.concat([sa_female, sa_male], axis=0)
        sa_asmr = sa_asmr.rename(columns={'Time': 'Time'})
        sa_asmr = sa_asmr.sort_values(['Time', 'Sex', 'AgeGrpStart'])
        sa_asmr.to_csv('South_Africa_ASMR.csv', index=False)
        
        print(f"✓ Created South_Africa_ASMR.csv with {len(sa_asmr)} records")
        print(f"  Year range: {sa_asmr['Time'].min()} - {sa_asmr['Time'].max()}")
        print(f"  Age range: {sa_asmr['AgeGrpStart'].min()} - {sa_asmr['AgeGrpStart'].max()}")
        print(f"  Mortality rate range: {sa_asmr['mx'].min():.6f} - {sa_asmr['mx'].max():.6f}")
        
        return sa_asmr
        
    except Exception as e:
        print(f"✗ Error extracting ASMR data: {e}")
        return None

def extract_south_africa_asfr():
    """Extract South Africa age-specific fertility rates from UN WPP 2025 CSV data."""
    print("Extracting South Africa age-specific fertility rates...")

    try:
        df = pd.read_csv('WPP2025_Fertility_by_Age1.csv')
        # Adjust these column names as needed based on your CSV
        # For example: Location, Year, Age, ASFR
        sa_fertility = df[df['Location'] == 'South Africa']
        # If needed, filter for ages 15-49
        sa_fertility = sa_fertility[(sa_fertility['Age'] >= 15) & (sa_fertility['Age'] <= 49)]
        sa_fertility = sa_fertility[['Year', 'Age', 'ASFR']]
        sa_fertility = sa_fertility.rename(columns={'Year': 'Time', 'Age': 'AgeGrp'})
        sa_fertility = sa_fertility.set_index(['Time', 'AgeGrp']).sort_index()
        sa_fertility.to_csv('South_Africa_ASFR.csv')

        print(f"✓ Created South_Africa_ASFR.csv with {len(sa_fertility)} records")
        print(f"  Year range: {sa_fertility.index.get_level_values('Time').min()} - {sa_fertility.index.get_level_values('Time').max()}")
        print(f"  Age range: {sa_fertility.index.get_level_values('AgeGrp').min()} - {sa_fertility.index.get_level_values('AgeGrp').max()}")

        return sa_fertility

    except Exception as e:
        print(f"✗ Error extracting ASFR data: {e}")
        return None

def create_south_africa_population_structure():
    """Create South Africa population age structure data for simulation initialization."""
    print("Creating South Africa population age structure...")
    
    # South Africa population age structure (approximate 1960 data)
    # Based on UN WPP data for South Africa
    age_structure = pd.DataFrame({
        'age': np.arange(0, 101, 5),
        'value': [
            12000, 10000, 8500, 7500, 6500, 5500, 4500, 3500, 2500, 2000,
            1500, 1200, 800, 500, 300, 150, 80, 40, 15, 5, 1
        ]
    })
    
    age_structure.to_csv('South_Africa_Age_Structure.csv', index=False)
    print("✓ Created South_Africa_Age_Structure.csv")
    
    return age_structure

def main():
    """Main function to extract all South Africa demographic data."""
    print("South Africa Demographic Data Extraction")
    print("=" * 50)
    
    # Check if required files exist
    if not download_wpp_data():
        return
    
    # Extract all demographic data
    cbr_data = extract_south_africa_cbr()
    asmr_data = extract_south_africa_asmr()
    asfr_data = extract_south_africa_asfr()
    age_structure = create_south_africa_population_structure()
    
    print("\n" + "=" * 50)
    print("Extraction Summary:")
    print(f"✓ CBR data: {'✓' if cbr_data is not None else '✗'}")
    print(f"✓ ASMR data: {'✓' if asmr_data is not None else '✗'}")
    print(f"✓ ASFR data: {'✓' if asfr_data is not None else '✗'}")
    print(f"✓ Age structure: {'✓' if age_structure is not None else '✗'}")
    
    if all(x is not None for x in [cbr_data, asmr_data, asfr_data, age_structure]):
        print("\n🎉 All South Africa demographic data extracted successfully!")
        print("\nFiles created:")
        print("  - South_Africa_CBR.csv")
        print("  - South_Africa_ASMR.csv") 
        print("  - South_Africa_ASFR.csv")
        print("  - South_Africa_Age_Structure.csv")
        print("\nYou can now update your TB simulation code to use these files instead of Vietnam data.")

if __name__ == "__main__":
    main()