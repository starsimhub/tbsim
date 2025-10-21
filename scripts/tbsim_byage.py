"""
Population Initialization from Demographic Data
================================================

This script demonstrates how to initialize a TBsim/Starsim population with specific
age and sex distributions from a CSV file, using the latest Starsim 3.0+ API.

Features:
---------
1. Loading age-sex cross-tabulated demographic data from a CSV file
2. Creating a population that matches the specified demographic structure
3. Validating the resulting population against the input data
4. Generating population pyramid visualizations

Two Initialization Methods:
---------------------------

1. **'exact' method**: Creates precise age-sex distribution
   - Each age-sex group matches CSV counts exactly
   - Best for: Calibration, validation, matching census data
   - Implementation: Post-initialization assignment of age/sex arrays
   
2. **'starsim' method**: Uses Starsim's native distribution sampling
   - Ages sampled from ss.histogram distribution
   - Sex sampled from ss.bernoulli with overall proportion
   - Best for: Flexible modeling, demographic uncertainty
   - Implementation: age_data parameter in ss.People()

CSV Format:
----------
age_min,age_max,male_count,female_count
0,5,450,430
5,10,480,460
...

Usage:
------
# Method 1: Exact matching
sim = initialize_population_from_demographics('data.csv', method='exact')

# Method 2: Starsim distribution sampling  
sim = initialize_population_from_demographics('data.csv', method='starsim')

# With population scaling
sim = initialize_population_from_demographics('data.csv', scale_factor=0.1)

Updated for Starsim 3.0+ API patterns.
"""

import pandas as pd
import numpy as np
import sciris as sc
import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt


def load_demographic_data(filepath):
    """
    Load demographic data from a CSV file.
    
    Expected CSV format:
    - age_min: minimum age for age group
    - age_max: maximum age for age group
    - male_count: number of males in this age group
    - female_count: number of females in this age group
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with demographic data
    """
    df = pd.read_csv(filepath)
    required_cols = ['age_min', 'age_max', 'male_count', 'female_count']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    return df


def create_age_distribution(demo_data):
    """
    Convert demographic data into age distribution format for Starsim.
    
    Starsim expects age data in histogram format with 'age' and 'value' columns,
    where 'age' represents bin edges and 'value' represents counts or proportions.
    
    Args:
        demo_data: DataFrame with demographic data (age_min, age_max, male_count, female_count)
        
    Returns:
        DataFrame with 'age' and 'value' columns suitable for ss.People(age_data=...)
    """
    age_bins = []
    age_counts = []
    
    for _, row in demo_data.iterrows():
        age_bins.append(row['age_min'])
        total_count = row['male_count'] + row['female_count']
        age_counts.append(total_count)
    
    # Add upper edge for the last age bin
    if len(demo_data) > 0:
        last_row = demo_data.iloc[-1]
        age_bins.append(last_row['age_max'])
        age_counts.append(0)  # Zero count to mark upper boundary
    
    age_dist_df = pd.DataFrame({
        'age': age_bins,
        'value': age_counts
    })
    
    return age_dist_df


def create_sex_distribution(demo_data):
    """
    Calculate the proportion of females from demographic data.
    
    Args:
        demo_data: DataFrame with demographic data
        
    Returns:
        Float representing proportion of females (for ss.bernoulli distribution)
    """
    total_male = demo_data['male_count'].sum()
    total_female = demo_data['female_count'].sum()
    total = total_male + total_female
    
    if total == 0:
        return 0.5  # Default to 50/50 if no data
    
    return total_female / total


def initialize_population_from_demographics(demo_filepath, scale_factor=1.0, method='starsim'):
    """
    Initialize a TB simulation population based on demographic data from a file.
    
    Uses latest Starsim 3.0+ API with age_data parameter for People initialization.
    
    Args:
        demo_filepath: Path to demographic data CSV file
        scale_factor: Factor to scale the population (1.0 = use counts as-is)
        method: 'starsim' for distribution sampling, 'exact' for precise age-sex matching
        
    Returns:
        Initialized simulation object
    """
    # Load demographic data
    demo_data = load_demographic_data(demo_filepath)
    
    # Calculate total population
    total_pop = int((demo_data['male_count'].sum() + demo_data['female_count'].sum()) * scale_factor)
    
    # Scale the demographic data if needed
    if scale_factor != 1.0:
        demo_data = demo_data.copy()
        demo_data['male_count'] = (demo_data['male_count'] * scale_factor).astype(int)
        demo_data['female_count'] = (demo_data['female_count'] * scale_factor).astype(int)
    
    if method == 'exact':
        # Create exact age-sex distribution
        ages = []
        sexes = []
        
        for _, row in demo_data.iterrows():
            age_min = row['age_min']
            age_max = row['age_max']
            male_count = int(row['male_count'])
            female_count = int(row['female_count'])
            
            # Generate ages uniformly within age group
            male_ages = np.random.uniform(age_min, age_max, male_count)
            female_ages = np.random.uniform(age_min, age_max, female_count)
            
            ages.extend(male_ages)
            ages.extend(female_ages)
            sexes.extend([False] * male_count)  # False = male
            sexes.extend([True] * female_count)  # True = female
        
        # Create People object
        people = ss.People(n_agents=total_pop)
        
        # Create simulation
        sim = ss.Sim(
            people=people,
            diseases=mtb.TB(),
            dt=ss.days(7),
            start=ss.date('2000-01-01'),
            stop=ss.date('2010-01-01'),
        )
        
        # Initialize the simulation
        sim.init()
        
        # Set exact ages and sexes
        sim.people.age[:] = np.array(ages)
        sim.people.female[:] = np.array(sexes)
        
    else:  # method == 'starsim'
        # Create age distribution for Starsim
        age_dist_df = create_age_distribution(demo_data)
        
        # Calculate sex distribution (proportion female)
        p_female = create_sex_distribution(demo_data)
        
        # Create People object with custom age and sex distributions
        # Starsim will sample from these distributions during initialization
        people = ss.People(
            n_agents=total_pop,
            age_data=age_dist_df,
        )
        
        # Override default female distribution with our calculated proportion
        people.female.default = ss.bernoulli(p=p_female, name='female')
        
        # Create simulation with custom population
        sim = ss.Sim(
            people=people,
            diseases=mtb.TB(),
            dt=ss.days(7),
            start=ss.date('2000-01-01'),
            stop=ss.date('2010-01-01'),
        )
        
        # Initialize the simulation
        sim.init()
    
    return sim


def validate_population_demographics(sim, demo_data, tolerance=0.1):
    """
    Validate that the generated population matches the input demographic data.
    
    Since Starsim samples from distributions, exact matches are not expected.
    Validation checks if actual values are within tolerance of expected values.
    
    Args:
        sim: Initialized simulation object
        demo_data: Original demographic data DataFrame
        tolerance: Acceptable relative deviation (default 0.1 = 10%)
        
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # Check total population
    total_expected = demo_data['male_count'].sum() + demo_data['female_count'].sum()
    total_actual = len(sim.people)
    rel_diff = abs(total_actual - total_expected) / total_expected if total_expected > 0 else 0
    validation['total_population'] = {
        'expected': int(total_expected),
        'actual': int(total_actual),
        'match': total_expected == total_actual,
        'rel_diff': rel_diff,
        'within_tolerance': rel_diff <= tolerance
    }
    
    # Check sex distribution
    male_expected = demo_data['male_count'].sum()
    female_expected = demo_data['female_count'].sum()
    male_actual = (~sim.people.female).sum()
    female_actual = sim.people.female.sum()
    
    male_rel_diff = abs(male_actual - male_expected) / male_expected if male_expected > 0 else 0
    female_rel_diff = abs(female_actual - female_expected) / female_expected if female_expected > 0 else 0
    
    validation['sex_distribution'] = {
        'male': {
            'expected': int(male_expected), 
            'actual': int(male_actual),
            'rel_diff': male_rel_diff,
            'within_tolerance': male_rel_diff <= tolerance
        },
        'female': {
            'expected': int(female_expected), 
            'actual': int(female_actual),
            'rel_diff': female_rel_diff,
            'within_tolerance': female_rel_diff <= tolerance
        }
    }
    
    # Check age distribution by group
    age_validation = []
    for _, row in demo_data.iterrows():
        age_min, age_max = row['age_min'], row['age_max']
        mask = (sim.people.age >= age_min) & (sim.people.age < age_max)
        
        actual_male = (~sim.people.female & mask).sum()
        actual_female = (sim.people.female & mask).sum()
        
        male_exp = row['male_count']
        female_exp = row['female_count']
        
        male_rel_diff = abs(actual_male - male_exp) / male_exp if male_exp > 0 else 0
        female_rel_diff = abs(actual_female - female_exp) / female_exp if female_exp > 0 else 0
        
        age_validation.append({
            'age_range': f"{age_min}-{age_max}",
            'male': {
                'expected': int(male_exp), 
                'actual': int(actual_male),
                'rel_diff': male_rel_diff,
                'within_tolerance': male_rel_diff <= tolerance
            },
            'female': {
                'expected': int(female_exp), 
                'actual': int(actual_female),
                'rel_diff': female_rel_diff,
                'within_tolerance': female_rel_diff <= tolerance
            }
        })
    
    validation['age_sex_distribution'] = age_validation
    
    return validation


def plot_population_pyramid(sim, demo_data, filename=None):
    """
    Create a population pyramid comparing expected vs actual distributions.
    
    Args:
        sim: Initialized simulation object
        demo_data: Original demographic data DataFrame
        filename: Optional filename to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract age groups
    age_labels = [f"{row['age_min']}-{row['age_max']}" for _, row in demo_data.iterrows()]
    y_pos = np.arange(len(age_labels))
    
    # Expected counts
    male_expected = demo_data['male_count'].values
    female_expected = demo_data['female_count'].values
    
    # Actual counts
    male_actual = []
    female_actual = []
    for _, row in demo_data.iterrows():
        age_min, age_max = row['age_min'], row['age_max']
        mask = (sim.people.age >= age_min) & (sim.people.age < age_max)
        male_actual.append((~sim.people.female & mask).sum())
        female_actual.append((sim.people.female & mask).sum())
    
    # Plot bars
    bar_height = 0.35
    # Plot expected values as solid bars
    ax.barh(y_pos, -np.array(male_expected), bar_height, 
            label='Male (Expected)', color='steelblue', alpha=0.8)
    ax.barh(y_pos, female_expected, bar_height, 
            label='Female (Expected)', color='coral', alpha=0.8)
    
    # Plot actual values as outlined bars on top
    ax.barh(y_pos, -np.array(male_actual), bar_height, 
            label='Male (Actual)', fill=False, edgecolor='navy', linewidth=2, linestyle='--')
    ax.barh(y_pos, female_actual, bar_height, 
            label='Female (Actual)', fill=False, edgecolor='darkred', linewidth=2, linestyle='--')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(age_labels)
    ax.set_xlabel('Population Count')
    ax.set_ylabel('Age Group')
    ax.set_title('Population Pyramid: Expected vs Actual')
    ax.legend()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Population pyramid saved to {filename}")
    
    plt.show()


def main():
    """
    Main function demonstrating the usage of demographic data loading with latest Starsim.
    """
    print("=" * 80)
    print("TB Simulation: Population Initialization from Demographic Data")
    print("Updated for Starsim 3.0+ API")
    print("=" * 80)
    print()
    
    # Path to demographic data
    demo_filepath = 'data/sample_demographics.csv'
    
    print(f"Loading demographic data from: {demo_filepath}")
    demo_data = load_demographic_data(demo_filepath)
    
    print(f"\nDemographic Summary:")
    print(f"  Age groups: {len(demo_data)}")
    print(f"  Total males: {int(demo_data['male_count'].sum())}")
    print(f"  Total females: {int(demo_data['female_count'].sum())}")
    print(f"  Total population: {int(demo_data['male_count'].sum() + demo_data['female_count'].sum())}")
    print(f"  Proportion female: {create_sex_distribution(demo_data):.3f}")
    print()
    
    # Method selection
    method = 'exact'  # Change to 'starsim' for distribution-based sampling
    scale_factor = 1.0  # Change this to scale population up or down
    
    print(f"Method: '{method}'")
    if method == 'exact':
        print("  - Creates exact age-sex distribution matching CSV data")
        print("  - Each age-sex group will match expected counts precisely")
    else:
        print("  - Uses Starsim's age_data distribution sampling")
        print("  - Overall distribution matches, but individual groups may vary")
    print()
    
    # Initialize population
    print("Initializing population from demographic data...")
    sim = initialize_population_from_demographics(demo_filepath, scale_factor=scale_factor, method=method)
    
    print(f"\nPopulation initialized with {len(sim.people)} agents")
    print(f"  Males: {(~sim.people.female).sum()}")
    print(f"  Females: {sim.people.female.sum()}")
    print(f"  Age range: {sim.people.age.min():.1f} - {sim.people.age.max():.1f}")
    print(f"  Mean age: {sim.people.age.mean():.1f} ± {sim.people.age.std():.1f}")
    print()
    
    # Validate the population
    tolerance = 0.02 if method == 'exact' else 0.1  # Tighter tolerance for exact method
    print(f"Validating population demographics (tolerance = {tolerance*100:.0f}%)...")
    validation = validate_population_demographics(sim, demo_data, tolerance=tolerance)
    
    print(f"\nValidation Results:")
    print(f"  Total population:")
    print(f"    Expected: {validation['total_population']['expected']}")
    print(f"    Actual: {validation['total_population']['actual']}")
    print(f"    Exact match: {validation['total_population']['match']}")
    print(f"    Within tolerance: {validation['total_population']['within_tolerance']} "
          f"(diff: {validation['total_population']['rel_diff']*100:.1f}%)")
    print()
    
    print(f"  Sex distribution:")
    male_val = validation['sex_distribution']['male']
    female_val = validation['sex_distribution']['female']
    print(f"    Males - Expected: {male_val['expected']}, Actual: {male_val['actual']} "
          f"(diff: {male_val['rel_diff']*100:.1f}%, within tolerance: {male_val['within_tolerance']})")
    print(f"    Females - Expected: {female_val['expected']}, Actual: {female_val['actual']} "
          f"(diff: {female_val['rel_diff']*100:.1f}%, within tolerance: {female_val['within_tolerance']})")
    print()
    
    # Check age-sex groups
    age_groups_ok = sum(1 for ag in validation['age_sex_distribution'] 
                       if ag['male']['within_tolerance'] and ag['female']['within_tolerance'])
    print(f"  Age-sex groups within tolerance: {age_groups_ok}/{len(validation['age_sex_distribution'])}")
    if age_groups_ok < len(validation['age_sex_distribution']):
        print(f"  Note: With '{method}' method, some age-sex groups may not match exactly")
    print()
    
    # Plot population pyramid
    print("Generating population pyramid...")
    plot_population_pyramid(sim, demo_data, filename='population_pyramid.png')
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print()
    print("Key takeaways:")
    print("  1. Use method='exact' for precise age-sex distribution matching")
    print("  2. Use method='starsim' for native Starsim age_data distribution sampling")
    print("  3. Starsim 3.0+ People() accepts age_data parameter for demographic structure")
    print("=" * 80)
    
    return sim, demo_data, validation


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the main demonstration
    sim, demo_data, validation = main()

