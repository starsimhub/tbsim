"""
Sample script demonstrating how to initialize a population with specific
age and sex distributions from a CSV file.

This script addresses the feature request in:
https://github.com/starsimhub/starsim/issues/1045

The script demonstrates:
1. Loading age-sex cross-tabulated demographic data from a CSV file
2. Creating a population that matches the specified demographic structure
3. Validating the resulting population against the input data
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


def create_age_sex_distribution(demo_data):
    """
    Convert demographic data into a format suitable for population initialization.
    
    Args:
        demo_data: DataFrame with demographic data
        
    Returns:
        Dictionary with age and sex arrays for initialization
    """
    ages = []
    sexes = []
    
    for _, row in demo_data.iterrows():
        age_min = row['age_min']
        age_max = row['age_max']
        male_count = int(row['male_count'])
        female_count = int(row['female_count'])
        
        # Generate ages uniformly distributed within the age group
        male_ages = np.random.uniform(age_min, age_max, male_count)
        female_ages = np.random.uniform(age_min, age_max, female_count)
        
        # Append to lists
        ages.extend(male_ages)
        ages.extend(female_ages)
        sexes.extend(['m'] * male_count)
        sexes.extend(['f'] * female_count)
    
    return {'ages': np.array(ages), 'sexes': np.array(sexes)}


def initialize_population_from_demographics(demo_filepath, scale_factor=1.0):
    """
    Initialize a TB simulation population based on demographic data from a file.
    
    Args:
        demo_filepath: Path to demographic data CSV file
        scale_factor: Factor to scale the population (1.0 = use counts as-is)
        
    Returns:
        Initialized simulation object
    """
    # Load demographic data
    demo_data = load_demographic_data(demo_filepath)
    
    # Calculate total population
    total_pop = int((demo_data['male_count'].sum() + demo_data['female_count'].sum()) * scale_factor)
    
    # Scale the demographic data if needed
    if scale_factor != 1.0:
        demo_data['male_count'] = (demo_data['male_count'] * scale_factor).astype(int)
        demo_data['female_count'] = (demo_data['female_count'] * scale_factor).astype(int)
    
    # Create age and sex distribution
    dist_data = create_age_sex_distribution(demo_data)
    
    # Initialize simulation with custom age distribution
    pars = sc.objdict(
        n_agents=total_pop,
        dt=ss.days(7),
        start=ss.date('2000-01-01'),
        stop=ss.date('2010-01-01'),
    )
    
    sim = ss.Sim(diseases=mtb.TB(), pars=pars)
    
    # Initialize the simulation to create the people object
    sim.init()
    
    # Set ages and sexes from the demographic data
    # Note: This is a workaround since direct loading isn't supported yet
    if len(dist_data['ages']) == len(sim.people):
        sim.people.age[:] = dist_data['ages']
        sim.people.female[:] = (dist_data['sexes'] == 'f')
    else:
        print(f"Warning: Population size mismatch. Expected {len(dist_data['ages'])}, got {len(sim.people)}")
    
    return sim


def validate_population_demographics(sim, demo_data):
    """
    Validate that the generated population matches the input demographic data.
    
    Args:
        sim: Initialized simulation object
        demo_data: Original demographic data DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # Check total population
    total_expected = demo_data['male_count'].sum() + demo_data['female_count'].sum()
    total_actual = len(sim.people)
    validation['total_population'] = {
        'expected': total_expected,
        'actual': total_actual,
        'match': total_expected == total_actual
    }
    
    # Check sex distribution
    male_expected = demo_data['male_count'].sum()
    female_expected = demo_data['female_count'].sum()
    male_actual = (~sim.people.female).sum()
    female_actual = sim.people.female.sum()
    
    validation['sex_distribution'] = {
        'male': {'expected': male_expected, 'actual': male_actual},
        'female': {'expected': female_expected, 'actual': female_actual}
    }
    
    # Check age distribution by group
    age_validation = []
    for _, row in demo_data.iterrows():
        age_min, age_max = row['age_min'], row['age_max']
        mask = (sim.people.age >= age_min) & (sim.people.age < age_max)
        
        actual_male = (~sim.people.female & mask).sum()
        actual_female = (sim.people.female & mask).sum()
        
        age_validation.append({
            'age_range': f"{age_min}-{age_max}",
            'male': {'expected': row['male_count'], 'actual': actual_male},
            'female': {'expected': row['female_count'], 'actual': actual_female}
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
    Main function demonstrating the usage of demographic data loading.
    """
    print("=" * 70)
    print("TB Simulation: Population Initialization from Demographic Data")
    print("=" * 70)
    print()
    
    # Path to demographic data
    demo_filepath = 'data/sample_demographics.csv'
    
    print(f"Loading demographic data from: {demo_filepath}")
    demo_data = load_demographic_data(demo_filepath)
    
    print(f"\nDemographic Summary:")
    print(f"  Age groups: {len(demo_data)}")
    print(f"  Total males: {demo_data['male_count'].sum()}")
    print(f"  Total females: {demo_data['female_count'].sum()}")
    print(f"  Total population: {demo_data['male_count'].sum() + demo_data['female_count'].sum()}")
    print()
    
    # Initialize population (with optional scaling)
    print("Initializing population from demographic data...")
    scale_factor = 1.0  # Change this to scale population up or down
    sim = initialize_population_from_demographics(demo_filepath, scale_factor=scale_factor)
    
    print(f"Population initialized with {len(sim.people)} agents")
    print(f"  Males: {(~sim.people.female).sum()}")
    print(f"  Females: {sim.people.female.sum()}")
    print(f"  Age range: {sim.people.age.min():.1f} - {sim.people.age.max():.1f}")
    print()
    
    # Validate the population
    print("Validating population demographics...")
    validation = validate_population_demographics(sim, demo_data)
    
    print(f"\nValidation Results:")
    print(f"  Total population match: {validation['total_population']['match']}")
    print(f"    Expected: {validation['total_population']['expected']}")
    print(f"    Actual: {validation['total_population']['actual']}")
    print()
    
    print(f"  Sex distribution:")
    print(f"    Males - Expected: {validation['sex_distribution']['male']['expected']}, "
          f"Actual: {validation['sex_distribution']['male']['actual']}")
    print(f"    Females - Expected: {validation['sex_distribution']['female']['expected']}, "
          f"Actual: {validation['sex_distribution']['female']['actual']}")
    print()
    
    # Plot population pyramid
    print("Generating population pyramid...")
    plot_population_pyramid(sim, demo_data, filename='population_pyramid.png')
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)
    
    return sim, demo_data, validation


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the main demonstration
    sim, demo_data, validation = main()

