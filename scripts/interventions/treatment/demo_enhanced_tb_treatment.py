#!/usr/bin/env python3
"""
Demonstration script for Enhanced TB Treatment module

This script shows how to use the EnhancedTBTreatment class in a real simulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, '/Users/mine/anaconda_projects/newgit/newtbsim')

import starsim as ss
from tbsim.interventions.enhanced_tb_treatment import (
    EnhancedTBTreatment, 
    create_dots_treatment, 
    create_dots_improved_treatment, 
    create_first_line_treatment
)
from tbsim.interventions.tb_drug_types import TBDrugType, TBDrugTypeParameters

def demo_basic_usage():
    """Demonstrate basic usage of Enhanced TB Treatment."""
    print("=== Basic Usage Demo ===")
    
    # Create different types of treatments
    dots_treatment = create_dots_treatment()
    improved_dots = create_dots_improved_treatment()
    first_line = create_first_line_treatment()
    
    print(f"DOTS Treatment: {dots_treatment.pars.drug_type.name}")
    print(f"  Cure rate: {dots_treatment.drug_parameters.cure_rate:.2f}")
    print(f"  Duration: {dots_treatment.drug_parameters.duration:.0f} days")
    print(f"  Cost: ${dots_treatment.drug_parameters.cost_per_course:.0f}")
    
    print(f"\nImproved DOTS Treatment: {improved_dots.pars.drug_type.name}")
    print(f"  Cure rate: {improved_dots.drug_parameters.cure_rate:.2f}")
    print(f"  Duration: {improved_dots.drug_parameters.duration:.0f} days")
    print(f"  Cost: ${improved_dots.drug_parameters.cost_per_course:.0f}")
    
    print(f"\nFirst Line Treatment: {first_line.pars.drug_type.name}")
    print(f"  Cure rate: {first_line.drug_parameters.cure_rate:.2f}")
    print(f"  Duration: {first_line.drug_parameters.duration:.0f} days")
    print(f"  Cost: ${first_line.drug_parameters.cost_per_course:.0f}")

def demo_simulation():
    """Demonstrate using Enhanced TB Treatment in a simulation."""
    print("\n=== Simulation Demo ===")
    
    # Create a treatment intervention
    treatment = EnhancedTBTreatment(
        pars={
            'drug_type': TBDrugType.FIRST_LINE_COMBO,
            'treatment_success_rate': 0.95,
            'reseek_multiplier': 2.0,
            'reset_flags': True
        }
    )
    
    # Create a simulation
    sim = ss.Sim(
        n_agents=500,
        start=2020,
        stop=2022,
        diseases='sir',
        interventions=[treatment]
    )
    
    print(f"Created simulation with {sim.pars.n_agents} agents")
    print(f"Simulation period: {sim.pars.start} - {sim.pars.stop}")
    print(f"Interventions: {len(sim.pars.interventions)}")
    
    # Initialize the simulation
    sim.init()
    print(f"Simulation initialized with modules: {list(sim.module_dict.keys())}")
    
    # Access the treatment intervention
    tb_treatment = sim.module_dict['enhancedtbtreatment']
    print(f"Treatment drug type: {tb_treatment.pars.drug_type.name}")
    print(f"Treatment success rate: {tb_treatment.pars.treatment_success_rate}")
    
    return sim

def demo_drug_comparison():
    """Demonstrate comparing different drug types."""
    print("\n=== Drug Comparison Demo ===")
    
    # Get all drug parameter sets
    all_drugs = TBDrugTypeParameters.get_all_parameter_sets()
    
    print("Drug Type Comparison:")
    print("-" * 80)
    print(f"{'Drug Type':<20} {'Cure Rate':<12} {'Duration':<12} {'Cost':<12} {'Adherence':<12}")
    print("-" * 80)
    
    for drug_type, params in all_drugs.items():
        print(f"{drug_type.name:<20} {params.cure_rate:<12.2f} {params.duration:<12.0f} {params.cost_per_course:<12.0f} {params.adherence_rate:<12.2f}")
    
    # Find the most cost-effective treatment
    cost_effectiveness = []
    for drug_type, params in all_drugs.items():
        if params.cure_rate > 0:  # Avoid division by zero
            ce_ratio = params.cure_rate / (params.cost_per_course / 100)  # Cure rate per $100
            cost_effectiveness.append((drug_type.name, ce_ratio, params.cure_rate, params.cost_per_course))
    
    # Sort by cost-effectiveness
    cost_effectiveness.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost cost-effective treatments (cure rate per $100):")
    for name, ce_ratio, cure_rate, cost in cost_effectiveness[:3]:
        print(f"  {name}: {ce_ratio:.3f} (cure rate: {cure_rate:.2f}, cost: ${cost:.0f})")

def demo_custom_parameters():
    """Demonstrate creating custom drug parameters."""
    print("\n=== Custom Parameters Demo ===")
    
    # Create a custom treatment with modified parameters
    custom_treatment = EnhancedTBTreatment(
        pars={
            'drug_type': TBDrugType.DOTS,
            'treatment_success_rate': 0.90,  # Override default
            'reseek_multiplier': 3.0,        # Higher reseek rate
            'reset_flags': False             # Don't reset flags
        }
    )
    
    print("Custom DOTS Treatment Configuration:")
    print(f"  Drug type: {custom_treatment.pars.drug_type.name}")
    print(f"  Treatment success rate: {custom_treatment.pars.treatment_success_rate}")
    print(f"  Reseek multiplier: {custom_treatment.pars.reseek_multiplier}")
    print(f"  Reset flags: {custom_treatment.pars.reset_flags}")
    
    # Show drug parameters
    drug_params = custom_treatment.drug_parameters
    print(f"\nDrug Parameters:")
    print(f"  Cure rate: {drug_params.cure_rate:.2f}")
    print(f"  Duration: {drug_params.duration:.0f} days")
    print(f"  Adherence rate: {drug_params.adherence_rate:.2f}")
    print(f"  Cost per course: ${drug_params.cost_per_course:.0f}")

def plot_drug_comparison():
    """Create plots comparing different drug types."""
    print("\n=== Creating Drug Comparison Plots ===")
    
    # Get all drug parameter sets
    all_drugs = TBDrugTypeParameters.get_all_parameter_sets()
    
    # Prepare data for plotting
    drug_names = []
    cure_rates = []
    durations = []
    costs = []
    adherence_rates = []
    
    for drug_type, params in all_drugs.items():
        drug_names.append(drug_type.name.replace('_', ' '))
        cure_rates.append(params.cure_rate)
        durations.append(params.duration)
        costs.append(params.cost_per_course)
        adherence_rates.append(params.adherence_rate)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TB Drug Treatment Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Cure Rates
    bars1 = ax1.bar(drug_names, cure_rates, color='skyblue', alpha=0.7)
    ax1.set_title('Cure Rates by Drug Type', fontweight='bold')
    ax1.set_ylabel('Cure Rate')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, cure_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Treatment Duration
    bars2 = ax2.bar(drug_names, durations, color='lightcoral', alpha=0.7)
    ax2.set_title('Treatment Duration by Drug Type', fontweight='bold')
    ax2.set_ylabel('Duration (days)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, duration in zip(bars2, durations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{duration:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Cost per Course
    bars3 = ax3.bar(drug_names, costs, color='lightgreen', alpha=0.7)
    ax3.set_title('Cost per Treatment Course', fontweight='bold')
    ax3.set_ylabel('Cost (USD)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, cost in zip(bars3, costs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'${cost:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Adherence Rates
    bars4 = ax4.bar(drug_names, adherence_rates, color='gold', alpha=0.7)
    ax4.set_title('Adherence Rates by Drug Type', fontweight='bold')
    ax4.set_ylabel('Adherence Rate')
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars4, adherence_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/tb_drug_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved drug comparison plot as 'plots/tb_drug_comparison.png'")
    plt.show()

def plot_cost_effectiveness():
    """Create a cost-effectiveness analysis plot."""
    print("\n=== Creating Cost-Effectiveness Analysis ===")
    
    # Get all drug parameter sets
    all_drugs = TBDrugTypeParameters.get_all_parameter_sets()
    
    # Calculate cost-effectiveness metrics
    drug_data = []
    for drug_type, params in all_drugs.items():
        if params.cure_rate > 0:  # Avoid division by zero
            ce_ratio = params.cure_rate / (params.cost_per_course / 100)  # Cure rate per $100
            drug_data.append({
                'name': drug_type.name.replace('_', ' '),
                'cure_rate': params.cure_rate,
                'cost': params.cost_per_course,
                'ce_ratio': ce_ratio,
                'duration': params.duration
            })
    
    # Sort by cost-effectiveness
    drug_data.sort(key=lambda x: x['ce_ratio'], reverse=True)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('TB Treatment Cost-Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Cost-Effectiveness Scatter
    names = [d['name'] for d in drug_data]
    ce_ratios = [d['ce_ratio'] for d in drug_data]
    costs = [d['cost'] for d in drug_data]
    cure_rates = [d['cure_rate'] for d in drug_data]
    
    scatter = ax1.scatter(costs, cure_rates, s=[r*100 for r in ce_ratios], 
                         c=ce_ratios, cmap='viridis', alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Cost per Course (USD)')
    ax1.set_ylabel('Cure Rate')
    ax1.set_title('Cost vs Cure Rate (bubble size = cost-effectiveness)')
    ax1.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, name in enumerate(names):
        ax1.annotate(name, (costs[i], cure_rates[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Cost-Effectiveness (cure rate per $100)')
    
    # Plot 2: Cost-Effectiveness Ranking
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, ce_ratios, color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Cost-Effectiveness (cure rate per $100)')
    ax2.set_title('Cost-Effectiveness Ranking')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, ce_ratios)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{ratio:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/tb_cost_effectiveness.png', dpi=300, bbox_inches='tight')
    print("✓ Saved cost-effectiveness analysis as 'plots/tb_cost_effectiveness.png'")
    plt.show()

def plot_treatment_timeline():
    """Create a timeline plot showing treatment progression."""
    print("\n=== Creating Treatment Timeline Plot ===")
    
    # Simulate treatment progression over time
    time_points = np.linspace(0, 180, 100)  # 6 months
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('TB Treatment Timeline Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Drug Effectiveness Over Time
    drug_types = ['DOTS', 'DOTS_IMPROVED', 'FIRST_LINE_COMBO']
    colors = ['blue', 'green', 'red']
    
    for drug_name, color in zip(drug_types, colors):
        # Get drug parameters
        drug_type = getattr(TBDrugType, drug_name)
        params = TBDrugTypeParameters.create_parameters_for_type(drug_type)
        
        # Calculate effectiveness over time with drug-specific models
        effectiveness = []
        for t in time_points:
            if drug_name == 'DOTS':
                # Standard DOTS: slower build-up, moderate decay
                if t <= 45:  # Slower build-up
                    eff = min(0.8, t / 45.0)
                else:
                    decay = np.exp(-(t - 45) / 90)  # Slower decay
                    eff = max(0.2, 0.8 * decay)
            elif drug_name == 'DOTS_IMPROVED':
                # Improved DOTS: faster build-up, slower decay
                if t <= 25:  # Faster build-up
                    eff = min(0.9, t / 25.0)
                else:
                    decay = np.exp(-(t - 25) / 120)  # Slower decay
                    eff = max(0.3, 0.9 * decay)
            else:  # FIRST_LINE_COMBO
                # First-line combo: very fast build-up, minimal decay
                if t <= 20:  # Very fast build-up
                    eff = min(0.95, t / 20.0)
                else:
                    decay = np.exp(-(t - 20) / 150)  # Very slow decay
                    eff = max(0.4, 0.95 * decay)
            
            effectiveness.append(eff)
        
        ax1.plot(time_points, effectiveness, label=f'{drug_name} (cure: {params.cure_rate:.2f})', 
                color=color, linewidth=2)
    
    ax1.set_xlabel('Days on Treatment')
    ax1.set_ylabel('Drug Effectiveness')
    ax1.set_title('Drug Effectiveness Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Cumulative Treatment Success
    success_rates = [0.85, 0.90, 0.95]  # DOTS, DOTS_IMPROVED, FIRST_LINE_COMBO
    
    for drug_name, success_rate, color in zip(drug_types, success_rates, colors):
        # Simulate cumulative success with different time constants for each drug
        if drug_name == 'DOTS':
            time_constant = 80  # Slower approach to final success rate
        elif drug_name == 'DOTS_IMPROVED':
            time_constant = 60  # Moderate approach
        else:  # FIRST_LINE_COMBO
            time_constant = 40  # Faster approach
        
        cumulative_success = success_rate * (1 - np.exp(-time_points / time_constant))
        ax2.plot(time_points, cumulative_success, label=f'{drug_name} ({success_rate:.0%} final)', 
                color=color, linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Days on Treatment')
    ax2.set_ylabel('Cumulative Success Rate')
    ax2.set_title('Cumulative Treatment Success Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('plots/tb_treatment_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Saved treatment timeline plot as 'plots/tb_treatment_timeline.png'")
    plt.show()

def plot_simulation_results():
    """Create plots showing simulation results."""
    print("\n=== Creating Simulation Results Plots ===")
    
    # Run a simple simulation with different treatments
    treatments = {
        'DOTS': create_dots_treatment(),
        'DOTS_IMPROVED': create_dots_improved_treatment(),
        'FIRST_LINE_COMBO': create_first_line_treatment()
    }
    
    # Simulate results over time (simplified)
    time_points = np.arange(0, 365, 30)  # Monthly for a year
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('TB Treatment Simulation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Treatment Success Over Time
    for treatment_name, treatment in treatments.items():
        # Simulate treatment success (simplified model)
        base_success = treatment.drug_parameters.cure_rate
        # Add some noise and trend
        success_over_time = []
        for t in time_points:
            # Simulate seasonal variation and improvement over time
            seasonal = 0.1 * np.sin(2 * np.pi * t / 365)
            improvement = 0.05 * (t / 365)
            noise = np.random.normal(0, 0.02)
            success = min(0.99, base_success + seasonal + improvement + noise)
            success_over_time.append(success)
        
        ax1.plot(time_points, success_over_time, label=treatment_name, linewidth=2, marker='o')
    
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Treatment Success Rate')
    ax1.set_title('Treatment Success Rate Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Cost-Benefit Analysis
    treatment_costs = []
    treatment_benefits = []
    treatment_names = []
    
    for treatment_name, treatment in treatments.items():
        cost = treatment.drug_parameters.cost_per_course
        benefit = treatment.drug_parameters.cure_rate * 1000  # Lives saved per 1000 treated
        treatment_costs.append(cost)
        treatment_benefits.append(benefit)
        treatment_names.append(treatment_name)
    
    bars = ax2.bar(treatment_names, treatment_benefits, 
                  color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
    ax2.set_ylabel('Lives Saved per 1000 Treated')
    ax2.set_title('Treatment Benefits (Lives Saved)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add cost annotations
    for bar, cost in zip(bars, treatment_costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'${cost:.0f}/course', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/tb_simulation_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved simulation results plot as 'plots/tb_simulation_results.png'")
    plt.show()

def main():
    """Run all demonstrations."""
    print("Enhanced TB Treatment Module Demonstration")
    print("=" * 50)
    
    try:
        # Basic demonstrations
        demo_basic_usage()
        sim = demo_simulation()
        demo_drug_comparison()
        demo_custom_parameters()
        
        # Plotting demonstrations
        print("\n" + "="*60)
        print("CREATING VISUALIZATION PLOTS")
        print("="*60)
        
        plot_drug_comparison()
        plot_cost_effectiveness()
        plot_treatment_timeline()
        plot_simulation_results()
        
        print(f"\n🎉 All demonstrations completed successfully!")
        print(f"The Enhanced TB Treatment module is ready for use in TBsim simulations.")
        print(f"\n📊 Generated plots:")
        print(f"  - plots/tb_drug_comparison.png: Comprehensive drug comparison")
        print(f"  - plots/tb_cost_effectiveness.png: Cost-effectiveness analysis")
        print(f"  - plots/tb_treatment_timeline.png: Treatment timeline analysis")
        print(f"  - plots/tb_simulation_results.png: Simulation results")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
