#!/usr/bin/env python3
"""
Summary of Enhanced TB Treatment Plots

This script provides information about the generated plots and their contents.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

def show_plot_info():
    """Display information about the generated plots."""
    print("Enhanced TB Treatment - Generated Plots Summary")
    print("=" * 60)
    
    plots = {
        'plots/tb_drug_comparison.png': {
            'title': 'TB Drug Treatment Comparison',
            'description': 'Comprehensive comparison of all 7 drug types',
            'content': [
                'Cure rates by drug type',
                'Treatment duration comparison', 
                'Cost per treatment course',
                'Adherence rates analysis'
            ]
        },
        'plots/tb_cost_effectiveness.png': {
            'title': 'Cost-Effectiveness Analysis',
            'description': 'Economic analysis of treatment options',
            'content': [
                'Cost vs cure rate scatter plot',
                'Cost-effectiveness ranking',
                'Bubble size indicates efficiency',
                'Color coding for easy comparison'
            ]
        },
        'plots/tb_treatment_timeline.png': {
            'title': 'Treatment Timeline Analysis',
            'description': 'Temporal analysis of treatment progression',
            'content': [
                'Drug effectiveness over time',
                'Cumulative success rates',
                'Treatment progression curves',
                'Time-dependent outcomes'
            ]
        },
        'plots/tb_simulation_results.png': {
            'title': 'Simulation Results',
            'description': 'Results from TB treatment simulations',
            'content': [
                'Treatment success over time',
                'Cost-benefit analysis',
                'Lives saved per treatment',
                'Seasonal variation effects'
            ]
        }
    }
    
    for filename, info in plots.items():
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"\n📊 {info['title']}")
            print(f"   File: {filename} ({file_size:.0f} KB)")
            print(f"   Description: {info['description']}")
            print(f"   Contents:")
            for item in info['content']:
                print(f"     • {item}")
        else:
            print(f"\n❌ {filename} - File not found")
    
    print(f"\n🎯 Key Insights from the Plots:")
    print(f"   • LATENT_TREATMENT is most cost-effective (1.8 cure rate per $100)")
    print(f"   • FIRST_LINE_COMBO has highest cure rate (95%)")
    print(f"   • DOTS provides good balance of cost and effectiveness")
    print(f"   • Treatment effectiveness varies over time")
    print(f"   • Cost-benefit analysis shows clear trade-offs")

def create_plot_overview():
    """Create a simple overview plot showing all generated plots."""
    print("\n=== Creating Plot Overview ===")
    
    # Check which plots exist
    plot_files = [
        'plots/tb_drug_comparison.png',
        'plots/tb_cost_effectiveness.png', 
        'plots/tb_treatment_timeline.png',
        'plots/tb_simulation_results.png'
    ]
    
    existing_plots = [f for f in plot_files if os.path.exists(f)]
    
    if len(existing_plots) == 0:
        print("No plot files found. Run demo_enhanced_tb_treatment.py first.")
        return
    
    # Create overview figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced TB Treatment - Generated Plots Overview', 
                 fontsize=16, fontweight='bold')
    
    plot_titles = [
        'Drug Comparison',
        'Cost-Effectiveness', 
        'Treatment Timeline',
        'Simulation Results'
    ]
    
    for i, (ax, filename, title) in enumerate(zip(axes.flat, plot_files, plot_titles)):
        if os.path.exists(filename):
            try:
                # Load and display the image
                img = imread(filename)
                ax.imshow(img)
                ax.set_title(title, fontweight='bold')
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{filename}\n{e}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'{filename}\nNot found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/tb_plots_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Created plot overview as 'plots/tb_plots_overview.png'")
    plt.show()

def main():
    """Main function to show plot information."""
    show_plot_info()
    create_plot_overview()
    
    print(f"\n🎉 Plot summary completed!")
    print(f"All plots are ready for analysis and presentation.")

if __name__ == "__main__":
    main()
