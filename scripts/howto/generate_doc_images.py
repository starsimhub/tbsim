#!/usr/bin/env python3
"""
Generate example images for DwtAnalyzer docstrings.

This script creates sample visualizations that will be used in the
documentation to illustrate what each plotting method produces.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add the project root to the path so we can import tbsim
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_sankey_example():
    """Create a Sankey diagram example for documentation."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a simplified Sankey-like representation
    states = ['Susceptible', 'Latent Slow', 'Latent Fast', 'Active', 'Treatment', 'Recovery']
    flows = [1000, 200, 150, 80, 60, 40]
    
    # Create horizontal bars representing flow
    y_pos = np.linspace(0, 1, len(states))
    colors = ['lightblue', 'orange', 'yellow', 'red', 'green', 'purple']
    
    for i, (state, flow, color) in enumerate(zip(states, flows, colors)):
        ax.barh(y_pos[i], flow, height=0.08, alpha=0.7, color=color, edgecolor='black')
        ax.text(flow + 20, y_pos[i], state, va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, max(flows) * 1.2)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Number of Agents')
    ax.set_title('TB State Transitions (Sankey Diagram Example)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])  # Hide y-axis ticks
    
    # Add flow arrows
    for i in range(len(states) - 1):
        ax.annotate('', xy=(flows[i+1], y_pos[i+1]), xytext=(flows[i], y_pos[i]),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_histogram_kde_example():
    """Create a histogram with KDE example for documentation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    states = ['Latent Slow', 'Latent Fast', 'Active', 'Treatment']
    
    for i, (ax, state) in enumerate(zip(axes.flat, states)):
        # Generate realistic dwell time data for each state
        if state == 'Latent Slow':
            data = np.random.exponential(365, 1000)  # Long dwell times
        elif state == 'Latent Fast':
            data = np.random.exponential(180, 1000)  # Medium dwell times
        elif state == 'Active':
            data = np.random.exponential(100, 1000)  # Shorter dwell times
        else:  # Treatment
            data = np.random.exponential(240, 1000)  # Treatment duration
            
        # Create histogram
        ax.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        ax.set_title(f'{state} State Dwell Times', fontweight='bold')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Dwell Time Distributions with Kernel Density Estimation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def create_network_graph_example():
    """Create a network graph example for documentation."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define nodes and their positions
    nodes = ['Susceptible', 'Latent\nSlow', 'Latent\nFast', 'Active\nPresymp', 'Active\nSMPos', 'Treatment', 'Recovery']
    node_pos = [(0, 0.5), (0.2, 0.8), (0.2, 0.2), (0.4, 0.6), (0.4, 0.4), (0.6, 0.5), (0.8, 0.5)]
    
    # Define edges with weights (for edge thickness)
    edges = [
        (0, 1, 50), (0, 2, 30),  # Susceptible -> Latent states
        (1, 3, 20), (2, 3, 15),  # Latent -> Active Presymp
        (3, 4, 25),              # Active Presymp -> Active SMPos
        (4, 5, 20),              # Active SMPos -> Treatment
        (5, 6, 15),              # Treatment -> Recovery
        (1, 0, 5), (2, 0, 3),    # Some recovery back to susceptible
    ]
    
    # Draw nodes
    for i, (node, pos) in enumerate(zip(nodes, node_pos)):
        ax.scatter(pos[0], pos[1], s=300, c=f'C{i}', alpha=0.8, edgecolors='black', linewidth=2)
        ax.annotate(node, (pos[0], pos[1]), xytext=(0, 0), textcoords='offset points',
                   ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw edges with thickness proportional to weight
    max_weight = max(weight for _, _, weight in edges)
    for start, end, weight in edges:
        x1, y1 = node_pos[start]
        x2, y2 = node_pos[end]
        thickness = 1 + (weight / max_weight) * 4
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=thickness, alpha=0.7, color='gray'))
        
        # Add edge label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'{weight}', ha='center', va='center', 
               fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-0.1, 0.9)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('TB State Transition Network\n(Edge thickness ∝ transition frequency)', 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    return fig

def create_reinfection_analysis_example():
    """Create a reinfection analysis example for documentation."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Example reinfection data
    reinfection_counts = [0, 1, 2, 3, 4, 5]
    percentages = [65, 20, 10, 3, 1.5, 0.5]  # Example percentages
    
    # Create bars with different colors
    colors = ['lightgreen', 'lightblue', 'orange', 'red', 'purple', 'brown']
    bars = ax.bar(reinfection_counts, percentages, color=colors, alpha=0.8, 
                 edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Number of Reinfections', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Population (%)', fontsize=12, fontweight='bold')
    ax.set_title('Reinfection Distribution Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{pct}%', ha='center', va='bottom', fontweight='bold')
    
    # Add some statistics
    total_infected = sum(percentages[1:])  # Exclude 0 reinfections
    ax.text(0.02, 0.98, f'Total infected: {total_infected}%\nNever infected: {percentages[0]}%', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_interactive_bar_example():
    """Create an interactive bar chart example for documentation."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Example state transition data with dwell time categories
    transitions = [
        'Latent→Active (0-30d)', 'Latent→Active (30-90d)', 'Latent→Active (90-180d)',
        'Active→Treatment (0-30d)', 'Active→Treatment (30-90d)', 'Active→Treatment (90-180d)',
        'Treatment→Recovery (0-90d)', 'Treatment→Recovery (90-180d)', 'Treatment→Recovery (180-365d)'
    ]
    counts = [45, 30, 15, 60, 25, 10, 40, 20, 8]
    
    # Create horizontal bars
    y_pos = np.arange(len(transitions))
    colors = plt.cm.Set3(np.linspace(0, 1, len(transitions)))
    
    bars = ax.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(transitions, fontsize=10)
    ax.set_xlabel('Number of Agents', fontsize=12, fontweight='bold')
    ax.set_title('State Transitions by Dwell Time Categories\n(Interactive Bar Chart Example)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, str(count),
               ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all example images for documentation."""
    # Ensure the _static directory exists
    static_dir = Path(__file__).parent.parent / 'docs' / '_static'
    static_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating documentation images in: {static_dir}")
    
    # Generate all examples
    examples = [
        ('sankey_diagram_example.png', create_sankey_example),
        ('histogram_kde_example.png', create_histogram_kde_example),
        ('network_graph_example.png', create_network_graph_example),
        ('reinfection_analysis_example.png', create_reinfection_analysis_example),
        ('interactive_bar_example.png', create_interactive_bar_example),
    ]
    
    for filename, create_func in examples:
        print(f"Creating {filename}...")
        fig = create_func()
        output_path = static_dir / filename
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  ✓ Saved to {output_path}")
    
    print("\nAll documentation images generated successfully!")
    print("You can now use these images in your docstrings with:")
    print(".. image:: _static/filename.png")
    print("    :width: 600px")
    print("    :alt: Description of the image")

if __name__ == "__main__":
    main() 