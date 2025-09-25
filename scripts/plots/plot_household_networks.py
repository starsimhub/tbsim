#!/usr/bin/env python3
"""
Household Network Plotting Script

This script provides comprehensive visualization tools for household networks in TBSim.
It includes basic network plots, advanced statistics, and simulation examples.

Usage:
    python scripts/plot_household_networks.py [--example basic|advanced|simulation|all]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add the parent directory to the path to import tbsim
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tbsim as mtb
import starsim as ss
from tbsim.networks import HouseholdNet, plot_household_structure


def plot_household_network_basic(households, title="Household Network", save_path=None):
    """
    Create a  visualization of household networks using NetworkX with dark theme.
    
    Args:
        households: List of lists, where each inner list contains agent UIDs in a household
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    # Set dark theme
    plt.style.use('dark_background')
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes (all agents)
    all_agents = [agent for hh in households for agent in hh]
    G.add_nodes_from(all_agents)
    
    # Create  color palette
    household_colors = plt.cm.viridis(np.linspace(0, 1, len(households)))
    
    # Create figure with dark background
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    # Add edges for each household (complete graph)
    for hh_idx, household in enumerate(households):
        if len(household) > 1:
            # Create complete graph for this household
            for i in range(len(household)):
                for j in range(i + 1, len(household)):
                    G.add_edge(household[i], household[j], 
                             household=hh_idx, color=household_colors[hh_idx])
    
    # Create layout with better spacing
    pos = nx.spring_layout(G, seed=42, k=3, iterations=100)
    
    # Draw edges first (behind nodes)
    for hh_idx, household in enumerate(households):
        household_edges = [(u, v) for u, v in G.edges() 
                          if u in household and v in household]
        nx.draw_networkx_edges(G, pos, 
                             edgelist=household_edges,
                             edge_color=household_colors[hh_idx],
                             width=3,
                             alpha=0.6,
                             style='solid',
                             ax=ax)
    
    for hh_idx, household in enumerate(households):
        node_positions = {node: pos[node] for node in household if node in pos}
        
        node_sizes = [400 + 100 * len(household)] * len(household)
        
        nx.draw_networkx_nodes(G, node_positions, 
                             nodelist=household,
                             node_color=[household_colors[hh_idx]] * len(household),
                             node_size=node_sizes,
                             alpha=0.9,
                             edgecolors='white',
                             linewidths=1,
                             ax=ax)
    
    # Draw labels with enhanced styling
    nx.draw_networkx_labels(G, pos, 
                           font_size=14, 
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    legend_elements = []
    for hh_idx, household in enumerate(households):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=household_colors[hh_idx],
                                        markersize=15, alpha=0.9, markeredgecolor='white',
                                        markeredgewidth=1,
                                        label=f'Household {hh_idx + 1} (n={len(household)})'))
    
    legend = ax.legend(handles=legend_elements, loc='upper right', 
                      bbox_to_anchor=(1.15, 1), 
                      frameon=True, 
                      fancybox=True, 
                      shadow=True,
                      facecolor='#2a2a2a',
                      edgecolor='#404040',
                      fontsize=10)
    
    ax.set_title(title, fontsize=20, fontweight='bold', color='white', pad=20)
    
    # Add subtle grid for reference
    ax.grid(True, alpha=0.1, color='white', linestyle='-', linewidth=0.5)
    
    # Remove axes
    ax.axis('off')
    
    # Add text box with network statistics
    total_agents = sum(len(hh) for hh in households)
    total_households = len(households)
    mean_size = np.mean([len(hh) for hh in households])
    
    stats_text = f"Total Agents: {total_agents}\n"
    stats_text += f"Total Households: {total_households}\n"
    stats_text += f"Mean Size: {mean_size:.1f}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#404040', alpha=0.8, edgecolor='#606060'),
            fontsize=10, color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return G


def plot_household_network_advanced(households, title="Advanced Household Network", save_path=None):
    """
    Create a  advanced visualization with household clustering and enhanced statistics.
    
    Args:
        households: List of lists, where each inner list contains agent UIDs in a household
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    # Set dark theme
    plt.style.use('dark_background')
    
    # Create figure with dark background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('#1a1a1a')
    ax1.set_facecolor('#1a1a1a')
    ax2.set_facecolor('#1a1a1a')
    
    # Left plot: Enhanced Network visualization
    G = nx.Graph()
    all_agents = [agent for hh in households for agent in hh]
    G.add_nodes_from(all_agents)
    
    # Add household membership as node attribute
    for hh_idx, household in enumerate(households):
        for agent in household:
            G.nodes[agent]['household'] = hh_idx
            G.nodes[agent]['household_size'] = len(household)
    
    # Add edges within households
    for hh_idx, household in enumerate(households):
        if len(household) > 1:
            for i in range(len(household)):
                for j in range(i + 1, len(household)):
                    G.add_edge(household[i], household[j], household=hh_idx)
    
    # Create  color palette
    household_colors = plt.cm.plasma(np.linspace(0, 1, len(households)))
    
    # Create enhanced layout with household clustering
    pos = {}
    
    # Position households in a more  pattern
    household_angles = np.linspace(0, 2*np.pi, len(households), endpoint=False)
    
    for hh_idx, household in enumerate(households):
        # Base position for this household with varying radius
        radius = 4 + 0.5 * len(household)  # Larger households get more space
        hh_x = radius * np.cos(household_angles[hh_idx])
        hh_y = radius * np.sin(household_angles[hh_idx])
        
        if len(household) == 1:
            pos[household[0]] = (hh_x, hh_y)
        else:
            # Arrange household members in a more  pattern
            if len(household) == 2:
                # Line arrangement for couples
                pos[household[0]] = (hh_x - 0.3, hh_y)
                pos[household[1]] = (hh_x + 0.3, hh_y)
            else:
                # Circular arrangement for larger households
                agent_angles = np.linspace(0, 2*np.pi, len(household), endpoint=False)
                inner_radius = 0.8 + 0.2 * len(household)
                
                for agent_idx, agent in enumerate(household):
                    pos[agent] = (hh_x + inner_radius * np.cos(agent_angles[agent_idx]),
                                 hh_y + inner_radius * np.sin(agent_angles[agent_idx]))
    
    # Draw edges first with enhanced styling
    for hh_idx, household in enumerate(households):
        household_edges = [(u, v) for u, v in G.edges() 
                          if u in household and v in household]
        nx.draw_networkx_edges(G, pos,
                             edgelist=household_edges,
                             edge_color=household_colors[hh_idx],
                             width=4,
                             alpha=0.7,
                             style='solid',
                             ax=ax1)
    
    # Draw nodes with enhanced styling
    for hh_idx, household in enumerate(households):
        node_positions = {node: pos[node] for node in household}
        # Dynamic node sizing based on household size
        base_size = 600
        size_multiplier = 1 + 0.3 * len(household)
        node_sizes = [base_size * size_multiplier for _ in household]
        
        nx.draw_networkx_nodes(G, node_positions,
                             nodelist=household,
                             node_color=[household_colors[hh_idx]] * len(household),
                             node_size=node_sizes,
                             alpha=0.9,
                             edgecolors='white',
                             linewidths=1,
                             ax=ax1)
    
    # Draw labels with enhanced styling
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', 
                           font_color='white', ax=ax1)
    
    # Add household labels
    for hh_idx, household in enumerate(households):
        if len(household) > 0:
            # Calculate household center
            hh_center_x = np.mean([pos[agent][0] for agent in household])
            hh_center_y = np.mean([pos[agent][1] for agent in household])
            
            # Add household label
            ax1.text(hh_center_x, hh_center_y + 1.5, f'HH{hh_idx+1}', 
                    fontsize=12, fontweight='bold', color=household_colors[hh_idx],
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', alpha=0.8))
    
    ax1.set_title('Enhanced Household Network Structure', fontsize=18, fontweight='bold', color='white', pad=20)
    ax1.axis('off')
    ax1.grid(True, alpha=0.1, color='white', linestyle='-', linewidth=0.5)
    
    # Right plot: Enhanced statistics with dark theme
    household_sizes = [len(hh) for hh in households]
    size_counts = {size: household_sizes.count(size) for size in set(household_sizes)}
    
    # Create gradient colors for bars
    bar_colors = plt.cm.viridis(np.linspace(0, 1, len(size_counts)))
    
    bars = ax2.bar(size_counts.keys(), size_counts.values(), 
                   color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels on bars with enhanced styling
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', 
                color='white', fontsize=12)
    
    ax2.set_xlabel('Household Size', fontsize=12, color='white')
    ax2.set_ylabel('Number of Households', fontsize=12, color='white')
    ax2.set_title('Household Size Distribution', fontsize=18, fontweight='bold', color='white', pad=20)
    
    # Enhanced grid styling
    ax2.grid(True, alpha=0.2, color='white', linestyle='-', linewidth=0.5)
    ax2.tick_params(colors='white', labelsize=12)
    
    # Enhanced summary statistics box
    total_agents = sum(household_sizes)
    mean_size = np.mean(household_sizes)
    max_size = max(household_sizes)
    min_size = min(household_sizes)
    std_size = np.std(household_sizes)
    
    stats_text = f"Total Agents: {total_agents}\n"
    stats_text += f"Total Households: {len(households)}\n"
    stats_text += f"Mean Size: {mean_size:.1f}\n"
    stats_text += f"Std Dev: {std_size:.1f}\n"
    stats_text += f"Size Range: {min_size}-{max_size}"
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#404040', alpha=0.9, 
                      edgecolor='#606060', linewidth=1),
             fontsize=10, color='white', fontweight='bold')
    
    # Add connection lines between households in the network
    for i in range(len(households)):
        for j in range(i+1, len(households)):
            # Calculate centers
            hh1_center = np.mean([pos[agent] for agent in households[i]], axis=0)
            hh2_center = np.mean([pos[agent] for agent in households[j]], axis=0)
            
            # Draw subtle connection lines
            ax1.plot([hh1_center[0], hh2_center[0]], [hh1_center[1], hh2_center[1]], 
                    '--', color='white', alpha=0.1, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return G


def demonstrate_household_network_simulation():
    """
    Demonstrate how to extract and plot household networks from a TBSim simulation.
    """
    print("Creating household network simulation...")
    
    # Define households
    households = [
        [0, 1, 2],      # Family of 3
        [3, 4],         # Couple
        [5, 6, 7, 8],   # Family of 4
        [9],            # Single person
        [10, 11, 12]    # Family of 3
    ]
    
    # Create network
    household_net = HouseholdNet(hhs=households)
    
    # Create simulation
    sim = ss.Sim(
        people=ss.People(n_agents=13),
        networks=[household_net],
        diseases=mtb.TB(),
        pars=dict(start=ss.date(2000), stop=ss.date(2001), dt=ss.months(1))
    )
    
    # Run simulation
    sim.run()
    
    print(f"Simulation completed with {len(households)} households")
    print(f"Total agents: {sum(len(hh) for hh in households)}")
    print(f"Network edges: {len(household_net.edges.p1) if hasattr(household_net.edges, 'p1') else 'None'}")
    
    # Plot the networks
    plot_household_network_basic(households, "TBSim Household Network - Basic View")
    plot_household_network_advanced(households, "TBSim Household Network - Advanced View")
    
    return sim, household_net


def create_sample_households():
    """Create  sample household structures for demonstration."""
    return {
        'simple': [[0, 1, 2], [3, 4], [5, 6, 7, 8]],
        'complex': [
            [0, 1, 2, 3],     # Large family
            [4, 5],           # Couple
            [6],              # Single
            [7, 8, 9],        # Family of 3
            [10, 11],         # Another couple
            [12, 13, 14, 15, 16]  # Very large family
        ],
        'realistic': [
            [0, 1, 2, 3, 4],  # Extended family
            [5, 6],           # Young couple
            [7],              # Elderly single
            [8, 9, 10],       # Nuclear family
            [11, 12, 13, 14], # Large family
            [15, 16],         # Couple
            [17, 18, 19],     # Family with children
            [20]              # Single person
        ],
        '': [
            [0, 1, 2, 3, 4, 5],    # Extended family with grandparents
            [6, 7],                # Young couple
            [8],                   # Elderly single
            [9, 10, 11],           # Nuclear family
            [12, 13, 14, 15],      # Large family
            [16, 17],              # Couple
            [18, 19, 20],          # Family with children
            [21],                  # Single person
            [22, 23, 24, 25, 26, 27, 28],  # Very large extended family
            [29, 30, 31],          # Another nuclear family
            [32, 33, 34, 35],      # Family of 4
            [36, 37],              # Another couple
            [38, 39, 40, 41, 42]   # Large family
        ],
        'community': [
            [0, 1, 2, 3, 4, 5, 6],     # Large extended family
            [7, 8],                     # Couple
            [9],                        # Single
            [10, 11, 12],              # Nuclear family
            [13, 14, 15, 16],          # Large family
            [17, 18],                  # Couple
            [19, 20, 21],              # Family with children
            [22],                      # Single person
            [23, 24, 25, 26, 27, 28, 29, 30],  # Very large family
            [31, 32, 33],              # Nuclear family
            [34, 35, 36, 37],          # Family of 4
            [38, 39],                  # Couple
            [40, 41, 42, 43, 44],      # Large family
            [45, 46, 47],              # Another nuclear family
            [48, 49, 50, 51, 52, 53],  # Extended family
            [54, 55],                  # Young couple
            [56, 57, 58, 59],          # Family with teenagers
            [60]                       # Elderly single
        ]
    }


def main():
    """Main function to run the household network plotting script."""
    parser = argparse.ArgumentParser(description='Plot household networks from TBSim')
    parser.add_argument('--example', choices=['basic', 'advanced', 'simulation', '', 'community', 'all'], 
                       default='all', help='Type of example to run')
    parser.add_argument('--save', action='store_true', 
                       help='Save plots to files')
    parser.add_argument('--output-dir', default='scripts/results', 
                       help='Output directory for saved plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    print("Household Network Plotting Script")
    print("=" * 50)
    print("🎨 Enhanced Dark Theme Visualizations")
    print("=" * 50)
    
    # Get sample households
    samples = create_sample_households()
    
    if args.example in ['basic', 'all']:
        print(f"\n🎯 Example 1:  Basic Network Visualization")
        households = samples['simple']
        save_path = os.path.join(args.output_dir, 'household_network_basic_dark.png') if args.save else None
        plot_household_network_basic(households, " Household Network", save_path)
    
    if args.example in ['advanced', 'all']:
        print(f"\n🚀 Example 2: Advanced Network Visualization with Enhanced Statistics")
        households = samples['complex']
        save_path = os.path.join(args.output_dir, 'household_network_advanced_dark.png') if args.save else None
        plot_household_network_advanced(households, "Advanced Household Network Analysis", save_path)
    
    if args.example in ['', 'all']:
        print(f"\n🌟 Example 3:  Community Network")
        households = samples['']
        save_path = os.path.join(args.output_dir, 'household_network__dark.png') if args.save else None
        plot_household_network_advanced(households, " Community Network", save_path)
    
    if args.example in ['community', 'all']:
        print(f"\n🏘️ Example 4: Large Community Network Analysis")
        households = samples['community']
        save_path = os.path.join(args.output_dir, 'household_network_community_dark.png') if args.save else None
        plot_household_network_advanced(households, "Large Community Network Analysis", save_path)
    
    if args.example in ['simulation', 'all']:
        print(f"\n⚡ Example 5: Full TBSim Simulation with Household Networks")
        sim, net = demonstrate_household_network_simulation()
    
    # Test the built-in plotting function with dark theme
    print(f"\n🔧 Example 6: Built-in TBSim Plotting Function (Dark Theme)")
    households = samples['realistic']
    save_path = os.path.join(args.output_dir, 'household_network_builtin_dark.png') if args.save else None
    
    # Apply dark theme to built-in function
    plt.style.use('dark_background')
    G = plot_household_structure(households, 'TBSim Built-in Household Plot (Dark Theme)')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Built-in plot saved to: {save_path}")
    
    print("\n" + "=" * 50)
    print(" Demonstration Complete!")
    print("=" * 50)
    print(f" Available Enhanced Plotting Functions:")
    print("   plot_household_network_basic():  network visualization with dark theme")
    print("   plot_household_network_advanced(): Advanced visualization with enhanced statistics")
    print("   plot_household_structure(): Built-in TBSim function (dark theme applied)")
    print("   demonstrate_household_network_simulation(): Full simulation example")
    print("\n Features:")
    print("  • Dark theme with  color palettes")
    print("  • Enhanced node and edge styling")
    print("  • Dynamic sizing based on household characteristics")
    print("  • Advanced statistics and visual elements")
    print("  • High-resolution output with proper dark backgrounds")


if __name__ == "__main__":
    main()
