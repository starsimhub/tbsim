"""
Enhanced DwtAnalyzer docstrings with embedded plots and visualizations.

This demonstrates how to add meaningful visual examples to docstrings
for the TB simulation analyzer class.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
import io

class EnhancedDwtAnalyzer:
    """
    Enhanced DwtAnalyzer with visual docstrings.
    
    This class demonstrates how to include plots and visualizations
    directly in docstrings for better documentation.
    """
    
    def sankey_agents(self, subtitle=""):
        """
        Creates a Sankey diagram showing agent flow between TB states.
        
        This method visualizes the movement of agents between different
        tuberculosis states throughout the simulation.
        
        .. image:: _static/sankey_diagram_example.png
            :width: 600px
            :alt: Sankey diagram showing TB state transitions
            :align: center
        
        The Sankey diagram above shows:
        - **Nodes**: TB states (Susceptible, Latent, Active, Treatment, etc.)
        - **Flows**: Agent transitions between states
        - **Width**: Proportional to the number of agents
        - **Colors**: Different state categories
        
        Mathematical Model:
            For each transition (state_i → state_j):
            - Flow width = N_ij / max(N) * max_width
            - Node width = Σ(N_ij) / max(Σ(N)) * max_node_width
            
        Args:
            subtitle (str): Additional subtitle for the plot
            
        Returns:
            None: Displays interactive Plotly Sankey diagram
            
        Example:
            >>> analyzer = DwtAnalyzer(scenario_name="example")
            >>> analyzer.sankey_agents(subtitle="TB State Transitions")
            # Opens interactive Sankey diagram in browser
        """
        pass
    
    def histogram_with_kde(self, subtitle=""):
        """
        Creates histograms with kernel density estimation for dwell time distributions.
        
        This method shows how long agents spend in each TB state, providing
        insights into the timing patterns of disease progression.
        
        .. image:: _static/histogram_kde_example.png
            :width: 500px
            :alt: Histogram with KDE showing dwell time distributions
            :align: center
        
        The plot above demonstrates:
        - **Histogram bars**: Frequency distribution of dwell times
        - **KDE curve**: Smooth probability density estimation
        - **Multi-panel layout**: One subplot per TB state
        - **Automatic scaling**: Optimized bin sizes and ranges
        
        Mathematical Model:
            For each state i and dwell time t:
            - Histogram: H_i(bin) = count(dwell_times ∈ bin)
            - KDE: f_i(t) = (1/nh) * Σ K((t-t_j)/h)
            - Bandwidth: h = 1.06 * σ * n^(-1/5) (Silverman's rule)
            
        Args:
            subtitle (str): Additional subtitle for the plot
            
        Returns:
            None: Displays matplotlib figure with subplots
            
        Example:
            >>> plotter = DwtPlotter(file_path='data.csv')
            >>> plotter.histogram_with_kde(subtitle="Dwell Time Analysis")
            # Displays multi-panel histogram with KDE
        """
        pass
    
    def graph_state_transitions_curved(self, states=None, subtitle="", colormap='Paired'):
        """
        Creates a curved network graph showing state transitions with statistics.
        
        This method generates a directed graph visualization where nodes represent
        TB states and edges show transitions with statistical annotations.
        
        .. image:: _static/network_graph_example.png
            :width: 700px
            :alt: Network graph of TB state transitions
            :align: center
        
        The network above illustrates:
        - **Nodes**: TB states (circles with state names)
        - **Edges**: State transitions (curved arrows)
        - **Edge thickness**: Proportional to transition frequency
        - **Edge labels**: Mean dwell time, mode, and agent count
        - **Color coding**: Different states and transition types
        
        Mathematical Model:
            For each transition (state_i → state_j):
            - Edge weight = N_ij / max(N) * max_thickness
            - Mean dwell time = μ_ij = Σ(t_k) / N_ij
            - Mode dwell time = most frequent value in transition
            
        Args:
            states (list, optional): Specific states to include
            subtitle (str): Additional subtitle for the plot
            colormap (str): Matplotlib colormap name
            
        Returns:
            None: Displays matplotlib network visualization
            
        Example:
            >>> analyzer.graph_state_transitions_curved(
            ...     subtitle="State Transition Network",
            ...     colormap='viridis'
            ... )
            # Displays network graph with curved edges
        """
        pass
    
    def reinfections_percents_bars_interactive(self, target_states, scenario=''):
        """
        Creates an interactive bar chart showing reinfection percentages.
        
        This method visualizes the percentage of the population that experienced
        different numbers of reinfections, important for TB epidemiology.
        
        .. image:: _static/reinfection_analysis_example.png
            :width: 800px
            :alt: Interactive bar chart of reinfection percentages
            :align: center
        
        The interactive chart above provides:
        - **Bar heights**: Percentage of population with each reinfection count
        - **Hover information**: Exact percentages and counts
        - **Color coding**: Different reinfection numbers
        - **Interactive features**: Zoom, pan, and export capabilities
        
        Mathematical Model:
            For each reinfection count k:
            - Count agents: N_k = count(agents with max_infection = k)
            - Calculate percentage: P_k = N_k / total_agents * 100
            - Create bar: height = P_k, x = k
            
        Args:
            target_states (list): States that count as infections
            scenario (str): Scenario name for plot title
            
        Returns:
            None: Displays interactive Plotly bar chart
            
        Example:
            >>> plotter.reinfections_percents_bars_interactive(
            ...     target_states=[0.0, 1.0],  # Active TB states
            ...     scenario="Population Reinfection Analysis"
            ... )
            # Opens interactive reinfection analysis in browser
        """
        pass

def create_documentation_plots():
    """
    Creates example plots for documentation purposes.
    
    This function generates sample visualizations that can be used
    in docstrings to illustrate the different plot types.
    """
    
    # Create _static directory if it doesn't exist
    import os
    os.makedirs('_static', exist_ok=True)
    
    # 1. Sankey diagram example (simplified)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create a simple flow diagram representation
    states = ['Susceptible', 'Latent', 'Active', 'Treatment', 'Recovery']
    flows = [100, 80, 20, 15, 10]
    
    y_pos = np.linspace(0, 1, len(states))
    ax.barh(y_pos, flows, height=0.1, alpha=0.7, color=['blue', 'orange', 'red', 'green', 'purple'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(states)
    ax.set_xlabel('Number of Agents')
    ax.set_title('TB State Transitions (Sankey Representation)')
    ax.grid(True, alpha=0.3)
    plt.savefig('_static/sankey_diagram_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram with KDE example
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    states = ['Latent', 'Active', 'Treatment', 'Recovery']
    
    for i, (ax, state) in enumerate(zip(axes.flat, states)):
        # Generate sample dwell time data
        if state == 'Latent':
            data = np.random.exponential(365, 1000)  # Long dwell times
        elif state == 'Active':
            data = np.random.exponential(100, 1000)  # Medium dwell times
        elif state == 'Treatment':
            data = np.random.exponential(180, 1000)  # Treatment duration
        else:
            data = np.random.exponential(50, 1000)   # Recovery time
            
        ax.hist(data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax.set_title(f'{state} State Dwell Times')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('_static/histogram_kde_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Network graph example
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a simple network representation
    nodes = ['Susceptible', 'Latent', 'Active', 'Treatment', 'Recovery']
    node_pos = [(0, 0.5), (0.25, 0.8), (0.25, 0.2), (0.5, 0.5), (0.75, 0.5)]
    
    # Draw nodes
    for i, (node, pos) in enumerate(zip(nodes, node_pos)):
        ax.scatter(pos[0], pos[1], s=200, c=f'C{i}', alpha=0.7)
        ax.annotate(node, (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points')
    
    # Draw edges (simplified)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for start, end in edges:
        x1, y1 = node_pos[start]
        x2, y2 = node_pos[end]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, alpha=0.7))
    
    ax.set_xlim(-0.1, 0.85)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('TB State Transition Network')
    ax.axis('off')
    plt.savefig('_static/network_graph_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Reinfection analysis example
    fig, ax = plt.subplots(figsize=(10, 6))
    
    reinfection_counts = [0, 1, 2, 3, 4]
    percentages = [60, 25, 10, 4, 1]  # Example percentages
    
    bars = ax.bar(reinfection_counts, percentages, color='lightcoral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Reinfections')
    ax.set_ylabel('Percentage of Population (%)')
    ax.set_title('Reinfection Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct}%', ha='center', va='bottom')
    
    plt.savefig('_static/reinfection_analysis_example.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Documentation plots created in _static/ directory")

if __name__ == "__main__":
    create_documentation_plots() 