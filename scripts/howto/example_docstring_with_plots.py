"""
Example of how to add plots to docstrings for the DwtAnalyzer class.

This demonstrates several approaches for including visualizations in docstrings.
"""

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
import io

def sankey_agents_with_example():
    """
    Creates a Sankey diagram showing agent flow between TB states.
    
    This method visualizes how agents move between different tuberculosis states
    (e.g., Susceptible → Latent → Active → Treatment → Recovery).
    
    .. image:: _static/sankey_example.png
        :width: 600px
        :alt: Sankey diagram showing TB state transitions
        :align: center
    
    The diagram above shows a typical Sankey visualization where:
    - Node width represents the number of agents in each state
    - Flow width represents the number of agents transitioning between states
    - Colors distinguish different state types
    
    Example:
        >>> analyzer = DwtAnalyzer(scenario_name="example")
        >>> analyzer.sankey_agents(subtitle="TB State Transitions")
        # Displays interactive Sankey diagram in browser
    
    Returns:
        None: Displays interactive Plotly Sankey diagram
    """
    pass

def histogram_with_kde_example():
    """
    Creates a histogram with kernel density estimation for dwell time distributions.
    
    This method shows the distribution of how long agents spend in each TB state,
    with both histogram bars and a smooth density curve overlay.
    
    .. image:: _static/histogram_example.png
        :width: 500px
        :alt: Histogram with KDE showing dwell time distributions
        :align: center
    
    The plot above demonstrates:
    - Histogram bars showing frequency of dwell times
    - KDE curve showing the underlying probability distribution
    - Different colors for each TB state
    - Automatic binning and scaling
    
    Mathematical Model:
        For each state i and dwell time t:
        - Histogram: H_i(bin) = count(dwell_times ∈ bin)
        - KDE: f_i(t) = (1/nh) * Σ K((t-t_j)/h) where K is the kernel function
    
    Example:
        >>> plotter = DwtPlotter(file_path='data.csv')
        >>> plotter.histogram_with_kde(subtitle="Dwell Time Analysis")
        # Displays multi-panel histogram with KDE
    
    Returns:
        None: Displays matplotlib figure with subplots
    """
    pass

def network_graph_example():
    """
    Creates a network graph showing state transitions with statistical annotations.
    
    This method generates a directed graph where nodes represent TB states and
    edges represent transitions, with edge thickness proportional to transition frequency.
    
    .. image:: _static/network_example.png
        :width: 700px
        :alt: Network graph of TB state transitions
        :align: center
    
    The network above illustrates:
    - Nodes: TB states (Susceptible, Latent, Active, etc.)
    - Edges: State transitions with thickness ∝ agent count
    - Edge labels: Mean dwell time, mode, and agent count
    - Color coding: Different states and transition types
    
    Example:
        >>> analyzer.graph_state_transitions_curved(
        ...     subtitle="State Transition Network",
        ...     colormap='viridis'
        ... )
        # Displays interactive network graph
    
    Returns:
        None: Displays matplotlib network visualization
    """
    pass

def interactive_plot_example():
    """
    Creates an interactive bar chart for detailed state transition analysis.
    
    This method generates an interactive Plotly visualization showing state
    transitions grouped by dwell time categories.
    
    .. image:: _static/interactive_example.png
        :width: 800px
        :alt: Interactive bar chart of state transitions
        :align: center
    
    The interactive chart above provides:
    - Hover information showing exact counts and percentages
    - Filterable state selections
    - Customizable dwell time bins
    - Zoom and pan capabilities
    - Export functionality
    
    Example:
        >>> plotter.barchar_all_state_transitions_interactive(
        ...     dwell_time_bins=[0, 30, 90, 180, 365, float('inf')],
        ...     filter_states=['Latent', 'Active', 'Treatment']
        ... )
        # Opens interactive plot in browser
    
    Returns:
        None: Displays interactive Plotly bar chart
    """
    pass

def create_example_plots():
    """
    Creates example plots for documentation purposes.
    
    This function generates sample plots that can be used in docstrings
    to illustrate the different visualization types available.
    """
    
    # Example 1: Simple line plot
    plt.figure(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Example Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('_static/example_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Example 2: Histogram
    plt.figure(figsize=(8, 6))
    data = np.random.normal(0, 1, 1000)
    plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Example Histogram')
    plt.grid(True, alpha=0.3)
    plt.savefig('_static/example_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Example 3: Scatter plot
    plt.figure(figsize=(8, 6))
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    plt.scatter(x, y, alpha=0.6, c='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Example Scatter Plot')
    plt.grid(True, alpha=0.3)
    plt.savefig('_static/example_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Example plots created in _static/ directory")

if __name__ == "__main__":
    create_example_plots() 