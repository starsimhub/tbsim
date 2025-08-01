"""
Example of enhanced DwtAnalyzer docstrings with embedded images.

This shows how to use the generated images in actual docstrings
for the TB simulation analyzer class.
"""

class DwtAnalyzer:
    """
    Dwell Time Analyzer for TB Simulation.
    
    This class provides comprehensive tools for analyzing and visualizing
    dwell time data from tuberculosis simulation runs.
    
    .. image:: _static/sankey_diagram_example.png
        :width: 600px
        :alt: Example Sankey diagram showing TB state transitions
        :align: center
    
    The analyzer tracks how long agents spend in each TB state and
    provides multiple visualization types including Sankey diagrams,
    network graphs, histograms, and interactive charts.
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
    
    def barchar_all_state_transitions_interactive(self, dwell_time_bins=None, filter_states=None):
        """
        Creates an interactive bar chart of all state transitions grouped by dwell time.
        
        This method provides detailed analysis of state transitions with
        customizable dwell time categories and state filtering.
        
        .. image:: _static/interactive_bar_example.png
            :width: 800px
            :alt: Interactive bar chart of state transitions by dwell time
            :align: center
        
        The interactive chart above shows:
        - **Horizontal bars**: Each representing a state transition
        - **Dwell time categories**: Grouped by time ranges (0-30d, 30-90d, etc.)
        - **Bar heights**: Number of agents making each transition
        - **Interactive features**: Hover details, zoom, and filtering
        
        Mathematical Model:
            For each transition (state_i → state_j) and dwell time bin [t_min, t_max):
            - Filter transitions: T where dwell_time ∈ [t_min, t_max)
            - Count agents: N_ij(bin) = count(T)
            - Create bar: height = N_ij(bin), label = "state_i → state_j (bin_range)"
            
        Args:
            dwell_time_bins (list, optional): Bin edges for categorizing dwell times
            filter_states (list, optional): States to include in analysis
            
        Returns:
            None: Displays interactive Plotly bar chart
            
        Example:
            >>> plotter.barchar_all_state_transitions_interactive(
            ...     dwell_time_bins=[0, 30, 90, 180, 365, float('inf')],
            ...     filter_states=['Latent', 'Active', 'Treatment']
            ... )
            # Opens interactive bar chart in browser
        """
        pass

# Example usage in a real docstring
def example_usage():
    """
    Example of how to use the DwtAnalyzer with visual documentation.
    
    This function demonstrates the workflow for analyzing TB simulation data
    and generating visualizations with embedded documentation.
    
    .. image:: _static/sankey_diagram_example.png
        :width: 400px
        :alt: Example Sankey diagram
        :align: left
    
    The workflow involves:
    1. Creating a simulation with the analyzer
    2. Running the simulation to collect data
    3. Generating various visualizations
    4. Using the plots in documentation
    
    .. image:: _static/histogram_kde_example.png
        :width: 400px
        :alt: Example histogram with KDE
        :align: right
    
    Each visualization type provides different insights into the
    simulation dynamics and can be used for different analytical purposes.
    """
    pass 