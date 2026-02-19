"""
TB Simulation Dwell Time Analysis and Visualization Module

This module provides comprehensive tools for analyzing and visualizing dwell time data
from tuberculosis simulation runs. It includes three main classes:

1. DwtAnalyzer: Records dwell times during simulation execution
2. DwtPlotter: Creates various visualizations of dwell time data
3. DwtPostProcessor: Aggregates and processes multiple simulation results

The module supports multiple visualization types including:
- Sankey diagrams for state transitions
- Interactive bar charts and histograms
- Network graphs of state transitions
- Kaplan-Meier survival curves
- Reinfection analysis

Key Features:
- Real-time dwell time tracking during simulation
- Multiple data aggregation methods
- Interactive and static visualizations
- Comprehensive statistical analysis
- Support for multiple simulation scenarios

Usage Examples:

Basic Analysis Setup:
```python
import starsim as ss
from tbsim import TB
from tbsim.analyzers import DwtAnalyzer

# Create simulation with analyzer
sim = ss.Sim(diseases=[TB()], analyzers=DwtAnalyzer(scenario_name="Baseline"))

sim.run()

# Access analyzer results
analyzer = sim.analyzers[0]
analyzer.plot_dwell_time_validation()
```

Post-Processing Multiple Runs:
```python
from tbsim.analyzers import DwtPostProcessor

# Aggregate multiple simulation results
postproc = DwtPostProcessor(directory='results', prefix='Baseline')
postproc.sankey_agents()
postproc.histogram_with_kde()
```

Direct Data Analysis:
```python
from tbsim.analyzers import DwtPlotter

# Analyze existing data file
plotter = DwtPlotter(file_path='results/Baseline-20240101120000.csv')
plotter.sankey_agents()
plotter.graph_state_transitions_curved()
```

References:
- NetworkX for graph visualizations
- Plotly for interactive plots
- Lifelines for survival analysis
- Seaborn for statistical visualizations
"""

import matplotlib.colors as mcolors
import pandas as pd
import starsim as ss
import numpy as np
import os
import tbsim.utils.config as cfg
import datetime as ddtt
import tbsim as mtb
from scipy import stats
from enum import IntEnum
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import networkx as nx
import plotly.graph_objects as go 
import warnings
import plotly.express as px
import itertools as it
        

__all__ = ['DwtAnalyzer', 'DwtPlotter', 'DwtPostProcessor']

warnings.simplefilter(action='ignore', category=FutureWarning)

class DwtPlotter:
    """
    Dwell Time Plotter for TB Simulation Data Visualization
    
    This class provides comprehensive visualization tools for analyzing dwell time data
    from tuberculosis simulations. It supports both interactive (Plotly) and static
    (Matplotlib) visualizations with various chart types including Sankey diagrams,
    bar charts, histograms, and network graphs.
    
    The class can work with data from:
    - Direct DataFrame input
    - CSV file paths
    - Aggregated simulation results
    
    Key Visualization Types:
    - Sankey diagrams for state transition flows
    - Interactive bar charts for dwell time distributions
    - Network graphs showing state transition relationships
    - Histograms with kernel density estimation
    - Kaplan-Meier survival curves
    - Reinfection analysis plots
    
    Attributes:
        data (pd.DataFrame): The dwell time data to be visualized
        data_file (str): Path to the source data file (if loaded from file)
        
    Example Usage:
    ```python
    # Load data from file
    plotter = DwtPlotter(file_path='results/simulation_data.csv')
    plotter.sankey_agents()
    plotter.histogram_with_kde()
    
    # Work with existing DataFrame
    df = pd.read_csv('data.csv')
    plotter = DwtPlotter(data=df)
    plotter.graph_state_transitions_curved()
    ```
    """
    
    def __init__(self, data=None, file_path=None):
        """
        Initialize the DwtPlotter with dwell time data.
        
        Args:
            data (pd.DataFrame, optional): DataFrame containing dwell time data.
                                         Must have columns: 'state_name', 'going_to_state', 
                                         'dwell_time', 'agent_id', 'age'
            file_path (str, optional): Path to CSV file containing dwell time data.
                                     If provided, data will be loaded and cleaned.
                                     
        Raises:
            ValueError: If neither data nor file_path is provided.
            
        Data Requirements:
            The data should contain the following columns:
            - state_name: Name of the current state
            - going_to_state: Name of the state being transitioned to
            - dwell_time: Time spent in the current state
            - agent_id: Unique identifier for each agent
            - age: Age of the agent at transition time
            
        Example:
        ```python
        # Initialize with file
        plotter = DwtPlotter(file_path='results/Baseline-20240101120000.csv')
        
        # Initialize with DataFrame
        df = pd.read_csv('data.csv')
        plotter = DwtPlotter(data=df)
        ```
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif file_path is not None:
            self.data = self.__cleandata__(file_path)
        else:
            raise ValueError("Either data or file_path must be provided.")
        if self.__data_error__():
            print("No data provided, or data is corrupted")
            
    def sankey_agents_by_age_subplots(self, bins=[0, 5, 200], scenario="", includecycles=False):
        """
        Generate Sankey diagrams of state transitions filtered by age bins.
        
        Creates multiple Sankey diagrams showing state transition flows for different
        age groups. Each diagram visualizes how agents move between states within
        specific age ranges.
        
        Mathematical Model:
            For each age bin [bin_min, bin_max):
            - Filter agents where age ∈ [bin_min, bin_max)
            - Count transitions: T(state_i → state_j) = count
            - Create Sankey diagram with nodes = states, flows = transition counts
            
        Args:
            bins (list): Age bin boundaries. Default [0, 5, 200] creates two bins:
                        [0-5) and [5-200) years
            scenario (str): Scenario name for plot titles
            includecycles (bool): Whether to include self-transitions (currently unused)
                                
        Returns:
            None: Displays interactive Plotly Sankey diagrams
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create age-specific Sankey diagrams
        plotter.sankey_agents_by_age_subplots(
            bins=[0, 18, 65, 100],  # Child, Adult, Elderly
            scenario="Baseline TB Model"
        )
        ```
        
        Visualization Features:
        - Interactive hover information showing transition counts
        - Color-coded nodes and links
        - Age-specific filtering for targeted analysis
        - Automatic layout optimization
        """
        data = self.data
        data['age'] = data['age'].astype(float)
        num_bins = 0
        # Define age bins
        age_min, age_max = data["age"].min(), data["age"].max()
        if bins is None:
            bin_width = (age_max - age_min) / 4
            bins = [age_min + i * bin_width for i in range(num_bins + 1)]
            # make sure the bins are integers 
            bins = [int(b) for b in bins]
        
        # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        num_bins = len(bins) - 1
        for i in range(num_bins):
            # Filter data for the age bin
            bin_min, bin_max = bins[i], bins[i + 1]
            df_bin = data[(data["age"] >= bin_min) & (data["age"] < bin_max)]

            if not df_bin.empty:
                transition_counts = df_bin.groupby(["state_name", "going_to_state"]).size().reset_index(name="count")

                # Create a dictionary to map unique labels to indices
                labels = list(set(df_bin["state_name"]) | set(df_bin["going_to_state"]))
                label_to_index = {label: idx for idx, label in enumerate(labels)}

                # Map source and target to indices
                source_indices = transition_counts["state_name"].map(label_to_index)
                target_indices = transition_counts["going_to_state"].map(label_to_index)
                values = transition_counts["count"]

                # Create Sankey plot
                sankey = go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        label=labels
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=values
                    )
                )
                sankey_fig = go.Figure(sankey)
                sankey_fig.update_layout(title_text=f"Sankey Diagram (Ages >= {bin_min} and <{bin_max}),<br> {scenario} <br> DwtPlotter.sankey_agents_by_age_subplots()")
                sankey_fig.show()

    def sankey_agents_even_age_ranges(self, number_of_plots=2, scenario=""):
        """
        Generate Sankey diagrams with evenly distributed age ranges.
        
        Creates multiple Sankey diagrams by automatically dividing the age range
        into equal-width bins. This is useful for comparing state transitions
        across different age groups without manual bin specification.
        
        Mathematical Model:
            age_range = max_age - min_age
            bin_width = age_range / number_of_plots
            bins[i] = min_age + i * bin_width for i = 0 to number_of_plots
            
        Args:
            number_of_plots (int): Number of age bins to create. Default 2
            scenario (str): Scenario name for plot titles
                            
        Returns:
            None: Displays interactive Plotly Sankey diagrams
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create 4 evenly-spaced age groups
        plotter.sankey_agents_even_age_ranges(
            number_of_plots=4,
            scenario="Age-stratified Analysis"
        )
        ```
        
        Features:
        - Automatic age range calculation
        - Even distribution across age spectrum
        - Interactive hover information
        - Consistent layout across all diagrams
        """
        data = self.data
        # Define age bins
        age_min, age_max = data["age"].min(), data["age"].max()
        num_bins = number_of_plots
        bin_width = (age_max - age_min) / num_bins
        bins = [age_min + i * bin_width for i in range(num_bins + 1)]

        for i in range(num_bins):
            # Filter data for the age bin
            bin_min, bin_max = bins[i], bins[i + 1]
            df_bin = data[(data["age"] >= bin_min) & (data["age"] < bin_max)]

            if not df_bin.empty:
                transition_counts = df_bin.groupby(["state_name", "going_to_state"]).size().reset_index(name="count")

                # Create a dictionary to map unique labels to indices
                labels = list(set(df_bin["state_name"]) | set(df_bin["going_to_state"]))
                label_to_index = {label: idx for idx, label in enumerate(labels)}

                # Map source and target to indices
                source_indices = transition_counts["state_name"].map(label_to_index)
                target_indices = transition_counts["going_to_state"].map(label_to_index)
                values = transition_counts["count"]

                # Create Sankey plot
                fig = go.Figure(go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        label=labels
                    ),
                    link=dict(
                        source=source_indices,
                        target=target_indices,
                        value=values
                    )
                ))
                fig.add_annotation(x = -0.05, y = -0.2, text = 'DwtPlotter.sankey_agents_even_age_ranges()', xref = 'paper',
                            yref = 'paper', showarrow = False,
                            font=dict(color='black',size=8))
        
                fig.update_layout(title_text=f"Sankey Diagram (Ages {bin_min:.1f} - {bin_max:.1f})<br> {scenario} <br> DwtPlotter.sankey_agents_even_age_ranges()")
                fig.show()

    def sankey_agents(self, subtitle = ""):
        """
        Generate a comprehensive Sankey diagram of all state transitions.
        
        Creates a single Sankey diagram showing the complete flow of agents
        between all states in the simulation. The diagram visualizes both
        the direction and magnitude of state transitions.
        
        .. image:: ../_static/sankey_diagram_example.png
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
            - Count agents: N_ij = count(agents transitioning from i to j)
            - Create flow: flow_i→j = N_ij
            - Node size ∝ total transitions (in + out)
            - Link width ∝ transition count
            
        Args:
            subtitle (str): Additional subtitle for the plot
                           
        Returns:
            None: Displays interactive Plotly Sankey diagram
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create comprehensive state transition diagram
        plotter.sankey_agents(subtitle="Baseline TB Model Results")
        ```
        
        Visualization Features:
        - Color-coded nodes and links
        - Interactive hover showing exact transition counts
        - Automatic layout optimization
        - Proportional sizing based on transition frequency
        - Professional styling with grid and annotations
        """
        
        if self.__data_error__():
            return
        
        df = self.data

        # Prepare data for Sankey plot
        source = df['state_name']
        target = df['going_to_state']

        # Count the number of agents for each transition
        transition_counts = df.groupby(['state_name', 'going_to_state']).size().reset_index(name='count')

        # Create a dictionary to map unique labels to indices
        labels = list(set(source) | set(target))
        label_to_index = {label: i for i, label in enumerate(labels)}

        # Map source and target to indices
        source_indices = transition_counts['state_name'].map(label_to_index)
        target_indices = transition_counts['going_to_state'].map(label_to_index)
        values = transition_counts['count']

        # Generate colors for nodes
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

        node_colors = [f'rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, 1.0)' for c in colors]

        # Generate lighter colors for links
        link_colors = [f'rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, 0.5)' for c in colors]

        # Map source indices to link colors
        link_color_map = {i: link_colors[i] for i in range(len(labels))}
        link_colors = [link_color_map[idx] for idx in source_indices]

        # Create the Sankey plot
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.2),
            label=labels,
            color=node_colors
            ),
            link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            color=link_colors,
            hovertemplate='%{source.label} → %{target.label}: %{value} agents<br>',
            line=dict(color="lightgray", width=0.1),
            label=values,  # Add labels to the links
            )
        ))
        # fig.update_layout(title_text=f"Sankey Diagram of State Transitions by Agent Count <\br>{subtitle}", font_size=10)

        fig.update_layout(
            hovermode = 'x',
            title=dict(text=f"State Transitions  - Agents Count<br>{subtitle} (DwtPlotter.sankey_agents())", font=dict(size=12)),
            font=dict(size = 12, color = 'black'),
        )
        fig.show()

    def sankey_dwelltimes(self, subtitle=''):
        """
        Generate a Sankey diagram weighted by dwell times.
        
        Creates a Sankey diagram where the flow thickness represents the total
        dwell time rather than the number of agents. This visualization shows
        the time-weighted importance of different state transitions.
        
        Mathematical Model:
            For each transition (state_i → state_j):
            - Sum dwell times: T_ij = Σ(dwell_time for i→j transitions)
            - Create flow: flow_i→j = T_ij
            - Link width ∝ total dwell time
            - Node size ∝ total time spent in state
            
        Args:
            subtitle (str): Additional subtitle for the plot
                           
        Returns:
            None: Displays interactive Plotly Sankey diagram
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create dwell time-weighted diagram
        plotter.sankey_dwelltimes(subtitle="Time-weighted Analysis")
        ```
        
        Features:
        - Dwell time-weighted flows
        - Interactive hover showing total time
        - Color-coded nodes and links
        - Professional styling and layout
        - Time annotations
        """
        

        if self.__data_error__():
            return
        
        df = self.data

        # Prepare data for Sankey plot
        source = df['state_name']
        target = df['going_to_state']
        value = df['dwell_time']

        # Create a dictionary to map unique labels to indices
        labels = list(set(source) | set(target))
        label_to_index = {label: i for i, label in enumerate(labels)}

        # Map source and target to indices
        source_indices = source.map(label_to_index)
        target_indices = target.map(label_to_index)

        # Generate colors for nodes
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        node_colors = [f'rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, 1.0)' for c in colors]
        link_colors = [f'rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, 0.5)' for c in colors]
        link_color_map = {i: link_colors[i] for i in range(len(labels))}
        link_colors = [link_color_map[idx] for idx in source_indices]

        # Create the Sankey plot
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.2),
                color=node_colors,
                label=labels
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                color=link_colors,
                value=value,
                hovertemplate='%{source.label} → %{target.label}: %{value}<br>',  #TODO: Add time unit
                line=dict(color="lightgray", width=0.1),
            )
        ))

        # fig.update_layout(title_text="Sankey Diagram of State Transitions -  DWELL TIME", font_size=10)

        fig.update_layout(
            # hovermode='x',
            title=dict(text=f"State Transitions - Dwell Times<br>{subtitle}  (DwtPlotter.sankey_dwelltimes())", font=dict(size=12)),
            font=dict(size=10, color='black'),
            margin=dict(l=20, r=20, t=40, b=20),
            # paper_bgcolor='white',
            # plot_bgcolor='white'
        )

        fig.show()

    def barchar_all_state_transitions_interactive(self, dwell_time_bins=None, filter_states=None):
        """
        Generate an interactive bar chart of state transitions grouped by dwell time categories.
        
        This method provides detailed analysis of state transitions with
        customizable dwell time categories and state filtering.
        
        .. image:: ../_static/interactive_bar_example.png
            :width: 800px
            :alt: Interactive bar chart of state transitions by dwell time
            :align: center
        
        The interactive chart above shows:
        - **Horizontal bars**: Each representing a state transition
        - **Dwell time categories**: Grouped by time ranges (0-30d, 30-90d, etc.)
        - **Bar heights**: Number of agents making each transition
        - **Interactive features**: Hover details, zoom, and filtering
        
        Mathematical Model:
            For each dwell time bin [bin_min, bin_max):
            - Filter transitions: T where dwell_time ∈ [bin_min, bin_max)
            - Count transitions: N(state_i → state_j) in bin
            - Create bar: height = N, label = "state_i → state_j (bin_range)"
            
        Args:
            dwell_time_bins (list, optional): Bin edges for categorizing dwell times.
                                            Default: [0, 50, 100, 150, 200, 250, ∞]
            filter_states (list, optional): States to include in analysis.
                                          If None, includes all states
                                          
        Returns:
            None: Displays interactive Plotly bar chart
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Custom dwell time bins
        bins = [0, 30, 90, 180, 365, float('inf')]  # Monthly, quarterly, etc.
        plotter.barchar_all_state_transitions_interactive(
            dwell_time_bins=bins,
            filter_states=['Latent', 'Active', 'Treatment']
        )
        ```
        
        Features:
        - Interactive hover information
        - Customizable dwell time bins
        - State filtering capabilities
        - Automatic height adjustment
        - Professional styling
        """

        

        if self.__data_error__():
            return

        # Set default bins if none are provided
        if dwell_time_bins is None:
            dwell_time_bins = [0, 50, 100, 150, 200, 250]

        # Append infinity to the bins if not already present
        if np.inf not in dwell_time_bins:
            dwell_time_bins.append(np.inf)

        # Create bin labels, handling infinity separately
        dwell_time_labels = []
        for b, d in zip(dwell_time_bins[:-1], dwell_time_bins[1:]):
            if b != np.inf and d != np.inf:
                dwell_time_labels.append(f"{int(b)}-{int(d)} step_time_units")
            elif b != np.inf and d == np.inf:
                dwell_time_labels.append(f"{int(b)}+ step_time_units")
            else:
                # Handle case where b might be inf (shouldn't happen with normal bins)
                dwell_time_labels.append(f"{b}-{d} step_time_units")

        # Create a dwell time category column
        self.data['dwell_time_category'] = pd.cut(
            self.data['dwell_time'],
            bins=dwell_time_bins,
            labels=dwell_time_labels,
            include_lowest=True
        )

        # Apply state filter if provided
        if filter_states is not None:
            filtered_logger = self.data[
                self.data['state_name'].isin(filter_states) |
                self.data['going_to_state'].isin(filter_states)
            ]
        else:
            filtered_logger = self.data

        # Group by state transitions and dwell time category
        grouped = filtered_logger.groupby(
            ['state_name', 'going_to_state', 'dwell_time_category'], observed=True
        ).size().reset_index(name='count')

        # Filter out empty ranges
        grouped = grouped[grouped['count'] > 0]

        # Interactive state transitions
        fig = go.Figure()

        for _, row in grouped.iterrows():
            state_label = row['state_name']
            going_to_state_label = row['going_to_state']
            dwell_time_category = row['dwell_time_category']
            count = row['count']
            unique_transition = f"{state_label} → {going_to_state_label} ({dwell_time_category})"

            # Add transition as a bar
            fig.add_trace(go.Bar(
                y=[unique_transition],
                x=[count],
                name=unique_transition,
                text=[count],
                textposition='auto',
                orientation='h'
            ))
        fig.add_annotation(x = -0.05, y = -0.2, text = 'DwtPlotter.interactive_all_state_transitions()', xref = 'paper',
            yref = 'paper', showarrow = False,
            font=dict(color='black',size=8))
        fig.update_layout(
            title="State Transitions Grouped by Dwell Time Categories <br>DwtPlotter.interactive_all_state_transitions()",
            yaxis_title="State Transitions",
            xaxis_title="Count",
            legend_title="Transitions",
            height=100 + 30 * len(grouped),
        )
        fig.show()

    def stacked_bars_states_per_agent_static(self):
        """
        Create a stacked bar chart showing cumulative dwell time per agent.
        
        Generates a static matplotlib visualization showing how much time each
        agent spent in different states throughout the simulation. Each bar
        represents an agent, with segments showing time spent in each state.
        
        Mathematical Model:
            For each agent i and state j:
            - Calculate cumulative time: T_ij = Σ(dwell_time for agent i in state j)
            - Create stacked bar: agent i = [T_i1, T_i2, ..., T_in]
            - Total bar height = Σ(T_ij) for all states j
            
        Returns:
            None: Displays static matplotlib stacked bar chart
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Show cumulative time per agent
        plotter.stacked_bars_states_per_agent_static()
        ```
        
        Features:
        - Color-coded state segments
        - Agent-level analysis
        - Cumulative time visualization
        - Professional styling with legends
        - Automatic figure sizing
        """
        if self.__data_error__():
            return
        df = self.data
        # Calculate cumulative dwell time for each agent and state
        df['cumulative_dwell_time'] = df.groupby(['agent_id', 'state_name'])['dwell_time'].cumsum()

        # Use cumulative dwell time directly
        df['cumulative_dwell_time_units'] = df['cumulative_dwell_time']

        # Ensure column order matches color mapping
        state_colors, cmap = Utils.colors()
        matching_colors = [state_colors[state] for state in df['state_name'].unique() if state in state_colors]
        cmap = mcolors.ListedColormap(matching_colors)


        # Pivot the data to get cumulative dwell time for each state
        pivot_df = df.pivot_table(index='agent_id', columns='state_name', values='cumulative_dwell_time_units', aggfunc='max', fill_value=0)
        pivot_df.plot(kind='bar', stacked=True, figsize=(15, 40), colormap=cmap )

        # Plot the data
        # pivot_df.plot(kind='bar', stacked=True, figsize=(15, 7))
        plt.title('Cumulative Time on Each State for All Agents')
        plt.annotate('DwtPlotter.stacked_bars_states_per_agent_static()', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=12)
        plt.xlabel('Agent ID')
        plt.ylabel('Cumulative Time (step_time_units)')
        plt.legend(title='State Name')
        plt.tight_layout()
        plt.show()


    def reinfections_age_bins_bars_interactive(self, target_states, barmode = 'group', scenario=''):
        """
        Analyze reinfection patterns by age groups using interactive bar charts.
        
        Creates an interactive visualization showing the distribution of maximum
        reinfection counts across different age groups. This helps identify
        age-specific patterns in TB reinfection rates.
        
        Mathematical Model:
            For each agent i:
            - Find maximum reinfection count: max_inf_i = max(infection_num for agent i)
            - Assign age group: age_group_i = bin(age_i)
            - Count by age group: N(age_group, reinfection_count)
            
        Args:
            target_states (list): States that count as infections (e.g., [0.0, 1.0])
            barmode (str): Bar mode for grouping ('group', 'stack', 'overlay')
            scenario (str): Scenario name for plot title
                           
        Returns:
            None: Displays interactive Plotly bar chart
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Analyze reinfections by age
        plotter.reinfections_age_bins_bars_interactive(
            target_states=[0.0, 1.0],  # Active TB states
            barmode='group',
            scenario="Age-stratified Reinfection Analysis"
        )
        ```
        
        Features:
        - Age group stratification
        - Interactive hover information
        - Multiple bar modes
        - Automatic age binning
        - Professional styling
        """


        if self.__data_error__():
            return
        

        # Ensure the 'infection_num' column exists
        if 'infection_num' not in self.data.columns:
            reinfections, total_count = self.__generate_reinfection_data__(target_states=target_states, scenario=Utils.to_filename_friendly(scenario) )
        else:
            reinfections = self.data

        # Calculate the maximum number of reinfections for each agent
        max_reinfections = reinfections.groupby('agent_id')['infection_num'].max().reset_index()
        # Merge with age data
        age_data = reinfections[['agent_id', 'age']].drop_duplicates()
        max_reinfections = max_reinfections.merge(age_data, on='agent_id')

        # Define age bins
        age_bins = [0, 5, 16, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
        age_labels = []
        for i, b in enumerate(age_bins[:-1]):
            if age_bins[i+1] != np.inf:
                age_labels.append(f'{int(b)}-{int(age_bins[i+1])}')
            else:
                age_labels.append(f'{int(b)}+')
        max_reinfections['age_group'] = pd.cut(max_reinfections['age'], bins=age_bins, labels=age_labels, right=False)

        # Plot the data
        fig = px.histogram(max_reinfections, x='infection_num', color='age_group', barmode=barmode,
                           labels={'infection_num': 'Number of Reinfections', 'age_group': 'Age Group'},
                           title=f'Distribution of Maximum Reinfections per Agent by Age Group {scenario}',
                           category_orders={'age_group': age_labels})
        fig.update_layout(bargap=0.2)
        fig.show()

    def reinfections_percents_bars_interactive(self, target_states, scenario=''):
        """
        Visualize reinfection percentages across the population.
        
        This method visualizes the percentage of the population that experienced
        different numbers of reinfections, important for TB epidemiology.
        
        .. image:: ../_static/reinfection_analysis_example.png
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
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Show reinfection percentages
        plotter.reinfections_percents_bars_interactive(
            target_states=[0.0, 1.0],
            scenario="Population Reinfection Analysis"
        )
        ```
        
        Features:
        - Percentage-based visualization
        - Interactive hover showing exact percentages
        - Color-coded bars
        - Professional styling with percentage formatting
        """
        if self.__data_error__():
            return

        # Ensure the 'infection_num' column exists
        if 'infection_num' not in self.data.columns:
            reinfections, total_count = self.__generate_reinfection_data__(target_states=target_states, scenario=Utils.to_filename_friendly(scenario))
        else:
            reinfections = self.data

        total_infections = reinfections[reinfections['infection_num'] > 0]['infection_num'].count()

        # Calculate the total sum of percent for each infection number
        percent_reinfections = reinfections.groupby('infection_num')['percent'].sum().reset_index()
        
        # Plot it
        fig = px.bar(percent_reinfections, x='infection_num', y='percent', color='infection_num', 
             labels={'infection_num': 'Number of Reinfections', 'percent': f'Total Percent {total_count:,}'},
             title=f'Distribution of Maximum Reinfections per Agent {scenario}')
        fig.update_layout(yaxis_tickformat='.1%')
        fig.show()

    def reinfections_bystates_bars_interactive(self, target_states, scenario='', barmode='group'):
        """
        Analyze reinfection patterns by state transitions.
        
        Creates an interactive visualization showing how reinfection patterns
        vary across different state transitions. This helps identify which
        transition paths are associated with higher reinfection rates.
        
        Mathematical Model:
            For each state transition (state_i → state_j):
            - Find agents with this transition: A_ij = agents with transition i→j
            - Calculate max reinfections: max_inf_ij = max(infection_num for A_ij)
            - Count by reinfection number: N(transition, reinfection_count)
            
        Args:
            target_states (list): States that count as infections
            scenario (str): Scenario name for plot title
            barmode (str): Bar mode for grouping
                           
        Returns:
            None: Displays interactive Plotly bar chart
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Analyze reinfections by state transitions
        plotter.reinfections_bystates_bars_interactive(
            target_states=[0.0, 1.0],
            barmode='group',
            scenario="State Transition Reinfection Analysis"
        )
        ```
        
        Features:
        - State transition analysis
        - Interactive hover information
        - Multiple bar modes
        - Professional styling
        """
        if self.__data_error__():
            return

        # Ensure the 'infection_num' column exists
        if 'infection_num' not in self.data.columns:
            reinfections, total_count = self.__generate_reinfection_data__(target_states=target_states, scenario=Utils.to_filename_friendly(scenario))
        else:
            reinfections = self.data

        # Calculate the maximum number of reinfections for each agent
        max_reinfections = reinfections.groupby('agent_id')['infection_num'].max().reset_index()
        # Merge with state_name and going_to_state data
        state_data = reinfections[['agent_id', 'state_name', 'going_to_state']] #.drop_duplicates() - interested on all of them
        max_reinfections = max_reinfections.merge(state_data, on='agent_id')

        # Combine state_name and going_to_state for better visualization
        max_reinfections['state_transition'] = max_reinfections['state_name'] + ' → ' + max_reinfections['going_to_state']

        # Plot the data
        fig = px.histogram(max_reinfections, x='infection_num', color='state_transition', barmode=barmode,
                           labels={'infection_num': 'Number of Reinfections ', 'state_transition': 'State Transition'},
                           title=f'Distribution of Maximum Reinfections per Agent by State Transition {scenario}')
        fig.update_layout(bargap=0.2)
        fig.show()

    def stackedbars_dwelltime_state_interactive(self, bin_size=3, num_bins=20):
        """
        Create interactive stacked bar charts of dwell times by state.
        
        Generates an interactive visualization showing the distribution of dwell
        times for each state, with transitions to other states stacked within
        each dwell time bin. This helps identify timing patterns in state transitions.
        
        Mathematical Model:
            For each state i and dwell time bin [b, b+bin_size):
            - Filter transitions: T where state=i and dwell_time ∈ [b, b+bin_size)
            - Count by target state: N(state_i → state_j) in bin
            - Create stacked bar: height = Σ(N) for all target states j
            
        Args:
            bin_size (int): Size of each dwell time bin. Default 3
            num_bins (int): Number of bins to create. Default 20
                           
        Returns:
            None: Displays interactive Plotly stacked bar chart
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Analyze dwell time patterns
        plotter.stackedbars_dwelltime_state_interactive(
            bin_size=5,    # 5 time unit bins
            num_bins=15    # 15 bins total
        )
        ```
        
        Features:
        - Interactive hover information
        - Stacked bar visualization
        - Customizable bin sizes
        - State-specific analysis
        - Professional styling
        """
        if self.__data_error__():
            return

        # Define bins for dwell times
        bins = np.arange(0, bin_size * num_bins, bin_size)
        bin_labels = [f"{int(b)}-{int(b + bin_size)} step_time_units" for b in bins[:-1]]

        # Create a figure with subplots for each state
        states = self.data['state_name'].unique()
        num_states = len(states)
        fig = go.Figure()

        for state in states:
            state_data = self.data[self.data['state_name'] == state]
            state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

            # Group by dwell time bins and going to state
            grouped = state_data.groupby(['dwell_time_bin', 'going_to_state'], observed=True).size().unstack(fill_value=0)

            for going_to_state in grouped.columns:
                fig.add_trace(go.Bar(
                    x=grouped.index,
                    y=grouped[going_to_state],
                    name=f"{state} to {going_to_state}",
                    text=grouped[going_to_state],
                    textposition='auto'
                ))

        fig.update_layout(
            barmode='stack',
            title=f"Stacked Bar Charts of Dwell Times by State - {self.__class__.__name__} stackedbars_dwelltime_state_interactive() ",
            xaxis_title="Dwell Time Bins",
            yaxis_title="Count",
            legend_title="State Transitions",
            # height=500 + 50 * num_states,
        )
        fig.update_layout(margin=dict(l=5,
                                r=25,
                                b=100,
                                t=50,
                                pad=20),
                  paper_bgcolor="LightSteelBlue")
        fig.show()

    def subplot_custom_transitions(self, transitions_dict=None):
        """
        Plot cumulative distribution of dwell times for custom state transitions.
        
        Creates subplots showing the cumulative distribution function (CDF) of
        dwell times for specific state transitions. This helps analyze the
        timing patterns of particular transition types.
        
        Mathematical Model:
            For each transition (state_i → state_j):
            - Extract dwell times: T_ij = {dwell_time for i→j transitions}
            - Sort times: T_sorted = sort(T_ij)
            - Create CDF: CDF(t) = count(T_sorted ≤ t) / |T_ij|
            - Plot: x = T_sorted, y = CDF(T_sorted)
            
        Args:
            transitions_dict (dict): Dictionary mapping source states to target state lists.
                                   Format: {'source_state': ['target1', 'target2', ...]}
                                   If None, uses default transitions
                                   
        Returns:
            None: Displays matplotlib subplot figure
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Custom transition analysis
        transitions = {
            'Latent': ['Active', 'Cleared'],
            'Active': ['Treatment', 'Death']
        }
        plotter.subplot_custom_transitions(transitions)
        ```
        
        Features:
        - Custom transition specification
        - Cumulative distribution plots
        - Multiple subplot layout
        - Professional styling with legends
        """
        if self.__data_error__():
            return

        if transitions_dict is None:
            transitions_dict = {
                '-1.0.None': ['0.0.Latent Slow', '1.0.Latent Fast']
            }

        # Create subplots
        num_plots = len(transitions_dict)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

        if num_plots == 1:
            axes = [axes]

        for ax, (state_name, going_to_states) in zip(axes, transitions_dict.items()):
            for transition in going_to_states:
                data = self.data[
                    (self.data['state_name'] == state_name) &
                    (self.data['going_to_state'] == transition)
                ]['dwell_time']
                ax.plot(np.sort(data), np.linspace(0, 1, len(data)), label=f"{state_name} -> {transition}")
            ax.set_title(f"Transitions from {state_name}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Cumulative Distribution")
            ax.legend()
        plt.annotate('DwtPlotter.plot_state_transition_lengths_custom()', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()

    def stackedbars_subplots_state_transitions(self, bin_size=1, num_bins=50):
        """
        Create subplot stacked bar charts for state transitions by dwell time.
        
        Generates a multi-panel visualization with each subplot showing the dwell
        time distribution for a specific state. Each subplot contains stacked bars
        representing transitions to different target states within dwell time bins.
        
        Mathematical Model:
            For each state i:
            - Create dwell time bins: [0, bin_size, 2*bin_size, ..., num_bins*bin_size)
            - For each bin [b, b+bin_size):
                - Filter transitions: T where state=i and dwell_time ∈ [b, b+bin_size)
                - Count by target: N(state_i → state_j) in bin
                - Create stacked bar: height = Σ(N) for all target states j
                
        Args:
            bin_size (int): Size of each dwell time bin in time units. Default 1
            num_bins (int): Number of bins to create. Default 50
                           
        Returns:
            None: Displays matplotlib subplot figure
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create detailed dwell time analysis
        plotter.stackedbars_subplots_state_transitions(
            bin_size=2,    # 2 time unit bins
            num_bins=25    # 25 bins total
        )
        ```
        
        Features:
        - Multi-panel layout (4 columns)
        - State-specific analysis
        - Stacked bar visualization
        - Color-coded transitions
        - Professional styling with legends
        """
        if self.__data_error__():
            return

        # Define bins for dwell times
        bins = np.arange(0, bin_size*num_bins, bin_size)
        bin_labels = [f"{int(b)}-{int(b+bin_size)}" for b in bins[:-1]]

        # Create a figure with subplots for each state
        states = self.data['state_name'].unique()
        num_states = len(states)
        num_cols = 4
        num_rows = (num_states + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), sharex=True)

        axes = axes.flatten()
        fig.suptitle(f'State Transitions by Dwell Time Bins)', fontsize=16)
        for ax, state in zip(axes, states):
            state_data = self.data[self.data['state_name'] == state].copy()
            state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

            # Group by dwell time bins and going to state
            grouped = state_data.groupby(['dwell_time_bin', 'going_to_state'], observed=True).size().unstack(fill_value=0)

            # Check if there's any data to plot
            if grouped.empty or grouped.shape[1] == 0:
                ax.text(0.5, 0.5, f'No data for {state}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'State: {state}')
                continue

            # Ensure column order matches color mapping
            state_colors, cmap = Utils.colors()
            matching_colors = [state_colors[state] for state in grouped.columns if state in state_colors]
            if matching_colors:
                cmap = mcolors.ListedColormap(matching_colors)
            else:
                cmap = 'tab10'  # fallback colormap

            # Plot stacked bar chart
            try:
                grouped.plot(kind='bar', stacked=True, ax=ax, colormap=cmap) #'tab20')
            except TypeError as e:
                # Handle case where there's no numeric data to plot
                ax.text(0.5, 0.5, f'No numeric data for {state}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'State: {state}')
                continue
            ax.set_title(f'State: {state}')
            ax.set_xlabel('Dwell Time Bins')
            ax.set_ylabel('Count')
            ax.legend(title='Going to State')

        # Remove any empty subplots
        for i in range(num_states, len(axes)):
            fig.delaxes(axes[i])
        plt.annotate('DwtPlotter.plot_binned_stacked_bars_state_transitions()', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()

    def histogram_with_kde(self, subtitle=""):
        """
        Create histograms with kernel density estimation for dwell time distributions.
        
        This method shows how long agents spend in each TB state, providing
        insights into the timing patterns of disease progression.
        
        .. image:: ../_static/histogram_kde_example.png
            :width: 500px
            :alt: Histogram with KDE showing dwell time distributions
            :align: center
        
        The plot above demonstrates:
        - **Histogram bars**: Frequency distribution of dwell times
        - **KDE curve**: Smooth probability density estimation
        - **Multi-panel layout**: One subplot per TB state
        - **Automatic scaling**: Optimized bin sizes and ranges
        
        Mathematical Model:
            For each state i:
            - Extract dwell times: T_i = {dwell_time for state i}
            - Create histogram: H_i(bin) = count(T_i in bin)
            - Calculate KDE: KDE_i(t) = Σ(K(t-t_j, h)) / (n*h)
                where K is kernel function, h is bandwidth, n = |T_i|
            - Plot: histogram + KDE curve
                
        Args:
            subtitle (str): Additional subtitle for the plot
                           
        Returns:
            None: Displays matplotlib subplot figure
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Analyze dwell time distributions
        plotter.histogram_with_kde(subtitle="Distribution Analysis")
        ```
        
        Features:
        - Multi-panel layout (4 columns)
        - Histogram + KDE visualization
        - State-specific analysis
        - Automatic bin sizing
        - Professional styling with legends
        """
        if self.__data_error__():
            return

        # Create DataFrame
        df = self.data

        # Create a figure with subplots for each state
        states = df['state_name'].unique()
        num_states = len(states)
        num_cols = 4
        num_rows = (num_states + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), sharex=False)

        axes = axes.flatten()
        fig.suptitle(f'State Transitions by Dwell Time Bins {subtitle}', fontsize=16)

        for ax, state in zip(axes, states):
            state_data = df[df['state_name'] == state].copy()

            # Automatically define the number of bins and bin size based on the data
            max_dwell_time = state_data['dwell_time'].max()
            bin_size = max(1, max_dwell_time // 15)  # Ensure at least 15 bins
            bins = np.arange(0, max_dwell_time + bin_size, bin_size)
            bin_labels = [f"{int(b)}-{int(b+bin_size)} step_time_units" for b in bins[:-1]]

            state_data['dwell_time_bin'] = pd.cut(
            state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True,
            )
            
            # Check if we have enough data points for KDE
            # KDE requires at least 2 unique values per group
            has_enough_data = True
            
            # Check each group defined by 'going_to_state'
            for group_name, group_data in state_data.groupby('going_to_state'):
                if group_data['dwell_time'].nunique() < 2:
                    has_enough_data = False
                    break
            
            # Only use KDE if we have enough data points
            kde_param = has_enough_data
            
            sns.histplot(data=state_data, 
                 x='dwell_time', 
                 bins=bins,
                 hue='going_to_state', 
                 kde=kde_param, 
                 palette='tab10',
                 multiple='stack',
                 legend=True,
                 ax=ax)

            ax.set_title(f'State: {state}')
            ax.set_xlabel('Dwell Time Bins')
            ax.set_ylabel('Count')
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(title='Going to State', loc='upper right')

        # Remove any unused subplots
        for i in range(num_states, len(axes)):
            fig.delaxes(axes[i])
        plt.annotate('DwtPlotter.histogram_with_kde()', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return
    
    def graph_state_transitions(self, states=None, subtitle="", layout=None, curved_ratio=0.1, colormap='Paired', onlymodel=True):
        """
        Create a network graph visualization of state transitions.
        
        Generates a directed graph showing the relationships between states,
        with edges representing transitions and annotations showing mean/mode
        dwell times and agent counts for each transition.
        
        Mathematical Model:
            For each transition (state_i → state_j): 
            - Calculate statistics: mean_ij, mode_ij, count_ij
            - Create edge: node_i → node_j
            - Annotate edge: "Mean: mean_ij, Mode: mode_ij, Agents: count_ij"
            - Node size ∝ total transitions (in + out)
            - Edge thickness ∝ transition count
            
        Args:
            states (list, optional): Specific states to include. If None, includes all states
            subtitle (str): Additional subtitle for the plot
            layout (int, optional): Layout algorithm (0-9). Default uses spring layout
            curved_ratio (float): Edge curvature factor. Default 0.1
            colormap (str): Matplotlib colormap name. Default 'Paired'
            onlymodel (bool): If True, exclude certain non-model states
                           
        Returns:
            None: Displays matplotlib network graph
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create state transition network
        plotter.graph_state_transitions(
            states=['Latent', 'Active', 'Treatment', 'Recovered'],
            subtitle="Core TB States",
            layout=1,  # Circular layout
            colormap='viridis'
        )
        ```
        
        Features:
        - Directed graph visualization
        - Statistical annotations on edges
        - Multiple layout algorithms
        - Color-coded nodes and edges
        - Professional styling
        """
        if self.__data_error__():  return
        df = self.data
        if states is not None: 
            df = df[df['going_to_state'].isin(states)]
            df = df[df['state_name'].isin(states)]

        if onlymodel: df = df[~df['going_to_state_id'].isin([-3.0, -2.0])]

        # Calculate mean, mode, and count for each state transition
        transitions = df.groupby(['state_name', 'going_to_state'])['dwell_time']   #Dweell time 

        stats_df = transitions.agg([
            'mean',
            lambda x: stats.mode(x, keepdims=True).mode[0] if len(x) > 0 else np.nan,
            'count'
        ]).reset_index()

        stats_df.columns = ['state_name', 'going_to_state', 'mean', 'mode', 'count']

        # Create a directed graph
        G = nx.DiGraph()
        # increase the size of the figure
        plt.figure(figsize=(15, 10), facecolor='black')
        
        # Add edges with mean and mode annotations
        for _, row in stats_df.iterrows():
            from_state = row['state_name']
            to_state = row['going_to_state']
            mean_dwell = round(row['mean'], 2) if not pd.isna(row['mean']) else "N/A"
            mode_dwell = round(row['mode'], 2) if not pd.isna(row['mode']) else "N/A"
            num_agents = row['count']

            G.add_edge(from_state, to_state,
                label=f"Mean: {mean_dwell}\nMo: {mode_dwell}\nAgents: {num_agents}")

        # Choose layout based on parameter
        if layout is not None:
            pos = self.__select_graph_pos__(G, layout, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)  # Default spring layout
        # Enhanced color handling
        colors = plt.colormaps.get_cmap(colormap)
        node_colors = [colors(i / max(1, len(G.nodes) - 1)) for i in range(len(G.nodes))]
        edge_colors = [colors(i / max(1, len(G.edges) - 1)) for i in range(len(G.edges))]
        edge_labels = nx.get_edge_attributes(G, 'label')

        nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors= "lightgray", alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1, alpha=0.7, arrowstyle="-|>", arrowsize=30, edge_color=edge_colors)
        nx.draw_networkx_labels(G, pos, font_size=11, font_color="black", font_weight="bold")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.title(f"State Transition Graph with Dwell Times: {subtitle}")
        plt.annotate('DwtPlotter.graph_state_transitions()', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=12)
        plt.show()
        return

    def graph_state_transitions_curved(self, states=None, subtitle="", layout=None, curved_ratio=0.09, colormap='Paired', onlymodel=True, graphseed=42):
        """
        Create a curved network graph with edge thickness proportional to agent count.
        
        This method generates a directed graph visualization where nodes represent
        TB states and edges show transitions with statistical annotations.
        
        .. image:: ../_static/network_graph_example.png
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
            - Calculate agent count: N_ij = count(agents with transition i→j)
            - Create edge: node_i → node_j
            - Edge thickness: thickness_ij = 1 + (4 * N_ij / max_count)
            - Annotate edge: "state_i → state_j, μ:mean_ij, Mo:mode_ij, Agents:N_ij"
            
        Args:
            states (list, optional): Specific states to include
            subtitle (str): Additional subtitle for the plot
            layout (int, optional): Layout algorithm (0-9)
            curved_ratio (float): Edge curvature factor. Default 0.09
            colormap (str): Matplotlib colormap name. Default 'Paired'
            onlymodel (bool): If True, exclude certain non-model states
            graphseed (int): Random seed for layout consistency. Default 42
                           
        Returns:
            None: Displays matplotlib network graph
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create curved transition network
        plotter.graph_state_transitions_curved(
            subtitle="Agent-weighted Transitions",
            curved_ratio=0.15,  # More curved edges
            colormap='plasma',
            graphseed=123
        )
        ```
        
        Features:
        - Curved edge visualization
        - Edge thickness proportional to agent count
        - Statistical annotations on edges
        - Color-coded nodes and edges
        - Professional styling with consistent layouts
        """
        if self.__data_error__():  
            return
        df = self.data
        if states is not None: 
            df = df[df['state_name'].isin(states)]
        
        if onlymodel: 
            df = df[~df['going_to_state_id'].isin([-3.0, -2.0])]

        # Compute transition statistics: Mean, Mode, Count (Agent Count)
        transitions = df.groupby(['state_name', 'going_to_state'])['dwell_time']
        
        stats_df = transitions.agg([
            'mean',
            lambda x: stats.mode(x, keepdims=True).mode[0] if len(x) > 0 else np.nan,
            'count'
        ]).reset_index()
        
        stats_df.columns = ['state_name', 'going_to_state', 'mean', 'mode', 'count']
        plt.figure(figsize=(15, 10), facecolor='black')
        # Create directed graph
        G = nx.DiGraph()

        # Find max agent count for scaling edge thickness
        max_agents = stats_df['count'].max() if not stats_df['count'].isna().all() else 1

        # Add edges with thickness proportional to agent count
        for _, row in stats_df.iterrows():
            from_state = row['state_name']
            to_state = row['going_to_state']
            mean_dwell = round(row['mean'], 2) if not pd.isna(row['mean']) else "N/A"
            mode_dwell = round(row['mode'], 2) if not pd.isna(row['mode']) else "N/A"
            num_agents = row['count']

            edge_thickness = 1 + (4 * num_agents / max_agents)  # Scale edge width
            fs_id = f"{from_state.split('.')[0]} → {to_state.split('.')[0]}\n"
            # Add edge to the graph
            G.add_edge(from_state, to_state,
                label=f"{fs_id},  μ:{mean_dwell},  Mo: {mode_dwell}\nAgents:{num_agents}",
                weight=edge_thickness)

        # Choose layout based on parameter
        if layout is not None:
            pos = self.__select_graph_pos__(G, layout, seed=graphseed)
        else:
            pos = nx.spring_layout(G, seed=graphseed)  # Default spring layout  

        # Enhanced color handling
        colors = plt.colormaps.get_cmap(colormap)
        node_colors = [colors(i / max(1, len(G.nodes) - 1)) for i in range(len(G.nodes))]
        edge_colors = [colors(i / max(1, len(G.edges) - 1)) for i in range(len(G.edges))]

        # Get edge attributes
        edge_labels = nx.get_edge_attributes(G, 'label')
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]

        # Draw the graph
        plt.figure(facecolor='black')
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, edgecolors="lightgray")
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.8, arrowstyle="-|>", arrowsize=30,
                            edge_color=edge_colors, connectionstyle=f'arc3,rad={curved_ratio}')
        nx.draw_networkx_labels(G, pos, font_size=11, font_color="black", font_weight="bold")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        # Display the graph
        plt.title(f"State Transition Graph - By Agents Count: {subtitle}", color='white')
        plt.annotate(text='DwtPlotter.graph_state_transitions_curved()', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return

    def graph_state_transitions_enhanced(self, states=None, subtitle="", layout=None, colormap='viridis', 
                                       onlymodel=True, graphseed=42, figsize=(16, 12), 
                                       node_size_scale=1000, edge_width_scale=8, font_size=10):
        """
        Create an enhanced network graph visualization of state transitions with improved styling.
        
        This enhanced version provides:
        - Better color schemes and visual hierarchy
        - Clearer edge annotations with better formatting
        - Improved node sizing based on transition importance
        - Enhanced readability with better fonts and spacing
        - Professional styling suitable for publications
        
        Mathematical Model:
            For each transition (state_i → state_j):
            - Calculate statistics: mean_ij, mode_ij, count_ij
            - Node importance: importance_i = Σ(count_ij for all j) + Σ(count_ji for all i)
            - Node size: size_i = node_size_scale * (importance_i / max_importance)
            - Edge thickness: thickness_ij = edge_width_scale * (count_ij / max_count)
            - Color intensity: intensity_ij = count_ij / max_count
            
        Args:
            states (list, optional): Specific states to include. If None, includes all states
            subtitle (str): Additional subtitle for the plot
            layout (int, optional): Layout algorithm (0-9). Default uses spring layout
            colormap (str): Matplotlib colormap name. Default 'viridis'
            onlymodel (bool): If True, exclude certain non-model states
            graphseed (int): Random seed for layout consistency. Default 42
            figsize (tuple): Figure size (width, height). Default (16, 12)
            node_size_scale (int): Scaling factor for node sizes. Default 1000
            edge_width_scale (int): Scaling factor for edge widths. Default 8
            font_size (int): Base font size. Default 10
                           
        Returns:
            None: Displays matplotlib network graph
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create enhanced state transition network
        plotter.graph_state_transitions_enhanced(
            subtitle="Enhanced TB State Transitions",
            colormap='plasma',
            figsize=(18, 14),
            node_size_scale=1200,
            edge_width_scale=10
        )
        ```
        
        Features:
        - Enhanced directed graph visualization
        - Professional statistical annotations on edges
        - Node size proportional to transition importance
        - Edge thickness proportional to agent count
        - Color-coded nodes and edges with improved schemes
        - Publication-ready styling
        - Better spacing and typography
        """
        if self.__data_error__():  
            return
            
        df = self.data
        if states is not None: 
            df = df[df['going_to_state'].isin(states)]
            df = df[df['state_name'].isin(states)]
        
        if onlymodel: 
            df = df[~df['going_to_state_id'].isin([-3.0, -2.0])]

        # Compute transition statistics
        transitions = df.groupby(['state_name', 'going_to_state'])['dwell_time']
        
        stats_df = transitions.agg([
            'mean',
            lambda x: stats.mode(x, keepdims=True).mode[0] if len(x) > 0 else np.nan,
            'count'
        ]).reset_index()
        
        stats_df.columns = ['state_name', 'going_to_state', 'mean', 'mode', 'count']
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Calculate node importance (total transitions in + out)
        node_importance = {}
        for _, row in stats_df.iterrows():
            from_state = row['state_name']
            to_state = row['going_to_state']
            count = row['count']
            
            # Add to importance calculations
            node_importance[from_state] = node_importance.get(from_state, 0) + count
            node_importance[to_state] = node_importance.get(to_state, 0) + count
            
            # Add edge to graph
            G.add_edge(from_state, to_state, weight=count)
        
        # Find scaling factors
        max_importance = max(node_importance.values()) if node_importance else 1
        max_count = stats_df['count'].max() if not stats_df['count'].isna().all() else 1
        
        # Set up the figure with enhanced styling
        plt.figure(figsize=figsize, facecolor='white')
        
        # Choose layout
        if layout is not None:
            pos = self.__select_graph_pos__(G, layout, seed=graphseed)
        else:
            pos = nx.spring_layout(G, seed=graphseed, k=3, iterations=50)
        
        # Enhanced color scheme
        colors = plt.colormaps.get_cmap(colormap)
        node_colors = [colors(i / max(1, len(G.nodes) - 1)) for i in range(len(G.nodes))]
        
        # Calculate node sizes based on importance
        node_sizes = [node_size_scale * (node_importance.get(node, 0) / max_importance) for node in G.nodes]
        node_sizes = [max(size, 300) for size in node_sizes]  # Minimum size
        
        # Calculate edge widths based on agent count
        edge_weights = [edge_width_scale * (G[u][v]['weight'] / max_count) for u, v in G.edges]
        edge_weights = [max(weight, 0.5) for weight in edge_weights]  # Minimum width
        
        # Edge colors based on transition intensity
        edge_colors = []
        for u, v in G.edges:
            intensity = G[u][v]['weight'] / max_count
            edge_colors.append(colors(intensity))
        
        # Draw nodes with enhanced styling
        nx.draw_networkx_nodes(
            G, pos, 
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors='black',
            linewidths=2,
            alpha=0.9
        )
        
        # Draw edges with enhanced styling
        nx.draw_networkx_edges(
            G, pos,
            width=edge_weights,
            edge_color=edge_colors,
            alpha=0.8,
            arrowstyle='-|>',
            arrowsize=25,
            connectionstyle='arc3,rad=0.1',
            min_source_margin=15,
            min_target_margin=15
        )
        
        # Enhanced node labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=font_size,
            font_color='black',
            font_weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        # Enhanced edge labels with better formatting
        edge_labels = {}
        for _, row in stats_df.iterrows():
            from_state = row['state_name']
            to_state = row['going_to_state']
            mean_dwell = row['mean']
            mode_dwell = row['mode']
            num_agents = row['count']
            
            # Format the label with better spacing and alignment
            if pd.isna(mean_dwell) or pd.isna(mode_dwell):
                label = f"Agents: {num_agents}"
            else:
                label = f"μ: {mean_dwell:.1f}\nMo: {mode_dwell:.1f}\nN: {num_agents}"
            
            edge_labels[(from_state, to_state)] = label
        
        # Draw edge labels with enhanced styling
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=font_size-2,
            font_color='darkblue',
            font_weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7, edgecolor='blue')
        )
        
        # Enhanced title and annotations
        plt.title(
            f"Enhanced State Transition Network\n{subtitle}",
            fontsize=font_size+4,
            fontweight='bold',
            pad=20,
            color='darkblue'
        )
        
        # Add informative subtitle
        total_transitions = stats_df['count'].sum()
        unique_states = len(G.nodes)
        plt.figtext(
            0.5, 0.02,
            f'Total Transitions: {total_transitions} | States: {unique_states} | '
            f'Node size ∝ transition importance | Edge thickness ∝ agent count',
            ha='center',
            fontsize=font_size-1,
            style='italic',
            color='gray'
        )
        
        # Add method attribution
        plt.figtext(
            0.5, -0.02,
            'DwtPlotter.graph_state_transitions_enhanced()',
            ha='center',
            fontsize=font_size-2,
            style='italic',
            color='lightgray'
        )
        
        # Remove axes for cleaner look
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return

    def plot_dwell_time_validation(self):
        """
        Create a histogram validation plot for dwell time distributions.
        
        Generates a single histogram showing the distribution of dwell times
        across all states, with different colors for each state. This provides
        a quick validation of dwell time data quality and distribution patterns.
        
        Mathematical Model:
            For each state i:
            - Extract dwell times: T_i = {dwell_time for state i}
            - Create histogram: H_i(bin) = count(T_i in bin)
            - Plot: overlay histograms for all states
            - Color coding: each state gets unique color
            
        Returns:
            None: Displays matplotlib histogram
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Validate dwell time distributions
        plotter.plot_dwell_time_validation()
        ```
        
        Features:
        - Overlaid histograms for all states
        - Color-coded state identification
        - Automatic bin sizing
        - Professional styling with legends
        - Data quality validation
        """
        # Plot the results of the dwell time validation. 
        if self.__data_error__():
            return
        fig, ax = plt.subplots()
        data = self.data
        model_states = data['state_name'].unique()
        plt.figure(figsize=(15, 10), facecolor='black')
        for state in model_states:
            dwell_times = data[data['state_name'] == state]['dwell_time']
            if dwell_times.empty:
                continue
            state_label = state
            ax.hist(dwell_times, bins=50, alpha=0.5, label=f'{state_label}')
            ax.hist(dwell_times, bins=50, alpha=0.5, label=f'{state}')
        ax.set_xlabel('Dwell Time')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.show()
        return
    
    def plot_dwell_time_validation_interactive(self):
        """
        Create an interactive histogram validation plot for dwell time distributions.
        
        Generates an interactive Plotly histogram showing the distribution of
        dwell times across all states, with hover information and interactive
        features for detailed analysis.
        
        Mathematical Model:
            For each state i:
            - Extract dwell times: T_i = {dwell_time for state i}
            - Create histogram: H_i(bin) = count(T_i in bin)
            - Plot: interactive histogram with state overlay
            - Hover information: exact counts and percentages
            
        Returns:
            None: Displays interactive Plotly histogram
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Interactive dwell time validation
        plotter.plot_dwell_time_validation_interactive()
        ```
        
        Features:
        - Interactive hover information
        - Overlaid histograms for all states
        - Color-coded state identification
        - Automatic bin sizing
        - Professional styling with legends
        """
        if self.__data_error__():
            return
        
        data = self.data
        fig = px.histogram(data, x='dwell_time', color='state_name', 
                            nbins=50, barmode='overlay', 
                            labels={'dwell_time': 'Dwell Time', 'state_name': 'State'},
                            title='Dwell Time Validation')
        fig.update_layout(bargap=0.1)
        fig.show()
        return
    
    def plot_kaplan_meier(self, dwell_time_col, event_observed_col=None):
        """
        Create a Kaplan-Meier survival curve for dwell time analysis.
        
        Generates a survival curve showing the probability of remaining in a
        state over time. This is useful for analyzing the "survival" of agents
        in different states and comparing transition timing patterns.
        
        Mathematical Model:
            For dwell times T = {t_1, t_2, ..., t_n}:
            - Sort times: T_sorted = sort(T)
            - Calculate survival function: S(t) = ∏(1 - d_i/n_i) for t_i ≤ t
                where d_i = events at time t_i, n_i = at risk at time t_i
            - Plot: x = time, y = S(t)
            
        Args:
            dwell_time_col (str): Column name containing dwell times
            event_observed_col (str, optional): Column indicating event observation (1=observed, 0=censored)
                If None, assumes all events are observed
                           
        Returns:
            None: Displays matplotlib survival curve
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Create survival curve for dwell times
        plotter.plot_kaplan_meier(
            dwell_time_col='dwell_time',
            event_observed_col='event_observed'
        )
        ```
        
        Features:
        - Kaplan-Meier survival analysis
        - Confidence intervals (if available)
        - Professional styling with grid
        - Clear axis labels and title
        - Statistical validation
        """
        if self.__data_error__():
            return
        
        data = self.data
        # Prepare the data
        durations = data[dwell_time_col]
        event_observed = data[event_observed_col] if event_observed_col else [1] * len(data)

        # Initialize Kaplan-Meier fitter
        kmf = KaplanMeierFitter()

        # Fit the data
        kmf.fit(durations, event_observed=event_observed)

        # Plot the survival function
        plt.figure(figsize=(10, 6))
        kmf.plot_survival_function()
        plt.title("TBSim Kaplan-Meier Survival Curve", fontsize=16)
        plt.figtext(0.5, 0.01, "DwtPlotter.plot_kaplan_meier()", ha="center", fontsize=12)
        plt.xlabel(f"Time ({dwell_time_col})", fontsize=14)
        plt.ylabel("Survival Probability", fontsize=14)
        plt.grid(True)
        plt.show()

    def __generate_reinfection_data__(self, file_path=None, target_states=[], scenario=''): 
        """
        Generate reinfection analysis data from dwell time logs.
        
        Processes the dwell time data to identify and count reinfections for each
        agent. Creates a new dataset with infection counts and percentages for
        population-level reinfection analysis.
        
        Mathematical Model:
            For each agent i:
            - Find transitions to target states: T_i = {transitions where going_to_state ∈ target_states}
            - Count infections: infection_count_i = |T_i|
            - Calculate percentage: percent_i = infection_count_i / total_infections * 100
            - Keep maximum: max_infection_i = max(infection_count_i for all transitions of agent i)
            
        Args:
            file_path (str, optional): Path to data file. If None, uses self.data
            target_states (list): States that count as infections (e.g., [0.0, 1.0])
            scenario (str): Scenario name for output file naming
                           
        Returns:
            tuple: (reinfection_dataframe, total_infection_count)
            
        Example:
        ```python
        plotter = DwtPlotter(file_path='data.csv')
        
        # Generate reinfection data
        reinfection_df, total_count = plotter.__generate_reinfection_data__(
            target_states=[0.0, 1.0],  # Active TB states
            scenario="Baseline"
        )
        print(f"Total infections: {total_count}")
        ```
        
        Output File:
        - Creates CSV file: {scenario}_WithReinfection.csv
        - Contains columns: agent_id, infection_num, percent, age, etc.
        """
        if file_path is None:
            df = self.data
            reinfection_file_name = Utils.to_filename_friendly(scenario) + '_WithReinfection.csv'
        else:
            file_path = self.data_file
            df = self.__cleandata__(file_path)
            reinfection_file_name = file_path.replace(".csv", "_WithReinfection.csv")

        ignore_states = [-2.0, -3.0] # -2 = ever infected, -3 = non-TB death

        df = df[~df['going_to_state_id'].isin(ignore_states)]
        relevant_rows = df[df['going_to_state_id'].isin(target_states)]
        
        # identify all the cases that landed in the selected states:
        total_count = len(df[df['going_to_state_id'].isin(target_states)])

        # identify the rows where the going_to_state_id is greater than state
        # this is to identify the reinfections
        df = relevant_rows[relevant_rows['going_to_state_id'] < relevant_rows['state']].copy()
        df.loc[:, 'infection_num'] = df.groupby('agent_id').cumcount()+1
        df.loc[:, 'percent'] = df['infection_num'].astype(float)/total_count
        # keep only the maximum infection_num for each agent_id
        df = df.loc[df.groupby('agent_id')['infection_num'].idxmax()]

        df = df.sort_values(by=['agent_id', 'entry_time'])
        df.to_csv(reinfection_file_name, index=False)
 
        return df, total_count
        
    def __data_error__(self):
        """
        Check for data availability and validity.
        
        Validates that the dwell time data is available and contains the required
        columns for analysis. This is a helper method used by all plotting methods.
        
        Returns:
            bool: True if data is missing or invalid, False if data is valid
            
        Validation Checks:
        - Data is not None
        - Data is not empty
        - 'dwell_time' column exists
        """
        # data error handling - check if data is available

        if self.data is None or self.data.empty or 'dwell_time' not in self.data.columns:
            print("No dwell time data available to plot.")
            return True
        return False

    def __cleandata__(self, filename):
        """
        Clean and validate dwell time data from CSV file.
        
        Reads a CSV file containing dwell time data and performs data cleaning
        operations including type conversion, missing value handling, and
        data validation.
        
        Data Cleaning Process:
        1. Read CSV as strings to avoid import errors
        2. Convert numeric columns with error handling
        3. Remove rows with missing critical data
        4. Reset index for clean data structure
        
        Args:
            filename (str): Path to CSV file containing dwell time data
            
        Returns:
            pd.DataFrame: Cleaned and validated dwell time data
            
        Required Columns:
        - agent_id: Unique agent identifier
        - state: Current state identifier
        - entry_time: Time entering current state
        - exit_time: Time exiting current state
        - dwell_time: Time spent in state
        - state_name: Human-readable state name
        - going_to_state_id: Target state identifier
        - going_to_state: Target state name
        - age: Agent age at transition time
        
        Example:
        ```python
        plotter = DwtPlotter()
        clean_data = plotter.__cleandata__('raw_data.csv')
        print(f"Cleaned {len(clean_data)} records")
        ```
        """
        # Define column types as expected
        dtype_dict = {
            "agent_id": float,  # Using float initially to avoid casting errors
            "state": float,
            "entry_time": float,
            "exit_time": float,  # Keep as float first, will convert later
            "dwell_time": float,
            "state_name": str,
            "going_to_state_id": float,
            "going_to_state": float,
            "age": float
        }

        # Read CSV as raw text to avoid errors during import
        df = pd.read_csv(filename, dtype=str)  # Read everything as strings first
        df = df.dropna(subset=["agent_id"])

        # Convert numeric columns, coercing invalid values to NaN
        numeric_columns = ["agent_id", "state", "entry_time", "exit_time", "dwell_time", "going_to_state_id", "age"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert and replace invalid with NaN

        # Drop rows with NaN in any of the expected numeric columns
        df_cleaned = df.dropna(subset=numeric_columns)

        # # Convert integer columns after filtering invalid data
        # df_cleaned["agent_id"] = df_cleaned["agent_id"].astype(int)
        # df_cleaned["exit_time"] = df_cleaned["exit_time"].astype(int)

        # Reset index after dropping rows
        df_cleaned.reset_index(drop=True, inplace=True)
        return df_cleaned

    @staticmethod
    def __select_graph_pos__(G, layout=4, seed=42):
        """
        Select graph layout algorithm for network visualizations.
        
        Provides access to various NetworkX layout algorithms for positioning
        nodes in network graphs. Each layout offers different visual characteristics
        suitable for different types of network analysis.
        
        Args:
            G (nx.Graph): NetworkX graph object
            layout (int): Layout algorithm selection (0-9)
            seed (int): Random seed for reproducible layouts. Default 42
            
        Returns:
            dict: Node positions as {node: (x, y)} dictionary
            
        Layout Options:
        0: Spring layout (default) - Force-directed placement
        1: Circular layout - Nodes arranged in circle
        2: Spiral layout - Nodes arranged in spiral pattern
        3: Spectral layout - Based on graph Laplacian eigenvectors
        4: Shell layout - Nodes in concentric shells
        5: Kamada-Kawai layout - Force-directed with Kamada-Kawai algorithm
        6: Planar layout - For planar graphs only
        7: Random layout - Random positioning
        8: Circular layout (alternative) - Another circular arrangement
        9: Fruchterman-Reingold layout - Force-directed with F-R algorithm
        
        Example:
        ```python
        import networkx as nx
        G = nx.DiGraph()
        # Add edges...
        
        # Use circular layout
        pos = DwtPlotter.__select_graph_pos__(G, layout=1, seed=123)
        nx.draw(G, pos)
        ```
        """
        import networkx as nx
        
        if layout == 1: 
            return nx.circular_layout(G)
            return pos
        elif layout == 2: 
            return nx.spiral_layout(G)
            return pos
        elif layout == 3: 
            return nx.spectral_layout(G)
            return pos
        elif layout == 4: 
            return nx.shell_layout(G)
        elif layout == 5: 
            return nx.kamada_kawai_layout(G)
        elif layout == 6: 
            return nx.planar_layout(G)
        elif layout == 7: 
            return nx.random_layout(G)
        elif layout == 9: 
            return nx.fruchterman_reingold_layout(G)
        else: 
            return nx.spring_layout(G, seed=seed)

        return pos

class DwtPostProcessor(DwtPlotter):
    """
    Post-Processing Analyzer for Multiple Simulation Results
    
    Extends DwtPlotter to provide aggregation and analysis capabilities for
    multiple simulation runs. This class can combine results from multiple
    CSV files and perform comparative analysis across different scenarios.
    
    Key Features:
    - Aggregates multiple simulation results by file prefix
    - Handles agent ID conflicts across runs
    - Provides batch processing capabilities
    - Supports scenario comparison analysis
    
    Mathematical Model:
        For multiple files with prefix P:
        - Load files: F = {f_1, f_2, ..., f_n} where f_i matches pattern P*.csv
        - Adjust agent IDs: agent_id_i = agent_id_i + (file_index * 10000)
        - Combine data: combined_data = ∪(data_i for all f_i in F)
        - Preserve file associations through agent ID ranges
        
    Example Usage:
    ```python
    # Aggregate multiple baseline runs
    postproc = DwtPostProcessor(directory='results', prefix='Baseline')
    postproc.sankey_agents()
    postproc.histogram_with_kde()
    
    # Save combined results
    postproc.save_combined_dataframe('baseline_combined.csv')
    ```
    
    Attributes:
        directory (str): Directory containing simulation result files
        prefix (str): File prefix for identifying related simulations
        debug (bool): Enable debug output for troubleshooting
        data (pd.DataFrame): Combined data from all matching files
    """
    
    def __init__(self, directory='', prefix='', data=None, debug=False, **kwargs) :
        """
        Initialize the post-processor with simulation result aggregation.
        
        Sets up the post-processor to work with multiple simulation results,
        either by aggregating files from a directory or using pre-loaded data.
        
        Args:
            directory (str): Directory containing CSV result files
            prefix (str): File prefix for identifying related simulations
                         (e.g., "BaselineLSHTM" or "BaselineTBSim")
            data (pd.DataFrame, optional): Pre-loaded data. If None, loads from directory
            debug (bool): Enable debug output. Default False
            **kwargs: Additional arguments passed to DwtPlotter
            
        Example:
        ```python
        # Initialize with directory and prefix
        postproc = DwtPostProcessor(
            directory='results',
            prefix='BaselineLSHTM',
            debug=True
        )
        
        # Initialize with existing data
        df = pd.read_csv('combined_data.csv')
        postproc = DwtPostProcessor(data=df)
        ```
        
        File Pattern:
        - Searches for files matching: {directory}/{prefix}*.csv
        - Example: results/BaselineLSHTM-20240101120000.csv
        """
        self.directory = directory
        self.prefix = prefix
        self.debug = debug

        if data is None:
            data = self.aggregate_simulation_results(self.directory, self.prefix)
        DwtPlotter.__init__(self, data=data)
        
        return


    def aggregate_simulation_results(self, directory: str, prefix: str) -> pd.DataFrame:
        """
        Aggregate multiple CSV files into a single DataFrame.
        
        Combines all CSV files with the same prefix into a single dataset,
        handling agent ID conflicts by adding offsets to ensure uniqueness
        across different simulation runs.
        
        Mathematical Model:
            For files F = {f_1, f_2, ..., f_n}:
            - Load each file: data_i = read_csv(f_i)
            - Adjust agent IDs: agent_id_i = agent_id_i + (i * 10000)
            - Combine: combined_data = concat(data_1, data_2, ..., data_n)
            - Result: unique agent IDs across all runs
            
        Args:
            directory (str): Directory containing CSV files
            prefix (str): Common prefix for files to aggregate
                         (e.g., "BaselineLSHTM" or "BaselineTBSim")
                         
        Returns:
            pd.DataFrame: Combined data from all matching files
            
        Example:
        ```python
        postproc = DwtPostProcessor()
        
        # Aggregate baseline results
        combined_data = postproc.aggregate_simulation_results(
            directory='results',
            prefix='BaselineLSHTM'
        )
        print(f"Aggregated {len(combined_data)} records from multiple runs")
        ```
        
        File Handling:
        - Uses glob pattern: {directory}/{prefix}*.csv
        - Skips files that can't be read
        - Reports number of files found and processed
        - Handles missing directories gracefully
        """
        import glob

        file_pattern = os.path.join(directory, f"{prefix}*.csv")
        file_list = glob.glob(file_pattern)

        if not file_list:
            print(f"No files found matching pattern: {file_pattern}")
            return pd.DataFrame()
        else:
            print(f"Found {len(file_list)} files matching pattern: {file_pattern}")
            if self.debug: print("\n".join(file_list))

        data_frames = []
        for index, file in enumerate(file_list):
            try:
                # Read the file into a DataFrame
                df = pd.read_csv(file)
                
                # Add the file index to the 'agent_id'
                df['agent_id'] = df['agent_id'] + ((index+1) * 10000)    # Add 10000 to avoid overlap or agent_ids
                                
                # Append the DataFrame to the list
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if not data_frames:
            print(f"No valid CSV files to aggregate for prefix: {prefix}")
            return pd.DataFrame()

        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df


    def save_combined_dataframe(self, output_file):
        """
        Save the combined DataFrame to a CSV file.
        
        Exports the aggregated simulation results to a CSV file for
        further analysis or sharing with other tools.
        
        Args:
            output_file (str): Path to the output CSV file
            
        Returns:
            None
            
        Example:
        ```python
        postproc = DwtPostProcessor(directory='results', prefix='Baseline')
        
        # Save combined results
        postproc.save_combined_dataframe('baseline_combined.csv')
        print("Combined data saved successfully")
        ```
        
        Error Handling:
        - Checks for data availability before saving
        - Reports success or failure messages
        - Handles file system errors gracefully
        """
        if self.data is None or self.data.empty:
            print("No data available to save.")
            return

        try:
            self.data.to_csv(output_file, index=False)
            if self.debug: print(f"Combined DataFrame saved to {output_file}")
        except Exception as e:
            print(f"Error saving DataFrame to {output_file}: {e}")
            
class DwtAnalyzer(ss.Analyzer, DwtPlotter):
    """
    Dwell Time Analyzer for TB Simulation
    
    Records and analyzes dwell times during simulation execution. This analyzer
    tracks how long agents spend in different states and provides comprehensive
    analysis capabilities for understanding state transition patterns.
    
    .. image:: ../_static/sankey_diagram_example.png
        :width: 600px
        :alt: Example Sankey diagram showing TB state transitions
        :align: center
    
    The analyzer tracks how long agents spend in each TB state and
    provides multiple visualization types including Sankey diagrams,
    network graphs, histograms, and interactive charts.
    
    Key Features:
        - Real-time dwell time tracking during simulation
        - Automatic state change detection
        - Support for multiple state enumeration systems
        - Comprehensive data export and analysis
        - Multiple visualization types (Sankey, Network, Histogram, Interactive)
    
    Mathematical Model:
    For each agent i and state transition:
        - Entry time: t_entry = time when agent enters state
        - Exit time: t_exit = time when agent leaves state  
        - Dwell time: dwell_time = t_exit - t_entry
        - State tracking: state_i(t) = current state of agent i at time t
        
    Example Usage:
        ```python
        import starsim as ss
        from tbsim import TB
        from tbsim.analyzers import DwtAnalyzer
        
        # Create simulation with analyzer
        sim = ss.Sim(diseases=mtb.TB(), analyzers=DwtAnalyzer(scenario_name="Baseline"), pars=dict(dt = ss.days(7), start = ss.date('1940'), stop = ss.date('2010')))
        sim.run()
        # Access analyzer results
        analyzer = sim.analyzers[0]
        analyzer.sankey_agents()
        analyzer.plot_dwell_time_validation()
        analyzer.sankey_agents()
        ```
    
    Attributes:
        eSTATES (IntEnum): State enumeration system (e.g., TBS, TBSL)
        scenario_name (str): Name of the simulation scenario
        data (pd.DataFrame): Collected dwell time data
        __latest_sts_df__ (pd.DataFrame): Internal state tracking
    """
    
    def __init__(self, states_ennumerator=mtb.TBS, scenario_name=''):
        """
        Initialize the dwell time analyzer.
        
        Sets up the analyzer to track dwell times during simulation execution
        with state enumeration system selection.
        
        Args:
            states_ennumerator (IntEnum): State enumeration class (e.g., TBS, TBSL).
                                        Default mtb.TBS
            scenario_name (str): Name for the simulation scenario. Default ''
                                
        Example:
            ```python
            # Basic analyzer
            analyzer = DwtAnalyzer()
            
            # Analyzer with custom state enumeration
            from tb_acf import TBSL
            analyzer = DwtAnalyzer(states_ennumerator=TBSL)
            ```
        
        State Tracking:
            - Tracks all state transitions during simulation
            - Handles births and deaths automatically
            - Records entry/exit times for each state
            - Calculates dwell times automatically
        """
        ss.Analyzer.__init__(self)
        self.eSTATES = states_ennumerator
        self.file_path = None
        self.scenario_name = scenario_name
        self.data = pd.DataFrame(columns=['agent_id', 'state', 'entry_time', 'exit_time', 'dwell_time', 'state_name', 'going_to_state_id','going_to_state'])
        self.__latest_sts_df__ = pd.DataFrame(columns=['agent_id', 'last_state', 'last_state_time'])      
        DwtPlotter.__init__(self, data=self.data)
        return
    
    def __initialize_dataframes__(self):
        """
        Initialize internal data structures for state tracking.
        
        Sets up the internal DataFrame that tracks the current state of each
        agent. This method is called automatically during the first simulation
        step to prepare the tracking system.
        
        Mathematical Model:
            For each agent i:
            - Initialize last_state_i = -1 (default state)
            - Initialize last_state_time_i = 0 (simulation start time)
            - Create tracking record: (agent_id_i, last_state_i, last_state_time_i)
            
        Implementation Details:
            - Creates DataFrame with columns: agent_id, last_state, last_state_time
            - Initializes all agents to state -1 (default/susceptible)
            - Sets all entry times to 0 (simulation start)
        """
        # Initialize the latest state dataframe
        # NOTE: This module assumes the default state is '-1'
        agent_ids = self.sim.people.auids.copy()
        population = len(agent_ids)
        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'last_state': np.full(population, -1.0),
            'last_state_time': np.zeros(population)
        })
        return
    
    def step(self):
        """
        Execute one time step of dwell time analysis.
        
        Called at each simulation time step to track state changes and record
        dwell times. This method handles state transitions, births, and deaths
        automatically.
        
        Step Process:
            1. Initialize tracking on first step (if needed)
            2. Check for new births and add to tracking
            3. Update state change data for existing agents
            4. Record natural deaths for deceased agents
        
        Mathematical Model:
            For each agent i at time t:
            - Current state: state_i(t) = TB.state[agent_i]
            - Previous state: last_state_i(t-1) = tracked state
            - If state_i(t) ≠ last_state_i(t-1):
                - Record dwell time: dwell_time = t - last_state_time_i
                - Update tracking: last_state_i(t) = state_i(t), last_state_time_i(t) = t
                
        Example:
        ```python
        # The step method is called automatically during simulation
        sim = ss.Sim(diseases=mtb.TB(), analyzers=DwtAnalyzer(scenario_name="Baseline"), pars=dict(dt = ss.days(7), start = ss.date('1940'), stop = ss.date('2010')))
        sim.run()  # step() is called internally at each time step
        ```
        """
        if not self.sim.ti: self.__initialize_dataframes__()
        self.__check_for_new_borns__()       
        self.__update_state_change_data__()
        self.__record_natural_deaths__()
        return
    
    def __update_state_change_data__(self):
        """
        Update dwell time data when agents change states.
        
        Detects state changes for all agents and records the dwell times
        for the previous state. This is the core method for tracking
        state transitions and calculating dwell times.
        
        Mathematical Model:
            For each agent i:
            - Current state: current_state_i = TB.state[agent_i]
            - Previous state: last_state_i = tracked last state
            - If current_state_i ≠ last_state_i:
                - Dwell time: dwell_time_i = current_time - last_state_time_i
                - Record transition: (agent_i, last_state_i, dwell_time_i, current_state_i)
                - Update tracking: last_state_i = current_state_i, last_state_time_i = current_time
                
        Implementation Details:
        - Compares current TB state with tracked last state
        - Calculates dwell times for state transitions
        - Updates internal tracking DataFrame
        - Handles only alive agents (filters by auids)
        """
        # Get the current state of the agents
        ti = self.ti
        tb = self.sim.diseases.tb
        uids = self.sim.people.auids.copy()  # People Alive

        # Filter rows in __latest_sts_df__ for the relevant agents (alive)
        relevant_rows = self.__latest_sts_df__[self.__latest_sts_df__['agent_id'].isin(uids)]

        # Identify agents whose last recorded state is different from the current state
        different_state_mask = relevant_rows['last_state'].values != tb.state[ss.uids(relevant_rows['agent_id'].values)]
        uids = ss.uids(relevant_rows['agent_id'].values[different_state_mask])

        # Log dwell times
        self._log_dwell_time(
            agent_ids=uids,
            states=relevant_rows['last_state'].values[different_state_mask],
            entry_times=relevant_rows['last_state_time'].values[different_state_mask],
            exit_times=np.full(len(uids), ti),
            going_to_state_ids=tb.state[uids].copy(),
            age=self.sim.people.age[uids].copy()
        )

        # Update the latest state dataframe with the new state
        self.__latest_sts_df__.loc[self.__latest_sts_df__['agent_id'].isin(uids), 'last_state'] = tb.state[uids]
        self.__latest_sts_df__.loc[self.__latest_sts_df__['agent_id'].isin(uids), 'last_state_time'] = ti
        return

    def __record_natural_deaths__(self):
        """
        Record dwell times for agents who died from natural causes.
        
        Identifies agents who died from non-TB causes and records their
        final state dwell time. This ensures complete tracking of all
        agent lifecycles.
        
        Mathematical Model:
            For each dead agent i:
            - If agent_i ∈ dead_agents and last_state_i < 0 (non-TB state):
                - Final dwell time: dwell_time_i = current_time - last_state_time_i
                - Record transition: (agent_i, last_state_i, dwell_time_i, -3.0)
                where -3.0 represents natural death
                
        Implementation Details:
        - Checks for agents in dead population
        - Filters for non-TB states (state < 0)
        - Avoids duplicate recording
        - Records final dwell time before death
        """
        # Get the current state of the agents
        ti = self.ti
        dead_uids = ss.uids(self.sim.people.dead)
        if not dead_uids.all():
            return
        # Filter rows in __latest_sts_df__ for the relevant agents
        relevant_rows = self.__latest_sts_df__[self.__latest_sts_df__['agent_id'].isin(dead_uids) & (self.__latest_sts_df__['last_state'] < 0)]

        # identify only those ones that are not already recorded
        relevant_rows = relevant_rows[~relevant_rows['agent_id'].isin(ss.uids(self.data['agent_id']))]
        if not relevant_rows.empty:
            self._log_dwell_time(
                agent_ids=relevant_rows['agent_id'].values,
                states=relevant_rows['last_state'].values,
                entry_times=relevant_rows['last_state_time'].values,
                exit_times=np.full(len(relevant_rows), ti),
                going_to_state_ids=np.full(len(relevant_rows), -3.0) , # State -3.0 is to represent Natural Death
                age= self.sim.people.age[ss.uids(relevant_rows['agent_id'].values)]
            )
            
    def __check_for_new_borns__(self):
        """
        Add new agents (births) to the state tracking system.
        
        Detects when new agents are added to the simulation (births) and
        initializes their state tracking. This ensures all agents are
        properly tracked throughout their lifecycle.
        
        Mathematical Model:
            For new agents N = {new_agent_1, new_agent_2, ...}:
            - Initialize: last_state_i = -1 (default state)
            - Initialize: last_state_time_i = current_time
            - Add to tracking: (agent_id_i, last_state_i, last_state_time_i)
            
        Implementation Details:
        - Compares current population size with tracking DataFrame size
        - Identifies new agent IDs
        - Initializes new agents to default state (-1)
        - Handles multiple births in single time step
        """
        # check if the number of agents has changed
        if len(self.sim.people.auids) != len(self.__latest_sts_df__):
            #identify which agent ids are new and add them to the __latest_sts_df__
            new_agent_ids = list(set(self.sim.people.auids) - set(self.__latest_sts_df__.agent_id))
            new_logs = pd.DataFrame({
                'agent_id': new_agent_ids,
                'last_state': np.full(len(new_agent_ids), -1.0),
                'last_state_time': np.zeros(len(new_agent_ids))
            })
            self.__latest_sts_df__ = pd.concat([self.__latest_sts_df__, new_logs], ignore_index=True) # Add new agents to the __latest_sts_df__ - more likely new borns
    

        # check if the number of agents has changed
        if len(self.sim.people.auids) != len(self.__latest_sts_df__):
            #identify which agent ids are new and add them to the __latest_sts_df__
            new_agent_ids = list(set(self.sim.people.auids) - set(self.__latest_sts_df__.agent_id))
            new_logs = pd.DataFrame({
                'agent_id': new_agent_ids,
                'last_state': np.full(len(new_agent_ids), -1.0),      # Never infected
                'last_state_time': np.zeros(len(new_agent_ids))
            })
            if not new_logs.empty:
                self.__latest_sts_df__ = pd.concat([self.__latest_sts_df__, new_logs.loc[:, ~new_logs.isna().all()]], 
                                            ignore_index=True, copy=False)
        return

    def finalize(self):
        """
        Finalize the dwell time analysis and save results.
        
        Called at the end of simulation to complete the analysis, process
        final state data, and save results to files. This method handles
        agents who never changed states and prepares data for analysis.
        
        Finalization Process:
        1. Record agents who never changed states (never infected)
        2. Detect and use appropriate state enumeration system
        3. Map state IDs to human-readable names
        4. Save results to CSV and metadata files
        
        Mathematical Model:
            For agents who never changed states:
            - Agent i with last_state_i = -1 and last_state_time_i = 0:
                - Record: (agent_i, -1, 0, simulation_end_time, -2.0)
                where -2.0 represents "never infected"
                
        Example:
        ```python
        # finalize() is called automatically at the end of simulation
        sim.run()  # finalize() is called internally
        
        # Access results after finalization
        analyzer = sim.analyzers[0]
        print(f"Recorded {len(analyzer.data)} state transitions")
        ```
        
        Output Files:
        - CSV file: {scenario_name}-{timestamp}.csv
        - Metadata file: {scenario_name}-{timestamp}.json
        """
        super().finalize()
        # record Never Infected (-2):
        # Identify agents with last_state == -1 (Not a single change of state was recorded)
        relevant_rows = self.__latest_sts_df__[(self.__latest_sts_df__['last_state'] == -1) & (self.__latest_sts_df__['last_state_time'] == 0.0)]
        if not relevant_rows.empty:
            self._log_dwell_time(
                agent_ids=relevant_rows['agent_id'].values,
                states=relevant_rows['last_state'].values,
                entry_times=relevant_rows['last_state_time'].values,
                exit_times=np.full(len(relevant_rows), self.sim.ti),
                going_to_state_ids=np.full(len(relevant_rows), -2.0) , # never infected
                age= self.sim.people.age[ss.uids(relevant_rows['agent_id'].values)].copy()
            )
        # TODO: this is a temporary solution to get the enum name, ideally this information 
        # should be available in the disease class
        # feel free to suggest an alternate way to do this:
        if 'LSHTM' in str(self.sim.diseases[0].__class__):
            print("====> Using model: str(self.sim.diseases[0].__class__)")
            import tb_acf as tbacf
            self.eSTATES = tbacf.TBSL

        # Create a dictionary to map state values to their names
        state_dict = {state.value: state.name.replace('_', ' ').title() for state in self.eSTATES}
        state_dict[-3] = 'NON-TB DEATH'  # Add a value for NaturalCauseDeath
        state_dict[-2] = 'NEVER INFECTED'
        self.data['state_name'] = self.data['state'].map(state_dict)
        self.data['going_to_state'] = self.data['going_to_state_id'].map(state_dict)
        self.data['going_to_state'] = self.data.apply(lambda row: f"{row['going_to_state_id']}.{row['going_to_state']}", axis=1 )
        self.data['state_name'] = self.data.apply(lambda row: f"{row['state']}.{row['state_name']}", axis=1 )        
        self.data['state_name'] = self.data['state_name'].replace('None', 'Susceptible')
        # self.data['compartment'] = 'tbd'
        
        
        self.file_path = self.__save_to_file__()
        return

    def _log_dwell_time(self, agent_ids, states, entry_times, exit_times, going_to_state_ids, age):
        """
        Log dwell time data for state transitions.
        
        Records dwell time information for a batch of agents who have
        changed states. This is the core data recording method used
        throughout the simulation.
        
        Mathematical Model:
            For each agent i in the batch:
            - Dwell time: dwell_time_i = exit_time_i - entry_time_i
            - Record: (agent_id_i, state_i, entry_time_i, exit_time_i, 
                      dwell_time_i, going_to_state_id_i, age_i)
                      
        Args:
            agent_ids (np.ndarray): Array of agent identifiers
            states (np.ndarray): Current states before transition
            entry_times (np.ndarray): Times when agents entered current states
            exit_times (np.ndarray): Times when agents exited current states
            going_to_state_ids (np.ndarray): Target state identifiers
            age (np.ndarray): Agent ages at transition time
            
        Implementation Details:
        - Calculates dwell times as exit_time - entry_time
        - Handles NaN values in entry times (converts to 0)
        - Creates DataFrame with all transition data
        - Appends to main data collection
        """
        entry_times = np.nan_to_num(entry_times, nan=0)
        dwell_times = exit_times - entry_times
        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'state': states,
            'entry_time': entry_times,
            'exit_time': exit_times,
            'dwell_time': dwell_times,
            'going_to_state_id': going_to_state_ids,
            'age': age
        })
        # append 
        self.data = pd.concat([self.data, new_logs], ignore_index=True)
        return
    
    def __save_to_file__(self):
        """
        Save dwell time data to CSV and metadata files.
        
        Creates timestamped files in the results directory with the
        dwell time data and simulation parameters for later analysis.
        
        File Naming Convention:
        - CSV file: {scenario_name}-{MMDDHHMMSS}.csv
        - Metadata file: {scenario_name}-{MMDDHHMMSS}.json
        
        Returns:
            str: Path to the saved CSV file
            
        Example:
        ```python
        # Files are saved automatically during finalize()
        analyzer = DwtAnalyzer(scenario_name="Baseline_TB")
        sim.run()
        # Results saved to: results/Baseline_TB-20240101120000.csv
        ```
        
        File Contents:
        - CSV: All dwell time data with columns for analysis
        - JSON: Simulation parameters and configuration
        """
        resdir = os.path.dirname(cfg.create_res_dir())
        t = ddtt.datetime.now()
        prefix = f'{Utils.to_filename_friendly(self.scenario_name)}'
        if prefix == '' or prefix is None: 
            prefix = 'dwt_logs'
        t = t.strftime("%m%d%H%M%S")
        fn = os.path.join(resdir, f'{prefix}-{t}.csv')
        self.data.to_csv(fn, index=False)

        fn_meta = os.path.join(resdir, f'{prefix}-{t}.json')
        with open(fn_meta, 'w') as f:
            f.write(f'{self.sim.pars}')

        print(f"===> Dwell time logs saved to:\n {fn}\n")
        return fn

    def validate_dwell_time_distributions(self, expected_distributions=None):
        """
        Validate dwell time distributions against expected patterns.
        
        Performs Kolmogorov-Smirnov tests to compare observed dwell time
        distributions with expected theoretical distributions. This is
        useful for model validation and quality assurance.
        
        Mathematical Model:
            For each state i:
            - Observed dwell times: T_obs = {dwell_times for state i}
            - Expected CDF: F_expected(t) = theoretical distribution
            - KS test: D = max|F_obs(t) - F_expected(t)|
            - P-value: probability of observing D under null hypothesis
            
        Args:
            expected_distributions (dict, optional): Dictionary mapping state IDs to expected CDF functions.
                                                   If None, uses self.expected_distributions
                                                   
        Returns:
            None: Prints validation results to console
            
        Example:
        ```python
        analyzer = DwtAnalyzer()
        sim.run()
        
        # Define expected distributions
        expected = {
            0: lambda x: 1 - np.exp(-x/10),  # Exponential with mean 10
            1: lambda x: 1 - np.exp(-x/5)    # Exponential with mean 5
        }
        
        # Validate distributions
        analyzer.validate_dwell_time_distributions(expected)
        ```
        
        Output:
        - Prints KS statistic and p-value for each state
        - Warns if distributions deviate significantly (p < 0.05)
        - Provides validation summary
        """
        from scipy.stats import ks_1samp, ks_2samp
        expected_distributions = expected_distributions or self.expected_distributions
       
        print("Validating dwell time distributions...")
        for state, expected_cdf in expected_distributions.items():
            dwell_times = self.data[self.data['state'] == state]['dwell_time']
            if dwell_times.empty:
                print(f"No data available for state {state}")
                continue
            stat, p_value = stats.kstest(dwell_times, expected_cdf)
            print(f"State {state}: KS Statistic={stat:.4f}, p-value={p_value:.4f}")
            if p_value < 0.05:
                print(f"WARNING: Dwell times for state {state} deviate significantly from expectations.")
        return



class Utils:
    """
    Utility functions for dwell time analysis.
    
    Provides helper functions for data processing, file handling, and
    visualization support in the dwell time analysis framework.
    """
    
    @staticmethod
    def to_filename_friendly(string=''):
        """
        Convert string to filename-safe format.
        
        Removes or replaces special characters that are not allowed
        in filenames across different operating systems.
        
        Args:
            string (str): Input string to convert
            
        Returns:
            str: Filename-safe string with only alphanumeric characters
            
        Example:
        ```python
        safe_name = Utils.to_filename_friendly("Baseline TB Model (v1.0)")
        print(safe_name)  # Output: "Baseline_TB_Model_v10"
        ```
        
        Conversion Rules:
        - Alphabetic characters: preserved
        - Numeric characters: preserved  
        - Special characters: replaced with underscores
        - Multiple underscores: collapsed to single underscore
        """
        import re
        string = "".join([c if c.isalpha() else "_" for c in string])
        return re.sub(r'[^a-zA-Z0-9]', '', string)
    
    @staticmethod
    def colors():
        """
        Get color mapping for TB state visualization.
        
        Returns a dictionary mapping state names to colors and a
        matplotlib ListedColormap for consistent visualization
        across all plotting functions.
        
        Returns:
            tuple: (state_colors_dict, matplotlib.colors.ListedColormap)
            
        Color Mapping:
        - Death states: cyan, gray, black
        - Susceptible states: yellow, orange, purple
        - Infection states: blue, purple, pink
        - Active states: brown, red, darkred, cyan
        - Treatment states: lightblue, darkgreen
        
        Example:
        ```python
        state_colors, cmap = Utils.colors()
        
        # Use in plotting
        plt.scatter(x, y, c=[state_colors[state] for state in states])
        
        # Or use colormap
        plt.imshow(data, cmap=cmap)
        ```
        """
        import matplotlib.colors as mcolors

        state_colors = {
            '-3.0.NON-TB DEATH': "cyan",  
            "-2.0.NEVER INFECTED": "orange",   
            "-1.0.Susceptible": "yellow",      
            "0.0.Infection":  "blue",     
            "1.0.Cleared": "green",        
            "2.0.Unconfirmed": "black",   
            "3.0.Recovered": "lightgreen",      
            "4.0.Asymptomatic": "gray",   
            "5.0.Symptomatic": "cyan",    
            "6.0.Treatment": "lightblue",      
            "7.0.Treated": "darkgreen",        
            "-1.0.None": "purple",
            "0.0.Latent Slow": "purple",
            "1.0.Latent Fast": "pink",
            "2.0.Active Presym": "brown",
            "3.0.Active Smpos": "red",
            "4.0.Active Smneg": "darkred",
            "5.0.Active Exptb": "cyan",
            "6.0.Dead": "gray",
            "8.0.Dead": "black",
        }
        
        cmap = mcolors.ListedColormap([state_colors[state] for state in state_colors])
        return state_colors, cmap


# Example usage and verification of the DwtAnalyzer and DwtPostProcessor classes
if __name__ == '__main__':

    debug = 0

    if debug == 0:
        # # # Initialize the DwtAnalyzer
        import starsim as ss
        sim = ss.Sim(diseases=mtb.TB(), analyzers=DwtAnalyzer(scenario_name="Baseline"), pars=dict(dt = ss.days(7), start = ss.date('1940'), stop = ss.date('2010')))
        sim.run()

        # # # Initialize the DwtPostProcessor
        londonpp = DwtPostProcessor(directory='results', prefix='BaselineLSHTM')
        londonpp.save_combined_dataframe('london_combined.csv')
        londonpp.sankey_agents()

    if debug == 1:
            results_path = '/Users/mine/git/tb_acf/results/'
            londonpp = DwtPostProcessor(directory=results_path, prefix='BaselineLSHTM')        # sample: results/BaselineLSHTM-20250128115433.csv
            londonpp.save_combined_dataframe('london_combined_01281704.csv')
            londonpp.sankey_agents()

    if debug == 2:
        # # # Initialize the DwtPlotter
        file = "/Users/mine/TEMP/results/LowDecliningTBsim-0206191748.csv"
        file = "/Users/mine/TEMP/results/LowDecliningLSHTM-0206192013.csv"
        pl = mtb.DwtPlotter(file_path=file)
        # pl.graph_state_transitions()
       
        #pl.stacked_bars_states_per_agent_static()
    
        # pl.barchar_all_state_transitions_interactive()
        # pl.graph_state_transitions(states=["-1.0.Susceptible", "0.0.Infection", "1.0.Cleared", "2.0.Unconfirmed", "3.0.Recovered",  "5.0.Symptomatic"], layout=9)
        # pl.graph_state_transitions_curved(graphseed=9)


 