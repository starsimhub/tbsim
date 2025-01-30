import starsim as ss
import pandas as pd
import numpy as np
import os
import tbsim.config as cfg
import datetime as ddtt
import tbsim as mtb
from scipy import stats
from enum import IntEnum
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import networkx as nx
import plotly.graph_objects as go


__all__ = ['DwtAnalyzer', 'DwtPlotter', 'DwtPostProcessor']

class DwtPlotter:
    def __init__(self, data=None, file_path=None):
        if data is not None:
            self.data = data
        elif file_path is not None:
            self.data = pd.read_csv(file_path, na_values=[], keep_default_na=False)
        else:
            raise ValueError("Either data or file_path must be provided.")

    # TODO: Kaplan-Meier
    def plot_kaplan_meier(self, dwell_time_col, event_observed_col=None):
        """
        Plots a Kaplan-Meier survival curve for the given data.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing survival data.
            dwell_time_col (str): Column name representing dwell times.
            event_observed_col (str, optional): Column indicating if the event was observed (1) or censored (0).
                If None, assumes all events are observed.

        Returns:
            None: Displays the Kaplan-Meier survival plot.
        """
        if self.data_error():
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
        plt.xlabel(f"Time ({dwell_time_col})", fontsize=14)
        plt.ylabel("Survival Probability", fontsize=14)
        plt.grid(True)
        plt.show()

    # looks good
    def state_transition_matrix(self):
        """
        Generates and plots a state transition matrix from the provided data.
        Parameters:
        file_path (str, optional): Path to the CSV file containing the data. The CSV file should have columns 'agent_id' and 'state'.
        self.data (pd.DataFrame, optional): DataFrame containing the data. Should have columns 'agent_id' and 'state'.
        Returns:
        None: The function plots the state transition matrix using seaborn's heatmap.
        Notes:
        - This is using plain 'state' columns recorded in the data - no need to 'going_to_state' column.
        - If both file_path and self.data are provided, file_path will be used.
        - If neither file_path nor self.data are provided, the function will print "No data provided." and return.
        - The transition matrix is normalized to show proportions. To display raw counts, comment out the normalization step.
        """

        if self.data_error():
            return
        df = self.data

        # Create a transition matrix
        # Get the unique states
        # unique_states = sorted(df['state'].dropna().unique())
        unique_states = sorted(df['state'].unique())
        # Initialize a matrix of zeros
        transition_matrix = pd.DataFrame(
            data=0, index=unique_states, columns=unique_states, dtype=int
        )

        # Fill the matrix with transitions
        for agent_id, group in df.groupby('agent_id'):
            states = group['state'].values
            for i in range(len(states) - 1):
                transition_matrix.loc[states[i], states[i + 1]] += 1

        # Normalize rows to show proportions (optional, can comment this out if counts are preferred)
        transition_matrix_normalized = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

        # Plot the state transition matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            transition_matrix_normalized, 
            annot=True, 
            fmt=".2f", 
            cmap="Blues", 
            xticklabels=unique_states, 
            yticklabels=unique_states
        )
        plt.title("State Transition Matrix (Normalized)")
        plt.xlabel("Next State")
        plt.ylabel("Current State")
        plt.show()

    def sankey_agents(self):
        """
        Generates and displays a Sankey diagram of state transitions based on the count of agents.

        Parameters:
        file_path (str, optional): The path to a CSV file containing the data. If not provided, 
                                    the method will use the data stored in self.data.

        The CSV file or self.data should contain the following columns:
        - 'state_name': The name of the current state.
        - 'going_to_state': The name of the state to which the transition is made.

        If neither file_path nor self.data is provided, the method will print "No data provided." and return.

        The method uses Plotly to create and display the Sankey diagram.
        """
        import plotly.graph_objects as go
        if self.data_error():
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

        fig.update_layout(title_text="Sankey Diagram of State Transitions by Agent Count", font_size=10)
        fig.show()

    #looks better- but still not perfect
    def sankey(self):
        """
        Generates and displays a Sankey diagram of state transitions and dwell times.

        Parameters:
        file_path (str, optional): The path to a CSV file containing the data. If not provided, 
                                   the method will use the data stored in self.data.

        The CSV file or self.data should contain the following columns:
        - 'state_name': The name of the current state.
        - 'going_to_state': The name of the state to which the transition is made.
        - 'dwell_time': The time spent in the current state before transitioning.

        If neither file_path nor self.data is provided, the method will print "No data provided." and return.

        The method uses Plotly to create and display the Sankey diagram.
        """
        import plotly.graph_objects as go

        if self.data_error():
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

        # Create the Sankey plot
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.2),
                label=labels
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=value,
                hovertemplate='%{source.label} → %{target.label}: %{value} days<br>',
                line=dict(color="lightgray", width=0.1),
            )
        ))

        fig.update_layout(title_text="Sankey Diagram of State Transitions and Dwell Times", font_size=10)
        fig.show()

    # looks good /
    def interactive_all_state_transitions(self, dwell_time_bins=None, filter_states=None):
        """
        Generates an interactive bar chart of state transitions grouped by dwell time categories.

        Parameters:
        dwell_time_bins (list, optional): List of bin edges for categorizing dwell times. 
                                          Defaults to [0, 50, 100, 150, 200, 250].
        filter_states (list, optional): List of states to filter the data by. If None, no filtering is applied.

        Returns:
        None: Displays an interactive Plotly bar chart.
        """

        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go

        if self.data_error():
            return

        # Set default bins if none are provided
        if dwell_time_bins is None:
            dwell_time_bins = [0, 50, 100, 150, 200, 250]

        #appemd infinity to the bins
        dwell_time_bins.append(np.inf)

        # Create bin labels, handling infinity separately
        dwell_time_labels = [
            f"{int(b)}-{int(d)} days" if d != np.inf else f"{int(b)}+ days"
            for b, d in zip(dwell_time_bins[:-1], dwell_time_bins[1:])
        ]

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
            ['state_name', 'going_to_state', 'dwell_time_category']
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

        fig.update_layout(
            title="State Transitions Grouped by Dwell Time Categories",
            yaxis_title="State Transitions",
            xaxis_title="Count",
            legend_title="Transitions",
            height=100 + 30 * len(grouped),
        )
        fig.show()

    # looks good - although crowded /
    def stacked_bars_states_per_agent_static(self):
        """
        Plots a stacked bar chart showing the cumulative dwell time in days for each state per agent.

        This function reads data from a CSV file or uses an existing DataFrame to calculate the cumulative dwell time
        for each agent and state. It then converts the dwell time to days and creates a pivot table to get the cumulative
        dwell time for each state. Finally, it plots a stacked bar chart to visualize the cumulative time in days spent
        in each state for all agents.

        Parameters:
        file_path (str, optional): The path to the CSV file containing the data. If not provided, the function will use
                                   the data stored in `self.data`.

        Returns:
        None
        """
        if self.data_error():
            return
        df = self.data
        # Calculate cumulative dwell time for each agent and state
        df['cumulative_dwell_time'] = df.groupby(['agent_id', 'state_name'])['dwell_time'].cumsum()

        # Convert dwell time to days
        df['cumulative_dwell_time_days'] = df['cumulative_dwell_time']#/24

        # Pivot the data to get cumulative dwell time for each state
        pivot_df = df.pivot_table(index='agent_id', columns='state_name', values='cumulative_dwell_time_days', aggfunc='max', fill_value=0)

        # Plot the data
        pivot_df.plot(kind='bar', stacked=True, figsize=(15, 7))
        plt.title('Cumulative Time in Days on Each State for All Agents')
        plt.xlabel('Agent ID')
        plt.ylabel('Cumulative Time (Days)')
        plt.legend(title='State Name')
        plt.tight_layout()
        plt.show()
        return

    # looks good
    def interactive_stacked_bar_charts_dt_by_state(self, bin_size=1, num_bins=20):
        """
        Generates an interactive stacked bar chart of dwell times by state using Plotly.

        Parameters:
        bin_size (int): The size of each bin for dwell times. Default is 50.
        num_bins (int): The number of bins to divide the dwell times into. Default is 20.
        
        Returns:
        None: Displays an interactive Plotly figure.
        
        Notes:
        - If the self.data DataFrame is empty, the function will print a message and return without plotting.
        - The function creates bins for dwell times and labels them in days.
        - It generates a stacked bar chart for each state, showing the count of transitions to other states within each dwell time bin.
        - The height of the figure is dynamically adjusted based on the number of states.
        """
        import plotly.express as px
        import plotly.graph_objects as go

        if self.data_error():
            return

        # Define bins for dwell times
        bins = np.arange(0, bin_size * num_bins, bin_size)
        bin_labels = [f"{int(b)}-{int(b + bin_size)} days" for b in bins[:-1]]

        # Create a figure with subplots for each state
        states = self.data['state_name'].unique()
        num_states = len(states)
        fig = go.Figure()

        for state in states:
            state_data = self.data[self.data['state_name'] == state]
            state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

            # Group by dwell time bins and going to state
            grouped = state_data.groupby(['dwell_time_bin', 'going_to_state']).size().unstack(fill_value=0)

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
            title="Stacked Bar Charts of Dwell Times by State",
            xaxis_title="Dwell Time Bins",
            yaxis_title="Count",
            legend_title="State Transitions",
            height=400 + 50 * num_states
        )
        fig.show()

    # looks good
    def plot_state_transition_lengths_custom(self, transitions_dict=None):
        """
        Plots the cumulative distribution of dwell times for different state transitions.

        This function generates a plot with individual lines representing the cumulative 
        distribution of dwell times for each specified state transition. If no transitions 
        dictionary is provided, a default one is used.

        Parameters:
        -----------
        transitions_dict (dict): A dictionary where keys are state names and values are 
            lists of states to which transitions are considered. If None, a default 
            dictionary is used.

            i.e.:

            transitions_dict = {
                'None': ['Latent Slow', 'Latent Fast'],
                'Active Presymp': ['Active Smpos', 'Active Smneg', 'Active Exptb'],
            }
        Returns:
        None: The function displays the plot and does not return any value.
        """

        import matplotlib.pyplot as plt
        import numpy as np

        if self.data_error():
            return

        if transitions_dict is None:
            transitions_dict = {
                'None': ['Latent Slow', 'Latent Fast'],
                'Active Presymp': ['Active Smpos', 'Active Smneg', 'Active Exptb'],
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
        plt.tight_layout()
        plt.show()

    # looks good /
    def plot_binned_by_compartment(self,  bin_size=1, num_bins=50):
        """
        Plots the dwell time data binned by compartment for each state.

        Parameters:
        -----------
        bin_size : int, optional
            The size of each bin for dwell times in days. Default is 50.
        num_bins : int, optional
            The number of bins to create. Default is 8.

        Returns:
        --------
        None
            This function does not return any value. It generates and displays a plot.

        Notes:
        ------
        - The function uses matplotlib to create a figure with subplots for each unique state in the data.
        - Each subplot shows a stacked bar chart of the count of dwell times binned by the specified bin size and grouped by compartment.
        - If the data is empty, the function prints a message and returns without plotting.
        - The function automatically adjusts the layout to fit all subplots and removes any empty subplots.
        """

        import matplotlib.pyplot as plt

        if self.data_error():
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
        fig.suptitle(f'State - Compartment Transitions)', fontsize=16)
        axes = axes.flatten()

        for ax, state in zip(axes, states):
            state_data = self.data[self.data['state_name'] == state]
            state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

            # Group by dwell time bins and going to state
            grouped = state_data.groupby(['dwell_time_bin', 'compartment']).size().unstack(fill_value=0)

            # Plot stacked bar chart
            grouped.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
            ax.set_title(f'State: {state}')
            ax.set_xlabel('Dwell Time Bins')
            ax.set_ylabel('Count')
            ax.legend(title='compartment')

        # Remove any empty subplots
        for i in range(num_states, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    # looks good /
    def plot_binned_stacked_bars_state_transitions(self, bin_size=1, num_bins=50):
        """
        Plots binned stacked bar charts for state transitions based on dwell times.

        Parameters:
        bin_size (int): The size of each bin for dwell times in days. Default is 50.
        num_bins (int): The number of bins to create. Default is 8.

        Returns:
        None: This function does not return any value. It displays a plot.

        Notes:
        - The function checks if the data is empty and prints a message if no data is available.
        - It creates bins for dwell times and labels them accordingly.
        - A figure with subplots is created for each unique state in the data.
        - Each subplot shows a stacked bar chart of state transitions grouped by dwell time bins.
        - Any empty subplots are removed before displaying the plot.
        """

        import matplotlib.pyplot as plt

        if self.data_error():
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
            state_data = self.data[self.data['state_name'] == state]
            state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

            # Group by dwell time bins and going to state
            grouped = state_data.groupby(['dwell_time_bin', 'going_to_state']).size().unstack(fill_value=0)

            # Plot stacked bar chart
            grouped.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
            ax.set_title(f'State: {state}')
            ax.set_xlabel('Dwell Time Bins')
            ax.set_ylabel('Count')
            ax.legend(title='Going to State')

        # Remove any empty subplots
        for i in range(num_states, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def histogram_with_kde(self):
        """
        Plots histograms with Kernel Density Estimation (KDE) for dwell times of different states.
        Parameters:
        num_bins (int): Number of bins for the histogram. Default is 50.
        bin_size (int): Size of each bin. Default is 30.
        Returns:
        None: Displays the histogram with KDE plots.
        Notes:
        - If the data is empty, the function will print a message and return without plotting.
        - The function creates subplots for each unique state in the data.
        - Each subplot shows the distribution of dwell times for a state, with KDE and stacked histograms based on the 'going_to_state' column.
        - Unused subplots are removed from the figure.
        """

        if self.data_error():
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
        fig.suptitle('State Transitions by Dwell Time Bins', fontsize=16)

        for ax, state in zip(axes, states):
            state_data = df[df['state_name'] == state]

            # Automatically define the number of bins and bin size based on the data
            max_dwell_time = state_data['dwell_time'].max()
            bin_size = max(1, max_dwell_time // 15)  # Ensure at least 15 bins
            bins = np.arange(0, max_dwell_time + bin_size, bin_size)
            bin_labels = [f"{int(b)}-{int(b+bin_size)} days" for b in bins[:-1]]

            state_data['dwell_time_bin'] = pd.cut(
            state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True,
            )
            sns.histplot(data=state_data, 
                 x='dwell_time', 
                 bins=bins,
                 hue='going_to_state', 
                 kde=True, 
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

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # looks good /
    def graph_state_transitions(self, states=None, layout=None, curved_ratio=0.05, colormap='tab20c'):
        """
        Plot a state transition graph with mean and mode dwell times annotated on the edges.
        
        Parameters:
        -----------
        curved_ratio (float, optional): Ratio to curve the edges. Default is 0.05.
        colormap (str, optional): Name of the colormap to use for coloring nodes. Default is 'tab20c'.
        states (list, optional): A list of states to include in the graph. If None, all states in the self.data will be included.
        layout (dict, optional): A dictionary specifying the layout positions of nodes. If None, a spring layout is used.
                0: (Default) Spring layout.
                1: Circular layout
                2: Spiral layout
                3: Spectral layout
                4: Shell layout
                5: Kamada-Kawai layout
                6: Planar layout
                7: Random layout
                8: Circular layout
                9: Fruchterman-Reingold layout

        Returns:
        --------
        None: This function does not return any value. It displays a plot of the state transition graph.

        Notes:
        -------
        - The function uses NetworkX to create a directed graph where nodes represent states and edges represent transitions.
        - Each edge is annotated with the mean and mode dwell times, as well as the number of agents that made the transition.
        - If the self.data is empty, the function prints a message and returns without plotting.
        - The graph layout is generated using a spring layout for better visualization.
        - Nodes are colored using a colormap, and edges are drawn with arrows to indicate direction.
        - The graph is displayed using Matplotlib.
        """
        import networkx as nx
        import itertools as it
        from scipy import stats

        if self.data_error():
            return
        
        if states is not None:
            self.data = self.data[self.data['state_name'].isin(states)]

        # Calculate mean, mode, and count for each state transition
        transitions = self.data.groupby(['state_name', 'going_to_state'])['dwell_time']
        stats_df = transitions.agg([
            'mean',
            lambda x: stats.mode(x, keepdims=True).mode[0] if len(x) > 0 else np.nan,
            'count'
        ]).reset_index()

        stats_df.columns = ['state_name', 'going_to_state', 'mean', 'mode', 'count']

        # Create a directed graph
        G = nx.DiGraph()

        # Add edges with mean and mode annotations
        for _, row in stats_df.iterrows():
            from_state = row['state_name']
            to_state = row['going_to_state']
            mean_dwell = round(row['mean'], 2) if not pd.isna(row['mean']) else "N/A"
            mode_dwell = round(row['mode'], 2) if not pd.isna(row['mode']) else "N/A"
            num_agents = row['count']

            # Add edge to the graph
            G.add_edge(from_state, to_state,
                label=f"Mean: {mean_dwell}\nMo: {mode_dwell}\nAgents: {num_agents}")

        # Generate a layout for the graph
        if layout is None:
            pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
        else:
            pos = self.select_graph_pos(G, layout=layout)

        colors =plt.colormaps.get_cmap(colormap) 
        node_colors = [colors(i) for i in range(len(G.nodes))]
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color=node_colors, alpha=0.9)
        
        # Draw edges with the same color as the origin node
        edge_colors = [node_colors[list(G.nodes).index(edge[0])] for edge in G.edges]
        nx.draw_networkx_edges(G, pos, width=1, arrowstyle="-|>", arrowsize=30, edge_color=edge_colors, connectionstyle=f"arc3,rad={curved_ratio}")
        
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")

        # Annotate edges with mean and mode
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Display the graph
        plt.tight_layout()
        plt.title("State Transition Graph with Dwell Times")
        plt.show()
        return

    # Looks good /
    def graph_compartments_transitions(self, states=None, layout=0, groups=[[]]):
        """
        /* UNDER CONSTRUCTION */
        Plots a directed graph of state transitions with dwell times.

        Parameters:
            states (list, optional): A list of state names to filter the self.data. If None, all states are included.
            groups (list of lists, optional): A list of groups for custom node coloring. Default is [[]].
            layout (int, optional): The layout type for the graph. 
                Default is 0 (spring layout).
                1: Circular layout
                2: Spiral layout
                3: Spectral layout
                4: Shell layout
                5: Kamada-Kawai layout
                6: Planar layout
                7: Random layout
                8: Circular layout
                9: Fruchterman-Reingold layout
            Notes:
            - The function expects self.data to be a pandas DataFrame containing columns 'state_name', 'compartment', and 'dwell_time'.
            - It uses NetworkX for graph creation and Matplotlib for plotting.
            - Then calculates mean, mode, and count of dwell times for each state transition.

        Returns:
        None: The function displays a plot of the state transition graph with annotations for mean, mode, and count of dwell times.
        """

        import networkx as nx
        import itertools as it
        from scipy import stats

        if self.data_error():
            return

        if states is not None:
            self.data = self.data[self.data['state_name'].isin(states)]

        # Calculate mean, mode, and count for each state transition
        transitions = self.data.groupby(['state_name', 'compartment'])['dwell_time']
        stats_df = transitions.agg([
            'mean',
            lambda x: stats.mode(x, keepdims=True).mode[0] if len(x) > 0 else np.nan,
            'count'
        ]).reset_index()

        stats_df.columns = ['state_name', 'compartment', 'mean', 'mode', 'count']

        # Create a directed graph
        G = nx.DiGraph()

        # Add edges with mean and mode annotations
        for _, row in stats_df.iterrows():
            from_state = row['state_name']
            to_compartment = row['compartment']
            mean_dwell = round(row['mean'], 2) if not pd.isna(row['mean']) else "N/A"
            mode_dwell = round(row['mode'], 2) if not pd.isna(row['mode']) else "N/A"
            num_agents = row['count']

            # Add edge to the graph
            G.add_edge(from_state, to_compartment,
                        label=f"Mean: {mean_dwell}, Mode: {mode_dwell}\nAgents: {num_agents}")

        # Generate a layout for the graph
        pos = self.select_graph_pos(G, layout)

        # Draw nodes and edges with curved lines
        colors =plt.colormaps.get_cmap('tab20')
        node_colors = [colors(i) for i in range(len(G.nodes))]
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=10, edge_color="black", connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")

        # Annotate edges with mean and mode
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        # Display the graph
        plt.title("State->Compartment Graph with Dwell Times")
        plt.show()
        return

    def plot_dwell_time_validation(self):
        """
        Plot the results of the dwell time validation.

        This method generates a histogram for the dwell times of different states
        in the dataset. Each state's dwell time is plotted in a separate histogram
        with a unique label.

        The method performs the following steps:
        1. Creates a figure and axis for the plot.
        2. Retrieves the unique states from the dataset.
        3. Iterates over each state and extracts the dwell times for that state.
        4. Plots a histogram of the dwell times for each state.
        5. Sets the x-axis label to 'Dwell Time' and the y-axis label to 'Frequency'.
        6. Adds a legend to the plot.
        7. Displays the plot.

        Returns:
            None
        """
        # Plot the results of the dwell time validation.
        if self.data_error():
            return
        fig, ax = plt.subplots()
        data = self.data
        model_states = data['state_name'].unique()
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
        Plots an interactive histogram for dwell time validation using Plotly.

        This method generates an interactive histogram plot of the dwell time data,
        categorized by state names. The histogram is overlaid with different colors
        representing different states, and the plot includes labels and a title for
        better readability.

        The histogram is displayed using Plotly's `show` method, which opens the plot
        in a web browser.

        Returns:
            None
        """

        import plotly.express as px
        if self.data_error():
            return
        
        data = self.data
        fig = px.histogram(data, x='dwell_time', color='state_name', 
                            nbins=50, barmode='overlay', 
                            labels={'dwell_time': 'Dwell Time', 'state_name': 'State'},
                            title='Dwell Time Validation')
        fig.update_layout(bargap=0.1)
        fig.show()
        return
    
    def data_error(self):
        # data error handling - check if data is available

        if self.data is None or self.data.empty or 'dwell_time' not in self.data.columns:
            print("No dwell time data available to plot.")
            return True
        return False

    @staticmethod
    def select_graph_pos(G, layout):
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
            return nx.spring_layout(G, seed=42)

        return pos

class DwtPostProcessor(DwtPlotter):
    def __init__(self, directory='', prefix='', data=None, **kwargs):
        """
        Initializes the post-analyzer with the data from the DwtAnalyzer.

        Args:
            data (pd.DataFrame): A DataFrame containing the dwell time data. Default is None.

        How to use it:
        1. Create an instance of the post-analyzer with the data from the DwtAnalyzer.
        2. Call the method you want to use.

        Example:
        ```
        postproc = DwtPostProcessor(directory='results', prefix='BaselineLSHTM')
        postproc.sandkey()
        ```
        """
        self.directory = directory
        self.prefix = prefix

        if data is None:
            data = self.aggregate_simulation_results(self.directory, self.prefix)
        DwtPlotter.__init__(self, data=data)
        
        return


    def aggregate_simulation_results(self, directory: str, prefix: str) -> pd.DataFrame:
        """
        Aggregates all CSV files of the same kind (e.g., BaselineLSHTM, BaselineTBSim)
        into a single DataFrame.

        Args:
            directory (str): The path to the directory containing the CSV files.
            prefix (str): The common prefix identifying the group of files to aggregate
                          (e.g., "BaselineLSHTM" or "BaselineTBSim").

        Returns:
            pd.DataFrame: A concatenated DataFrame containing all data from matching files.
        """
        import os
        import pandas as pd
        import glob

        file_pattern = os.path.join(directory, f"{prefix}-*.csv")
        file_list = glob.glob(file_pattern)

        if not file_list:
            print(f"No files found matching pattern: {file_pattern}")
            return pd.DataFrame()
        else:
            print(f"Found {len(file_list)} files matching pattern: {file_pattern}")
            print("\n".join(file_list))

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
        Saves the combined DataFrame to a specified CSV file.

        Args:
            output_file (str): The path to the output CSV file.

        Returns:
            None
        """
        if self.data is None or self.data.empty:
            print("No data available to save.")
            return

        try:
            self.data.to_csv(output_file, index=False)
            print(f"Combined DataFrame saved to {output_file}")
        except Exception as e:
            print(f"Error saving DataFrame to {output_file}: {e}")
            
class DwtAnalyzer(ss.Analyzer, DwtPlotter):
    def __init__(self, adjust_to_unit=False, unit=1.0, states_ennumerator=mtb.TBS, scenario_name=''):
        """
        Initializes the analyzer with optional adjustments to days and unit specification.

        Args:
            adjust_to_unit (bool): If True, adjusts the dwell times to days by multiplying the recorded dwell_time by the sim.pars.dt.
            Default is True.
            
            unit (float | ss.t ):  TODO: Implement its use.
            states_ennumerator (IntEnum): An IntEnum class that enumerates the states in the simulation. Default is mtb.TBS but it will accept any equivalent IntEnum class.

        How to use it:
        1. Add the analyzer to the sim object.
        2. Run the simulation.
        3. Optional: Create an instance of the analyzer and call the method you want to use.
        
        Example:
        ```
        sim = tb.Sim()
        sim.add_analyzer(DwtAnalyzer())
        sim.run()
        analyzer = sim.analyzers[0]
        analyzer.plot_dwell_time_validation()
        ```
        
        """
        ss.Analyzer.__init__(self)
        self.eSTATES = states_ennumerator
        self.adjust_to_unit = adjust_to_unit
        self.unit = unit
        self.file_path = None
        self.scenario_name = scenario_name
        self.data = pd.DataFrame(columns=['agent_id', 'state', 'entry_time', 'exit_time', 'dwell_time', 'state_name', 'going_to_state_id','going_to_state'])
        self._latest_sts_df = pd.DataFrame(columns=['agent_id', 'last_state', 'last_state_time'])      
        DwtPlotter.__init__(self, data=self.data)
        return
    
    def _initialize_dataframes(self):
        # Initialize the latest state dataframe
        if self.unit is None:
            self.unit = self.sim.pars.unit
        agent_ids = self.sim.people.auids
        population = len(agent_ids)
        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'last_state': np.full(population, -1),
            'last_state_time': np.zeros(population)
        })
        self._latest_sts_df = pd.concat([self._latest_sts_df, new_logs], ignore_index=True)
        return
    
    def step(self):
        if not self.sim.ti: self._initialize_dataframes()
        sim = self.sim
        ti = sim.ti
        self._check_for_new_borns()       
        
        # check if the number of agents has changed
        if len(self.sim.people.auids) != len(self._latest_sts_df):
            #identify which agent ids are new and add them to the _latest_sts_df
            new_agent_ids = list(set(self.sim.people.auids) - set(self._latest_sts_df.agent_id))
            new_logs = pd.DataFrame({
                'agent_id': new_agent_ids,
                'last_state': np.full(len(new_agent_ids), -1),
                'last_state_time': np.zeros(len(new_agent_ids))
            })
            self._latest_sts_df = pd.concat([self._latest_sts_df, new_logs], ignore_index=True)
        self._update_state_change_data(ti)
        self._record_natural_deaths(ti)
        return
    
    def _update_state_change_data(self, ti):
        # Get the current state of the agents
        tb = self.sim.diseases.tb
        uids = self.sim.people.auids  # People Alive

        # Filter rows in _latest_sts_df for the relevant agents
        relevant_rows = self._latest_sts_df[self._latest_sts_df['agent_id'].isin(uids)]

        # Identify agents whose last recorded state is different from the current state
        different_state_mask = relevant_rows['last_state'].values != tb.state[ss.uids(relevant_rows['agent_id'].values)]
        uids = ss.uids(relevant_rows['agent_id'].values[different_state_mask])

        # Log dwell times
        self._log_dwell_time(
            agent_ids=uids,
            states=relevant_rows['last_state'].values[different_state_mask],
            entry_times=relevant_rows['last_state_time'].values[different_state_mask],
            exit_times=np.full(len(uids), ti),
            going_to_state_ids=tb.state[uids]
        )

        # Update the latest state dataframe
        self._latest_sts_df.loc[self._latest_sts_df['agent_id'].isin(uids), 'last_state'] = tb.state[uids]
        self._latest_sts_df.loc[self._latest_sts_df['agent_id'].isin(uids), 'last_state_time'] = ti
        return

    # TODO:  IN PROGRESS
    def _record_natural_deaths(self, ti):
        # Get the current state of the agents
        dead_uids = ss.uids(self.sim.people.dead)
        if not dead_uids.all():
            return
        # Filter rows in _latest_sts_df for the relevant agents
        relevant_rows = self._latest_sts_df[self._latest_sts_df['agent_id'].isin(dead_uids) & (self._latest_sts_df['last_state'] == -1)]

        # identify only those ones that are not already recorded
        relevant_rows = relevant_rows[~relevant_rows['agent_id'].isin(ss.uids(self.data['agent_id']))]
        if not relevant_rows.empty:
            self._log_dwell_time(
                agent_ids=relevant_rows['agent_id'].values,
                states=relevant_rows['last_state'].values,
                entry_times=relevant_rows['last_state_time'].values,
                exit_times=np.full(len(relevant_rows), ti),
                going_to_state_ids=np.full(len(relevant_rows), 100)  # State 100 is to represent Natural Death
            )
            
    def _check_for_new_borns(self):
        # check if the number of agents has changed
        if len(self.sim.people.auids) != len(self._latest_sts_df):
            #identify which agent ids are new and add them to the _latest_sts_df
            new_agent_ids = list(set(self.sim.people.auids) - set(self._latest_sts_df.agent_id))
            new_logs = pd.DataFrame({
                'agent_id': new_agent_ids,
                'last_state': np.full(len(new_agent_ids), -1),
                'last_state_time': np.zeros(len(new_agent_ids))
            })
            self._latest_sts_df = pd.concat([self._latest_sts_df, new_logs], ignore_index=True) # Add new agents to the _latest_sts_df - more likely new borns
    
    def finalize(self):
        super().finalize()
        
        # TODO: this is a temporary solution to get the enum name, ideally this information 
        # should be available in the disease class
        # feel free to suggest an alternate way to do this:
        if 'LSHTM' in str(self.sim.diseases[0].__class__):
            print("Using model: str(self.sim.diseases[0].__class__)")
            import tb_acf as tbacf
            self.eSTATES = tbacf.TBSL

        # Create a dictionary to map state values to their names
        state_dict = {state.value: state.name.replace('_', ' ').title() for state in self.eSTATES}
        state_dict[100] = 'NON-TB DEATH'  # Add a value for NaturalCauseDeath

        self.data['state_name'] = self.data['state'].map(state_dict)
        # Replace any state_name with 'None' with "Susceptible"
        self.data['state_name'] = self.data['state_name'].replace('None', 'Susceptible')

        self.data['going_to_state'] = self.data['going_to_state_id'].map(state_dict)
        # self.data['compartment'] = 'tbd'
        if self.adjust_to_unit:
            self.data['dwell_time_raw'] = self.data['dwell_time']   # Save the original recorded values for comparison or later use
            if isinstance(self.unit, (float, int)):
                self.data['dwell_time'] = self.data['dwell_time'] * self.unit   
            elif isinstance(self.unit, str):
                self.data['dwell_time'] = self.data['dwell_time'] * (self.sim.pars.dt / ss.rate(self.unit))
                # self.data['dwell_time'] = self.data['dwell_time'].apply(lambda x: eval(f"{x} {self.unit}"))
        
        self.file_path = self._save_to_file()
        return

    def _log_dwell_time(self, agent_ids, states, entry_times, exit_times, going_to_state_ids):
        entry_times = np.nan_to_num(entry_times, nan=0)
        dwell_times = exit_times - entry_times

        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'state': states,
            'entry_time': entry_times,
            'exit_time': exit_times,
            'dwell_time': dwell_times,
            'going_to_state_id': going_to_state_ids
        })
        self.data = pd.concat([self.data, new_logs], ignore_index=True)
                # Map state codes to their corresponding names
        return
    
    def _save_to_file(self):
        resdir = os.path.dirname(cfg.create_res_dir())
        t = ddtt.datetime.now()
        prefix = f'{self.to_filename_friendly(self.scenario_name)}'
        if prefix == '' or prefix is None: 
            prefix = 'dwt_logs'
        t = t.strftime("%m%d%H%M%S")
        fn = os.path.join(resdir, f'{prefix}-{t}.csv')
        self.data.to_csv(fn, index=False)

        fn_meta = os.path.join(resdir, f'{prefix}-{t}.json')
        with open(fn_meta, 'w') as f:
            f.write(f'{self.sim.pars}')

        print(f"Dwell time logs saved to:\n {fn}\n")
        return fn

    @staticmethod
    def to_filename_friendly(string=''):
        import re
        string = "".join([c if c.isalpha() else "_" for c in string])

        return re.sub(r'[^a-zA-Z0-9]', '', string)


    def validate_dwell_time_distributions(self, expected_distributions=None):
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



# Example usage
if __name__ == '__main__':
    results_path = '/Users/mine/git/tb_acf/results/'
    londonpp = DwtPostProcessor(directory=results_path, prefix='BaselineLSHTM')        # sample: results/BaselineLSHTM-20250128115433.csv
    londonpp.save_combined_dataframe('london_combined_01281704.csv')
    londonpp.sankey_agents()