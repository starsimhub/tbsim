import matplotlib.colors as mcolors
import pandas as pd
import starsim as ss
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
import warnings
import plotly.express as px
import itertools as it
        

__all__ = ['DwtAnalyzer', 'DwtPlotter', 'DwtPostProcessor']

warnings.simplefilter(action='ignore', category=FutureWarning)
class DwtPlotter:
    def __init__(self, data=None, file_path=None):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif file_path is not None:
            self.data = self.__cleandata__(file_path)
        else:
            raise ValueError("Either data or file_path must be provided.")
        if self.__data_error__():
            print("No data provided, or data is corrupted")
            
    def sankey_agents_by_age_subplots(self, bins=[5, 16, 200], scenario="", includecycles=False):
        """
        Generates and displays a single figure with multiple Sankey diagrams of state transitions,
        filtering data by age bins .

        Parameters:
        - data (pd.DataFrame): A DataFrame containing columns:
        - 'state_name': The name of the current state.
        - 'going_to_state': The name of the state to which the transition is made.
        - 'age': The age of the agents.
        
        This function automatically determines four age bins and arranges them
        in a 2x2 subplot layout.
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
        # axes = axes.flatten()
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

    def sankey_agents_even_age_ranges(self, number_of_plots=8):
        """
        Generates and displays multiple Sankey diagrams of state transitions,
        filtering data by age bins.

        Parameters:
        - data (pd.DataFrame): A DataFrame containing columns:
        - 'state_name': The name of the current state.
        - 'going_to_state': The name of the state to which the transition is made.
        - 'age': The age of the agents.

        This function automatically determines four age bins and generates
        separate Sankey diagrams for each bin.
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
        
                fig.update_layout(title_text=f"Sankey Diagram (Ages {bin_min:.1f} - {bin_max:.1f})")
                fig.show()

    def sankey_agents(self, subtitle = ""):
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
        NOTE: Very large data sets may cause the plot to be slow or unresponsive.
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
                hovertemplate='%{source.label} → %{target.label}: %{value} step_time_units<br>',
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
        Generates an interactive bar chart of state transitions grouped by dwell time categories.

        Parameters:
        dwell_time_bins (list, optional): List of bin edges for categorizing dwell times. 
                                          Defaults to [0, 50, 100, 150, 200, 250].
        filter_states (list, optional): List of states to filter the data by. If None, no filtering is applied.

        Returns:
        None: Displays an interactive Plotly bar chart.
        """


        

        if self.__data_error__():
            return

        # Set default bins if none are provided
        if dwell_time_bins is None:
            dwell_time_bins = [0, 50, 100, 150, 200, 250]

        #appemd infinity to the bins
        dwell_time_bins.append(np.inf)

        # Create bin labels, handling infinity separately
        dwell_time_labels = [
            f"{int(b)}-{int(d)} step_time_units" if d != np.inf else f"{int(b)}+ step_time_units"
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
        Plots a stacked bar chart showing the cumulative dwell time in step_time_units for each state per agent.

        This function reads data from a CSV file or uses an existing DataFrame to calculate the cumulative dwell time
        for each agent and state. It then converts the dwell time to step_time_units and creates a pivot table to get the cumulative
        dwell time for each state. Finally, it plots a stacked bar chart to visualize the cumulative time in step_time_units spent
        in each state for all agents.

        Parameters:
        file_path (str, optional): The path to the CSV file containing the data. If not provided, the function will use
                                   the data stored in `self.data`.

        Returns:
        """
        if self.__data_error__():
            return
        df = self.data
        # Calculate cumulative dwell time for each agent and state
        df['cumulative_dwell_time'] = df.groupby(['agent_id', 'state_name'])['dwell_time'].cumsum()

        # Convert dwell time to suplied step_time_units
        df['cumulative_dwell_time_units'] = df['cumulative_dwell_time']#/24

        # Ensure column order matches color mapping
        state_colors, cmap = Utils.colors()
        matching_colors = [state_colors[state] for state in df['state_name'].unique() if state in state_colors]
        cmap = mcolors.ListedColormap(matching_colors)


        # Pivot the data to get cumulative dwell time for each state
        pivot_df = df.pivot_table(index='agent_id', columns='state_name', values='cumulative_dwell_time_units', aggfunc='max', fill_value=0)
        pivot_df.plot(kind='bar', stacked=True, figsize=(15, 40), colormap=cmap )

        # Plot the data
        # pivot_df.plot(kind='bar', stacked=True, figsize=(15, 7))
        plt.title('Cumulative Time in step_time_units on Each State for All Agents')
        plt.annotate('DwtPlotter.stacked_bars_states_per_agent_static()', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=12)
        plt.xlabel('Agent ID')
        plt.ylabel('Cumulative Time (step_time_units)')
        plt.legend(title='State Name')
        plt.tight_layout()
        plt.show()


    def reinfections_age_bins_bars_interactive(self, target_states, barmode = 'group', scenario=''):
        """
        Plots the maximum number of reinfections for agents and groups them interactively using Plotly.

        This method calculates the maximum number of reinfections for each agent
        and generates an interactive bar plot to visualize the distribution of reinfections by age bins.

        Returns:
            None
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
        age_labels = [f'{int(b)}-{int(age_bins[i+1])}' if age_bins[i+1] != np.inf else f'{int(b)}+' for i, b in enumerate(age_bins[:-1])]
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
        Plots the maximum number of reinfections for agents and groups them interactively using Plotly.

        This method calculates the maximum number of reinfections for each agent
        and generates an interactive bar plot to visualize the distribution of reinfections by age bins.

        Returns:
            None
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
        Plots the maximum number of reinfections for agents and groups them interactively using Plotly.

        This method calculates the maximum number of reinfections for each agent
        and generates an interactive bar plot to visualize the distribution of reinfections by state transitions.

        Returns:
            None
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
        Generates an interactive stacked bar chart of dwell times by state using Plotly.

        Parameters:
        bin_size (int): The size of each bin for dwell times. Default is 50.
        num_bins (int): The number of bins to divide the dwell times into. Default is 20.
        
        Returns:
        None: Displays an interactive Plotly figure.
        
        Notes:
        - If the self.data DataFrame is empty, the function will print a message and return without plotting.
        - The function creates bins for dwell times and labels them in step_time_units.
        - It generates a stacked bar chart for each state, showing the count of transitions to other states within each dwell time bin.
        - The height of the figure is dynamically adjusted based on the number of states.
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
        Plots binned stacked bar charts for state transitions based on dwell times.

        Parameters:
        bin_size (int): The size of each bin for dwell times in step_time_units. Default is 50.
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
            state_data = self.data[self.data['state_name'] == state]
            state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

            # Group by dwell time bins and going to state
            grouped = state_data.groupby(['dwell_time_bin', 'going_to_state']).size().unstack(fill_value=0)

            # Ensure column order matches color mapping
            state_colors, cmap = Utils.colors()
            matching_colors = [state_colors[state] for state in grouped.columns if state in state_colors]
            cmap = mcolors.ListedColormap(matching_colors)

            # Plot stacked bar chart
            grouped.plot(kind='bar', stacked=True, ax=ax, colormap=cmap) #'tab20')
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
            state_data = df[df['state_name'] == state]

            # Automatically define the number of bins and bin size based on the data
            max_dwell_time = state_data['dwell_time'].max()
            bin_size = max(1, max_dwell_time // 15)  # Ensure at least 15 bins
            bins = np.arange(0, max_dwell_time + bin_size, bin_size)
            bin_labels = [f"{int(b)}-{int(b+bin_size)} step_time_units" for b in bins[:-1]]

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
        plt.annotate('DwtPlotter.histogram_with_kde()', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        return
    
    def graph_state_transitions(self, states=None, subtitle="", layout=None, curved_ratio=0.1, colormap='Paired', onlymodel=True):
        """
        Plot a state transition graph with mean and mode dwell times annotated on the edges.
        
        Parameters:
        -----------
        curved_ratio (float, optional): Ratio to curve the edges. Default is 0.05.
        colormap (str, optional): Name of the colormap to use for coloring nodes. Default is 'tab20'.
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

        # Add edges with mean and mode annotations
        for _, row in stats_df.iterrows():
            from_state = row['state_name']
            to_state = row['going_to_state']
            mean_dwell = round(row['mean'], 2) if not pd.isna(row['mean']) else "N/A"
            mode_dwell = round(row['mode'], 2) if not pd.isna(row['mode']) else "N/A"
            num_agents = row['count']

            G.add_edge(from_state, to_state,
                label=f"Mean: {mean_dwell}\nMo: {mode_dwell}\nAgents: {num_agents}")

        pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
        colors =plt.colormaps.get_cmap(colormap) 
        node_colors = [colors(i) for i in range(len(G.nodes))]
        edge_colors = [node_colors[list(G.nodes).index(edge[0])] for edge in G.edges]
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
        Plot a state transition graph with curved edges, where edge thickness is proportional to agent count.
        
        Parameters:
        -----------
        states (list, optional): A list of states to include in the graph. If None, all states in self.data will be included.
        subtitle (str, optional): Subtitle for the graph.
        layout (int, optional): Layout type for node positioning.
        curved_ratio (float, optional): Curve factor for edges.
        colormap (str, optional): Matplotlib colormap for node coloring.
        onlymodel (bool, optional): If True, exclude certain state transitions.
        graphseed (int, optional): Random seed for layout consistency.

        Returns:
        --------
        None: Displays a directed graph of state transitions.
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

        # Choose layout
        pos = nx.spring_layout(G, seed=graphseed)  

        # Color nodes and edges
        colors = plt.get_cmap(colormap)
        node_colors = [colors(i / max(1, len(G.nodes))) for i in range(len(G.nodes))]
        edge_colors = [node_colors[list(G.nodes).index(edge[0])] for edge in G.edges]

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
        if self.__data_error__():
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
        Plots a Kaplan-Meier survival curve for the given data.

        Parameters:
            data (pd.DataFrame): Input DataFrame containing survival data.
            dwell_time_col (str): Column name representing dwell times.
            event_observed_col (str, optional): Column indicating if the event was observed (1) or censored (0).
                If None, assumes all events are observed.

        Returns:
            None: Displays the Kaplan-Meier survival plot.
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
        # data error handling - check if data is available

        if self.data is None or self.data.empty or 'dwell_time' not in self.data.columns:
            print("No dwell time data available to plot.")
            return True
        return False

    def __cleandata__(self, filename):

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
    def __select_graph_pos__(G, layout, seed=42):
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
    def __init__(self, directory='', prefix='', data=None, debug=False, **kwargs) :
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
        self.debug = debug

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
            if self.debug: print(f"Combined DataFrame saved to {output_file}")
        except Exception as e:
            print(f"Error saving DataFrame to {output_file}: {e}")
            
class DwtAnalyzer(ss.Analyzer, DwtPlotter):
    def __init__(self, adjust_to_unit=False, unit=1.0, states_ennumerator=mtb.TBS, scenario_name=''):
        """
        Initializes the analyzer with optional adjustments to step_time_units.

        Args:
            adjust_to_unit (bool): If True, adjusts the dwell times to step_time_units by multiplying the recorded dwell_time by the provided multiplier.
            Default is True.
            
            unit (float | ss.t ):  TODO: Implement its use.
            states_ennumerator (IntEnum): An IntEnum class that enumerates the states in the simulation. Default is mtb.TBS but it will accept any equivalent IntEnum class.


        Notes:
        Please note, states -2 has been added to represent Agents NEVER INFECTED state and state -3.0 is to represent NON-TB DEAD.

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
        self.__latest_sts_df__ = pd.DataFrame(columns=['agent_id', 'last_state', 'last_state_time'])      
        DwtPlotter.__init__(self, data=self.data)
        return
    
    def __initialize_dataframes__(self):
        # Initialize the latest state dataframe
        # NOTE: This module assumes the default state is '-1'
        if self.unit is None:
            self.unit = self.sim.pars.unit
        agent_ids = self.sim.people.auids
        population = len(agent_ids)
        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'last_state': np.full(population, -1.0),
            'last_state_time': np.zeros(population)
        })
        return
    
    def step(self):
        if not self.sim.ti: self.__initialize_dataframes__()
        self.__check_for_new_borns__()       
        self.__update_state_change_data__()
        self.__record_natural_deaths__()
        return
    
    def __update_state_change_data__(self):
        # Get the current state of the agents
        ti = self.ti
        tb = self.sim.diseases.tb
        uids = self.sim.people.auids  # People Alive

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
            going_to_state_ids=tb.state[uids],
            age=self.sim.people.age[uids]
        )

        # Update the latest state dataframe with the new state
        self.__latest_sts_df__.loc[self.__latest_sts_df__['agent_id'].isin(uids), 'last_state'] = tb.state[uids]
        self.__latest_sts_df__.loc[self.__latest_sts_df__['agent_id'].isin(uids), 'last_state_time'] = ti
        return

    # TODO:  IN PROGRESS
    def __record_natural_deaths__(self):
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
                age= self.sim.people.age[ss.uids(relevant_rows['agent_id'].values)]
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
        if self.adjust_to_unit:
            self.data['dwell_time_raw'] = self.data['dwell_time']   # Save the original recorded values for comparison or later use
            if isinstance(self.unit, (float, int)):
                self.data['dwell_time'] = self.data['dwell_time'] * self.unit   
            elif isinstance(self.unit, str):
                self.data['dwell_time'] = self.data['dwell_time'] * (self.sim.pars.dt / ss.rate(self.unit))
                # self.data['dwell_time'] = self.data['dwell_time'].apply(lambda x: eval(f"{x} {self.unit}"))
        self.file_path = self.__save_to_file__()
        return

    def _log_dwell_time(self, agent_ids, states, entry_times, exit_times, going_to_state_ids, age):
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
    def to_filename_friendly(string=''):
        import re
        string = "".join([c if c.isalpha() else "_" for c in string])
        return re.sub(r'[^a-zA-Z0-9]', '', string)
    
    def colors():
        """
        Returns a ListedColormap and corresponding state labels for TB states.
        
        Returns:
            cmap (ListedColormap): Colormap object for use in plots.
            state_labels (list): Ordered state names.
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

    debug = 1

    if debug == 0:
        # # # Initialize the DwtAnalyzer
        import starsim as ss
        sim = ss.Sim()
        
        sim.add_analyzer(DwtAnalyzer())
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


 