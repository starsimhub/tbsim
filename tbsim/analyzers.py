import starsim as ss
import pandas as pd
import numpy as np
import os
import tbsim.config as cfg
import datetime as ddtt
import tbsim.plotdwelltimes as pdt
import tbsim as mtb
import matplotlib.pyplot as plt
from scipy import stats
from pandas.plotting import parallel_coordinates
import plotly.express as px

class DTAn(ss.Module):
    def __init__(self):
        super().__init__()
        return

    def init_results(self):
        super().init_results()
        self.latest_sts_df = pd.DataFrame(columns=['agent_id', 'last_state', 'last_state_time'])
        self.dwell_time_logger = pd.DataFrame(columns=['agent_id', 'state', 'entry_time', 'exit_time', 'dwell_time', 'state_name', 'going_to_state'])
        
        agent_ids = self.sim.people.auids
        population = len(agent_ids)
        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'last_state': np.full(population, -1),
            'last_state_time': np.zeros(population)
        })
        self.latest_sts_df = pd.concat([self.latest_sts_df, new_logs], ignore_index=True)
        return

    def step(self):
        sim = self.sim
        ti = sim.ti
        
        # check if the number of agents has changed
        if len(self.sim.people.auids) != len(self.latest_sts_df):
            #identify which agent ids are new and add them to the latest_sts_df
            new_agent_ids = list(set(self.sim.people.auids) - set(self.latest_sts_df.agent_id))
            new_logs = pd.DataFrame({
                'agent_id': new_agent_ids,
                'last_state': np.full(len(new_agent_ids), -1),
                'last_state_time': np.zeros(len(new_agent_ids))
            })
            self.latest_sts_df = pd.concat([self.latest_sts_df, new_logs], ignore_index=True)
        return
    
    def update_results(self):
        super().update_results()

        tb = self.sim.diseases.tb
        ti = self.ti
        uids = self.sim.people.auids

        # Filter rows in latest_sts_df for the relevant agents
        relevant_rows = self.latest_sts_df[self.latest_sts_df['agent_id'].isin(uids)]

        # Identify agents whose last recorded state is different from the current state
        different_state_mask = relevant_rows['last_state'].values != tb.state[ss.uids(relevant_rows['agent_id'].values)]
        uids = ss.uids(relevant_rows['agent_id'].values[different_state_mask])

        # Log dwell times
        self.log_dwell_time(
            agent_ids=uids,
            states=relevant_rows['last_state'].values[different_state_mask],
            entry_times=relevant_rows['last_state_time'].values[different_state_mask],
            exit_times=np.full(len(uids), ti),
            going_to_states=tb.state[uids]
        )

        # Update the latest state dataframe
        self.latest_sts_df.loc[self.latest_sts_df['agent_id'].isin(uids), 'last_state'] = tb.state[uids]
        self.latest_sts_df.loc[self.latest_sts_df['agent_id'].isin(uids), 'last_state_time'] = ti


    def finalize(self):
        super().finalize()
        self.dwell_time_logger['state_name'] = self.dwell_time_logger['state'].apply(lambda x: mtb.TBS(x).name.replace('_', ' ').title())
        self.dwell_time_logger['going_to_state'] = self.dwell_time_logger['going_to_state'].apply(lambda x: mtb.TBS(x).name.replace('_', ' ').title())
        self.file_name = self.save_to_file()

    def finalize_results(self):
        super().finalize_results()
        print(self.ti)
        print(self.latest_sts_df)
        print(self.dwell_time_logger)
        return

    def log_dwell_time(self, agent_ids, states, entry_times, exit_times, going_to_states):
        entry_times = np.nan_to_num(entry_times, nan=0)
        dwell_times = exit_times - entry_times

        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'state': states,
            'entry_time': entry_times,
            'exit_time': exit_times,
            'dwell_time': dwell_times,
            'going_to_state': going_to_states
        })
        self.dwell_time_logger = pd.concat([self.dwell_time_logger, new_logs], ignore_index=True)
                # Map state codes to their corresponding names

    def save_to_file(self):
        resdir = os.path.dirname(cfg.create_res_dir())
        t = ddtt.datetime.now()
        fn = os.path.join(resdir, f'dwell_time_logger_{t.strftime("%Y%m%d%H%M%S")}.csv')
        self.dwell_time_logger.to_csv(fn, index=False)
        print(f"Dwell time logs saved to {fn}")

        return fn

    def validate_dwell_time_distributions(self, expected_distributions=None):
        """
        Validate dwell times against expected distributions using statistical tests.
        """
        from scipy.stats import ks_1samp, ks_2samp

        
        expected_distributions = expected_distributions or self.expected_distributions
       
        print("Validating dwell time distributions...")
        for state, expected_cdf in expected_distributions.items():
            dwell_times = self.dwell_time_logger[self.dwell_time_logger['state'] == state]['dwell_time']
            if dwell_times.empty:
                print(f"No data available for state {state}")
                continue
            stat, p_value = stats.kstest(dwell_times, expected_cdf)
            print(f"State {state}: KS Statistic={stat:.4f}, p-value={p_value:.4f}")
            if p_value < 0.05:
                print(f"WARNING: Dwell times for state {state} deviate significantly from expectations.")

    def plot_dwell_time_validation(self):
        """
        Plot the results of the dwell time validation.
        """

        fig, ax = plt.subplots()
        for state in self.dwell_time_logger['state'].unique():
            dwell_times = self.dwell_time_logger[self.dwell_time_logger['state'] == state]['dwell_time']
            if dwell_times.empty:
                continue
            state_label = mtb.TBS(state).name.replace('_', ' ').title()
            ax.hist(dwell_times, bins=50, alpha=0.5, label=f'{state_label}')
            ax.hist(dwell_times, bins=50, alpha=0.5, label=f'{state}')
        ax.set_xlabel('Dwell Time')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.show()
        return
    
    def plot_dwell_time_validation_interactive(self):
        """
        Plot the results of the dwell time validation interactively using Plotly.
        """
        import plotly.express as px
        fig = px.histogram(self.dwell_time_logger, x='dwell_time', color='state_name', 
                            nbins=50, barmode='overlay', 
                            labels={'dwell_time': 'Dwell Time', 'state_name': 'State'},
                            title='Dwell Time Validation')
        fig.update_layout(bargap=0.1)
        fig.show()
        return

    
    def plot_agent_dynamics(self, dwell_time_bins=None, filter_states=None):
        """
        Plot the state transitions and/or dwell time distributions of agents interactively,
        with dwell times grouped into predefined ranges.

        Parameters:
        - dwell_time_bins (list): List of bin edges for grouping dwell times.
                                  Default is [0, 50, 100, 150, 200, 250, np.inf].
        - filter_states (list): List of states to include in the plot. If None, include all states.
        """
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go

        if self.dwell_time_logger.empty:
            print("No dwell time data available to plot.")
            return

        # Set default bins if none are provided
        if dwell_time_bins is None:
            dwell_time_bins = [0, 50, 100, 150, 200, 250, np.inf]

        # Create bin labels, handling infinity separately
        dwell_time_labels = [
            f"{int(b)}-{int(d)} days" if d != np.inf else f"{int(b)}+ days"
            for b, d in zip(dwell_time_bins[:-1], dwell_time_bins[1:])
        ]

        # Create a dwell time category column
        self.dwell_time_logger['dwell_time_category'] = pd.cut(
            self.dwell_time_logger['dwell_time'],
            bins=dwell_time_bins,
            labels=dwell_time_labels,
            include_lowest=True
        )

        # Apply state filter if provided
        if filter_states is not None:
            filtered_logger = self.dwell_time_logger[
                self.dwell_time_logger['state_name'].isin(filter_states) |
                self.dwell_time_logger['going_to_state'].isin(filter_states)
            ]
        else:
            filtered_logger = self.dwell_time_logger

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
            going_to_state_label = mtb.TBS(row['going_to_state']).name.replace('_', ' ').title()
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


    def plot_state_transition_graph_static(self):
        """
        Plot a state transition graph with mean and mode dwell times annotated on the edges.
        """
        import networkx as nx
        import itertools as it

        if self.dwell_time_logger.empty:
            print("No data available to plot.")
            return

        # Calculate mean, mode, and count for each state transition
        transitions = self.dwell_time_logger.groupby(['state_name', 'going_to_state'])['dwell_time']
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
                    label=f"Mean: {mean_dwell}\nMode: {mode_dwell}\nAgents: {num_agents}")

        # Generate a layout for the graph
        pos = nx.spring_layout(G, seed=42)  # Use spring layout for better visualization

        # Draw nodes and edges with curved lines
        colors = plt.cm.get_cmap('tab20', len(G.nodes))
        node_colors = [colors(i) for i in range(len(G.nodes))]
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=10, edge_color="black") #connectionstyle="arc3,rad=0.2")
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black", font_weight="bold")

        # Annotate edges with mean and mode
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Display the graph
        plt.title("State Transition Graph with Dwell Times")
        plt.show()
        return
        

    def plot_stacked_bars_by_state(self, bin_size=50):
        """
        Plot stacked bar charts for each state showing the distribution of dwell times in configurable bins.

        Parameters:
        - bin_size (int): Size of each bin for grouping dwell times. Default is 25 days.
        """
        import matplotlib.pyplot as plt

        if self.dwell_time_logger.empty:
            print("No dwell time data available to plot.")
            return

        # Define bins for dwell times
        # bins = np.arange(0, self.dwell_time_logger['dwell_time'].max() + bin_size, bin_size)
        # bin_labels = [f"{int(b)}-{int(b+bin_size)} days" for b in bins[:-1]]

        bins = np.arange(0, bin_size*8, bin_size)
        bin_labels = [f"{int(b)}-{int(b+bin_size)} days" for b in bins[:-1]]

        # Create a figure with subplots for each state
        states = self.dwell_time_logger['state_name'].unique()
        num_states = len(states)
        fig, axes = plt.subplots(num_states, 1, figsize=(20, 5 * num_states), sharex=True)

        if num_states == 1:
            axes = [axes]

        for ax, state in zip(axes, states):
            state_data = self.dwell_time_logger[self.dwell_time_logger['state_name'] == state]
            state_data['dwell_time_bin'] = pd.cut(state_data['dwell_time'], bins=bins, labels=bin_labels, include_lowest=True)

            # Group by dwell time bins and going to state
            grouped = state_data.groupby(['dwell_time_bin', 'going_to_state']).size().unstack(fill_value=0)

            # Plot stacked bar chart
            grouped.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
            ax.set_title(f'State: {state}')
            ax.set_xlabel('Dwell Time Bins')
            ax.set_ylabel('Count')
            ax.legend(title='Going to State')

        plt.tight_layout()
        plt.show()