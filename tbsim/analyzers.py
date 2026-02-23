"""TBsim custom analyzers"""


import matplotlib.colors as mcolors
import pandas as pd
import starsim as ss
import numpy as np
import os
import sciris as sc
import datetime as ddtt
import tbsim
from scipy import stats
from enum import IntEnum
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import networkx as nx
import plotly.graph_objects as go
import warnings

__all__ = ['DwellTime', 'DwtAnalyzer', 'DwtPlotter', 'DwtPostProcessor']

warnings.simplefilter(action='ignore', category=FutureWarning)


class DwellTime(ss.Analyzer):
    """
    Unified dwell-time analysis, aggregation, and visualization.

    Constructor mode is auto-detected from the arguments supplied:

    * **Plotter mode** – pass ``data`` (DataFrame) or ``file_path`` (CSV path).
    * **Aggregate mode** – pass ``directory`` (and optional ``prefix``).
    * **Analyzer mode** – pass none of the above; the instance will be
      attached to a Starsim simulation and record dwell times at each step.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Pre-loaded dwell-time DataFrame.
    file_path : str, optional
        Path to a single CSV file with dwell-time data.
    directory : str, optional
        Directory containing CSV result files for aggregation.
    prefix : str, optional
        File prefix filter used with *directory*.
    states_ennumerator : IntEnum, optional
        State enumeration class (default ``tbsim.TBS``).
    scenario_name : str, optional
        Label for the simulation scenario.
    debug : bool, optional
        Enable verbose output.

    Visualization types (via ``plot(kind=...)``):

    - ``'sankey'`` – Sankey diagram of agent state transitions
    - ``'histogram'`` – Histograms with KDE per state
    - ``'kaplan_meier'`` – Kaplan-Meier survival curves
    - ``'network'`` – Directed graph of state transitions
    - ``'validation'`` – Observed dwell-time histograms for validation

    Examples
    --------
    Direct data visualization::

        dt = DwellTime(file_path='results/Baseline-20240101120000.csv')
        dt.plot('sankey')

    Aggregating multiple runs::

        dt = DwellTime(directory='results', prefix='Baseline')
        dt.plot('network')

    During simulation::

        sim = ss.Sim(diseases=[TB()], analyzers=DwellTime(scenario_name="Baseline"))
        sim.run()
        sim.analyzers[0].plot('validation')
    """

    def __init__(self, data=None, file_path=None, directory=None, prefix='',
                 states_ennumerator=None, scenario_name='', debug=False):
        """Auto-detect mode (plotter, aggregate, or analyzer) from the supplied arguments."""
        self.debug = debug
        self.scenario_name = scenario_name
        self.eSTATES = states_ennumerator or tbsim.TBS
        self.file_path = file_path

        if isinstance(data, pd.DataFrame):
            # Plotter mode – data supplied directly
            self.data = data
        elif file_path is not None:
            # Plotter mode – load from CSV
            self.data = self._cleandata(file_path)
        elif directory is not None:
            # Aggregate mode
            self.directory = directory
            self.prefix = prefix
            agg = self._aggregate_simulation_results(directory, prefix)
            self.data = agg
        else:
            # Analyzer mode – will be attached to a simulation
            ss.Analyzer.__init__(self)
            self.data = pd.DataFrame(
                columns=['agent_id', 'state', 'entry_time', 'exit_time',
                         'dwell_time', 'state_name', 'going_to_state_id',
                         'going_to_state'])
            self._latest_sts_df = pd.DataFrame(
                columns=['agent_id', 'last_state', 'last_state_time'])
            return  # skip data-error check for analyzer mode

        if self._data_error():
            print("No data provided, or data is corrupted")

    # ------------------------------------------------------------------
    # plot() dispatcher
    # ------------------------------------------------------------------

    _PLOT_KINDS = None  # populated after method definitions

    def plot(self, kind='sankey', **kwargs):
        """
        Create a visualization of the dwell-time data.

        Parameters
        ----------
        kind : str
            One of ``'sankey'``, ``'histogram'``, ``'kaplan_meier'``,
            ``'network'``, ``'validation'``.
        **kwargs
            Forwarded to the underlying plot method.

        Raises
        ------
        ValueError
            If *kind* is not recognised.
        """
        kinds = self._PLOT_KINDS
        if kind not in kinds:
            valid = ', '.join(sorted(kinds))
            raise ValueError(
                f"Unknown plot kind {kind!r}. Valid kinds: {valid}")
        return kinds[kind](self, **kwargs)

    # ------------------------------------------------------------------
    # Kept plotting methods (private)
    # ------------------------------------------------------------------

    def _plot_sankey(self, subtitle=""):
        """Sankey diagram of agent state transitions."""
        if self._data_error():
            return

        df = self.data
        source = df['state_name']
        target = df['going_to_state']

        transition_counts = df.groupby(
            ['state_name', 'going_to_state']).size().reset_index(name='count')

        labels = list(set(source) | set(target))
        label_to_index = {label: i for i, label in enumerate(labels)}

        source_indices = transition_counts['state_name'].map(label_to_index)
        target_indices = transition_counts['going_to_state'].map(label_to_index)
        values = transition_counts['count']

        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        node_colors = [
            f'rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, 1.0)' for c in colors]
        link_colors_base = [
            f'rgba({c[0]*255}, {c[1]*255}, {c[2]*255}, 0.5)' for c in colors]
        link_color_map = {i: link_colors_base[i] for i in range(len(labels))}
        link_colors = [link_color_map[idx] for idx in source_indices]

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15, thickness=20,
                line=dict(color="black", width=0.2),
                label=labels, color=node_colors),
            link=dict(
                source=source_indices, target=target_indices,
                value=values, color=link_colors,
                hovertemplate='%{source.label} → %{target.label}: %{value} agents<br>',
                line=dict(color="lightgray", width=0.1),
                label=values)))

        fig.update_layout(
            hovermode='x',
            title=dict(
                text=f"State Transitions  - Agents Count<br>{subtitle}",
                font=dict(size=12)),
            font=dict(size=12, color='black'))
        fig.show()

    def _plot_histogram(self, subtitle=""):
        """Histograms with KDE for dwell-time distributions per state."""
        if self._data_error():
            return

        df = self.data
        states = df['state_name'].unique()
        num_states = len(states)
        num_cols = 4
        num_rows = (num_states + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(20, 5 * num_rows), sharex=False)

        axes = axes.flatten()
        fig.suptitle(
            f'State Transitions by Dwell Time Bins {subtitle}', fontsize=16)

        for ax, state in zip(axes, states):
            state_data = df[df['state_name'] == state].copy()
            max_dwell_time = state_data['dwell_time'].max()
            bin_size = max(1, max_dwell_time // 15)
            bins = np.arange(0, max_dwell_time + bin_size, bin_size)
            bin_labels = [
                f"{int(b)}-{int(b+bin_size)} step_time_units"
                for b in bins[:-1]]

            state_data['dwell_time_bin'] = pd.cut(
                state_data['dwell_time'], bins=bins, labels=bin_labels,
                include_lowest=True)

            has_enough_data = True
            for _, group_data in state_data.groupby('going_to_state'):
                if group_data['dwell_time'].nunique() < 2:
                    has_enough_data = False
                    break

            sns.histplot(
                data=state_data, x='dwell_time', bins=bins,
                hue='going_to_state', kde=has_enough_data, palette='tab10',
                multiple='stack', legend=True, ax=ax)

            ax.set_title(f'State: {state}')
            ax.set_xlabel('Dwell Time Bins')
            ax.set_ylabel('Count')
            handles, labels_ = ax.get_legend_handles_labels()
            if len(handles) > 0:
                ax.legend(title='Going to State', loc='upper right')

        for i in range(num_states, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def _plot_kaplan_meier(self, dwell_time_col='dwell_time',
                           event_observed_col=None):
        """Kaplan-Meier survival curve for dwell times."""
        if self._data_error():
            return

        data = self.data
        durations = data[dwell_time_col]
        event_observed = (data[event_observed_col]
                          if event_observed_col else [1] * len(data))

        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=event_observed)

        plt.figure(figsize=(10, 6))
        kmf.plot_survival_function()
        plt.title("TBSim Kaplan-Meier Survival Curve", fontsize=16)
        plt.xlabel(f"Time ({dwell_time_col})", fontsize=14)
        plt.ylabel("Survival Probability", fontsize=14)
        plt.grid(True)
        plt.show()

    def _plot_network(self, states=None, subtitle="", layout=None,
                      curved_ratio=0.09, colormap='Paired', onlymodel=True,
                      graphseed=42):
        """Directed graph of state transitions with curved edges."""
        if self._data_error():
            return

        df = self.data
        if states is not None:
            df = df[df['state_name'].isin(states)]

        if onlymodel:
            df = df[~df['going_to_state_id'].isin([-3.0, -2.0])]

        transitions = df.groupby(
            ['state_name', 'going_to_state'])['dwell_time']

        stats_df = transitions.agg([
            'mean',
            lambda x: stats.mode(x, keepdims=True).mode[0]
            if len(x) > 0 else np.nan,
            'count'
        ]).reset_index()
        stats_df.columns = [
            'state_name', 'going_to_state', 'mean', 'mode', 'count']

        plt.figure(figsize=(15, 10), facecolor='black')
        G = nx.DiGraph()

        max_agents = (stats_df['count'].max()
                      if not stats_df['count'].isna().all() else 1)

        for _, row in stats_df.iterrows():
            from_state = row['state_name']
            to_state = row['going_to_state']
            mean_dwell = (round(row['mean'], 2)
                          if not pd.isna(row['mean']) else "N/A")
            mode_dwell = (round(row['mode'], 2)
                          if not pd.isna(row['mode']) else "N/A")
            num_agents = row['count']
            edge_thickness = 1 + (4 * num_agents / max_agents)
            fs_id = (f"{from_state.split('.')[0]} → "
                     f"{to_state.split('.')[0]}\n")
            G.add_edge(
                from_state, to_state,
                label=(f"{fs_id},  μ:{mean_dwell},  Mo: {mode_dwell}\n"
                       f"Agents:{num_agents}"),
                weight=edge_thickness)

        if layout is not None:
            pos = self._select_graph_pos(G, layout, seed=graphseed)
        else:
            pos = nx.spring_layout(G, seed=graphseed)

        cmap = plt.colormaps.get_cmap(colormap)
        node_colors = [
            cmap(i / max(1, len(G.nodes) - 1)) for i in range(len(G.nodes))]
        edge_colors = [
            cmap(i / max(1, len(G.edges) - 1)) for i in range(len(G.edges))]

        edge_labels = nx.get_edge_attributes(G, 'label')
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]

        plt.figure(facecolor='black')
        nx.draw_networkx_nodes(
            G, pos, node_size=300, node_color=node_colors,
            edgecolors="lightgray")
        nx.draw_networkx_edges(
            G, pos, width=edge_weights, alpha=0.8, arrowstyle="-|>",
            arrowsize=30, edge_color=edge_colors,
            connectionstyle=f'arc3,rad={curved_ratio}')
        nx.draw_networkx_labels(
            G, pos, font_size=11, font_color="black", font_weight="bold")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=7)

        plt.title(
            f"State Transition Graph - By Agents Count: {subtitle}",
            color='white')
        plt.tight_layout()
        plt.show()

    def _plot_validation(self):
        """Overlaid histograms for dwell-time validation."""
        if self._data_error():
            return

        fig, ax = plt.subplots()
        data = self.data
        model_states = data['state_name'].unique()
        plt.figure(figsize=(15, 10), facecolor='black')
        for state in model_states:
            dwell_times = data[data['state_name'] == state]['dwell_time']
            if dwell_times.empty:
                continue
            ax.hist(dwell_times, bins=50, alpha=0.5, label=f'{state}')
        ax.set_xlabel('Dwell Time')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.show()

    # ------------------------------------------------------------------
    # Aggregate helpers (from former DwtPostProcessor)
    # ------------------------------------------------------------------

    def _aggregate_simulation_results(self, directory, prefix):
        """Aggregate multiple CSV files into a single DataFrame."""
        import glob

        file_pattern = os.path.join(directory, f"{prefix}*.csv")
        file_list = glob.glob(file_pattern)

        if not file_list:
            print(f"No files found matching pattern: {file_pattern}")
            return pd.DataFrame()
        else:
            print(
                f"Found {len(file_list)} files matching pattern: "
                f"{file_pattern}")
            if self.debug:
                print("\n".join(file_list))

        data_frames = []
        for index, file in enumerate(file_list):
            try:
                df = pd.read_csv(file)
                df['agent_id'] = df['agent_id'] + ((index + 1) * 10000)
                data_frames.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if not data_frames:
            print(f"No valid CSV files to aggregate for prefix: {prefix}")
            return pd.DataFrame()

        return pd.concat(data_frames, ignore_index=True)

    def save_combined_dataframe(self, output_file):
        """Save the combined DataFrame to a CSV file."""
        if self.data is None or self.data.empty:
            print("No data available to save.")
            return
        try:
            self.data.to_csv(output_file, index=False)
            if self.debug:
                print(f"Combined DataFrame saved to {output_file}")
        except Exception as e:
            print(f"Error saving DataFrame to {output_file}: {e}")

    # ------------------------------------------------------------------
    # Analyzer methods (from former DwtAnalyzer)
    # ------------------------------------------------------------------

    def _initialize_dataframes(self):
        """Initialize internal DataFrames on the first simulation step."""
        agent_ids = self.sim.people.auids.copy()
        population = len(agent_ids)
        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'last_state': np.full(population, -1.0),
            'last_state_time': np.zeros(population)
        })
        # The original code did not assign new_logs; keeping the same
        # behaviour (the _check_for_new_borns call fills _latest_sts_df).
        return

    def step(self):
        """Execute one time step of dwell time analysis."""
        if not self.sim.ti:
            self._initialize_dataframes()
        self._check_for_new_borns()
        self._update_state_change_data()
        self._record_natural_deaths()

    def _update_state_change_data(self):
        """Detect state changes and record dwell times."""
        ti = self.ti
        tb = self.sim.diseases.tb
        uids = self.sim.people.auids.copy()

        relevant_rows = self._latest_sts_df[
            self._latest_sts_df['agent_id'].isin(uids)]

        different_state_mask = (
            relevant_rows['last_state'].values
            != tb.state[ss.uids(relevant_rows['agent_id'].values)])
        uids = ss.uids(
            relevant_rows['agent_id'].values[different_state_mask])

        self._log_dwell_time(
            agent_ids=uids,
            states=relevant_rows['last_state'].values[different_state_mask],
            entry_times=relevant_rows['last_state_time'].values[
                different_state_mask],
            exit_times=np.full(len(uids), ti),
            going_to_state_ids=tb.state[uids].copy(),
            age=self.sim.people.age[uids].copy())

        self._latest_sts_df.loc[
            self._latest_sts_df['agent_id'].isin(uids),
            'last_state'] = tb.state[uids]
        self._latest_sts_df.loc[
            self._latest_sts_df['agent_id'].isin(uids),
            'last_state_time'] = ti

    def _record_natural_deaths(self):
        """Record dwell times for agents who died from natural causes."""
        ti = self.ti
        dead_uids = ss.uids(self.sim.people.dead)
        if not dead_uids.all():
            return

        relevant_rows = self._latest_sts_df[
            self._latest_sts_df['agent_id'].isin(dead_uids)
            & (self._latest_sts_df['last_state'] < 0)]

        relevant_rows = relevant_rows[
            ~relevant_rows['agent_id'].isin(
                ss.uids(self.data['agent_id']))]
        if not relevant_rows.empty:
            self._log_dwell_time(
                agent_ids=relevant_rows['agent_id'].values,
                states=relevant_rows['last_state'].values,
                entry_times=relevant_rows['last_state_time'].values,
                exit_times=np.full(len(relevant_rows), ti),
                going_to_state_ids=np.full(len(relevant_rows), -3.0),
                age=self.sim.people.age[
                    ss.uids(relevant_rows['agent_id'].values)])

    def _check_for_new_borns(self):
        """Add newly born agents to the state tracking system."""
        if len(self.sim.people.auids) != len(self._latest_sts_df):
            new_agent_ids = list(
                set(self.sim.people.auids)
                - set(self._latest_sts_df.agent_id))
            new_logs = pd.DataFrame({
                'agent_id': new_agent_ids,
                'last_state': np.full(len(new_agent_ids), -1.0),
                'last_state_time': np.zeros(len(new_agent_ids))
            })
            self._latest_sts_df = pd.concat(
                [self._latest_sts_df, new_logs], ignore_index=True)

        if len(self.sim.people.auids) != len(self._latest_sts_df):
            new_agent_ids = list(
                set(self.sim.people.auids)
                - set(self._latest_sts_df.agent_id))
            new_logs = pd.DataFrame({
                'agent_id': new_agent_ids,
                'last_state': np.full(len(new_agent_ids), -1.0),
                'last_state_time': np.zeros(len(new_agent_ids))
            })
            if not new_logs.empty:
                self._latest_sts_df = pd.concat(
                    [self._latest_sts_df,
                     new_logs.loc[:, ~new_logs.isna().all()]],
                    ignore_index=True, copy=False)

    def finalize(self):
        """Finalize the analysis: map state names and save to file."""
        super().finalize()

        relevant_rows = self._latest_sts_df[
            (self._latest_sts_df['last_state'] == -1)
            & (self._latest_sts_df['last_state_time'] == 0.0)]
        if not relevant_rows.empty:
            self._log_dwell_time(
                agent_ids=relevant_rows['agent_id'].values,
                states=relevant_rows['last_state'].values,
                entry_times=relevant_rows['last_state_time'].values,
                exit_times=np.full(len(relevant_rows), self.sim.ti),
                going_to_state_ids=np.full(len(relevant_rows), -2.0),
                age=self.sim.people.age[
                    ss.uids(relevant_rows['agent_id'].values)].copy())

        if 'LSHTM' in str(self.sim.diseases[0].__class__):
            print("====> Using model: str(self.sim.diseases[0].__class__)")
            import tb_acf as tbacf
            self.eSTATES = tbacf.TBSL

        state_dict = {
            state.value: state.name.replace('_', ' ').title()
            for state in self.eSTATES}
        state_dict[-3] = 'NON-TB DEATH'
        state_dict[-2] = 'NEVER INFECTED'
        self.data['state_name'] = self.data['state'].map(state_dict)
        self.data['going_to_state'] = self.data[
            'going_to_state_id'].map(state_dict)
        self.data['going_to_state'] = self.data.apply(
            lambda row: f"{row['going_to_state_id']}.{row['going_to_state']}",
            axis=1)
        self.data['state_name'] = self.data.apply(
            lambda row: f"{row['state']}.{row['state_name']}", axis=1)
        self.data['state_name'] = self.data['state_name'].replace(
            'None', 'Susceptible')

        self.file_path = self._save_to_file()

    def _log_dwell_time(self, agent_ids, states, entry_times, exit_times,
                        going_to_state_ids, age):
        """Record dwell time data for a batch of state transitions."""
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
        self.data = pd.concat([self.data, new_logs], ignore_index=True)

    def _save_to_file(self):
        """Save dwell time data to CSV and metadata files."""
        resdir = os.path.join(os.getcwd(), 'results')
        os.makedirs(resdir, exist_ok=True)
        t = ddtt.datetime.now()
        prefix = sc.sanitizefilename(self.scenario_name)
        if not prefix:
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
        """Validate dwell time distributions using KS tests."""
        expected_distributions = (
            expected_distributions or self.expected_distributions)

        print("Validating dwell time distributions...")
        for state, expected_cdf in expected_distributions.items():
            dwell_times = self.data[
                self.data['state'] == state]['dwell_time']
            if dwell_times.empty:
                print(f"No data available for state {state}")
                continue
            stat, p_value = stats.kstest(dwell_times, expected_cdf)
            print(f"State {state}: KS Statistic={stat:.4f}, "
                  f"p-value={p_value:.4f}")
            if p_value < 0.05:
                print(f"WARNING: Dwell times for state {state} deviate "
                      f"significantly from expectations.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _data_error(self):
        """Return True if data is missing or invalid."""
        if (self.data is None or self.data.empty
                or 'dwell_time' not in self.data.columns):
            print("No dwell time data available to plot.")
            return True
        return False

    def _cleandata(self, filename):
        """Clean and validate dwell time data from a CSV file."""
        df = pd.read_csv(filename, dtype=str)
        df = df.dropna(subset=["agent_id"])

        numeric_columns = [
            "agent_id", "state", "entry_time", "exit_time",
            "dwell_time", "going_to_state_id", "age"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df_cleaned = df.dropna(subset=numeric_columns)
        df_cleaned.reset_index(drop=True, inplace=True)
        return df_cleaned

    @staticmethod
    def _select_graph_pos(G, layout=4, seed=42):
        """Select a NetworkX layout algorithm for graph visualizations."""
        if layout == 1:
            return nx.circular_layout(G)
        elif layout == 2:
            return nx.spiral_layout(G)
        elif layout == 3:
            return nx.spectral_layout(G)
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


# Wire up the plot dispatcher after the methods are defined
DwellTime._PLOT_KINDS = {
    'sankey': DwellTime._plot_sankey,
    'histogram': DwellTime._plot_histogram,
    'kaplan_meier': DwellTime._plot_kaplan_meier,
    'network': DwellTime._plot_network,
    'validation': DwellTime._plot_validation,
}

# Backward-compatibility aliases
DwtPlotter = DwellTime
DwtPostProcessor = DwellTime
DwtAnalyzer = DwellTime
