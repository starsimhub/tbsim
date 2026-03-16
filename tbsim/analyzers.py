"""TBsim custom analyzers"""


import os
import datetime as ddtt

import numpy as np
import pandas as pd
from scipy import stats
from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import sciris as sc
import starsim as ss
import tbsim

__all__ = ['DwellTime', 'DwtAnalyzer', 'DwtPlotter', 'DwtPostProcessor', 'HouseholdStats']


class DwellTime(ss.Analyzer):
    """
    Unified dwell-time analysis, aggregation, and visualization.

    Constructor mode is auto-detected from the arguments supplied:

    * **Plotter mode** -- pass ``data`` (DataFrame) or ``file_path`` (CSV path).
    * **Aggregate mode** -- pass ``directory`` (and optional ``prefix``).
    * **Analyzer mode** -- pass none of the above; the instance will be
      attached to a Starsim simulation and record dwell times at each step.

    Args:
        data (pd.DataFrame, optional):        Pre-loaded dwell-time DataFrame.
        file_path (str, optional):             Path to a single CSV file with dwell-time data.
        directory (str, optional):             Directory containing CSV result files for aggregation.
        prefix (str, optional):                File prefix filter used with *directory*.
        scenario_name (str, optional):         Label for the simulation scenario.
        debug (bool, optional):                Enable verbose output.

    Visualization types (via ``plot(kind=...)``):

    - ``'histogram'`` -- Histograms with KDE per state
    - ``'kaplan_meier'`` -- Kaplan-Meier survival curves
    - ``'network'`` -- Directed graph of state transitions
    - ``'validation'`` -- Observed dwell-time histograms for validation

    Example:

        Aggregating multiple runs::

            dt = DwellTime(directory='results', prefix='Baseline')
            dt.plot('network')

        During simulation::

            sim = ss.Sim(diseases=[TB_LSHTM()], analyzers=DwellTime(scenario_name="Baseline"))
            sim.run()
            sim.analyzers[0].plot('validation')
    """

    def __init__(self, data=None, file_path=None, directory=None, prefix='',
                 scenario_name='', debug=False):
        """Auto-detect mode (plotter, aggregate, or analyzer) from the supplied arguments."""
        self.debug = debug
        self.scenario_name = scenario_name
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

        return

    # ------------------------------------------------------------------
    # plot() dispatcher
    # ------------------------------------------------------------------

    _PLOT_KINDS = None  # populated after method definitions

    def plot(self, kind='histogram', **kwargs):
        """
        Create a visualization of the dwell-time data.

        Args:
            kind (str): One of ``'histogram'``, ``'kaplan_meier'``,
                ``'network'``, ``'validation'``.
            **kwargs: Forwarded to the underlying plot method.

        Raises:
            ValueError: If *kind* is not recognised.
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

        return

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

        return

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

        return

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

        return

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

        return

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

        return

    def _update_state_change_data(self):
        """Detect state changes and record dwell times."""
        ti = self.ti
        tb = tbsim.get_tb(self.sim)
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

        return

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

        return

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

        return

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

        state_dict = {
            state.value: state.name.replace('_', ' ').title()
            for state in tbsim.TBSL}
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
        self.data['state_name'] = self.data['state_name'].replace('None', 'Susceptible')

        return

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

        return

    def save(self, filename='dwelltime.csv'):
        """Save dwell time data to CSV and metadata files."""
        sc.makefilepath(filename, makedirs=True)
        self.data.to_csv(filename, index=False)
        return

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

        return

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
    def _select_graph_pos(G, layout='shell_layout', seed=42):
        """Select a NetworkX layout algorithm for graph visualizations."""
        func = getattr(x, layout)
        if layout == 'spring_layout':
            return func(G, seed=seed)
        else:
            return func(G)


# Wire up the plot dispatcher after the methods are defined
DwellTime._PLOT_KINDS = {
    'histogram': DwellTime._plot_histogram,
    'kaplan_meier': DwellTime._plot_kaplan_meier,
    'network': DwellTime._plot_network,
    'validation': DwellTime._plot_validation,
}

# Backward-compatibility aliases
DwtPlotter = DwellTime
DwtPostProcessor = DwellTime
DwtAnalyzer = DwellTime


class HouseholdStats(ss.Analyzer):
    """
    Track household size, age, and contact-mixing statistics over time.

    Works with any network that exposes a ``household_ids`` state (e.g.
    ``starsim.HouseholdNet``).  At each timestep the
    analyzer counts alive agents per household and records summary statistics.

    Args:
        network_name (str): Name of the household network on ``sim.networks``
            (default ``'householdnet'``).
        age_bins (tuple): Bin edges for the age distribution histogram
            (default ``(0, 5, 15, 50, 100)``).

    Results (per timestep):
        mean_hh_size, median_hh_size, max_hh_size, n_households,
        mean_hh_age, median_hh_age, mean_age, and one count per age bin.

    Plots:
        ``plot_stats()`` -- four-panel figure: household size over time,
            number of households, average household age over time, and
            initial vs. final household size distribution.
        ``plot_matrix()`` -- three-panel figure: age distribution over time,
            age-mixing matrix at simulation start, age-mixing matrix at end.
        ``plot_normalized_matrix()`` -- four-panel figure: contacts per person
            (C_ij / n_j) and proportionate mixing ratio (C_ij / E_ij) at
            simulation start and end.
        ``plot()`` -- calls all three of the above.

    Example::

        import starsim as ss
        import starsim_examples as sse
        import tbsim

        dhs_data = ...  # pandas DataFrame with hh_id and ages columns
        net = ss.HouseholdNet(dhs_data=dhs_data)
        analyzer = tbsim.HouseholdStats(network_name='householdnet')
        sim = ss.Sim(
            diseases='sis', networks=net,
            demographics=[ss.Pregnancy(fertility_rate=20), ss.Deaths(death_rate=10)],
            analyzers=analyzer,
        )
        sim.run()
        analyzer.plot()
    """

    def __init__(self, network_name='householdnet', age_bins=(0, 5, 15, 50, 100), **kwargs):
        super().__init__(**kwargs)
        self.network_name = network_name
        self.age_bins = list(age_bins)
        self.hh_size_hists = []         # per-timestep arrays of per-household sizes
        self.age_mixing_initial = None  # age-mixing matrix at t=0
        self.age_mixing_final = None    # age-mixing matrix at last step
        self.age_counts_initial = None  # per-bin population counts at t=0
        self.age_counts_final = None    # per-bin population counts at last step
        return

    def init_results(self):
        super().init_results()
        results = [
            ss.Result('mean_hh_size'),
            ss.Result('median_hh_size'),
            ss.Result('max_hh_size'),
            ss.Result('n_households'),
            ss.Result('mean_hh_age'),
            ss.Result('median_hh_age'),
            ss.Result('mean_age'),
        ]
        for lo, hi in zip(self.age_bins[:-1], self.age_bins[1:]):
            results.append(ss.Result(f'age_{lo}_{hi}'))
        self.define_results(*results)
        return

    def step(self):
        sim = self.sim
        ti = sim.ti
        net = sim.networks[self.network_name]
        ppl = sim.people
        alive = ppl.alive

        # --- Gather per-agent data and sort by household for efficient grouping ---
        hh_ids = net.household_ids[alive.uids]
        ages = ppl.age[alive.uids]

        sort_idx = np.argsort(hh_ids)
        hh_ids_sorted = hh_ids[sort_idx]
        ages_sorted = ages[sort_idx]

        unique_ids, counts = np.unique(hh_ids_sorted, return_counts=True)
        split_pts = np.cumsum(counts)[:-1]
        hh_age_groups = np.split(ages_sorted, split_pts)

        # --- Household sizes ---
        self.results['mean_hh_size'][ti] = np.mean(counts) if len(counts) else 0
        self.results['median_hh_size'][ti] = np.median(counts) if len(counts) else 0
        self.results['max_hh_size'][ti] = np.max(counts) if len(counts) else 0
        self.results['n_households'][ti] = len(unique_ids)
        self.hh_size_hists.append(counts.copy())

        # --- Per-household mean age ---
        hh_mean_ages = np.array([np.mean(g) for g in hh_age_groups])
        self.results['mean_hh_age'][ti] = np.mean(hh_mean_ages) if len(hh_mean_ages) else 0
        self.results['median_hh_age'][ti] = np.median(hh_mean_ages) if len(hh_mean_ages) else 0

        # --- Population age distribution ---
        self.results['mean_age'][ti] = np.mean(ages) if len(ages) else 0
        for lo, hi in zip(self.age_bins[:-1], self.age_bins[1:]):
            self.results[f'age_{lo}_{hi}'][ti] = np.sum((ages >= lo) & (ages < hi))

        # --- Age-mixing matrix (pairwise network contacts) ---
        n_bins = len(self.age_bins) - 1
        mix = np.zeros((n_bins, n_bins), dtype=float)
        p1, p2 = net.edges.p1, net.edges.p2
        if len(p1) > 0:
            bins_p1 = np.clip(np.digitize(ppl.age[p1], self.age_bins) - 1, 0, n_bins - 1)
            bins_p2 = np.clip(np.digitize(ppl.age[p2], self.age_bins) - 1, 0, n_bins - 1)
            np.add.at(mix, (bins_p1, bins_p2), 1)
            mix = (mix + mix.T) / 2

        age_counts = np.array([
            np.sum((ages >= lo) & (ages < hi))
            for lo, hi in zip(self.age_bins[:-1], self.age_bins[1:])
        ], dtype=float)

        if ti == 0:
            self.age_mixing_initial = mix.copy()
            self.age_counts_initial = age_counts.copy()
        self.age_mixing_final = mix          # updated every step; final value persists
        self.age_counts_final = age_counts   # updated every step; final value persists

        return

    def finalize_results(self):
        super().finalize_results()
        self.hh_size_hists = [np.array(h) for h in self.hh_size_hists]
        return

    def plot(self, **kwargs):
        """Plot all household statistics (calls plot_stats, plot_matrix, and plot_normalized_matrix)."""
        fig1 = self.plot_stats(**kwargs)
        fig2 = self.plot_matrix(**kwargs)
        fig3 = self.plot_normalized_matrix(**kwargs)
        return fig1, fig2, fig3

    def plot_stats(self, **kwargs):
        """Four-panel figure: household size, n_households, average household age, size distribution."""
        kw = ss.plot_args(kwargs)
        timevec = self.sim.t.timevec

        with ss.style(**kw.style):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # -- Household size over time --
            ax = axes[0, 0]
            ax.plot(timevec, self.results['mean_hh_size'], label='Mean')
            ax.plot(timevec, self.results['median_hh_size'], label='Median', linestyle='--')
            ax.set_ylabel('Household size')
            ax.set_xlabel('Year')
            ax.set_title('Household size over time')
            ax.legend()
            ax.set_ylim(bottom=0)

            # -- Number of households --
            ax = axes[0, 1]
            ax.plot(timevec, self.results['n_households'])
            ax.set_ylabel('Count')
            ax.set_xlabel('Year')
            ax.set_title('Number of households')
            ax.set_ylim(bottom=0)

            # -- Average household age over time --
            ax = axes[1, 0]
            ax.plot(timevec, self.results['mean_hh_age'], label='Mean')
            ax.plot(timevec, self.results['median_hh_age'], label='Median', linestyle='--')
            ax.set_ylabel('Mean age (years)')
            ax.set_xlabel('Year')
            ax.set_title('Average household age over time')
            ax.legend()
            ax.set_ylim(bottom=0)

            # -- Initial vs final household size distribution --
            ax = axes[1, 1]
            if len(self.hh_size_hists) >= 2:
                initial_sizes = self.hh_size_hists[0]
                final_sizes = self.hh_size_hists[-1]
                max_size = int(max(np.max(initial_sizes), np.max(final_sizes)))
                size_range = np.arange(1, max_size + 1)
                initial_counts = np.array([np.sum(initial_sizes == s) for s in size_range]) / len(initial_sizes)
                final_counts = np.array([np.sum(final_sizes == s) for s in size_range]) / len(final_sizes)
                width = 0.35
                ax.bar(size_range - width/2, initial_counts, width, edgecolor='black', label=f'Initial (t={float(timevec[0]):.0f})')
                ax.bar(size_range + width/2, final_counts, width, edgecolor='black', label=f'Final (t={float(timevec[-1]):.0f})')
                ax.set_xlabel('Household size')
                ax.set_ylabel('Proportion of households')
                ax.set_title('Household size distribution')
                ax.set_xticks(size_range)
                ax.legend()

            plt.tight_layout()

        return ss.return_fig(fig, **kw.return_fig)

    def plot_normalized_matrix(self, **kwargs):
        """Four-panel figure: contacts-per-person and proportionate-mixing ratio at start and end.

        Top row: contacts per person in the x-axis age group (C_ij / n_j).
        Bottom row: departure from proportionate mixing (C_ij / E_ij), where
        E_ij = C_total * p_i * p_j and p_k = n_k / N.
        Left column: simulation start; right column: simulation end.
        """
        kw = ss.plot_args(kwargs)
        timevec = self.sim.t.timevec
        bin_labels = [f'{lo}-{hi}' for lo, hi in zip(self.age_bins[:-1], self.age_bins[1:])]

        def _contacts_per_person(mix, counts):
            """C_ij / n_j — average contacts with age-i individuals per person of age j."""
            n_j = counts.copy()
            n_j[n_j == 0] = np.nan
            return mix / n_j[np.newaxis, :]

        def _prop_mixing_ratio(mix, counts):
            """C_ij / E_ij where E_ij = C_total * p_i * p_j."""
            C_total = mix.sum()
            if C_total == 0:
                return np.full_like(mix, np.nan)
            N = counts.sum()
            if N == 0:
                return np.full_like(mix, np.nan)
            p = counts / N
            expected = C_total * np.outer(p, p)
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = np.where(expected > 0, mix / expected, np.nan)
            return ratio

        panels = [
            (self.age_mixing_initial, self.age_counts_initial,
             f't={float(timevec[0]):.0f}'),
            (self.age_mixing_final, self.age_counts_final,
             f't={float(timevec[-1]):.0f}'),
        ]

        with ss.style(**kw.style):
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            for col, (mix, counts, t_label) in enumerate(panels):
                if mix is None or counts is None:
                    continue

                # -- Contacts per person --
                ax = axes[0, col]
                cpp = _contacts_per_person(mix, counts)
                sns.heatmap(
                    cpp, ax=ax,
                    xticklabels=bin_labels, yticklabels=bin_labels,
                    cmap='YlOrRd', annot=True, fmt='.2f',
                    cbar_kws={'label': 'Contacts per person'},
                )
                ax.set_title(f'Contacts per person ({t_label})')
                ax.set_xlabel('Age group (contact)')
                ax.set_ylabel('Age group')
                ax.invert_yaxis()

                # -- Proportionate mixing ratio --
                ax = axes[1, col]
                ratio = _prop_mixing_ratio(mix, counts)
                # Centre the diverging colormap on 1 (proportionate mixing)
                vmax = np.nanmax(np.abs(ratio - 1)) + 1 if not np.all(np.isnan(ratio)) else 2
                sns.heatmap(
                    ratio, ax=ax,
                    xticklabels=bin_labels, yticklabels=bin_labels,
                    cmap='RdBu_r', center=1, vmin=max(0, 2 - vmax), vmax=vmax,
                    annot=True, fmt='.2f',
                    cbar_kws={'label': 'Observed / expected contacts'},
                )
                ax.set_title(f'Proportionate mixing ratio ({t_label})')
                ax.set_xlabel('Age group (contact)')
                ax.set_ylabel('Age group')
                ax.invert_yaxis()

            plt.tight_layout()

        return ss.return_fig(fig, **kw.return_fig)

    def plot_matrix(self, **kwargs):
        """Three-panel figure: age distribution over time, initial and final age-mixing matrices."""
        kw = ss.plot_args(kwargs)
        timevec = self.sim.t.timevec
        bin_labels = [f'{lo}-{hi}' for lo, hi in zip(self.age_bins[:-1], self.age_bins[1:])]

        with ss.style(**kw.style):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # -- Age distribution over time (stacked area) --
            ax = axes[0]
            bin_data = [np.array(self.results[f'age_{lo}_{hi}']) for lo, hi in zip(self.age_bins[:-1], self.age_bins[1:])]
            ax.stackplot(timevec, *bin_data, labels=bin_labels, alpha=0.8)
            ax.set_ylabel('Number of agents')
            ax.set_xlabel('Year')
            ax.set_title('Age distribution over time')
            ax.legend(loc='upper left', fontsize=8)

            # -- Age-mixing matrix: initial --
            ax = axes[1]
            if self.age_mixing_initial is not None:
                sns.heatmap(
                    self.age_mixing_initial, ax=ax,
                    xticklabels=bin_labels, yticklabels=bin_labels,
                    cmap='YlOrRd', annot=True, fmt='.0f',
                    cbar_kws={'label': 'Contact pairs'},
                )
                ax.set_title(f'Age-mixing matrix (t={float(timevec[0]):.0f})')
                ax.set_xlabel('Age group')
                ax.set_ylabel('Age group')
                ax.invert_yaxis()

            # -- Age-mixing matrix: final --
            ax = axes[2]
            if self.age_mixing_final is not None:
                sns.heatmap(
                    self.age_mixing_final, ax=ax,
                    xticklabels=bin_labels, yticklabels=bin_labels,
                    cmap='YlOrRd', annot=True, fmt='.0f',
                    cbar_kws={'label': 'Contact pairs'},
                )
                ax.set_title(f'Age-mixing matrix (t={float(timevec[-1]):.0f})')
                ax.set_xlabel('Age group')
                ax.set_ylabel('Age group')
                ax.invert_yaxis()

            plt.tight_layout()

        return ss.return_fig(fig, **kw.return_fig)
