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

class DwtAnalyzer(ss.Analyzer):
    def __init__(self, adjust_to_days=True, unit='days'):
        """
        Initializes the analyzer with optional adjustments to days and unit specification.

        Args:
            adjust_to_days (bool): If True, adjusts the dwell times to days by multiplying the recorded dwell_time by the sim.pars.dt.
            Default is True.
            
            unit (str): The unit of time for the analysis. Default is 'days'. TODO: Implement its use.

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
        super().__init__()
        self.adjust_to_days = adjust_to_days
        self.unit = unit
        self.file_name = None
        self.data = pd.DataFrame(columns=['agent_id', 'state', 'entry_time', 'exit_time', 'dwell_time', 'state_name', 'going_to_state_id','going_to_state'])
        self._latest_sts_df = pd.DataFrame(columns=['agent_id', 'last_state', 'last_state_time'])      
        # self.initialize_dfs() <-- This is called in the step method to ensure that 'people object' is available
        return
    
    def initialize_dfs(self):

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
        if not self.sim.ti: self.initialize_dfs()
        sim = self.sim
        ti = sim.ti
        
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
        return
    
    def update_results(self):
        super().update_results()

        tb = self.sim.diseases.tb
        ti = self.ti
        uids = self.sim.people.auids

        # Filter rows in _latest_sts_df for the relevant agents
        relevant_rows = self._latest_sts_df[self._latest_sts_df['agent_id'].isin(uids)]

        # Identify agents whose last recorded state is different from the current state
        different_state_mask = relevant_rows['last_state'].values != tb.state[ss.uids(relevant_rows['agent_id'].values)]
        uids = ss.uids(relevant_rows['agent_id'].values[different_state_mask])

        # Log dwell times
        self.log_dwell_time(
            agent_ids=uids,
            states=relevant_rows['last_state'].values[different_state_mask],
            entry_times=relevant_rows['last_state_time'].values[different_state_mask],
            exit_times=np.full(len(uids), ti),
            going_to_state_ids=tb.state[uids]
        )

        # Update the latest state dataframe
        self._latest_sts_df.loc[self._latest_sts_df['agent_id'].isin(uids), 'last_state'] = tb.state[uids]
        self._latest_sts_df.loc[self._latest_sts_df['agent_id'].isin(uids), 'last_state_time'] = ti


    def finalize(self):
        super().finalize()
        self.data['state_name'] = self.data['state'].apply(lambda x: mtb.TBS(x).name.replace('_', ' ').title())
        self.data['going_to_state'] = self.data['going_to_state_id'].apply(lambda x: mtb.TBS(x).name.replace('_', ' ').title())
        self.data['compartment'] = self.data['going_to_state_id'].apply(self.resolve_compartment)
        if self.adjust_to_days:
            self.data['dwell_time'] = self.data['dwell_time'] * self.sim.pars.dt

        self.file_name = self.save_to_file()

    def finalize_results(self):
        super().finalize_results()
        print(self.ti)
        print(self._latest_sts_df)
        print(self.data)
        return
    

    
    def log_dwell_time(self, agent_ids, states, entry_times, exit_times, going_to_state_ids):
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
    
    def resolve_compartment(self, going_to_state_id):
        if going_to_state_id in [0, 1]:      return 'LATENT'
        elif going_to_state_id in [3, 4, 5]: return 'ACTIVE'
        elif going_to_state_id == 2:         return 'PRESYMPTOMATIC'
        elif going_to_state_id == -1:        return 'SUSCEPTIBLE'
        elif going_to_state_id == 8:         return 'REMOVED'
        else:                                return 'OTHER'

    def save_to_file(self):
        resdir = os.path.dirname(cfg.create_res_dir())
        t = ddtt.datetime.now()
        fn = os.path.join(resdir, f'dwell_time_logger_{t.strftime("%Y%m%d%H%M%S")}.csv')
        self.data.to_csv(fn, index=False)
        print(f"Dwell time logs saved to:\n {fn}\n")
        return fn

    def validate_dwell_time_distributions(self, expected_distributions=None):
        """
        Validate dwell times against expected distributions using statistical tests.
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

    def plot_dwell_time_validation(self):
        """
        Plot the results of the dwell time validation.
        """

        fig, ax = plt.subplots()
        for state in self.data['state'].unique():
            dwell_times = self.data[self.data['state'] == state]['dwell_time']
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
        fig = px.histogram(self.data, x='dwell_time', color='state_name', 
                            nbins=50, barmode='overlay', 
                            labels={'dwell_time': 'Dwell Time', 'state_name': 'State'},
                            title='Dwell Time Validation')
        fig.update_layout(bargap=0.1)
        fig.show()
        return

    def graph_state_transitions(self, states=None, pos=None):
        pdt.graph_state_transitions(dwell_time_logger=self.data, states=states, pos=pos)
        return

    def graph_compartments_transitions(self, states=None, pos=None):
        pdt.graph_compartments_transitions(dwell_time_logger=self.data, states=states, pos=pos)
        return

    def plot_binned_stacked_bars_state_transitions(self, bin_size=50, num_bins=8):
        pdt.plot_binned_stacked_bars_state_transitions(dwell_time_logger=self.data, bin_size=bin_size, num_bins=num_bins)
        return

    def plot_binned_by_compartment(self, num_bins=50):
        pdt.plot_binned_by_compartment(dwell_time_logger=self.data, num_bins=num_bins)
        return
    
    def interactive_all_state_transitions(self, dwell_time_bins=None, filter_states=None):
        pdt.interactive_all_state_transitions(dwell_time_logger=self.data, dwell_time_bins=dwell_time_bins, filter_states=filter_states)
        return
    
    def stacked_bars_states_per_agent_static(self):
        pdt.stacked_bars_states_per_agent_static(dwell_time_logger=self.data)
        return
        
    def interactive_stacked_bar_charts_dt_by_state(self):
        pdt.interactive_stacked_bar_charts_dt_by_state(dwell_time_logger=self.data)
        return
    
    def sankey(self):
        pdt.sankey(dwell_time_logger=self.data)
        return
    
    def plot_state_transition_lengths_custom(self, transitions_dict):
        pdt.plot_state_transition_lengths_custom(dwell_time_logger=self.data, transitions_dict=transitions_dict)
        return

    def plot_binned_by_compartment(self, bin_size=50, num_bins=50):
        pdt.plot_binned_by_compartment(dwell_time_logger=self.data, bin_size=bin_size, num_bins=num_bins)
        return
    
    def plot_binned_stacked_bars_state_transitions(self, bin_size=50, num_bins=20):
        pdt.plot_binned_stacked_bars_state_transitions(dwell_time_logger=self.data, bin_size=bin_size, num_bins=num_bins)
        return