import starsim as ss
import pandas as pd
import numpy as np
import os
import tbsim.config as cfg
import datetime as ddtt
import tbsim.plotdwelltimes as plotter
import tbsim as mtb
from scipy import stats
from pandas.plotting import parallel_coordinates
from enum import IntEnum

class DwtAnalyzer(ss.Analyzer, plotter.DwtPlotter):
    def __init__(self, adjust_to_days=False, unit=None, states_ennumerator=None):
        """
        Initializes the analyzer with optional adjustments to days and unit specification.

        Args:
            adjust_to_days (bool): If True, adjusts the dwell times to days by multiplying the recorded dwell_time by the sim.pars.dt.
            Default is True.
            
            unit (str): The unit of time for the analysis. Default is 'days'. TODO: Implement its use.
            states_ennumerator (IntEnum): An IntEnum class that enumerates the states in the simulation. Default is None which will result in the use of the mtb.TBS class.

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
        if self.eSTATES is None:
            self.eSTATES = mtb.TBS
        self.adjust_to_days = adjust_to_days
        self.unit = unit
        self.file_path = None
        self.data = pd.DataFrame(columns=['agent_id', 'state', 'entry_time', 'exit_time', 'dwell_time', 'state_name', 'going_to_state_id','going_to_state'])
        self._latest_sts_df = pd.DataFrame(columns=['agent_id', 'last_state', 'last_state_time'])      
        plotter.DwtPlotter.__init__(self, data=self.data)
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
        self._update_data(ti)
        return
    
    def _update_data(self, ti):
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

    def finalize(self):
        super().finalize()
        self.data['state_name'] = self.data['state'].apply(lambda x: self.eSTATES(x).name.replace('_', ' ').title())
        self.data['going_to_state'] = self.data['going_to_state_id'].apply(lambda x: self.eSTATES(x).name.replace('_', ' ').title())
        self.data['compartment'] = 'tbd'
        if self.adjust_to_days:
            self.data['dwell_time_recorded'] = self.data['dwell_time']
            self.data['dwell_time'] = self.data['dwell_time'] * self.sim.pars.dt
        
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
        fn = os.path.join(resdir, f'dwell_time_logger_{t.strftime("%Y%m%d%H%M%S")}.csv')
        self.data.to_csv(fn, index=False)
        fn_meta = os.path.join(resdir, f'dwell_time_logger_{t.strftime("%Y%m%d%H%M%S")}.json')
        with open(fn_meta, 'w') as f:
            f.write('{"sim_units": "%s", "specified_units": "%s"}' % (self.sim.pars.unit, self.unit))

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
