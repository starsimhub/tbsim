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

class DTAn(ss.Module):
    def __init__( self):
        super().__init__()
        return

    def init_results(self):
        super().init_results()
        self.latest_sts_df = pd.DataFrame(columns=['agent_id', 'last_state', 'last_state_time'])
        self.dwell_time_logger = pd.DataFrame(columns=['agent_id', 'state', 'entry_time', 'exit_time', 'dwell_time'])
        
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
            states=tb.state[uids],
            entry_times=self.latest_sts_df[self.latest_sts_df['agent_id'].isin(uids)]['last_state_time'].values,
            exit_times=np.full(len(uids), ti)
        )

        # Update the latest state dataframe
        self.latest_sts_df.loc[self.latest_sts_df['agent_id'].isin(uids), 'last_state'] = tb.state[uids]
        self.latest_sts_df.loc[self.latest_sts_df['agent_id'].isin(uids), 'last_state_time'] = ti


    def finalize(self):
        super().finalize()
        self.dwell_time_logger['state_name'] = self.dwell_time_logger['state'].apply(lambda x: mtb.TBS(x).name.replace('_', ' ').title())
        self.file_name = self.save_to_file()


    def finalize_results(self):
        super().finalize_results()
        print(self.ti)
        print(self.latest_sts_df)
        print(self.dwell_time_logger)
        return

    def log_dwell_time(self, agent_ids, states, entry_times, exit_times):
        entry_times = np.nan_to_num(entry_times, nan=0)
        dwell_times = exit_times - entry_times

        new_logs = pd.DataFrame({
            'agent_id': agent_ids,
            'state': states,
            'entry_time': entry_times,
            'exit_time': exit_times,
            'dwell_time': dwell_times,
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