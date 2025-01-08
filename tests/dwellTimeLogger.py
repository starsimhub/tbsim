import pandas as pd
import time

class DwellTimeLogger:
    def __init__(self):
        self.logs = []  # It will store dwell time logs

    def log_state_transition(self, agent_id, state, entry_time, exit_time):
        dwell_time = exit_time - entry_time
        self.logs.append({
            'agent_id': agent_id,
            'state': state,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'dwell_time': dwell_time
        })

    def save_logs(self, filename="dwell_times.csv"):
        # Save logs as a CSV file
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"Logs saved to {filename}")
