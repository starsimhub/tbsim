import pytest
import os
import pandas as pd
from dwellTimeLogger import DwellTimeLogger

def test_log_state_transition():
    logger = DwellTimeLogger()
    logger.log_state_transition(agent_id=1, state="active", entry_time=10, exit_time=20)
    
    assert len(logger.logs) == 1
    log = logger.logs[0]
    assert log['agent_id'] == 1
    assert log['state'] == "active"
    assert log['entry_time'] == 10
    assert log['exit_time'] == 20
    assert log['dwell_time'] == 10

def test_save_logs(tmp_path):
    logger = DwellTimeLogger()
    logger.log_state_transition(agent_id=1, state="active", entry_time=10, exit_time=20)
    logger.log_state_transition(agent_id=2, state="inactive", entry_time=15, exit_time=25)
    
    file_path = tmp_path / "dwell_times.csv"
    logger.save_logs(file_path)
    
    assert os.path.exists(file_path)
    
    df = pd.read_csv(file_path)
    assert len(df) == 2
    assert df.iloc[0]['agent_id'] == 1
    assert df.iloc[0]['state'] == "active"
    assert df.iloc[0]['entry_time'] == 10
    assert df.iloc[0]['exit_time'] == 20
    assert df.iloc[0]['dwell_time'] == 10
    assert df.iloc[1]['agent_id'] == 2
    assert df.iloc[1]['state'] == "inactive"
    assert df.iloc[1]['entry_time'] == 15
    assert df.iloc[1]['exit_time'] == 25
    assert df.iloc[1]['dwell_time'] == 10