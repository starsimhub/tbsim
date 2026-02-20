import pytest
import pandas as pd
from tbsim.analyzers import DwellTime
import os
from unittest.mock import patch

@pytest.fixture
def sample_data():
    base_dir = os.path.dirname(__file__)
    data_files = [
        'data/HighDecliningLSHTM-01.csv',
        # 'data/HighDecliningLSHTM-02.csv',
        # 'data/LowFlatLSHTMAcute-01.csv',
        'data/LowFlatLSHTMAcute-02.csv',
        'data/LowFlatTBsim-01.csv',
        # 'data/LowFlatTBsim-02.csv'
    ]
    dataframes = {
        'LSHTM': [pd.read_csv(os.path.join(base_dir, file)) for file in data_files if 'HighDecliningLSHTM' in file],
        'HTMAcute': [pd.read_csv(os.path.join(base_dir, file)) for file in data_files if 'LowFlatLSHTMAcute' in file],
        'TBsim': [pd.read_csv(os.path.join(base_dir, file)) for file in data_files if 'LowFlatTBsim' in file]
    }
    return dataframes

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
@patch('plotly.graph_objects.Figure.show')
def test_sankey_agents(mock_show, sample_data, model_type):
    for df in sample_data[model_type]:
        dt = DwellTime(data=df)
        dt.plot('sankey', subtitle="Test Sankey")
        assert dt.data is not None
        assert 'state_name' in dt.data.columns
        assert 'going_to_state' in dt.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
@patch('matplotlib.pyplot.show')
def test_histogram_with_kde(mock_show, sample_data, model_type):
    for df in sample_data[model_type]:
        dt = DwellTime(data=df)
        dt.plot('histogram', subtitle="Test Histogram")
        assert dt.data is not None
        assert 'dwell_time' in dt.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
@patch('matplotlib.pyplot.show')
def test_graph_state_transitions_curved(mock_show, sample_data, model_type):
    for df in sample_data[model_type]:
        dt = DwellTime(data=df)
        dt.plot('network', states=['A', 'B'], subtitle="Test Curved Graph")
        assert dt.data is not None
        assert 'state_name' in dt.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
@patch('matplotlib.pyplot.show')
def test_plot_dwell_time_validation(mock_show, sample_data, model_type):
    for df in sample_data[model_type]:
        dt = DwellTime(data=df)
        dt.plot('validation')
        assert dt.data is not None
        assert 'dwell_time' in dt.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
@patch('matplotlib.pyplot.show')
def test_plot_kaplan_meier(mock_show, sample_data, model_type):
    for df in sample_data[model_type]:
        dt = DwellTime(data=df)
        dt.plot('kaplan_meier', dwell_time_col='dwell_time')
        assert dt.data is not None
        assert 'dwell_time' in dt.data.columns

def test_plot_invalid_kind(sample_data):
    df = sample_data['TBsim'][0]
    dt = DwellTime(data=df)
    with pytest.raises(ValueError, match="Unknown plot kind"):
        dt.plot('nonexistent_kind')

if __name__ == '__main__':
    pytest.main()
