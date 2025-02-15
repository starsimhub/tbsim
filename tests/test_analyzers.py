import pytest
import pandas as pd
from tbsim.analyzers import DwtPlotter
import os
import tb_acf
import starsim as ss

def init():
    plotter = DwtPlotter()


    pop = ss.People(1000)
    tb_pars = dict(
        beta = ss.rate_prob(0.5, unit='month'),
        acuinf = ss.days(ss.lognorm_ex(mean=60, std=34.2)),
        alpha = 1.0,
    )
    tb = tb_acf.TB_LSHTM_Acute(tb_pars, name='tb')

    nets = ss.MixingPool(
        diseases = 'tb',
        src      = ss.AgeGroup(0, None),
        dst      = ss.AgeGroup(0, None),
        beta     = tb.pars.beta,
        contacts = ss.poisson(3), # ss.constant(1),
    )

    # Simulation parameters
    sim_pars = dict(
        unit = 'day',
        dt = 30,
        start = '1950-01-01',
        stop = '2019-12-31',
    )

    sim = ss.Sim(pars=sim_pars, people=pop, diseases=tb, networks=nets)
    sim.run()

    return sim

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
def test_sankey_agents_even_age_ranges(sample_data, model_type, mocker):
    mocker.patch('plotly.graph_objects.Figure.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.sankey_agents_even_age_ranges(number_of_plots=3)
        assert plotter.data is not None
        assert 'age' in plotter.data.columns
        assert plotter.data['age'].dtype in [int, float]

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])   # verify
def test_sankey_agents(sample_data, model_type, mocker):
    mocker.patch('plotly.graph_objects.Figure.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.sankey_agents(subtitle="Test Sankey")
        assert plotter.data is not None
        assert 'state_name' in plotter.data.columns
        assert 'going_to_state' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['TBsim']) # ['LSHTM', 'HTMAcute', 'TBsim'])
def test_sankey_dwelltimes(sample_data, model_type, mocker):
    mocker.patch('plotly.graph_objects.Figure.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.sankey_dwelltimes(subtitle="Test Dwell Times")
        assert plotter.data is not None
        assert 'dwell_time' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_barchar_all_state_transitions_interactive(sample_data, model_type, mocker):
    mocker.patch('plotly.graph_objects.Figure.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.barchar_all_state_transitions_interactive(dwell_time_bins=[0, 2, 4, 6])
        assert plotter.data is not None
        assert 'dwell_time' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['TBsim'])  # it takes too long to run
def test_stacked_bars_states_per_agent_static(sample_data, model_type, mocker):
    mocker.patch('matplotlib.pyplot.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.stacked_bars_states_per_agent_static()
        assert plotter.data is not None
        assert 'agent_id' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_reinfections_bystates_bars_interactive(sample_data, model_type, mocker):
    mocker.patch('plotly.graph_objects.Figure.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.reinfections_bystates_bars_interactive(target_states=[-1.0, 0.0, 1.0])
        assert plotter.data is not None
        assert 'state_name' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_stackedbars_dwelltime_state_interactive(sample_data, model_type, mocker):
    mocker.patch('plotly.graph_objects.Figure.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.stackedbars_dwelltime_state_interactive(bin_size=1, num_bins=6)
        assert plotter.data is not None
        assert 'dwell_time' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_subplot_custom_transitions(sample_data, model_type, mocker):
    mocker.patch('matplotlib.pyplot.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        transitions_dict = {'A': ['B', 'C'], 'B': ['A', 'C']}
        plotter.subplot_custom_transitions(transitions_dict=transitions_dict)
        assert plotter.data is not None
        assert 'state_name' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_stackedbars_subplots_state_transitions(sample_data, model_type, mocker):
    mocker.patch('matplotlib.pyplot.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.stackedbars_subplots_state_transitions(bin_size=1, num_bins=6)
        assert plotter.data is not None
        assert 'state_name' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_histogram_with_kde(sample_data, model_type, mocker):
    mocker.patch('matplotlib.pyplot.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.histogram_with_kde(subtitle="Test Histogram")
        assert plotter.data is not None
        assert 'dwell_time' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_graph_state_transitions(sample_data, model_type, mocker):
    mocker.patch('matplotlib.pyplot.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.graph_state_transitions(states=['A', 'B'], subtitle="Test Graph")
        assert plotter.data is not None
        assert 'state_name' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_graph_state_transitions_curved(sample_data, model_type, mocker):
    mocker.patch('matplotlib.pyplot.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.graph_state_transitions_curved(states=['A', 'B'], subtitle="Test Curved Graph")
        assert plotter.data is not None
        assert 'state_name' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_plot_dwell_time_validation(sample_data, model_type, mocker):
    mocker.patch('matplotlib.pyplot.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.plot_dwell_time_validation()
        assert plotter.data is not None
        assert 'dwell_time' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_plot_dwell_time_validation_interactive(sample_data, model_type, mocker):
    mocker.patch('plotly.graph_objects.Figure.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.plot_dwell_time_validation_interactive()
        assert plotter.data is not None
        assert 'dwell_time' in plotter.data.columns

@pytest.mark.parametrize("model_type", ['LSHTM', 'HTMAcute', 'TBsim'])
def test_plot_kaplan_meier(sample_data, model_type, mocker):
    mocker.patch('matplotlib.pyplot.show')
    for df in sample_data[model_type]:
        plotter = DwtPlotter(data=df)
        plotter.plot_kaplan_meier(dwell_time_col='dwell_time')
        assert plotter.data is not None
        assert 'dwell_time' in plotter.data.columns

if __name__ == '__main__':
    pytest.main()