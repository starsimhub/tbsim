"""
Start the slowest tests first, each paired with a quick one: pytest-xdist (-n) gives each worker two tests at a time
from the front of the list, so this way they start at once on separate workers, rather than queue behind each other
"""

slowest = ['test_tb_hiv_jointly', 'test_care', 'test_contact_tracing', 'test_two_strain_abm_vs_ode[1.0]', 'test_two_strain_abm_vs_ode[0]', 'test_hiv_speeds_progression', 'test_tpt_infection', 'test_natural_history', 'test_two_drugs', 'test_plot_accepts_all_input_types', 'test_monitoring[abm]', 'test_second_line[abm]', 'test_agents[LSHTMResistant]']


def pytest_collection_modifyitems(items):
    slow = sorted([item for item in items if item.name in slowest], key=lambda item: slowest.index(item.name))
    rest = [item for item in items if item.name not in slowest]
    paired = [item for pair in zip(slow, rest) for item in pair]
    items[:] = paired + slow[len(rest):] + rest[len(slow):]
