"""
Illustrate malnutrition, and TB-malnutrition with connector.
"""

import starsim as ss
import tbsim
import matplotlib.pyplot as plt


def make_malnutrition():
    """Standalone malnutrition simulation."""
    nut = tbsim.Malnutrition()
    sim = tbsim.Sim(
        n_agents=200,
        start='1990',
        stop='2020',
        diseases=[nut],
        demographics=[ss.Births(pars=dict(birth_rate=5)), ss.Deaths(pars=dict(death_rate=5))],
    )
    return sim


def make_tb_nut():
    """TB + malnutrition simulation with connector."""
    nut = tbsim.Malnutrition()
    connector = tbsim.TB_Nutrition_Connector()

    sim = tbsim.Sim(
        n_agents=1000,
        start=1980,
        stop=1995,
        dt=float(ss.days(7))/365,
        beta=0.01,
        init_prev=0.25,
        diseases=[nut],
        demographics=[ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))],
        connectors=connector,
    )
    sim.pars.verbose = float(sim.pars.dt) / 5
    return sim


if __name__ == '__main__':
    # Make Malnutrition simulation
    sim_n = make_malnutrition()
    sim_n.run()
    results = {'malnutrition': sim_n.results.flatten()}
    tbsim.plot(results, n_cols=3)

    # Make TB-malnutrition simulation
    sim_tbn = make_tb_nut()
    sim_tbn.run()
    tbsim.get_tb(sim_tbn).plot()
    plt.show()
