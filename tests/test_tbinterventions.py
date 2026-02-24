import pytest
import tbsim
import starsim as ss
import numpy as np

@pytest.mark.skip(reason='ActiveCaseFinding has moved out of this repository')
def test_ACF():
    import networkx as nx
    ppl = ss.People(1000)
    acf = tbsim.ActiveCaseFinding()
    graph = nx.fast_gnp_random_graph(n=ppl.n_uids, p=1, seed=None, directed=False)
    net = ss.StaticNet(graph=graph, seed=True)
    tb = tbsim.TB_EMOD(beta=ss.peryear(0.2))
    sim = ss.Sim(dt=ss.days(7), start=ss.date('2013-01-01'), stop=ss.date('2016-12-31'), people=ppl, diseases=tb, interventions=acf, networks=net)
    sim.run()
    return

@pytest.mark.skip(reason='not fully implemented yet')
def test_campaign():
    class TB_Treatment(ss.Product):
        def administer(self, people, uids):
            self.sim.tb.on_treatment[uids] = True
            return

    campaign = ss.campaign_triage(
        product = TB_Treatment(),
        eligibility = lambda sim: sim.interventions.screening.outcomes['positive'],
        prob = [0.85, 0.85, 0.85],
        years = [2014, 2015, 2016],
    )

    tb = tbsim.TB_EMOD()
    sim = ss.Sim(dt=ss.days(7), start=ss.date('2013-01-01'), stop=ss.date('2016-12-31'), n_agents=1000, diseases=tb, interventions=campaign)
    sim.run()
    return


# Test for BetaByYear

def test_beta_intervention_changes_beta():
    import sciris as sc
    # Set up a minimal simulation with a known beta
    initial_beta = 0.01
    x_beta = 0.5
    intervention_year = 2005
    stop_year = 2010
    
    tb_pars = dict(beta=initial_beta, init_prev=0.25)
    sim_pars = dict(start=f'{intervention_year-1}-01-01', stop=f'{stop_year}-01-01', dt=ss.days(1), rand_seed=42)
    
    pop = ss.People(n_agents=100)
    tb = tbsim.TB_EMOD(pars=tb_pars)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    beta_intv = tbsim.BetaByYear(pars={'years': [intervention_year], 'x_beta': x_beta})
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        interventions=[beta_intv],
        pars=sim_pars,
    )
    
    sim.init()
    beta = tbsim.get_tb(sim).pars['beta']
    # Check beta before intervention
    assert np.isclose(beta.value, initial_beta)
    # Run up to just before the intervention year
    while sim.t.now('year') < intervention_year:
        sim.run_one_step()
    # Should still be unchanged
    assert np.isclose(beta.value, initial_beta)
    # Step into the intervention year
    sim.run_one_step()
    # Beta should now be changed
    expected_beta = initial_beta * x_beta
    assert np.isclose(beta.value, expected_beta)
    # Step again, beta should not change further
    sim.run_one_step()
    assert np.isclose(beta.value, expected_beta)


if __name__ == '__main__':
    sim = ss.Sim(people=ss.People(n_agents=500), networks=ss.RandomNet(), diseases=tbsim.TB_EMOD(), pars=dict(start=1990, stop=2021, dt=ss.days(0.5)))
    pytest.main()
