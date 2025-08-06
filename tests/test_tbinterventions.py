import pytest
import tbsim as mtb
import starsim as ss
import numpy as np
import sciris as sc

@pytest.fixture
def sample_product():
    return mtb.Product(name="TestVaccine", efficacy=0.95, doses=2)

@pytest.fixture
def sample_campaign(sample_product):
    return mtb.TBVaccinationCampaign(year=2020, product=sample_product, rate=0.015, target_gender='All', target_age=10, target_state='susceptible')

@pytest.mark.skip(reason='ActiveCaseFinding has moved out of this repository')
def test_ACF():
    import networkx as nx
    ppl = ss.People(1000)
    acf = mtb.ActiveCaseFinding()
    graph = nx.fast_gnp_random_graph(n=ppl.n_uids, p=1, seed=None, directed=False)
    net = ss.StaticNet(graph=graph, seed=True)
    tb = mtb.TB(beta=ss.prob(0.2))
    sim = ss.Sim(dt=ss.days(), dt=ss.days(7), start=sc.date('2013-01-01'), stop=sc.date('2016-12-31'), people=ppl, diseases=tb, interventions=acf, networks=net)
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

    tb = mtb.TB()
    sim = ss.Sim(dt=ss.days(), dt=ss.days(7), start=sc.date('2013-01-01'), stop=sc.date('2016-12-31'), n_agents=1000, diseases=tb, interventions=campaign)
    sim.run()
    return

def test_product_initialization(sample_product):
    assert sample_product.name == "TestVaccine"
    assert sample_product.efficacy == 0.95
    assert sample_product.doses == 2

def test_product_repr(sample_product):
    assert repr(sample_product) == "Product(name=TestVaccine, efficacy=0.95, doses=2)"

def test_campaign_initialization(sample_campaign, sample_product):
    assert np.array_equal(sample_campaign.year, np.array([2020]))
    assert np.array_equal(sample_campaign.rate, np.array([0.015]))
    assert sample_campaign.target_gender == 'All'
    assert sample_campaign.target_age == 10
    assert sample_campaign.target_state == 'susceptible'
    assert sample_campaign.product == sample_product

# Assuming a Simulation class and necessary setup exists in starsim for a complete test
@pytest.mark.skip(reason="TODO: Implement the test_campaign_apply test")
def test_campaign_apply():
    pass


# Test for BetaByYear

def test_beta_intervention_changes_beta():
    import sciris as sc
    # Set up a minimal simulation with a known beta
    initial_beta = 0.01
    x_beta = 0.5
    intervention_year = 2005
    stop_year = 2010
    
    tb_pars = dict(beta=initial_beta, init_prev=0.25)
    sim_pars = dict(start=intervention_year-1, stop=stop_year, dt=ss.days(1), rand_seed=42)
    
    pop = ss.People(n_agents=100)
    tb = mtb.TB(pars=tb_pars)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    beta_intv = mtb.BetaByYear(pars={'years': [intervention_year], 'x_beta': x_beta})
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        interventions=[beta_intv],
        pars=sim_pars,
    )
    
    sim.init()
    # Check beta before intervention
    assert np.isclose(sim.diseases.tb.pars['beta'].v, initial_beta)
    # Run up to just before the intervention year
    while sim.t.now('year') < intervention_year:
        sim.run_one_step()
    # Should still be unchanged
    assert np.isclose(sim.diseases.tb.pars['beta'].v, initial_beta)
    # Step into the intervention year
    sim.run_one_step()
    # Beta should now be changed
    expected_beta = initial_beta * x_beta
    assert np.isclose(sim.diseases.tb.pars['beta'].v, expected_beta)
    # Step again, beta should not change further
    sim.run_one_step()
    assert np.isclose(sim.diseases.tb.pars['beta'].v, expected_beta)


if __name__ == '__main__':
    sim = ss.Sim(people=ss.People(n_agents=500), networks=ss.RandomNet(), diseases=mtb.TB(), pars=dict(start=1990, stop = 2021, dt=ss.days(0).5))
    pytest.main()
