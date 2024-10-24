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

def test_ACF():
    import networkx as nx
    ppl = ss.People(1000)
    acf = mtb.ActiveCaseFinding()
    graph = nx.fast_gnp_random_graph(n=ppl.n_uids, p=1, seed=None, directed=False)
    net = ss.StaticNet(graph=graph, seed=True)
    tb = mtb.TB(beta=ss.beta(0.2))
    sim = ss.Sim(unit='day', dt=7, start=sc.date('2013-01-01'), stop=sc.date('2016-12-31'), people=ppl, diseases=tb, interventions=acf, networks=net)
    sim.run()
    return

@pytest.mark.skip(reason="not working")
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
    sim = ss.Sim(unit='day', dt=7, start=sc.date('2013-01-01'), stop=sc.date('2016-12-31'), n_agents=1000, diseases=tb, interventions=campaign)
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

if __name__ == '__main__':
    sim = ss.Sim(people=ss.People(n_agents=500), networks=ss.RandomNet(), diseases=mtb.TB(), pars=dict(start=1990, stop = 2021, dt=0.5))
    pytest.main()
