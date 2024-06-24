import unittest
import tbsim as mtb
import starsim as ss
import numpy as np

class TestProduct(unittest.TestCase):
    def setUp(self):
        self.product = mtb.Product(name="TestVaccine", efficacy=0.95, doses=2)

    def test_product_initialization(self):
        self.assertEqual(self.product.name, "TestVaccine")  # Test the name attribute
        self.assertEqual(self.product.efficacy, 0.95)     # Test the efficacy attribute
        self.assertEqual(self.product.doses, 2)          # Test the doses attribute

    def test_product_repr(self):
        self.assertEqual(repr(self.product), "Product(name=TestVaccine, efficacy=0.95, doses=2)") # Cannonical representation

class TestTBVaccinationCampaign(unittest.TestCase):
    def setUp(self):
        self.product = mtb.Product(name="TestVaccine", efficacy=0.95, doses=2)
        self.campaign = mtb.TBVaccinationCampaign(year=2020, product=self.product, rate=0.015, target_gender='All', target_age=10, target_state='susceptible')

    def test_campaign_initialization(self):
        self.assertEqual(self.campaign.year, np.array([2020]))  # Test the year attribute
        self.assertEqual(self.campaign.rate, np.array([0.015])) # Test the rate attribute
        self.assertEqual(self.campaign.target_gender, 'All')    # Test the target_gender attribute
        self.assertEqual(self.campaign.target_age, 10)        # Test the target_age attribute
        self.assertEqual(self.campaign.target_state, 'susceptible') # Test the target_state attribute
        self.assertEqual(self.campaign.product, self.product)   # Test the product attribute

            
    # Assuming a Simulation class and necessary setup exists in starsim for a complete test
    def test_campaign_apply(self):
        sim = ss.Sim(people=ss.People(n_agents=500), networks=ss.RandomNet(), diseases=mtb.TB(), pars=dict(start = 1900, end = 2000))
        self.campaign.apply(sim)
        # Further assertions depending on the simulation outcome

if __name__ == '__main__':
    sim = ss.Sim(people=ss.People(n_agents=500), networks=ss.RandomNet(), diseases=mtb.TB(), pars=dict(start = 1900, end = 2000))

    
    
    unittest.main()