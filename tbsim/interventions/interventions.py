import starsim as ss
import sciris as sc
import numpy as np
from tbsim import TBS
import datetime as dt
import tbsim as mtb
import pandas as pd

__all__ = ['Product', 'TBVaccinationCampaign', 'get_extrastates']


def get_extrastates():
    exs = [ss.State('sought_care', default=False),
        ss.FloatArr('care_seeking_multiplier', default=1.0),
        ss.State('multiplier_applied', default=False),
        ss.FloatArr('n_times_tested', default=0.0),
        ss.FloatArr('n_times_treated', default=0.0),
        ss.State('returned_to_community', default=False),
        ss.State('received_tpt', default=False),
        ss.State('tb_treatment_success', default=False),
        ss.State('tested', default=False),
        ss.State('test_result', default=np.nan),
        ss.State('diagnosed', default=False),
        ss.State('on_tpt', default=True),
        ss.State('tb_smear', default=False),
        ss.State('hiv_positive', default=False),
        ss.State('eptb', default=False),
        ss.State('symptomatic', default=False),
        ss.State('presymptomatic', default=False),
        ss.State('non_symptomatic', default=True),
        ss.State('screen_negative', default=True),
        ss.State('household_contact', default=False),
        ss.State('treatment_success', default=False),
        ss.State('treatment_failure', default=False),
        ss.State('hhid', default=False),
        ss.FloatArr('vaccination_year', default=np.nan),]
    return exs
   

class Product(ss.Module):
    """
    Class to define a vaccine product with specific attributes.
    """
    def __init__(self, name, efficacy, doses):
        self.name = name
        self.efficacy = efficacy
        self.doses = doses
        
    def init_pre(self, sim):
        if not self.initialized:
            super().init_pre(sim)
        else:
            return
    def __repr__(self):
        return f"Product(name={self.name}, efficacy={self.efficacy}, doses={self.doses})"
    
    def administer(self, people, inds):
        """ Adminster a Product - implemented by derived classes """
        print("vaccine administered to people")
        
        return

class TBVaccinationCampaign(ss.Intervention):
    """
    Base class for any intervention that uses campaign delivery; handles interpolation of input years.
    """

    def __init__(self, year=1900, product=None, rate =.015, target_gender='All', target_age=10, target_state=None, new_value_fraction=1, prob=None, *args, **kwargs):
        if product is None:
            raise NotImplementedError('No product specified')
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        self.target_gender = target_gender
        self.target_age = target_age
        self.prob = sc.promotetoarray(prob)
        self.product = product
        self.target_state = target_state
        self.new_value_fraction = new_value_fraction
        super().__init__(*args, **kwargs)
        self.p = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.rate*sim.dt))
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        return
    
    def update(self, sim):
        if sim.year < self.year[0]:
            return
        if self.product is None:
            raise NotImplementedError('No product specified')

        tb = sim.diseases['tb']
        ppl = sim.people
        
        # eligible = (tb.state == self.target_state) & ppl.alive & (ppl.age >= self.target_age) & (ppl.gender == self.target_gender)
        eligible = (tb.state == self.target_state) & ppl.alive
        
        eligible_uids = eligible.uids
        
        change_uids = self.p.filter(eligible_uids)
        
        # TODO: Add the actual value of the product's effectiveness here...
        tb.rel_LS_prog[change_uids] = tb.rel_LS_prog[change_uids]*0.9 # *self.product.efficacy   
        tb.rel_LF_prog[change_uids] = tb.rel_LF_prog[change_uids]*0.9  # *self.product.efficacy   
        
        return len(change_uids)

# Sample calling function below
if __name__ == '__main__':

    print('care_seeking_multiplier' in [s.name for s in mtb.get_extrastates()])
    print('n_times_tested' in [s.name for s in mtb.get_extrastates()])