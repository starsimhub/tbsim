# tb_intervention.py

import starsim as ss
import sciris as sc
import numpy as np
from tbsim.tb import TBS
import pandas as pd
from enum import IntEnum, auto

__all__ = ['Product', 'TBVaccinationCampaign', 'ActiveCaseFinding']

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
    

class ActiveCaseFinding(ss.Intervention):
    """
    Intervention that simulates active case finding in active Tb cases. In more detail, this intervention --
    - Identifies the active TB individuals. This could be any one of the following states:
        ACTIVE_PRESYMP  (Active TB, pre-symptomatic)
        ACTIVE_SMPOS    (Active TB, smear positive)
        ACTIVE_SMNEG    (Active TB, smear negative)
        ACTIVE_EXPTB    (Active TB, extra-pulmonary)
    - Assign test sensitivity to accurately identify as Active Tb
    - With some coverage rate, the intervention identifies the active TB cases and assigns them to treatment.
    - The treatment will at some rate move individuals from repective Active TB state to Susceptible state -- similar to the TB module.
    - May need to think about how to keep track of individuals that get treated #TODO- perhaps an analyser?
    """
    def __init__(self, pars=None, *args, **kwargs):
        """
        Initialize the intervention.
        """
        super().__init__(*args, **kwargs)

        self.define_states(
            ss.State('found', default=False)
        )
        
        # Updated default parameters with time-aware p_found
        self.define_pars(
            p_found = ss.bernoulli(p=self.cov_fun),
            intv_year=np.arange(2000, 2004, 1),
            coverage=np.linspace(0.5, 0.1, 5),
            target_age_range=[15, 100],

            # test sensitivity relative to ACTIVE_SMPOS
            rel_sens_presymp=0.9,
            rel_sens_smpos=1.0,
            rel_sens_smneg=0.8,
            rel_sens_exptb=0.1
        )
        
        self.update_pars(pars, **kwargs)
       
        
        # make sure that the intv_range and coverage have the same dimensions
        assert np.shape(self.pars.intv_year) == np.shape(self.pars.coverage), "intv_year and coverage must have the same dimensions"
            
    @staticmethod
    def cov_fun(self, sim, uids):
        
        baseline_coverage = np.interp(self.now, xp=self.pars.intv_range, fp=self.pars.coverage)
                        
        p = np.select(
            condlist=[
            sim.diseases.tb.state[uids] == TBS.ACTIVE_PRESYMP,
            sim.diseases.tb.state[uids] == TBS.ACTIVE_SMPOS,
            sim.diseases.tb.state[uids] == TBS.ACTIVE_SMNEG,
            sim.diseases.tb.state[uids] == TBS.ACTIVE_EXPTB
            ],
            choicelist=[
            self.pars.rel_sens_presymp * baseline_coverage,
            self.pars.rel_sens_smpos * baseline_coverage,
            self.pars.rel_sens_smneg * baseline_coverage,
            self.pars.rel_sens_exptb * baseline_coverage
            ],
            default=0
        )
        
        # exclusion criterion for the intervention by age
        too_young = sim.people.age[uids] < self.pars.target_age_range[0]
        p[too_young] = 0
        too_old = sim.people.age[uids] > self.pars.target_age_range[1]
        p[too_old] = 0

        return p
    

    def init_results(self):
        
        self.define_results(
            ss.Result('n_aff_presym',   dtype=int, label='Treated Active Pre-Symptomatic'),
            ss.Result('n_aff_smpos',    dtype=int, label='Treated Active Smear Positive'),
            ss.Result('n_aff_smneg',    dtype=int, label='Treated Active Smear Negative'),
            ss.Result('n_aff_exptb',    dtype=int, label='Treated Active Extra-Pulmonary'),
            ss.Result('cum_aff_presym', dtype=int, label='Total Treated Active Pre-Symptomatic'),
            ss.Result('cum_aff_smpos',  dtype=int, label='Total Treated Active Smear Positive'),
            ss.Result('cum_aff_smneg',  dtype=int, label='Total Treated Active Smear Negative'),
            ss.Result('cum_aff_exptb',  dtype=int, label='Total Treated Active Extra-Pulmonary')
        )
    
        return

    def step(self):
        """
        Apply the intervention
        """
        
        super().step()

        sim = self.sim
        # ti = self.ti

        # check if the time is between the intv_range
        # when to use sim.now vs sim.ti ?
        if not np.any(np.isclose(sim.now, self.pars.intv_year, atol=1e-2)):
            return
            
        tb = sim.diseases['tb']

        active = np.isin(tb.state, [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
        eligible_uids = ss.uids(active & ~tb.on_treatment)
        
        # apply coverage
        found_uids = self.pars.p_found.filter(eligible_uids)
        # track the found individuals 
        # Mark the individuals as being on treatment
        self.found[found_uids] = True
        
        # apply treatment
        tb.start_treatment(found_uids)
        
        # append the results 
        # self.results.n_aff_presym[ti] = len(treated_presym_uids)
        # self.results.n_aff_smpos[ti] = len(treated_smpos_uids)
        # self.results.n_aff_smneg[ti] = len(treated_smneg_uids)
        # self.results.n_aff_exptb[ti] = len(treated_exptb_uids)
        return 

    def finalize_results(self):
        super().finalize_results()    
        self.results.cum_aff_presym = np.cumsum(self.results.n_aff_presym)
        self.results.cum_aff_smpos = np.cumsum(self.results.n_aff_smpos)
        self.results.cum_aff_smneg = np.cumsum(self.results.n_aff_smneg)
        self.results.cum_aff_exptb = np.cumsum(self.results.n_aff_exptb)
        return
    
    



