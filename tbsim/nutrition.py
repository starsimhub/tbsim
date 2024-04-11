"""
Define non-communicable disease (Nutrition) model
"""

import numpy as np
import starsim as ss
import sciris as sc
from starsim.diseases.ncd import NCD

__all__ = ['Nutrition']

class Nutrition(ss.Disease):
    """
    This class implements a basic Nutrition model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having a condition, or accelerate a condition 
    (e.g., TB, HIV) and associated mortality.
    """
    # Possible references:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9971264/
    # https://www.espen.org/files/ESPEN-guidelines-on-definitions-and-terminology-of-clinical-nutrition.pdf 
    
    def __init__(self, pars=None):
        # According to https://www.who.int/news-room/questions-and-answers/item/malnutrition
        # Actual Disease class initialization.
        pars = ss.omergeleft(pars,
               # prognosis defaults
               # c: c>1 might be appropriate if the risk of death increases 
               # the longer the individual remains malnourished without effective intervention.
               c = 0,
               # scale:  This parameter would depend significantly on factors such as the severity of malnutrition, available healthcare, and intervention effectiveness. A higher scale value 
               # might suggest a longer survival time, potentially indicating better healthcare access or intervention.
               scale = 0,
        )
        pars = ss.omergeleft(pars,
            undern_init_risk = ss.bernoulli(p=0.3),     # undernutrition initial prevalence of risk factors
            vmd_init_risk = ss.bernoulli(p=0.1),        # VMD Initial prevalence of risk factors
            dur_risk = ss.expon(scale=10),
            
            prognosis = ss.weibull(c=2, scale=5),       # 
        )      
        super().__init__(pars=pars)

        # Additional Nutrition States
        self.add_states(
            ss.State('at_risk', bool, False),           # Normal- at_risk nutrition state (no malnutrition or overnutrition
            ss.State('undernutrition', bool, False),    # malnutrition, this includes stunting, wasting, underweight 
            ss.State('vmd', bool, False),               # Vitamin and Mineral Deficiency
        )
        self.add_states(
            ss.State('ti_undern', int, ss.INT_NAN),     # Time at which undernutrition was diagnosed
            ss.State('ti_vmd', int, ss.INT_NAN),        # Time at which micronutrient deficiency was diagnosed
            ss.State('ti_dead', int, ss.INT_NAN),       # Time at which the person died
        )
        return

    @property
    def not_at_risk(self):
        return ~self.at_risk

    def set_prognoses(self, sim, new_cases):
        super().set_prognoses(sim, new_cases)
        return
    
    def make_new_cases(self, sim):

        new_cases = ss.true(self.ti_undern == sim.ti)
        self.undernutrition[new_cases] = True
       
        prog_years = self.pars.prognosis.rvs(new_cases)
        self.ti_dead[new_cases] = sim.ti + sc.randround(prog_years / sim.dt)
        super().set_prognoses(sim, new_cases)



        self.set_prognoses(sim, new_cases)
        return new_cases
    
    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'n_not_at_risk', sim.npts, dtype=int),
            ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
            ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
        ]
        return
    
    def update_pre(self, sim):
        # Make all the updates from the NCD model 
        deaths = ss.true(self.ti_dead == sim.ti)
        sim.people.request_death(deaths)
        self.log.add_data(deaths, died=True)
        self.results.new_deaths[sim.ti] = len(deaths) # Log deaths attributable to this module
        
        # At risk --> Undernutrition
        inds = ss.true(self.at_risk & (self.ti_undern <= sim.ti))
        if len(inds):
            self.at_risk[inds] = False
            self.undernutrition[inds] = True
            
            
        inds = ss.true(self.at_risk & (self.ti_vmd <= sim.ti))
        if len(inds):
            self.at_risk[inds] = False
            self.micronutrient_deficiency[inds] = True
            
        return

    



class MalnutritionX(NCD):
    """_summary_
    This class implements a basic Malnutrition model with risk of developing a condition
    including subforms of malnutrition.
    """
    def __init__(self, pars=None, par_dists=None, *args, **kwargs):


        pars = ss.omergeleft(pars,
            init_prev = 0.01,   # Initial prevalence - TODO: Check if there is one
            beta = 0.25,         # Transmission rate  - TODO: Check if there is one
        )
        
                # Subforms of malnutrition Nutrition States
        self.add_states(
            ss.State('stunting', bool, False),                  # undernutrition sub-form: low height for age
            ss.State('wasting', bool, False),                   # undernutrition sub-form: low weight for height
            ss.State('underweight', bool, False),               # undernutrition sub-form: low weight for age
            ss.State('mvd_vitamins', bool, False),              # micronutrient deficiencies, a lack of important vitamins 
            ss.State('mvd_minerals', bool, False),              # micronutrient deficiencies, a lack of important minerals.
            ss.State('macronutrient_deficiency', bool, False),  # macronutrient deficiencies 
            ss.State('micronutrient_excess', bool, False),      # micro nutrient excess
            ss.State('macronutrient_excess', bool, False),      # macro nutrient excess
            ss.State('overweight', bool, False),                # BMI of 30 or more
            ss.State('obesity', bool, False),                   # BMI of 30 or more
        )