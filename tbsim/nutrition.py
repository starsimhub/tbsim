"""
Define non-communicable disease (Nutrition) model
"""

import numpy as np
import starsim as ss
import sciris as sc
from starsim.diseases.ncd import NCD

__all__ = ['Nutrition']

class Nutrition(NCD):
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
        super().__init__(pars=pars)

        # Generic Nutrition States
        self.add_states(
            # ss.State('at_risk', bool, True),                     # At risk for malnutrition - PART OF NCD BASE CLASS
            ss.State('normal', bool, True),                     # Normal nutrition state (no malnutrition or overnutrition
            ss.State('undernutrition', bool, False),            # malnutrition, this includes stunting, wasting, underweight 
            ss.State('overnutrition', bool, False),             # overweight, obesity, and the extention of this would be the related NCD  
            ss.State('micronutrient_deficiency', bool, False),  # micronutrient-imbalance:  allow use of the disease state without being too specific
        )
        self.add_states(
            ss.State('ti_undernutrition', int, ss.INT_NAN),     # Time at which undernutrition was diagnosed
            ss.State('ti_overnutrition', int, ss.INT_NAN),      # Time at which overnutrition was diagnosed
            ss.State('ti_micronutrient_deficiency', int, ss.INT_NAN), # Time at which micronutrient deficiency was diagnosed
        )
        return

    @property
    def not_at_risk(self):
        return ~self.at_risk

    def update_pre(self, sim):
        # Make all the updates from the NCD model 
        super().update_pre(sim)
        
        # At risk --> Undernutrition
        inds = ss.true(self.normal & (self.ti_undernutrition <= sim.ti))
        if len(inds):
            self.normal[inds] = False
            self.undernutrition[inds] = True
            
        inds = ss.true(self.normal & (self.ti_overnutrition <= sim.ti))
        if len(inds):
            self.normal[inds] = False
            self.undernutrition[inds] = True
            
        inds = ss.true(self.normal & (self.ti_micronutrient_deficiency <= sim.ti))
        if len(inds):
            self.normal[inds] = False
            self.undernutrition[inds] = True
            
        return

    def make_new_cases(self, sim):
        super().make_new_cases(sim)

        new_cases = ss.true(self.ti_undernutrition == sim.ti)
        self.undernutrition[new_cases] = True
        # prog_years = self.pars.prognosis.rvs(new_cases)
        # self.ti_dead[new_cases] = sim.ti + sc.randround(prog_years / sim.dt)

        self.set_prognoses(sim, new_cases)
        return new_cases
    
    def set_prognoses(self, sim, new_cases):
        super().set_prognoses(sim, new_cases)
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
            ss.State('micronutrient_deficiency', bool, False),  # micronutrient deficiencies, a lack of important vitamins and minerals.
            ss.State('macronutrient_deficiency', bool, False),  # macronutrient deficiencies 
            ss.State('micronutrient_excess', bool, False),      # micro nutrient excess
            ss.State('macronutrient_excess', bool, False),      # macro nutrient excess
            ss.State('overweight', bool, False),                # BMI of 30 or more
            ss.State('obesity', bool, False),                   # BMI of 30 or more
        )