"""
Define non-communicable disease (Nutrition) model
"""

import numpy as np
import starsim as ss
import sciris as sc

__all__ = ['Nutrition']

class Nutrition(ss.Disease):
    """
    This class implements a basic Nutrition model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having a condition, or accelerate a condition 
    (e.g., TB, HIV) and associated mortality.
    """
    def __init__(self, params=None, param_dists=None, param_defaults=None):
        # According to https://www.who.int/news-room/questions-and-answers/item/malnutrition

        # Generic Nutrition States
        self.add_states(
            ss.State('normal', bool, True),                     # Normal nutrition
            ss.State('undernutrition', bool, False),            # malnutrition, this includes stunting, wasting, underweight 
            ss.State('overnutrition', bool, False),             # overweight, obesity, and the extention of this would be the related NCD  
            ss.State('micronutrient_deficiency', bool, False),  # micronutrient-imbalance:  allow use of the disease state without being too specific
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
        
        params = ss.omergeleft(params, 
            # TODO: Natural history parameters, specify them in <days|weeks|months|years> 
            # TODO: Decide if makes sense to have these values incorporated and if it's possible to get them
            # one possibility that I found was: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
                dur_undernutrition = 0,            # Add source
                dur_overnutrition= 0,              # Add source
                dur_micronutrient_deficiency = 0,  # Add source
                )
        
        param_dists = ss.omergeleft(param_dists, 
            # TODO: Does it makes sense to only have the 3 mayor subdivitions of malnutrtion for param_dists duration?
            dur_undernutrition = ss.lognorm_ex,                 
            dur_overnutrition = ss.lognorm_ex,                      
            dur_micronutrient_imbalance = ss.lognorm_ex,
            )
        
        param_defaults = dict(
            prognosis = ss.weibull_min(c=2, scale=5),   # Time in years between first becoming undernutrition and death
            initial_risk = .07                          # Baseline case 
        )

        # Actual Disease class initialization.
        super().__init__(ss.omerge( params, param_dists, param_defaults))
        return

    @property
    def not_normal(self):
        return ~self.normal

    def set_initial_states(self, sim):
        """
        Setting initial nutrition states
        """
        alive_uids = ss.true(sim.people.alive)      
        initial_risk = self.params['initial_risk'].filter(alive_uids)
        # TODO: in-progress module.
        self.normal[initial_risk] = True
        self.ti_undernutrition[initial_risk] = sim.ti + sc.randround(self.params['dur_risk'].rvs(initial_risk) / sim.dt)
        return initial_risk

    def make_new_cases(self, sim):
        # TODO: in-progress module.
        new_cases = ss.true(self.ti_undernutrition == sim.ti)
        self.undernutrition[new_cases] = True
        prog_years = self.params['prognosis'].rvs(new_cases)
        self.ti_dead[new_cases] = sim.ti + sc.randround(prog_years / sim.dt)
        super().set_prognoses(sim, new_cases)
        return new_cases

    # def init_results(self, sim):
    #     """
    #     Initialize results
    #     """
    #     super().init_results(sim)
    #     self.results += [
    #         ss.Result(self.name, 'n_not_normal', sim.npts, dtype=int),
    #         ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
    #         ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
    #     ]
    #     return

    # def update_results(self, sim):
    #     super().update_results(sim)
    #     ti = sim.ti
    #     alive = sim.people.alive
    #     self.results.n_not_normal[ti] = np.count_nonzero(self.not_normal & alive)
    #     self.results.prevalence[ti]    = np.count_nonzero(self.undernutrition & alive)/alive.count()
    #     self.results.new_deaths[ti]    = np.count_nonzero(self.ti_dead == ti)
    #     return


