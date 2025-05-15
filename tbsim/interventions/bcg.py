
import numpy as np
import starsim as ss

__all__ = ['BCGProtection']

class BCGProtection(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            year=[1900],
            coverage=0.95,
            target_age=5,
        )
        self.update_pars(pars=pars, **kwargs)
        self.vaccinated = None

    def initialize(self, sim):
        super().initialize(sim)
        ppl = sim.people
        eligible = (ppl.age < self.pars.target_age) & ppl.alive
        n_eligible = np.count_nonzero(eligible)
        vaccinated_mask = sim.rng.binomial(1, self.pars.coverage, n_eligible)
        self.vaccinated = eligible.uids[vaccinated_mask == 1]
        ppl['vaccination_year'][self.vaccinated] = sim.year

    def step(self):
        if self.vaccinated is None:
            return

        tb = self.sim.diseases.tb
        ppl = self.sim.people

        for uid in self.vaccinated:
            if not ppl.alive[uid]:
                continue

            age = ppl.age[uid]
            tb.rel_SL_prog[uid] *= 0.81

            if age < 3:
                tb.rel_LF_prog[uid] *= 0.58
            elif 3 <= age <= 15:
                tb.rel_LF_prog[uid] *= 0.81

            if age < 5:
                tb.rel_EP_prog[uid] *= 0.63

            if age < 5:
                tb.rel_ATB_death[uid] *= 0.20
                tb.rel_EP_death[uid] *= 0.20
            elif 5 <= age <= 14:
                tb.rel_ATB_death[uid] *= 0.13
                tb.rel_EP_death[uid] *= 0.13

    def init_results(self):
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )
        
    def update_results(self):
        self.results['n_vaccinated'][self.ti] = len(self.vaccinated) if self.vaccinated is not None else 0
        self.results['n_eligible'][self.ti] = np.count_nonzero(self.sim.people.age < self.pars.target_age)
        
   