"""
TB and malnutrition: TB with and without the effects of nutrition

The malnutrition model tracks each agent's weight and height (as percentiles of growth
curves) and micronutrient status. Nutrition acts on TB through modifiers
(`tbsim.nutrition_effects()`): here, a low BMI speeds progression to active TB
(`tbsim.lonnroth_bmi_rr`), and poor micronutrient status doubles susceptibility.
Malnutrition is agent-based, so the sim runs as agents.
"""

import starsim as ss
import tbsim


def make_sim(nutrition=True, label=None):
    """ A TB-malnutrition sim, with or without the effects of nutrition on TB """
    tb = tbsim.TB(init=dict(infection=0.25))
    effects = tbsim.nutrition_effects(tb, rr_activation=tbsim.lonnroth_bmi_rr) if nutrition else None
    sim = tbsim.Sim(
        tb = tb,
        diseases = tbsim.Malnutrition(),
        connectors = effects,
        solver = 'abm',
        use_aging = True, # Agents age, so that their height and weight follow the growth curves
        n_agents = 2000,
        start = '1980-01-01',
        stop = '2000-01-01',
        rand_seed = 1,
        verbose = 0,
        label = label,
    )
    return sim


if __name__ == '__main__':
    sims = [
        make_sim(nutrition=False, label='No nutrition effects'),
        make_sim(nutrition=True, label='With nutrition effects'),
    ]
    for sim in sims:
        print(f'Running: {sim.label}')
        sim.run()

    msim = ss.MultiSim(sims)
    tbsim.plot(msim, select=['prevalence_active', 'active', 'infection', 'symptomatic'], title='TB and malnutrition', n_cols=2)
