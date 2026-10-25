"""
Health-seeking behavior with the LSHTM TB model.

People with symptomatic TB seek care, are tested with Xpert, and start treatment if
they test positive: the care cascade, tbsim.care(). Three scenarios are run over 20 years
and compared:

  Baseline              – no care seeking
  Low rate (1/year)     – people with symptomatic TB seek care once a year on average
  High rate (4/year)    – four times a year on average

Seeking care is a rate, so people who have not sought care yet can seek it later, and
people whose test is falsely negative can seek care again.
"""

import matplotlib.pyplot as plt
import starsim as ss
import tbsim


def build_sim(label, seeking=None):
    """ A TB simulation with DOTS treatment and, if a rate of seeking care is given, the care cascade """
    tb = tbsim.TB(beta=9, **tbsim.regimen('dots')).stratify(age=[0, 15]) # Xpert's sensitivity differs for children and adults
    care = tbsim.care(tb, seeking=seeking) if seeking is not None else None
    return tbsim.Sim(tb=tb, interventions=care, start=1990, stop=2010, label=label, verbose=0)


scenarios = {
    'Baseline': None,
    'Low rate (1/year)': ss.peryear(1),
    'High rate (4/year)': ss.peryear(4),
}


if __name__ == '__main__':
    sims = [build_sim(label, seeking) for label,seeking in scenarios.items()]
    msim = ss.MultiSim(sims)
    msim.run(parallel=False)

    for sim in msim.sims:
        res = sim.results
        diagnosed = res.care.positive.sum() if 'care' in res else 0
        print(f'{sim.label:20s}: {diagnosed:6.0f} diagnosed, {res.tb.tb_death.cum[-1]:5.0f} TB deaths, prevalence of infectious TB {res.tb.prevalence_active[-1]*1e5:.0f} per 100,000')

    tbsim.plot(
        msim,
        select = ['symptomatic', 'prevalence_active', 'transmission', 'tb_death', 'diagnosis', 'treatment'],
        title = 'Health-seeking behavior – LSHTM TB model',
        n_cols = 3,
    )
    plt.show()
