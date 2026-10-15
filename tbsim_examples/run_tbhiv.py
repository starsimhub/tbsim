"""
TB-HIV coinfection: TB with and without HIV

HIV acts on TB through modifiers (`tbsim.hiv_effects()`) that speed progression from
latent infection to active TB, by HIV state. People are infected with HIV at a given
incidence from 2000 onwards, and half of the people with HIV are put on ART in 2010 and
again in 2020 (`ss.transfer()`). With no network, the sim runs as an ODE, with TB and HIV
tracked jointly; for agents, use `solver='abm'`.
"""

import starsim as ss
import tbsim


def make_sim(hiv_incidence=0, label=None):
    """ A TB-HIV sim; with `hiv_incidence`, people are infected with HIV at that rate per year from 2000 """
    tb = tbsim.TB(init=dict(infection=0.25))
    incidence = ss.timeseries({1980: ss.peryear(0.0), 2000: ss.peryear(hiv_incidence)})
    hiv = tbsim.HIV(incidence=incidence)
    art = ss.transfer(art=True, coverage=0.5, years=[2010, 2020], eligible=hiv.acute | hiv.latent | hiv.aids)
    sim = tbsim.Sim(
        tb = tb,
        diseases = hiv,
        connectors = tbsim.hiv_effects(tb, hiv),
        interventions = art,
        start = 1980,
        stop = 2035,
        verbose = 0,
        label = label,
    )
    return sim


if __name__ == '__main__':
    sims = [
        make_sim(label='No HIV'),
        make_sim(hiv_incidence=0.02, label='HIV incidence 2% a year'),
    ]
    for sim in sims:
        print(f'Running: {sim.label}')
        sim.run()

    msim = ss.MultiSim(sims)
    tbsim.plot(msim, select=['prevalence_active', 'active', 'n_symptomatic', 'infected', 'hiv_prevalence', 'n_aids'], title='TB-HIV coinfection', n_cols=3)
