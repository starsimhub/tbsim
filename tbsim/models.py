"""
The reference model of drug resistance, declared as states and transitions (Starsim v4)

- `LSHTMResistant`: the model of `compartmental/two_strain_ode.R`, with strains resistant to any
  number of drugs. Its natural history is that of `tbsim.TB`, but treatment is as in the
  reference: it starts at a rate from asymptomatic and from symptomatic TB, and a failure
  returns people to where they were. With `drugs=['R']` (the default) it is the reference ODE.

Rates are per year. Births replace every death (background and TB), so the population
stays the same size, as in the reference ODE.
"""
import sciris as sc
import starsim as ss
import tbsim

from .tb import tb_pars, make_pars

__all__ = ['LSHTMResistant']


def reference_pars():
    """ The parameters of the reference model, per year (as in two_strain_ode.R); see also `tbsim.resistance_pars()` """
    pars = tb_pars()
    pars.beta = ss.par(ss.peryear(16.45), bounds=[1, 50], doc=pars.beta.doc)
    for key in ['theta', 'phi', 'rr_reinfection_inf', 'dur_reinfection_protection', 'k_latent', 'inf_non_remote', 'inf_asy_remote', 'p_relapse', 'dur_relapse']: # Not in the reference model
        pars.pop(key)
    pars.update(
        r_treat_asym = ss.peryear(0.01),  # Start of treatment from asymptomatic TB
        r_treat_sym  = ss.peryear(1.204), # ... and from symptomatic TB
    )
    return pars


class LSHTMResistant(ss.Disease):
    """
    The LSHTM model with strains resistant to some drugs: the reference ODE of drug resistance (`two_strain_ode.R`)

    The states of infection and disease are split by `resistance`, whose levels are the sets of
    strains people carry (`DS`, `R`, and `DS+R` with one drug); see `tbsim.add_resistance()`.
    Treatment is split by the state it started from (`treat_asy`, `treat_sym`).

    The parameters of `two_strain_ode.R` map to these: `fit_b` is `fitness`, `eff_a` is `cure`,
    `eff_b` is `cure*rr_cure`, and `rr_reinfection_inf`, `_non`, `_asy`, and `_sym` are
    `rr_superinfection_latent`, `_non`, `_asy`, and `_sym`.

    Args:
        init (dict): the number (or share) of people starting in each state (default: 1,010 symptomatic, 1% of them with a resistant strain)
        drugs (list): the drugs that strains can be resistant to
        max_strains (int): the most strains anyone can carry at once
        strains (dict): the share of each set of strains among people infected at the start (default: 1% resistant to the first drug)
        cure (float): the chance that treatment cures a drug-susceptible strain
        kwargs (dict): parameters, by name (see `reference_pars()` and `tbsim.resistance_pars()`)
    """
    def __init__(self, init=None, drugs=('R',), max_strains=2, strains=None, cure=0.75, name='tb', **kwargs):
        res_kw = {key:kwargs.pop(key) for key in list(kwargs) if key in tbsim.resistance_pars()}
        p = make_pars(reference_pars(), kwargs)

        S = sc.objdict()
        for key,label in [('susceptible', 'Susceptible'), ('cleared', 'Cleared (from latent)'), ('recovered', 'Recovered (from non-infectious)'), ('treated', 'Completed treatment'),
                          ('infection', 'Latent infection'), ('non_infectious', 'Non-infectious TB'), ('asymptomatic', 'Asymptomatic TB'), ('symptomatic', 'Symptomatic TB'),
                          ('treat_asy', 'On treatment, from asymptomatic'), ('treat_sym', 'On treatment, from symptomatic')]:
            S[key] = ss.State(name=key, label=label, infectious=key in ['asymptomatic', 'symptomatic'])

        T = sc.objdict()
        def add(name_, src, dst, rate):
            T[name_] = ss.Transition(S[src], S[dst], rate=rate, name=name_)

        everyone = S.susceptible
        for st in S.values()[1:]:
            everyone = everyone | st
        foi = p.beta*(p.trans_asymp*S.asymptomatic + S.symptomatic)
        add('transmission', 'susceptible', 'infection', foi)
        add('reinfect_cle', 'cleared', 'infection', p.rr_reinfection_cleared*foi)
        add('reinfect_rec', 'recovered', 'infection', p.rr_reinfection_rec*foi)
        add('reinfect_trd', 'treated', 'infection', p.rr_reinfection_treat*foi)
        add('clearance', 'infection', 'cleared', p.inf_cle)
        add('latent_non', 'infection', 'non_infectious', p.inf_non)
        add('latent_asy', 'infection', 'asymptomatic', p.inf_asy)
        add('recovery', 'non_infectious', 'recovered', p.non_rec)
        add('worsening', 'non_infectious', 'asymptomatic', p.non_asy)
        add('improvement', 'asymptomatic', 'non_infectious', p.asy_non)
        add('symptoms', 'asymptomatic', 'symptomatic', p.asy_sym)
        add('remission', 'symptomatic', 'asymptomatic', p.sym_asy)
        T.tb_death = ss.death('symptomatic ->', rate=p.sym_dead, name='tb_death')
        add('start_treat_asy', 'asymptomatic', 'treat_asy', p.r_treat_asym)
        add('start_treat_sym', 'symptomatic', 'treat_sym', p.r_treat_sym)
        for kind,back in [('asy', 'asymptomatic'), ('sym', 'symptomatic')]: # Treatment ends at one rate; its outcome depends on the strains
            add(f'completion_{kind}', f'treat_{kind}', 'treated', p.delta)
            add(f'failure_{kind}', f'treat_{kind}', back, p.delta)

        strat = tbsim.add_resistance(S, T, p, foi, drugs=drugs, max_strains=max_strains, split=sc.ifelse(strains, {'DS':100/101, drugs[0]:1/101}),
            superinfect = dict(infection='rr_superinfection_latent', non_infectious='rr_superinfection_non', asymptomatic='rr_superinfection_asy', symptomatic='rr_superinfection_sym'),
            bottleneck  = ['latent_asy', 'worsening'],
            symptoms    = ['symptoms'],
            clear       = ['clearance', 'recovery'],
            denovo      = ['latent_non', 'latent_asy'],
            treatment   = [('completion_asy', 'failure_asy'), ('completion_sym', 'failure_sym')],
            cure        = cure,
            **res_kw)

        # Demography: births replace every death
        T.births  = ss.birth('-> susceptible', rate=p.mu, name='births')
        T.replace = ss.null >> S.susceptible @ (p.sym_dead*S.symptomatic.n) # Births that replace TB deaths
        T.deaths  = ss.death(rate=p.mu, name='deaths')

        super().__init__(
            *S.values(),
            **T,
            infectious = S.asymptomatic + S.symptomatic,
            active     = S.non_infectious + S.asymptomatic + S.symptomatic,
            init       = sc.ifelse(init, dict(symptomatic=1010)),
            name       = name,
        )
        self.stratify(**strat)
        return
