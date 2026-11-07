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

from .tb import tb_pars

__all__ = ['LSHTMResistant']


def reference_pars():
    """ The parameters of the reference model, per year (as in two_strain_ode.R); see also `tbsim.resistance_pars()` """
    pars = tb_pars()
    pars.beta = ss.par(ss.peryear(16.45), bounds=[0, 50], doc=pars.beta.doc)
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

    The model has strains (`self.strains`, an `ss.Strains` called `resistance`), whose sets people
    carry (`DS`, `R`, and `DS+R` with one drug); see `tbsim.add_resistance()`. Treatment is split
    by the state it started from (`treat_asy`, `treat_sym`).

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
    def __init__(self, init=None, init_n=None, drugs=('R',), max_strains=2, strains=None, cure=0.75, name='tb', **kwargs):
        res_kw = {key:kwargs.pop(key) for key in list(kwargs) if key in tbsim.resistance_pars()}
        p = ss.pars(reference_pars(), **kwargs)

        # Treatment ends at one rate, starting from asymptomatic or symptomatic TB; its outcome depends on the strains, and a failure returns people to where they were
        regimens = [dict(name=f'completion_{kind}', src=f'treat_{kind}', cured='treated', failed=back, rate=p.delta, cure=cure) for kind,back in [('asy', 'asymptomatic'), ('sym', 'symptomatic')]]
        split = sc.ifelse(strains, {'DS':100/101, drugs[0]:1/101}) # By default, 1% of infections resistant to the first drug
        bottleneck = ['latent_asy', 'worsening'] # The transitions to asymptomatic TB
        strains = tbsim.make_strains(drugs, max_strains=max_strains, split=split, regimens=regimens, alone=bottleneck, **res_kw)

        super().__init__(
            init    = init,
            init_n  = sc.ifelse(init_n, None if init else dict(symptomatic=1010)),
            strains = strains,
            name    = name,
        )

        # The states
        self.define_states(
            ss.State('susceptible',    label='Susceptible'),
            ss.State('cleared',        label='Cleared (from latent)'),
            ss.State('recovered',      label='Recovered (from non-infectious)'),
            ss.State('treated',        label='Completed treatment'),
            ss.State('infection',      label='Latent infection'),
            ss.State('non_infectious', label='Non-infectious TB'),
            ss.State('asymptomatic',   label='Asymptomatic TB'),
            ss.State('symptomatic',    label='Symptomatic TB'),
            ss.State('treat_asy',      label='On treatment, from asymptomatic'),
            ss.State('treat_sym',      label='On treatment, from symptomatic'),
        )

        # Infection, and reinfection at a relative risk that depends on how people stopped being infected
        foi = p.beta*(p.trans_asymp*self.asymptomatic + self.symptomatic)
        self.define_transitions(
            transmission = self.susceptible >> self.infection @ foi,
            reinfect_cle = self.cleared     >> self.infection @ p.rr_reinfection_cleared*foi,
            reinfect_rec = self.recovered   >> self.infection @ p.rr_reinfection_rec*foi,
            reinfect_trd = self.treated     >> self.infection @ p.rr_reinfection_treat*foi,
        )

        # Latent infection, and the spectrum of TB disease (as in tbsim.TB)
        self.define_transitions(
            clearance   = self.infection      >> self.cleared        @ p.inf_cle,
            latent_non  = self.infection      >> self.non_infectious @ p.inf_non,
            latent_asy  = self.infection      >> self.asymptomatic   @ p.inf_asy,
            recovery    = self.non_infectious >> self.recovered      @ p.non_rec,
            worsening   = self.non_infectious >> self.asymptomatic   @ p.non_asy,
            improvement = self.asymptomatic   >> self.non_infectious @ p.asy_non,
            symptoms    = self.asymptomatic   >> self.symptomatic    @ p.asy_sym,
            remission   = self.symptomatic    >> self.asymptomatic   @ p.sym_asy,
            tb_death    = ss.death(self.symptomatic, rate=p.sym_dead),
        )

        # Treatment starts at a rate from asymptomatic and from symptomatic TB
        self.define_transitions(
            start_treat_asy = self.asymptomatic >> self.treat_asy @ p.r_treat_asym,
            start_treat_sym = self.symptomatic  >> self.treat_sym @ p.r_treat_sym,
        )

        # Drug resistance: superinfection, progression and treatment by strain, and resistance acquired
        tbsim.add_resistance(self, strains, foi,
            superinfect = dict(infection='rr_superinfection_latent', non_infectious='rr_superinfection_non', asymptomatic='rr_superinfection_asy', symptomatic='rr_superinfection_sym'),
            bottleneck  = bottleneck,
            symptoms    = ['symptoms'],
            clear       = ['clearance', 'recovery'],
            denovo      = ['latent_non', 'latent_asy'],
            regimens    = regimens,
            **res_kw,
        )

        # Demography: births replace every death
        self.define_transitions(
            births  = ss.birth(self.susceptible, rate=p.mu),
            replace = ss.unborn >> self.susceptible @ p.sym_dead*self.symptomatic.n, # Births that replace TB deaths
            deaths  = ss.death(rate=p.mu),
        )

        # Results, besides those of every state and transition
        infectious = self.asymptomatic + self.symptomatic
        self.define_results(
            infectious = infectious,
            active     = self.non_infectious + infectious,
        )
        return
