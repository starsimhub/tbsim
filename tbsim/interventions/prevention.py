"""
Prevention: BCG vaccination and TB preventive therapy (TPT), as Starsim v4 products

A product acts on the transitions of `tbsim.TB` for the people who hold it (see
`ss.Vaccine` and `ss.Treatment`), so it runs under every solver. Deliver them with
`ss.vaccinate()` and `ss.treat()`, e.g.::

    tb = tbsim.TB()
    bcg = ss.vaccinate(tbsim.BCG(), eligible=ppl.age < 1, coverage=0.9)
    tpt = ss.treat(tbsim.TPT(), eligible=tb.infection, coverage=0.3, interval=ss.years(1))

With drug resistance, TPT protects only against the strains its drugs act on: see `TPT(drugs=...)`.
"""
import numpy as np
import starsim as ss

__all__ = ['BCG', 'TPT']


def BCG(take=0.8, activation=0.575, clearance=1.4, death=0.1, dur=ss.years(10), **kwargs):
    """
    The BCG vaccine: an `ss.Vaccine` that slows progression to active TB, speeds recovery, and lowers TB mortality

    Each effect is a multiplier on the rate for people who respond to the vaccine
    (a share `take` of those vaccinated); the vaccine's efficacy is the average over everyone vaccinated.

    Args:
        take (float): the share of those vaccinated who respond
        activation (float): the multiplier on progression from latent infection to active TB
        clearance (float): the multiplier on recovery from non-infectious TB
        death (float): the multiplier on TB mortality
        dur (dur): the mean time protection lasts (exponential)
        kwargs (dict): passed to `ss.Vaccine`, e.g. `cost=`
    """
    eff = lambda mult: take*(1 - mult) # The efficacy is 1 minus the multiplier, averaged over responders and others
    efficacy = dict(latent_asy=eff(activation), latent_non=eff(activation), recovery=eff(clearance), tb_death=eff(death))
    kwargs.setdefault('name', 'bcg')
    return ss.Vaccine(efficacy=efficacy, waning=dur, **kwargs)


def TPT(efficacy=0.6, activation=0.4, dur=ss.years(2), tb=None, drugs=None, **kwargs):
    """
    TB preventive therapy: an `ss.Treatment` that slows progression from latent infection to active TB while it protects

    With `tb`, it acts on every transition from latent infection to active TB (including from a
    remote infection, and those on which resistance is acquired); otherwise on `latent_asy` and
    `latent_non`. With `drugs` as well, it is strain-aware: it protects only against the strains
    that are susceptible to all of its drugs, so someone who carries several strains is
    protected in proportion to the share of them it covers, and someone with only resistant
    strains not at all. (A model with `tpt=` gives its drugs, and the other effects of TPT on
    the infection of the people who hold it: see `tbsim.TB`.)

    Args:
        efficacy (float): the share of those treated whom it protects
        activation (float): the multiplier on progression to active TB for those it protects
        dur (dur): the mean time protection lasts (exponential)
        tb (`tbsim.TB`): the TB model
        drugs (list): the drugs of the regimen (e.g. `['INH']` for isoniazid, `['INH', 'RIF']` for 3HP), with a model with drugs (default: the model's `tpt=` drugs, if any)
        kwargs (dict): passed to `ss.Treatment`, e.g. `cost=`

    Example::

        tb = tbsim.TB(drugs=['RIF', 'INH'])
        tpt = ss.treat(tbsim.TPT(tb=tb, drugs=['INH']), eligible=tb.infection, coverage=0.3, interval=ss.years(1))
    """
    eff = efficacy*(1 - activation)
    if drugs is not None and tb is None:
        errormsg = 'TPT(drugs=...) needs the TB model too, for its strains: TPT(tb=tb, drugs=...)'
        raise ValueError(errormsg)
    if drugs is None and getattr(tb, 'tpt', None) and tb.tpt.drugs:
        drugs = tb.tpt.drugs
    if tb is None:
        keys = ['latent_asy', 'latent_non']
    else:
        latent = {'infection', 'infection_remote'}
        active = {'non_infectious', 'asymptomatic', 'symptomatic'}
        keys = [name for name,tr in tb.graph.transitions.items() if set(s.name for s in tr.src) <= latent and any(s.name in active for s in tr.dst)]
    if drugs is not None: # The share of the strains people carry that the regimen covers, at each level of resistance
        from .care import resistant_levels
        resistant_levels(tb, drugs) # Checks that the model has these drugs
        drugs = [drugs] if isinstance(drugs, str) else list(drugs)
        labels = [strat for strat in tb.graph.strats if strat.name == 'resistance'][0].labels
        covered = [np.mean([not any(d in strain.split('/') for d in drugs) for strain in label.split('+')]) for label in labels]
        eff = ss.by(resistance=[eff*cov for cov in covered])
    kwargs.setdefault('name', 'tpt')
    product = ss.Treatment(efficacy={key:eff for key in keys}, waning=dur, **kwargs)
    product.is_tpt = True # So the model's effects of TPT on the infection (tbsim.TB(tpt=...)) apply to the people who hold it
    return product
