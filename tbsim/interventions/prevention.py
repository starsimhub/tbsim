"""
Prevention: BCG vaccination and TB preventive therapy (TPT), as Starsim v4 products

A product acts on the transitions of `tbsim.TB` for the people who hold it (see
`ss.Vaccine` and `ss.Treatment`), so it runs under every solver. Deliver them with
`ss.vaccinate()` and `ss.treat()`, e.g.::

    tb = tbsim.TB()
    bcg = ss.vaccinate(tbsim.BCG(), eligible=ppl.age < 1, coverage=0.9)
    tpt = ss.treat(tbsim.TPT(), eligible=tb.infection, coverage=0.3, interval=ss.years(1))
"""
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


def TPT(efficacy=0.6, activation=0.4, dur=ss.years(2), **kwargs):
    """
    TB preventive therapy: an `ss.Treatment` that slows progression from latent infection to active TB while it protects

    Args:
        efficacy (float): the share of those treated whom it protects
        activation (float): the multiplier on progression to active TB for those it protects
        dur (dur): the mean time protection lasts (exponential)
        kwargs (dict): passed to `ss.Treatment`, e.g. `cost=`
    """
    eff = efficacy*(1 - activation)
    kwargs.setdefault('name', 'tpt')
    return ss.Treatment(efficacy=dict(latent_asy=eff, latent_non=eff), waning=dur, **kwargs)
