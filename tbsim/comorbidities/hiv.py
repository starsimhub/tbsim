"""
HIV, declared as states and transitions (so that it runs as an ODE, as a stochastic compartmental model, or as agents), and its effect on TB
"""
import sciris as sc
import starsim as ss
import tbsim

__all__ = ['HIV', 'hiv_pars', 'hiv_effects']


def hiv_pars():
    """ The parameters of the HIV model """
    return sc.objdict(
        incidence       = ss.par(ss.peryear(0.0), bounds=[0, 0.2], doc='New HIV infections per person at risk per year'),
        acute_to_latent = ss.peryear(52/12),     # ACUTE -> LATENT (12 weeks)
        latent_to_aids  = ss.peryear(1/8),      # LATENT -> AIDS (8 years)
        art_start       = ss.peryear(0.0),      # The rate at which people with HIV start ART (0: none, or by a delivery, e.g. ss.Delivery(move=dict(art=True), coverage=0.8, at=[2010]))
        art_progression = 0.1,                  # The multiplier on progression for people on ART
    )


class HIV(ss.Disease):
    """
    A simple HIV model, declared as states and transitions

    People at risk (`atrisk`) are infected at a given incidence (`acute`), and progress
    to chronic infection (`latent`) and advanced disease (`aids`). People with HIV are
    stratified by `art`, whether they are on ART, which slows progression; new infections
    start off ART. People start ART at the rate `art_start`, or by a delivery,
    e.g. `ss.Delivery(move=dict(art=True), coverage=0.8, at=[2010])`.

    Results, besides those of every state and transition: `infected` (people with HIV)
    and `hiv_prevalence` (as a share of everyone). The number on ART is
    `sim.results.hiv.infected.by('art')`.

    Args:
        pars (dict): parameters, by name (as `kwargs`)
        init_prev (float): the share of people with (acute) HIV at the start
        init_onart (float): the share of them on ART at the start
        kwargs (dict): parameters, by name (see `hiv_pars()`): numbers (rates per day for the progression, per year for the others), or `ss.par` objects;
            `incidence` can vary in time, e.g. `ss.timeseries({2000: ss.peryear(0.01), 2020: ss.peryear(0.002)})`

    Examples::

        hiv = tbsim.HIV(incidence=0.01, art_start=0.2, init_prev=0.1)
        tb = tbsim.TB()
        sim = ss.Sim(tb, diseases=hiv, connectors=tbsim.hiv_effects(tb, hiv), start=2000, stop=2030)
    """
    def __init__(self, pars=None, init_prev=0.0, init_onart=0.0, name='hiv', **kwargs):
        kwargs = sc.mergedicts(pars, kwargs)
        p = ss.pars(hiv_pars(), **kwargs)
        atrisk = ss.State('atrisk', label='At risk')
        acute  = ss.State('acute', label='Acute HIV')
        latent = ss.State('latent', label='Chronic HIV')
        aids   = ss.State('aids', label='AIDS')
        infection = ss.Transition(atrisk, acute, rate=p.incidence, move=dict(art=False), name='infection')
        to_latent = ss.Transition(acute, latent, rate=p.acute_to_latent, name='to_latent')
        to_aids   = ss.Transition(latent, aids, rate=p.latent_to_aids, name='to_aids')
        super().__init__(
            atrisk, acute, latent, aids,
            infection      = infection,
            to_latent      = to_latent,
            to_aids        = to_aids,
            infected       = acute + latent + aids,
            hiv_prevalence = (acute + latent + aids)/(atrisk + acute + latent + aids),
            init           = dict(acute=init_prev),
            first          = 'atrisk',
            name           = name,
        )
        starts = isinstance(kwargs.get('art_start'), ss.par) or p.art_start.raw.value > 0 # Given to be fitted, or not zero
        slow = [1.0, p.art_progression.raw]
        move = dict(move=ss.transfer(art=True, rate=p.art_start)) if starts else {}
        self.stratify(art=[False, True], states=['acute', 'latent', 'aids'], split=[1 - init_onart, init_onart], adjust=dict(to_latent=slow, to_aids=slow), **move)
        return


def hiv_effects(tb, hiv, acute=1.2202, latent=1.9001, aids=2.5955):
    """
    The effect of HIV on TB: a multiplier on progression from latent infection to active TB, by HIV state

    Returns modifiers, to pass to the sim as `connectors`. In a compartmental run, the states
    of TB and HIV are then tracked jointly (see `ss.Crossing`).

    Args:
        tb (`tbsim.TB`): the TB model
        hiv (`tbsim.HIV`): the HIV model
        acute, latent, aids (float): the multiplier for each HIV state (TODO: citation needed)

    Example::

        tb = tbsim.TB()
        hiv = tbsim.HIV(incidence=0.01, init_prev=0.1)
        sim = ss.Sim(tb, diseases=hiv, connectors=tbsim.hiv_effects(tb, hiv))
    """
    out = []
    for state,mult in dict(acute=acute, latent=latent, aids=aids).items():
        for tr in [tb.latent_asy, tb.latent_non]:
            out.append(ss.modify(tr, multiply=mult, where=getattr(hiv, state), name=f'hiv_{state}_{tr.name}'))
    return out
