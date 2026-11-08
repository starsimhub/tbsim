"""
The TB model, declared as states and transitions (Starsim v4), so that it runs as an ODE, as a stochastic compartmental model, or as agents
"""
import sciris as sc
import starsim as ss
import tbsim

__all__ = ['TB', 'tb_pars', 'tpt_pars', 'treatment', 'get_tb']


def tb_pars():
    """ The parameters of the TB model: rates are per year (as in the LSHTM model, lshtm_ode.R) """
    return sc.objdict(
        beta                   = ss.par(ss.peryear(9.0), bounds=[0, 50], doc='Contacts per person per year (with homogeneous mixing; per contact with a network)'),
        trans_asymp            = 0.82,             # Infectiousness of asymptomatic TB, relative to symptomatic (kappa)
        rr_reinfection_cleared = 1.00,             # Relative risk of reinfection after clearing latent infection
        rr_reinfection_rec     = 0.21,             # ... after recovering from non-infectious TB (pi)
        rr_reinfection_treat   = 3.15,             # ... after completing treatment (rho)
        rr_reinfection_inf     = 0.21,             # ... of someone with a remote latent infection, which makes it recent again (sigma_L)
        dur_reinfection_protection = None,         # How long the protection (or risk) after clearance, recovery, or treatment lasts; None: for life
        inf_cle                = ss.peryear(1.90), # INFECTION -> CLEARED
        inf_non                = ss.peryear(0.16), # INFECTION -> NON_INFECTIOUS
        inf_asy                = ss.peryear(0.06), # INFECTION -> ASYMPTOMATIC
        k_latent               = ss.peryear(0.0),  # Front-loading: the rate at which a recent latent infection becomes remote (0: no remote stage)
        inf_non_remote         = None,             # INFECTION_REMOTE -> NON_INFECTIOUS (None: as for a recent infection)
        inf_asy_remote         = ss.peryear(0.0),  # INFECTION_REMOTE -> ASYMPTOMATIC
        non_rec                = ss.peryear(0.18), # NON_INFECTIOUS -> RECOVERED
        non_asy                = ss.peryear(0.25), # NON_INFECTIOUS -> ASYMPTOMATIC
        asy_non                = ss.peryear(1.66), # ASYMPTOMATIC -> NON_INFECTIOUS
        asy_sym                = ss.peryear(0.88), # ASYMPTOMATIC -> SYMPTOMATIC
        sym_asy                = ss.peryear(0.54), # SYMPTOMATIC -> ASYMPTOMATIC
        sym_dead               = ss.peryear(0.34), # SYMPTOMATIC -> dead (TB mortality)
        theta                  = ss.peryear(0.0),  # Diagnosis: active TB -> TREATMENT (0 by default: it is driven by a delivery, e.g. tbsim.care())
        phi                    = ss.peryear(0.0),  # Treatment failure: TREATMENT -> SYMPTOMATIC
        delta                  = ss.peryear(2.0),  # Treatment completion: TREATMENT -> TREATED (6 months)
        p_relapse              = 0.0,              # The share of completed treatments that relapse to symptomatic TB
        dur_relapse            = ss.years(1.5),    # The time from the end of treatment to relapse
        mu                     = ss.peryear(1/70), # Background mortality, with births replacing every death (0: none, e.g. to use ss.Births and ss.Deaths as agents)
        immigration            = ss.peryear(0.0),  # People arriving per year (a number, not per person), into the states given by immigrants=
        emigration             = ss.peryear(0.0),  # People leaving, per person per year
    )


def tpt_pars():
    """ The effects of TB preventive therapy on the infection itself (see `tbsim.TB(tpt=...)`); its protection is the product's (see `tbsim.TPT()`) """
    return sc.objdict(
        drugs          = None,             # The drugs of the regimen, with a model with drugs (default: all of them)
        clearance      = ss.peryear(0.0),  # The rate at which it clears the strains it acts on, while held
        acquisition    = 0.0,              # With drugs: the rate per year at which a strain it does not clear becomes resistant to each of its drugs, while held (a number, or a dict by drug)
        rr_acquisition = dict(infection=0.05, non_infectious=0.5, asymptomatic=1.0, symptomatic=1.0), # The acquisition, relative to the above, by state
    )




class TB(ss.Disease):
    """
    The LSHTM spectrum-of-disease TB model (Schwalb et al. 2025), declared as states and transitions

    It runs as an ODE (where it is `compartmental/lshtm_ode.R`), as a stochastic compartmental
    model, or as agents (with networks, if any), from the same object: see `ss.Sim(solver=...)`.

    States: `susceptible`, `infection` (latent), `non_infectious`, `asymptomatic` and
    `symptomatic` (both infectious), `treatment`, and three states for people who are no
    longer infected, which differ in their risk of reinfection: `cleared` (from latent
    infection), `recovered` (from non-infectious TB), and `treated` (after treatment).
    Optionally, `infection_remote` (with `k_latent > 0`: latent infections become remote,
    after which progression is slower, so it is front-loaded) and `relapsing` (with
    `p_relapse > 0`: treated people who will relapse).

    The transitions are named so that interventions and other conditions can act on them
    (e.g. `ss.modify(tb.latent_asy, multiply=0.5, who=...)`): `transmission` and `reinfect_*`;
    `clearance`, `latent_non`, and `latent_asy` (from latent infection); `recovery` and
    `worsening` (from non-infectious TB); `improvement` and `symptoms` (from asymptomatic TB);
    `remission` and `tb_death` (from symptomatic TB); `diagnosis` (into
    treatment: driven by a delivery, see `tbsim.care()`), `failure`, and `completion`.

    Results, besides those of every state and transition: `infectious` (asymptomatic and
    symptomatic), `active` (also non-infectious), and `prevalence_active` (infectious, as
    a share of everyone).

    With `drugs`, there are strains resistant to them, which people can carry several of
    at once: the model has strains (`tb.strains`, an `ss.Strains` called `resistance`; see
    `tbsim.add_resistance()`), and results can be read by set of strains, e.g.
    `sim.results.tb.symptomatic.select(resistance='DS+RIF')`. Treatment then ends at the
    rate `delta + phi` (`completion`), in success for people with every strain cured.

    With `second_line`, there is a second regimen, for people found to carry a resistant
    strain: the state `second_line`, entered by `diagnosis_second_line` (driven by a
    delivery, see `tbsim.care(dst=...)`) and left by `completion_second_line`;
    `switch_second_line` moves people from first-line treatment to it
    (driven by a delivery, see `tbsim.monitoring()`).

    With `tpt`, TB preventive therapy acts on the infection of people who hold it (a product
    from `tbsim.TPT()`), as well as protecting them: it clears the strains it acts on
    (`tpt_clearance`, to `cleared`; with drugs, someone who also carries a strain it does
    not act on keeps that one), and with drugs, strains it does not clear can become
    resistant to its drugs (`tpt_acquisition_<state>_<strain>_to_<strain>`), mostly in people who have
    active TB that was not found.

    Args:
        pars (dict): parameters, by name (as `kwargs`)
        init (dict): the number (or share) of people starting in each state (default: 5% with a latent infection)
        drugs (list): the drugs that strains can be resistant to (default: none, so one strain)
        max_strains (int): with drugs, the most strains anyone can carry at once
        strains (dict): with drugs, the share of each set of strains among people infected at the start (default: all drug-susceptible), e.g. `dict(DS=0.99, RIF=0.01)`
        tpt (dict): the effects of TB preventive therapy on the infection (see `tbsim.tpt_pars()`), e.g. `dict(clearance=2.0)`, or with drugs, `dict(drugs=['INH'], clearance=2.0, acquisition=0.05)`
        second_line (bool/dict): with drugs, a second-line regimen: True for `tbsim.regimen('second_line_combo')`, or a dict
            of `delta`, `phi`, and `p_relapse` (as `tbsim.regimen()` gives); and optionally `rr_cure` (the chance that it cures
            a strain resistant to each drug, relative to a drug-susceptible strain; default 1, for drugs it does not use) and
            `q_treat` (the chance that a strain it does not cure becomes resistant to each drug; default 0)
        immigrants (dict): with `immigration`, the share of people arriving in each state (default: all susceptible), e.g. `dict(susceptible=0.7, infection=0.3)`
        kwargs (dict): parameters, by name (see `tb_pars()`, and `tbsim.resistance_pars()` with drugs): numbers (rates per year), or `ss.par` objects

    Examples::

        tb = tbsim.TB(beta=12)
        sim = ss.Sim(tb, start=1950, stop=2020, total_pop=1e5)
        sim.run().plot()

        tb = tbsim.TB(drugs=['RIF', 'INH'], fitness=dict(RIF=0.6, INH=0.9), theta=0.5)
        tb = tbsim.TB(drugs=['RIF'], second_line=True) # With tbsim.care(tb, dst=tbsim.DST(tb))
    """
    def __init__(self, pars=None, init=None, init_n=None, drugs=None, max_strains=2, strains=None, immigrants=None, second_line=None, tpt=None, name='tb', **kwargs):
        kwargs = sc.mergedicts(pars, kwargs)
        res_kw = {key:kwargs.pop(key) for key in list(kwargs) if key in tbsim.resistance_pars()}
        if res_kw and not drugs:
            errormsg = f'The parameters {sc.strjoin(res_kw.keys())} are for drug resistance: give drugs= as well'
            raise ValueError(errormsg)
        if second_line and not drugs:
            errormsg = 'second_line= is a regimen for people with a resistant strain: give drugs= as well'
            raise ValueError(errormsg)
        p = ss.pars(tb_pars(), **kwargs)
        second = second_line_pars(second_line) if second_line else None
        tpt = check_tpt(tpt, drugs)
        front = p.k_latent.raw.value > 0 # Latent infections become remote, after which progression is slower
        relapse = p.p_relapse.raw > 0 or (second is not None and second.p_relapse > 0) # Some treated people relapse
        latent = ['infection', 'infection_remote'] if front else ['infection'] # The states of latent infection
        on_tpt = tbsim.TPT.held if tpt else None # Only for the people who hold TPT

        # With drugs, strains resistant to them, whose transitions are added below; the outcome of treatment depends on the strains
        regimens = []
        resistance = None
        if drugs:
            regimens.append(treatment('completion', 'treatment', p.delta, p.phi, p.p_relapse.raw))
            if second: # People found to carry a resistant strain
                regimens.append(treatment('completion_second_line', 'second_line', second.delta, second.phi, second.p_relapse, rr_cure=second.rr_cure, q_treat=second.q_treat))
            bottleneck = ['latent_asy', 'worsening'] + (['remote_asy'] if front else []) # The transitions to asymptomatic TB
            tpt_strains = sc.mergedicts(tpt, dict(on=on_tpt, latent=latent)) if tpt else None
            resistance = tbsim.make_strains(drugs, max_strains=max_strains, split=strains, regimens=regimens, tpt=tpt_strains, alone=bottleneck, **res_kw)

        super().__init__(
            init    = None if init_n else sc.ifelse(init, dict(infection=0.05)),
            init_n  = init_n,
            strains = resistance,
            name    = name,
        )

        # The states
        self.define_states(
            susceptible    = ss.State('Susceptible'),
            infection      = ss.State('Latent infection'),
            cleared        = ss.State('Cleared (from latent)'),
            recovered      = ss.State('Recovered (from non-infectious)'),
            non_infectious = ss.State('Non-infectious TB'),
            asymptomatic   = ss.State('Asymptomatic TB'),
            symptomatic    = ss.State('Symptomatic TB'),
            treatment      = ss.State('On treatment'),
            treated        = ss.State('Completed treatment'),
        )
        if front:
            self.define_states(infection_remote=ss.State('Remote latent infection'))
        if second:
            self.define_states(second_line=ss.State('On second-line treatment'))
        if relapse:
            self.define_states(relapsing=ss.State('Treated, will relapse'))

        # Infection, and reinfection of those no longer infected, at a relative risk that depends on how they stopped being infected
        foi = p.beta*(p.trans_asymp*self.asymptomatic + self.symptomatic) # Exposure to asymptomatic and symptomatic TB (through networks, as agents)
        self.define_transitions(
            transmission = self.susceptible >> self.infection @ foi,
            reinfect_cle = self.cleared     >> self.infection @ p.rr_reinfection_cleared*foi,
            reinfect_rec = self.recovered   >> self.infection @ p.rr_reinfection_rec*foi,
            reinfect_trd = self.treated     >> self.infection @ p.rr_reinfection_treat*foi,
        )

        # Latent infection: cleared, or progressing to TB
        self.define_transitions(
            clearance  = self.infection >> self.cleared        @ p.inf_cle,
            latent_non = self.infection >> self.non_infectious @ p.inf_non,
            latent_asy = self.infection >> self.asymptomatic   @ p.inf_asy,
        )
        if front: # The hazard of progression falls with time since infection: recent, then remote
            inf_non_remote = p.inf_non_remote if p.inf_non_remote is not None else p.inf_non
            self.define_transitions(
                remote           = self.infection        >> self.infection_remote @ p.k_latent,
                reinfect_remote  = self.infection_remote >> self.infection        @ p.rr_reinfection_inf*foi,
                clearance_remote = self.infection_remote >> self.cleared          @ p.inf_cle,
                remote_non       = self.infection_remote >> self.non_infectious   @ inf_non_remote,
                remote_asy       = self.infection_remote >> self.asymptomatic     @ p.inf_asy_remote,
            )

        # The spectrum of TB disease, and death from it
        self.define_transitions(
            recovery    = self.non_infectious >> self.recovered      @ p.non_rec,
            worsening   = self.non_infectious >> self.asymptomatic   @ p.non_asy,
            improvement = self.asymptomatic   >> self.non_infectious @ p.asy_non,
            symptoms    = self.asymptomatic   >> self.symptomatic    @ p.asy_sym,
            remission   = self.symptomatic    >> self.asymptomatic   @ p.sym_asy,
            tb_death    = ss.death(self.symptomatic, rate=p.sym_dead),
        )

        # Treatment: started by diagnosis (driven by a delivery, e.g. tbsim.care()); with drugs, it ends by the strains (below)
        active = [self.non_infectious, self.asymptomatic, self.symptomatic]
        self.define_transitions(diagnosis=ss.Transition(active, self.treatment, rate=p.theta))
        if not drugs:
            self.define_transitions(
                failure    = self.treatment >> self.symptomatic @ p.phi,
                completion = ss.Transition(self.treatment, {self.treated: None, self.relapsing: p.p_relapse} if relapse else self.treated, rate=p.delta),
            )
        if second: # As treatment, at its own rates; both are driven by deliveries (tbsim.care(dst=...), tbsim.monitoring())
            self.define_transitions(
                diagnosis_second_line = ss.Transition(active, self.second_line, rate=ss.peryear(0.0)),
                switch_second_line    = ss.Transition(self.treatment, self.second_line, rate=ss.peryear(0.0)),
            )

        # TPT clears a latent infection, for those who hold it (with drugs, by strain: below)
        if tpt and not drugs:
            for src in latent:
                name_ = 'tpt_clearance' + ('' if src == 'infection' else '_remote')
                self.define_transitions(ss.Transition(getattr(self, src), self.cleared, rate=tpt.clearance, who=on_tpt, name=name_))

        # Relapse after treatment
        if relapse:
            self.define_transitions(relapse=ss.Transition(self.relapsing, self.symptomatic, dur=p.dur_relapse))

        # Protection (or risk) after clearance, recovery, or treatment wanes
        if p.dur_reinfection_protection is not None:
            for src in ['cleared', 'recovered', 'treated']:
                self.define_transitions(ss.Transition(getattr(self, src), self.susceptible, dur=p.dur_reinfection_protection, name=f'waning_{src}'))

        # Demography: births replace every death (with mu=0, none, so ss.Births and ss.Deaths, or demographics=True, can be used instead)
        used = lambda key: isinstance(kwargs.get(key), ss.par) or p[key].raw.value > 0 # Given to be fitted, or not zero
        if used('mu'):
            self.define_transitions(
                births  = ss.birth(self.susceptible, rate=p.mu),
                replace = ss.unborn >> self.susceptible @ p.sym_dead*self.symptomatic.n, # Births that replace TB deaths
                deaths  = ss.death(rate=p.mu),
            )

        # Migration: people arriving from elsewhere, at ages like the population's, into the states given by immigrants=
        if used('immigration'):
            immigrants = sc.ifelse(immigrants, dict(susceptible=1.0))
            total = sum(immigrants.values())
            for key,share in immigrants.items():
                self.define_transitions(ss.immigration(getattr(self, key), n=p.immigration*(share/total), name=f'immigrants_{key}'))
        elif immigrants is not None:
            errormsg = 'immigrants= gives the states that people arrive in: give immigration= as well'
            raise ValueError(errormsg)
        if used('emigration'):
            self.define_transitions(emigrants=ss.emigration(rate=p.emigration))

        # Drug resistance: superinfection, progression and treatment by strain, and resistance acquired
        if drugs:
            tbsim.add_resistance(self, resistance, foi,
                superinfect = dict(infection='rr_superinfection_latent', non_infectious='rr_superinfection_non', asymptomatic='rr_superinfection_asy', symptomatic='rr_superinfection_sym'),
                bottleneck  = bottleneck,
                symptoms    = ['symptoms'],
                clear       = ['clearance', 'recovery'] + (['clearance_remote'] if front else []),
                denovo      = ['latent_non', 'latent_asy'] + (['remote_non', 'remote_asy'] if front else []),
                regimens    = regimens,
                tpt         = tpt_strains,
                **res_kw,
            )

        # Results, besides those of every state and transition
        infectious = self.asymptomatic + self.symptomatic
        self.define_results(
            infectious        = infectious,
            active            = self.non_infectious + infectious,
            prevalence_active = infectious.frac, # As a share of everyone
        )
        self.tpt = tpt
        return


def treatment(name, src, delta, phi, p_relapse=0.0, **kwargs):
    """
    A regimen of treatment, for a model with drug resistance (see `tbsim.add_resistance()`)

    Treatment ends at the rate of completion plus that of failure; people with every strain cured
    go to `treated` (or a share `p_relapse` of them to `relapsing`), and the others to `symptomatic`.

    Args:
        name (str): the name of its transitions
        src (str): the state of being on it
        delta (ss.par): the rate of completion
        phi (ss.par): the rate of failure
        p_relapse (float): the share of those cured who will relapse
        kwargs (dict): `rr_cure` and `q_treat`, if not the model's (see `tbsim.resistance_pars()`)
    """
    d, f = delta.raw.value, phi.raw.value
    cure = d/(d + f) if d + f > 0 else 1.0 # The chance that it cures a drug-susceptible strain
    return dict(name=name, src=src, cured='treated', failed='symptomatic', rate=delta + phi, cure=cure, p_relapse=p_relapse, **kwargs)


def check_tpt(tpt, drugs):
    """ The effects of TPT on the infection (see `tpt_pars()`), with its rate as an `ss.par`; None if there are none """
    if not tpt:
        return None
    unknown = set(tpt) - set(tpt_pars())
    if unknown:
        errormsg = f'tpt= has {sc.strjoin(unknown)}, which are not among {sc.strjoin(tpt_pars().keys())}'
        raise ValueError(errormsg)
    out = sc.mergedicts(tpt_pars(), tpt)
    if not drugs and (out.drugs or out.acquisition):
        errormsg = 'The drugs of TPT, and resistance acquired under it, are for a model with drug resistance: give drugs= as well'
        raise ValueError(errormsg)
    unknown = set(sc.tolist(out.drugs)) - set(drugs or [])
    if unknown:
        errormsg = f'The drugs of TPT, {sc.strjoin(unknown)}, are not among those of the model ({sc.strjoin(drugs)})'
        raise ValueError(errormsg)
    out.drugs = sc.tolist(out.drugs) if out.drugs is not None else None
    out.rr_acquisition = sc.mergedicts(tpt_pars().rr_acquisition, tpt.get('rr_acquisition'))
    clearance = out.clearance
    if not isinstance(clearance, ss.par):
        clearance = ss.par(clearance if isinstance(clearance, ss.TimePar) else ss.peryear(clearance))
    if clearance.name is None:
        clearance.name = 'tpt_clearance_rate'
    out.clearance = clearance
    return out


def second_line_pars(second_line):
    """ The parameters of a second-line regimen (see `tbsim.TB`), with its rates as `ss.par` objects """
    from .interventions.care import regimen # Not at the top, since the interventions import this module
    pars = sc.mergedicts(regimen('second_line_combo') if second_line is True else {}, second_line if isinstance(second_line, dict) else {})
    unknown = set(pars) - {'delta', 'phi', 'p_relapse', 'rr_cure', 'q_treat'}
    if unknown or not {'delta', 'phi'} <= set(pars):
        errormsg = f'second_line= takes True, or a dict of delta, phi, and optionally p_relapse, rr_cure, and q_treat; not {second_line}'
        raise ValueError(errormsg)
    out = sc.objdict(p_relapse=pars.get('p_relapse', 0.0), rr_cure=pars.get('rr_cure', 1.0), q_treat=pars.get('q_treat', 0.0))
    for key in ['delta', 'phi']:
        val = pars[key]
        if not isinstance(val, ss.par):
            val = ss.par(val if isinstance(val, ss.TimePar) else ss.peryear(val))
        if val.name is None:
            val.name = f'{key}_second_line'
        out[key] = val
    return out


def get_tb(sim):
    """ The TB module of a sim: the first disease that is a `tbsim.TB` or a `tbsim.LSHTMResistant` """
    for disease in sim.diseases.values():
        if isinstance(disease, (TB, tbsim.LSHTMResistant)):
            return disease
    errormsg = 'No TB module found in sim.diseases'
    raise ValueError(errormsg)
