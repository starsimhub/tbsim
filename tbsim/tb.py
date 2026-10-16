"""
The TB model, declared as states and transitions (Starsim v4), so that it runs as an ODE, as a stochastic compartmental model, or as agents
"""
import numpy as np
import sciris as sc
import starsim as ss
import tbsim

__all__ = ['TB', 'tb_pars', 'make_pars', 'get_tb', 'state_codes']


def tb_pars():
    """ The parameters of the TB model: rates are per year (as in the LSHTM model, lshtm_ode.R) """
    return sc.objdict(
        beta                   = ss.par(ss.peryear(9.0), bounds=[1, 50], doc='Contacts per person per year (with homogeneous mixing; per contact with a network)'),
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


def make_pars(defaults, kwargs):
    """ Each parameter as an `ss.par` named after it (so it can be fitted, or set with `ss.Sim(pars=...)`); a bare number for a rate is per year """
    unknown = set(kwargs) - set(defaults)
    if unknown:
        errormsg = f'Unknown parameter(s) {sc.strjoin(unknown)}; the parameters are {sc.strjoin(defaults.keys())}'
        raise ValueError(errormsg)
    pars = sc.objdict()
    for key,default in defaults.items():
        val = kwargs.get(key)
        if isinstance(val, ss.par):
            if val.name is None: # Named after its key, as the defaults are
                val.name = key
            pars[key] = val
            continue
        raw = default.raw if isinstance(default, ss.par) else default
        if val is None:
            val = raw
        elif isinstance(raw, ss.TimePar) and sc.isnumber(val): # A bare number is in the default's unit (per year, or years)
            val = type(raw)(val)
        if val is None: # An optional parameter that is not used
            pars[key] = None
            continue
        bounds = default.bounds if isinstance(default, ss.par) else None
        doc = default.doc if isinstance(default, ss.par) else None
        pars[key] = ss.par(val, bounds=bounds, doc=doc, name=key)
    return pars


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
    (e.g. `ss.modify(tb.latent_asy, mult=0.5, where=...)`): `transmission` and `reinfect_*`;
    `clearance`, `latent_non`, and `latent_asy` (from latent infection); `recovery` and
    `worsening` (from non-infectious TB); `improvement` and `symptoms` (from asymptomatic TB);
    `remission` and `tb_death` (from symptomatic TB); `diagnosis` (into
    treatment: driven by a delivery, see `tbsim.care()`), `failure`, and `completion`.

    Results, besides those of every state and transition: `infectious` (asymptomatic and
    symptomatic), `active` (also non-infectious), and `prevalence_active` (infectious, as
    a share of everyone).

    With `drugs`, there are strains resistant to them, which people can carry several of
    at once: the model is stratified by `resistance`, whose levels are the sets of strains
    people carry (see `tbsim.add_resistance()`), and results can be read by strain, e.g.
    `sim.results.tb.n_symptomatic.by('resistance')`. Treatment then ends at the rate
    `delta + phi`, in success for people with every strain cured.

    With `second_line`, there is a second regimen, for people found to carry a resistant
    strain: the state `second_line`, entered by `diagnosis_second_line` (driven by a
    delivery, see `tbsim.care(dst=...)`) and left by `completion_second_line` and
    `failure_second_line`.

    Args:
        pars (dict): parameters, by name (as `kwargs`)
        init (dict): the number (or share) of people starting in each state (default: 5% with a latent infection)
        drugs (list): the drugs that strains can be resistant to (default: none, so one strain)
        max_strains (int): with drugs, the most strains anyone can carry at once
        strains (dict): with drugs, the share of each set of strains among people infected at the start (default: all drug-susceptible), e.g. `dict(DS=0.99, RIF=0.01)`
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
    def __init__(self, pars=None, init=None, drugs=None, max_strains=2, strains=None, immigrants=None, second_line=None, name='tb', **kwargs):
        kwargs = sc.mergedicts(pars, kwargs)
        res_kw = {key:kwargs.pop(key) for key in list(kwargs) if key in tbsim.resistance_pars()}
        if res_kw and not drugs:
            errormsg = f'The parameters {sc.strjoin(res_kw.keys())} are for drug resistance: give drugs= as well'
            raise ValueError(errormsg)
        if second_line and not drugs:
            errormsg = 'second_line= is a regimen for people with a resistant strain: give drugs= as well'
            raise ValueError(errormsg)
        p = make_pars(tb_pars(), kwargs)
        second = second_line_pars(second_line) if second_line else None
        S = sc.objdict() # The states, by name
        add_state = lambda name_, label, **kw: S.setdefault(name_, ss.State(name=name_, label=label, **kw))
        add_state('susceptible', 'Susceptible')
        add_state('infection', 'Latent infection')
        add_state('cleared', 'Cleared (from latent)')
        add_state('recovered', 'Recovered (from non-infectious)')
        add_state('non_infectious', 'Non-infectious TB')
        add_state('asymptomatic', 'Asymptomatic TB', infectious=True)
        add_state('symptomatic', 'Symptomatic TB', infectious=True)
        add_state('treatment', 'On treatment')
        add_state('treated', 'Completed treatment')
        front = p.k_latent.raw.value > 0
        relapse = p.p_relapse.raw > 0 or (second is not None and second.p_relapse > 0)
        if front:
            add_state('infection_remote', 'Remote latent infection')
        if second:
            add_state('second_line', 'On second-line treatment')
        if relapse:
            add_state('relapsing', 'Treated, will relapse')

        T = sc.objdict() # The transitions, by name
        def add(name_, src, dst, rate, **kw):
            T[name_] = ss.Transition(S[src], S[dst] if isinstance(dst, str) else [S[d] for d in dst], rate=rate, name=name_, **kw)

        foi = p.beta*(p.trans_asymp*S.asymptomatic + S.symptomatic) # Exposure to asymptomatic and symptomatic TB (through networks, as agents)
        add('transmission', 'susceptible', 'infection', foi)
        add('reinfect_cle', 'cleared', 'infection', p.rr_reinfection_cleared*foi)
        add('reinfect_rec', 'recovered', 'infection', p.rr_reinfection_rec*foi)
        add('reinfect_trd', 'treated', 'infection', p.rr_reinfection_treat*foi)
        add('clearance', 'infection', 'cleared', p.inf_cle)
        add('latent_non', 'infection', 'non_infectious', p.inf_non)
        add('latent_asy', 'infection', 'asymptomatic', p.inf_asy)
        if front: # The hazard of progression falls with time since infection: recent, then remote
            add('remote', 'infection', 'infection_remote', p.k_latent)
            add('reinfect_remote', 'infection_remote', 'infection', p.rr_reinfection_inf*foi)
            add('clearance_remote', 'infection_remote', 'cleared', p.inf_cle)
            add('remote_non', 'infection_remote', 'non_infectious', p.inf_non_remote if p.inf_non_remote is not None else p.inf_non)
            add('remote_asy', 'infection_remote', 'asymptomatic', p.inf_asy_remote)
        add('recovery', 'non_infectious', 'recovered', p.non_rec)
        add('worsening', 'non_infectious', 'asymptomatic', p.non_asy)
        add('improvement', 'asymptomatic', 'non_infectious', p.asy_non)
        add('symptoms', 'asymptomatic', 'symptomatic', p.asy_sym)
        add('remission', 'symptomatic', 'asymptomatic', p.sym_asy)
        T.tb_death = ss.death('symptomatic ->', rate=p.sym_dead, name='tb_death')
        T.diagnosis = ss.Transition([S.non_infectious, S.asymptomatic, S.symptomatic], S.treatment, rate=p.theta, name='diagnosis')
        end, fail = (p.delta + p.phi, p.delta + p.phi) if drugs else (p.delta, p.phi) # With drugs, treatment ends at one rate, and its outcome depends on the strains: see tbsim.resistance()
        add('failure', 'treatment', 'symptomatic', fail)
        def add_completion(name_, src, rate, p_relapse):
            if relapse:
                add(name_, src, ['treated', 'relapsing'], rate, p=[None, p_relapse])
            else:
                add(name_, src, 'treated', rate)
        add_completion('completion', 'treatment', end, p.p_relapse)
        if second: # As treatment, at its own rates
            T.diagnosis_second_line = ss.Transition([S.non_infectious, S.asymptomatic, S.symptomatic], S.second_line, rate=ss.peryear(0.0), name='diagnosis_second_line')
            end_2 = second.delta + second.phi
            add('failure_second_line', 'second_line', 'symptomatic', end_2)
            add_completion('completion_second_line', 'second_line', end_2, second.p_relapse)
        if relapse:
            T.relapse = ss.Transition(S.relapsing, S.symptomatic, dur=p.dur_relapse, name='relapse')
        if p.dur_reinfection_protection is not None: # Protection (or risk) after clearance wanes
            for src in ['cleared', 'recovered', 'treated']:
                T[f'waning_{src}'] = ss.Transition(S[src], S.susceptible, dur=p.dur_reinfection_protection, name=f'waning_{src}')

        # Demography: births replace every death
        T.births  = ss.birth('-> susceptible', rate=p.mu, name='births')
        T.replace = ss.null >> S.susceptible @ (p.sym_dead*S.symptomatic.n) # Births that replace TB deaths
        T.deaths  = ss.death(rate=p.mu, name='deaths')
        migrates = lambda key: isinstance(kwargs.get(key), ss.par) or p[key].raw.value > 0 # Given to be fitted, or not zero
        if migrates('immigration'): # People arriving from elsewhere, at ages like the population's
            immigrants = sc.ifelse(immigrants, dict(susceptible=1.0))
            total = sum(immigrants.values())
            for key,share in immigrants.items():
                T[f'immigrants_{key}'] = ss.immigration(f'-> {key}', rate=p.immigration*(share/total), name=f'immigrants_{key}')
        elif immigrants is not None:
            errormsg = 'immigrants= gives the states that people arrive in: give immigration= as well'
            raise ValueError(errormsg)
        if migrates('emigration'):
            T.emigrants = ss.emigration(rate=p.emigration, name='emigrants')

        if drugs: # Strains resistant to them, as a dimension
            remote = ['remote_asy'] if front else []
            delta, phi = p.delta.raw.value, p.phi.raw.value
            treatment = [('completion', 'failure')]
            if second:
                d2, f2 = second.delta.raw.value, second.phi.raw.value
                treatment.append(('completion_second_line', 'failure_second_line', dict(cure=d2/(d2 + f2) if d2 + f2 > 0 else 1.0, rr_cure=second.rr_cure, q_treat=second.q_treat)))
            strat = tbsim.add_resistance(S, T, p, foi, drugs=drugs, max_strains=max_strains, split=strains,
                superinfect = dict(infection='rr_superinfection_latent', non_infectious='rr_superinfection_non', asymptomatic='rr_superinfection_asy', symptomatic='rr_superinfection_sym'),
                bottleneck  = ['latent_asy', 'worsening'] + remote,
                symptoms    = ['symptoms'],
                clear       = ['clearance', 'recovery'] + (['clearance_remote'] if front else []),
                denovo      = ['latent_non', 'latent_asy'] + (['remote_non'] + remote if front else []),
                treatment   = treatment,
                cure        = delta/(delta + phi) if delta + phi > 0 else 1.0,
                **res_kw)

        everyone = S.susceptible
        for st in S.values()[1:]:
            everyone = everyone | st
        super().__init__(
            *S.values(),
            **T,
            infectious        = S.asymptomatic + S.symptomatic,
            active            = S.non_infectious + S.asymptomatic + S.symptomatic,
            prevalence_active = (S.asymptomatic + S.symptomatic)/everyone,
            init              = sc.ifelse(init, dict(infection=0.05)),
            name              = name,
        )
        if drugs:
            self.stratify(**strat)
        return


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
    """ The TB module of a sim: the first disease that is a `tbsim.TB` (or has a `tb_death` transition, as the resistant models do) """
    for disease in sim.diseases.values():
        if isinstance(disease, TB) or 'tb_death' in getattr(getattr(disease, 'graph', None), 'transitions', {}):
            return disease
    errormsg = 'No TB module found in sim.diseases'
    raise ValueError(errormsg)


def state_codes(tb):
    """
    As agents: the state of each agent as a number, its position in `tb.graph.states` (-1 for none, e.g. the dead)

    Returns an array indexed by UID, e.g. `state_codes(tb)[uids]`; `list(tb.graph.states)` gives the names.
    """
    out = np.full(len(tb.sim.people.uid.raw), -1.0)
    for i,name in enumerate(tb.graph.states):
        out[tb[name].raw] = i
    return out
