"""
Drug resistance: strains of TB that are resistant to some drugs, as the strains of the TB model (`ss.Strains`)

A strain is the set of drugs it is resistant to (`DS` for none, `RIF`, `RIF/INH`, ...). People can
carry several strains at once, up to `max_strains`; the sets they carry (`DS`, `RIF`, `DS+RIF`, ...)
are made by `ss.Strains`, and are the levels of the model's `resistance` dimension. Everything else
about the model is unchanged, so it runs as an ODE, as a stochastic compartmental model, or as
agents, and every intervention for `tbsim.TB` works with it.

The model is `two_strain_ode.R` (in `tbsim/compartmental`), generalized to any number of drugs:

- Transmission: each strain has its own force of infection. Someone who carries several strains
  is as infectious as the most infectious of them, and passes on one, in proportion to fitness
  (`transmit='one'`); or each at its own fitness (`transmission_independent=True`, `transmit='each'`).
- Superinfection: people with a latent infection or TB who are infected with a strain they do not
  carry carry it as well (at a rate relative to the force of infection, by state).
- Progression: of someone who carries several strains, all progress to asymptomatic TB with
  probability `p_multi`, and otherwise one of them (`alone=True`); resistance can be acquired on
  progression from a latent infection with one strain (`q_prog`), added to the strain
  (`prog_resist_mix=True`) or replacing it.
- Treatment: each strain carried is cured independently (`per='strain'`), with the chance of
  curing a drug-susceptible strain times `rr_cure` for each drug it is resistant to; people with
  every strain cured complete treatment, and the others fail, with the strains not cured, which
  can acquire resistance to each drug (`q_treat`; for people with several strains left, all of
  them acquire it together).
"""
import itertools
import functools
import operator
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['resistance_pars', 'add_resistance', 'resistant', 'resistant_share']


def resistance_pars():
    """ The parameters of drug resistance (as in two_strain_ode.R): a value for each drug can be a number (the same for every drug) or a dict by drug """
    return sc.objdict(
        fitness                   = 0.575, # The transmission fitness of a strain resistant to each drug (relative to a drug-susceptible strain); a strain's is the product over its drugs
        rr_cure                   = 1/3,   # The chance that treatment cures a strain resistant to each drug, relative to a drug-susceptible strain
        q_treat                   = 0.05,  # The chance that a strain not cured by treatment becomes resistant to each drug
        q_prog                    = 1e-4,  # The chance that a strain becomes resistant to each drug on progression from a latent infection
        rr_superinfection_latent  = ss.par(1.0, doc='Superinfection of a latent infection, relative to the force of infection (sigma_L)'),
        rr_superinfection_non     = ss.par(1.0, doc='... of non-infectious TB (sigma_N)'),
        rr_superinfection_asy     = ss.par(0.0, doc='... of asymptomatic TB'),
        rr_superinfection_sym     = ss.par(0.0, doc='... of symptomatic TB'),
        p_multi                   = 1.0,   # The chance that every strain carried progresses to asymptomatic TB together (otherwise one does)
        rr_prog_super             = 1.0,   # Progression to asymptomatic and symptomatic TB with several strains, relative to one (psi)
        rr_clear_super            = 1.0,   # Clearance and recovery with several strains, relative to one (omega)
        transmission_independent  = False, # Whether someone with several strains transmits each at its own fitness (otherwise one, at the highest)
        prog_resist_mix           = True,  # Whether resistance acquired on progression is added to the strain (giving two) or replaces it
        prog_select_fitness       = False, # Whether the strain that progresses alone is chosen by fitness (otherwise equally)
    )


def drugs_of(strain):
    """ The drugs a strain is resistant to, from its name ('DS', 'RIF', 'RIF/INH') """
    return frozenset(getattr(strain, 'name', strain).split('/')) - {'DS'}


def per_drug(drugs, value, key):
    """ A value for each drug, from a number or a dict by drug """
    if isinstance(value, dict):
        unknown = set(value) - set(drugs)
        if unknown:
            errormsg = f'{key} has values for {sc.strjoin(unknown)}, which are not among the drugs ({sc.strjoin(drugs)})'
            raise ValueError(errormsg)
        return {d:value.get(d, 1.0 if key in ['fitness', 'rr_cure'] else 0.0) for d in drugs}
    return {d:value for d in drugs}


def gains(have, drugs, q):
    """ The resistance a strain resistant to `have` gains, by each drug independently with chance q[d]: {drugs gained: probability}, including none """
    free = [d for d in drugs if d not in have]
    out = {}
    for k in range(len(free) + 1):
        for gained in itertools.combinations(free, k):
            out[frozenset(gained)] = np.prod([q[d] if d in gained else 1 - q[d] for d in free])
    return out


def add_resistance(S, T, p, foi, drugs=('RIF',), max_strains=2, split=None, superinfect=None, bottleneck=None, symptoms=None, clear=None, denovo=None, regimens=None, tpt=None, **kwargs):
    """
    Add drug resistance to a TB model that is being built: its strains, and the transitions they need

    The roles of the model's transitions are given by name, so this works for any model with the
    structure of the LSHTM model (`tbsim.TB`, and the reference structure of `tbsim.LSHTMResistant`).

    Args:
        S (dict): the states, by name
        T (dict): the transitions, by name (changed and added to here)
        p (dict): the parameters of the model (as `ss.par`)
        foi (Expr): the force of infection
        drugs (list): the drugs
        max_strains (int): the most strains anyone can carry at once
        split (dict): the share of each strain, or set of strains carried together, among people infected at the start, e.g. `dict(DS=0.9, RIF=0.1)` (default: all `DS`)
        superinfect (dict): the states whose people can be superinfected, and the parameter of each (e.g. dict(infection='rr_superinfection_latent'))
        bottleneck (list): the transitions to asymptomatic TB, through which several strains progress together or one alone
        symptoms (list): the transitions from asymptomatic to symptomatic TB (faster with several strains, by `rr_prog_super`)
        clear (list): the transitions of clearance and recovery (faster with several strains, by `rr_clear_super`)
        denovo (list): the transitions of progression from a latent infection, on which resistance can be acquired
        regimens (list): the treatments, each a dict: `name` (of its transitions), `src` (the state on treatment), `cured` (where
            people with every strain cured go), `failed` (where the others go), `rate` (at which treatment ends), `cure` (the chance
            that it cures a drug-susceptible strain), and optionally `rr_cure`, `q_treat` (as the parameters), and `relapse` (a pair: the
            state, and the share of those cured who go there instead)
        tpt (dict): the effects of TPT on the infection (see `tbsim.tpt_pars()`), with `on` (who holds it) and `latent` (the states of latent infection)
        kwargs (dict): the parameters of resistance (see `resistance_pars()`)

    Returns:
        The strains (an `ss.Strains`), for `ss.Disease(strains=...)`
    """
    unknown = set(kwargs) - set(resistance_pars())
    if unknown:
        errormsg = f'Unknown parameter(s) of resistance: {sc.strjoin(unknown)}; they are {sc.strjoin(resistance_pars().keys())}'
        raise ValueError(errormsg)
    pars = sc.mergedicts(resistance_pars(), kwargs)
    drugs = [drugs] if isinstance(drugs, str) else list(drugs)
    if not drugs or len(set(drugs)) != len(drugs):
        errormsg = f'drugs must be a list of different drugs, not {drugs}'
        raise ValueError(errormsg)
    have = [frozenset(c) for k in range(len(drugs) + 1) for c in itertools.combinations(drugs, k)] # Each strain, as the drugs it resists
    names = ['/'.join(d for d in drugs if d in h) or 'DS' for h in have]
    index = {h:i for i,h in enumerate(have)}
    fit = per_drug(drugs, pars.fitness, 'fitness')
    fitness = np.array([np.prod([fit[d] for d in h]) for h in have])
    q_prog = per_drug(drugs, pars.q_prog, 'q_prog')

    # The values of the strains: transmissibility, each regimen's chance of cure, and the weights of the strain that progresses alone
    values = dict(rel_trans=fitness)
    for reg in regimens or []:
        rr = per_drug(drugs, reg.get('rr_cure', pars.rr_cure), 'rr_cure')
        values[f'cure_{reg["name"]}'] = [min(1.0, reg['cure']*np.prod([rr[d] for d in h])) for h in have]
    if tpt: # TPT clears the strains susceptible to all of its drugs
        values['tpt_cure'] = [float(not (h & set(tpt.drugs or drugs))) for h in have]
    alone = [name for name in bottleneck or []] if pars.p_multi < 1 else []
    for name in alone:
        values[f'{name}_one'] = fitness if pars.prog_select_fitness else np.ones(len(have))
    split = sc.ifelse(split, {'DS': 1.0})
    split = {label:share/sum(split.values()) for label,share in split.items()}
    strains = ss.Strains(names, max_strains=max_strains, transmit='each' if pars.transmission_independent else 'one', split=split, name='resistance', **values)
    sets = [frozenset(s) for s in strains.sets] # The strains each level carries
    several = lambda s: len(s) > 1
    def by(func):
        """ A value for each level of resistance, from the strains it carries """
        return ss.by(resistance=[func(s) for s in sets])
    def scale(name, func): # A transition, with its rate times a value by level
        tr = T[name]
        T[name] = ss.Transition(tr.src, tr.dst, rate=tr.rate*by(func), name=name, where=tr.where)

    base = dict(T) # The transitions as given, before their rates are changed by level

    # Superinfection: infected with a strain they do not carry, people carry it as well
    for state,key in (superinfect or {}).items():
        rr = kwargs.get(key)
        if not isinstance(rr, ss.par): # A number, or the default (a new parameter, so that each model has its own)
            default = resistance_pars()[key]
            rr = ss.par(default.raw if rr is None else rr, doc=default.doc, name=key)
        T[f'superinfection_{state}'] = ss.Transition(S[state], S[state], rate=rr*foi, name=f'superinfection_{state}')

    # Progression with several strains: faster or slower, and together or one alone
    ps, om = pars.rr_prog_super, pars.rr_clear_super
    for name in bottleneck or []:
        scale(name, lambda s: ps*pars.p_multi if several(s) else 1)
    for name in alone: # One strain progresses alone: the rate of the set is the sum of its strains' weights, so divided by it
        w = values[f'{name}_one']
        tr = base[name]
        T[f'{name}_one'] = ss.Transition(tr.src, tr.dst, rate=tr.rate*ps*(1 - pars.p_multi)*by(lambda s: 1/sum(w[j] for j in s) if several(s) else 0), per='strain', alone=True, name=f'{name}_one')
    for name in symptoms or []:
        scale(name, lambda s: ps if several(s) else 1)
    for name in clear or []:
        scale(name, lambda s: om if several(s) else 1)

    # Resistance acquired on progression from a latent infection with one strain: added to it, or replacing it
    for name in denovo or []:
        tr = base[name]
        scale(name, lambda s: gains(have[min(s)], drugs, q_prog)[frozenset()] if not several(s) else 1)
        for j,h in enumerate(have):
            for gained,prob in gains(h, drugs, q_prog).items():
                new = strains[index[h | gained]]
                if gained and prob > 0:
                    move = strains[j] >> ((strains[j], new) if pars.prog_resist_mix and max_strains > 1 else new)
                    T[f'{name}_{names[j]}_to_{new.name}'] = ss.Transition(tr.src, tr.dst, rate=tr.rate*prob*by(lambda s: float(s == {j})), move=move, name=f'{name}_{names[j]}_to_{new.name}')

    # The outcome of treatment: each strain is cured independently; those not cured can acquire resistance
    for reg in regimens or []:
        add_regimen(S, T, reg, strains, have, names, sets, by, drugs, pars)

    # TPT clears the strains it acts on (leaving the others), and the others can become resistant to its drugs
    if tpt:
        for src in tpt.latent:
            name = 'tpt_clearance' + ('' if src == 'infection' else '_remote')
            T[name] = ss.Transition(S[src], [S.cleared, S[src]], rate=tpt.clearance, p='tpt_cure', per='strain', where=tpt.on, name=name)
        tpt_drugs = tpt.drugs or drugs
        acq = per_drug(drugs, tpt.acquisition if isinstance(tpt.acquisition, dict) else {d:tpt.acquisition for d in tpt_drugs}, 'acquisition')
        for state,rr in tpt.rr_acquisition.items():
            for src in (tpt.latent if state == 'infection' else [state]):
                for j,h in enumerate(have):
                    for gained,prob in gains(h, drugs, acq).items():
                        if gained and prob > 0:
                            name = f'tpt_acquisition_{src}_{names[j]}_to_{names[index[h | gained]]}'
                            T[name] = ss.Transition(S[src], S[src], rate=ss.peryear(rr*prob), move=strains[j] >> strains[index[h | gained]], where=tpt.on, name=name)
    return strains


def add_regimen(S, T, reg, strains, have, names, sets, by, drugs, pars):
    """ The transitions that end a treatment: see `add_resistance()` """
    name, rate, cure = reg['name'], reg['rate'], f'cure_{reg["name"]}'
    src, cured, failed = S[reg['src']], S[reg['cured']], S[reg['failed']]
    relapse, p_relapse = reg.get('relapse') or (None, 0.0)
    q = per_drug(drugs, reg.get('q_treat', pars.q_treat), 'q_treat')
    c = np.array(strains.values[cure])
    none = np.prod([1 - q[d] for d in drugs]) # The chance that no strain acquires resistance

    # No resistance acquired: each strain is cured or not, by itself
    if none > 0:
        T[name] = ss.Transition(src, [cured, failed], rate=rate*none*(1 - p_relapse), p=cure, per='strain', name=name)
        if p_relapse:
            T[f'{name}_relapse'] = ss.Transition(src, [S[relapse], failed], rate=rate*none*p_relapse, p=cure, per='strain', name=f'{name}_relapse')
    if none == 1:
        return

    # Resistance acquired to some drugs by every strain not cured: as a change from each set carried to each set after
    T[f'{name}_acquired_cured'] = ss.Transition(src, [cured, S[relapse]] if p_relapse else cured, rate=rate*(1 - none)*by(lambda s: np.prod(c[list(s)])), p=[None, p_relapse] if p_relapse else None, name=f'{name}_acquired_cured')
    outcomes = {} # {(set before, set after): chance}
    for gained,pg in gains(frozenset(), drugs, q).items():
        if not gained:
            continue
        for s in set(sets):
            for k in range(1, len(s) + 1):
                for left in itertools.combinations(sorted(s), k):
                    after = frozenset(have.index(have[j] | gained) for j in left)
                    p_left = np.prod([1 - c[j] if j in left else c[j] for j in s])
                    outcomes[(s, after)] = outcomes.get((s, after), 0) + pg/(1 - none)*p_left
    T[f'{name}_acquired'] = ss.Transition(src, failed, rate=rate*(1 - none)*by(lambda s: outcomes.get((s, s), 0)), name=f'{name}_acquired') # Nothing to acquire
    for (before,after),prob in outcomes.items():
        if before != after and prob > 0:
            key = f'{name}_acquired_{"+".join(names[j] for j in sorted(before))}_to_{"+".join(names[j] for j in sorted(after))}'
            move = strains.move(tuple(strains[j] for j in sorted(before)), tuple(strains[j] for j in sorted(after)))
            T[key] = ss.Transition(src, failed, rate=rate*(1 - none)*prob*by(lambda s: float(s == before)), move=move, name=key)
    return


def resistant(tb, drugs=None):
    """ The people who carry a strain resistant to any of some drugs (default: any), as a condition, e.g. `where=tbsim.resistant(tb, 'RIF')` """
    if getattr(tb, 'strains', None) is None:
        errormsg = f'The TB model "{tb.name}" has no drug resistance: give it drugs='
        raise ValueError(errormsg)
    known = set().union(*[drugs_of(s) for s in tb.strains])
    drugs = known if drugs is None else {drugs} if isinstance(drugs, str) else set(drugs)
    if drugs - known:
        errormsg = f'The drugs {sc.strjoin(drugs - known)} are not among those of "{tb.name}" ({sc.strjoin(sorted(known))})'
        raise ValueError(errormsg)
    return functools.reduce(operator.or_, [s.cond for s in tb.strains if drugs_of(s) & drugs])


def resistant_share(sim, drug=None, states=('non_infectious', 'asymptomatic', 'symptomatic'), disease='tb'):
    """
    The share of people with active TB (or in other states) who carry a strain resistant to a drug, at each time

    Args:
        sim (`ss.Sim`): a sim that has been run, with a TB model with drugs
        drug (str): the drug (default: any)
        states (list): the states to count
        disease (str): the name of the TB model

    Example::

        sim = tbsim.Sim(tb=tbsim.TB(drugs=['RIF'], theta=0.5)).run()
        plt.plot(sim.timevec, tbsim.resistant_share(sim))
    """
    res = sim.results[disease]
    st = sim.diseases[disease].strains
    labels = [label for label,s in zip(st.labels, st.sets) if any(drugs_of(st[j]) & ({drug} if drug else drugs_of(st[j])) for j in s)]
    counts = sum(np.asarray(res[state].select(resistance=labels)) for state in states)
    total = sum(res[state].values for state in states)
    return np.divide(counts, total, out=np.zeros(len(total)), where=total>0)
