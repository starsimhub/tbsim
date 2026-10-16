"""
Drug resistance: strains of TB that are resistant to some drugs, as a strain dimension of the TB model

A strain is the set of drugs it is resistant to (`DS` for none, `RIF`, `RIF/INH`, ...).
People can carry several strains at once, up to `max_strains`: the levels of the dimension
are the sets of strains people can carry (`DS`, `RIF`, `DS+RIF`, ...). The TB model's states
of infection and disease are split by these levels; everything else about the model is unchanged,
so it runs as an ODE, as a stochastic compartmental model, or as agents, and every intervention
for `tbsim.TB` works with it.

The model is `two_strain_ode.R` (in `tbsim/compartmental`), generalized to any number of drugs:

- Transmission: each strain has its own force of infection. Someone who carries several strains
  is as infectious as the most infectious of them (or as all of them together, with
  `transmission_independent=True`), and passes each on in proportion to its fitness.
- Superinfection: people with a latent infection or TB who are infected with a strain they do not
  carry carry it as well (at a rate relative to the force of infection, by state).
- Progression: of someone who carries several strains, all progress to asymptomatic TB with
  probability `p_multi`, and otherwise one of them; resistance can be acquired on progression
  from a latent infection with one strain (`q_prog`), added to the strain (`prog_resist_mix=True`)
  or replacing it.
- Treatment: each strain carried is cured independently, with the chance of curing a drug-susceptible
  strain times `rr_cure` for each drug it is resistant to; people with every strain cured complete
  treatment, and the others fail, with the strains not cured, each of which can acquire resistance to
  each drug (`q_treat`).
"""
import itertools
import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['resistance_pars', 'Strains', 'add_resistance', 'resistant_share']


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
        transmission_independent  = False, # Whether someone with several strains transmits each at its own fitness (otherwise at the highest, split between them)
        prog_resist_mix           = True,  # Whether resistance acquired on progression is added to the strain (giving two) or replaces it
        prog_select_fitness       = False, # Whether the strain that progresses alone is chosen by fitness (otherwise equally)
    )


class Strains(sc.prettyobj):
    """
    The strains for a set of drugs, and the sets of them that people can carry

    Args:
        drugs (list): the drugs, e.g. `['RIF', 'INH']`
        max_strains (int): the most strains anyone can carry at once

    Attributes:
        strains (list): each strain, as the set of drugs it is resistant to (the first is drug-susceptible)
        sets (list): each set of strains that people can carry: the levels of the dimension, in order
        labels (list): the label of each set, e.g. 'DS', 'RIF', 'DS+RIF'
    """
    def __init__(self, drugs=('RIF',), max_strains=2):
        self.drugs = [drugs] if isinstance(drugs, str) else list(drugs)
        if not self.drugs or len(set(self.drugs)) != len(self.drugs):
            errormsg = f'drugs must be a list of different drugs, not {drugs}'
            raise ValueError(errormsg)
        self.max_strains = max_strains
        self.strains = [frozenset(c) for k in range(len(self.drugs) + 1) for c in itertools.combinations(self.drugs, k)]
        self.sets = [frozenset(c) for k in range(1, max_strains + 1) for c in itertools.combinations(self.strains, k)]
        self.labels = [self.label(s) for s in self.sets]
        self.index = {s:i for i,s in enumerate(self.sets)}
        return

    def name(self, strain):
        """ The name of a strain: the drugs it is resistant to, or DS (drug-susceptible) """
        return '/'.join(d for d in self.drugs if d in strain) or 'DS'

    def label(self, carried):
        """ The label of a set of strains, e.g. 'DS+RIF' """
        return '+'.join(self.name(s) for s in self.strains if s in carried)

    def per_drug(self, value, key):
        """ A value for each drug, from a number or a dict by drug """
        if isinstance(value, dict):
            unknown = set(value) - set(self.drugs)
            if unknown:
                errormsg = f'{key} has values for {sc.strjoin(unknown)}, which are not among the drugs ({sc.strjoin(self.drugs)})'
                raise ValueError(errormsg)
            return {d:value.get(d, 1.0 if key in ['fitness', 'rr_cure'] else 0.0) for d in self.drugs}
        return {d:value for d in self.drugs}

    def safe(self, carried):
        """ A label usable in the name of a transition, e.g. DS_and_RIF """
        return self.label(carried).replace('+', '_and_').replace('/', '_')


def acquire(strain, drugs, q):
    """ The strains that a strain can become by acquiring resistance to each of some drugs independently, with their probabilities: {strain: p} """
    out = {}
    free = [d for d in drugs if d not in strain]
    for k in range(len(free) + 1):
        for gained in itertools.combinations(free, k):
            p = np.prod([q[d] if d in gained else 1 - q[d] for d in free])
            new = strain | frozenset(gained)
            out[new] = out.get(new, 0) + p
    return out


def add_resistance(S, T, p, foi, drugs=('RIF',), max_strains=2, split=None, superinfect=None, bottleneck=None, symptoms=None, clear=None, denovo=None, treatment=None, cure=None, **kwargs):
    """
    Add drug resistance to a TB model that is being built: the transitions it needs, and the stratification to apply once it is built

    The roles of the model's transitions are given by name, so this works for any model with
    the structure of the LSHTM model (`tbsim.TB`, and the reference structure of `tbsim.LSHTMResistant`).

    Args:
        S (dict): the states, by name
        T (dict): the transitions, by name (new ones are added)
        p (dict): the parameters of the model (as `ss.par`)
        foi (Expr): the force of infection
        drugs (list): the drugs
        max_strains (int): the most strains anyone can carry at once
        split (dict): the share of each set of strains among people infected at the start, by label (default: all drug-susceptible)
        superinfect (dict): the states whose people can be superinfected, and the parameter of each (e.g. dict(infection='rr_superinfection_latent'))
        bottleneck (list): the transitions to asymptomatic TB, through which several strains progress together or one alone
        symptoms (list): the transitions from asymptomatic to symptomatic TB (faster with several strains, by `rr_prog_super`)
        clear (list): the transitions of clearance and recovery (faster with several strains, by `rr_clear_super`)
        denovo (list): the transitions of progression from a latent infection, on which resistance can be acquired
        treatment (list): pairs of the transitions that end treatment, (completion, failure): both have the rate at which treatment ends, and are split here by its outcome;
            a third entry, a dict, gives that regimen's own `cure`, `rr_cure`, or `q_treat`
        cure (float): the chance that treatment cures a drug-susceptible strain
        kwargs (dict): the parameters of resistance (see `resistance_pars()`)

    Returns:
        A dict of keywords for `stratify()`
    """
    pars = sc.mergedicts(resistance_pars(), kwargs)
    unknown = set(kwargs) - set(resistance_pars())
    if unknown:
        errormsg = f'Unknown parameter(s) of resistance: {sc.strjoin(unknown)}; they are {sc.strjoin(resistance_pars().keys())}'
        raise ValueError(errormsg)
    st = Strains(drugs, max_strains)
    fit = st.per_drug(pars.fitness, 'fitness')
    rr_cure = st.per_drug(pars.rr_cure, 'rr_cure')
    q_treat = st.per_drug(pars.q_treat, 'q_treat')
    q_prog = st.per_drug(pars.q_prog, 'q_prog')
    fitness = {s:np.prod([fit[d] for d in s]) for s in st.strains}
    n = len(st.sets)
    adjust = sc.objdict()

    def mult(name, level, value):
        adjust.setdefault(name, np.ones(n))
        adjust[name][level] *= value
        return

    def add_move(base, l, m, value):
        """ A transition like another, from people carrying one set of strains, who carry another after it """
        if value <= 0 or l == m:
            return
        tr = T[base]
        name = f'{base}_{st.safe(st.sets[l])}_to_{st.safe(st.sets[m])}'
        T[name] = ss.Transition(tr.src, tr.dst, rate=tr.rate*value, name=name, move=dict(resistance=f'{st.labels[l]} -> {st.labels[m]}'))
        return

    # Transmission: someone with several strains is as infectious as the fittest (or all of them), and passes each on in proportion to its fitness
    rel_trans = np.array([sum(fitness[s] for s in c) if pars.transmission_independent else max(fitness[s] for s in c) for c in st.sets])

    # Superinfection: infected with a strain they do not carry, people carry it as well
    for state,key in (superinfect or {}).items():
        rr = kwargs.get(key)
        if not isinstance(rr, ss.par): # A number, or the default (a new parameter, so that each model has its own)
            default = resistance_pars()[key]
            rr = ss.par(default.raw if rr is None else rr, doc=default.doc, name=key)
        T[f'superinfection_{state}'] = ss.Transition(S[state], S[state], rate=rr*foi, name=f'superinfection_{state}')

    for l,carried in enumerate(st.sets):
        several = len(carried) > 1
        if several: # Progression with several strains: together, or one alone
            for name in bottleneck or []:
                mult(name, l, pars.rr_prog_super*pars.p_multi)
                total = sum(fitness[s] for s in carried)
                for s in carried:
                    h = fitness[s]/total if pars.prog_select_fitness else 1/len(carried)
                    add_move(name, l, st.index[frozenset([s])], pars.rr_prog_super*(1 - pars.p_multi)*h)
            for name in symptoms or []:
                mult(name, l, pars.rr_prog_super)
            for name in clear or []:
                mult(name, l, pars.rr_clear_super)
        else: # Resistance acquired on progression from a latent infection with one strain
            [s] = carried
            for new,prob in acquire(s, st.drugs, q_prog).items():
                if new == s:
                    for name in denovo or []:
                        mult(name, l, prob)
                    continue
                target = frozenset([s, new]) if pars.prog_resist_mix else frozenset([new])
                m = st.index.get(target, st.index[frozenset([new])]) # Replaced, if there is no level for carrying both
                for name in denovo or []:
                    add_move(name, l, m, prob)

        # The outcome of treatment: each strain is cured independently; those not cured can acquire resistance
        for completion,failure,*opts in treatment or []:
            opts = opts[0] if opts else {}
            p_cure = opts.get('cure', cure)
            rr = st.per_drug(opts['rr_cure'], 'rr_cure') if 'rr_cure' in opts else rr_cure
            q = st.per_drug(opts['q_treat'], 'q_treat') if 'q_treat' in opts else q_treat
            c = {s:min(1.0, p_cure*np.prod([rr[d] for d in s])) for s in carried}
            outcomes = {}
            for k in range(1, len(carried) + 1):
                for left in itertools.combinations(sorted(carried, key=st.strains.index), k):
                    p_left = np.prod([1 - c[s] if s in left else c[s] for s in carried])
                    for combo in itertools.product(*[acquire(s, st.drugs, q).items() for s in left]):
                        target = frozenset(new for new,_ in combo)
                        prob = p_left*np.prod([q_ for _,q_ in combo])
                        m = st.index.get(target, l)
                        outcomes[m] = outcomes.get(m, 0) + prob
            mult(completion, l, np.prod(list(c.values())))
            mult(failure, l, outcomes.get(l, 0))
            for m,prob in outcomes.items():
                add_move(failure, l, m, prob)

    split = sc.ifelse(split, {'DS': 1.0})
    shares = np.zeros(n)
    for label,share in split.items():
        if label not in st.labels:
            errormsg = f'The split of strains has "{label}", which is not one of {sc.strjoin(st.labels)}'
            raise ValueError(errormsg)
        shares[st.labels.index(label)] = share
    adjust['rel_trans'] = rel_trans
    strained = [name for name in S if name not in ['susceptible', 'cleared', 'recovered', 'treated']]
    return dict(resistance=st.labels, split=shares, adjust=dict(adjust), states=strained)


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
    strat = [s for s in sim.diseases[disease].graph.strats if s.name == 'resistance'][0]
    def resists(label): # Whether a set of strains, e.g. 'DS+RIF/INH', includes one resistant to the drug
        strains = label.split('+')
        if drug is None:
            return any(strain != 'DS' for strain in strains)
        return any(drug in strain.split('/') for strain in strains)
    resistant = np.array([resists(label) for label in strat.labels])
    counts = sum(res[f'n_{state}'].by('resistance').values for state in states)
    total = counts.sum(axis=1)
    return np.divide(counts[:, resistant].sum(axis=1), total, out=np.zeros(len(total)), where=total>0)
