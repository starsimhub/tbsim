"""
Diagnosis and treatment: tests, treatment regimens, and the care cascade (seeking care, testing, starting treatment)

These are Starsim v4 products and deliveries, so they run under every solver. A treatment
course is part of the TB model (its `treatment` state): a delivery starts it, and the model's
`completion`, `failure`, and `relapse` transitions end it (see `regimen()` for their values).
"""
import sciris as sc
import starsim as ss

__all__ = ['tests', 'test', 'Xpert', 'OralSwab', 'FujiLAM', 'CAD', 'regimens', 'regimen', 'care']


# The probability of a positive result, by TB state, for children (under 15) and adults. FujiLAM's values are for people without HIV; with HIV they are 0.579 and 0.75 (not yet used: see tbsim.comorbidities)
tests = sc.objdict(
    xpert    = dict(symptomatic=[0.73, 0.909], asymptomatic=[0.73, 0.775], non_infectious=[0.73, 0.775]),
    oralswab = dict(symptomatic=[0.25, 0.80],  asymptomatic=[0.25, 0.30],  non_infectious=[0.25, 0.30]),
    fujilam  = dict(symptomatic=[0.51, 0.58],  asymptomatic=[0.51, 0.58],  non_infectious=[0.51, 0.58]),
    cad      = dict(symptomatic=[0.66, 0.66],  asymptomatic=[0.66, 0.66],  non_infectious=[0.66, 0.66]),
)


def test(kind='xpert', by_age=True, specificity=1.0, **kwargs):
    """
    A TB test: an `ss.Test` whose sensitivity depends on the TB state, and on age (children under 15, and adults)

    Args:
        kind (str/dict): one of `tbsim.tests` ('xpert', 'oralswab', 'fujilam', 'cad'), or a dict of the
            probability of a positive result for each state, as [children, adults]
        by_age (bool): whether the sensitivity differs by age; needs the TB model to be stratified by age at 15 (`tb.stratify(age=[0, 15])`). If False, the adult values are used
        specificity (float): the probability that someone without active TB tests negative
        kwargs (dict): passed to `ss.Test`, e.g. `cost=` or `delay=`
    """
    table = tests[kind.lower()] if isinstance(kind, str) else kind
    sensitivity = {state:(ss.by(age=vals) if by_age else vals[-1]) for state,vals in table.items()}
    kwargs.setdefault('name', kind.lower() if isinstance(kind, str) else None)
    return ss.Test(sensitivity=sensitivity, specificity=specificity, **kwargs)


def Xpert(**kwargs):    return test('xpert', **kwargs)
def OralSwab(**kwargs): return test('oralswab', **kwargs)
def FujiLAM(**kwargs):  return test('fujilam', **kwargs)
def CAD(**kwargs):      return test('cad', **kwargs)


# Treatment regimens: the chance of cure among those who adhere, the chance of adhering, the length of the course, and the chance of relapse after a cure
regimens = sc.objdict(
    dots              = dict(cure=0.85, adherence=0.85, duration=ss.days(180), relapse=0.05),
    dots_improved     = dict(cure=0.90, adherence=0.90, duration=ss.days(180), relapse=0.03),
    empiric_treatment = dict(cure=0.70, adherence=0.75, duration=ss.days(90),  relapse=0.10),
    first_line_combo  = dict(cure=0.95, adherence=0.88, duration=ss.days(120), relapse=0.02),
    second_line_combo = dict(cure=0.75, adherence=0.80, duration=ss.days(240), relapse=0.08),
    third_line_combo  = dict(cure=0.60, adherence=0.70, duration=ss.days(360), relapse=0.15),
)


def regimen(kind='dots', **kwargs):
    """
    The parameters of `tbsim.TB` that describe a treatment regimen: `tbsim.TB(**tbsim.regimen('dots'))`

    A course ends in success (completion, to `treated`) with probability `cure × adherence`,
    and otherwise in failure (back to symptomatic TB), after a time of mean `duration`;
    a share `relapse` of successes relapse later.

    Args:
        kind (str): one of `tbsim.regimens`
        kwargs (dict): values to use instead of the regimen's (cure, adherence, duration, relapse)
    """
    r = sc.mergedicts(regimens[kind], kwargs)
    success = r['cure']*r['adherence']
    years = r['duration'].years
    return dict(delta=success/years, phi=(1 - success)/years, p_relapse=r['relapse'])


def care(tb, test=None, seeking=ss.perday(0.1), eligible=None, every=ss.weeks(1), treat=1.0, start=None, stop=None, name='care'):
    """
    The care cascade: people with symptomatic TB seek care, are tested, and those who test positive start treatment

    Seeking care is a round of testing every `every`, offered to each eligible person with
    the chance of seeking care in that time. A positive test starts treatment (the model's
    `diagnosis` transition, into `treatment`) for a share `treat` of those who test positive.

    Args:
        tb (`tbsim.TB`): the TB model
        test (`ss.Test`): the test (default: `tbsim.Xpert()`, by age if the model is stratified by age at 15)
        seeking (rate): the rate at which eligible people seek care
        eligible (Cond): who seeks care (default: people with symptomatic TB)
        every (dur): how often the rounds are; shorter is closer to seeking care at any time
        treat (float): the share of those who test positive who start treatment
        start, stop: the period of the cascade
        name (str): the name of the delivery (its follow-up is `<name>_positive`)

    Returns:
        An `ss.screen` delivery
    """
    if test is None:
        ages = [strat.edges for strat in tb.graph.strats if strat.kind == 'age']
        by_age = len(ages) == 1 and ages[0] is not None and list(ages[0]) == [0, 15]
        test = Xpert(by_age=by_age)
    eligible = sc.ifelse(eligible, tb.symptomatic)
    p_seek = float(seeking.to_prob(every)) if isinstance(seeking, ss.TimePar) else float(seeking) # The chance of seeking care in each round
    start_treatment = ss.treat(ss.Treatment(efficacy=1.0, transition='diagnosis', name=f'{name}_tx'), coverage=treat)
    return ss.screen(test, eligible=eligible, coverage=p_seek, interval=every, start=start, stop=stop, on_positive=start_treatment, name=name)
