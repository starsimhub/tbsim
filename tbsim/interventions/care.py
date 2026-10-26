"""
Diagnosis and treatment: tests, treatment regimens, and the care cascade (seeking care, testing, starting treatment)

These are Starsim v4 products and deliveries, so they run under every solver. A treatment
course is part of the TB model (its `treatment` state): a delivery starts it, and the model's
`completion`, `failure`, and `relapse` transitions end it (see `regimen()` for their values).
With drug resistance, a test for it (`DST()`) can start a second-line regimen instead (see
`tbsim.TB(second_line=...)`).
"""
import numpy as np
import sciris as sc
import starsim as ss
import starsim.library as ssl
from .prevention import TPT
from ..resistance import resistant

__all__ = ['tests', 'test', 'Xpert', 'OralSwab', 'FujiLAM', 'CAD', 'DST', 'regimens', 'regimen', 'treated_within', 'start_treatment', 'care', 'monitoring', 'contact_tracing']


# The probability of a positive result, by TB state, for children (under 15) and adults. FujiLAM's values are for people without HIV; with HIV they are 0.579 and 0.75 (not yet used: see tbsim.comorbidities)
tests = sc.objdict(
    xpert    = dict(symptomatic=[0.73, 0.909], asymptomatic=[0.73, 0.775], non_infectious=[0.73, 0.775]),
    oralswab = dict(symptomatic=[0.25, 0.80],  asymptomatic=[0.25, 0.30],  non_infectious=[0.25, 0.30]),
    fujilam  = dict(symptomatic=[0.51, 0.58],  asymptomatic=[0.51, 0.58],  non_infectious=[0.51, 0.58]),
    cad      = dict(symptomatic=[0.66, 0.66],  asymptomatic=[0.66, 0.66],  non_infectious=[0.66, 0.66]),
)


def test(kind='xpert', by_age=True, specificity=1.0, **kwargs):
    """
    A TB test: an `ss.Diagnostic` whose sensitivity depends on the TB state, and on age (children under 15, and adults)

    Args:
        kind (str/dict): one of `tbsim.tests` ('xpert', 'oralswab', 'fujilam', 'cad'), or a dict of the
            probability of a positive result for each state, as [children, adults]
        by_age (bool): whether the sensitivity differs by age; needs the TB model to be stratified by age at 15 (`tb.stratify(age=[0, 15])`). If False, the adult values are used
        specificity (float): the probability that someone without active TB tests negative
        kwargs (dict): passed to `ss.Diagnostic`, e.g. `cost=` or `delay=`
    """
    table = tests[kind.lower()] if isinstance(kind, str) else kind
    sensitivity = {state:(ss.by(age=vals) if by_age else vals[-1]) for state,vals in table.items()}
    kwargs.setdefault('name', kind.lower() if isinstance(kind, str) else None)
    return ss.Diagnostic(sensitivity=sensitivity, specificity=specificity, **kwargs)


def Xpert(**kwargs):    return test('xpert', **kwargs)
def OralSwab(**kwargs): return test('oralswab', **kwargs)
def FujiLAM(**kwargs):  return test('fujilam', **kwargs)
def CAD(**kwargs):      return test('cad', **kwargs)


def DST(tb, drugs=None, sensitivity=1.0, specificity=1.0, **kwargs):
    """
    A drug-susceptibility test: an `ss.Diagnostic` that is positive for people who carry a strain resistant to any of some drugs

    Use it in the care cascade, so that people found resistant start the second-line regimen
    (`tbsim.care(tb, dst=tbsim.DST(tb))`), or with `ss.screen()`.

    Args:
        tb (`tbsim.TB`): the TB model, with drugs
        drugs (list): the drugs the test detects resistance to (default: all of the model's), e.g. 'RIF' for Xpert MTB/RIF
        sensitivity (float): the probability that someone with a resistant strain tests positive
        specificity (float): the probability that someone without one tests negative
        kwargs (dict): passed to `ss.Diagnostic`, e.g. `cost=` or `delay=`
    """
    detects = resistant(tb, drugs)
    kwargs.setdefault('name', 'dst')
    return ss.Diagnostic(sensitivity=sensitivity, specificity=specificity, detects=detects, **kwargs)


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


def care(tb, test=None, seeking=ss.perday(0.1), where=None, every=ss.weeks(1), treat=1.0, dst=None, retreatment=None, start=None, stop=None, name='care'):
    """
    The care cascade: people with symptomatic TB seek care, are tested, and those who test positive start treatment

    With `dst`, those who test positive are also tested for resistance: those found resistant
    start the second-line regimen (`diagnosis_second_line`; see `tbsim.TB(second_line=...)`),
    and the others first-line treatment. With `retreatment` as well, only people treated
    within that time before are tested for resistance (agents only; see `start_treatment()`).

    Seeking care is a round of testing every `every`, offered to each eligible person with
    the chance of seeking care in that time. A positive test starts treatment (the model's
    `diagnosis` transition, into `treatment`) for a share `treat` of those who test positive.

    Args:
        tb (`tbsim.TB`): the TB model
        test (`ss.Diagnostic`): the test (default: `tbsim.Xpert()`, by age if the model is stratified by age at 15)
        seeking (rate): the rate at which eligible people seek care (or a number: the chance of seeking care in each round)
        where (Cond): who seeks care (default: people with symptomatic TB)
        every (dur): how often the rounds are; shorter is closer to seeking care at any time
        treat (float): the share of those who test positive who start treatment
        dst (`ss.Diagnostic`): a test for resistance (see `tbsim.DST()`), for a model with a second-line regimen
        retreatment (dur): with `dst`, test only people who started treatment within this time before (agents only), e.g. `ss.years(2)`
        start, stop: the period of the cascade
        name (str): the name of the delivery (its follow-up is `<name>_positive`)

    Returns:
        A list of deliveries: the `ss.screen`, and its follow-ups (see `start_treatment()`)
    """
    if test is None:
        by_age = 'age' in tb.strats and list(tb.strats.age.edges) == [0, 15]
        test = Xpert(by_age=by_age)
    screen = ss.screen(test, where=sc.ifelse(where, tb.symptomatic), coverage=seeking, interval=every, start=start, stop=stop, name=name) # A rate of seeking care is a chance in each round
    return [screen, *start_treatment(tb, screen.positive, treat, dst, retreatment, name=name)]


def treated_within(tb, within):
    """
    Whether people started treatment (first- or second-line) within some time before: a condition, for `where=` or `detects=` (agents only)

    Args:
        tb (`tbsim.TB`): the TB model
        within (dur): the time, e.g. `ss.years(2)`
    """
    recent = tb.diagnosis.since <= within
    if 'diagnosis_second_line' in tb.transitions:
        recent = recent | (tb.diagnosis_second_line.since <= within)
    return recent


def start_treatment(tb, who, treat=1.0, dst=None, retreatment=None, name='care'):
    """
    The follow-up of a positive test for TB: a share `treat` start treatment (the model's `diagnosis`)

    With `dst`, they are tested for resistance first, and those found resistant start the
    second-line regimen (`diagnosis_second_line`) instead. With `retreatment` as well, only
    people who started treatment within that time before are tested for resistance, which
    their records show (`<name>_history`, a test that is always right); the others start
    first-line treatment. This needs agents, which keep when each person was last treated.

    Args:
        tb (`tbsim.TB`): the TB model
        treat (float): the share who start treatment
        dst (`ss.Diagnostic`): a test for resistance (see `tbsim.DST()`)
        retreatment (dur): with `dst`, test only people who started treatment within this time before (agents only), e.g. `ss.years(2)`
        name (str): the start of the names of the products (`<name>_tx` and `<name>_tx_second_line`) and of the test for resistance (`<name>_dst`, whose follow-ups are `<name>_dst_positive` and `<name>_dst_negative`)

    Returns:
        A list of deliveries: an `ss.treat`, or with `dst`, an `ss.screen` and the treatments that follow it
    """
    first_line = lambda who: ss.treat(ss.Treatment(moves='diagnosis', name=f'{name}_tx'), who=who, coverage=treat, name=f'{name}_positive')
    if dst is None:
        if retreatment is not None:
            errormsg = 'retreatment= says who is tested for resistance: give dst= as well'
            raise ValueError(errormsg)
        return [first_line(who)]
    if 'diagnosis_second_line' not in tb.transitions:
        errormsg = f'dst= sends people found resistant to the second-line regimen, which "{tb.name}" does not have: give it second_line='
        raise ValueError(errormsg)
    def resistance(who):
        screen = ss.screen(dst, who=who, name=f'{name}_dst')
        second_line = ss.treat(ss.Treatment(moves='diagnosis_second_line', name=f'{name}_tx_second_line'), who=screen.positive, coverage=treat, name=f'{name}_dst_positive')
        first = ss.treat(ss.Treatment(moves='diagnosis', name=f'{name}_tx_dst_negative'), who=screen.negative, coverage=treat, name=f'{name}_dst_negative')
        return [screen, second_line, first]
    if retreatment is None:
        return resistance(who)
    history = ss.screen(ss.Diagnostic(detects=treated_within(tb, retreatment), name=f'{name}_records'), who=who, name=f'{name}_history')
    return [history, *resistance(history.positive), first_line(history.negative)]


def monitoring(tb, dst=None, every=ss.months(2), coverage=1.0, switch=1.0, start=None, stop=None, name='monitoring'):
    """
    Treatment monitoring: people on first-line treatment are tested for resistance at intervals, and those found resistant switch to the second-line regimen

    The switch is the model's `switch_second_line` transition (see `tbsim.TB(second_line=...)`).

    Args:
        tb (`tbsim.TB`): the TB model, with a second-line regimen
        dst (`ss.Diagnostic`): the test for resistance (default: `tbsim.DST(tb)`)
        every (dur): how often people on treatment are tested
        coverage (float): the share of people on treatment tested in each round
        switch (float): the share of those found resistant who switch
        start, stop: the period of monitoring
        name (str): the name of the delivery (the switch is `<name>_positive`)

    Example::

        tb = tbsim.TB(drugs=['RIF'], second_line=True)
        interventions = [tbsim.care(tb), tbsim.monitoring(tb)]
    """
    if 'switch_second_line' not in tb.transitions:
        errormsg = f'Monitoring switches people found resistant to the second-line regimen, which "{tb.name}" does not have: give it second_line='
        raise ValueError(errormsg)
    dst = sc.ifelse(dst, DST(tb, name=f'{name}_dst'))
    screen = ss.screen(dst, where=tb.treatment, coverage=coverage, interval=every, start=start, stop=stop, name=name)
    switching = ss.treat(ss.Treatment(moves='switch_second_line', name=f'{name}_tx'), who=screen.positive, coverage=switch, name=f'{name}_positive')
    return [screen, switching]


def contact_tracing(tb, network='householdnet', test=None, tpt=None, coverage=1.0, treat=1.0, dst=None, on=None, name='tracing'):
    """
    Household contact tracing: the contacts of people who start treatment are tested; those who test positive start treatment, and those with a latent infection who test negative receive TPT

    This is agent-based only, since it needs to know who each person's contacts are (it uses
    `ss.screen(trace=...)`).

    Args:
        tb (`tbsim.TB`): the TB model
        network (str/Network): the network whose contacts are traced (default: `ssl.HouseholdNet`'s)
        test (`ss.Diagnostic`): the test of contacts (default: `tbsim.Xpert(by_age=False)`)
        tpt (`ss.Treatment`): the preventive therapy for contacts with a latent infection who test negative, delivered by `<name>_prevention` (default: `tbsim.TPT(tb=tb)`); False for none
        coverage (float): the share of contacts who are traced and tested
        treat (float): the share of those who test positive who start treatment
        dst (`ss.Diagnostic`): a test for resistance for those who test positive, as for `tbsim.care()`
        on (Transition/Cond): whose contacts are traced (default: people starting treatment)
        name (str): the name of the delivery

    Example::

        tb = tbsim.TB()
        household = ssl.HouseholdNet(dhs_data='default', dynamic=False)
        interventions = [tbsim.care(tb), tbsim.contact_tracing(tb)]
        sim = ss.Sim(tb, networks=household, interventions=interventions, solver='abm')
    """
    second = 'second_line' in tb.states
    if on is None:
        on = tb.diagnosis | tb.diagnosis_second_line if second else tb.diagnosis
    test = sc.ifelse(test, Xpert(by_age=False, name=f'{name}_xpert'))
    tpt = TPT(tb=tb, name=f'{name}_tpt') if tpt is None else tpt
    on_treatment = tb.treatment | tb.second_line if second else tb.treatment
    latent = tb.infection | tb.infection_remote if 'infection_remote' in tb.states else tb.infection
    screen = ss.screen(test, on=on, trace=network, coverage=coverage, where=~on_treatment, name=name)
    prevent = [ss.treat(tpt, who=screen.negative, where=latent, name=f'{name}_prevention')] if tpt else []
    return [screen, *start_treatment(tb, screen.positive, treat, dst, name=name), *prevent]
