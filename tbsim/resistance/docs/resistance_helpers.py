"""Shared helpers for resistance overlay tests (MultiStrainTB sim builders)."""

import numpy as np
import pytest
import starsim as ss

import tbsim
from tbsim import TBS
from tbsim.resistance import (
    MultiStrainTB,
    ResistanceConnector,
    StrainCatalog,
    StrainSpec,
)


class DetRng:
    """Deterministic drop-in for ``ss.random`` in unit tests."""

    initialized = True

    def __init__(self, values):
        self._vals = np.asarray(values, dtype=float)
        self._i = 0

    def rvs(self, n):
        n = int(n)
        out = self._vals[self._i : self._i + n]
        self._i = (self._i + n) % len(self._vals)
        return out


def uniform_det_rng(n=100_000, seed=42):
    return DetRng(np.random.default_rng(seed).uniform(size=n))


def default_strains(init_prev_pan=0.05, include_mdr=True):
    strains = [
        StrainSpec('pan', {'INH': 0, 'RIF': 0, 'BDQ': 0}, fitness=1.0, init_prev=init_prev_pan),
        StrainSpec('inh_r', {'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95, init_prev=0.0),
        StrainSpec('rif_r', {'INH': 0, 'RIF': 1, 'BDQ': 0}, fitness=0.90, init_prev=0.0),
    ]
    if include_mdr:
        strains.append(StrainSpec('mdr', {'INH': 1, 'RIF': 1, 'BDQ': 0}, fitness=0.85, init_prev=0.0))
    return strains


def default_catalog(strains=None):
    return StrainCatalog(strains if strains is not None else default_strains())


def two_strain_catalog(init_prev_pan=0.0):
    """Pan-susceptible + INH-resistant; INH-only regimen covers only ``pan``."""
    strains = [
        StrainSpec('pan', {'INH': 0, 'RIF': 0}, fitness=1.0, init_prev=init_prev_pan),
        StrainSpec('inh_r', {'INH': 1, 'RIF': 0}, fitness=0.95, init_prev=0.0),
    ]
    return strains, StrainCatalog(strains)


def spec_strains():
    """Four strains mirroring the spec notation table (RIF, BDQ, FQ)."""
    return [
        StrainSpec('x1', {'RIF': 0, 'BDQ': 0, 'FQ': 0}, fitness=1.00, init_prev=0.05),
        StrainSpec('x2', {'RIF': 1, 'BDQ': 0, 'FQ': 0}, fitness=0.90, init_prev=0.0),
        StrainSpec('x3', {'RIF': 1, 'BDQ': 0, 'FQ': 1}, fitness=0.80, init_prev=0.0),
        StrainSpec('x4', {'RIF': 0, 'BDQ': 1, 'FQ': 0}, fitness=0.70, init_prev=0.0),
    ]


def basic_sim(diseases, interventions=None, n_agents=20, connectors=None):
    sim = ss.Sim(
        n_agents=n_agents,
        diseases=diseases,
        networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=30)),
        interventions=interventions or [],
        connectors=connectors,
        start=ss.date('2000-01-01'),
        stop=ss.date('2000-04-01'),
        dt=ss.days(14),
    )
    sim.pars.verbose = 0
    return sim


def make_ss_sim(n_agents=200, strains=None, progression_mode='bottleneck',
                p_multi=1.0, p_random_acquisition=None, interventions=None,
                connectors=None, analyzers=None, alpha_super=0.21,
                alpha_act=None, beta=None, init_prev=0.05,
                start='2000-01-01', stop='2001-12-31', dt_days=14):
    """Build a resistance-enabled ``ss.Sim``."""
    tb_kwargs = dict(
        strains=strains if strains is not None else default_strains(),
        pars=dict(init_prev=ss.bernoulli(init_prev)),
        progression_mode=progression_mode,
        p_multi=p_multi,
        p_random_acquisition=p_random_acquisition,
        alpha_super=alpha_super,
    )
    if alpha_act is not None:
        tb_kwargs['alpha_act'] = alpha_act
    if beta is not None:
        tb_kwargs['pars']['beta'] = beta
    tb = MultiStrainTB(**tb_kwargs)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
    sim_kw = dict(
        n_agents=n_agents, diseases=tb, networks=net,
        start=ss.date(start), stop=ss.date(stop), dt=ss.days(dt_days),
        connectors=connectors if connectors is not None else ResistanceConnector(),
    )
    if interventions is not None:
        sim_kw['interventions'] = interventions
    if analyzers is not None:
        sim_kw['analyzers'] = analyzers
    sim = ss.Sim(**sim_kw)
    sim.pars.verbose = 0
    return sim


def make_example_sim(strains=None, n_agents=600, stop='2020-01-01', rand_seed=1,
                     connectors=None, interventions=None, alpha_super=0.21,
                     alpha_act=None, p_random_acquisition=None,
                     beta=0.5, init_prev=0.15, run=True):
    """Resistance sim for spec-quantitative example tests."""
    tb_kwargs = dict(
        pars=dict(
            init_prev=ss.bernoulli(init_prev),
            beta=ss.permonth(beta),
        ),
        alpha_super=alpha_super,
    )
    if alpha_act is not None:
        tb_kwargs['alpha_act'] = alpha_act
    if p_random_acquisition is not None:
        tb_kwargs['p_random_acquisition'] = p_random_acquisition
    tb = MultiStrainTB(
        strains=strains if strains is not None else spec_strains(),
        **tb_kwargs,
    )
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=8), dur=30))
    sim = ss.Sim(
        n_agents=n_agents,
        diseases=tb,
        networks=net,
        connectors=connectors if connectors is not None else ResistanceConnector(),
        interventions=interventions or [],
        start=ss.date('2000-01-01'),
        stop=ss.date(stop),
        dt=ss.days(14),
        rand_seed=rand_seed,
    )
    sim.pars.verbose = 0
    if run:
        sim.run()
    return sim


def make_scenario_sim(with_resistance=False, rand_seed=1, p_selective=0.0,
                      resistant_fitness=0.95, resistant_init_prev=0.0,
                      include_treatment=False, n_agents=800,
                      stop=ss.date('2006-01-01'),
                      p_random_acquisition=None):
    """Scenario-level sim comparing burden and resistance dynamics."""
    from tbsim.resistance import (
        Regimen,
        StrainAwareTx,
        StrainAwareTxDelivery,
        StrainResults,
    )

    if with_resistance:
        strains = [
            StrainSpec('pan', {'INH': 0, 'RIF': 0}, fitness=1.0, init_prev=0.05),
            StrainSpec('inh_r', {'INH': 1, 'RIF': 0}, fitness=resistant_fitness,
                       init_prev=resistant_init_prev),
        ]
        tb = MultiStrainTB(
            strains=strains,
            pars=dict(init_prev=ss.bernoulli(0.05), beta=ss.permonth(0.22)),
            p_random_acquisition=p_random_acquisition,
        )
        interventions = []
        if include_treatment:
            regimen = Regimen(
                'inh_first_line',
                drugs=['INH'],
                per_drug_efficacy={'INH': 0.05},
            )
            tx_product = StrainAwareTx(
                regimen=regimen,
                catalog=tb._strain_catalog,
                p_selective_acquisition={'INH': p_selective},
                acq_state_modifiers={
                    'infection': 1.0,
                    'non_infectious': 1.0,
                    'asymptomatic': 1.0,
                    'symptomatic': 1.0,
                    'treatment': 0.0,
                    'cleared': 0.0,
                },
                adherence=1.0,
            )

            def _tx_elig(sim):
                tb_local = tbsim.get_tb(sim)
                active = ((tb_local.state == TBS.NON_INFECTIOUS) |
                          (tb_local.state == TBS.ASYMPTOMATIC) |
                          (tb_local.state == TBS.SYMPTOMATIC)).uids
                return active.intersect(sim.people.alive.uids).intersect(tb_local.on_treatment.false())

            interventions = [
                StrainAwareTxDelivery(product=tx_product, name='tx_first', eligibility=_tx_elig),
            ]
        analyzers = [StrainResults()]
    else:
        tb = tbsim.TB(
            pars=dict(init_prev=ss.bernoulli(0.05), beta=ss.permonth(0.22)),
        )
        interventions = []
        analyzers = []

    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
    sim = tbsim.Sim(
        tb_model=tb,
        sim_pars=dict(
            n_agents=n_agents,
            start=ss.date('2000-01-01'),
            stop=stop,
            dt=ss.days(14),
            rand_seed=rand_seed,
        ),
        networks=[net],
        connectors=ResistanceConnector(),
        interventions=interventions,
        analyzers=analyzers,
    )
    sim.pars.verbose = 0
    return sim


def set_agent(tb, profile, state, strain_names):
    """Configure one alive agent for unit testing."""
    alive = tb.sim.people.auids
    if len(alive) == 0:
        pytest.skip('No alive agents in test sim')
    target = alive[:1]
    active_states = {TBS.ASYMPTOMATIC, TBS.SYMPTOMATIC, TBS.NON_INFECTIOUS,
                     TBS.INFECTION, TBS.TREATMENT}
    if state in active_states:
        tb.infected[target] = True
        tb.susceptible[target] = False
    else:
        tb.infected[target] = False
        tb.susceptible[target] = True
    tb.state[target] = state
    profile.clear_all(target)
    for name in strain_names:
        profile.add_strain(target, name)
    return target


def cum_new_inh_r(sim):
    from tbsim.resistance import StrainResults
    analyzer = next(a for a in sim.analyzers.values() if isinstance(a, StrainResults))
    return float(np.asarray(analyzer.results['new_carriers_inh_r'][:]).sum())


def final_active_resistant_share(sim):
    tb = tbsim.get_tb(sim)
    active = ((tb.state == TBS.NON_INFECTIOUS) |
              (tb.state == TBS.ASYMPTOMATIC) |
              (tb.state == TBS.SYMPTOMATIC)).uids
    if len(active) == 0 or tb.agent_strains is None:
        return 0.0
    n_r = len(tb.agent_strains.carriers('inh_r').intersect(active))
    n_pan = len(tb.agent_strains.carriers('pan').intersect(active))
    den = n_r + n_pan
    return n_r / den if den else 0.0
