"""Resistance wiring tests: analyzers, init_prev, connector, migration."""

import warnings

import numpy as np
import pytest
import starsim as ss

import tbsim
from tbsim import TBS
from tbsim.resistance import (
    DuplicateStrainAnalyzer,
    MultiStrainTB,
    ResistanceConnector,
    StrainResults,
    StrainSpec,
)

from resistance_helpers import basic_sim, default_strains, make_ss_sim, two_strain_catalog


class TestStrainAnalyzers:
    def test_strain_results_runs(self):
        sim = make_ss_sim(n_agents=100, analyzers=[StrainResults()])
        sim.run()
        analyzer = [a for a in sim.analyzers.values() if isinstance(a, StrainResults)][0]
        for uid in ['pan', 'inh_r', 'rif_r', 'mdr']:
            assert f'n_carriers_{uid}' in analyzer.results
            assert f'n_active_{uid}' in analyzer.results
            assert f'new_carriers_{uid}' in analyzer.results
        assert analyzer.results['n_carriers_pan'][:].max() > 0

    def test_duplicate_analyzer_runs(self):
        sim = make_ss_sim(n_agents=100, analyzers=[DuplicateStrainAnalyzer()])
        sim.run()
        analyzer = [a for a in sim.analyzers.values() if isinstance(a, DuplicateStrainAnalyzer)][0]
        vals = analyzer.results['n_duplicate_blocked'][:]
        assert (vals >= 0).all()

    def test_strain_results_requires_strain_overlay(self):
        tb = tbsim.TB(pars=dict(init_prev=ss.bernoulli(0.01)))
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
        sim = ss.Sim(
            n_agents=20, diseases=tb, networks=net,
            start=ss.date('2000-01-01'), stop=ss.date('2000-12-31'),
            dt=ss.days(14), analyzers=[StrainResults()],
        )
        sim.pars.verbose = 0
        with pytest.raises(RuntimeError):
            sim.init()

    def test_strain_results_new_carriers_counts_incident_not_net(self):
        sim = make_ss_sim(n_agents=40, analyzers=[StrainResults()])
        sim.init()
        tb = tbsim.get_tb(sim)
        analyzer = [a for a in sim.analyzers.values() if isinstance(a, StrainResults)][0]

        all_uids = sim.people.auids
        u0, u1 = all_uids[0], all_uids[1]
        tb.agent_strains.clear_all(all_uids)
        tb.agent_strains.add_strain(ss.uids([u1]), 'pan')
        analyzer._prev_carriers['pan'] = ss.uids([u0])

        analyzer.step()
        assert int(analyzer.results['new_carriers_pan'][sim.ti]) == 1


class TestInitPrevFallback:
    """Do not silently assign strain 0 when init_prev weights are all zero."""

    def test_unresolved_source_without_init_prev_leaves_unassigned(self):
        strains = [
            StrainSpec('pan', {'INH': 0, 'RIF': 0}, init_prev=0.0),
            StrainSpec('inh_r', {'INH': 1, 'RIF': 0}, init_prev=0.0),
        ]
        tb = MultiStrainTB(
            strains=strains,
            pars=dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0)),
        )
        sim = basic_sim(tb)
        sim.init()
        tb = tbsim.get_tb(sim)
        uid = sim.people.auids[:1]
        with pytest.warns(RuntimeWarning, match='init_prev'):
            tb.seed_strains(uid, sources=None)
        assert tb.agent_strains.n_strains_per_agent(uid)[0] == 0


class TestMissingResistanceConnector:
    """MultiStrainTB without ResistanceConnector should warn at init."""

    def test_warns_when_connector_absent(self):
        strains, _ = two_strain_catalog()
        tb = MultiStrainTB(strains=strains)
        sim = basic_sim(tb)
        with pytest.warns(RuntimeWarning, match='ResistanceConnector'):
            sim.init()

    def test_no_warn_when_connector_present(self):
        strains, _ = two_strain_catalog()
        tb = MultiStrainTB(strains=strains)
        sim = ss.Sim(
            n_agents=20,
            diseases=tb,
            connectors=ResistanceConnector(),
            networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=30)),
            start=ss.date('2000-01-01'),
            stop=ss.date('2000-04-01'),
            dt=ss.days(14),
        )
        sim.pars.verbose = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sim.init()
        assert not any('ResistanceConnector' in str(w.message) for w in caught)


class TestMigrationStrainSeeding:
    """Immigrants with active TB should receive a strain profile."""

    def test_infected_immigrants_carry_strain(self):
        strains = [
            StrainSpec('pan', {'INH': 0, 'RIF': 0}, init_prev=1.0),
            StrainSpec('inh_r', {'INH': 1, 'RIF': 0}, init_prev=0.0),
        ]
        tb = MultiStrainTB(
            strains=strains,
            pars=dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0)),
        )
        sim = ss.Sim(
            n_agents=100,
            diseases=tb,
            connectors=ResistanceConnector(),
            demographics=tbsim.Migration(pars=dict(
                immigration_rate=ss.freqperyear(500),
                emigration_rate=ss.freqperyear(0),
                tb_state_distribution=dict(ASYMPTOMATIC=1.0),
            )),
            networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=30)),
            start=ss.date('2000-01-01'),
            stop=ss.date('2001-01-01'),
            dt=ss.days(30),
            rand_seed=3,
        )
        sim.pars.verbose = 0
        sim.run()
        tb = tbsim.get_tb(sim)
        mig = sim.demographics['migration']
        immigrants = mig.is_immigrant.uids
        assert len(immigrants) > 0
        active_imm = immigrants[
            np.isin(tb.state[immigrants], [TBS.INFECTION, TBS.NON_INFECTIOUS,
                                           TBS.ASYMPTOMATIC, TBS.SYMPTOMATIC])
        ]
        assert len(active_imm) > 0
        assert tb.agent_strains.n_strains_per_agent(active_imm).min() >= 1

