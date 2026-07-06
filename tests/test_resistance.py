"""
Tests for the drug-resistance overlay (StrainSpec, StrainCatalog, AgentStrains,
ResistanceConnector, and the strain hooks added to TB).
"""

import numpy as np
import pytest
import starsim as ss
import tbsim
from tbsim import TBS
from tbsim.resistance import (
    ResistanceConnector,
    AgentStrains,
    StrainCatalog,
    StrainSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_default_strains(init_prev_pan=0.05):
    return [
        StrainSpec(uid='pan',   resistance={'INH': 0, 'RIF': 0, 'BDQ': 0},
                   fitness=1.0, init_prev=init_prev_pan),
        StrainSpec(uid='inh_r', resistance={'INH': 1, 'RIF': 0, 'BDQ': 0},
                   fitness=0.95, init_prev=0.0),
        StrainSpec(uid='rif_r', resistance={'INH': 0, 'RIF': 1, 'BDQ': 0},
                   fitness=0.90, init_prev=0.0),
    ]


def make_resistance_sim(n_agents=200, **kwargs):
    """Build a small TB sim with the resistance overlay enabled."""
    strains = kwargs.pop('strains', None) or make_default_strains()
    tb = tbsim.MultiStrainTB(strains=strains, pars=dict(init_prev=ss.bernoulli(0.05)))
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
    sim = ss.Sim(
        n_agents=n_agents,
        diseases=tb,
        networks=net,
        connectors=ResistanceConnector(),
        start=ss.date('2000-01-01'),
        stop=ss.date('2001-12-31'),
        dt=ss.days(14),
        **kwargs,
    )
    sim.pars.verbose = 0
    return sim


# ---------------------------------------------------------------------------
# StrainSpec
# ---------------------------------------------------------------------------

class TestStrainSpec:
    def test_basic_construction(self):
        s = StrainSpec('pan', {'INH': 0, 'RIF': 0})
        assert s.uid == 'pan'
        assert s.resistance == {'INH': 0, 'RIF': 0}
        assert s.fitness == 1.0
        assert s.init_prev == 0.0
        assert s.label == 'pan'

    def test_label_defaults_to_uid(self):
        assert StrainSpec('rif_r', {'RIF': 1}).label == 'rif_r'

    def test_explicit_label(self):
        assert StrainSpec('rif_r', {'RIF': 1}, label='RIF-R').label == 'RIF-R'

    def test_is_resistant_to(self):
        s = StrainSpec('rif_r', {'INH': 0, 'RIF': 1})
        assert s.is_resistant_to('RIF') is True
        assert s.is_resistant_to('INH') is False
        assert s.is_resistant_to('BDQ') is False  # missing drug treated as susceptible

    def test_bool_values_coerced_to_int(self):
        s = StrainSpec('mixed', {'INH': True, 'RIF': False})
        assert s.resistance == {'INH': 1, 'RIF': 0}

    @pytest.mark.parametrize('bad_uid', ['', None, 0])
    def test_rejects_bad_uid(self, bad_uid):
        with pytest.raises(ValueError):
            StrainSpec(bad_uid, {'INH': 0})

    def test_rejects_empty_resistance(self):
        with pytest.raises(ValueError):
            StrainSpec('x', {})

    def test_rejects_non_binary_resistance(self):
        with pytest.raises(ValueError):
            StrainSpec('x', {'INH': 2})

    @pytest.mark.parametrize('bad', [-0.1, 1.5])
    def test_rejects_bad_fitness(self, bad):
        with pytest.raises(ValueError):
            StrainSpec('x', {'INH': 0}, fitness=bad)

    @pytest.mark.parametrize('bad', [-0.1, 1.5])
    def test_rejects_bad_init_prev(self, bad):
        with pytest.raises(ValueError):
            StrainSpec('x', {'INH': 0}, init_prev=bad)


# ---------------------------------------------------------------------------
# StrainCatalog
# ---------------------------------------------------------------------------

class TestStrainCatalog:
    def test_basic_catalog(self):
        reg = StrainCatalog(make_default_strains())
        assert reg.n == 3
        assert reg.drugs == ['INH', 'RIF', 'BDQ']
        assert reg.uids == ['pan', 'inh_r', 'rif_r']

    def test_resistance_matrix_shape(self):
        reg = StrainCatalog(make_default_strains())
        assert reg.resistance.shape == (3, 3)
        # pan is all-zero
        assert (reg.resistance[0] == 0).all()
        # inh_r resists INH only
        assert reg.resistance[1, 0] == 1
        assert (reg.resistance[1, 1:] == 0).all()

    def test_fitness_array(self):
        reg = StrainCatalog(make_default_strains())
        np.testing.assert_allclose(reg.fitness, [1.0, 0.95, 0.90])

    def test_index_lookup(self):
        reg = StrainCatalog(make_default_strains())
        assert reg.index('pan') == 0
        assert reg.index('rif_r') == 2

    def test_index_unknown_raises(self):
        reg = StrainCatalog(make_default_strains())
        with pytest.raises(KeyError):
            reg.index('unknown')

    def test_spec_returns_strain(self):
        reg = StrainCatalog(make_default_strains())
        assert reg.spec('inh_r').uid == 'inh_r'
        assert reg.spec(2).uid == 'rif_r'

    def test_rejects_duplicate_uids(self):
        with pytest.raises(ValueError):
            StrainCatalog([
                StrainSpec('pan', {'INH': 0}),
                StrainSpec('pan', {'INH': 0}),
            ])

    def test_rejects_init_prev_sum_above_one(self):
        with pytest.raises(ValueError):
            StrainCatalog([
                StrainSpec('a', {'INH': 0}, init_prev=0.6),
                StrainSpec('b', {'INH': 0}, init_prev=0.6),
            ])

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            StrainCatalog([])

    def test_explicit_drug_ordering(self):
        reg = StrainCatalog(make_default_strains(), drugs=['BDQ', 'RIF', 'INH'])
        assert reg.drugs == ['BDQ', 'RIF', 'INH']
        # pan still all-zero, inh_r now in column 2
        assert reg.resistance[1, 2] == 1
        assert reg.resistance[1, 0] == 0


# ---------------------------------------------------------------------------
# TB integration: opt-in multi-strain subclass
# ---------------------------------------------------------------------------

class TestTBStrainHook:
    def test_no_agent_strains_when_not_configured(self):
        tb = tbsim.TB()
        assert not hasattr(tb, 'agent_strains')

    def test_agent_strains_created_when_configured(self):
        tb = tbsim.MultiStrainTB(strains=make_default_strains())
        assert isinstance(tb.agent_strains, AgentStrains)
        assert tb.agent_strains.catalog.n == 3

    def test_strain_states_registered_on_tb(self):
        tb = tbsim.MultiStrainTB(strains=make_default_strains())
        for name in tb.agent_strains.names:
            assert name.startswith('carries_')

    def test_default_sim_runs_without_strains(self):
        # Backward compatibility: original behaviour intact when strains is None
        tb = tbsim.TB(pars=dict(init_prev=ss.bernoulli(0.05)))
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
        sim = ss.Sim(n_agents=100, diseases=tb, networks=net,
                     start=ss.date('2000-01-01'), stop=ss.date('2000-12-31'),
                     dt=ss.days(14))
        sim.pars.verbose = 0
        sim.run()
        assert not hasattr(tb, 'agent_strains')
        # No new strain-related state arrays should exist on TB
        assert not any(attr.startswith('carries_') for attr in dir(tb))


# ---------------------------------------------------------------------------
# AgentStrains per-agent operations
# ---------------------------------------------------------------------------

class TestAgentStrains:
    def test_initial_seeding_assigns_strain(self):
        # pan has init_prev=0.5; all initial cases should carry pan only
        strains = [
            StrainSpec('pan', {'INH': 0, 'RIF': 0}, init_prev=1.0, fitness=1.0),
            StrainSpec('rif_r', {'INH': 0, 'RIF': 1}, init_prev=0.0, fitness=0.9),
        ]
        sim = make_resistance_sim(n_agents=200, strains=strains)
        sim.init()
        tb = tbsim.get_tb(sim)
        infected = tb.infected.uids
        assert len(infected) > 0
        pan_carriers = tb.agent_strains.carriers('pan')
        rif_carriers = tb.agent_strains.carriers('rif_r')
        assert set(pan_carriers).issuperset(set(infected))
        assert len(rif_carriers) == 0

    def test_clear_all_removes_all_strains(self):
        sim = make_resistance_sim(n_agents=80)
        sim.init()
        tb = tbsim.get_tb(sim)
        infected = tb.infected.uids
        tb.agent_strains.clear_all(infected)
        for uid in tb.agent_strains.catalog.uids:
            assert not tb.agent_strains.carries(uid, infected).any()

    def _clean_targets(self, tb, count):
        """Return ``count`` UIDs that currently carry no strains."""
        all_uids = ss.uids(np.arange(len(tb.sim.people)))
        counts = tb.agent_strains.n_strains_per_agent(all_uids)
        clean = all_uids[counts == 0]
        assert len(clean) >= count, 'not enough clean targets for the test'
        return clean[:count]

    def test_add_strain_blocks_duplicate(self):
        sim = make_resistance_sim(n_agents=40)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = self._clean_targets(tb, 3)
        tb.agent_strains.add_strain(target, 'pan')
        # Second add should be a no-op (returns no new uids)
        added_again = tb.agent_strains.add_strain(target, 'pan')
        assert len(added_again) == 0
        assert tb.agent_strains.carries('pan', target).all()

    def test_replace_strain(self):
        sim = make_resistance_sim(n_agents=40)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = self._clean_targets(tb, 2)
        tb.agent_strains.add_strain(target, 'pan')
        tb.agent_strains.replace_strain(target, old='pan', new='rif_r')
        assert not tb.agent_strains.carries('pan', target).any()
        assert tb.agent_strains.carries('rif_r', target).all()

    def test_effective_rel_trans_takes_max_fitness(self):
        sim = make_resistance_sim(n_agents=40)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = self._clean_targets(tb, 2)
        tb.agent_strains.add_strain(target, 'pan')         # fitness 1.0
        tb.agent_strains.add_strain(target, 'rif_r')       # fitness 0.9
        eff = tb.agent_strains.effective_rel_trans(target)
        np.testing.assert_allclose(eff, [1.0, 1.0])

    def test_n_strains_per_agent(self):
        sim = make_resistance_sim(n_agents=40)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = self._clean_targets(tb, 3)
        tb.agent_strains.add_strain(target[:2], 'pan')
        tb.agent_strains.add_strain(target[1:], 'rif_r')
        counts = tb.agent_strains.n_strains_per_agent(target)
        np.testing.assert_array_equal(counts, [1, 2, 1])


# ---------------------------------------------------------------------------
# Resistance simulation end-to-end
# ---------------------------------------------------------------------------

class TestResistanceSim:
    def test_short_sim_runs_with_overlay(self):
        sim = make_resistance_sim(n_agents=150)
        sim.run()
        tb = tbsim.get_tb(sim)
        assert tb.agent_strains is not None
        # At least some agents should carry a strain at some point
        any_strain = False
        for uid in tb.agent_strains.catalog.uids:
            if len(tb.agent_strains.carriers(uid)) > 0:
                any_strain = True
                break
        assert any_strain, 'Expected some strain carriers during the sim run.'

    def test_dead_agents_carry_no_strain(self):
        sim = make_resistance_sim(n_agents=200)
        sim.run()
        tb = tbsim.get_tb(sim)
        dead = ss.uids(tb.state == TBS.DEAD)
        if len(dead) == 0:
            pytest.skip('No deaths in this short run; cannot assert')
        counts = tb.agent_strains.n_strains_per_agent(dead)
        assert (counts == 0).all(), 'Dead agents should carry no strains'

    def test_cleared_agents_carry_no_strain(self):
        sim = make_resistance_sim(n_agents=200)
        sim.run()
        tb = tbsim.get_tb(sim)
        cleared = ss.uids(tb.state == TBS.CLEARED)
        if len(cleared) == 0:
            pytest.skip('No CLEARED agents in this short run; cannot assert')
        counts = tb.agent_strains.n_strains_per_agent(cleared)
        # Per the spec, natural clearance removes all strains.
        assert (counts == 0).all(), 'CLEARED agents should carry no strains'

    def test_susceptible_agents_carry_no_strain(self):
        sim = make_resistance_sim(n_agents=200)
        sim.run()
        tb = tbsim.get_tb(sim)
        susc = ss.uids(tb.state == TBS.SUSCEPTIBLE)
        if len(susc) == 0:
            pytest.skip('No SUSCEPTIBLE agents; cannot assert')
        counts = tb.agent_strains.n_strains_per_agent(susc)
        assert (counts == 0).all()

    def test_no_duplicate_strains_per_agent_no_superinfection(self):
        # Without overlay-driven superinfection logic in Phase 1, each carrier
        # array is boolean: an agent carries a strain at most once per strain.
        sim = make_resistance_sim(n_agents=150)
        sim.run()
        tb = tbsim.get_tb(sim)
        # Boolean by construction; just sanity-check via counts <= n_strains.
        all_uids = ss.uids(np.arange(len(tb.sim.people)))
        counts = tb.agent_strains.n_strains_per_agent(all_uids)
        assert counts.max() <= tb.agent_strains.catalog.n


if __name__ == '__main__':
    pytest.main(['-x', '-v', __file__])
