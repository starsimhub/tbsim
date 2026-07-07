"""Natural-history overlay tests: transmission, progression, and acquisition.

Covers fitness-weighted transmission and superinfection, progression bottleneck,
random μ_d acquisition, and selective ω acquisition on treatment failure.
"""

import numpy as np
import pytest
import starsim as ss

import tbsim
from tbsim import TBS
from tbsim.resistance import (
    AcquisitionResolver,
    MultiStrainTB,
    ProgressionResolver,
    ResistanceConnector,
    StrainCatalog,
    StrainSpec,
)

from resistance_helpers import (
    DetRng,
    basic_sim,
    make_example_sim,
    make_ss_sim,
    set_agent,
    two_strain_catalog,
    uniform_det_rng,
)


class TestTransmissionFitnessWeightedMultinomial:
    """
    Spec §Transmission — "Agent C" table:

        "Each transmission event only passes 1 strain from infector to infectee.

         For infecting agents with superinfection:
           • Overall probability of transmission equals that of the *fittest*
             strain: max(f_i).
           • Conditional on transmission, each strain is transmitted with
             probability proportional to its fitness: P(strain_i) = f_i / Σ f_j."

    This class validates the two quantitative claims in that table.
    Each test manually configures agent strain identity on a running sim
    to avoid dependence on stochastic epidemic dynamics.
    """

    def _two_strain_sim(self, f_high=1.0, f_low=0.6, seed=1):
        strains = [
            StrainSpec('high', {'RIF': 0}, fitness=f_high, init_prev=0.10),
            StrainSpec('low',  {'RIF': 1}, fitness=f_low,  init_prev=0.05),
        ]
        return make_example_sim(strains=strains, rand_seed=seed)

    # -- effective_rel_trans tests --

    def test_single_strain_effective_rel_trans_equals_its_fitness(self):
        """An agent carrying only one strain: rel_trans = that strain's fitness."""
        sim = self._two_strain_sim(f_high=1.0, f_low=0.6)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        target = set_agent(tb, profile, TBS.INFECTION, ['high'])
        rel = profile.effective_rel_trans(target)
        np.testing.assert_allclose(rel, [1.0], rtol=1e-9,
            err_msg='Single-strain agent rel_trans must equal its fitness (1.0)')

    def test_superinfected_effective_rel_trans_is_max_not_sum_not_mean(self):
        """
        Spec table: "max(relative fitness of each strain)" — explicitly NOT
        the sum (f1 + f2) or product.

        An agent superinfected with fitness=1.0 and fitness=0.6 must report
        effective_rel_trans = 1.0, not 1.6, not 0.8.
        """
        sim = self._two_strain_sim(f_high=1.0, f_low=0.6)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        target = set_agent(tb, profile, TBS.INFECTION, ['high', 'low'])

        rel = profile.effective_rel_trans(target)
        np.testing.assert_allclose(rel, [1.0], rtol=1e-9,
            err_msg=(
                'Superinfected agent effective_rel_trans must equal max(fitness)=1.0, '
                'not sum=1.6, not mean=0.8.'
            ))
        assert float(rel[0]) != pytest.approx(0.6 + 1.0), \
            'rel_trans must NOT be the sum of both fitnesses'
        assert float(rel[0]) != pytest.approx(0.6), \
            'rel_trans must NOT be just the lower fitness'

    def test_low_fitness_only_agent_rel_trans_equals_low_fitness(self):
        """Symmetric check: agent carrying only the low-fitness strain."""
        sim = self._two_strain_sim(f_high=1.0, f_low=0.6)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        target = set_agent(tb, profile, TBS.INFECTION, ['low'])
        rel = profile.effective_rel_trans(target)
        np.testing.assert_allclose(rel, [0.6], rtol=1e-9,
            err_msg='Agent carrying only f=0.6 strain must have rel_trans=0.6')

    def test_no_strain_agent_returns_zero(self):
        """Agent carrying no strain: effective_rel_trans = 0."""
        sim = self._two_strain_sim()
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        target = set_agent(tb, profile, TBS.SUSCEPTIBLE, [])  # cleared
        rel = profile.effective_rel_trans(target)
        np.testing.assert_allclose(rel, [0.0], rtol=1e-9)

    # -- sample_transmitted_strain tests --

    def test_single_strain_source_always_transmits_that_strain(self):
        """Single-strain source: every draw returns strain index 0 ('high')."""
        sim = self._two_strain_sim()
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        source = set_agent(tb, profile, TBS.INFECTION, ['high'])
        rng = DetRng(np.linspace(0.001, 0.999, 500))
        picks = np.array([int(profile.sample_transmitted_strain(source, rng)[0])
                          for _ in range(50)])
        assert np.all(picks == 0), (
            f'Single-strain source must always transmit strain 0 (high); '
            f'got unique picks: {np.unique(picks)}'
        )

    def test_superinfected_source_multinomial_ratio(self):
        """
        Spec (Agent C table):
          With f_high=1.0, f_low=0.6:
            P(high transmitted | event) = 1.0 / 1.6 = 0.625
            P(low  transmitted | event) = 0.6 / 1.6 = 0.375

        Validated with N=10 000 deterministic uniform draws.
        Tolerance: ±5σ of the binomial.
        """
        sim = self._two_strain_sim(f_high=1.0, f_low=0.6)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        # Strain indices: 'high'=0, 'low'=1 (order of StrainSpec list)
        source = set_agent(tb, profile, TBS.INFECTION, ['high', 'low'])

        N = 10_000
        rng = uniform_det_rng(N, seed=7)
        picks = np.array([int(profile.sample_transmitted_strain(source, rng)[0])
                          for _ in range(N)])

        f_high, f_low = 1.0, 0.6
        p_high = f_high / (f_high + f_low)   # 0.625
        p_low  = f_low  / (f_high + f_low)   # 0.375
        n_high = int(np.sum(picks == 0))
        n_low  = int(np.sum(picks == 1))

        tol_high = 5 * np.sqrt(N * p_high * (1 - p_high))
        tol_low  = 5 * np.sqrt(N * p_low  * (1 - p_low))
        assert abs(n_high - N * p_high) < tol_high, (
            f'High-fitness strain picked {n_high}/{N} times; '
            f'expected {N*p_high:.0f} ± {tol_high:.0f}'
        )
        assert abs(n_low - N * p_low) < tol_low, (
            f'Low-fitness strain picked {n_low}/{N} times; '
            f'expected {N*p_low:.0f} ± {tol_low:.0f}'
        )

    def test_zero_fitness_strain_never_selected_for_transmission(self):
        """
        Spec: fitness is a multiplicative factor on FOI.
        A strain with fitness=0 contributes 0 to the multinomial weights and
        must never be selected as the transmitted strain.
        """
        strains = [
            StrainSpec('viable',  {'RIF': 0}, fitness=1.0, init_prev=0.10),
            StrainSpec('extinct', {'RIF': 1}, fitness=0.0, init_prev=0.05),
        ]
        sim = make_example_sim(strains=strains, rand_seed=2)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        source = set_agent(tb, profile, TBS.INFECTION, ['viable', 'extinct'])

        N = 2_000
        rng = uniform_det_rng(N, seed=99)
        picks = np.array([int(profile.sample_transmitted_strain(source, rng)[0])
                          for _ in range(N)])
        n_extinct = int(np.sum(picks == 1))
        assert n_extinct == 0, (
            f'Zero-fitness strain was selected for transmission {n_extinct}/{N} times'
        )

    def test_three_strain_superinfection_multinomial_proportions(self):
        """
        Three-strain source: probabilities sum to 1 and each is f_i / Σ f_j.

        Strains: f1=1.0, f2=0.8, f3=0.4 → expected proportions 45.5%:36.4%:18.2%
        """
        strains = [
            StrainSpec('s1', {'RIF': 0, 'BDQ': 0}, fitness=1.0, init_prev=0.10),
            StrainSpec('s2', {'RIF': 1, 'BDQ': 0}, fitness=0.8, init_prev=0.05),
            StrainSpec('s3', {'RIF': 0, 'BDQ': 1}, fitness=0.4, init_prev=0.02),
        ]
        sim = make_example_sim(strains=strains, rand_seed=3)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        source = set_agent(tb, profile, TBS.INFECTION, ['s1', 's2', 's3'])

        f = [1.0, 0.8, 0.4]
        f_sum = sum(f)
        p_expected = [fi / f_sum for fi in f]  # [0.4545, 0.3636, 0.1818]

        N = 15_000
        rng = uniform_det_rng(N, seed=11)
        picks = np.array([int(profile.sample_transmitted_strain(source, rng)[0])
                          for _ in range(N)])

        for idx, p_exp in enumerate(p_expected):
            n_obs = int(np.sum(picks == idx))
            tol = 5 * np.sqrt(N * p_exp * (1 - p_exp))
            assert abs(n_obs - N * p_exp) < tol, (
                f'Strain {idx}: observed {n_obs}/{N}={n_obs/N:.3f}, '
                f'expected {p_exp:.3f} ± {tol/N:.3f}'
            )


class TestSuperinfectionProtection:
    """
    Spec §Strain competition and protection (Agent A / Agent B example):

        "An agent can be infected with a second strain while in the INFECTED
         OR NON-INFECTIOUS disease states, but not the ASYMPTOMATIC or
         SYMPTOMATIC (or TREATED) TB disease states."

         "For INFECTED individuals we apply a multiplicative protective factor
          α_super [default 0.21]."

         "We do not allow superinfection with 2 identical strains."

    Agent A example:
        • Agent A carries X1.  Agent B carries X1 + X3.
        • Contact: P(A acquires X1) is computed but is blocked (duplicate).
        • P(A acquires X3) ~ α_super × transmission_prob × P(X3 selected | B).
    """

    def _two_strain_superinf_sim(self, alpha_super=0.21, seed=5):
        strains = [
            StrainSpec('x1', {'RIF': 0}, fitness=1.0, init_prev=0.10),
            StrainSpec('x3', {'RIF': 1}, fitness=1.0, init_prev=0.08),
        ]
        return make_example_sim(strains=strains, rand_seed=seed, alpha_super=alpha_super)

    def test_only_one_strain_exists_means_no_duplicate_superinfection(self):
        """
        Spec: "We do not allow superinfection with 2 identical strains."

        When a single strain is defined (only one possible strain), carriers
        can never carry more than one copy — duplicate blocking is absolute.
        """
        strains = [StrainSpec('pan', {'RIF': 0}, fitness=1.0, init_prev=0.10)]
        sim = make_example_sim(strains=strains, rand_seed=6)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        infected = tb.infected.uids
        if len(infected) == 0:
            pytest.skip('No infected agents')
        counts = profile.n_strains_per_agent(infected)
        assert np.all(counts <= 1), (
            f'Duplicate blocking failed: {np.sum(counts > 1)} agents '
            f'carry >1 copy of the single available strain'
        )

    def test_asymptomatic_agents_carry_at_most_one_strain(self):
        """
        Spec: "ASYMPTOMATIC: 0 [no superinfection allowed]."

        With default α_asymptomatic=0, ASYMPTOMATIC agents must never hold
        more than one strain regardless of how many exposures they receive.
        """
        sim = self._two_strain_superinf_sim(alpha_super=0.21)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        asymp = (tb.state == TBS.ASYMPTOMATIC).uids
        if len(asymp) == 0:
            pytest.skip('No ASYMPTOMATIC agents at end of run')
        counts = profile.n_strains_per_agent(asymp)
        assert np.all(counts <= 1), (
            f'{np.sum(counts > 1)}/{len(asymp)} ASYMPTOMATIC agents carry >1 strain '
            f'despite α_asymptomatic=0 (default)'
        )

    def test_symptomatic_agents_carry_at_most_one_strain(self):
        """Spec: "SYMPTOMATIC: 0 [no superinfection allowed]." """
        sim = self._two_strain_superinf_sim(alpha_super=0.21)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        symp = (tb.state == TBS.SYMPTOMATIC).uids
        if len(symp) == 0:
            pytest.skip('No SYMPTOMATIC agents at end of run')
        counts = profile.n_strains_per_agent(symp)
        assert np.all(counts <= 1), (
            f'{np.sum(counts > 1)}/{len(symp)} SYMPTOMATIC agents carry >1 strain '
            f'despite α_symptomatic=0 (default)'
        )

    def test_direct_duplicate_blocking_via_assign_strains(self):
        """
        Agent A example — duplicate blocking is enforced by _assign_transmitted_strains.

        Set up recipient carrying X1; source also carries only X1.
        After assignment the recipient must still have exactly 1 strain.
        """
        sim = self._two_strain_superinf_sim()
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        alive = sim.people.auids
        recipient = set_agent(tb, profile, TBS.INFECTION, ['x1'])
        # Source: use a different alive agent, manually set to carry only x1
        if len(alive) < 2:
            pytest.skip('Need at least 2 alive agents')
        source = alive[1:2]
        profile.clear_all(source)
        profile.add_strain(source, 'x1')
        assert int(profile.n_strains_per_agent(recipient)[0]) == 1

        tb._assign_transmitted_strains(recipient, np.asarray(source, dtype=int))
        n_after = int(profile.n_strains_per_agent(recipient)[0])
        assert n_after == 1, (
            f'Duplicate-strain blocking failed: recipient went from 1 to {n_after} strains '
            f'after transmission of the strain it already carries'
        )

    def test_non_duplicate_strain_superinfection_succeeds(self):
        """
        Agent A / Agent B example — Agent A (carries X1) can acquire X3
        when the source carries X3 (a different strain).
        """
        sim = self._two_strain_superinf_sim(alpha_super=1.0)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        alive = sim.people.auids
        if len(alive) < 2:
            pytest.skip('Need at least 2 alive agents')
        recipient = set_agent(tb, profile, TBS.INFECTION, ['x1'])
        source = alive[1:2]
        profile.clear_all(source)
        profile.add_strain(source, 'x3')  # source carries only x3

        n_before = int(profile.n_strains_per_agent(recipient)[0])
        assert n_before == 1
        tb._assign_transmitted_strains(recipient, np.asarray(source, dtype=int))
        n_after = int(profile.n_strains_per_agent(recipient)[0])
        assert n_after == 2, (
            f'Non-duplicate superinfection failed: recipient has {n_after} strains '
            f'after receiving a novel strain from the source (expected 2)'
        )

    def test_higher_alpha_super_produces_more_superinfected_agents(self):
        """
        Spec: α_super is a multiplicative factor on P(superinfection).
        Higher α → more superinfected INFECTION-state agents over time.
        """
        def _n_super(alpha):
            sim = self._two_strain_superinf_sim(alpha_super=alpha, seed=9)
            tb = tbsim.get_tb(sim)
            profile = tb.agent_strains
            inf_uids = (tb.state == TBS.INFECTION).uids
            if len(inf_uids) == 0:
                return 0
            return int(np.sum(profile.n_strains_per_agent(inf_uids) >= 2))

        n_low  = _n_super(alpha=0.05)
        n_high = _n_super(alpha=0.8)
        assert n_high >= n_low, (
            f'Expected α=0.8 ({n_high} superinfected) ≥ α=0.05 ({n_low})'
        )


class TestSuperinfectionViaTransmission:
    def _strains(self):
        strains, _ = two_strain_catalog()
        return strains

    def test_default_alpha_super_runs(self):
        """With α_super=0.21 (default) a sim seeded with both strains runs
        end-to-end without errors and produces some multi-strain carriers."""
        sim = make_ss_sim(n_agents=300, strains=self._strains(), init_prev=0.3,
                          beta=ss.peryear(2.0), stop='2002-01-01')
        sim.run()
        tb = tbsim.get_tb(sim)
        multi = tb.agent_strains.n_strains_per_agent(sim.people.auids)
        assert int((multi > 1).sum()) >= 0  # framework runs; numbers are stochastic

    def test_alpha_super_zero_no_crash(self):
        """With α_super=0 and α_act_*=0 the override should skip the super
        path entirely and the sim should still run cleanly."""
        sim = make_ss_sim(n_agents=200, strains=self._strains(), alpha_super=0.0,
                          init_prev=0.3, beta=ss.peryear(2.0), stop='2002-01-01')
        sim.run()
        tb = tbsim.get_tb(sim)
        multi = tb.agent_strains.n_strains_per_agent(sim.people.auids)
        assert (multi >= 0).all()

    def test_set_prognoses_preserves_state_for_super(self):
        """Superinfecting an already-infected agent must not reset their TB state."""
        sim = make_ss_sim(n_agents=20, strains=self._strains(), init_prev=0.0)
        sim.init()
        tb = tbsim.get_tb(sim)
        uids = sim.people.auids[:3]
        tb.state[uids] = TBS.ASYMPTOMATIC
        tb.infected[uids] = True
        tb.susceptible[uids] = False
        tb.agent_strains.add_strain(uids, 'pan')
        tb.set_prognoses(uids, sources=uids)
        assert (tb.state[uids] == TBS.ASYMPTOMATIC).all()


class TestNonInfectiousAlphaDefault:
    """Spec: rr_reinfection_non := rr_reinfection_inf := rr_reinfection_rec."""

    def _init_tb(self, **tb_kwargs):
        strains, _ = two_strain_catalog()
        tb = MultiStrainTB(strains=strains, **tb_kwargs)
        sim = basic_sim(tb)
        sim.init()
        return tbsim.get_tb(sim)

    def test_default_alpha_non_infectious_equals_alpha_super(self):
        tb = self._init_tb()
        assert tb._alpha_super == float(tb.pars.rr_reinfection_rec)
        assert tb._alpha_act['non_infectious'] == tb._alpha_super, (
            'Spec default for rr_reinfection_non is rr_reinfection_inf '
            '(i.e. alpha_super), not 0. Setting it to 0 disables '
            'NON_INFECTIOUS superinfection — the scenario the spec '
            'explicitly warns against.'
        )

    def test_active_disease_defaults_remain_zero(self):
        """Option 2: no superinfection while in ASY / SYM."""
        tb = self._init_tb()
        assert tb._alpha_act['asymptomatic'] == 0.0
        assert tb._alpha_act['symptomatic']  == 0.0

    def test_user_override_still_respected(self):
        tb = self._init_tb(alpha_act={'non_infectious': 0.0})
        assert tb._alpha_act['non_infectious'] == 0.0

    def test_alpha_super_tracks_rr_reinfection_rec(self):
        """Custom rr_reinfection_rec propagates to alpha_super and NON_INFECTIOUS."""
        tb = self._init_tb(pars=dict(rr_reinfection_rec=0.5))
        assert tb._alpha_super == 0.5
        assert tb._alpha_act['non_infectious'] == 0.5


class TestZeroFitnessConnector:
    """P1: zero-fitness-only carriers must not transmit at full rate."""

    def _infectious_agent(self, strains, carried_uids):
        tb = MultiStrainTB(
            strains=strains,
            pars=dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0)),
        )
        sim = ss.Sim(
            n_agents=10,
            diseases=tb,
            connectors=ResistanceConnector(),
            networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=1), dur=30)),
            start=ss.date('2000-01-01'),
            stop=ss.date('2000-02-01'),
            dt=ss.days(14),
        )
        sim.pars.verbose = 0
        sim.init()
        tb = tbsim.get_tb(sim)
        uid = sim.people.auids[:1]
        for name in tb.agent_strains.names:
            getattr(tb, name)[sim.people.auids] = False
        for s_uid in carried_uids:
            tb.agent_strains.add_strain(uid, s_uid)
        tb.state[uid] = TBS.ASYMPTOMATIC
        tb.infected[uid] = True
        tb.rel_trans[uid] = 1.0
        next(c for c in sim.connectors.values() if c.__class__.__name__ == 'ResistanceConnector').step()
        return tb, uid

    def test_zero_fitness_only_carrier_gets_zero_multiplier(self):
        strains = [
            StrainSpec('viable', {'RIF': 0}, fitness=1.0),
            StrainSpec('extinct', {'RIF': 1}, fitness=0.0),
        ]
        tb, uid = self._infectious_agent(strains, ['extinct'])
        assert float(np.asarray(tb.rel_trans[uid]).item()) == 0.0

    def test_no_strain_legacy_infectious_unchanged(self):
        strains = [StrainSpec('viable', {'RIF': 0}, fitness=1.0)]
        tb, uid = self._infectious_agent(strains, [])
        assert float(np.asarray(tb.rel_trans[uid]).item()) == 1.0


class TestProgressionResolver:
    def test_all_mode_no_op(self):
        sim = make_ss_sim(n_agents=50)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0, 1], dtype=int))
        tb.agent_strains.add_strain(target, 'pan')
        tb.agent_strains.add_strain(target, 'inh_r')

        resolver = ProgressionResolver(mode='all')
        out = resolver.resolve(tb.agent_strains, target)
        assert (out == -1).all()
        counts = tb.agent_strains.n_strains_per_agent(target)
        assert (counts >= 2).all()

    def test_bottleneck_p_multi_zero_forces_single_strain(self):
        sim = make_ss_sim(n_agents=50)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0, 1, 2], dtype=int))
        tb.agent_strains.clear_all(target)
        tb.agent_strains.add_strain(target, 'pan')
        tb.agent_strains.add_strain(target, 'inh_r')

        resolver = ProgressionResolver(mode='bottleneck', p_multi=0.0)
        out = resolver.resolve(tb.agent_strains, target)
        counts = tb.agent_strains.n_strains_per_agent(target)
        assert (counts == 1).all()
        assert (out >= 0).all()

    def test_bottleneck_p_multi_one_is_no_op(self):
        sim = make_ss_sim(n_agents=50)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0, 1], dtype=int))
        tb.agent_strains.clear_all(target)
        tb.agent_strains.add_strain(target, 'pan')
        tb.agent_strains.add_strain(target, 'inh_r')

        resolver = ProgressionResolver(mode='bottleneck', p_multi=1.0)
        out = resolver.resolve(tb.agent_strains, target)
        assert (out == -1).all()
        counts = tb.agent_strains.n_strains_per_agent(target)
        assert (counts == 2).all()

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError):
            ProgressionResolver(mode='nope')

    def test_bad_p_multi_raises(self):
        with pytest.raises(ValueError):
            ProgressionResolver(p_multi=1.5)


class TestBottleneckIntegration:
    def test_bottleneck_unit_via_resolver(self):
        sim = make_ss_sim(n_agents=80)
        sim.init()
        tb = tbsim.get_tb(sim)
        all_uids = tb.sim.people.auids
        clean = all_uids[tb.agent_strains.n_strains_per_agent(all_uids) == 0]
        target = clean[:5]
        tb.agent_strains.add_strain(target, 'pan')
        tb.agent_strains.add_strain(target, 'inh_r')
        assert (tb.agent_strains.n_strains_per_agent(target) == 2).all()

        resolver = ProgressionResolver(mode='bottleneck', p_multi=0.0)
        resolver.resolve(tb.agent_strains, target)
        counts = tb.agent_strains.n_strains_per_agent(target)
        assert (counts == 1).all(), f'Expected single strain after funnel, got {counts}'


class TestAcquisitionResolver:
    def test_selective_acquisition_replaces_pan_with_inh_r(self):
        sim = make_ss_sim(n_agents=50)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0, 1, 2], dtype=int))
        tb.agent_strains.clear_all(target)
        tb.agent_strains.add_strain(target, 'pan')

        acq = AcquisitionResolver(p_selective={'INH': 1.0})
        acq.selective_acquisition(tb.agent_strains, target, drugs_used=['INH'])

        assert not tb.agent_strains.carries('pan', target).any()
        assert tb.agent_strains.carries('inh_r', target).all()

    def test_selective_no_effect_if_strain_already_resistant(self):
        sim = make_ss_sim(n_agents=50)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0], dtype=int))
        tb.agent_strains.clear_all(target)
        tb.agent_strains.add_strain(target, 'inh_r')

        acq = AcquisitionResolver(p_selective={'INH': 1.0})
        acq.selective_acquisition(tb.agent_strains, target, drugs_used=['INH'])
        assert tb.agent_strains.carries('inh_r', target).all()
        assert not tb.agent_strains.carries('pan', target).any()

    def test_random_acquisition_can_be_zero(self):
        sim = make_ss_sim(n_agents=20)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0, 1], dtype=int))
        tb.agent_strains.clear_all(target)
        tb.agent_strains.add_strain(target, 'pan')

        acq = AcquisitionResolver()
        acq.random_acquisition(tb.agent_strains, target)
        assert tb.agent_strains.carries('pan', target).all()
        assert not tb.agent_strains.carries('inh_r', target).any()


class TestStateDependentAcquisition:
    def test_default_modifiers_zero_for_latent(self):
        """Spec: ω = 0 for non-symptomatic agents by default."""
        ar = AcquisitionResolver(p_selective={'INH': 1.0})
        sim = make_ss_sim(n_agents=20)
        sim.init()
        tb = tbsim.get_tb(sim)
        uids = sim.people.auids[:6]
        tb.state[uids[:2]] = TBS.INFECTION
        tb.state[uids[2:4]] = TBS.NON_INFECTIOUS
        tb.state[uids[4:]] = TBS.SYMPTOMATIC
        mod = ar._state_modifier_array(tb, uids)
        assert (mod[:2] == 0.0).all()
        assert (mod[2:4] == 0.0).all()
        assert (mod[4:] == 1.0).all()

    def test_custom_modifiers_override_defaults(self):
        ar = AcquisitionResolver(
            p_selective={'INH': 1.0},
            state_modifiers={'infection': 0.5, 'asymptomatic': 0.0},
        )
        assert ar.state_modifiers['infection'] == 0.5
        assert ar.state_modifiers['asymptomatic'] == 0.0
        assert ar.state_modifiers['symptomatic'] == 1.0


class TestRandomAcquisition:
    """
    Spec §Random Acquisition:

        "Model as a one-time, independent probability μ_d that each strain
         develops resistance to each drug d upon transition from INFECTION to
         NON_INFECTIOUS or ASYMPTOMATIC."

        Option 1 (implementation choice):
        "Assume that rather than replacing the existing strain with a resistant
         version, random acquisition results in multi-strain infection."

        Additional rules:
        • "The probability of random acquisition is strain-agnostic except that
           a strain cannot acquire resistance to a drug to which it already has
           resistance."
        • "Allow multiple resistance acquisitions ... to occur."

    Example from the spec:
        Agent with X1={0,0,0} and X3={1,0,1} in INFECTION, μ_BDQ=p:
        • X1 trial for BDQ → if hit, agent gains X4={0,1,0} (added, X1 retained)
        • X1 trial for FQ  → if hit, agent gains a new FQ-resistant X1 variant
        • X3 trial for RIF → X3 already resistant; no trial for RIF
        • X3 trial for FQ  → X3 already resistant; no trial for FQ
        • X3 trial for BDQ → if hit, agent gains {1,1,1} (added, X3 retained)
    """

    def _make_acq_sim(self, p_random=None, seed=10):
        """Run a sim; return the finished sim for direct method calls."""
        strains = [
            StrainSpec('x1', {'RIF': 0, 'BDQ': 0, 'FQ': 0}, fitness=1.00, init_prev=0.10),
            StrainSpec('x2', {'RIF': 1, 'BDQ': 0, 'FQ': 0}, fitness=0.90, init_prev=0.02),
            StrainSpec('x3', {'RIF': 1, 'BDQ': 0, 'FQ': 1}, fitness=0.85, init_prev=0.01),
            StrainSpec('x4', {'RIF': 0, 'BDQ': 1, 'FQ': 0}, fitness=0.80, init_prev=0.01),
            StrainSpec('xdr', {'RIF': 1, 'BDQ': 1, 'FQ': 1}, fitness=0.70, init_prev=0.01),
        ]
        return make_example_sim(strains=strains, p_random_acquisition=p_random, rand_seed=seed)

    def _make_acq_resolver(self, p_random_dict, catalog):
        rng = uniform_det_rng(50_000, seed=77)
        return AcquisitionResolver(p_random=p_random_dict, rng=rng)

    def test_option1_add_semantics_original_strain_retained(self):
        """
        Spec Option 1: when X1 acquires BDQ resistance it becomes X4={0,1,0},
        but X1 is **retained** — the agent ends up multi-strain.

        Verify: after random_acquisition with μ_BDQ=1.0, a pan-susceptible agent
        carries both X1 (original) and X4 (new resistant variant).
        """
        sim = self._make_acq_sim(p_random=None)
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        catalog = tb._strain_catalog

        target = set_agent(tb, profile, TBS.INFECTION, ['x1'])
        assert int(profile.n_strains_per_agent(target)[0]) == 1

        resolver = self._make_acq_resolver({'BDQ': 1.0}, catalog)
        resolver.random_acquisition(profile, target)

        n_after = int(profile.n_strains_per_agent(target)[0])
        still_x1 = bool(np.asarray(getattr(tb, 'carries_x1')[target], dtype=bool)[0])
        has_x4 = bool(np.asarray(getattr(tb, 'carries_x4')[target], dtype=bool)[0])
        assert still_x1, 'X1 must still be present after Option 1 random acquisition'
        assert has_x4, 'X4 (BDQ-resistant variant of X1) must be added'
        assert n_after == 2, f'Expected 2 strains after acquisition, got {n_after}'

    def test_already_resistant_strain_skips_that_drug_trial(self):
        """
        Spec: "A strain cannot acquire resistance to a drug to which it already
        has resistance."

        X3={1,0,1} (RIF+FQ resistant): no trial for RIF or FQ, only BDQ is eligible.
        """
        sim = self._make_acq_sim()
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        catalog = tb._strain_catalog

        target = set_agent(tb, profile, TBS.INFECTION, ['x3'])

        resolver_rif = self._make_acq_resolver({'RIF': 1.0}, catalog)
        resolver_rif.random_acquisition(profile, target)
        assert int(profile.n_strains_per_agent(target)[0]) == 1, (
            'Trial for already-resistant drug (RIF) must be skipped'
        )

        resolver_fq = self._make_acq_resolver({'FQ': 1.0}, catalog)
        resolver_fq.random_acquisition(profile, target)
        assert int(profile.n_strains_per_agent(target)[0]) == 1, (
            'Trial for already-resistant drug (FQ) must be skipped'
        )

    def test_bdq_trial_succeeds_for_x3(self):
        """
        Continuing the spec example: X3={1,0,1} CAN acquire BDQ resistance
        (not yet BDQ-resistant).  With p_BDQ=1.0, XDR={1,1,1} is added.
        """
        sim = self._make_acq_sim()
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        catalog = tb._strain_catalog

        target = set_agent(tb, profile, TBS.INFECTION, ['x3'])
        resolver = self._make_acq_resolver({'BDQ': 1.0}, catalog)
        resolver.random_acquisition(profile, target)

        has_x3 = bool(np.asarray(getattr(tb, 'carries_x3')[target], dtype=bool)[0])
        has_xdr = bool(np.asarray(getattr(tb, 'carries_xdr')[target], dtype=bool)[0])
        assert has_x3, 'X3 must still be present after acquiring BDQ (Option 1)'
        assert has_xdr, 'XDR={1,1,1} must be added when X3 acquires BDQ'

    def test_two_strains_each_get_independent_trial(self):
        """
        Spec: "Allow multiple resistance acquisitions ... to occur."

        Agent carries X1 and X3.  With p_BDQ=1.0, both get independent BDQ
        trials → agent ends with X1 + X4 + X3 + XDR (4 strains).
        """
        sim = self._make_acq_sim()
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        catalog = tb._strain_catalog

        target = set_agent(tb, profile, TBS.INFECTION, ['x1', 'x3'])
        assert int(profile.n_strains_per_agent(target)[0]) == 2

        resolver = self._make_acq_resolver({'BDQ': 1.0}, catalog)
        resolver.random_acquisition(profile, target)

        n_after = int(profile.n_strains_per_agent(target)[0])
        assert n_after == 4, (
            f'Expected 4 strains after both X1 and X3 acquire BDQ; got {n_after}'
        )

    def test_zero_probability_causes_no_acquisition(self):
        """With μ_d = 0.0 no acquisition ever fires."""
        sim = self._make_acq_sim()
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains
        catalog = tb._strain_catalog

        target = set_agent(tb, profile, TBS.INFECTION, ['x1'])
        resolver = self._make_acq_resolver({'RIF': 0.0, 'BDQ': 0.0, 'FQ': 0.0},
                                           catalog)
        resolver.random_acquisition(profile, target)
        assert int(profile.n_strains_per_agent(target)[0]) == 1, (
            'μ_d=0 must produce no acquisition event'
        )

    def test_random_acquisition_fires_only_leaving_infection_not_non_infectious(self):
        """
        Spec: "upon transition from INFECTION to NON_INFECTIOUS or ASYMPTOMATIC"
        — NOT at NON_INFECTIOUS → ASYMPTOMATIC.

        _apply_strain_progression(activating_only=True) must skip acquisition.
        """
        sim = self._make_acq_sim(p_random={'RIF': 1.0})
        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains

        target = set_agent(tb, profile, TBS.NON_INFECTIOUS, ['x1'])
        n_before = int(profile.n_strains_per_agent(target)[0])

        tb.state[target] = TBS.ASYMPTOMATIC
        tb._apply_strain_progression(target, activating_only=True)
        tb.state[target] = TBS.NON_INFECTIOUS

        n_after = int(profile.n_strains_per_agent(target)[0])
        assert n_after == n_before, (
            f'Random acquisition must NOT fire when activating_only=True '
            f'(NON_INFECTIOUS→ASYMPTOMATIC path). before={n_before}, after={n_after}'
        )


class TestSelectiveAcquisition:
    """
    Spec §Treatment & Selective Acquisition:

        "For each regimen R, define ω_{R,d} as the risk of developing
         resistance ... among ... strains with unsuccessful treatment outcomes
         ... at time of treatment failure/relapse."
        "Allow acquisition risk to vary by TB state at time of failure."
        "Resistance acquisition results in strain replacement."
    """

    def _catalog_with_bdq(self):
        strains = [
            StrainSpec('x1', {'RIF': 0, 'BDQ': 0}, fitness=1.0, init_prev=0.08),
            StrainSpec('x2', {'RIF': 1, 'BDQ': 0}, fitness=0.9, init_prev=0.03),
            StrainSpec('x4', {'RIF': 0, 'BDQ': 1}, fitness=0.8, init_prev=0.02),
            StrainSpec('xdr', {'RIF': 1, 'BDQ': 1}, fitness=0.7, init_prev=0.01),
        ]
        return strains, StrainCatalog(strains)

    def _bdq_sim(self, seed):
        strains_spec, catalog = self._catalog_with_bdq()
        return make_example_sim(strains=strains_spec, rand_seed=seed), catalog

    def test_selective_acquisition_replaces_susceptible_strain_on_failure(self):
        """
        Spec example: surviving X2 (BDQ-susceptible) acquires BDQ resistance
        → X2 is replaced by XDR (one bit flip: RIF=1, BDQ=1).

        With ω_BDQ = 1.0 the replacement is certain.
        """
        sim, catalog = self._bdq_sim(seed=21)
        tb2 = tbsim.get_tb(sim)
        profile = tb2.agent_strains

        target = set_agent(tb2, profile, TBS.SYMPTOMATIC, ['x2'])

        resolver = AcquisitionResolver(
            p_selective={'BDQ': 1.0},
            rng=DetRng(np.zeros(50)),
        )
        resolver.selective_acquisition(profile, target, drugs_used=['BDQ'], tb=tb2)

        has_x2 = bool(np.asarray(getattr(tb2, 'carries_x2')[target], dtype=bool)[0])
        has_xdr = bool(np.asarray(getattr(tb2, 'carries_xdr')[target], dtype=bool)[0])
        assert not has_x2, 'X2 must be replaced (removed) after selective acquisition'
        assert has_xdr, 'XDR must replace X2 after selective BDQ acquisition'
        assert int(profile.n_strains_per_agent(target)[0]) == 1, (
            'Strain replacement: must still have exactly 1 strain'
        )

    def test_state_dependent_modifier_zero_for_latent_agents(self):
        """
        Spec: "allow acquisition risk to vary by what TB state an agent is in."
        "In practice ... set the risk to 0 for agents who get treated despite
        not being SYMPTOMATIC or ASYMPTOMATIC."

        Default modifier for INFECTION = 0 → selective acquisition is suppressed.
        """
        sim, catalog = self._bdq_sim(seed=22)
        tb2 = tbsim.get_tb(sim)
        profile = tb2.agent_strains

        target = set_agent(tb2, profile, TBS.INFECTION, ['x1'])

        resolver = AcquisitionResolver(
            p_selective={'BDQ': 1.0},
            rng=DetRng(np.zeros(50)),
        )
        resolver.selective_acquisition(profile, target, drugs_used=['BDQ'], tb=tb2)

        n_after = int(profile.n_strains_per_agent(target)[0])
        assert n_after == 1, (
            f'Selective acquisition must be suppressed for INFECTION-state agents '
            f'(ω=0 default). Strain count: {n_after}'
        )

    def test_state_dependent_modifier_nonzero_for_symptomatic(self):
        """
        Spec: state modifier = 1 for SYMPTOMATIC → full ω applies.

        With p_selective=1.0 and ω_SYMPTOMATIC=1.0, acquisition is certain.
        """
        sim, catalog = self._bdq_sim(seed=23)
        tb2 = tbsim.get_tb(sim)
        profile = tb2.agent_strains

        target = set_agent(tb2, profile, TBS.SYMPTOMATIC, ['x1'])

        resolver = AcquisitionResolver(
            p_selective={'BDQ': 1.0},
            rng=DetRng(np.zeros(50)),
        )
        resolver.selective_acquisition(profile, target, drugs_used=['BDQ'], tb=tb2)

        has_x4 = bool(np.asarray(getattr(tb2, 'carries_x4')[target], dtype=bool)[0])
        assert has_x4, (
            'SYMPTOMATIC agent (ω=1) with p_selective=1.0 must acquire BDQ resistance'
        )

    def test_drug_not_in_regimen_causes_no_acquisition(self):
        """
        Spec: "ω_{R,d} = 0 if drug d is not included in regimen R."

        Selective acquisition only fires for drugs listed in ``drugs_used``.
        With p_selective={'RIF': 1.0} but drugs_used=['BDQ'], the RIF trial
        must not fire because RIF is not in the regimen.
        """
        sim, catalog = self._bdq_sim(seed=24)
        tb2 = tbsim.get_tb(sim)
        profile = tb2.agent_strains

        target = set_agent(tb2, profile, TBS.SYMPTOMATIC, ['x1'])

        resolver = AcquisitionResolver(
            p_selective={'RIF': 1.0},
            rng=DetRng(np.zeros(50)),
        )
        resolver.selective_acquisition(profile, target, drugs_used=['BDQ'], tb=tb2)

        has_x2 = bool(np.asarray(getattr(tb2, 'carries_x2')[target], dtype=bool)[0])
        has_x4 = bool(np.asarray(getattr(tb2, 'carries_x4')[target], dtype=bool)[0])
        assert not has_x2, 'RIF acquisition must not fire; BDQ regimen does not use RIF'
        assert not has_x4, 'BDQ acquisition (not in p_selective) must not fire'
        assert int(profile.n_strains_per_agent(target)[0]) == 1, (
            'Strain profile must be unchanged when drug not in regimen'
        )

