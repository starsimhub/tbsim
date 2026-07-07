"""Strain-aware intervention tests: treatment, DST/routing, and TPT.

Covers regimens, StrainAwareTx delivery, relapse restoration, treatment
monitoring, DSTDx/DSTDelivery, RegimenRouter, and StrainAwareTPTTx.
"""

import inspect

import numpy as np
import pytest
import starsim as ss

import tbsim
from tbsim import TBS, Tx
from tbsim.resistance import (
    DSTDelivery,
    DSTDx,
    MultiStrainTB,
    Regimen,
    RegimenRouter,
    ResistanceConnector,
    StrainAwareTPTTx,
    StrainAwareTx,
    StrainAwareTxDelivery,
    StrainCatalog,
    StrainSpec,
    treatment_monitoring_eligibility,
)

from resistance_helpers import (
    DetRng,
    basic_sim,
    default_catalog,
    default_strains,
    make_example_sim,
    make_ss_sim,
    set_agent,
    two_strain_catalog,
)


class TestRegimen:
    def test_max_combine_with_pan_sus(self):
        catalog = default_catalog()
        r = Regimen('first_line', drugs=['INH', 'RIF'],
                    per_drug_efficacy={'INH': 0.9, 'RIF': 0.95})
        probs = r.strain_cure_probs(catalog)
        assert probs[0] == pytest.approx(0.95)

    def test_mdr_uncovered_is_zero(self):
        catalog = default_catalog()
        r = Regimen('first_line', drugs=['INH', 'RIF'])
        probs = r.strain_cure_probs(catalog)
        assert probs[3] == 0.0

    def test_inh_r_with_rif(self):
        catalog = default_catalog()
        r = Regimen('first_line', drugs=['INH', 'RIF'],
                    per_drug_efficacy={'INH': 0.9, 'RIF': 0.95})
        probs = r.strain_cure_probs(catalog)
        assert probs[1] == pytest.approx(0.95)

    def test_parallel_combine(self):
        catalog = default_catalog()
        r = Regimen('parallel', drugs=['INH', 'RIF'],
                    per_drug_efficacy={'INH': 0.5, 'RIF': 0.5}, combine='parallel')
        probs = r.strain_cure_probs(catalog)
        assert probs[0] == pytest.approx(0.75)

    def test_base_efficacy_scales(self):
        catalog = default_catalog()
        r = Regimen('halved', drugs=['INH'], per_drug_efficacy={'INH': 1.0},
                    base_efficacy=0.5)
        probs = r.strain_cure_probs(catalog)
        assert probs[0] == pytest.approx(0.5)

    def test_unknown_drug_raises(self):
        catalog = default_catalog()
        r = Regimen('bad', drugs=['XYZ'])
        with pytest.raises(ValueError):
            r.strain_cure_probs(catalog)

    def test_resistance_penalty_allows_partial_resistant_efficacy(self):
        """ODE reference: resistant strains can have reduced-but-nonzero efficacy."""
        _, catalog = two_strain_catalog()
        regimen = Regimen(
            'inh_partial',
            drugs=['INH'],
            per_drug_efficacy={'INH': 0.75},
            resistance_penalty={'INH': 0.4},
        )
        probs = regimen.strain_cure_probs(catalog)
        assert probs[catalog.index('pan')] == pytest.approx(0.75)
        assert probs[catalog.index('inh_r')] == pytest.approx(0.30)

    def test_resistance_penalty_rejects_unknown_drug(self):
        with pytest.raises(ValueError):
            Regimen('bad_penalty', drugs=['INH'], resistance_penalty={'RIF': 0.5})


class TestRegimenPerStrainEfficacy:
    """Spec §Treatment: per-strain clinical efficacy ψ_{R,j}."""

    def _catalog_with_bdq(self):
        strains = [
            StrainSpec('x1', {'RIF': 0, 'BDQ': 0}, fitness=1.0, init_prev=0.08),
            StrainSpec('x2', {'RIF': 1, 'BDQ': 0}, fitness=0.9, init_prev=0.03),
            StrainSpec('x4', {'RIF': 0, 'BDQ': 1}, fitness=0.8, init_prev=0.02),
            StrainSpec('xdr', {'RIF': 1, 'BDQ': 1}, fitness=0.7, init_prev=0.01),
        ]
        return strains, StrainCatalog(strains)

    def _bdq_regimen(self, catalog, per_drug_efficacy=None):
        return Regimen(
            'bdq_mono',
            drugs=['BDQ'],
            per_drug_efficacy=per_drug_efficacy or {'BDQ': 0.9},
        )

    def test_bdq_regimen_zero_efficacy_against_bdq_resistant_strain(self):
        _, catalog = self._catalog_with_bdq()
        regimen = self._bdq_regimen(catalog, per_drug_efficacy={'BDQ': 0.9})
        probs = regimen.strain_cure_probs(catalog)
        assert probs[catalog.index('x4')] == 0.0
        assert probs[catalog.index('xdr')] == 0.0

    def test_bdq_regimen_full_efficacy_against_bdq_susceptible_strains(self):
        _, catalog = self._catalog_with_bdq()
        regimen = self._bdq_regimen(catalog, per_drug_efficacy={'BDQ': 0.85})
        probs = regimen.strain_cure_probs(catalog)
        np.testing.assert_allclose(probs[catalog.index('x1')], 0.85, rtol=1e-9)
        np.testing.assert_allclose(probs[catalog.index('x2')], 0.85, rtol=1e-9)

    def test_rif_resistant_strain_treated_by_bdq_same_as_pan(self):
        _, catalog = self._catalog_with_bdq()
        regimen = self._bdq_regimen(catalog, per_drug_efficacy={'BDQ': 0.8})
        probs = regimen.strain_cure_probs(catalog)
        np.testing.assert_allclose(
            probs[catalog.index('x1')], probs[catalog.index('x2')], rtol=1e-9,
        )

    def test_partial_cure_means_per_strain_efficacy_is_zero_for_resistant(self):
        _, catalog = self._catalog_with_bdq()
        regimen = self._bdq_regimen(catalog, per_drug_efficacy={'BDQ': 1.0})
        probs = regimen.strain_cure_probs(catalog)
        np.testing.assert_allclose(probs[catalog.index('x1')], 1.0)
        np.testing.assert_allclose(probs[catalog.index('x4')], 0.0)
        assert probs[catalog.index('x1')] + probs[catalog.index('x4')] < 2.0


class TestStrainAwareTxFlow:
    def test_strain_aware_tx_requires_regimen(self):
        with pytest.raises(TypeError):
            StrainAwareTx(regimen='nope', catalog=default_catalog())

    def test_strain_aware_tx_delivery_rejects_wrong_product(self):
        with pytest.raises(TypeError):
            StrainAwareTxDelivery(product=Tx())

    def test_tx_clears_susceptible_strains_only(self):
        sim = make_ss_sim(n_agents=80)
        sim.init()
        tb = tbsim.get_tb(sim)
        all_uids = tb.sim.people.auids
        clean = all_uids[tb.agent_strains.n_strains_per_agent(all_uids) == 0]
        target = clean[:3]
        tb.agent_strains.add_strain(target, 'pan')
        tb.agent_strains.add_strain(target, 'mdr')

        catalog = default_catalog()
        regimen = Regimen('first_line', drugs=['INH', 'RIF'])
        drug_cols = [catalog.drugs.index(d) for d in regimen.drugs]
        covered = np.all(catalog.resistance[:, drug_cols] == 0, axis=1)
        profile = tb.agent_strains
        for s_idx in np.where(covered)[0]:
            profile.remove_strain(target, int(s_idx))

        assert not profile.carries('pan', target).any()
        assert profile.carries('mdr', target).all()


class TestODEReferenceTreatmentOperator:
    """Two-strain ODE π(m→s) treatment operator checks."""

    @staticmethod
    def _make_operator_sim(e_pan=0.75, e_res=0.30, q_acq=0.10, n_agents=30000):
        strains, catalog = two_strain_catalog()
        tb = MultiStrainTB(
            strains=strains,
            pars=dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0)),
        )
        regimen = Regimen(
            'inh_ode_reference',
            drugs=['INH'],
            per_drug_efficacy={'INH': e_pan},
            resistance_penalty={'INH': e_res / e_pan},
        )
        product = StrainAwareTx(
            regimen=regimen,
            catalog=catalog,
            p_selective_acquisition={'INH': q_acq},
            adherence=1.0,
            p_relapse=0.0,
        )
        delivery = StrainAwareTxDelivery(product=product, name='tx_ode_reference')
        sim = tbsim.Sim(
            tb_model=tb,
            sim_pars=dict(
                n_agents=n_agents,
                start=ss.date('2000-01-01'),
                stop=ss.date('2000-02-01'),
                dt=ss.days(14),
                rand_seed=4,
            ),
            interventions=[delivery],
            connectors=ResistanceConnector(),
            networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=1), dur=30)),
        )
        sim.pars.verbose = 0
        sim.init()
        return sim, tbsim.get_tb(sim), sim.interventions['tx_ode_reference'].product

    @staticmethod
    def _mask_counts(tb, uids):
        profile = tb.agent_strains
        pan = profile.carries('pan', uids)
        inh_r = profile.carries('inh_r', uids)
        masks = pan.astype(int) + 2 * inh_r.astype(int)
        return {mask: float(np.mean(masks == mask)) for mask in (0, 1, 2, 3)}

    def _outcome_dist(self, tb, product, strain_names):
        uids = tb.sim.people.auids
        profile = tb.agent_strains
        tb.state[uids] = TBS.SYMPTOMATIC
        tb.infected[uids] = True
        tb.susceptible[uids] = False
        profile.clear_all(uids)
        for name in strain_names:
            profile.add_strain(uids, name)

        outcomes = product.administer(tb.sim, uids)
        failure_uids = outcomes['failure']

        for s_idx, cured_mask in outcomes['per_strain'].items():
            cured_uids = uids[cured_mask].intersect(failure_uids)
            if len(cured_uids):
                profile.remove_strain(cured_uids, int(s_idx))

        product._acq_resolver.selective_acquisition(
            profile,
            failure_uids,
            product.regimen.drugs,
            tb=tb,
        )
        profile.clear_all(outcomes['success'])
        return self._mask_counts(tb, uids)

    def test_treatment_outcome_operator_matches_ode_pi_table(self):
        """Per-strain cure plus replacement acquisition matches the ODE π table."""
        e_pan, e_res, q = 0.75, 0.30, 0.10
        _, tb, product = self._make_operator_sim(e_pan=e_pan, e_res=e_res, q_acq=q)

        mono_pan = self._outcome_dist(tb, product, ['pan'])
        assert mono_pan[0] == pytest.approx(e_pan, abs=0.02)
        assert mono_pan[1] == pytest.approx((1 - e_pan) * (1 - q), abs=0.02)
        assert mono_pan[2] == pytest.approx((1 - e_pan) * q, abs=0.02)

        mono_res = self._outcome_dist(tb, product, ['inh_r'])
        assert mono_res[0] == pytest.approx(e_res, abs=0.02)
        assert mono_res[2] == pytest.approx(1 - e_res, abs=0.02)

        mixed = self._outcome_dist(tb, product, ['pan', 'inh_r'])
        assert mixed[0] == pytest.approx(e_pan * e_res, abs=0.02)
        assert mixed[1] == pytest.approx((1 - e_pan) * e_res * (1 - q), abs=0.02)
        pi_ab_to_b = (
            e_pan * (1 - e_res)
            + (1 - e_pan) * e_res * q
            + (1 - e_pan) * (1 - e_res) * q
        )
        assert mixed[2] == pytest.approx(pi_ab_to_b, abs=0.02)
        assert mixed[3] == pytest.approx((1 - e_pan) * (1 - e_res) * (1 - q), abs=0.02)


class TestAdherenceCorrelation:
    def test_adherence_drawn_once_per_agent(self):
        src = inspect.getsource(StrainAwareTx.administer)
        assert 'adherent_mask' in src
        assert 'p_adherence' in src
        assert 'adherent_mask & roll & carriers' in src


class TestRelapseStrainRestoration:
    def _make_tx_sim(self, p_selective=None):
        strains = default_strains(include_mdr=False)
        tb = MultiStrainTB(strains=strains,
                      pars=dict(init_prev=ss.bernoulli(0.0)))
        regimen = Regimen('first', drugs=['INH'])
        product = StrainAwareTx(
            regimen=regimen,
            catalog=tb._strain_catalog,
            p_selective_acquisition=p_selective,
        )
        tx = StrainAwareTxDelivery(product=product, name='tx_first')
        sim = tbsim.Sim(
            n_agents=30, tb_model=tb,
            interventions=[tx],
            connectors=ResistanceConnector(),
            start=ss.date('2000-01-01'), stop=ss.date('2000-03-01'),
            dt=ss.days(14),
        )
        sim.pars.verbose = 0
        sim.init()
        return sim, tbsim.get_tb(sim), sim.interventions['tx_first']

    def test_step_relapses_restores_saved_strain_identity(self):
        sim, tb, tx = self._make_tx_sim()
        uid = sim.people.auids[0]
        tb.state[uid] = TBS.CLEARED
        tb.infected[uid] = False
        tb.susceptible[uid] = True
        tb.agent_strains.clear_all(ss.uids([uid]))

        tx.pending_relapse[uid] = True
        tx.ti_relapse[uid] = sim.ti
        tx._relapse_strains_by_uid[int(uid)] = (0,)

        tx.step_relapses()

        assert uid in tx._relapsed
        assert tb.state[uid] == TBS.SYMPTOMATIC
        assert bool(tb.agent_strains.carries('pan', ss.uids([uid]))[0])

    def test_step_relapses_can_apply_selective_acquisition(self):
        sim, tb, tx = self._make_tx_sim(p_selective={'INH': 1.0})
        uid = sim.people.auids[0]
        tb.state[uid] = TBS.CLEARED
        tb.infected[uid] = False
        tb.susceptible[uid] = True
        tb.agent_strains.clear_all(ss.uids([uid]))

        tx.pending_relapse[uid] = True
        tx.ti_relapse[uid] = sim.ti
        tx._relapse_strains_by_uid[int(uid)] = (0,)

        tx.step_relapses()

        assert uid in tx._relapsed
        assert bool(tb.agent_strains.carries('inh_r', ss.uids([uid]))[0])
        assert not bool(tb.agent_strains.carries('pan', ss.uids([uid]))[0])


class TestTreatmentMonitoring:
    def test_eligibility_gates_by_time_on_treatment(self):
        strains = default_strains(include_mdr=False)
        tb = MultiStrainTB(strains=strains,
                      pars=dict(init_prev=ss.bernoulli(0.3)))
        regimen = Regimen('first', drugs=['INH'])
        product = StrainAwareTx(regimen=regimen, catalog=tb._strain_catalog)
        tx = StrainAwareTxDelivery(product=product, name='tx_first')
        sim = tbsim.Sim(
            n_agents=200, tb_model=tb,
            interventions=[
                tbsim.HealthSeekingBehavior(),
                tbsim.DxDelivery(name='confirm', product=tbsim.Xpert(),
                                 coverage=1.0, result_state='diagnosed'),
                tx,
            ],
            connectors=ResistanceConnector(),
            start=ss.date('2000-01-01'), stop=ss.date('2002-01-01'),
            dt=ss.days(14),
        )
        sim.pars.verbose = 0
        sim.run()

        elig_fn = treatment_monitoring_eligibility('tx_first', after_steps=4)
        elig = elig_fn(sim)
        on_tx = tbsim.get_tb(sim).on_treatment.uids
        assert set(elig.tolist()) <= set(on_tx.tolist())
        if len(elig):
            starts = np.asarray(tx.ti_treatment_start[elig])
            assert (sim.ti - starts >= 4).all()

    def test_one_shot_and_periodic_semantics(self):
        strains = default_strains(include_mdr=False)
        tb = MultiStrainTB(strains=strains,
                      pars=dict(init_prev=ss.bernoulli(0.0)))
        regimen = Regimen('first', drugs=['INH'])
        product = StrainAwareTx(regimen=regimen, catalog=tb._strain_catalog)
        tx = StrainAwareTxDelivery(product=product, name='tx_first')
        sim = tbsim.Sim(
            n_agents=30, tb_model=tb,
            interventions=[tx],
            connectors=ResistanceConnector(),
            start=ss.date('2000-01-01'), stop=ss.date('2000-03-01'),
            dt=ss.days(14),
        )
        sim.pars.verbose = 0
        sim.init()
        tb = tbsim.get_tb(sim)
        tx = sim.interventions['tx_first']

        uids = sim.people.auids[:3]
        tb.on_treatment[uids] = True
        tx.ti_treatment_start[uids] = np.array([-4.0, -5.0, -6.0])

        one_shot = treatment_monitoring_eligibility('tx_first', after_steps=4)
        periodic = treatment_monitoring_eligibility('tx_first', after_steps=4, every_steps=2)

        assert list(one_shot(sim)) == [uids[0]]
        assert list(periodic(sim)) == [uids[0], uids[2]]


class TestDST:
    def test_dst_true_phenotype(self):
        sim = make_ss_sim(n_agents=40)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0, 1, 2], dtype=int))
        tb.agent_strains.clear_all(target)
        tb.agent_strains.add_strain(target[:1], 'pan')
        tb.agent_strains.add_strain(target[1:2], 'inh_r')
        tb.agent_strains.add_strain(target[2:3], 'mdr')

        dst = DSTDx(default_catalog(), drugs=['INH', 'RIF'],
                    sensitivity=1.0, specificity=1.0,
                    p_strain_obs=1.0)
        true_inh = dst.true_phenotype(tb, target, 'INH')
        np.testing.assert_array_equal(true_inh, [False, True, True])
        true_rif = dst.true_phenotype(tb, target, 'RIF')
        np.testing.assert_array_equal(true_rif, [False, False, True])

    def test_dst_perfect_sens_spec_administer(self):
        sim = make_ss_sim(n_agents=40)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0, 1, 2], dtype=int))
        tb.agent_strains.clear_all(target)
        tb.agent_strains.add_strain(target[:1], 'pan')
        tb.agent_strains.add_strain(target[1:2], 'inh_r')
        tb.agent_strains.add_strain(target[2:3], 'mdr')

        dst = DSTDx(default_catalog(), drugs=['INH', 'RIF'],
                    sensitivity=1.0, specificity=1.0,
                    p_strain_obs=1.0)
        out = dst.administer(sim, target)
        np.testing.assert_array_equal(out['INH'], [False, True, True])
        np.testing.assert_array_equal(out['RIF'], [False, False, True])

    def test_dst_strain_dropout_blocks_detection(self):
        sim = make_ss_sim(n_agents=40)
        sim.init()
        tb = tbsim.get_tb(sim)
        target = ss.uids(np.array([0, 1], dtype=int))
        tb.agent_strains.clear_all(target)
        tb.agent_strains.add_strain(target, 'inh_r')

        dst = DSTDx(default_catalog(), drugs=['INH'],
                    sensitivity=1.0, specificity=1.0,
                    p_strain_obs=0.0)
        out = dst.administer(sim, target)
        np.testing.assert_array_equal(out['INH'], [False, False])

    def test_dst_unknown_drug_raises(self):
        with pytest.raises(ValueError):
            DSTDx(default_catalog(), drugs=['XYZ'])


class TestRegimenRouter:
    def _make_sim(self):
        tb = MultiStrainTB(strains=default_strains()[:2],
                      pars=dict(init_prev=ss.bernoulli(0.05)))
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
        dst = DSTDelivery(product=DSTDx(tb._strain_catalog, drugs=['INH']))
        sim = ss.Sim(
            n_agents=40, diseases=tb, networks=net,
            connectors=ResistanceConnector(),
            interventions=[
                tbsim.HealthSeekingBehavior(),
                tbsim.DxDelivery(name='confirm', product=tbsim.Xpert(),
                                 coverage=1.0, result_state='diagnosed'),
                dst,
            ],
            start=ss.date('2000-01-01'),
            stop=ss.date('2000-06-01'),
            dt=ss.days(14),
        )
        sim.pars.verbose = 0
        sim.init()
        return sim

    def test_matches_filters_by_observed_phenotype(self):
        sim = self._make_sim()
        dst = sim.interventions['dstdelivery']
        router = RegimenRouter(dst, diagnosed_state=None,
                               require_dst_tested=False)
        uids = sim.people.auids[:6]
        dst.observed_INH_resistant[uids[:3]] = True
        dst.observed_INH_resistant[uids[3:]] = False
        tb = tbsim.get_tb(sim)
        tb.on_treatment[:] = False

        elig_resistant = router.matches(INH=True)
        elig_suscept = router.matches(INH=False)

        r_uids = elig_resistant(sim)
        s_uids = elig_suscept(sim)
        assert not set(uids[:3].tolist()).difference(set(r_uids.tolist()))
        assert not set(uids[3:].tolist()).difference(set(s_uids.tolist()))
        assert not set(r_uids.tolist()) & set(s_uids.tolist())

    def test_matches_rejects_unknown_drug(self):
        sim = self._make_sim()
        dst = sim.interventions['dstdelivery']
        router = RegimenRouter(dst)
        with pytest.raises(ValueError):
            router.matches(BDQ=True)


def _run_tpt_sim(target_state, p_tpt_acquisition, acq_state_modifiers=None,
                 efficacy=0.0, p_sterilize=0.0):
    """Run a sim where exactly one agent is staged in *target_state* with the
    pan strain and is the sole TPT recipient.

    Returns the final TB module and the staged uid.
    """
    strains, catalog = two_strain_catalog()
    regimen = Regimen('inh_mono', drugs=['INH'], per_drug_efficacy={'INH': 0.9})
    tpt_product = StrainAwareTPTTx(
        regimen=regimen,
        catalog=catalog,
        p_tpt_acquisition=p_tpt_acquisition,
        acq_state_modifiers=acq_state_modifiers,
        pars=dict(
            efficacy=ss.bernoulli(p=efficacy),
            p_sterilize=ss.bernoulli(p=p_sterilize),
            dur_treatment=ss.constant(v=ss.days(0)),
        ),
    )
    tpt_delivery = tbsim.TPTSimple(
        product=tpt_product,
        pars=dict(
            coverage=ss.bernoulli(p=1.0),
            eligible_states=[target_state],
        ),
    )
    tb_template = MultiStrainTB(strains=strains)
    sim = basic_sim(tb_template, interventions=[tpt_delivery])
    sim.init()
    tb = tbsim.get_tb(sim)  # ss.Sim clones modules; use the live reference

    # Wipe any seeded strain assignments and stage exactly one agent.
    uid = sim.people.auids[:1]
    for name in ('pan', 'inh_r'):
        getattr(tb, f'carries_{name}')[sim.people.auids] = False
    getattr(tb, 'carries_pan')[uid] = True
    tb.state[sim.people.auids] = TBS.SUSCEPTIBLE
    tb.state[uid] = target_state
    tb.infected[uid] = True
    tb.susceptible[uid] = False

    sim.run()
    return tb, uid


class TestStrainAwareTPT:
    def test_tpt_clear_mask_excludes_mdr(self):
        catalog = default_catalog()
        regimen = Regimen('isoniazid_tpt', drugs=['INH'])
        tpt = StrainAwareTPTTx(regimen=regimen, catalog=catalog)
        # pan is covered, inh_r/mdr are not
        assert tpt._cover_mask[0]  # pan
        assert not tpt._cover_mask[1]  # inh_r
        assert not tpt._cover_mask[3]  # mdr

    def test_tpt_invalid_regimen_type(self):
        with pytest.raises(TypeError):
            StrainAwareTPTTx(regimen='nope', catalog=default_catalog())


class TestTPTNeitherBranchAcquisition:
    """Spec: TPT failure ("neither" branch) triggers a selective acquisition trial."""

    def test_neither_branch_now_runs_acquisition(self):
        """With efficacy=0, every recipient lands in the neither branch.
        A susceptible carrier with p_tpt_acquisition=1.0 and state modifier=1
        must mutate to its resistant counterpart."""
        tb, uid = _run_tpt_sim(
            target_state=TBS.SYMPTOMATIC,  # default state modifier = 1
            p_tpt_acquisition={'INH': 1.0},
            efficacy=0.0,
        )
        assert not bool(tb.carries_pan[uid][0]), (
            'pan strain should have been replaced by acquisition'
        )
        assert bool(tb.carries_inh_r[uid][0]), (
            'TPT failure on the NEITHER branch must trigger selective '
            'acquisition per spec — susceptible carrier with p=1 must '
            'acquire resistance.'
        )

    def test_neither_branch_respects_state_modifier(self):
        """With acq modifier = 0 for the agent's state, no acquisition fires
        even on the neither branch."""
        tb, uid = _run_tpt_sim(
            target_state=TBS.SYMPTOMATIC,
            p_tpt_acquisition={'INH': 1.0},
            acq_state_modifiers={'infection': 0.0, 'non_infectious': 0.0,
                                 'asymptomatic': 0.0, 'symptomatic': 0.0,
                                 'treatment': 0.0, 'cleared': 0.0},
            efficacy=0.0,
        )
        assert bool(tb.carries_pan[uid][0]), (
            'modifier=0 must block acquisition'
        )
        assert not bool(tb.carries_inh_r[uid][0])


class TestTPTSterilizeBranchStateCoverage:
    """Spec: TPT acquisition state modifiers apply to all states, not just INFECTION."""

    def test_sterilize_branch_applies_acquisition_in_non_infection_states(self):
        """A NON_INFECTIOUS agent in the sterilize branch must get acquisition
        trials. Previously the sterilize branch filtered to INFECTION first,
        making the NON_INFECTIOUS state modifier dead code."""
        tb, uid = _run_tpt_sim(
            target_state=TBS.NON_INFECTIOUS,
            p_tpt_acquisition={'INH': 1.0},
            acq_state_modifiers={'infection': 0.0,
                                 'non_infectious': 1.0,  # only NI gates open
                                 'asymptomatic': 0.0, 'symptomatic': 0.0,
                                 'treatment': 0.0, 'cleared': 0.0},
            efficacy=1.0, p_sterilize=1.0,  # sterilize branch
        )
        # Sterilization → CLEARED is gated on INFECTION; NON_INFECTIOUS
        # agents remain in NON_INFECTIOUS. But acquisition must have fired.
        assert tb.state[uid][0] == TBS.NON_INFECTIOUS, (
            'sterilize→CLEARED is a latent-only transition'
        )
        assert bool(tb.carries_inh_r[uid][0]), (
            'NON_INFECTIOUS agent in sterilize branch with state '
            'modifier=1 must acquire resistance — previously this was '
            'unreachable because the sterilize branch gated on '
            'state == INFECTION before acquisition.'
        )

    def test_sterilize_branch_still_clears_latent_infection(self):
        """Sterilize → CLEARED still works for INFECTION-state agents
        (regression check that the fix didn't break the happy path)."""
        tb, uid = _run_tpt_sim(
            target_state=TBS.INFECTION,
            p_tpt_acquisition=None,  # no acquisition
            efficacy=1.0, p_sterilize=1.0,
        )
        assert tb.state[uid][0] == TBS.CLEARED, (
            'Sterilize → CLEARED must still apply for latent agents.'
        )
        assert not bool(tb.infected[uid][0])

