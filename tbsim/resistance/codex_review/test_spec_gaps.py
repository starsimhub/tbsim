"""Intentionally failing acceptance tests for gaps in the resistance tech spec.

These tests are *not* marked xfail: each is intended to turn green when the
corresponding missing capability is implemented.  Run this file separately from
the project's passing regression suite.
"""

from pathlib import Path

import numpy as np

import tbsim


def test_treatment_accepts_an_explicit_per_strain_efficacy_vector():
    """Treatment must be able to represent arbitrary T_l={t_1,l,...,t_m,l}."""
    strains = tbsim.Strains(["RIF", "BDQ"])
    requested = np.array([0.90, 0.70, 0.40, 0.35])

    product = tbsim.TxR(strains=strains, efficacy_by_strain=requested)

    np.testing.assert_allclose(product.eff_by_id, requested)


def test_treatment_accepts_an_agent_varying_adherence_distribution():
    """Adherence should be a sampled agent-level value shared by all strains."""
    strains = tbsim.Strains(["RIF"])

    # The proposed public contract is deliberately simple: a callable receives
    # UIDs and returns one adherence value per treated agent. Current TxR passes
    # this object to a Bernoulli probability instead of sampling adherence values.
    adherence_distribution = lambda uids: np.linspace(0.2, 0.9, len(uids))
    product = tbsim.TxR(strains=strains, adherence=adherence_distribution)

    assert callable(getattr(product, "adherence_distribution", None))


def test_treatment_history_can_classify_failure_versus_new_case():
    """DST/retreatment routing needs the spec's time-since-last-treatment classifier."""
    assert hasattr(tbsim.TxDeliveryR, "failure_case_eligibility"), (
        "TxDeliveryR records course-local start/end times but exposes no durable "
        "failure-versus-new-case classification for later DST/treatment routing"
    )


def test_requested_lai_tpt_burden_validation_artifact_exists():
    """The requested before/after per-100k burden comparison must be reproducible."""
    review_dir = Path(__file__).resolve().parent
    artifact = review_dir / "lai_tpt_burden_validation.csv"
    assert artifact.exists(), (
        "No calibrated resistance-off versus resistance-on burden table exists "
        "for prevalence, asymptomatic incidence, and TB mortality per 100,000"
    )

