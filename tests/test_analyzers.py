"""
Test TBsim analyzers
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim
import pytest


def make_sim(n_agents=1000, interventions=None, **kwargs):
    """Create and return a tbsim.Sim with HSB and optional interventions."""
    sim = tbsim.Sim(
        n_agents=n_agents,
        interventions=interventions,
        sim_pars=dict(start='2000-01-01', stop='2005-12-31'),
        tb_pars=dict(init_prev=0.30),
        **kwargs
    )
    return sim


def test_householdstats():
    """Test HouseholdStats: results, matrices, snapshots, and plots."""
    # Build sim with a HouseholdNet and the analyzer
    np.random.seed(0)
    hh_ids = np.arange(50)
    ages = [sc.strjoin(np.random.randint(1, 70, np.random.randint(2, 6))) for _ in hh_ids]
    dhs = sc.dataframe(hh_id=hh_ids, ages=ages)
    net = ss.HouseholdNet(dhs_data=dhs, dynamic=False)
    az = tbsim.HouseholdStats(save_at=['2001-01-01', '2004-01-01'])
    sim = make_sim(analyzers=az, networks=net, copy_inputs=False)
    sim.run()

    # Documented results are present and sensible
    n_bins = len(az.age_bins) - 1
    assert (np.array(az.results['n_households']) > 0).all()
    assert (np.array(az.results['max_hh_size']) >= np.array(az.results['mean_hh_size'])).all()
    assert az.age_mixing_initial.shape == (n_bins, n_bins)
    assert np.allclose(az.age_mixing_final, az.age_mixing_final.T)
    assert az.age_mixing_snapshots and az.age_counts_snapshots

    # Plot methods run
    az.plot()
    return


if __name__ == '__main__':
    pytest.main(["-x", "-v", __file__])
