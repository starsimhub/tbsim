"""Tests for the JAX-backed compartmental TB model."""

import numpy as np
import pytest

from tbsim.compartmental import jax_ode


pytestmark = pytest.mark.skipif(not jax_ode.available(), reason='JAX not installed')


def test_single_run_matches_scipy():
    """JAX integration should match SciPy to tight ODE tolerance across regimes."""
    from tbsim.compartmental.lshtm_ode import TB_ODE

    names = ['SUSCEPTIBLE', 'INFECTION', 'CLEARED', 'RECOVERED',
             'NON_INFECTIOUS', 'ASYMPTOMATIC', 'SYMPTOMATIC',
             'TREATMENT', 'TREATED']
    for beta in [6.0, 9.0, 12.0]:
        scipy_run = TB_ODE(beta=beta).run(start_time=1900, end_time=1950)
        jax_run = jax_ode.TB_JAX_ODE(pars=dict(beta=beta),
                                     start=1900, stop=1950, dt=1.0).run()
        for i, name in enumerate(names):
            s = scipy_run[name][-1]
            j = jax_run.results[-1, i]
            rel = abs(j - s) / max(abs(s), 1.0)
            assert rel < 1e-6, f'{name} (beta={beta}): scipy={s}, jax={j}, rel={rel}'


def test_gradient_matches_finite_difference():
    """Autodiff through odeint should match central finite differences."""
    import jax
    import jax.numpy as jnp

    pars_vec = jax_ode._PARS_DEFAULT.copy()
    beta_idx = jax_ode._PARS_KEYS.index('beta')
    N_idx = jax_ode._PARS_KEYS.index('N')
    y0 = jnp.asarray(jax_ode._default_y0(pars_vec[N_idx]))
    t = jnp.linspace(1900, 2000, 101)

    def f(beta):
        v = jnp.asarray(pars_vec).at[beta_idx].set(beta)
        res = jax_ode._integrate(y0, t, v)
        return res[-1, 6] / res[-1].sum()  # SYMPTOMATIC fraction

    grad = float(jax.grad(f)(9.0))
    eps = 1e-3
    fd = (float(f(9.0 + eps)) - float(f(9.0 - eps))) / (2 * eps)
    rel = abs(grad - fd) / abs(fd)
    assert rel < 1e-4, f'autodiff={grad}, fd={fd}, rel={rel}'


def test_handoff_state_sums_to_one():
    ode = jax_ode.TB_JAX_ODE(start=1900, stop=1950).run()
    init_prev, seed_kwargs = ode.handoff_state(1950)
    assert 0 < init_prev < 1
    assert abs(sum(seed_kwargs.values()) - 1.0) < 1e-6


def test_batch_run_respects_N_override():
    """Default y0 should use each parameter set's ``N``, not ``default_pars.N``."""
    pars_list = [dict(N=1e5), dict(N=2e5)]
    _, res_batch = jax_ode.batch_run(pars_list, start=1900, stop=1950)
    for i, p in enumerate(pars_list):
        single = jax_ode.TB_JAX_ODE(pars=p, start=1900, stop=1950).run().results
        np.testing.assert_allclose(res_batch[i], single, rtol=1e-4)


def test_batch_run_matches_serial():
    """Batched vmap output should match a Python loop element-wise."""
    pars_list = [
        dict(beta=8.0),
        dict(beta=9.0),
        dict(beta=10.0),
        dict(beta=11.0),
    ]
    t_batch, res_batch = jax_ode.batch_run(pars_list, start=1900, stop=1950)
    assert res_batch.shape == (4, len(t_batch), 9)

    for i, p in enumerate(pars_list):
        single = jax_ode.TB_JAX_ODE(pars=p, start=1900, stop=1950).run().results
        np.testing.assert_allclose(res_batch[i], single, rtol=1e-4)


def test_calibrate_recovers_known_parameter():
    """Synthetic test: generate targets with known beta, recover via gradient."""
    # 1. Generate target with truth beta
    truth = jax_ode.TB_JAX_ODE(pars=dict(beta=9.0), start=1900, stop=2000).run()
    final = truth.results[-1]
    total = final.sum()
    targets = {
        'SYMPTOMATIC': float(final[6] / total),
        'ASYMPTOMATIC': float(final[5] / total),
    }

    # 2. Start far from truth and calibrate
    result = jax_ode.calibrate(
        initial_pars=dict(beta=5.0),
        targets=targets,
        calib_keys=['beta'],
        start=1900, stop=2000,
        lr=0.1, n_steps=200,
    )
    recovered = result['best_pars']['beta']
    assert abs(recovered - 9.0) < 0.5, f'expected ~9.0, got {recovered}'
    assert result['final_loss'] < 1e-6
