"""
Differentiable + batched JAX implementation of the LSHTM compartmental TB model.

This is the *real* JAX payoff for TBsim: a single ODE integration is small enough
that JAX has no edge over SciPy, but two things make JAX dramatically faster than
existing tooling:

1. ``batch_run`` uses ``jax.vmap`` to integrate many parameter sets in parallel.
   For history-matching / LHS sweeps (e.g. ``run_history_matching_reidentify.py``)
   this replaces N sequential ``scipy.odeint`` calls with one batched call.

2. ``calibrate`` uses ``jax.grad`` (autodiff) + Adam to fit ODE parameters to
   targets without finite-difference gradients or Optuna trials. For a tractable
   surrogate (or the burn-in phase) this is orders of magnitude faster than
   black-box optimisation.

Requires ``pip install tbsim[jax]``.
"""

import numpy as np
import sciris as sc

from .lshtm_ode import default_pars

__all__ = ['available', 'TB_JAX_ODE', 'batch_run', 'mse_loss', 'calibrate']

# Compartments in fixed order
_STATE_NAMES = [
    'SUSCEPTIBLE', 'INFECTION', 'CLEARED', 'RECOVERED', 'NON_INFECTIOUS',
    'ASYMPTOMATIC', 'SYMPTOMATIC', 'TREATMENT', 'TREATED',
]
_N_STATES = len(_STATE_NAMES)

# Parameters in fixed order (matches default_pars). Stored as a tuple so JAX
# can carry it as a static argument.
_PARS_KEYS = tuple(default_pars.keys())
_PARS_DEFAULT = np.array([float(default_pars[k]) for k in _PARS_KEYS])


try:
    import jax
    # float64 is essential: rates × dt × small probabilities lose precision in fp32
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.experimental.ode import odeint

    _HAS_JAX = True

    def _deriv(state, t, pars_vec):
        """LSHTM TB ODE right-hand side. Mirrors ``TB_ODE.run`` exactly."""
        (S, I, C, R, N, A, Y, T, Tr) = state
        # Unpack parameter vector by fixed index
        p = {k: pars_vec[i] for i, k in enumerate(_PARS_KEYS)}

        foi = (p['beta'] / p['N']) * (p['trans_asymp'] * A + Y)

        dS  = p['mu'] * p['N'] + p['sym_dead'] * Y - foi * S - p['mu'] * S
        dI  = foi * (S + p['rr_reinfection_cleared'] * C
                       + p['rr_reinfection_rec'] * R
                       + p['rr_reinfection_treat'] * Tr) \
              - (p['inf_cle'] + p['inf_non'] + p['inf_asy'] + p['mu']) * I
        dC  = p['inf_cle'] * I - foi * p['rr_reinfection_cleared'] * C - p['mu'] * C
        dR  = p['non_rec'] * N - foi * p['rr_reinfection_rec'] * R - p['mu'] * R
        dN  = p['inf_non'] * I + p['asy_non'] * A \
              - (p['non_rec'] + p['non_asy'] + p['mu']) * N
        dA  = p['inf_asy'] * I + p['non_asy'] * N + p['sym_asy'] * Y \
              - (p['asy_non'] + p['asy_sym'] + p['mu']) * A
        dY  = p['asy_sym'] * A - p['sym_asy'] * Y \
              - p['theta'] * Y + p['phi'] * T \
              - p['sym_dead'] * Y - p['mu'] * Y
        dT  = p['theta'] * Y - p['phi'] * T - p['delta'] * T - p['mu'] * T
        dTr = p['delta'] * T - foi * p['rr_reinfection_treat'] * Tr - p['mu'] * Tr

        return jnp.array([dS, dI, dC, dR, dN, dA, dY, dT, dTr])

    @jax.jit
    def _integrate(y0, t, pars_vec):
        """Single ODE integration. JIT-compiled, differentiable."""
        return odeint(_deriv, y0, t, pars_vec)

    # vmap over the first axis of pars_vec → run a batch of parameter sets in
    # parallel. Same t and y0 for all members of the batch.
    _batch_integrate = jax.jit(jax.vmap(_integrate, in_axes=(None, None, 0)))

except ImportError:
    _HAS_JAX = False


def available():
    """Return True if JAX is installed."""
    return _HAS_JAX


def _default_y0(N):
    """LSHTM-style initial conditions: 0.1% seed in SYMPTOMATIC."""
    y0 = np.zeros(_N_STATES)
    y0[0] = N - 1e3   # SUSCEPTIBLE
    y0[6] = 1e3       # SYMPTOMATIC
    return y0


def _pars_to_vec(pars):
    """Pack a dict of parameter overrides into the canonical vector."""
    vec = _PARS_DEFAULT.copy()
    for k, v in pars.items():
        if k not in _PARS_KEYS:
            raise KeyError(f'Unknown parameter: {k}; expected one of {_PARS_KEYS}')
        vec[_PARS_KEYS.index(k)] = float(v)
    return vec


class TB_JAX_ODE(sc.prettyobj):
    """
    Single-run JAX-backed compartmental TB model with differentiable integration.

    Drop-in alternative to ``TB_ODE`` for cases where you need batched runs or
    autodiff. For a single run it is comparable to (slightly slower than) SciPy
    after the first JIT compile.

    Args:
        pars (dict): parameter overrides (subset of ``default_pars``)
        start (float): start year
        stop (float): stop year
        dt (float): time-step (years)
        y0 (array-like): initial compartment vector; default is 0.1% in SYMPTOMATIC

    Example:
        ::

            from tbsim.compartmental.jax_ode import TB_JAX_ODE

            ode = TB_JAX_ODE(start=1910, stop=1950)
            ode.run()
            init_prev, seed_kwargs = ode.handoff_state(1950)
    """

    def __init__(self, pars=None, start=1500, stop=2020, dt=1.0, y0=None):
        if not _HAS_JAX:
            raise ImportError('JAX required. Install with: pip install tbsim[jax]')
        self.pars = sc.mergedicts(default_pars, pars)
        self.start = float(start)
        self.stop = float(stop)
        self.dt = float(dt)
        self.y0 = y0 if y0 is not None else _default_y0(self.pars.N)
        self.results = None
        self.t = None
        return

    def run(self):
        """Integrate from start to stop and store results."""
        n_steps = int(np.ceil((self.stop - self.start) / self.dt)) + 1
        t = jnp.linspace(self.start, self.stop, n_steps)
        pars_vec = jnp.asarray(_pars_to_vec(self.pars))
        y0 = jnp.asarray(self.y0)
        self.t = np.asarray(t)
        self.results = np.asarray(_integrate(y0, t, pars_vec))
        return self

    def handoff_state(self, year):
        """
        Return ``(init_prev, seed_kwargs)`` for ABM handoff at *year*.

        Mirrors ``utils.ode_state_to_init_params`` output format.
        """
        if self.results is None:
            raise RuntimeError('Call run() first')
        idx = int(np.argmin(np.abs(self.t - year)))
        S, I, C, R, N, A, Y, T, Tr = self.results[idx]
        ever = I + C + R + N + A + Y + T + Tr
        total = S + ever
        init_prev = float(ever / total) if total else 0.0
        if ever <= 0:
            return init_prev, {k: 0.0 for k in
                ['init_inf', 'init_non', 'init_asy', 'init_sym',
                 'init_cle', 'init_rec', 'init_treat']}
        seed_kwargs = dict(
            init_inf   = float(I  / ever),
            init_non   = float(N  / ever),
            init_asy   = float(A  / ever),
            init_sym   = float(Y  / ever),
            init_cle   = float(C  / ever),
            init_rec   = float(R  / ever),
            init_treat = float(Tr / ever),
        )
        return init_prev, seed_kwargs

    def to_dataframe(self):
        """Return results as a pandas DataFrame."""
        import pandas as pd
        if self.results is None:
            raise RuntimeError('Call run() first')
        df = pd.DataFrame(self.results, columns=_STATE_NAMES)
        df['time'] = self.t
        return df


def batch_run(pars_batch, start=1500, stop=2020, dt=1.0, y0=None):
    """
    Run a batch of parameter sets in parallel via ``jax.vmap``.

    Use for LHS, history-matching sweeps, or any "run the ODE for these N
    parameter combinations" workflow. The batch is run as a single JIT-compiled
    op; speedup over a Python loop scales with batch size.

    Args:
        pars_batch (list of dict OR (n_batch, n_pars) array): parameter
            override dicts, or pre-packed vector matching ``_PARS_KEYS`` order
        start, stop, dt: same as ``TB_JAX_ODE``
        y0 (array): shared initial condition; default 0.1% SYMPTOMATIC

    Returns:
        t (array, shape (n_steps,))
        results (array, shape (n_batch, n_steps, n_states))
    """
    if not _HAS_JAX:
        raise ImportError('JAX required. Install with: pip install tbsim[jax]')

    if isinstance(pars_batch, np.ndarray):
        pars_vecs = pars_batch
    else:
        pars_vecs = np.stack([_pars_to_vec(p) for p in pars_batch])

    N_pop = float(default_pars.N if y0 is None else default_pars.N)
    if y0 is None:
        y0 = _default_y0(N_pop)

    n_steps = int(np.ceil((stop - start) / dt)) + 1
    t = jnp.linspace(start, stop, n_steps)
    results = _batch_integrate(jnp.asarray(y0), t, jnp.asarray(pars_vecs))
    return np.asarray(t), np.asarray(results)


# ---- Gradient-based calibration --------------------------------------------

def mse_loss(pars_vec, t, y0, target_indices, target_values, weights=None):
    """
    Mean-squared error between final-time compartment fractions and targets.

    Args:
        pars_vec (array): packed parameter vector (length ``len(_PARS_KEYS)``)
        t (array): time grid
        y0 (array): initial compartment vector
        target_indices (array): compartment indices to compare (0..8)
        target_values (array): target fractions of total population
        weights (array): optional per-target weights

    Returns:
        scalar loss (JAX-traceable, differentiable).
    """
    res = _integrate(y0, t, pars_vec)
    final = res[-1]
    total = jnp.sum(final)
    fractions = final[target_indices] / total
    diffs = fractions - target_values
    if weights is not None:
        diffs = diffs * jnp.sqrt(weights)
    return jnp.mean(diffs ** 2)


def calibrate(initial_pars, targets, calib_keys,
              start=1500, stop=2020, dt=1.0, y0=None,
              lr=0.01, n_steps=300, verbose=False):
    """
    Gradient-based calibration of a subset of ODE parameters to targets.

    Uses ``jax.grad`` (autodiff) + Adam (hand-rolled, no Optax dependency)
    to find parameter values that minimise MSE between final-time compartment
    fractions and target values.

    Identifiability warning: with K free parameters and fewer than K independent
    targets, the inverse problem is under-determined and the optimiser will
    happily settle on a parameter set whose forward solution matches the targets
    but is not the "true" generator. Use ``batch_run`` or history matching when
    you need to characterise the full set of acceptable parameter combinations.

    Args:
        initial_pars (dict): full parameter dict (defaults filled in)
        targets (dict): ``{compartment_name: fraction_of_total}`` (e.g.
            ``{'SYMPTOMATIC': 0.003, 'ASYMPTOMATIC': 0.005}``)
        calib_keys (list of str): parameter names to optimise; others stay fixed
        start, stop, dt: ODE time grid
        y0 (array): initial conditions
        lr (float): Adam learning rate
        n_steps (int): number of optimisation steps
        verbose (bool): print loss every 50 steps

    Returns:
        dict with keys ``best_pars``, ``loss_history``, ``final_loss``
    """
    if not _HAS_JAX:
        raise ImportError('JAX required. Install with: pip install tbsim[jax]')

    pars = sc.mergedicts(default_pars, initial_pars)
    full_vec = _pars_to_vec(pars)
    calib_idx = jnp.array([_PARS_KEYS.index(k) for k in calib_keys])

    tgt_idx = jnp.array([_STATE_NAMES.index(name) for name in targets.keys()])
    tgt_vals = jnp.array([float(v) for v in targets.values()])

    n_steps_ode = int(np.ceil((stop - start) / dt)) + 1
    t = jnp.linspace(start, stop, n_steps_ode)
    y0_arr = jnp.asarray(y0 if y0 is not None else _default_y0(pars.N))

    def loss_fn(theta):
        # theta is the subset being optimised; insert it back into full_vec
        full = jnp.asarray(full_vec).at[calib_idx].set(theta)
        return mse_loss(full, t, y0_arr, tgt_idx, tgt_vals)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Hand-rolled Adam (avoid optax dependency)
    theta = jnp.asarray([full_vec[i] for i in [_PARS_KEYS.index(k) for k in calib_keys]])
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history = []

    for step in range(1, n_steps + 1):
        loss, grad = loss_and_grad(theta)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mhat = m / (1 - b1 ** step)
        vhat = v / (1 - b2 ** step)
        theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)
        # Clip to positive — all TB rates are non-negative
        theta = jnp.maximum(theta, 1e-6)
        history.append(float(loss))
        if verbose and step % 50 == 0:
            print(f'  step {step:4d}  loss = {float(loss):.6g}')

    best_pars = dict(zip(calib_keys, [float(x) for x in np.asarray(theta)]))
    return dict(best_pars=best_pars, loss_history=history, final_loss=history[-1])
