"""
Benchmark JAX vmap batched ODE vs sequential SciPy odeint.

Run from repo root:
    python tests/benchmark_jax.py
"""

import time

import numpy as np

from tbsim.compartmental import jax_ode
from tbsim.compartmental.lshtm_ode import TB_ODE


def bench_single():
    """Single-run comparison: JAX vs SciPy. Expected: roughly comparable."""
    print('--- Single ODE run, 1900-2020 ---')

    # SciPy baseline
    t0 = time.perf_counter()
    for _ in range(20):
        TB_ODE().run(start_time=1900, end_time=2020)
    t_scipy = (time.perf_counter() - t0) / 20

    # JAX (after warmup)
    jax_ode.TB_JAX_ODE(start=1900, stop=2020).run()  # JIT warmup
    t0 = time.perf_counter()
    for _ in range(20):
        jax_ode.TB_JAX_ODE(start=1900, stop=2020).run()
    t_jax = (time.perf_counter() - t0) / 20

    print(f'  SciPy odeint: {t_scipy * 1000:.2f} ms/run')
    print(f'  JAX odeint:   {t_jax * 1000:.2f} ms/run')


def bench_batch(batch_size):
    """Batched LHS-style sweep: JAX vmap vs SciPy loop. Expected: big JAX win."""
    print(f'--- LHS sweep, batch={batch_size}, 1900-2020 ---')
    rng = np.random.default_rng(0)
    beta_grid = rng.uniform(5.0, 15.0, batch_size)
    pars_list = [dict(beta=float(b)) for b in beta_grid]

    # SciPy: sequential loop
    t0 = time.perf_counter()
    for p in pars_list:
        TB_ODE(**p).run(start_time=1900, end_time=2020)
    t_scipy = time.perf_counter() - t0

    # JAX: single vmapped call (warm up first)
    jax_ode.batch_run(pars_list[:4], start=1900, stop=2020)
    t0 = time.perf_counter()
    jax_ode.batch_run(pars_list, start=1900, stop=2020)
    t_jax = time.perf_counter() - t0

    print(f'  SciPy loop:  {t_scipy:.3f}s  ({t_scipy / batch_size * 1000:.2f} ms/run)')
    print(f'  JAX vmap:    {t_jax:.3f}s  ({t_jax / batch_size * 1000:.2f} ms/run)')
    if t_jax > 0:
        print(f'  Speedup:     {t_scipy / t_jax:.1f}x')


def bench_calibration():
    """Gradient-based calibration: recover known beta from synthetic targets."""
    print('--- Gradient calibration (1 parameter, 200 steps) ---')

    truth = jax_ode.TB_JAX_ODE(pars=dict(beta=9.0), start=1900, stop=2000).run()
    final = truth.results[-1]
    total = final.sum()
    targets = {
        'SYMPTOMATIC': float(final[6] / total),
        'ASYMPTOMATIC': float(final[5] / total),
    }

    t0 = time.perf_counter()
    result = jax_ode.calibrate(
        initial_pars=dict(beta=5.0),
        targets=targets,
        calib_keys=['beta'],
        start=1900, stop=2000,
        lr=0.1, n_steps=200,
    )
    elapsed = time.perf_counter() - t0

    print(f'  Initial beta:    5.000')
    print(f'  Recovered beta:  {result["best_pars"]["beta"]:.3f}  (truth 9.000)')
    print(f'  Final loss:      {result["final_loss"]:.2e}')
    print(f'  Wall time:       {elapsed:.2f}s')


def main():
    if not jax_ode.available():
        print('JAX not installed. Run: pip install -e ".[jax]"')
        return
    print(f'JAX available: True\n')
    bench_single()
    print()
    for n in [16, 64, 256]:
        bench_batch(n)
        print()
    bench_calibration()


if __name__ == '__main__':
    main()
