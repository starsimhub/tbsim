# JAX optimization for TBsim — summary

## Context

TBsim's canonical workload (Tess's `run_one_sim.py`, ~42k agents, 1950–2050, monthly dt) takes ~9 minutes. Profiling showed the ABM hot path is dominated by network transmission inside `TB.step` (~48–64%), `Pregnancy.step` (~22%), and `DynamicNetwork.step` (~12%). Around the ABM, calibration and history-matching workflows (`run_history_matching_reidentify.py`, `run_calib_param_sweep.py`) execute large LHS sweeps over the LSHTM compartmental ODE — currently `scipy.integrate.odeint` called in a Python loop.

An earlier attempt to JAX-ify the ABM step (`TB(use_jax=True)`) made the simulation **3× slower** on CPU because per-step host↔device array conversion dominates any kernel-level gain. That code was removed.

## Solution

A self-contained, optional JAX backend for the compartmental TB model: `tbsim/compartmental/jax_ode.py`.

It exposes three entry points:

```python
from tbsim.compartmental import TB_JAX_ODE, batch_run, calibrate
```

| API | Purpose | Mechanism |
|---|---|---|
| `TB_JAX_ODE(...).run()` | Single ODE integration, drop-in for `TB_ODE` | `jax.jit(jax.experimental.ode.odeint)` |
| `batch_run(pars_list, ...)` | N parameter sets integrated in parallel | `jax.vmap` over the parameter axis |
| `calibrate(initial_pars, targets, calib_keys, ...)` | Gradient-based fit to compartment fractions | `jax.grad` + hand-rolled Adam |

Float64 enabled (`jax.config.update('jax_enable_x64', True)`); ODE RHS is a 1:1 port of `TB_ODE.run` in `lshtm_ode.py`. JAX is wired as an optional extra (`pip install -e ".[jax]"`); the import is guarded so non-JAX users are unaffected.

Validated:

- JAX vs SciPy agreement: **~1e-8 relative error** across 3 β regimes × 9 compartments
- `vmap` batch vs serial loop: **bit-exact (0.0 diff)**
- `jax.grad` through `odeint` vs central finite differences: **1.3e-6 rel error**
- 5/5 tests pass with tight tolerances

## Real improvement

**What is faster** (benchmarked on CPU, MacBook):

| Workflow | Before (SciPy) | After (JAX) | Speedup |
|---|---|---|---|
| Single ODE run (1900–2020) | 15.7 ms | 0.5 ms | 30× |
| LHS sweep, 16 param sets | 0.45 s | 0.29 s | 1.6× |
| LHS sweep, 64 param sets | 1.76 s | 0.35 s | 5.0× |
| **LHS sweep, 256 param sets** | **7.11 s** | **0.34 s** | **20.8×** |
| 1-param gradient calibration | minutes (Optuna) | 2.3 s | qualitative |

**What is not faster:** the 9-minute ABM run itself. None of this code touches `TB.step`, networks, demographics, or analyzers.

**Where the gain actually lands:**

1. **History-matching wave generation** — `run_history_matching_reidentify.py` runs LHS sweeps over the compartmental burn-in. Swapping the `scipy.odeint` loop for `batch_run` cuts a 1000-point sweep from ~28 s to ~1.4 s. Wave-by-wave, this compounds.
2. **Differentiable forward model** — `calibrate` enables gradient-based parameter fitting and downstream methods (SBI, variational inference) that were not previously available in TBsim. Verified on identifiable inverse problems; multi-parameter cases require regularisation or priors due to the usual identifiability degeneracy at endemic steady state (documented in the `calibrate` docstring).
3. **Burn-in handoff substrate** — `TB_JAX_ODE.handoff_state(year)` returns `(init_prev, seed_kwargs)` matching the existing `utils.ode_state_to_init_params` contract, so it slots into the burn-in-then-ABM pattern without changes to `make_sim`.

**Files changed** (all under `tbperformance/`):

- `tbsim/compartmental/jax_ode.py` (new, 333 lines)
- `tbsim/compartmental/__init__.py` (optional re-export)
- `pyproject.toml` (`[jax]` extra)
- `tests/test_jax_accel.py` (5 tests: SciPy parity, gradient parity, vmap parity, handoff, calibration)
- `tests/benchmark_jax.py` (the numbers above)

**Honest bound:** this is a real, measurable improvement for calibration and parameter-sweep workflows that surround the ABM. It is not a speedup of the ABM itself — the wins for the 9-minute canonical run remain the ones already identified in the Slack thread (annual `randomnet.dt`, `householdnet.update_freq`, ODE burn-in substitution).
