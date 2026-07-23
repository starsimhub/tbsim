# Compartmental models

This folder implements two families of deterministic (ODE) TB models used as validation references for the agent-based models:

- **Single-strain LSHTM model** in three versions -- the original model in R (`lshtm_ode.R`), an exact translation into Python (`lshtm_ode.py:TB_ODE()`), and a translation into Starsim (`lshtm_ode.py:TB_ODE_SS()`).
- **Two-strain (drug-resistance) model** -- the original model in R (`two_strain_ode.R`) and its Python port (`two_strain_ode.py:TwoStrainODE`), used as the deterministic reference for `tbsim.TBResistant`.

See `tests/test_compartmental.py` for related tests.