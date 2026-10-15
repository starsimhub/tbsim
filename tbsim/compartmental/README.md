# Compartmental models

This folder holds the reference ODEs for the TB models in `tbsim/models.py`, which are declared as states and transitions (Starsim v4) and so run as an ODE, as a stochastic compartmental model, or as agents:

- **Single-strain LSHTM model** -- the original model in R (`lshtm_ode.R`) and an exact translation into Python (`lshtm_ode.py:TB_ODE()`), the reference for `tbsim.TB`.
- **Two-strain (drug-resistance) model** -- the original model in R (`two_strain_ode.R`) and its Python port (`two_strain_ode.py:TwoStrainODE`), the reference for `tbsim.LSHTMResistant` and for `tbsim.TBResistant`.

See `tests/test_models.py` and `tests/test_compartmental.py` for the tests.
