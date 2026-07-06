"""
Developer tests for the multi-strain (drug-resistance) TB extension.

These go deeper than ``tests/test_resistance.py`` (the concise CI-level suite): they are
organized around the spec's *questions of interest* (``tbsim/resistance/docs/model-tests.md``
§10) and validate the agent-based ``tbsim.TBResistant`` against the deterministic two-strain
reference ODE (``tbsim.compartmental.TwoStrainODE``). Shared ABM↔ODE plumbing is in
``ode_utils.py``. Run with, e.g.::

    /software/conda/bin/python -m pytest tbsim/resistance/devtests -v
"""
