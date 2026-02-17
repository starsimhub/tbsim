TB LSHTM Model
==============

.. currentmodule:: tbsim.tb_lshtm

This module defines an individual-based tuberculosis (TB) natural history model
reflecting the LSHTM (London School of Hygiene & Tropical Medicine) structure,
implemented for use in the Starsim simulation framework. The model adopts the
LSHTM state-transition structure and is provided as a subclass of
:class:`starsim.Infection`.

State definitions in this model
-------------------------------

- ``TBSL.SUSCEPTIBLE``: Susceptible to TB infection.
- ``TBSL.INFECTION``: Latent infection (not yet active TB).
- ``TBSL.CLEARED``: Cleared infection without developing active TB.
- ``TBSL.NON_INFECTIOUS``: Non-infectious TB (early/smear-negative).
- ``TBSL.RECOVERED``: Recovered from non-infectious TB (susceptible to reinfection).
- ``TBSL.ASYMPTOMATIC``: Active TB, asymptomatic (infectious).
- ``TBSL.SYMPTOMATIC``: Active TB, symptomatic (infectious).
- ``TBSL.TREATMENT``: On TB treatment.
- ``TBSL.TREATED``: Completed treatment (susceptible to reinfection).
- ``TBSL.DEAD``: TB-related death.
- ``TBSL.ACUTE``: Acute infection immediately after exposure (TB_LSHTM_Acute only).

State transition diagram
------------------------

.. mermaid::

   graph TD
       S[Susceptible] -->|λ| I[INFECTION]
       I -->|infcle| C[CLEARED]
       I -->|infnon| N[NON_INFECTIOUS]
       I -->|infasy| A[ASYMPTOMATIC]
       N -->|nonrec| R[RECOVERED]
       N -->|nonasy| A
       A -->|asynon| N
       A -->|asysym| Sym[SYMPTOMATIC]
       Sym -->|symasy| A
       Sym -->|θ| T[TREATMENT]
       Sym -->|μTB| D[DEAD]
       T -->|φ| Sym
       T -->|δ| Tr[TREATED]
       C -.->|λ| I
       R -.->|λ·π| I
       Tr -.->|λ·ρ| I

Transition rate names in :class:`TB_LSHTM` ``pars`` (e.g. ``inf_cle``, ``asy_sym``,
``theta``) correspond to the diagram labels above: infcle→inf_cle, infnon→inf_non,
infasy→inf_asy, nonrec→non_rec, nonasy→non_asy, asynon→asy_non, asysym→asy_sym,
symasy→sym_asy, θ→theta, μTB→mu_tb, φ→phi, δ→delta.

API Reference
------------

.. automodule:: tbsim.tb_lshtm
   :members:
   :undoc-members:
   :show-inheritance:
