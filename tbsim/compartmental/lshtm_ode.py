"""
Define compartmental TB models.

TB_ODE is a translation of the R model (lshtm_ode.R) into Python, integrated with scipy.odeint. It is
the reference for ``tbsim.TB``, which is the same model declared as states and transitions, and
runs as an ODE or as agents.

State and parameter names are aligned with the agent-based ``tbsim.TB`` class. The ODE keeps
``CLEARED`` split into three sub-states (CLEARED, RECOVERED, TREATED) so that the appropriate
reinfection multiplier can be applied to each pathway, matching the per-agent
``rr_reinfection_cleared`` / ``rr_reinfection_rec`` / ``rr_reinfection_treat`` mechanism in
``tbsim.TB``. Treatment dynamics (theta, phi, delta) are off by default — enable explicitly
when modelling a post-treatment-era period.
"""

import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
from scipy.integrate import odeint

__all__ = ['default_pars', 'TB_ODE']


# Default parameters. Natural-history rates and reinfection multipliers are aligned with
# tbsim.TB defaults so that the same numeric value produces the same expected per-year flux
# in both models. Note that ``beta`` here is a contact-rate-times-transmissibility product
# under homogeneous mixing, with different semantics than tbsim.TB's per-edge ``beta``.
default_pars = sc.objdict(
    # Transmission
    beta        = 9,    # Contact (per person/year). Different semantics than tbsim.TB beta (per-edge per-dt).
    trans_asymp = 0.82, # Relative infectiousness, ASYMPTOMATIC vs SYMPTOMATIC (formerly kappa)

    # Reinfection multipliers on FOI for each cleared sub-state
    rr_reinfection_cleared = 1.0,  # CLEARED   (cleared from latent INFECTION)
    rr_reinfection_rec     = 0.21, # RECOVERED (recovered from NON_INFECTIOUS) — formerly pi
    rr_reinfection_treat   = 3.15, # TREATED   (completed TREATMENT)            — formerly rho

    # From INFECTION (latent)
    inf_cle = 1.90, # INFECTION -> CLEARED
    inf_non = 0.16, # INFECTION -> NON_INFECTIOUS
    inf_asy = 0.06, # INFECTION -> ASYMPTOMATIC

    # From NON_INFECTIOUS
    non_rec = 0.18, # NON_INFECTIOUS -> RECOVERED
    non_asy = 0.25, # NON_INFECTIOUS -> ASYMPTOMATIC

    # From ASYMPTOMATIC
    asy_non = 1.66, # ASYMPTOMATIC -> NON_INFECTIOUS
    asy_sym = 0.88, # ASYMPTOMATIC -> SYMPTOMATIC

    # From SYMPTOMATIC
    sym_asy  = 0.54, # SYMPTOMATIC -> ASYMPTOMATIC
    sym_dead = 0.34, # SYMPTOMATIC -> DEAD (constant; replaces time-varying mutb_ini/mutb_fin)

    # Treatment (off by default; set theta>0 to enable diagnosis/treatment dynamics)
    theta = 0.0, # Diagnosis rate, SYMPTOMATIC -> TREATMENT
    phi   = 0.0, # Treatment failure, TREATMENT -> SYMPTOMATIC
    delta = 2,   # Treatment completion, TREATMENT -> TREATED (6-month duration)

    # Demography
    N  = 1e5,  # Population size (held constant)
    mu = 1/70, # Background mortality rate
)


class TB_ODE(sc.prettyobj):
    """
    Continuous-time compartmental TB model integrated with scipy.odeint.

    See ``default_pars`` for parameter definitions.

    Example
    -------
    ::

        from tbsim.compartmental.lshtm_ode import TB_ODE

        tb = TB_ODE()
        tb.run()
        tb.plot()
    """
    def __init__(self, **kwargs):
        """Initialize with ``default_pars``; keyword arguments override individual parameters."""
        self.pars = sc.dcp(default_pars)
        self.pars.update(kwargs)
        return

    def run(self, start_time=1500, end_time=2020):
        """Integrate the ODE system from ``start_time`` to ``end_time`` and store results."""
        p = self.pars

        def des(state, t):
            (SUSCEPTIBLE, INFECTION, CLEARED, RECOVERED, NON_INFECTIOUS,
             ASYMPTOMATIC, SYMPTOMATIC, TREATMENT, TREATED) = state

            foi = (p.beta / p.N) * (p.trans_asymp * ASYMPTOMATIC + SYMPTOMATIC)

            dSUSCEPTIBLE    = (p.mu * p.N) + (p.sym_dead * SYMPTOMATIC) \
                              - foi * SUSCEPTIBLE - p.mu * SUSCEPTIBLE
            dINFECTION      = foi * (SUSCEPTIBLE
                                     + p.rr_reinfection_cleared * CLEARED
                                     + p.rr_reinfection_rec     * RECOVERED
                                     + p.rr_reinfection_treat   * TREATED) \
                              - (p.inf_cle + p.inf_non + p.inf_asy + p.mu) * INFECTION
            dCLEARED        = p.inf_cle * INFECTION \
                              - foi * p.rr_reinfection_cleared * CLEARED - p.mu * CLEARED
            dRECOVERED      = p.non_rec * NON_INFECTIOUS \
                              - foi * p.rr_reinfection_rec * RECOVERED - p.mu * RECOVERED
            dNON_INFECTIOUS = p.inf_non * INFECTION + p.asy_non * ASYMPTOMATIC \
                              - (p.non_rec + p.non_asy + p.mu) * NON_INFECTIOUS
            dASYMPTOMATIC   = p.inf_asy * INFECTION + p.non_asy * NON_INFECTIOUS + p.sym_asy * SYMPTOMATIC \
                              - (p.asy_non + p.asy_sym + p.mu) * ASYMPTOMATIC
            dSYMPTOMATIC    = p.asy_sym * ASYMPTOMATIC - p.sym_asy * SYMPTOMATIC \
                              - p.theta * SYMPTOMATIC + p.phi * TREATMENT \
                              - p.sym_dead * SYMPTOMATIC - p.mu * SYMPTOMATIC
            dTREATMENT      = p.theta * SYMPTOMATIC - p.phi * TREATMENT \
                              - p.delta * TREATMENT - p.mu * TREATMENT
            dTREATED        = p.delta * TREATMENT \
                              - foi * p.rr_reinfection_treat * TREATED - p.mu * TREATED

            return [dSUSCEPTIBLE, dINFECTION, dCLEARED, dRECOVERED, dNON_INFECTIOUS,
                    dASYMPTOMATIC, dSYMPTOMATIC, dTREATMENT, dTREATED]

        # Initial conditions: 1% start in SYMPTOMATIC active TB
        yini = [p.N - 1e3, 0, 0, 0, 0, 0, 1e3, 0, 0]

        times = np.arange(start_time, end_time + 1, 1)
        out = odeint(des, yini, times)

        labels = ['SUSCEPTIBLE', 'INFECTION', 'CLEARED', 'RECOVERED', 'NON_INFECTIOUS',
                  'ASYMPTOMATIC', 'SYMPTOMATIC', 'TREATMENT', 'TREATED']
        results = sc.objdict(time=times)
        for i, label in enumerate(labels):
            results[label] = out[:, i]

        # Derived quantities
        results.TBc = results.ASYMPTOMATIC + results.SYMPTOMATIC                # TB prevalence
        results.Mor = p.sym_dead * results.SYMPTOMATIC                          # TB mortality
        results.Dxs = p.theta * results.SYMPTOMATIC                             # TB notifications
        with np.errstate(divide='ignore', invalid='ignore'):
            results.Spr = np.where(results.TBc > 0,
                                   results.ASYMPTOMATIC / results.TBc, 0)      # Proportion asymptomatic

        self.results = results
        return results

    def plot(self, **kwargs):
        """Plot all results, each in a separate axes."""
        res = self.results
        labels = sc.objdict(
            SUSCEPTIBLE    = 'Susceptible',
            INFECTION      = 'Latent infection',
            CLEARED        = 'Cleared (from latent)',
            RECOVERED      = 'Recovered (from active)',
            NON_INFECTIOUS = 'Non-infectious',
            ASYMPTOMATIC   = 'Asymptomatic',
            SYMPTOMATIC    = 'Symptomatic',
            TREATMENT      = 'On treatment',
            TREATED        = 'Completed treatment',
            TBc            = 'TB prevalence',
            Mor            = 'TB mortality',
            Dxs            = 'TB notifications',
            Spr            = 'Proportion asymptomatic',
        )
        kw = sc.mergedicts(dict(lw=2, alpha=0.8), kwargs)
        with sc.options.with_style('fancy'):
            fig, axs = sc.getrowscols(len(labels), make=True)
            for ax, key in zip(axs.flatten(), labels):
                ax.plot(res.time, res[key], **kw)
                ax.set_title(labels[key])
                ax.set_xlabel('Year')
                sc.boxoff(ax)
                sc.commaticks(ax)
                if res[key].min() >= 0:
                    ax.set_ylim(bottom=0)
            sc.figlayout()
        return fig
