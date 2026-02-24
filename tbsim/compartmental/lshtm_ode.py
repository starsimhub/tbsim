"""
Define compartmental TB models

TB_ODE is a literal translation of the R model (lshtm_ode.R) into Python, with exact integration.

TB_SS is a Starsim implementation of the TB_ODE model, with Euler integration.
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
from scipy.integrate import odeint

__all__ = ['default_pars', 'TB_ODE', 'TB_SS']


# Default parameters
default_pars = sc.objdict(
  beta = 9,         # Contact (per person/year) parameter
  kappa = 0.75,     # Relative infectiousness
  infcle = 1.83,    # Infected -> Cleared
  infmin = 0.21,    # Infected -> Minimal
  minrec = 0.16,    # Minimal -> Recovered
  pi = 0.21,        # Protection from reinfection
  minsub = 0.25,    # Minimal -> Subclinical
  infsub = 0.07,    # Infected -> Subclinical
  submin = 1.58,    # Subclinical -> Minimal
  subcln = 0.77,    # Subclinical -> Clinical
  clnsub = 0.53,    # Clinical -> Subclinical
  mutb_ini = 0.3,   # TB mortality (Initial)
  mutb_fin = 0.23,  # TB mortality (Final)
  theta_ini = 0.44, # Diagnosis (Initial)
  theta_fin = 0.9,  # Diagnosis (Final)
  phi_ini = 0.69,   # Treatment failure (Initial)
  phi_fin = 0.09,   # Treatment failure (Final)
  rho = 3.25,       # Risk of reinfection
  N = 1e5,          # Population size
  mu = 1/70,        # Mortality rate
  delta = 2,        # Treatment duration (6 months)
)


class TB_ODE(sc.prettyobj):
    """
    Exact translation of the R model (lshtm_ode.R) into Python. No Starsim.

    See default_pars for parameter definitions.

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
        self.pars = default_pars
        self.pars.update(kwargs)
        return

    def run(self, start_time=1500, end_time=2020):
        """Integrate the ODE system from ``start_time`` to ``end_time`` and store results."""
        p = self.pars

        # Forcing functions (piecewise linear interpolation, constant extrapolation)
        # Equivalent to R's approxfun(method="linear", rule=2)
        interp_times  = [start_time, 1999, end_time]
        mutb_vals   = [p.mutb_ini,  p.mutb_ini,  p.mutb_fin]
        theta_vals  = [p.theta_ini, p.theta_ini, p.theta_fin]
        phi_vals    = [p.phi_ini,   p.phi_ini,   p.phi_fin]

        def force_mutb(t):  return np.interp(t, interp_times, mutb_vals)
        def force_theta(t): return np.interp(t, interp_times, theta_vals)
        def force_phi(t):   return np.interp(t, interp_times, phi_vals)

        # ODE system
        def des(state, t):
            SUS, INF, CLE, REC, MIN, SUB, CLN, TXT, TRE = state

            mutb  = force_mutb(t)
            theta = force_theta(t)
            phi   = force_phi(t)
            foi   = (p.beta / p.N) * (p.kappa * SUB + CLN)  # Force of infection

            dSUS = (p.mu * p.N) + (mutb * CLN) - foi * SUS - p.mu * SUS
            dINF = foi * (SUS + CLE + p.pi * REC + p.rho * TRE) - p.infcle * INF - p.infmin * INF - p.infsub * INF - p.mu * INF
            dCLE = p.infcle * INF - foi * CLE - p.mu * CLE
            dREC = p.minrec * MIN - foi * (p.pi * REC) - p.mu * REC
            dMIN = p.infmin * INF + p.submin * SUB - p.minrec * MIN - p.minsub * MIN - p.mu * MIN
            dSUB = p.infsub * INF + p.minsub * MIN + p.clnsub * CLN - p.submin * SUB - p.subcln * SUB - p.mu * SUB
            dCLN = p.subcln * SUB - p.clnsub * CLN - theta * CLN + phi * TXT - mutb * CLN - p.mu * CLN
            dTXT = theta * CLN - phi * TXT - p.delta * TXT - p.mu * TXT
            dTRE = p.delta * TXT - foi * (p.rho * TRE) - p.mu * TRE

            return [dSUS, dINF, dCLE, dREC, dMIN, dSUB, dCLN, dTXT, dTRE]

        # Initial conditions
        yini = [1e5 - 1e3, 0, 0, 0, 0, 0, 1e3, 0, 0]

        # Solve ODE
        times = np.arange(start_time, end_time + 1, 1)
        out = odeint(des, yini, times)

        # Build results
        labels = ['SUS', 'INF', 'CLE', 'REC', 'MIN', 'SUB', 'CLN', 'TXT', 'TRE']
        results = sc.objdict(time=times)
        for i, label in enumerate(labels):
            results[label] = out[:, i]

        # Derived quantities (matching R output columns)
        mutb_arr  = np.array([force_mutb(t) for t in times])
        theta_arr = np.array([force_theta(t) for t in times])
        results.TBc = results.SUB + results.CLN                    # TB prevalence (per 100k)
        results.Mor = mutb_arr * results.CLN                       # TB mortality (per 100k)
        results.Dxs = theta_arr * results.CLN                      # TB notifications (per 100k)
        results.Spr = results.SUB / (results.SUB + results.CLN)    # Proportion subclinical TB (%)

        self.results = results
        return results

    def plot(self, **kwargs):
        """ Plot all results, each in a separate axes """
        res = self.results
        labels = sc.objdict(
            SUS='Susceptible', INF='Infected', CLE='Cleared', REC='Recovered',
            MIN='Minimal', SUB='Subclinical', CLN='Clinical', TXT='On treatment',
            TRE='Treated', TBc='TB prevalence', Mor='TB mortality',
            Dxs='TB notifications', Spr='Proportion subclinical',
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


class TB_SS(ss.Module):
    """
    Compartmental Starsim implementation of the LSHTM TB model (Euler integration).

    Because this is a self-contained module, it does not need a network or People

    See default_pars for parameter definitions.

    **Example**:

        import starsim as ss
        import tbsim.compartmental as tbc
        tb = tbc.TB_SS()
        sim = ss.Sim(modules=tb, start=1920, stop=2020, dt=0.1, n_agents=1)
        sim.run()
        sim.plot()
    """
    def __init__(self, **kwargs):
        """Initialize compartmental states and parameters; keyword arguments override defaults."""
        super().__init__()
        self.define_pars(
            **default_pars,
            start_time = 1500,
            mid_time = 1999,
            end_time = 2020,
        )
        self.update_pars(**kwargs)

        # Compartment labels
        self.c_labels = sc.objdict(
            SUS='Susceptible',
            INF='Infected',
            CLE='Cleared',
            REC='Recovered',
            MIN='Minimal',
            SUB='Subclinical',
            CLN='Clinical',
            TXT='On treatment',
            TRE='Treated',
        )

        # Compartments (scalars, not per-agent states)
        self.c = sc.objdict({key:0 for key in self.c_labels}) # Compartment names
        return

    def init_post(self):
        """ Set initial conditions """
        super().init_post()
        p = self.pars
        c = self.c
        init_cln = 1e3
        c.SUS = p.N - init_cln
        c.CLN = init_cln
        self.interp_time = [p.start_time, p.mid_time, p.end_time]
        return

    def force_mutb(self, t):
        """ Forcing function for TB mortality (piecewise linear, constant extrapolation) """
        p = self.pars
        return np.interp(t, self.interp_time, [p.mutb_ini, p.mutb_ini, p.mutb_fin])

    def force_theta(self, t):
        """ Forcing function for diagnosis rate """
        p = self.pars
        return np.interp(t, self.interp_time, [p.theta_ini, p.theta_ini, p.theta_fin])

    def force_phi(self, t):
        """ Forcing function for treatment failure rate """
        p = self.pars
        return np.interp(t, self.interp_time, [p.phi_ini, p.phi_ini, p.phi_fin])

    def step(self):
        """ Euler integration of the ODE system """
        p = self.pars
        c = self.c
        t = self.now
        dt = float(self.dt) # Note: this is hard-coded in years

        # Time-varying parameters
        mutb  = self.force_mutb(t)
        theta = self.force_theta(t)
        phi   = self.force_phi(t)
        foi   = (p.beta / p.N) * (p.kappa * c.SUB + c.CLN)  # Force of infection

        # Compute derivatives (matching R ODE)
        d = sc.objdict()
        d.SUS = (p.mu * p.N) + (mutb * c.CLN) - foi * c.SUS - p.mu * c.SUS
        d.INF = foi * (c.SUS + c.CLE + p.pi * c.REC + p.rho * c.TRE) - p.infcle * c.INF - p.infmin * c.INF - p.infsub * c.INF - p.mu * c.INF
        d.CLE = p.infcle * c.INF - foi * c.CLE - p.mu * c.CLE
        d.REC = p.minrec * c.MIN - foi * (p.pi * c.REC) - p.mu * c.REC
        d.MIN = p.infmin * c.INF + p.submin * c.SUB - p.minrec * c.MIN - p.minsub * c.MIN - p.mu * c.MIN
        d.SUB = p.infsub * c.INF + p.minsub * c.MIN + p.clnsub * c.CLN - p.submin * c.SUB - p.subcln * c.SUB - p.mu * c.SUB
        d.CLN = p.subcln * c.SUB - p.clnsub * c.CLN - theta * c.CLN + phi * c.TXT - mutb * c.CLN - p.mu * c.CLN
        d.TXT = theta * c.CLN - phi * c.TXT - p.delta * c.TXT - p.mu * c.TXT
        d.TRE = p.delta * c.TXT - foi * (p.rho * c.TRE) - p.mu * c.TRE
        d.SUS = (p.mu * p.N) + (mutb * c.CLN) - foi * c.SUS - p.mu * c.SUS
        d.INF = foi * (c.SUS + c.CLE + p.pi * c.REC + p.rho * c.TRE) - p.infcle * c.INF - p.infmin * c.INF - p.infsub * c.INF - p.mu * c.INF
        d.CLE = p.infcle * c.INF - foi * c.CLE - p.mu * c.CLE
        d.REC = p.minrec * c.MIN - foi * (p.pi * c.REC) - p.mu * c.REC

        # Euler update
        for key in d:
            c[key] += d[key] * dt
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.define_results(
            *[ss.Result(key, label=value) for key, value in self.c_labels.items()],
            ss.Result('TBc', label='TB prevalence'),
            ss.Result('Mor', label='TB mortality'),
            ss.Result('Dxs', label='TB notifications'),
            ss.Result('Spr', label='Proportion subclinical', scale=False),
        )
        return

    def update_results(self):
        """ Store the current state """
        super().update_results()
        ti = self.ti
        c = self.c
        for key in c:
            self.results[key][ti] = c[key]

        # Derived quantities
        tb_prev = c.SUB + c.CLN
        self.results.TBc[ti] = tb_prev
        self.results.Mor[ti] = self.force_mutb(self.now) * c.CLN
        self.results.Dxs[ti] = self.force_theta(self.now) * c.CLN
        self.results.Spr[ti] = c.SUB / tb_prev if tb_prev > 0 else 0
        return

    def plot(self, **kwargs):
        """ Plot all results, each in a separate axes """
        results = list(self.results.all_results)
        kw = sc.mergedicts(dict(lw=2, alpha=0.8), kwargs)
        with sc.options.with_style('fancy'):
            fig = plt.figure()
            nrows, ncols = sc.getrowscols(len(results))
            for i, res in enumerate(results):
                ax = plt.subplot(nrows, ncols, i + 1)
                res.plot(ax=ax, **kw)
                sc.boxoff(ax)
                if res.min() >= 0:
                    ax.set_ylim(bottom=0)
            sc.figlayout()
        return ss.return_fig(fig)