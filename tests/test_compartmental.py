"""
Compare pure compartmental, compartmental Starsim, and agent-based Starsim LSHTM models.

Original R code:

```r
parms = c(
  beta = 9,        # Contact (per person/year) parameter
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
  rho = 3.25)       # Risk of reinfection

ode <- function(parms, end_time = 2020) {

  # Static parameters
  N <- 1e5 # Population size
  mu <- 1/70 # Age expectancy adult
  delta <- 2 # Treatment duration (6 months)

  # Function parameters
  forcer_mutb <- matrix(c(1500, parms['mutb_ini'], 1999, parms['mutb_ini'], 2020, parms['mutb_fin']), ncol = 2, byrow = TRUE)
  force_func_mutb <- approxfun(x = forcer_mutb[,1], y = forcer_mutb[,2], method = "linear", rule = 2)

  forcer_theta <- matrix(c(1500, parms['theta_ini'], 1999, parms['theta_ini'], 2020, parms['theta_fin']), ncol = 2, byrow = TRUE)
  force_func_theta <- approxfun(x = forcer_theta[,1], y = forcer_theta[,2], method = "linear", rule = 2)

  forcer_phi <- matrix(c(1500, parms['phi_ini'], 1999, parms['phi_ini'], 2020, parms['phi_fin']), ncol = 2, byrow = TRUE)
  force_func_phi <- approxfun(x = forcer_phi[,1], y = forcer_phi[,2], method = "linear", rule = 2)

  des <- function(time, state, parms) {

    with(as.list(c(state, parms)), {
      
      dSUS = ((mu * N) + (force_func_mutb(time) * CLN)) - (((beta / N) * ((kappa * SUB) + CLN)) * SUS) - (mu * SUS)
      dINF = (((beta / N) * ((kappa * SUB) + CLN)) * (SUS + CLE + (pi * REC) + (rho * TRE))) - (infcle * INF) - (infmin * INF) - (infsub * INF) - (mu * INF)
      dCLE = (infcle * INF) - (((beta / N) * ((kappa * SUB) + CLN)) * CLE) - (mu * CLE)
      dREC = (minrec * MIN) - (((beta / N) * ((kappa * SUB) + CLN)) * (pi * REC)) - (mu * REC)
      dMIN = (infmin * INF) + (submin * SUB) - (minrec * MIN) - (minsub * MIN) - (mu * MIN)
      dSUB = (infsub * INF) + (minsub * MIN) + (clnsub * CLN) - (submin * SUB) - (subcln * SUB) - (mu * SUB)
      dCLN = (subcln * SUB) - (clnsub * CLN) - (force_func_theta(time) * CLN) + (force_func_phi(time) * TXT) - (force_func_mutb(time) * CLN) - (mu * CLN)
      dTXT = (force_func_theta(time) * CLN) - (force_func_phi(time) * TXT) - (delta * TXT) - (mu * TXT)
      dTRE = (delta * TXT) - (((beta / N) * ((kappa * SUB) + CLN)) * (rho * TRE)) - (mu * TRE)

      return(list(c( 
        dSUS, dINF, dCLE, dREC, dMIN, dSUB, dCLN, dTXT, dTRE),
        TBc   = (SUB + CLN), # TB prevalence (per 100k)
        Mor   = (force_func_mutb(time) * CLN), # TB mortality (per 100k)
        Dxs   = (force_func_theta(time) * CLN), # TB notifications (per 100k)
        Spr   = (SUB / (SUB + CLN)))) # Proportion subclinical TB (%)
    })
  }
  
  yini <- c(SUS = 1e5 - 1e3, INF = 0, CLE = 0, REC = 0,
            MIN = 0, SUB = 0, CLN = 1e3, TXT = 0, TRE = 0)
  
  times <- seq(1500, end_time, by = 1)
  out <- deSolve::ode(yini, times, des, parms)
  return(out)
}
```
"""

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Default parameters; identical to above
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


# Define literal Python translation of R model
class TB_R(sc.prettyobj):
    
    def __init__(self, **kwargs):
        self.pars = default_pars
        self.pars.update(kwargs)
        return

    def run(self, end_time=2020):
        p = self.pars

        # Forcing functions (piecewise linear interpolation, constant extrapolation)
        # Equivalent to R's approxfun(method="linear", rule=2)
        mutb_times  = [1500, 1999, 2020]
        mutb_vals   = [p.mutb_ini,  p.mutb_ini,  p.mutb_fin]
        theta_times = [1500, 1999, 2020]
        theta_vals  = [p.theta_ini, p.theta_ini, p.theta_fin]
        phi_times   = [1500, 1999, 2020]
        phi_vals    = [p.phi_ini,   p.phi_ini,   p.phi_fin]

        def force_mutb(t):  return np.interp(t, mutb_times, mutb_vals)
        def force_theta(t): return np.interp(t, theta_times, theta_vals)
        def force_phi(t):   return np.interp(t, phi_times, phi_vals)

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
        times = np.arange(1500, end_time + 1, 1)
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


class TB_R_SS(ss.Module):
    """ Compartmental Starsim implementation of the LSHTM TB model (Euler integration) """

    def __init__(self, **kwargs):
        super().__init__()
        self.define_pars(
            beta      = 9,       # Contact (per person/year) parameter
            kappa     = 0.75,    # Relative infectiousness
            infcle    = 1.83,    # Infected -> Cleared
            infmin    = 0.21,    # Infected -> Minimal
            minrec    = 0.16,    # Minimal -> Recovered
            pi        = 0.21,    # Protection from reinfection
            minsub    = 0.25,    # Minimal -> Subclinical
            infsub    = 0.07,    # Infected -> Subclinical
            submin    = 1.58,    # Subclinical -> Minimal
            subcln    = 0.77,    # Subclinical -> Clinical
            clnsub    = 0.53,    # Clinical -> Subclinical
            mutb_ini  = 0.3,     # TB mortality (Initial)
            mutb_fin  = 0.23,    # TB mortality (Final)
            theta_ini = 0.44,    # Diagnosis (Initial)
            theta_fin = 0.9,     # Diagnosis (Final)
            phi_ini   = 0.69,    # Treatment failure (Initial)
            phi_fin   = 0.09,    # Treatment failure (Final)
            rho       = 3.25,    # Risk of reinfection
            N         = 1e5,     # Population size
            mu        = 1/70,    # Mortality rate
            delta     = 2,       # Treatment duration (6 months)
        )
        self.update_pars(**kwargs)

        # Compartments (scalars, not per-agent states)
        self.SUS = 0
        self.INF = 0
        self.CLE = 0
        self.REC = 0
        self.MIN = 0
        self.SUB = 0
        self.CLN = 0
        self.TXT = 0
        self.TRE = 0
        return

    def force_mutb(self, t):
        """ Forcing function for TB mortality (piecewise linear, constant extrapolation) """
        p = self.pars
        return np.interp(t, [1500, 1999, 2020], [p.mutb_ini, p.mutb_ini, p.mutb_fin])

    def force_theta(self, t):
        """ Forcing function for diagnosis rate """
        p = self.pars
        return np.interp(t, [1500, 1999, 2020], [p.theta_ini, p.theta_ini, p.theta_fin])

    def force_phi(self, t):
        """ Forcing function for treatment failure rate """
        p = self.pars
        return np.interp(t, [1500, 1999, 2020], [p.phi_ini, p.phi_ini, p.phi_fin])

    def init_post(self):
        """ Set initial conditions """
        super().init_post()
        p = self.pars
        init_cln = 1e3
        self.SUS = p.N - init_cln
        self.CLN = init_cln
        return

    def step(self):
        """ Euler integration of the ODE system """
        p = self.pars
        t = self.now
        dt = float(self.dt)

        # Time-varying parameters
        mutb  = self.force_mutb(t)
        theta = self.force_theta(t)
        phi   = self.force_phi(t)
        foi   = (p.beta / p.N) * (p.kappa * self.SUB + self.CLN)  # Force of infection

        # Compute derivatives (matching R ODE)
        dSUS = (p.mu * p.N) + (mutb * self.CLN) - foi * self.SUS - p.mu * self.SUS
        dINF = foi * (self.SUS + self.CLE + p.pi * self.REC + p.rho * self.TRE) - p.infcle * self.INF - p.infmin * self.INF - p.infsub * self.INF - p.mu * self.INF
        dCLE = p.infcle * self.INF - foi * self.CLE - p.mu * self.CLE
        dREC = p.minrec * self.MIN - foi * (p.pi * self.REC) - p.mu * self.REC
        dMIN = p.infmin * self.INF + p.submin * self.SUB - p.minrec * self.MIN - p.minsub * self.MIN - p.mu * self.MIN
        dSUB = p.infsub * self.INF + p.minsub * self.MIN + p.clnsub * self.CLN - p.submin * self.SUB - p.subcln * self.SUB - p.mu * self.SUB
        dCLN = p.subcln * self.SUB - p.clnsub * self.CLN - theta * self.CLN + phi * self.TXT - mutb * self.CLN - p.mu * self.CLN
        dTXT = theta * self.CLN - phi * self.TXT - p.delta * self.TXT - p.mu * self.TXT
        dTRE = p.delta * self.TXT - foi * (p.rho * self.TRE) - p.mu * self.TRE

        # Euler update
        self.SUS += dSUS * dt
        self.INF += dINF * dt
        self.CLE += dCLE * dt
        self.REC += dREC * dt
        self.MIN += dMIN * dt
        self.SUB += dSUB * dt
        self.CLN += dCLN * dt
        self.TXT += dTXT * dt
        self.TRE += dTRE * dt
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.define_results(
            ss.Result('SUS', label='Susceptible'),
            ss.Result('INF', label='Infected'),
            ss.Result('CLE', label='Cleared'),
            ss.Result('REC', label='Recovered'),
            ss.Result('MIN', label='Minimal'),
            ss.Result('SUB', label='Subclinical'),
            ss.Result('CLN', label='Clinical'),
            ss.Result('TXT', label='On treatment'),
            ss.Result('TRE', label='Treated'),
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
        self.results['SUS'][ti] = self.SUS
        self.results['INF'][ti] = self.INF
        self.results['CLE'][ti] = self.CLE
        self.results['REC'][ti] = self.REC
        self.results['MIN'][ti] = self.MIN
        self.results['SUB'][ti] = self.SUB
        self.results['CLN'][ti] = self.CLN
        self.results['TXT'][ti] = self.TXT
        self.results['TRE'][ti] = self.TRE

        # Derived quantities
        tb_prev = self.SUB + self.CLN
        self.results['TBc'][ti] = tb_prev
        self.results['Mor'][ti] = self.force_mutb(self.now) * self.CLN
        self.results['Dxs'][ti] = self.force_theta(self.now) * self.CLN
        self.results['Spr'][ti] = self.SUB / tb_prev if tb_prev > 0 else 0
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
            sc.figlayout()
        return ss.return_fig(fig)
  