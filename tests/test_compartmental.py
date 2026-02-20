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
    
    def __init__(self, pars=default_pars, **kwargs):
        self.pars = pars
        self.pars.update(kwargs)
        return

    def run(self, end_time=2020):
        p = self.pars

        # Forcing functions (piecewise linear interpolation, constant extrapolation)
        # Equivalent to R's approxfun(method="linear", rule=2)
        mutb_times  = [1500, 1999, 2020]
        mutb_vals   = [p['mutb_ini'],  p['mutb_ini'],  p['mutb_fin']]
        theta_times = [1500, 1999, 2020]
        theta_vals  = [p['theta_ini'], p['theta_ini'], p['theta_fin']]
        phi_times   = [1500, 1999, 2020]
        phi_vals    = [p['phi_ini'],   p['phi_ini'],   p['phi_fin']]

        def force_mutb(t):  return np.interp(t, mutb_times, mutb_vals)
        def force_theta(t): return np.interp(t, theta_times, theta_vals)
        def force_phi(t):   return np.interp(t, phi_times, phi_vals)

        # ODE system
        def des(state, t):
            SUS, INF, CLE, REC, MIN, SUB, CLN, TXT, TRE = state

            mutb  = force_mutb(t)
            theta = force_theta(t)
            phi   = force_phi(t)
            foi   = (p['beta'] / N) * (p['kappa'] * SUB + CLN)  # Force of infection

            dSUS = (mu * N) + (mutb * CLN) - foi * SUS - mu * SUS
            dINF = foi * (SUS + CLE + p['pi'] * REC + p['rho'] * TRE) - p['infcle'] * INF - p['infmin'] * INF - p['infsub'] * INF - mu * INF
            dCLE = p['infcle'] * INF - foi * CLE - mu * CLE
            dREC = p['minrec'] * MIN - foi * (p['pi'] * REC) - mu * REC
            dMIN = p['infmin'] * INF + p['submin'] * SUB - p['minrec'] * MIN - p['minsub'] * MIN - mu * MIN
            dSUB = p['infsub'] * INF + p['minsub'] * MIN + p['clnsub'] * CLN - p['submin'] * SUB - p['subcln'] * SUB - mu * SUB
            dCLN = p['subcln'] * SUB - p['clnsub'] * CLN - theta * CLN + phi * TXT - mutb * CLN - mu * CLN
            dTXT = theta * CLN - phi * TXT - delta * TXT - mu * TXT
            dTRE = delta * TXT - foi * (p['rho'] * TRE) - mu * TRE

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
        results['TBc'] = results.SUB + results.CLN                    # TB prevalence (per 100k)
        results['Mor'] = mutb_arr * results.CLN                       # TB mortality (per 100k)
        results['Dxs'] = theta_arr * results.CLN                      # TB notifications (per 100k)
        results['Spr'] = results.SUB / (results.SUB + results.CLN)    # Proportion subclinical TB (%)

        self.results = results
        return results
