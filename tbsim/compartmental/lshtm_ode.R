# Direct copy of Alvaro Schwalb's ACF model, from:
# https://github.com/aschwalbc/ACF-VN/blob/main/scripts/02_fit.R#L64

# Install dependencies if needed
if (!requireNamespace("deSolve", quietly = TRUE)) {
  install.packages("deSolve", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
}
library(deSolve) # Solvers for ordinary differential equations
library(ggplot2) # Plotting

# Parameters
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

# Define the model
ode <- function(parms, start_time = 1500, end_time = 2020) {

  # Static parameters
  N <- 1e5 # Population size
  mu <- 1/70 # Age expectancy adult
  delta <- 2 # Treatment duration (6 months)

  # Function parameters
  forcer_mutb <- matrix(c(start_time, parms['mutb_ini'], 1999, parms['mutb_ini'], 2020, parms['mutb_fin']), ncol = 2, byrow = TRUE)
  force_func_mutb <- approxfun(x = forcer_mutb[,1], y = forcer_mutb[,2], method = "linear", rule = 2)

  forcer_theta <- matrix(c(start_time, parms['theta_ini'], 1999, parms['theta_ini'], 2020, parms['theta_fin']), ncol = 2, byrow = TRUE)
  force_func_theta <- approxfun(x = forcer_theta[,1], y = forcer_theta[,2], method = "linear", rule = 2)

  forcer_phi <- matrix(c(start_time, parms['phi_ini'], 1999, parms['phi_ini'], 2020, parms['phi_fin']), ncol = 2, byrow = TRUE)
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

  times <- seq(start_time, end_time, by = 1)
  out <- deSolve::ode(yini, times, des, parms)
  return(out)
}

# Run ODE model
out <- ode(parms)
times <- out[, 1]
states <- out[, -1]
state_names <- colnames(states)

# Plot results
df <- data.frame(
  time = rep(times, times = ncol(states)),
  variable = rep(state_names, each = length(times)),
  value = c(states)
)
plt <- ggplot(df, aes(x = time, y = value)) +
  geom_line() +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +
  labs(x = "Time", y = NULL) +
  theme_minimal()

print(plt)