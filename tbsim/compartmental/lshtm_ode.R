# Compartmental TB model (LSHTM ODE).
#
# State and parameter names are aligned with the agent-based tbsim.TB class.
# CLEARED is split into three sub-states (CLEARED, RECOVERED, TREATED) so the
# pathway-specific reinfection multiplier can be applied. Treatment dynamics
# (theta, phi, delta) are off by default; enable explicitly to model a
# post-treatment-era period.
#
# Originally adapted from Alvaro Schwalb's ACF model:
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

# Parameters (aligned with tbsim.TB defaults; treatment off by default)
parms = c(
  # Transmission
  beta        = 9,    # Contact (per person/year). Different semantics than tbsim.TB beta.
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
  sym_dead = 0.34, # SYMPTOMATIC -> DEAD (constant)

  # Treatment (off by default; set theta > 0 to enable)
  theta = 0.0, # Diagnosis rate, SYMPTOMATIC -> TREATMENT
  phi   = 0.0, # Treatment failure, TREATMENT -> SYMPTOMATIC
  delta = 2)   # Treatment completion, TREATMENT -> TREATED (6-month duration)

# Define the model
ode <- function(parms, start_time = 1500, end_time = 2020) {

  # Static parameters
  N <- 1e5  # Population size (held constant)
  mu <- 1/70 # Background mortality rate

  des <- function(time, state, parms) {

    with(as.list(c(state, parms)), {

      foi <- (beta / N) * ((trans_asymp * ASYMPTOMATIC) + SYMPTOMATIC)

      dSUSCEPTIBLE    <- (mu * N) + (sym_dead * SYMPTOMATIC) - (foi * SUSCEPTIBLE) - (mu * SUSCEPTIBLE)
      dINFECTION      <- (foi * (SUSCEPTIBLE +
                                 (rr_reinfection_cleared * CLEARED) +
                                 (rr_reinfection_rec     * RECOVERED) +
                                 (rr_reinfection_treat   * TREATED))) -
                         ((inf_cle + inf_non + inf_asy + mu) * INFECTION)
      dCLEARED        <- (inf_cle * INFECTION) - (foi * rr_reinfection_cleared * CLEARED) - (mu * CLEARED)
      dRECOVERED      <- (non_rec * NON_INFECTIOUS) - (foi * rr_reinfection_rec * RECOVERED) - (mu * RECOVERED)
      dNON_INFECTIOUS <- (inf_non * INFECTION) + (asy_non * ASYMPTOMATIC) -
                         ((non_rec + non_asy + mu) * NON_INFECTIOUS)
      dASYMPTOMATIC   <- (inf_asy * INFECTION) + (non_asy * NON_INFECTIOUS) + (sym_asy * SYMPTOMATIC) -
                         ((asy_non + asy_sym + mu) * ASYMPTOMATIC)
      dSYMPTOMATIC    <- (asy_sym * ASYMPTOMATIC) - (sym_asy * SYMPTOMATIC) -
                         (theta * SYMPTOMATIC) + (phi * TREATMENT) -
                         (sym_dead * SYMPTOMATIC) - (mu * SYMPTOMATIC)
      dTREATMENT      <- (theta * SYMPTOMATIC) - (phi * TREATMENT) - (delta * TREATMENT) - (mu * TREATMENT)
      dTREATED        <- (delta * TREATMENT) - (foi * rr_reinfection_treat * TREATED) - (mu * TREATED)

      return(list(c(
        dSUSCEPTIBLE, dINFECTION, dCLEARED, dRECOVERED, dNON_INFECTIOUS,
        dASYMPTOMATIC, dSYMPTOMATIC, dTREATMENT, dTREATED),
        TBc = (ASYMPTOMATIC + SYMPTOMATIC),                                   # TB prevalence
        Mor = (sym_dead * SYMPTOMATIC),                                       # TB mortality
        Dxs = (theta * SYMPTOMATIC),                                          # TB notifications
        Spr = ifelse((ASYMPTOMATIC + SYMPTOMATIC) > 0,                        # Proportion asymptomatic
                     ASYMPTOMATIC / (ASYMPTOMATIC + SYMPTOMATIC), 0)))
    })
  }

  yini <- c(SUSCEPTIBLE    = 1e5 - 1e3,
            INFECTION      = 0,
            CLEARED        = 0,
            RECOVERED      = 0,
            NON_INFECTIOUS = 0,
            ASYMPTOMATIC   = 0,
            SYMPTOMATIC    = 1e3,
            TREATMENT      = 0,
            TREATED        = 0)

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
