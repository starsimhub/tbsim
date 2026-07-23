# =============================================================================
# Two-strain compartmental TB model (deterministic ODE).
#
# Extends the single-strain LSHTM spectrum-of-disease model (lshtm_ode.R, in this
# folder) into a two-strain system:
#   strain A = treatment-susceptible, strain B = treatment-resistant.
#
# Every "question of interest" is a parameter or mode switch on this one system;
# the defaults reproduce the intended baseline. The Python port is two_strain_ode.py
# (tbsim.compartmental.TwoStrainODE), against which tbsim.TBResistant is validated.
#
# State names map to the model symbols as follows:
#   SUS = S, CLE = C, REC = R, TRD = W                  (strain-agnostic)
#   L_*  = L^s  (INFECTION / latent)                    s in {A, B, AB}
#   N_*  = N^s  (NON_INFECTIOUS)
#   AS_* = A^s  (ASYMPTOMATIC, infectious, weight kappa)
#   SY_* = Y^s  (SYMPTOMATIC,  infectious, weight 1)
#   TA_* = T^{s,a}  (on treatment, initiated from ASYMPTOMATIC)
#   TY_* = T^{s,y}  (on treatment, initiated from SYMPTOMATIC)
#   DTH  = cumulative deaths accumulator (NOT part of the conserved N)
#
# Single-letter names S/C/R/T/W/F are avoided because some collide with R
# built-ins inside with(as.list(...)).
# =============================================================================

if (!requireNamespace("deSolve", quietly = TRUE)) {
  install.packages("deSolve", repos = "https://cloud.r-project.org")
}
library(deSolve)

# Canonical state ordering. yini and the derivative list are built in this order.
TWO_STRAIN_STATES <- c(
  "SUS", "CLE", "REC", "TRD",
  "L_A",  "L_B",  "L_AB",
  "N_A",  "N_B",  "N_AB",
  "AS_A", "AS_B", "AS_AB",
  "SY_A", "SY_B", "SY_AB",
  "TA_A", "TA_B", "TA_AB",
  "TY_A", "TY_B", "TY_AB",
  "DTH"
)

# -----------------------------------------------------------------------------
# Default parameters (South-Africa 2-strain calibration).
#
# Natural-history rates inherit the LSHTM / tbsim.TB defaults. Mode switches are
# encoded as 0/1 numeric indicators so the whole vector stays numeric and can be
# unpacked with with(as.list(c(state, parms))). Use two_strain_modes() for a
# human-readable way to set them.
# -----------------------------------------------------------------------------
two_strain_defaults <- function() {
  c(
    # --- Population / demography ---
    N  = 1e5,      # total population (held constant by recycling deaths -> births)
    mu = 1 / 70,   # background mortality (per year)

    # --- Transmission ---
    # Calibrated to South Africa: beta = 17.3 gives ~400/100k/yr active-TB
    # incidence under the treatment-on baseline below. (Active-TB prevalence then
    # comes out ~0.25%, lower than survey ~0.5%, because the high treatment rate
    # shortens the infectious period; incidence is the prioritized target. The
    # original single-strain value of 9 gives R0 < 1 once treatment is on.)
    beta        = 16.45, # effective contact rate (frequency-dependent, beta/N)
    trans_asymp = 0.82,  # rel. infectiousness ASYMPTOMATIC vs SYMPTOMATIC (kappa)

    # --- Natural-history rates (per year) ---
    inf_cle = 1.90, # INFECTION      -> CLEARED
    inf_non = 0.16, # INFECTION      -> NON_INFECTIOUS
    inf_asy = 0.06, # INFECTION      -> ASYMPTOMATIC
    non_rec = 0.18, # NON_INFECTIOUS -> RECOVERED
    non_asy = 0.25, # NON_INFECTIOUS -> ASYMPTOMATIC
    asy_non = 1.66, # ASYMPTOMATIC   -> NON_INFECTIOUS (reversion)
    asy_sym = 0.88, # ASYMPTOMATIC   -> SYMPTOMATIC
    sym_asy = 0.54, # SYMPTOMATIC    -> ASYMPTOMATIC (reversion)
    sym_dead = 0.34, # SYMPTOMATIC   -> DEAD (TB mortality)

    # --- Reinfection multipliers on FOI (rho_*) for the cleared sub-states ---
    rr_reinfection_cleared = 1.00, # CLEARED   (rho_C)
    rr_reinfection_rec     = 0.21, # RECOVERED (rho_R)
    rr_reinfection_treat   = 3.15, # TREATED   (rho_W)

    # --- Strain fitness (transmission) ---
    fit_a = 1.0,    # r_a
    fit_b = 0.575,  # r_b (~43% fitness cost; calibrated so transmitted resistance
                    #      holds the resistant fraction near the SA target ~4%)

    # --- Superinfection susceptibility, rel. to fully susceptible (sigma_*) ---
    rr_reinfection_inf = 1, # sigma_L: susceptibility of mono L to 2nd strain
    rr_reinfection_non = 1, # sigma_N: of mono N
    rr_reinfection_asy = 0, # sigma_A: of mono A (off by default)
    rr_reinfection_sym = 0, # sigma_Y: of mono Y (off by default)

    # --- Progression ---
    p_multi       = 1, # prob. both strains co-progress at L/N -> A (1 = no bottleneck)
    rr_prog_super = 1, # psi:   progression-rate multiplier for AB sources
    rr_clear_super = 1, # omega: clearance/recovery multiplier for AB (off by default)

    # --- Treatment (ON in the resistance-model baseline; strain B's advantage
    #     only manifests under treatment. Set both to 0 to recover the
    #     treatment-free single-strain natural history.) ---
    r_treat_asym = 0.01,  # treatment initiation rate from ASYMPTOMATIC (~1%/yr prob.)
    r_treat_sym  = 1.204, # treatment initiation rate from SYMPTOMATIC (~70%/yr prob., SA)
    delta        = 2.0,   # treatment exit (completion) rate (~6-month duration)
    eff_a        = 0.75,  # per-strain cure prob., strain A (e_a; ~10% tx failure)
    eff_b        = 0.25,  # per-strain cure prob., strain B (e_b; ~30% tx failure, e_a > e_b)

    # --- Resistance acquisition (low: most resistance is transmitted, not acquired) ---
    q_treat = 0.05,   # prob. acquire resistance per A-surviving tx failure (q, always replacement)
    q_prog  = 0.0001, # prob. de novo resistance per L^A -> N and L^A -> A event (q_p)

    # --- Mode switches (numeric indicators) ---
    transmission_independent = 0, # 0 = bottleneck (default), 1 = independent       (Q5)
    prog_resist_mix          = 1, # 1 = de novo q_p -> AB (mixed, default), 0 = -> B (replacement) (Q4)
    prog_select_fitness      = 0  # 0 = random (1/2,1/2) selection, 1 = fitness-weighted (g_A,g_B) (Q2b)
  )
}

# -----------------------------------------------------------------------------
# Convenience: set mode switches by name. Returns the numeric indicator overrides
# to splice into a parameter vector, e.g.
#   p <- two_strain_defaults(); p[names(m)] <- (m <- two_strain_modes(...))
# -----------------------------------------------------------------------------
two_strain_modes <- function(transmission = c("bottleneck", "independent"),
                             prog_resist  = c("mixed", "replacement"),
                             prog_select  = c("random", "fitness")) {
  transmission <- match.arg(transmission)
  prog_resist  <- match.arg(prog_resist)
  prog_select  <- match.arg(prog_select)
  c(
    transmission_independent = as.numeric(transmission == "independent"),
    prog_resist_mix          = as.numeric(prog_resist == "mixed"),
    prog_select_fitness      = as.numeric(prog_select == "fitness")
  )
}

# -----------------------------------------------------------------------------
# Initial conditions. Supply seed counts by state name; SUS absorbs the balance
# so the 22 dynamic compartments sum to N. DTH starts at 0.
#   two_strain_init(N = 1e5, SY_A = 1e3)   # single A-strain symptomatic seed
# -----------------------------------------------------------------------------
two_strain_init <- function(N = 1e5, ...) {
  seeds <- c(...)
  yini <- setNames(numeric(length(TWO_STRAIN_STATES)), TWO_STRAIN_STATES)
  if (length(seeds)) {
    unknown <- setdiff(names(seeds), TWO_STRAIN_STATES)
    if (length(unknown)) stop("Unknown state(s) in seed: ", paste(unknown, collapse = ", "))
    yini[names(seeds)] <- seeds
  }
  dynamic <- setdiff(TWO_STRAIN_STATES, "DTH")
  yini["SUS"] <- 0
  yini["SUS"] <- N - sum(yini[dynamic])
  if (yini["SUS"] < 0) stop("Seeds exceed N; SUS would be negative.")
  yini
}

# -----------------------------------------------------------------------------
# Derivative function. Returns the 23 derivatives in
# TWO_STRAIN_STATES order plus a set of observable outputs.
# -----------------------------------------------------------------------------
two_strain_des <- function(time, state, parms) {
  with(as.list(c(state, parms)), {

    # --- Strain fitness summaries (Section 2) ---
    g_A   <- fit_a / (fit_a + fit_b)   # transmitted-strain fraction, A
    g_B   <- fit_b / (fit_a + fit_b)   # transmitted-strain fraction, B
    r_max <- max(fit_a, fit_b)

    # --- Infectious pressure by strain content (Section 3) ---
    P_A  <- trans_asymp * AS_A  + SY_A
    P_B  <- trans_asymp * AS_B  + SY_B
    P_AB <- trans_asymp * AS_AB + SY_AB

    # --- Force of infection per strain (Section 3) ---
    if (transmission_independent == 1) {
      # Independent: each strain transmits at its own full rate from AB sources.
      lambda_A <- (beta / N) * fit_a * (P_A + P_AB)
      lambda_B <- (beta / N) * fit_b * (P_B + P_AB)
    } else {
      # Bottleneck (default): AB source transmits at r_max, split g_A : g_B.
      lambda_A <- (beta / N) * (fit_a * P_A + g_A * r_max * P_AB)
      lambda_B <- (beta / N) * (fit_b * P_B + g_B * r_max * P_AB)
    }
    Lambda <- lambda_A + lambda_B

    # --- Susceptible pool weighted by reinfection protection (Section 7) ---
    U <- SUS +
      rr_reinfection_cleared * CLE +
      rr_reinfection_rec     * REC +
      rr_reinfection_treat   * TRD

    # --- Births recycle all deaths to hold N constant (Section 7) ---
    sumY     <- SY_A + SY_B + SY_AB
    B_birth  <- mu * N + sym_dead * sumY

    # --- Progression bottleneck operator at L/N -> A (Section 5) ---
    Pi_AB <- rr_prog_super * inf_asy * L_AB + rr_prog_super * non_asy * N_AB
    if (prog_select_fitness == 1) { h_A <- g_A;  h_B <- g_B }  # Q2b fitness-weighted
    else                          { h_A <- 0.5;  h_B <- 0.5 }  # default random
    G_AB <- p_multi * Pi_AB
    G_A  <- (1 - p_multi) * h_A * Pi_AB
    G_B  <- (1 - p_multi) * h_B * Pi_AB

    # --- De novo resistance at L^A progression (Section 5.1, Q4) ---
    mix <- prog_resist_mix       # 1 = mixed (-> AB), 0 = replacement (-> B)
    rep <- 1 - mix

    # --- Treatment-outcome operator (Section 6) ---
    # Per-strain cures are independent; failures return to origin disease state.
    # Among A-surviving failures, resistance acquired w.p. q_treat, always A->B.
    ea <- eff_a; eb <- eff_b; q <- q_treat
    # pi(m -> s): exit probabilities from on-treatment content m to outcome s.
    pi_A_to_A   <- (1 - ea) * (1 - q)
    pi_A_to_B   <- (1 - ea) * q
    pi_B_to_B   <- (1 - eb)
    pi_AB_to_A  <- (1 - ea) * eb * (1 - q)
    pi_AB_to_B  <- ea * (1 - eb) + (1 - ea) * eb * q + (1 - ea) * (1 - eb) * q
    pi_AB_to_AB <- (1 - ea) * (1 - eb) * (1 - q)
    # Treatment-failure return inflows Phi^s_o, origin o in {a (asymp), y (symp)}.
    Phi_A_a  <- delta * (pi_A_to_A * TA_A + pi_AB_to_A * TA_AB)
    Phi_B_a  <- delta * (pi_A_to_B * TA_A + pi_B_to_B * TA_B + pi_AB_to_B * TA_AB)
    Phi_AB_a <- delta * (pi_AB_to_AB * TA_AB)
    Phi_A_y  <- delta * (pi_A_to_A * TY_A + pi_AB_to_A * TY_AB)
    Phi_B_y  <- delta * (pi_A_to_B * TY_A + pi_B_to_B * TY_B + pi_AB_to_B * TY_AB)
    Phi_AB_y <- delta * (pi_AB_to_AB * TY_AB)
    # Cures flow to TRD (W).
    cure_flow <- delta * (ea * (TA_A + TY_A) + eb * (TA_B + TY_B) + ea * eb * (TA_AB + TY_AB))

    # =====================  ODE system (Section 7)  =====================

    # --- Strain-agnostic ---
    dSUS <- B_birth - Lambda * SUS - mu * SUS
    dCLE <- inf_cle * (L_A + L_B) + rr_clear_super * inf_cle * L_AB -
            rr_reinfection_cleared * Lambda * CLE - mu * CLE
    dREC <- non_rec * (N_A + N_B) + rr_clear_super * non_rec * N_AB -
            rr_reinfection_rec * Lambda * REC - mu * REC
    dTRD <- cure_flow - rr_reinfection_treat * Lambda * TRD - mu * TRD

    # --- Latent (INFECTION) ---
    dL_A  <- lambda_A * U -
             (inf_cle + inf_non + inf_asy + rr_reinfection_inf * lambda_B + mu) * L_A
    dL_B  <- lambda_B * U -
             (inf_cle + inf_non + inf_asy + rr_reinfection_inf * lambda_A + mu) * L_B
    dL_AB <- rr_reinfection_inf * (lambda_B * L_A + lambda_A * L_B) -
             (rr_clear_super * inf_cle + inf_non + rr_prog_super * inf_asy + mu) * L_AB

    # --- Non-infectious ---
    dN_A  <- inf_non * (1 - q_prog) * L_A + asy_non * AS_A -
             (non_rec + non_asy + rr_reinfection_non * lambda_B + mu) * N_A
    dN_B  <- inf_non * L_B + rep * inf_non * q_prog * L_A + asy_non * AS_B -
             (non_rec + non_asy + rr_reinfection_non * lambda_A + mu) * N_B
    dN_AB <- inf_non * L_AB + mix * inf_non * q_prog * L_A + asy_non * AS_AB +
             rr_reinfection_non * (lambda_B * N_A + lambda_A * N_B) -
             (rr_clear_super * non_rec + rr_prog_super * non_asy + mu) * N_AB

    # --- Asymptomatic ---
    dAS_A  <- inf_asy * (1 - q_prog) * L_A + non_asy * N_A + sym_asy * SY_A +
              G_A + Phi_A_a -
              (asy_non + asy_sym + r_treat_asym + rr_reinfection_asy * lambda_B + mu) * AS_A
    dAS_B  <- inf_asy * L_B + rep * inf_asy * q_prog * L_A + non_asy * N_B + sym_asy * SY_B +
              G_B + Phi_B_a -
              (asy_non + asy_sym + r_treat_asym + rr_reinfection_asy * lambda_A + mu) * AS_B
    dAS_AB <- sym_asy * SY_AB + G_AB + mix * inf_asy * q_prog * L_A +
              rr_reinfection_asy * (lambda_B * AS_A + lambda_A * AS_B) + Phi_AB_a -
              (asy_non + rr_prog_super * asy_sym + r_treat_asym + mu) * AS_AB

    # --- Symptomatic ---
    dSY_A  <- asy_sym * AS_A + Phi_A_y -
              (sym_asy + sym_dead + r_treat_sym + rr_reinfection_sym * lambda_B + mu) * SY_A
    dSY_B  <- asy_sym * AS_B + Phi_B_y -
              (sym_asy + sym_dead + r_treat_sym + rr_reinfection_sym * lambda_A + mu) * SY_B
    dSY_AB <- rr_prog_super * asy_sym * AS_AB +
              rr_reinfection_sym * (lambda_B * SY_A + lambda_A * SY_B) + Phi_AB_y -
              (sym_asy + sym_dead + r_treat_sym + mu) * SY_AB

    # --- Treatment (origin asymptomatic 'a' and symptomatic 'y') ---
    dTA_A  <- r_treat_asym * AS_A  - (delta + mu) * TA_A
    dTA_B  <- r_treat_asym * AS_B  - (delta + mu) * TA_B
    dTA_AB <- r_treat_asym * AS_AB - (delta + mu) * TA_AB
    dTY_A  <- r_treat_sym  * SY_A  - (delta + mu) * TY_A
    dTY_B  <- r_treat_sym  * SY_B  - (delta + mu) * TY_B
    dTY_AB <- r_treat_sym  * SY_AB - (delta + mu) * TY_AB

    # --- Cumulative deaths accumulator (recycled into births; not part of N) ---
    dynamic_sum <- SUS + CLE + REC + TRD +
      L_A + L_B + L_AB + N_A + N_B + N_AB +
      AS_A + AS_B + AS_AB + SY_A + SY_B + SY_AB +
      TA_A + TA_B + TA_AB + TY_A + TY_B + TY_AB
    dDTH <- mu * dynamic_sum + sym_dead * sumY

    # --- Observable outputs ---
    active      <- AS_A + AS_B + AS_AB + SY_A + SY_B + SY_AB
    active_B    <- AS_B + SY_B + AS_AB + SY_AB        # active TB carrying B
    active_AB   <- AS_AB + SY_AB                      # active superinfections
    latent_all  <- L_A + L_B + L_AB
    # All current TB infections (INFECTION + NON_INFECTIOUS + active), and the
    # B-carrying (resistant) subset. Excludes cleared (C/R/W) and on-treatment.
    infected     <- L_A + L_B + L_AB + N_A + N_B + N_AB + active
    infected_B   <- L_B + L_AB + N_B + N_AB + active_B   # any B-carrying infection
    infected_AB  <- L_AB + N_AB + active_AB              # any superinfection
    incid_active <- inf_asy * (L_A + L_B + L_AB) + non_asy * (N_A + N_B + N_AB) # flux into A
    tb_deaths   <- sym_dead * sumY
    notif       <- r_treat_asym * (AS_A + AS_B + AS_AB) + r_treat_sym * sumY    # flux into T
    # Resistance-origin fluxes (Section 11): de novo, treatment-acquired, transmitted.
    flux_denovo  <- q_prog * (inf_non + inf_asy) * L_A
    flux_txacq   <- delta * (pi_A_to_B * (TA_A + TY_A) +
                             ((1 - ea) * eb * q + (1 - ea) * (1 - eb) * q) * (TA_AB + TY_AB))
    flux_transB  <- lambda_B * U + rr_reinfection_inf * lambda_A * L_B  # new B infections (mono primary + superinf of L^B)

    derivs <- c(
      dSUS, dCLE, dREC, dTRD,
      dL_A,  dL_B,  dL_AB,
      dN_A,  dN_B,  dN_AB,
      dAS_A, dAS_B, dAS_AB,
      dSY_A, dSY_B, dSY_AB,
      dTA_A, dTA_B, dTA_AB,
      dTY_A, dTY_B, dTY_AB,
      dDTH
    )

    list(
      derivs,
      prev_active   = active / N,
      frac_resist   = ifelse(active > 0, active_B / active, 0),
      frac_super    = ifelse(active > 0, active_AB / active, 0),
      prev_infected   = infected / N,
      frac_resist_all = ifelse(infected > 0, infected_B / infected, 0),
      frac_super_all  = ifelse(infected > 0, infected_AB / infected, 0),
      latent_all    = latent_all,
      incid_active  = incid_active,
      tb_deaths     = tb_deaths,
      notif         = notif,
      flux_denovo   = flux_denovo,
      flux_txacq    = flux_txacq,
      flux_transB   = flux_transB,
      lambda_A      = lambda_A,
      lambda_B      = lambda_B
    )
  })
}

# -----------------------------------------------------------------------------
# Solver wrapper. Returns the deSolve output (a deSolve matrix / data.frame).
#   run_two_strain()                                  # defaults, A+B seeded
#   run_two_strain(parms = p, init = y0, times = ...) # custom
# -----------------------------------------------------------------------------
run_two_strain <- function(parms = two_strain_defaults(),
                           init  = NULL,
                           start_time = 1500,
                           end_time   = 2020,
                           by = 1,
                           as_data_frame = TRUE) {
  if (is.null(init)) {
    N <- unname(parms["N"])
    init <- two_strain_init(N = N, SY_A = 1e3, SY_B = 1e1)
  }
  # Ensure state order matches the derivative output order.
  init <- init[TWO_STRAIN_STATES]
  times <- seq(start_time, end_time, by = by)
  out <- deSolve::ode(y = init, times = times, func = two_strain_des, parms = parms)
  if (as_data_frame) as.data.frame(out) else out
}

# -----------------------------------------------------------------------------
# Strain-summed view: collapse the A/B/AB triplets so the two-strain run can be
# compared compartment-for-compartment against the single-strain model.
# Returns a data.frame with the single-strain state names.
# -----------------------------------------------------------------------------
two_strain_collapse <- function(out_df) {
  with(out_df, data.frame(
    time           = time,
    SUSCEPTIBLE    = SUS,
    INFECTION      = L_A + L_B + L_AB,
    CLEARED        = CLE,
    RECOVERED      = REC,
    NON_INFECTIOUS = N_A + N_B + N_AB,
    ASYMPTOMATIC   = AS_A + AS_B + AS_AB,
    SYMPTOMATIC    = SY_A + SY_B + SY_AB,
    TREATMENT      = TA_A + TA_B + TA_AB + TY_A + TY_B + TY_AB,
    TREATED        = TRD
  ))
}

# -----------------------------------------------------------------------------
# Mean sojourn time per visit (years) in each infected / on-treatment state,
# = 1 / (total per-capita exit rate). At endemic equilibrium this equals the
# population mean residence time per visit (Little's law: stock / outflow).
#
# The exit rates whose superinfection terms depend on the force of infection
# (L and N always; A and Y when rr_reinfection_asy/sym > 0) are evaluated at the
# equilibrium FOI of a solved `run` if supplied; otherwise lambda = 0 (sojourn
# ignoring superinfection pressure).
#
# This is the time *per stay*. It is NOT the total time over an infection
# episode, which sums across revisits (e.g. ASYMPTOMATIC <-> NON_INFECTIOUS
# cycling) and needs the absorbing-chain fundamental matrix.
#
# NOTE: these exit rates mirror the negative (loss) coefficients in
# two_strain_des(); keep the two in sync if the model structure changes.
# -----------------------------------------------------------------------------
two_strain_sojourn <- function(parms = two_strain_defaults(), run = NULL) {
  lambda_A <- 0; lambda_B <- 0
  if (!is.null(run)) {
    last <- run[nrow(run), ]
    lambda_A <- last$lambda_A; lambda_B <- last$lambda_B
  }
  with(c(as.list(parms), list(lambda_A = lambda_A, lambda_B = lambda_B)), {
    exit <- c(
      L_A   = inf_cle + inf_non + inf_asy + rr_reinfection_inf * lambda_B + mu,
      L_B   = inf_cle + inf_non + inf_asy + rr_reinfection_inf * lambda_A + mu,
      L_AB  = rr_clear_super * inf_cle + inf_non + rr_prog_super * inf_asy + mu,
      N_A   = non_rec + non_asy + rr_reinfection_non * lambda_B + mu,
      N_B   = non_rec + non_asy + rr_reinfection_non * lambda_A + mu,
      N_AB  = rr_clear_super * non_rec + rr_prog_super * non_asy + mu,
      AS_A  = asy_non + asy_sym + r_treat_asym + rr_reinfection_asy * lambda_B + mu,
      AS_B  = asy_non + asy_sym + r_treat_asym + rr_reinfection_asy * lambda_A + mu,
      AS_AB = asy_non + rr_prog_super * asy_sym + r_treat_asym + mu,
      SY_A  = sym_asy + sym_dead + r_treat_sym + rr_reinfection_sym * lambda_B + mu,
      SY_B  = sym_asy + sym_dead + r_treat_sym + rr_reinfection_sym * lambda_A + mu,
      SY_AB = sym_asy + sym_dead + r_treat_sym + mu,
      TA    = delta + mu,   # on treatment, origin asymptomatic (any strain)
      TY    = delta + mu    # on treatment, origin symptomatic  (any strain)
    )
    data.frame(state = names(exit), exit_rate = unname(exit),
               mean_years = unname(1 / exit), row.names = NULL)
  })
}

# -----------------------------------------------------------------------------
# Fundamental matrix of the infected/on-treatment transient states.
#
# Builds the 18x18 sub-generator T (12 disease + 6 treatment states): off-
# diagonal = internal transition rates (progression, regression, superinfection,
# treatment-failure returns); diagonal = -(total exit rate). Leaks to absorbing
# states (CLEARED / RECOVERED / TREATED / DEAD) are NOT columns of T. Then
# M = (-T)^-1, where M[i, j] = expected TOTAL time spent in transient state j
# before absorption, starting from state i (summed across all revisits).
#
# FOI-dependent rates (superinfection out of L, N, and A/Y) are evaluated at the
# equilibrium force of infection of a solved `run` if supplied, else lambda = 0.
#
# NOTE: transitions mirror two_strain_des(); keep in sync with the model.
# -----------------------------------------------------------------------------
.two_strain_fundamental <- function(parms = two_strain_defaults(), run = NULL) {
  lambda_A <- 0; lambda_B <- 0
  if (!is.null(run)) {
    last <- run[nrow(run), ]
    lambda_A <- last$lambda_A; lambda_B <- last$lambda_B
  }
  st <- c("L_A", "L_B", "L_AB", "N_A", "N_B", "N_AB",
          "AS_A", "AS_B", "AS_AB", "SY_A", "SY_B", "SY_AB",
          "TA_A", "TA_B", "TA_AB", "TY_A", "TY_B", "TY_AB")
  with(c(as.list(parms), list(lambda_A = lambda_A, lambda_B = lambda_B)), {
    n <- length(st)
    Q <- matrix(0, n, n, dimnames = list(st, st))
    mix <- prog_resist_mix; repl <- 1 - mix
    ea <- eff_a; eb <- eff_b; q <- q_treat
    piAA   <- (1 - ea) * (1 - q);  piAB2 <- (1 - ea) * q
    piBB   <- (1 - eb)
    piABA  <- (1 - ea) * eb * (1 - q)
    piABB  <- ea * (1 - eb) + (1 - ea) * eb * q + (1 - ea) * (1 - eb) * q
    piABAB <- (1 - ea) * (1 - eb) * (1 - q)

    # --- internal transitions (rate from -> to) ---
    Q["L_A", "N_A"]   <- inf_non * (1 - q_prog)
    Q["L_A", "N_B"]   <- repl * inf_non * q_prog
    Q["L_A", "N_AB"]  <- mix  * inf_non * q_prog
    Q["L_A", "AS_A"]  <- inf_asy * (1 - q_prog)
    Q["L_A", "AS_B"]  <- repl * inf_asy * q_prog
    Q["L_A", "AS_AB"] <- mix  * inf_asy * q_prog
    Q["L_A", "L_AB"]  <- rr_reinfection_inf * lambda_B

    Q["L_B", "N_B"]   <- inf_non
    Q["L_B", "AS_B"]  <- inf_asy
    Q["L_B", "L_AB"]  <- rr_reinfection_inf * lambda_A

    Q["L_AB", "N_AB"]  <- inf_non
    Q["L_AB", "AS_AB"] <- rr_prog_super * inf_asy

    Q["N_A", "AS_A"]  <- non_asy
    Q["N_A", "N_AB"]  <- rr_reinfection_non * lambda_B
    Q["N_B", "AS_B"]  <- non_asy
    Q["N_B", "N_AB"]  <- rr_reinfection_non * lambda_A
    Q["N_AB", "AS_AB"] <- rr_prog_super * non_asy

    Q["AS_A", "N_A"]   <- asy_non
    Q["AS_A", "SY_A"]  <- asy_sym
    Q["AS_A", "TA_A"]  <- r_treat_asym
    Q["AS_A", "AS_AB"] <- rr_reinfection_asy * lambda_B
    Q["AS_B", "N_B"]   <- asy_non
    Q["AS_B", "SY_B"]  <- asy_sym
    Q["AS_B", "TA_B"]  <- r_treat_asym
    Q["AS_B", "AS_AB"] <- rr_reinfection_asy * lambda_A
    Q["AS_AB", "N_AB"]  <- asy_non
    Q["AS_AB", "SY_AB"] <- rr_prog_super * asy_sym
    Q["AS_AB", "TA_AB"] <- r_treat_asym

    Q["SY_A", "AS_A"]  <- sym_asy
    Q["SY_A", "TY_A"]  <- r_treat_sym
    Q["SY_A", "SY_AB"] <- rr_reinfection_sym * lambda_B
    Q["SY_B", "AS_B"]  <- sym_asy
    Q["SY_B", "TY_B"]  <- r_treat_sym
    Q["SY_B", "SY_AB"] <- rr_reinfection_sym * lambda_A
    Q["SY_AB", "AS_AB"] <- sym_asy
    Q["SY_AB", "TY_AB"] <- r_treat_sym

    # treatment failures return to origin disease state (TA->AS, TY->SY)
    Q["TA_A", "AS_A"]   <- delta * piAA
    Q["TA_A", "AS_B"]   <- delta * piAB2
    Q["TA_B", "AS_B"]   <- delta * piBB
    Q["TA_AB", "AS_A"]  <- delta * piABA
    Q["TA_AB", "AS_B"]  <- delta * piABB
    Q["TA_AB", "AS_AB"] <- delta * piABAB
    Q["TY_A", "SY_A"]   <- delta * piAA
    Q["TY_A", "SY_B"]   <- delta * piAB2
    Q["TY_B", "SY_B"]   <- delta * piBB
    Q["TY_AB", "SY_A"]  <- delta * piABA
    Q["TY_AB", "SY_B"]  <- delta * piABB
    Q["TY_AB", "SY_AB"] <- delta * piABAB

    # --- leaks to absorbing states (clearance / recovery / cure / death) ---
    leak <- setNames(numeric(n), st)
    leak["L_A"]  <- inf_cle + mu
    leak["L_B"]  <- inf_cle + mu
    leak["L_AB"] <- rr_clear_super * inf_cle + mu
    leak["N_A"]  <- non_rec + mu
    leak["N_B"]  <- non_rec + mu
    leak["N_AB"] <- rr_clear_super * non_rec + mu
    leak[c("AS_A", "AS_B", "AS_AB")] <- mu
    leak[c("SY_A", "SY_B", "SY_AB")] <- sym_dead + mu
    leak[c("TA_A", "TY_A")]   <- delta * ea + mu
    leak[c("TA_B", "TY_B")]   <- delta * eb + mu
    leak[c("TA_AB", "TY_AB")] <- delta * ea * eb + mu

    Tm <- Q
    diag(Tm) <- -(rowSums(Q) + leak)
    solve(-Tm)   # fundamental matrix M[i, j]
  })
}

# -----------------------------------------------------------------------------
# Expected total time (years) spent in each transient state per infection
# episode, by starting strain (new mono-A infection entering L_A, or mono-B
# entering L_B). Counts revisits (e.g. ASYMPTOMATIC <-> NON_INFECTIOUS cycling).
# -----------------------------------------------------------------------------
two_strain_residence <- function(parms = two_strain_defaults(), run = NULL) {
  M <- .two_strain_fundamental(parms, run)
  data.frame(state = colnames(M),
             from_new_A = unname(M["L_A", ]),
             from_new_B = unname(M["L_B", ]),
             row.names = NULL)
}

# -----------------------------------------------------------------------------
# Episode-level summary: total time (years) per infection episode in grouped
# state categories, for a new A-infection and a new B-infection. "resistant" =
# any B-carrying state; "super" = AB superinfection states.
# -----------------------------------------------------------------------------
two_strain_episode_summary <- function(parms = two_strain_defaults(), run = NULL) {
  M  <- .two_strain_fundamental(parms, run)
  st <- colnames(M)
  grp <- list(
    duration  = st,
    latent    = grep("^L_",     st, value = TRUE),
    noninf    = grep("^N_",     st, value = TRUE),
    active    = grep("^(AS|SY)_", st, value = TRUE),
    treatment = grep("^T[AY]_", st, value = TRUE),
    resistant = grep("_B$|_AB$", st, value = TRUE),
    super     = grep("_AB$",    st, value = TRUE)
  )
  agg <- function(start) vapply(grp, function(cols) sum(M[start, cols]), numeric(1))
  rbind(
    data.frame(start = "new_A", as.list(agg("L_A")), check.names = FALSE),
    data.frame(start = "new_B", as.list(agg("L_B")), check.names = FALSE)
  )
}

# -----------------------------------------------------------------------------
# Reach-probability decomposition of the residence times. For an absorbing chain
# the total time in state j per episode factors exactly as
#
#     M[i, j]  =  P(ever reach j | start i)  x  M[j, j],
#
# where M[j, j] (= time_once_arrived) is the expected total time accrued in j
# once you first arrive there -- memoryless, identical for every origin -- and
# the reach probability P(ever reach j | start i) = M[i, j] / M[j, j] carries ALL
# of the origin (strain-of-infection) dependence. Returns, per transient state:
#   time_once_arrived = M[j, j]
#   reach_from_A/B    = P(ever reach j | new mono-A / mono-B infection)
#   time_from_A/B     = M[L_A or L_B, j]  (= reach * time_once_arrived)
# -----------------------------------------------------------------------------
two_strain_reach <- function(parms = two_strain_defaults(), run = NULL) {
  M   <- .two_strain_fundamental(parms, run)
  Mjj <- diag(M)
  data.frame(
    state             = colnames(M),
    time_once_arrived = unname(Mjj),
    reach_from_A      = unname(M["L_A", ] / Mjj),
    reach_from_B      = unname(M["L_B", ] / Mjj),
    time_from_A       = unname(M["L_A", ]),
    time_from_B       = unname(M["L_B", ]),
    row.names = NULL
  )
}
