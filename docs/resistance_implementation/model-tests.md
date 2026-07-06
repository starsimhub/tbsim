# Model tests

We want to test several structural assumptions or choices for how we choose to model TB resistance as a multistrain phenomena. In lieu of using the full TBsim model to do this, we want to build a simple model of ODEs or (less ideal) a deterministic individual-based simulation that allows us to test several features.

# Basic model description

First, the TB natural history should be the same as currently exists in TBsim (see `~/active/tbsim` for more details), with a few key differences. This includes susceptible (class SUSCEPTIBLE) individuals become infected (class INFECTION) with some probability upon contact with transmissible individuals. Infected individuals can either clear this early infection (class CLEARED) or proceed to the non-infectious state (class NON_INFECTIOUS). From the non-infectious state, individuals again have a chance to resolve their infection (class RECOVERED) or proceed to asymptomatic active TB (class ASYMPTOMATIC). Asymptomatic individuals can progress to symptomatic disease (class SYMPTOMATIC). We also track states for individuals who are treated (class TREATMENT), individuals who die (class DEAD), and individuals who emigrate from the population (class REMOVED). Individuals who are successfully treated move into the class TREATED. Individuals can also move backwards (e.g., regress from asymptomatic active TB to non-infectious TB), at rates as defined in TBsim.

One difference between TBsim and the compartmental model we wish to build is the separation of the TBsim CLEARED state into three separate states. Individuals who recover from the NON_INFECTIOUS state to RECOVERED acquire protection against reinfection dictated by the parameter `rr_reinfection_rec`. Individuals who recover from the TREATMENT class to TREATED acquire protection against reinfection dictated by the parameter `rr_reinfection_treat`. Individuals who clear their infection in the INFECTED state to CLEARED acquire protection against reinfection dictated by the parameter `rr_reinfection_clear`, but this is by default set to 1 to model no protection following clearance.

Individuals are infectious only in the active TB states ASYMPTOMATIC or SYMPTOMATIC. In the simple compartmental model, we model transmission solely as a function of the number of infectious individuals multiplied by a parameter $\beta$ (the rate of effective contacts), by the fitness of the infecting strain (see below), and by any level of protecton against reinfection that exists in the target compartment. Note that we will vary which states are eligible to be infected or reinfected, as described below.

## Resistance

We wish to incorporate resistance as a two-strain model, where strain A is fully susceptible to available treatments and strain B carries resistance to treatment. Therefore, each TB state (INFECTED, NON_INFECTIOUS, ASYMPTOMATIC, SYMPTOMATIC, TREATED) above should be in triplicate: strain A only, strain B only, or strain A-B superinfection.

We assume that each strain has a defined fitness cost $r_i$ that affects the probability of transmission. When an individual is mono-infected, the rate of onward transmission per individual is simply $\beta * r_i$. When an individual is superinfected, we assume that only one strain will be transmitted. The overall rate of transmission from the superinfected individual is equal to the highest rate of transmission across both strains (in our model, where $r_a$ will typically be greater than $r_b$ reflecting greater fitness of the susceptible strain, this means overall rate of transmission from the superinfected individual will be $\beta * r_a$). We will then use the relative fitness of each strain to determine which strain is transmitted (i.e., probability that A is transmitted, conditional on infection occurring, is $\frac{r_a}{r_a + r_b}$). 

We assume that, when superinfection occurs, individuals remain in the same TB state (INFECTED, NON_INFECTIOUS, ASYMPTOMATIC, SYMPTOMATIC, TREATED). By default, we assume that only individuals in INFECTED and NON_INFECTIOUS are eligible to be reinfected, although we hope to explore the impacts of allowing superinfection to occur in ASYMPTOMATIC and SYMPTOMATIC. This should be governed by some state-specific parameters like `rr_reinfection_inf`, `rr_reinfection_non`, `rr_reinfection_asy`, `rr_reinfection_sym`.

If an individual in the INFECTION or NON-INFECTIOUS state progresses to ASYMPTOMATIC, they retain both strains with some probability `p_multi`, and otherwise each strain has an equal probability of being the single strain that progresses (i.e., we do not model fitness costs via a progression bottleneck – only a transmission bottleneck). However, as described below, we do wish to test a model variant where the same transmission fitness parameters are applied to the disease progression bottleneck when `p_multi` $<1$.

We assume that if a superinfected individual naturally (i.e., absent intervention) clears their infection (INFECTION to CLEARED) or spontaneously resolves (NON-INFECTIOUS to RECOVERED), all strains are cleared. Any natural protection acquired following clearance or recovery is strain-agnostic. Clearance rates (and other transition rates between TB states in the natural history) are not affected by superinfection in the null model, although we will test modifications of this (see below). 

We also assume a low rate of acquisition of resistance for individuals infected with strain A, representing the (random) occurrence of mutations occurring during infection. This can be modeled as a one-time probability that occurs upon progression from INFECTION to NON_INFECTIOUS or from INFECTION to ASYMPTOMATIC. We wish to model two mechanisms here: one where this random acquisition of replacement results in a mixed infection (i.e., persistence of some of the original A strain, but growth of strain B as well resulting in mixed A/B infection) and one where random acquisition leads to full switch to strain B (i.e. all strain A bacteria are replaced by strain B bacteria, for monoinfection with strain B).

## Treatment & Resistance

We define state-specific rates of treatment (e.g., `r_treat_sym`, `r_treat_asym`) that reflect different case-finding and treatment modalities. As an initial simplification, we do not vary treatment efficacy by TB state.

We assume that treatment will have lower efficacy against strain B $eff_b$ versus strain A $eff_a$, reflecting the additional resistance of strain B. For superinfected individuals, we treat cure of each strain as an independent probability (i.e. the probability of clearing both strains is $eff_a * eff_b$, the probability of clearing only A is $eff_a * (1-eff_b)$ and the probability of clearing only B is $(1-eff_a) * eff_b$). If a regimen cures/clears only 1 strain of an individual with superinfection/multi-strain disease, the individual remains in the same TB natural history state with just the surviving strain.

We define $q$ as the risk of developing resistance (i.e., moving from strain A to strain B in the two-strain model) among individuals infected with strain A with unsuccessful treatment outcome. We assume acquisition of resistance leads to replacement of strain A, although we will test modeling acquisition as an implicit superinfection (i.e., adding strain B to strain A).

# Questions of interest

We want to test the influence of several of our modeling assumptions, listed roughly in order of priority:

1. Changing rate of progression with superinfections

There is evidence that superinfection events may increase the rates of progression to active TB/symptomatic disease. What are the implications if we assume that superinfections have different rates of progression than monoinfections?

2. Adding disease progression bottleneck

What is the influence of assuming that both strains of a superinfection progress together to active TB (`p_multi=1`) vs adding in some probability that only one strain is selected? 

What is the influence of assuming that selection of the progression strain is random vs incorporates information on the relative fitness of each strain?

3. Allowing superinfection only in infected vs later disease states

First, we want to look at the implications of allowing superinfection to occur only for individuals in the INFECTED state vs INFECTED and NON_INFECTIOUS.

Secondary to this, we want to consider what would happen if we allow superinfections during active TB states (ASYMPTOMATIC/SYMPTOMATIC), particularly when treatment is applied.

4. Treating random acquisition of resistance as a replacement event vs superinfection

As a default, we assume that the random acquisition of resistance (i.e., not following treamtent failure) results in a mixed strain A-strain B superinfection. What is the influence of assuming that this random acquisition of resistance results in strain replacement (i.e., strain A becomes strain B)?

We assume that, due to within-host bottleneck effects, resistance acquired following treatment failure will always results in a switch of all strain A bacteria to strain B. This means that a superinfected individual that fails treatment and acquires resistance could 'reduce' from a strain A-B infection to strain B infection only.

5. Transmission bottleneck vs independence

By default, we plan to model a transmission bottleneck (that is, using the maximum probability of infection across both strains to model the probability of any infection occurring, then selecting which strain is transmitted based on relative fitness). What are the implications of instead modeling transmission of each strain as independent transmission probabilities?

6. [DEPRIORITIZED] Allowing superinfection with 2 identical strains

This test will not be completed due to additional modeling complexity require. This is retained for record-keeping purposes only.

In the default 2-strain model, we implicitly ignore superinfections with the identical strain (i.e., an individual already infected with strain A and B might be biologically eligible for another infection by strain B, but implicitly in the model this would do nothing to affect their TB state). What if we did track the number of times an individual was infected, and allowed that to modify the relative transmission potentials of each strain? Note this would presumalby be more easily accomplished with a deterministic individual-based model, rather than a compartmental model.

# Analytic Plan

## 0. Overview and relationship to existing code

We extend the existing single-strain compartmental model in `~/active/tbsim/tbsim/compartmental/lshtm_ode.py` (`TB_ODE` / `TB_SS`) — which already splits `CLEARED` into `CLEARED / RECOVERED / TREATED` with pathway-specific reinfection multipliers — into a **two-strain** model. Strain **A** is treatment-susceptible; strain **B** is treatment-resistant. We keep the LSHTM "spectrum of disease" natural history (states and default rates) and the frequency-dependent force of infection ($\beta/N$). The only structural additions are (i) strain stratification of the infection/disease/treatment states, (ii) superinfection flows, (iii) a transmission bottleneck, (iv) a progression bottleneck, (v) de novo resistance at progression out of `INFECTION`, and (vi) a strain-resolved treatment-outcome operator.

The model is deliberately built so that **every question of interest is a parameter or mode switch on one master system of equations** (Section 10). The default ("null") settings recover the intended baseline; each question is answered by sweeping one switch.

**Reduction check (verification target):** with a single seeded strain, $q=q_p=0$, no superinfection, $p_\text{multi}=1$, and identical strain parameters, the strain-summed dynamics must reproduce the existing `TB_ODE` trajectory (Section 8).

---

## 1. Compartments

Let strain content be $s \in \{A, B, AB\}$ where $AB$ denotes a superinfection carrying both strains.

**Strain-agnostic (4):**

| Symbol | State | Notes |
|---|---|---|
| $S$ | SUSCEPTIBLE | never (currently) infected |
| $C$ | CLEARED | cleared from latent `INFECTION`; reinfection RR $\rho_C$ |
| $R$ | RECOVERED | recovered from `NON_INFECTIOUS`; reinfection RR $\rho_R$ |
| $W$ | TREATED | completed/cured treatment; reinfection RR $\rho_W$ |

**Strain-structured natural-history states (4 states × 3 = 12):**

| Symbol | State | Infectious? |
|---|---|---|
| $L^s$ | INFECTION (latent) | no |
| $N^s$ | NON_INFECTIOUS (active, early) | no |
| $A^s$ | ASYMPTOMATIC (active) | yes (weight $\kappa$) |
| $Y^s$ | SYMPTOMATIC (active) | yes (weight $1$) |

**Treatment, strain × origin (3 × 2 = 6):** origin $o \in \{a, y\}$ records the state treatment was initiated from (asymptomatic / symptomatic), because failures return to origin.

| Symbol | Meaning |
|---|---|
| $T^{s,a}$ | on treatment, strain content $s$, initiated from ASYMPTOMATIC |
| $T^{s,y}$ | on treatment, strain content $s$, initiated from SYMPTOMATIC |

**Absorbing:** $D$ (dead). Background and TB deaths are recycled as births into $S$ to hold $N$ constant.

Total: **22 dynamic compartments** ($+D$).

---

## 2. Parameters

Natural-history rates (per year) inherit the LSHTM/`tbsim.TB` defaults; new resistance parameters are listed below them.

| Symbol | Code name | Default | Meaning |
|---|---|---|---|
| $\beta$ | `beta` | ~9/yr | effective contact rate (freq.-dependent, $\beta/N$) |
| $\kappa$ | `trans_asymp` | 0.82 | rel. infectiousness asymptomatic vs symptomatic |
| | `inf_cle` | 1.90 | $L \to$ CLEARED |
| | `inf_non` | 0.16 | $L \to N$ |
| | `inf_asy` | 0.06 | $L \to A$ |
| | `non_rec` | 0.18 | $N \to$ RECOVERED |
| | `non_asy` | 0.25 | $N \to A$ |
| | `asy_non` | 1.66 | $A \to N$ (reversion) |
| | `asy_sym` | 0.88 | $A \to Y$ |
| | `sym_asy` | 0.54 | $Y \to A$ (reversion) |
| | `sym_dead` | 0.34 | $Y \to D$ (TB mortality) |
| $\mu$ | `mu` | 1/70 | background mortality |
| $\rho_C$ | `rr_reinfection_cleared` | 1.0 | reinfection RR after clearing latent |
| $\rho_R$ | `rr_reinfection_rec` | 0.21 | reinfection RR after recovery |
| $\rho_W$ | `rr_reinfection_treat` | 3.15 | reinfection RR after treatment |
| **Strain / fitness** | | | |
| $r_a, r_b$ | `fit_a`, `fit_b` | $r_a > r_b$, e.g. 1.0, 0.9 | per-strain transmission fitness |
| **Superinfection susceptibility** (rel. to fully susceptible) | | | |
| $\sigma_L$ | `rr_reinfection_inf` | 1 | susceptibility of $L^{\text{mono}}$ to 2nd strain |
| $\sigma_N$ | `rr_reinfection_non` | 1 | of $N^{\text{mono}}$ |
| $\sigma_A$ | `rr_reinfection_asy` | 0 | of $A^{\text{mono}}$ |
| $\sigma_Y$ | `rr_reinfection_sym` | 0 | of $Y^{\text{mono}}$ |
| **Progression** | | | |
| $p_\text{multi}$ | `p_multi` | 1 | prob. both strains co-progress at $\to A$ |
| $(h_A,h_B)$ | `prog_select` | $(0.5,0.5)$ | strain selected when only one progresses |
| $\psi$ | `rr_prog_super` | 1 | progression-rate multiplier for $AB$ (into $A$ and into $Y$) |
| $\omega$ | `rr_clear_super` | 1 | clearance/recovery-rate multiplier for $AB$ (off by default) |
| **Treatment** | | | |
| $r^A_\text{tx}$ | `r_treat_asym` | e.g. 0.2 | treatment initiation from $A$ |
| $r^Y_\text{tx}$ | `r_treat_sym` | e.g. 2.0 | treatment initiation from $Y$ |
| $\delta$ | `delta` | 2.0 | treatment exit (completion) rate |
| $e_a, e_b$ | `eff_a`, `eff_b` | e.g. 0.85, 0.50 | per-strain cure probability ($e_a>e_b$) |
| $q$ | `q_treat` | e.g. 0.02 | prob. of acquiring resistance per A-surviving treatment **failure** (always replacement A$\to$B; not a test variable) |
| **De novo resistance (on progression)** | | | |
| $q_p$ | `q_prog` | e.g. 0.005 | prob. of *de novo* resistance at each $L^A\to N$ and $L^A\to A$ progression event (mono-A only) |
| **Mode switches** | | | |
| — | `transmission_mode` | `bottleneck` | `bottleneck` vs `independent` (Q5) |
| — | `prog_resist_mode` | `mixed` | de novo $q_p$ event yields `mixed` ($L^A\to AB$) vs `replacement` ($L^A\to B$) (Q4) |

Define for the bottleneck the transmitted-strain fractions for a superinfected source:
$$ g_A = \frac{r_a}{r_a + r_b}, \qquad g_B = \frac{r_b}{r_a + r_b}, \qquad r_{\max} = \max(r_a, r_b). $$

---

## 3. Force of infection

Infectious "pressure" by strain content (asymptomatic down-weighted by $\kappa$):
$$ P_A = \kappa A^A + Y^A, \quad P_B = \kappa A^B + Y^B, \quad P_{AB} = \kappa A^{AB} + Y^{AB}. $$

**Bottleneck mode (default).** A superinfected source transmits at total rate $r_{\max}$, apportioned to A vs B by $g_A : g_B$:
$$
\lambda_A = \frac{\beta}{N}\Big[\, r_a P_A + g_A\, r_{\max} P_{AB} \,\Big], \qquad
\lambda_B = \frac{\beta}{N}\Big[\, r_b P_B + g_B\, r_{\max} P_{AB} \,\Big].
$$

**Independent mode (Q5).** A superinfected source transmits each strain at its own full rate:
$$
\lambda_A = \frac{\beta}{N}\, r_a \big(P_A + P_{AB}\big), \qquad
\lambda_B = \frac{\beta}{N}\, r_b \big(P_B + P_{AB}\big).
$$

The two modes differ **only** in the $P_{AB}$ contribution. Since $r_a + r_b > r_{\max}$, independent transmission yields strictly more total onward transmission from superinfected sources (no bottleneck competition between co-resident strains).

---

## 4. Infection and superinfection hazards

**Primary infection** (always yields mono-infection; simultaneous dual acquisition is $O(\mathrm{d}t^2)$, neglected):

$$ S,\;C,\;R,\;W \;\xrightarrow{\;\lambda_A \cdot \{1,\rho_C,\rho_R,\rho_W\}\;}\; L^A, \qquad S,\;C,\;R,\;W \;\xrightarrow{\;\lambda_B \cdot \{1,\rho_C,\rho_R,\rho_W\}\;}\; L^B. $$

**Superinfection** (mono $\to AB$, governed by state-specific $\sigma$):

$$
L^A \xrightarrow{\sigma_L \lambda_B} L^{AB}, \quad L^B \xrightarrow{\sigma_L \lambda_A} L^{AB}, \qquad
N^A \xrightarrow{\sigma_N \lambda_B} N^{AB}, \quad N^B \xrightarrow{\sigma_N \lambda_A} N^{AB},
$$
$$
A^A \xrightarrow{\sigma_A \lambda_B} A^{AB}, \quad A^B \xrightarrow{\sigma_A \lambda_A} A^{AB}, \qquad
Y^A \xrightarrow{\sigma_Y \lambda_B} Y^{AB}, \quad Y^B \xrightarrow{\sigma_Y \lambda_A} Y^{AB}.
$$

By default $\sigma_A=\sigma_Y=0$ (no superinfection during active disease).

---

## 5. Progression bottleneck operator (at $\to A$ only)

Total progression flux **into ASYMPTOMATIC from $AB$ sources** is
$$ \Pi_{AB} \;=\; \psi\,\texttt{inf\_asy}\,L^{AB} \;+\; \psi\,\texttt{non\_asy}\,N^{AB}. $$
This flux is split by the bottleneck:
$$
\Pi_{AB} \to A^{AB} \text{ w.p. } p_\text{multi}, \qquad
\to A^{A} \text{ w.p. } (1-p_\text{multi})h_A, \qquad
\to A^{B} \text{ w.p. } (1-p_\text{multi})h_B.
$$
So the per-destination inflows are
$$
G^{AB} = p_\text{multi}\,\Pi_{AB}, \qquad G^{A} = (1-p_\text{multi})h_A\,\Pi_{AB}, \qquad G^{B} = (1-p_\text{multi})h_B\,\Pi_{AB}.
$$

- **Random selection (default):** $h_A = h_B = \tfrac12$.
- **Fitness-weighted selection (Q2b):** $h_A = g_A,\; h_B = g_B$.

Note: $L^{AB}\to N^{AB}$ (`inf_non`) carries **both** strains (no bottleneck, no $\psi$). The bottleneck does **not** fire on the $A\to Y$ step ($A^{AB}\to Y^{AB}$ keeps both strains, at rate $\psi\,\texttt{asy\_sym}$).

### 5.1 De novo resistance at progression out of INFECTION (Q4)

Independently of the bottleneck, mono-A latents acquire resistance *de novo* (random within-host mutation) with one-time probability $q_p$ at **each** progression event out of `INFECTION` — i.e. at $L^A\to N$ (rate `inf_non`) and $L^A\to A$ (rate `inf_asy`). This applies to mono-A only ($L^B$, $L^{AB}$ unaffected: B is already present or absent of A). The total exit rate of $L^A$ is unchanged; only the destination strain content is re-routed. With the mode indicators $[\text{mix}]+[\text{rep}]=1$ for `prog_resist_mode`:

$$
L^A \xrightarrow{\texttt{inf\_non}}
\begin{cases}
N^{A} & \text{w.p. } 1-q_p\\
N^{AB} & \text{w.p. } q_p\,[\text{mix}] \quad(\text{mixed, default})\\
N^{B} & \text{w.p. } q_p\,[\text{rep}] \quad(\text{replacement})
\end{cases}
\qquad
L^A \xrightarrow{\texttt{inf\_asy}}
\begin{cases}
A^{A} & \text{w.p. } 1-q_p\\
A^{AB} & \text{w.p. } q_p\,[\text{mix}]\\
A^{B} & \text{w.p. } q_p\,[\text{rep}]
\end{cases}
$$

(No bottleneck interaction: this is a single mono-A progression that spawns B, not two strains co-progressing. In `mixed` mode it is one of the few default ways $AB$ first appears in $N$ or $A$ even when $\sigma_A=\sigma_Y=0$.)

---

## 6. Treatment-outcome operator

On-treatment compartments leave at rate $\delta$. Each strain is cured independently ($e_a$ for A, $e_b$ for B). Failures return to the **origin** disease state (asymptomatic if $o=a$, symptomatic if $o=y$). Among failures in which strain **A survives**, resistance is acquired with probability $q$, **always as replacement** (the surviving A becomes B; per spec, within-host bottleneck effects fully switch A$\to$B). Consequently an $AB$ individual whose treatment fails and acquires resistance can *reduce* to mono-B. The exit probability from on-treatment content $m$ to outcome content $s$ (or cure $\to W$) is:

| On-treatment $m$ | $\to W$ (cured) | $\to A$-mono | $\to B$-mono | $\to AB$ |
|---|---|---|---|---|
| $A$ | $e_a$ | $(1-e_a)(1-q)$ | $(1-e_a)q$ | $0$ |
| $B$ | $e_b$ | — | $(1-e_b)$ | — |
| $AB$ | $e_a e_b$ | $(1-e_a)e_b(1-q)$ | $e_a(1-e_b) + (1-e_a)e_b\,q + (1-e_a)(1-e_b)\,q$ | $(1-e_a)(1-e_b)(1-q)$ |

Each row sums to 1 (verified: A-, B-, AB-row totals $= 1-e_a$, $1-e_b$, $1$ over their non-$W$ entries combined with cure). Denote the table entry $\pi(m \to s)$. Note the treatment-failure $q$ here is a **fixed** modelling assumption (always replacement), not a test variable — distinct from the de novo $q_p$ of §5.1, whose mechanism *is* the Q4 test.

Define the treatment-failure **return inflows** into each disease state (origin $o$, target $X(a)=A$, $X(y)=Y$):
$$ \Phi^{s}_{o} \;=\; \delta \sum_{m\in\{A,B,AB\}} \pi(m \to s)\; T^{m,o}, \qquad o\in\{a,y\},\; s\in\{A,B,AB\}. $$
Cures flow to $W$: $\;\delta\big[e_a(T^{A,a}+T^{A,y}) + e_b(T^{B,a}+T^{B,y}) + e_a e_b(T^{AB,a}+T^{AB,y})\big].$

---

## 7. Full ODE system

Births recycle all deaths: $\;B_{\text{birth}} = \mu N + \texttt{sym\_dead}\,(Y^A + Y^B + Y^{AB})$. Let $\Lambda = \lambda_A + \lambda_B$ and $U = S + \rho_C C + \rho_R R + \rho_W W$ (the susceptible pool weighted by reinfection protection).

**Strain-agnostic:**
$$
\dot S = B_{\text{birth}} - \Lambda S - \mu S
$$
$$
\dot C = \texttt{inf\_cle}\,(L^A + L^B) + \omega\,\texttt{inf\_cle}\,L^{AB} - \rho_C \Lambda C - \mu C
$$
$$
\dot R = \texttt{non\_rec}\,(N^A + N^B) + \omega\,\texttt{non\_rec}\,N^{AB} - \rho_R \Lambda R - \mu R
$$
$$
\dot W = \delta\big[e_a(T^{A,a}{+}T^{A,y}) + e_b(T^{B,a}{+}T^{B,y}) + e_a e_b(T^{AB,a}{+}T^{AB,y})\big] - \rho_W \Lambda W - \mu W
$$

**Latent ($L$):**
$$
\dot L^A = \lambda_A U - (\texttt{inf\_cle}+\texttt{inf\_non}+\texttt{inf\_asy}+\sigma_L\lambda_B+\mu)\,L^A
$$
$$
\dot L^B = \lambda_B U - (\texttt{inf\_cle}+\texttt{inf\_non}+\texttt{inf\_asy}+\sigma_L\lambda_A+\mu)\,L^B
$$
$$
\dot L^{AB} = \sigma_L(\lambda_B L^A + \lambda_A L^B) - (\omega\,\texttt{inf\_cle}+\texttt{inf\_non}+\psi\,\texttt{inf\_asy}+\mu)\,L^{AB}
$$

**Non-infectious ($N$):**
$$
\dot N^A = \texttt{inf\_non}(1-q_p)\,L^A + \texttt{asy\_non}\,A^A - (\texttt{non\_rec}+\texttt{non\_asy}+\sigma_N\lambda_B+\mu)\,N^A
$$
$$
\dot N^B = \texttt{inf\_non}\,L^B + [\text{rep}]\,\texttt{inf\_non}\,q_p\,L^A + \texttt{asy\_non}\,A^B - (\texttt{non\_rec}+\texttt{non\_asy}+\sigma_N\lambda_A+\mu)\,N^B
$$
$$
\dot N^{AB} = \texttt{inf\_non}\,L^{AB} + [\text{mix}]\,\texttt{inf\_non}\,q_p\,L^A + \texttt{asy\_non}\,A^{AB} + \sigma_N(\lambda_B N^A + \lambda_A N^B) - (\omega\,\texttt{non\_rec}+\psi\,\texttt{non\_asy}+\mu)\,N^{AB}
$$

**Asymptomatic ($A$):** with bottleneck inflows $G^s$ (Section 5) and treatment returns $\Phi^s_a$ (Section 6):
$$
\dot A^A = \texttt{inf\_asy}(1-q_p)\,L^A + \texttt{non\_asy}\,N^A + \texttt{sym\_asy}\,Y^A + G^A + \Phi^A_a - (\texttt{asy\_non}+\texttt{asy\_sym}+r^A_\text{tx}+\sigma_A\lambda_B+\mu)\,A^A
$$
$$
\dot A^B = \texttt{inf\_asy}\,L^B + [\text{rep}]\,\texttt{inf\_asy}\,q_p\,L^A + \texttt{non\_asy}\,N^B + \texttt{sym\_asy}\,Y^B + G^B + \Phi^B_a - (\texttt{asy\_non}+\texttt{asy\_sym}+r^A_\text{tx}+\sigma_A\lambda_A+\mu)\,A^B
$$
$$
\dot A^{AB} = \texttt{sym\_asy}\,Y^{AB} + G^{AB} + [\text{mix}]\,\texttt{inf\_asy}\,q_p\,L^A + \sigma_A(\lambda_B A^A + \lambda_A A^B) + \Phi^{AB}_a - (\texttt{asy\_non}+\psi\,\texttt{asy\_sym}+r^A_\text{tx}+\mu)\,A^{AB}
$$

**Symptomatic ($Y$):** with treatment returns $\Phi^s_y$:
$$
\dot Y^A = \texttt{asy\_sym}\,A^A + \Phi^A_y - (\texttt{sym\_asy}+\texttt{sym\_dead}+r^Y_\text{tx}+\sigma_Y\lambda_B+\mu)\,Y^A
$$
$$
\dot Y^B = \texttt{asy\_sym}\,A^B + \Phi^B_y - (\texttt{sym\_asy}+\texttt{sym\_dead}+r^Y_\text{tx}+\sigma_Y\lambda_A+\mu)\,Y^B
$$
$$
\dot Y^{AB} = \psi\,\texttt{asy\_sym}\,A^{AB} + \sigma_Y(\lambda_B Y^A + \lambda_A Y^B) + \Phi^{AB}_y - (\texttt{sym\_asy}+\texttt{sym\_dead}+r^Y_\text{tx}+\mu)\,Y^{AB}
$$

**Treatment ($T$):** for $s\in\{A,B,AB\}$,
$$
\dot T^{s,a} = r^A_\text{tx}\,A^s - (\delta+\mu)\,T^{s,a}, \qquad
\dot T^{s,y} = r^Y_\text{tx}\,Y^s - (\delta+\mu)\,T^{s,y}.
$$

---

## 8. Conservation and reduction checks

1. **Population conservation.** Summing all 22 equations: births $\mu N + \texttt{sym\_dead}\sum Y$ exactly offset $\mu\sum(\text{compartments}) + \texttt{sym\_dead}\sum Y$, and the $\delta T$ outflows are fully redistributed (cures $+$ failures sum to 1). Hence $\dot N = 0$; $N$ is constant. Use $|\sum_i \dot x_i| < \varepsilon$ as a unit test.
2. **Single-strain reduction.** Seed only strain A ($L^B(0)=0$, all-B compartments 0), set $q=q_p=0$, $\sigma=0$. The A-only subsystem $\{S,C,R,W,L^A,N^A,A^A,Y^A,T^{A,a},T^{A,y}\}$ must reproduce a treatment-augmented `TB_ODE`. With treatment off ($r^A_\text{tx}=r^Y_\text{tx}=0$) it must reproduce `TB_ODE` exactly.
3. **Strain symmetry.** With $r_a=r_b$, $e_a=e_b$, $q=q_p=0$, $p_\text{multi}=1$, and symmetric seeding, $A$- and $B$-compartments must remain identical for all $t$, and their sum must match the single-strain model.

---

## 9. Invasion / $R_0$ analysis (analytic anchor)

For a single strain $i$ in isolation (no treatment, no superinfection), the infected states $\{L,N,A,Y\}$ give a next-generation matrix $K=FV^{-1}$ with transmission matrix $F$ (new infections enter $L$ only):

$$
F_{L,A} = \beta r_i \kappa, \quad F_{L,Y} = \beta r_i, \quad \text{(all other } F=0\text{)},
$$
and transition matrix
$$
V = \begin{pmatrix}
d_L & 0 & 0 & 0\\
-\texttt{inf\_non} & d_N & -\texttt{asy\_non} & 0\\
-\texttt{inf\_asy} & -\texttt{non\_asy} & d_A & -\texttt{sym\_asy}\\
0 & 0 & -\texttt{asy\_sym} & d_Y
\end{pmatrix},
$$
with diagonal exit rates $d_L=\texttt{inf\_cle}+\texttt{inf\_non}+\texttt{inf\_asy}+\mu$, $d_N=\texttt{non\_rec}+\texttt{non\_asy}+\mu$, $d_A=\texttt{asy\_non}+\texttt{asy\_sym}+\mu$, $d_Y=\texttt{sym\_asy}+\texttt{sym\_dead}+\mu$ (add $r^A_\text{tx}$ to $d_A$ and $r^Y_\text{tx}$ to $d_Y$ when treatment is present). Then
$$
R_0^{(i)} = \beta r_i\Big(\kappa\,[V^{-1}]_{A,L} + [V^{-1}]_{Y,L}\Big) = \beta r_i\big(\kappa\,\mathbb{E}[\text{time in }A \mid \text{start }L] + \mathbb{E}[\text{time in }Y \mid \text{start }L]\big).
$$

This gives a clean read on the competition: the **fitness cost** scales $R_0$ linearly through $r_i$, while **treatment** shortens the infectious sojourn (raising $d_A,d_Y$) and so lowers the *effective* $R_0$ of the treatable strain A relative to B. The sign of $R_0^{A,\text{eff}} - R_0^{B,\text{eff}}$ predicts which strain dominates and whether resistance invades — a useful check on every numerical experiment, and a way to set $\beta$ to a target baseline prevalence.

---

## 10. Mapping questions of interest to model variants

Every question is a switch on the master system above. Null/default column reproduces the intended baseline.

| # | Question | Knob(s) | Null | Variant(s) |
|---|---|---|---|---|
| 1 | Faster progression with superinfection | $\psi$ (`rr_prog_super`) on `inf_asy`,`non_asy`,`asy_sym` for $AB$ | $\psi=1$ | $\psi>1$ (e.g. 1.5, 2, 5); optionally a per-transition vector |
| 2a | Progression bottleneck | $p_\text{multi}$ | $p_\text{multi}=1$ | sweep $1\to 0$ |
| 2b | Selection rule when one strain progresses | $(h_A,h_B)$ | random $(\tfrac12,\tfrac12)$ | fitness-weighted $(g_A,g_B)$ |
| 3a | Superinfection eligibility | $\sigma_L,\sigma_N$ | $\sigma_L{=}\sigma_N{=}1$ | INFECTED-only: $\sigma_L{=}1,\sigma_N{=}0$ |
| 3b | Superinfection during active disease | $\sigma_A,\sigma_Y$ | $0$ | $>0$, examined with treatment on |
| 4 | De novo (random) resistance acquisition mechanism, at $L^A$ progression | `prog_resist_mode` (with $q_p>0$) | `mixed` ($L^A\to AB$) | `replacement` ($L^A\to B$) — §5.1 |
| 5 | Transmission bottleneck vs independence | `transmission_mode` | `bottleneck` | `independent` (Section 3) |
| ~~6~~ | ~~Repeated/identical-strain superinfection~~ | — | **deprioritized** | not implemented — see Section 12 |

Fixed assumptions (not test variables): treatment-failure resistance $q$ is **always replacement** (A$\to$B), per spec §"Treatment & Resistance"; an $AB$ that fails and mutates reduces to mono-B.

Optional sensitivity knob shared across questions: $\omega$ (`rr_clear_super`) to test superinfection-modified clearance (spec §"Clearance rates ... not affected ... although we will test").

---

## 11. Outputs and observables

Per run, track:

- **Burden:** active-TB prevalence $\sum_s (A^s + Y^s)/N$; TB incidence (flux into active disease); TB mortality $\texttt{sym\_dead}\sum_s Y^s$; notifications (flux into $T$).
- **Resistance composition:** fraction of active TB carrying B $= \dfrac{A^B+Y^B+A^{AB}+Y^{AB}}{\sum_s(A^s+Y^s)}$; superinfection fraction $\dfrac{A^{AB}+Y^{AB}}{\dots}$; latent reservoir composition.
- **Resistance origin decomposition** (three sources, now separable): (i) *de novo* resistance flux at progression (the $q_p$ terms in §5.1), (ii) *treatment-acquired* resistance flux (the $\delta\,q$ terms in the treatment operator), and (iii) *transmitted* resistance flux (new $L^B$ infections from B-carrying sources). This decomposition is the key mechanistic output distinguishing the variants — note Q4 changes how (i) lands ($AB$ vs mono-B) and so feeds (iii) differently.
- **Equilibrium outcomes:** does B persist / invade / exclude A? Coexistence region in $(r_b/r_a,\, e_b/e_a,\, q)$ space, compared against the $R_0$ prediction of Section 9.

The headline comparison for each question is the **shift in steady-state resistant fraction and in time-to-establishment of resistance** relative to the null.

---

## 12. Question 6 — repeated/identical-strain superinfection (DEPRIORITIZED — not implemented)

Per the updated spec, this question will **not** be carried out; it is retained here for record only. For completeness, the reason it is out of scope for the ODE: "number of times infected" is an unbounded extra dimension — letting transmission potential depend on infection multiplicity $k$ would require stratifying each infectious state by $(k_A, k_B)$, giving combinatorial blow-up and no natural closure (a compartmental approximation would have to truncate $k$ or carry a summary moment, both lossy). If revisited, the natural route is a deterministic individual-based simulation sharing the same rates, with per-agent counts $(k_A,k_B)$ and per-strain transmission $r_i(k_i)$; the ODE of Sections 1–11 is its $r_i(k_i)\equiv r_i$ special case and would serve as the validation target. Q1–Q5 are fully answered by the ODE.

---

## 13. Suggested experiment sequence

1. **Calibrate baseline.** Single-strain ($A$ only), tune $\beta$ to target active-TB prevalence/incidence using the $R_0$ formula (Section 9) as a starting guess; confirm reduction check (Section 8.2).
2. **Introduce strain B at equilibrium.** Seed a small $L^B$ into the strain-A-only endemic equilibrium; record invasion (yes/no) and time-to-establishment under null settings.
3. **Sweep questions in priority order (Q1→Q5)**, one knob at a time, then key 2-way interactions flagged by the spec (esp. Q3b × treatment, and Q4 × Q5 — the de novo-resistance mechanism interacts with how superinfected sources transmit). Report each as a shift in steady-state resistant fraction, the three-way resistance-origin split (§11), and overall burden.
4. Q6 is out of scope (§12).

