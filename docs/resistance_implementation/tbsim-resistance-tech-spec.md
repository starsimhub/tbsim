**<u>TBSim Resistance Technical Specifications</u>**

**Individual strain resistance profiles:**

- $x_{i,j}$ represents the phenotypic resistance of strain $j$ to drug/class $i$

- Flexibly defined -- starting categories for $i$: RIF, BDQ

  - Others we may want to eventually build in: INH, FQ, other second-line companion drugs

  - For $n$ drugs/classes, there will be ${m = 2}^{n}$ possible strains.

- Track binary presence/absence of resistance phenotype for each strain and each category, to create strain profile $\mathbf{X}_{\mathbf{j}}\mathbf{=}\{ x_{1,j},\ x_{2,j}\ ,\ldots,\ x_{n,j}\}$ where each $x_{i,j}\ \epsilon\ \{ 1,\ 0\}$.

| Strain | x~RIF,j~ | x~BDQ,j~ | x~FQ,j~ | Notation | Interpretation                     |
|--------|----------|----------|---------|----------|------------------------------------|
| X~1~   | 0        | 0        | 0       | {0,0,0}  | Strain 1 is pan-susceptible        |
| X~2~   | 1        | 0        | 0       | {1,0,0}  | Strain 2 is resistant to RIF only  |
| X~3~   | 1        | 0        | 1       | {1,0,1}  | Strain 3 is resistant to RIF & FQs |
| ...    |          |          |         |          |                                    |

> **Implementation — `tbsim/resistance/strains.py`.** The `Strains` registry realizes $x_{i,j}$ exactly: `drugs` is the ordered list of names (list index = resistance bit $i$; `RIF`/`BDQ` are just user-supplied string labels — nothing biological is hard-coded), and `Strains.profile` is the $(m, n)$ boolean matrix with `profile[j, i]` $= x_{i,j}$. Strain ids run $0 \ldots m-1$ with `m = 2**n`; id 0 is pan-susceptible and bit $i$ of the id is resistance to drug $i$ (so id 5 decodes to $\{1,0,1\}$ = RIF+FQ). Per-drug fitness costs $r_i$ live in the name-keyed `rel_fitness` dict (default 1.0), and `Strains.fitness[j]` is their product over the drugs strain $j$ resists. Adding INH / FQ / a second-line companion needs no code change — just append the name to `drugs`.

**Allow for multi-strain infections (i.e., superinfections):**

- Each agent$\ k$ can be infected with multiple strains $\mathbf{X}_{\mathbf{j}}$. Define the strain profile of agent $k$ as $\mathbf{Y}_{\mathbf{k}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,k}}\mathbf{,\ \ldots,\ }\mathbf{X}_{\mathbf{m,k}} \right\}\mathbf{,\ }where\ \mathbf{X}_{\mathbf{1,k}}\mathbf{=}\left\{ x_{1,1},\ x_{2,1},\ \ldots,\ x_{n,1} \right\},\ \mathbf{X}_{\mathbf{m,k}}\mathbf{=}\left\{ x_{1,m},x_{2,m},\ldots,x_{n,m} \right\},$ etc.

- Example, for drugs $\{ RIF,\ BDQ,\ FQ\}$ as above:

  1.  Agent A is infected with 1 strain that is resistant to RIF only: $\mathbf{Y}_{\mathbf{A}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,A}} \right\}\mathbf{= \{}\left\{ 1,0,0 \right\}\}$

  2.  Agent B is infected with 2 strains -- 1 pan-susceptible strain and 1 strain resistant to RIF and FQs: $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\{\left\{ 0,0,0 \right\},\ \left\{ 1,0,1 \right\}\}$

> **Implementation — per-agent `strain_mask` (`tb_resistant.py`).** Agent profile $Y_k$ is stored as one integer per agent, `ss.IntArr('strain_mask')`, where bit $j$ set = agent carries strain $j$ (mask 0 = uninfected). This packs an arbitrary *set* of strains into a single 1-D Starsim array (Starsim has no native 2-D per-agent state), so Agent B's $\{\{0,0,0\},\{1,0,1\}\}$ is `mask = (1<<0) | (1<<5)`. `Strains.carried(mask)` decodes a mask array back into the $(k, m)$ membership matrix used everywhere downstream. There is no cap on the number of strains an agent may carry.

**Transmission:**

- Susceptible agents can only acquire strains within the infecting agent's strain profile (no emergence of resistance during transmission events)

- For infecting agents with single infection:

  1.  Consider strain fitness: each phenotype $i$ associated with some multiplicative reduction $r_{i}$ on FOI, where $r_{i}\epsilon\lbrack 0,1\rbrack$; reductions of each phenotype are considered independent & multiplicative

  2.  Example: if probability of transmission per contact with an agent with pan-susceptible strain $\{ 0,0,0\}$ equals $\beta,\$ then probability of transmission per contact with an agent with strain $\left\{ 1,0,0 \right\}$ equals $r_{RIF}\beta$, probability of transmission per contact with strain $\{ 1,1,0\}$ equals $r_{RIF}r_{BDQ}\beta$, etc.

- For infecting agents with superinfection:

  1.  Each transmission event only passes 1 strain from infector to infectee.

  2.  Consider strain fitness: overall probability of transmission per contact equals the probability of transmission per contact of the fittest strain, using the same approach for agents with single infection (above). Samples from a multinomial distribution based on each strain's fitness then determine which strain is passed to the infectee.

  3.  Implication: super-infection does not lower the overall risk of transmission by an infectious agent, but it does disadvantage both strains compared to a single infection with either strain. See table below for an illustration of these dynamics.

  4.  Example: say $\mathbf{Y}_{\mathbf{C}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,C}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,C}} \right\}\mathbf{=}\ \{\left\{ 1,0,0 \right\},\ \left\{ 1,1,0 \right\}\}$. Then the probability of transmission per contact with a susceptible agent equals $r_{RIF}\beta$, since $r_{RIF}\beta > r_{RIF}r_{BDQ}\beta.$ Conditional on transmission occurring, the probability that strain $\mathbf{X}_{\mathbf{1,C}}\mathbf{\ }$is passed equals $\frac{r_{RIF}}{r_{RIF} + r_{RIF}r_{BDQ}} = \frac{1}{1 + r_{BDQ}}$ and the probability that strain $\mathbf{X}_{\mathbf{2,C}}\mathbf{\ }$is passed equals $\frac{r_{RIF}r_{BDQ}}{r_{RIF} + r_{RIF}r_{BDQ}} = \frac{r_{BDQ}}{1 + r_{BDQ}}$.

<table style="width:100%;">
<colgroup>
<col style="width: 21%" />
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 18%" />
<col style="width: 18%" />
</colgroup>
<thead>
<tr>
<th rowspan="2" style="text-align: center;"><strong>Example for agent C,
with illustrative numbers</strong></th>
<th rowspan="2" style="text-align: center;"><strong>Single infection w/
strain</strong> <span
class="math inline">{<strong>1</strong><strong>,</strong> <strong>0</strong><strong>,</strong> <strong>0</strong>}</span></th>
<th rowspan="2" style="text-align: center;"><strong>Single infection w/
strain</strong> <span
class="math inline">{<strong>1</strong><strong>,</strong> <strong>1</strong><strong>,</strong> <strong>0</strong>}</span></th>
<th colspan="2"
style="text-align: center;"><strong>Superinfection</strong></th>
</tr>
<tr>
<th style="text-align: center;"><strong>Strain</strong> <span
class="math inline">{<strong>1</strong><strong>,</strong> <strong>0</strong><strong>,</strong> <strong>0</strong>}</span></th>
<th style="text-align: center;"><strong>Strain</strong> <span
class="math inline">{<strong>1</strong><strong>,</strong> <strong>1</strong><strong>,</strong> <strong>0</strong>}</span></th>
</tr>
</thead>
<tbody>
<tr>
<td>Relative fitness <span
class="math inline"><em>r</em><sub><em>i</em></sub></span></td>
<td style="text-align: center;"><span
class="math display"><em>r</em><sub><em>R</em><em>I</em><em>F</em> </sub> = 0.5</span></td>
<td style="text-align: center;"><span
class="math display"><em>r</em><sub><em>B</em><em>D</em><em>Q</em></sub><em>r</em><sub><em>R</em><em>I</em><em>F</em></sub> = 0.5 * 0.8 = 0.4</span></td>
<td style="text-align: center;"><span
class="math display"><em>r</em><sub><em>R</em><em>I</em><em>F</em> </sub> = 0.5</span></td>
<td style="text-align: center;"><span
class="math display"><em>r</em><sub><em>B</em><em>D</em><em>Q</em></sub><em>r</em><sub><em>R</em><em>I</em><em>F</em></sub> = 0.4</span></td>
</tr>
<tr>
<td>Relative effective contact rate</td>
<td style="text-align: center;"><p>= relative fitness</p>
<p><span
class="math display"><em>r</em><sub><em>R</em><em>I</em><em>F</em> </sub> = 0.5</span></p></td>
<td style="text-align: center;"><p>= relative fitness</p>
<p><span
class="math display"><em>r</em><sub><em>B</em><em>D</em><em>Q</em></sub><em>r</em><sub><em>R</em><em>I</em><em>F</em></sub> = 0.4</span></p></td>
<td colspan="2" style="text-align: center;">max(relative fitness of each
strain) = <em><br />
</em><span
class="math display"><em>r</em><sub><em>R</em><em>I</em><em>F</em> </sub> = 0.5</span></td>
</tr>
<tr>
<td>Overall rate of transmission of any strain to another agent</td>
<td style="text-align: center;"><span
class="math display"><em>r</em><sub><em>R</em><em>I</em><em>F</em></sub><em>β</em> = <strong>0.5</strong><strong>β</strong></span></td>
<td style="text-align: center;"><span
class="math display"><em>r</em><sub><em>R</em><em>I</em><em>F</em></sub><em>r</em><sub><em>B</em><em>D</em><em>Q</em></sub><em>β</em> = <strong>0.4</strong><strong>β</strong></span></td>
<td colspan="2" style="text-align: center;"><span
class="math inline"><em>r</em><sub><em>R</em><em>I</em><em>F</em></sub><em>β</em> = <strong>0.5</strong><strong>β</strong></span></td>
</tr>
<tr>
<td>Probability that each strain is the strain that gets
transmitted</td>
<td style="text-align: center;"><span
class="math display"><strong>100</strong><strong>%</strong></span></td>
<td style="text-align: center;"><span
class="math display"><strong>100</strong><strong>%</strong></span></td>
<td style="text-align: center;"><span class="math display">$$\frac{1}{1
+ r_{BDQ}} = \frac{1}{1.8} = \mathbf{56\%}$$</span></td>
<td style="text-align: center;"><span
class="math display">$$\frac{r_{BDQ}}{1 + r_{BDQ}} = \frac{0.8}{1.8} =
\mathbf{44\%}$$</span></td>
</tr>
<tr>
<td>Rate of transmission of each strain to another agent</td>
<td style="text-align: center;"><span
class="math display">100% * 0.5<em>β</em> = <strong>0.5</strong><strong>β</strong></span></td>
<td style="text-align: center;"><span
class="math display">100% * 0.4<em>β</em> = <strong>0.4</strong><strong>β</strong></span></td>
<td style="text-align: center;"><span
class="math display">56% * 0.5<em>β</em> = <strong>0.28</strong><strong>β</strong></span></td>
<td style="text-align: center;"><span
class="math display">44% * 0.5<em>β</em> = <strong>0.22</strong><strong>β</strong></span></td>
</tr>
</tbody>
</table>

> **Implementation — reuse-the-FOI design (`step_bookkeeping` + `set_prognoses`).** Rather than a bespoke transmission loop, the model reuses Starsim's common-random-number force-of-infection engine. `step_bookkeeping` sets `rel_trans = Strains.max_fitness(strain_mask)` — the fittest carried strain, i.e. the table's "relative effective contact rate = max(relative fitness)" row — so a superinfected source transmits at its single fittest strain's rate and superinfection does *not* lower total infectiousness. `set_prognoses` then draws *which* strain is passed with `Strains.transmit_probs` (∝ carried-strain fitness — the multinomial $r_j / \sum r$ split) via the CRN-safe `choice2d` sampler. Single-infection fitness is $\prod r_i$ over resistant drugs (`Strains.fitness`), matching $r_{RIF}\beta$, $r_{RIF}r_{BDQ}\beta$, etc. The two stated constraints — infectees acquire only strains in the infector's profile, and no resistance emerges during transmission — hold by construction, since the drawn strain is always one the source already carries. This reproduces the worked Table exactly (0.28β / 0.22β, 56% / 44%); see `tests/test_resistance.py`.

**Strain competition and protection against reinfection:**

- TBsim already considers agents who spontaneously resolve (transition from early TB/"NONINFECTIOUS" state to "CLEARED" state) to retain some protection against reinfection, via the $rr\_ reinfection\_ rec$ parameter. In a multistrain model, we must also allow currently infected agents to be protected against superinfection. This allows for competition between strains.

- The potential infectee agent will have different protection against secondary (super) infections depending on the disease state in which they are in. An agent can be infected with a second strain while in the INFECTED OR NON-INFECTIOUS disease states, but not the ASYMPTOMATIC or SYMPTOMATIC (or TREATED) TB disease states.

| **Disease State** | **Relative risk of being superinfected vs fully susceptible individual** |
|----|----|
| INFECTED | $rr\_reinfection\_inf$ |
| NON-INFECTIOUS | $rr\_reinfection\_non$ |
| ASYMPTOMATIC | 0 [no superinfection allowed] |
| SYMPTOMATIC | 0 [no superinfection allowed] |

- For INFECTED individuals, we apply a multiplicative protective factor $rr\_reinfection\_inf{\  \epsilon\  \lbrack 0, 1\rbrack}$ which is strain-agnostic and also agnostic to number of strains (an agent who is already super-infected does not gain additional protection to infection with a third strain compared to an agent who is only infected with 1 strain).
  1.  By default, we set $rr\_reinfection\_ inf\  = rr\_ reinfection\_ rec$

- For NON-INFECTIOUS individuals, we similarly apply a multiplicative protective factor $rr\_ reinfection\_{non}{\ \epsilon\ \lbrack 0,\ 1\rbrack}$ that is again strain-agnostic and agnostic to number of previous infections.

  1.  By default, we set $rr\_ reinfection\_ non\  = rr\_ reinfection\_ inf$

  2.  Rationale: this state somewhat substitutes for a 2^nd^ latent state in our natural history model. Only \~1 year is spent in INFECTION on average, vs. \> 2 years in NON-INFECTIOUS, so not allowing for infection with a second strain while in NON-INFECTIOUS could reduce the prevalence of superinfection compared to typical TB models.

- For ASYMPTOMTIC or SYMPTOMATIC individuals, by default we allow for no secondary infections. This could be handled in 2 different ways, and we defer to software on how to implement this:

  1.  Option 1: Protection from reinfection can be handled similarly as above, governed by 2 parameters$,\ rr\_ reinfection\_ asy;\ rr\_ reinfection\_ sym$, that by default are set to 0. This allows for flexibility and assessment of any biases these assumptions induce, but is less simple/parsimonious.

  2.  Option 2: hard-code the model so that agents cannot be newly infected with another strains while ASYMPTOMATIC or SYMPTOMATIC. Still set $rr\_ reinfection\_ non = rr\_ reinfection\_ inf = rr\_ reinfection\_ rec$

  3.  Note that the current approach of modeling natural history at the agent level rather than the strain level implies that, if $rr\_ reinfection\_ asy > 0$ or $rr\_ reinfection\_ sym > 0$, the new strain has already progressed to active disease immediately upon acquisition. This is for the most part unrealistic and is the rationale for not allowing new infection of ASYMPTOMATIC and SYMPTOMATIC agents (if we did allow this, it could represent an agent progressing to disease experiencing some sort of hyper-susceptibility to further progression, e.g., representing some breakdown of immune responses that could "fast-track" the new strain).

- We do not allow superinfection with 2 identical strains.

  1.  Implication: this may result in some bias towards favoring low-prevalence strains, since an agent is more likely to be infected with high-prevalence strains multiple times than rare, low-prevalence strains. Regardless of whether we allow for multiple strains to progress to disease (see next section), because we assume only 1 strain gets transmitted per transmission event, this will result in somewhat less transmission of high-prevalence strains/more transmission of low-prevalence strains, relative to a scenario in which we allow for superinfection with identical strains. This could favor the emergence of drug-resistant strains and is something we should continue to assess as this functionality gets built out.

  2.  An alternative that does not require declaring the relative \[continuous\] frequency of each strain is to "count" how many instances of each strain an agent has, and use these counts to determine which strains progress to disease and/or get transmitted. We could reassess this option later (or variations on this option with different functional relationships) if we are concerned about bias after implementing the current specification.

  3.  This is the main decision we have not tested with an ODE model but which we might want to test in the future. For now: include an analyzer that allows us to quantify how much this comes up. That is, for each agent, count the number of times an agent would become infected if we did not block superinfection with 2 identical strains (e.g. going by the ordering/random number draws in TBsim, $B$ contacts $A,$ then $B$ infects $A$, then $B$ infects $A$ with strain $X$, but $A$ is already infected with strain $X$, so no infection occurs).

  4.  Note that for appropriate consideration of time-varying risk of progression (see section below), we may also wish to track the time since an individual was successfully exposed to a strain of their same infection (i.e., would still reset time since infection), even if we make no other changes to their strain profile or disease state.

- Example: consider Agent $A$ who is already infected such that $\mathbf{Y}_{\mathbf{A}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,A}} \right\} = \{\left\{ 0,0,0 \right\}\}$. Conditional upon contact with super-infected agent $B$, where $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\ \{\left\{ 0,0,0 \right\},\ \left\{ 1,0,1 \right\}\}$:

  1.  Probability that agent $A$ becomes infected from agent $B$ equals $rr\_ reinfection\_ inf\ *\beta\$

  2.  The probability that the infecting strain is the same strain agent $A$ already has (strain $\mathbf{X}_{\mathbf{1,B}}\mathbf{=}\left\{ 0,0,0 \right\}\mathbf{\ }$**)** is $\frac{1}{1 + r_{RIF}r_{FQ}}$. However, since agent $B$ cannot have superinfection with 2 identical strains, this would not result in changes to strain profile or disease state.

  3.  Agent $A$ becomes superinfected (acquires strain $\mathbf{X}_{\mathbf{2,B}}\mathbf{=}\{ 1,0,1\}$) with probability $\frac{r_{RIF}r_{FQ}}{1 + r_{RIF}r_{FQ}}$

  4.  So in this scenario, the overall probability that agent $A$ becomes superinfected, conditional on contact with agent $B,$ is $rr\_ reinfection\_ inf\ *\beta*\frac{r_{RIF}r_{FQ}}{1 + r_{RIF}r_{FQ}}$

  5.  Note, if agent $A$ had instead been super-infected already, such that $\mathbf{Y}_{\mathbf{A}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,A}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,A}} \right\} = \{\left\{ 0,0,0 \right\},\ \{ 1,0,0\}\}$ this would not have affected their probability of becoming infected with strain $\mathbf{X}_{\mathbf{2,B}}\mathbf{=}\{ 1,0,1\}$, since strain $\mathbf{X}_{\mathbf{2,B}}$ does not match any of agent A's current strains and since we do not apply extra protection for super-infected agents.

> **Implementation — state-dependent `rel_sus` + identical-strain blocking (`step_bookkeeping`, `set_prognoses`).** Each state's superinfection susceptibility is written to `rel_sus`: `rr_reinfection_inf` (σ_L) for INFECTION, `rr_reinfection_non` (σ_N) for NON_INFECTIOUS, and `rr_reinfection_asy` / `rr_reinfection_sym` for ASYMPTOMATIC / SYMPTOMATIC. **Option chosen for ASY/SYM: Option 1** (explicit parameters, default 0), not the hard-coded Option 2 — it preserves the flexibility to probe the induced bias and maps directly onto the reference ODE's σ_A / σ_Y; when either is 0 the state is simply dropped from the `susceptible` mask, which recovers Option 2's behavior for free. Protection is strain- and count-agnostic (a superinfected agent gains no extra protection against a third strain), as specified. **Honest divergence on defaults:** the spec suggests defaulting `rr_reinfection_inf = rr_reinfection_rec` and `rr_reinfection_non = rr_reinfection_inf`; I instead default σ_L = σ_N = **1.0** (the reference-ODE null), so the spec's coupling must be set explicitly if wanted. **Identical-strain blocking** is enforced in `set_prognoses` (the `already` mask): re-exposure to a carried strain changes nothing but resets `ti_infected` (the "reset the clock" rule). The requested "how often does this happen" analyzer is the `new_blocked_superinf` result, counted per step where the block fires.

**Progression to disease**

In addition to multi-strain infection, we also allow for multi-strain TB disease. Balancing simplicity with accuracy to the extent possible in the context of the somewhat more complex natural history model in TBsim, we assume the following:

- We do *not* track separate natural histories for each strain. Natural history remains an agent-level characteristic, not a strain-level characteristic.

- Progression risk and other transition rates between TB natural history states are the same regardless of how many strains an agent is infected with.

  1.  If we implement time-varying risk of disease progression, each new infection should 'reset the clock' on an individual's time since infection, which can functionally allow for different risk of progression. Ideally, this would also include successful exposure to strains with which the agent is already infected, even if we do not track those as explicit infection events. However, the number of previously-infecting strains does not affect how progression risk changes with time since infection (i.e., someone going from 0 -\> 1 infecting strains or from 1 -\> 2 infecting strains will follow the same temporal pattern of progression risk).

- If an agent in the INFECTION state, with superinfection, progresses to NON-INFECTIOUS, they retain all strains.

- If an agent in the INFECTION or NON-INFECTIOUS state progresses to ASYMPTOMATIC, they retain all strains with some probability $p\_ multi$, and otherwise each strain has an equal probability of being the single strain that progresses

  1.  We will set $p\_ multi = 1$, but can assess the importance of this parameter in one-way sensitivity analysis.

> **Implementation — `step_transitions` + `_bottleneck`.** Natural history stays agent-level (one `TB.state` per agent); strains are an overlay and progression rates are strain-count-independent, as specified. INFECTION→NON_INFECTIOUS retains all strains (no bottleneck there). The **progression bottleneck** applies only at →ASYMPTOMATIC (`_progress(..., bottleneck=True)`): a multi-strain agent keeps all strains with probability `p_multi` (default 1), else exactly one strain progresses. Strain selection defaults to **equal probability** (`prog_select='random'`, matching "each strain has equal probability"); I added an optional `'fitness'`-weighted mode as a superset. Time-varying progression that "resets the clock" is not yet modeled, but the hook exists — `ti_infected` is reset on every successful (and blocked) exposure. The optional multi-strain progression-rate multiplier ψ (`rr_prog_super`, default 1) — the reference ODE's question-of-interest #1 — scales the →ASYMPTOMATIC and A→Y rates for multi-strain agents.

**Clearance**

- We assume that if a superinfected agent naturally (i.e., absent intervention) clears their infection (INFECTION to CLEARED) or spontaneously resolves (NON-INFECTIOUS to CLEARED) all strains are cleared.

  1.  Rationale: biologically, natural clearance/resolution represents an immune response that should equally apply to multiple strains, while clearance/recovery via TPT or treatment selectively clear only certain strains because of their resistance to drugs in the TPT/treatment regimen.

- Clearance rates (and other transition rates between TB states in the natural history) are not affected by superinfection.

> **Implementation — clear-all-on-natural-clearance (`step_transitions`).** Natural clearance/resolution (INFECTION→CLEARED and NON_INFECTIOUS→CLEARED) sets `strain_mask = 0`, wiping every strain — the specified immune-mediated, strain-agnostic behavior (contrast with treatment, which clears selectively). Clearance rates are strain-count-independent except for the optional ω multiplier (`rr_clear_super`, default 1) scaling the natural-clearance rate for multi-strain agents.

**(Random) Acquisition**:

- Allow for some random/endogenous acquisition of $x_{i}$ among people with active TB (background mutation rate; presumably low and could be zero for some drugs, i.e., RIF)

- Model as a one-time, independent probability, $p\_ rand_{i}$ that each strain $j$ develops resistance to each drug $i$ upon transition from INFECTION to NON-INFECTIOUS or ASYMPTOMATIC.

  - Rationale: the alternative of a per-timestep rate (a) could result in some people who have undiagnosed TB for a very long time having an unrealistically high probability of developing resistance mutations; (b) requires us to loop over everyone with TB every timestep to see if they acquire resistance, which is inefficient; (c) is more challenging to estimate from data and may require calibration to some overall prevalence of naturally-occurring resistance.

- The probability of random acquisition $p\_ rand_{i}$ is strain-agnostic (except that a strain cannot acquire resistance to a drug to which it already has resistance).

- In testing using an ODE, whether random/*de novo* acquisition resulted in strain replacement vs. superinfection did not affect disease or resistance dynamics, so we leave it up to software how to implement this:

  - Option 1 (Superinfection): Assume that rather than replacing the existing strain with a resistant version, random acquisition results in multi-strain infection.

  - Option 2 (Replacement): Assume that a strain with the newly acquired resistance replaces the existing strain

- Allow multiple resistance acquisitions (including across multiple strains for the same agent, or for multiple drugs in the same strain) to occur.

  - Rationale: this is a very rare event and agents are unlikely to actually acquire resistance to more than 1 drug, but should simplify how this is coded/modeled

- Example: if agent $B$ currently has $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\ \{\left\{ 0,0,0 \right\},\ \left\{ 1,1,0 \right\}\}$ and is transitioning from INFECTED to NON-INFECTIOUS:

  - Each strain could independently acquire resistance to FQs with probability $p\_ rand_{FQ}$

  - Strain \#1, $\mathbf{X}_{\mathbf{1,B}}$ can also acquire resistance to RIF with probability $p\_ rand_{RIF}$ and BDQ with probability $p\_ rand_{BDQ}$

  - Say the random draws that determine resistance acquisition result in only strain \#1, $\mathbf{X}_{\mathbf{1,B}}$**,** acquiring BDQ resistance. Then:

    - Option 1 (Superinfection): agent $B$'s updated strain profile is: $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{3,B}} \right\}\mathbf{=}\ \{\left\{ 0,0,0 \right\},\ \left\{ 1,1,0 \right\},\ \left\{ 0,1,0 \right\}\}$ -- that is, a new version of strain \#1 with BDQ resistance (i.e., strain \#3) is added to agent $B$'s resistance profile.

    - Option 2 (Replacement): agent $B$'s updated strain profile is: $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\ \{\left\{ 0,1,0 \right\},\ \left\{ 1,1,0 \right\}\}$ -- that is, a strain with BDQ resistance replaces strain \#1.

> **Implementation — `_denovo` (`tb_resistant.py`).** De-novo resistance is a one-time draw at progression out of INFECTION (→NON_INFECTIOUS and →ASYMPTOMATIC), not a per-timestep rate — matching the spec's stated rationale. **Options:** configurable via `prog_resist_mode`, defaulting to **Option 1 (mixed → superinfection)**; `'replacement'` gives Option 2. (The reference ODE showed the choice doesn't affect dynamics, so the default is arbitrary-but-documented.) **Honest simplifications:** (1) the acquisition probability is a single scalar `q_prog` applied per not-yet-resistant drug, *not* a per-drug `p_rand_i` dict — so you cannot yet set RIF's rate to 0 while another drug's is positive without extending it; (2) de-novo is applied only to **mono-strain** agents (`nstr == 1`), whereas the spec allows each strain of a multi-strain agent to mutate independently — this matches the n=1 reference and treats multi-strain de-novo (rare-on-rare) as negligible; (3) for n>1 the per-drug draws share one CRN stream, so within-agent multi-drug acquisitions are correlated. Counts feed the `new_denovo_resistance` result.

**Treatment & (Selective) Acquisition:**

- Treatment efficacy:

  - For a given treatment regimen $l$, define the clinical efficacy against each possible strain $\mathbf{X}_{\mathbf{1}}\mathbf{,\ldots}\mathbf{X}_{\mathbf{m}}\mathbf{\ }$ as $\mathbf{T}_{\mathbf{l}} = \{ t_{1,l},\ldots,t_{m,l}\}$. For example, $t_{i,l}$ will be regimen $l's$ full efficacy for strains without resistance to drugs/drug classes included in the regimen, and less than full efficacy for strains with resistance to drugs/drug classes in the regimen.

  - This approach also allows us to vary efficacy for shorter regimens containing the same drugs/classes.

  - Currently, treatment in TBsim includes an adherence argument that can adjust the clinical efficacy of a regimen. During multistrain infection, we would like to be able to use this adherence parameter in such a way that it can induce agent-level correlation in treatment efficacy across strains, by allowing the user to change adherence from a regimen-level value to a regimen-level distribution that can vary by agent and would get applied across all strains during a given treatment course.

  - If a regimen cures/clears only 1 strain of an agent with superinfection/multi-strain disease, the agent remains in the same TB natural history state with just a subset of strains (and retains no memory of the cured/cleared strain, although we do want the model to track that they were treated; see section below on Treatment Monitoring).

- Selective pressure/acquisition risk:

  - For each regimen $l$, define $q_{l,i}$ as the risk of developing resistance to modeled drugs/classes $i$ included in $l$ among baseline $i$-susceptible strains with unsuccessful treatment outcomes (including failure, relapse, and LTFU if that is eventually modeled as a separate outcome -- right now it isn't).

    - $q_{l,i} = 0$ if drug $i$ is not included in regimen $l$

    - $q_{l,i} > 0$ if drug $i$ is included in regimen $l$

    - Risk of acquisition therefore occurs once per treatment episode (i.e., at the time of treatment failure/relapse)

  - Allow acquisition risk to vary by what TB state an agent is in at time of treatment failure. .

    - In practice, we will probably set the risk to 0 for agents who get treated despite *not* being SYMPTOMATIC or ASYMPTOMATIC, and set the risk to be equal for SYMPTOMATIC and ASYMPTOMATIC agents, but it will be good to build in this flexibility. This could be implemented via some RR term on q (that is default 1 for ASYMPTOMATIC and SYMPTOMATIC, 0 for other states); this RR term need not vary by strain or treatment regimen.

  - Resistance acquisition results in strain replacement.

  - Example:

    - Agent $B$ currently has SYMPTOMATIC TB with $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\ \{\left\{ 0,0,0 \right\},\ \left\{ 1,0,1 \right\}\}$ and gets treated with regimen $l$, which for the purposes of this example contains BDQ but neither RIF nor an FQ. Therefore $q_{l,RIF} = q_{l,FQ} = 0$ and $q_{l,BDQ} > 0$.

    - Because none of agent $B's$ strains are BDQ-resistant, the clinical efficacy against each strain is the full regimen efficacy; that is, $t_{1,l} = t_{2,l} = t_{l}^{*}$

    - Say (by random chance) the regimen clears/cures strain $\mathbf{X}_{\mathbf{2,B}}$ but not $\mathbf{X}_{\mathbf{1,B}}$ **.** Then $\mathbf{X}_{\mathbf{1,B}}$ acquires resistance to BDQ with probability $q_{l,BDQ}$.

    - If $\mathbf{X}_{\mathbf{1,B}}$ does acquire resistance to BDQ, agent $B's$ resulting strain profile becomes $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{3,B}} \right\}\mathbf{= \{}\left\{ 0,1,0 \right\}\mathbf{\}}$**.**

  - Note: proposed approach does not vary $q_{l,i}$ by an agent's current strain/resistance profile. However, an agent will be more likely to fail treatment if their strain is resistant to drugs in the treatment regimen, and therefore they will also be more likely to acquire new resistance (because acquisition is contingent on unsuccessful treatment outcome).

  - Note: proposed approach could vary $q_{l,i}$ by regimen duration (not just which drugs are in a regimen) -- or we could allow overall resistance risk to be increased for shorter regimens via lower efficacy. We are scoping this out flexibly enough that it should allow for both approaches.

> **Implementation — `TxR` (product) + `TxDeliveryR` (delivery), `treatments.py`.** Per-strain efficacy $T_l$ is `TxR.eff_by_id` = `base_efficacy` × ∏ `resist_penalty` over each strain's resistant drugs, so resistant strains are cured at reduced probability. Adherence induces the specified agent-level correlation across strains: a single per-agent `_adh_rng` draw gates the whole course (a non-adherent agent clears no strains that course). Partial cure works as specified — if only a subset of a multi-strain agent's strains clear, the agent returns to the state treatment was initiated from (`prior_state`) carrying the surviving `strain_mask`, with no memory of cleared strains. Acquisition-on-failure ($q_{l,i}$) is `q_acq`, applied once per failed episode as **replacement** (spec-mandated) in `TxR.acquire`. **Design choice:** I built a dedicated rate-based `TxDeliveryR` (state-specific initiation rates `rate_asym` / `rate_sym`) instead of retrofitting the existing cascade `TxDelivery`, per the "optimal fresh-field" guidance; the cascade path remains reachable via the `eligibility=` callable. **Honest gaps vs spec:** (1) adherence is binary (adherent/not), not the fuller "regimen-level distribution that varies by agent"; (2) `q_acq` is a single scalar per regimen, not a per-drug $q_{l,i}$ with a state-varying RR term — in practice delivery only treats ASYMPTOMATIC/SYMPTOMATIC so the "0 for other states" default holds, but the RR-by-state flexibility isn't built.

**TPT**

- For the most part, we can take a similar approach to TPT as TB treatment.

- Efficacy can be modeled very similarly, as varying by regimen and resistance/strain.

  - Because we will model superinfection, this means that TPT can clear a drug-susceptible strain, increasing the risk that a resistant strain goes on to <u>progress</u> to active TB (i.e., the dynamics in Ted's papers cited at the end of this document) and <u>get transmitted</u> to others. Note that if we set $p\_ multi$ above to 1, only the transmission dynamic and not the progression dynamic will be present.

- Acquisition can also be modeled similarly: some percentage of agents for whom TPT was not effective (provided neither clearance nor longer-term protection from progression) have a probability of acquiring resistance to the drugs/classes included in the TPT regimen. This probability should vary by TB state at time of failure (e.g., highest risk for ASYMPTOMATIC and SYMPTOMATIC agents, very low risk for INFECTED agents, low-medium risk for NONINFECTIOUS agents)

> **Implementation — NOT DONE.** Strain-aware TPT is not implemented. The existing `tbsim` TPT intervention is not strain-aware — it does not clear strains selectively, drive the resistant-strain progression/transmission dynamic described here, or model TPT-failure resistance acquisition. This is a documented deferral; `TxR` / `DST` are the intended templates when it is built.

**Diagnostics & Treatment Modification:**

- DST:

  - (Likely a separate class from current diagnostics for TB state) Allow for a diagnostic test to create observed resistance profile per agent $\mathbf{X}_{\mathbf{obs,k}} = \left\{ \ x_{obs,1,k},\ \ldots x_{obs,n,k} \right\},\$

    - Testing follows logic of current diagnostic class, in that we define sensitivity/specificity for each $x_{i,k}$ to have user-defined outcomes $x_{obs,i,k}$ (likely defaults, positive, negative, indeterminant) for whether DST detected resistance to drug $i$ for agent $k$.

      - Estimating the probability of detecting a given phenotype in mixed infections requires consideration of the probability that all (or multiple) strains are observed, due to possible bottlenecks at time of sample collection, culture, or other sample processing.

      - First, we should apply DST sensitivity and specificity at the strain level to indicate whether each strain-specific phenotype is observed. Then, we will aggregate these observed profiles to determine whether any $x_{obs,i,k} = 1$. At this step, there should be an additional probability (e.g., p_strain_obs) that the profile of a given strain is observed at all. When p_strain_obs=1 (i.e. we assume all strains are observed in the test), this will create a higher probability that a given resistance phenotype is detected if it exists in more than one infecting strain. However, when p_strain_obs \< 1, overall DST sensitivity will be reduced due to this strain drop-out.

      - By default, set \`p_strain_obs\` equal to strain fitness, as a proxy for within-host bacillary loads and culture growth potential. Allow p_strain_obs to be re-defined in a strain agnostic way.

    - DST will not identify specific strains; that is, the agent resistance profile will have only *n* objects for the chosen drugs/classes

  - Eligibility for DST can either be immediate (i.e., immediately following diagnosis) or dependent on treatment failure.

    - Should track time since last treatment initiation to inform whether later treatment is managed as treatment failure, with need for DST/second-line treatments, or as a new case.

  - Treatment provision can be dependent on observed diagnostic profile

    - Similar logic to current intervention structure -- certain treatments only provided to eligible individuals that match a certain profile

> **Implementation — `DST` + `DSTDelivery` (`dst.py`).** DST is a separate product from the TB-state diagnostics, producing an observed n-bit profile $X_{obs,k}$ (`dst_profile`, one integer per agent). It applies per-drug `sens` / `spec` **at the strain level**, then aggregates over strains under the `p_strain_obs` bottleneck (default = strain fitness, the spec's bacillary-load proxy; overridable scalar/dict): a drug is called resistant if any *observed* resistant strain passes sensitivity or any observed susceptible strain fails specificity. So `p_strain_obs = 1` raises detection when a phenotype is present in multiple strains, and `p_strain_obs < 1` lowers overall sensitivity via strain drop-out — exactly as specified. DST reports only the n-drug profile, never strain identities. Treatment provision keyed to the observed profile is wired via `DSTDelivery.observed_resistant(drug)`, which returns a `sim → uids` callable to feed `TxDeliveryR(eligibility=...)`. **Honest gaps:** outcomes are binary (no "indeterminate" call), and eligibility defaults to immediate (post-diagnosis); the treatment-failure-vs-new-case distinction (tracking time since last treatment initiation) is not implemented.

- Treatment monitoring:

  - Eligibility for treatment monitoring diagnostic depends on time under treatment (i.e., track time since treatment initiation).

  - Use current diagnostic class to identify those who are still bacteriologically positive in a state-dependent manner.

  - For those who remain bacteriologically positive, treatment regimen can be extended and/or changed:

    - Any extensions or changes in regimen are to be handled as a new treatment product, with eligibility conditional on the results of the treatment monitoring diagnostic.

      - Means we may need a way to prematurely stop/change an ongoing treatment regimen if someone becomes eligible (through treatment monitoring diagnostic) for another regimen

    - Treatment efficacy and risk of resistance following failure are handled as described above. Treatment efficacy parameters may need to be adjusted (external to TBsim) to reflect the conditional efficacy given prior treatment (e.g., the conditional probability of clearance in months 3-4 of an extended regimen given treatment failure from months 1-2).

> **Implementation — NOT DONE.** Treatment monitoring is not implemented: there is no monitoring diagnostic keyed to time-under-treatment, no regimen extension/switching, and no mechanism to prematurely stop or change an ongoing course (the spec's flagged "interrupt an ongoing regimen" need). `TxDeliveryR` does track a course end time (`ti_treatment_end`) and resolves at a fixed duration, which is the natural hook, but the monitoring/switching layer itself is deferred — this was the least-specified piece of the spec.

**Notation:** software should feel free to diverge from the notation specified here in whatever way makes the most sense from a code perspective. Some of the notation is currently very dependent on the drugs being a consistent order within different arrays, so we will need to figure out either how to maintain that consistent ordering or use naming/keys in an expedient way (such as via an intermediate naming system, e.g., user defines $X_{1}\  = \ \{ 1,0,0\}\  = \ "RIF"$, or position 1 = "RIF", etc.). We are open to suggestions here.

> **Implementation — divergence taken as invited.** Strains are integer ids / bitmasks and all per-drug parameters (`rel_fitness`, `resist_penalty`, `sens`, `spec`) are **name-keyed dicts** resolved through `Strains.drug_idx`, so consistent array ordering is never demanded of the user. This is exactly the spec's suggested "position 1 = RIF" intermediate-naming scheme, promoted to the primary interface.

**Testing:** It would be helpful to see:

- Tests comparing key burden results output by models before vs. after resistance is added (not necessarily burden metrics that are focused on resistance, but overall TB disease prevalence per 100,000, annual incidence of new asymptomatic disease per 100,000, annual TB mortality per 100,000). We do not expect multistrain functionality to substantially shift overall burden of disease -- but it is possible that it will, in which case it would be helpful to interrogate why/under what conditions this occurs.

- Tests demonstrating that raising/lowering key resistance parameters (such as risk of acquisition during treatment, fitness costs, relative treatment efficacy against resistant strains, and overall treatment rate) has the anticipated effect on both overall disease burden and the % of active TB that is resistant.

- For these tests, you could use the best-fitting parameter set (and model configuration) from tb_LAI_TPT if it is helpful to have a starting parameter set that produces reasonable values: <https://github.com/starsimhub/tb_LAI_TPT/blob/main/run_one_sim.py>

> **Implementation — `tests/test_resistance.py` (12 tests) + `scripts/validate_resistance_abm_vs_ode.py`.** The before/after check is a single-strain-reduction test (a `TBResistant` with `drugs=['TX']` and no resistance reproduces plain `TB` dynamics), plus a full transient ABM-vs-ODE validation against a bit-identical Python port of `ode.r` (`tbsim/compartmental/two_strain_ode.py`). Parameter-effect tests cover the specified levers: fitness-cost-driven competitive exclusion, treatment selecting for resistance, superinfection requiring σ > 0, and de-novo mixed-vs-replacement equivalence. **Honest gap:** validation used the South Africa reference-ODE parameter set, not the `tb_LAI_TPT` set; a burden-per-100,000 (prevalence / asymptomatic incidence / mortality) before-vs-after table using that set has not been produced.

**Sources:** the proposed approach is based off of these 4 papers, which may be helpful for additional context and examples:

1.  Cohen T, Lipsitch M, Walensky RP, Murray M. Beneficial and perverse effects of isoniazid preventive therapy for latent tuberculosis infection in HIV--tuberculosis coinfected populations. PNAS. 2006;103(18):7042--7047. <https://doi.org/10.1073/pnas.0600349103>

2.  Mills HL, Cohen T, Colijn C. Community-wide isoniazid preventive therapy drives drug-resistant tuberculosis: a model-based analysis. Science Translational Medicine. 2013;5(180):180ra49. <https://doi.org/10.1126/scitranslmed.3005260>

3.  Kunkel A, Crawford FW, Shepherd J, Cohen T. Benefits of continuous isoniazid preventive therapy may outweigh resistance risks in a declining tuberculosis/HIV coepidemic. AIDS. 2016;30(17):2715--2723. <https://doi.org/10.1097/QAD.0000000000001235>

4.  Kunkel A, Colijn C, Lipsitch M, Cohen T. How could preventive therapy affect the prevalence of drug resistance? Causes and consequences. Philosophical Transactions of the Royal Society B. 2015;370(1670):20140306. <https://doi.org/10.1098/rstb.2014.0306>
