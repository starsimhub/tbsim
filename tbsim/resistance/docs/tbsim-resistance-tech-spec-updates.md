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

**Allow for multi-strain infections (i.e., superinfections):**

- Each agent$\ k$ can be infected with multiple strains $\mathbf{X}_{\mathbf{j}}$. Define the strain profile of agent $k$ as $\mathbf{Y}_{\mathbf{k}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,k}}\mathbf{,\ \ldots,\ }\mathbf{X}_{\mathbf{m,k}} \right\}\mathbf{,\ }where\ \mathbf{X}_{\mathbf{1,k}}\mathbf{=}\left\{ x_{1,1},\ x_{2,1},\ \ldots,\ x_{n,1} \right\},\ \mathbf{X}_{\mathbf{m,k}}\mathbf{=}\left\{ x_{1,m},x_{2,m},\ldots,x_{n,m} \right\},$ etc.

- Example, for drugs $\{ RIF,\ BDQ,\ FQ\}$ as above:

  1.  Agent A is infected with 1 strain that is resistant to RIF only: $\mathbf{Y}_{\mathbf{A}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,A}} \right\}\mathbf{= \{}\left\{ 1,0,0 \right\}\}$

  2.  Agent B is infected with 2 strains -- 1 pan-susceptible strain and 1 strain resistant to RIF and FQs: $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\{\left\{ 0,0,0 \right\},\ \left\{ 1,0,1 \right\}\}$

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

- We **do** allow superinfection with 2 identical strains.

  1.  Implication: this avoids bias towards favoring low-prevalence strains, since an agent is more likely to be infected with high-prevalence strains multiple times than rare, low-prevalence strains. Because we assume only 1 strain gets transmitted per transmission event, not allowing for superinfection with the same strain could result in somewhat less transmission of high-prevalence strains/more transmission of low-prevalence strains, relative to a scenario in which we allow for superinfection with identical strains.

  2.  How to do this: "count" how many instances of each strain an agent has, and use these counts to determine which strains progress to disease and/or get transmitted. We could reassess this option later (or variations on this option with different functional relationships) if we are concerned about bias after implementing the current specification.

      - Example: an agent is infected with 1 susceptible and 1 resistant strain, with no fitness costs. If they progressed to disease, there would be a 50/50 chance per transmission event of the susceptible vs. resistant strain being passed on. If this agent becomes reinfected with the susceptible strain before progressing to disease, they now carry 2 susceptible and 1 resistant strains. If they subsequently progress to disease, there is instead a 2/3 vs. 1/3 chance per transmission event of the susceptible vs. resistant strain being passed on. The above logic of fitness costs still applies as in the table on page 2 (so both fitness costs and strain counts serve as multipliers here for the multinomial draws determining which strain is passed, and the overall probability of transmission with any strain is still based on the probability with the most-fit strain and is not affected by strain count).

      - The newly-infected agent will always start with a strain count of 1 (representing transmission of one founding strain), even if the infecting agent has \>1 strain of a given phenotype.

  3.  This is the main decision we have not tested with an ODE model but which we might want to test in the future.

      - We had previously requested only including an analyzer that allows us to quantify how much this comes up. (That is, for each agent, count the number of times an agent would become infected if we did not block superinfection with 2 identical strains (e.g. going by the ordering/random number draws in TBsim, $B$ contacts $A,$ then $B$ infects $A$, then $B$ infects $A$ with strain $X$, but $A$ is already infected with strain $X$, so no infection occurs).)

  4.  Note that for appropriate consideration of time-varying risk of progression (see section below), we may also wish to track the time since an individual was successfully exposed to a strain of their same infection (i.e., would still reset time since infection), even if we make no other changes to their strain profile or disease state.

      - Note: this still applies. Once we add time-varying progression, reinfection with any strain (same or different from current strain profile) should reset the progression clock.

  5.  Note: the same-strain counter only affects the transmission and, if $p\_ multi < 1$, progression bottlenecks from individuals with multi-strain infections. It does not affect transition rates (such as progression or clearance), the overall transmission probability from mono-infected or superinfected individuals, DST results, treatment effectiveness, or additional resistance acquisition. For DST, treatment, and additional resistance acquisition, same-strains will always be coupled (details in the relevant sections below).

- Example: consider Agent $A$ who is already infected such that $\mathbf{Y}_{\mathbf{A}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,A}} \right\} = \{\left\{ 0,0,0 \right\}\}$. Conditional upon contact with super-infected agent $B$, where $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\ \{\left\{ 0,0,0 \right\},\ \left\{ 1,0,1 \right\}\}$, and the strain count for each strain is 1:

  1.  Probability that agent $A$ becomes infected from agent $B$ equals $rr\_ reinfection\_ inf\ *\beta\$

  2.  The probability that the infecting strain is the same strain agent $A$ already has (strain $\mathbf{X}_{\mathbf{1,B}}\mathbf{=}\left\{ 0,0,0 \right\}\mathbf{\ }$**)** is $\frac{1}{1 + r_{RIF}r_{FQ}}$. If this occurs, agent A's strain count for strain $\{ 0,0,0\}$ moves from 1 to 2.

  3.  Agent $A$ becomes superinfected (acquires strain $\mathbf{X}_{\mathbf{2,B}}\mathbf{=}\{ 1,0,1\}$) with probability $\frac{r_{RIF}r_{FQ}}{1 + r_{RIF}r_{FQ}}$

  4.  If agent $B$ with $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\ \{\left\{ 0,0,0 \right\},\ \left\{ 1,0,1 \right\}\}$ had a strain count of 2 for the susceptible strain $\left\{ 0,0,0 \right\}$, the probability that agent A is infected with the susceptible strain would be $\frac{2}{2 + r_{RIF}r_{FQ}}$. The probability that agent A becomes superinfected would then be $\frac{r_{RIF}r_{FQ}}{2 + r_{RIF}r_{FQ}}$.

**Progression to disease**

In addition to multi-strain infection, we also allow for multi-strain TB disease. Balancing simplicity with accuracy to the extent possible in the context of the somewhat more complex natural history model in TBsim, we assume the following:

- We do *not* track separate natural histories for each strain. Natural history remains an agent-level characteristic, not a strain-level characteristic.

- Progression risk and other transition rates between TB natural history states are the same regardless of how many strains an agent is infected with.

  1.  If we implement time-varying risk of disease progression, each new infection should 'reset the clock' on an individual's time since infection, which can functionally allow for different risk of progression. Ideally, this would also include successful exposure to strains with which the agent is already infected, even if we do not track those as explicit infection events. However, the number of previously-infecting strains does not affect how progression risk changes with time since infection (i.e., someone going from 0 -\> 1 infecting strains or from 1 -\> 2 infecting strains will follow the same temporal pattern of progression risk).

- If an agent in the INFECTION state, with superinfection, progresses to NON-INFECTIOUS, they retain all strains.

- If an agent in the INFECTION or NON-INFECTIOUS state progresses to ASYMPTOMATIC, they retain all strains with some probability $p\_ multi$, and otherwise each strain has an equal probability of being the single strain that progresses

  1.  We will set $p\_ multi = 1$, but can assess the importance of this parameter in one-way sensitivity analysis.

  2.  Note: we would again like to use the strain counter here to determine which strain is progressed. For example, if an agent is infected with 1 susceptible and 1 resistant strain, and $p\_ multi < 1$, and the binomial draw from $p\_ multi$ is such that this agent does not retain all strains upon progression, then there is a 50/50 chance that the agent progresses with only the susceptible strain vs. only the resistant strain. If the agent instead was infected with 2 susceptible strains and 1 resistant strain, there would be a 2/3 vs. 1/3 chance that the agent progresses with only the susceptible strain vs. only the resistant strain.

**Clearance**

- We assume that if a superinfected agent naturally (i.e., absent intervention) clears their infection (INFECTION to CLEARED) or spontaneously resolves (NON-INFECTIOUS to CLEARED) all strains are cleared and all infection counters are reset to 0.

  1.  Rationale: biologically, natural clearance/resolution represents an immune response that should equally apply to multiple strains, while clearance/recovery via TPT or treatment selectively clear only certain strains because of their resistance to drugs in the TPT/treatment regimen.

- Clearance rates (and other transition rates between TB states in the natural history) are not affected by superinfection.

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

  - However, if an agent is infected with multiple of the same strain (e.g., 2 susceptible and 1 resistant), the same-strains are coupled (i.e., either both of the agent's susceptible strains acquire the same type of resistance or neither acquires any resistance). Strain count does not affect the probability of resistance.

- Example: if agent $B$ currently has $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\ \{\left\{ 0,0,0 \right\},\ \left\{ 1,1,0 \right\}\}$ and is transitioning from INFECTED to NON-INFECTIOUS:

  - Each strain could independently acquire resistance to FQs with probability $p\_ rand_{FQ}$

  - Strain \#1, $\mathbf{X}_{\mathbf{1,B}}$ can also acquire resistance to RIF with probability $p\_ rand_{RIF}$ and BDQ with probability $p\_ rand_{BDQ}$

  - Say the random draws that determine resistance acquisition result in only strain \#1, $\mathbf{X}_{\mathbf{1,B}}$**,** acquiring BDQ resistance. Then:

    - Option 1 (Superinfection): agent $B$'s updated strain profile is: $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{3,B}} \right\}\mathbf{=}\ \{\left\{ 0,0,0 \right\},\ \left\{ 1,1,0 \right\},\ \left\{ 0,1,0 \right\}\}$ -- that is, a new version of strain \#1 with BDQ resistance (i.e., strain \#3) is added to agent $B$'s resistance profile.

    - Option 2 (Replacement): agent $B$'s updated strain profile is: $\mathbf{Y}_{\mathbf{B}}\mathbf{=}\left\{ \mathbf{X}_{\mathbf{1,B}}\mathbf{,\ }\mathbf{X}_{\mathbf{2,B}} \right\}\mathbf{=}\ \{\left\{ 0,1,0 \right\},\ \left\{ 1,1,0 \right\}\}$ -- that is, a strain with BDQ resistance replaces strain \#1.

**Treatment & (Selective) Acquisition:**

- Treatment efficacy:

  - For a given treatment regimen $l$, define the clinical efficacy against each possible strain $\mathbf{X}_{\mathbf{1}}\mathbf{,\ldots}\mathbf{X}_{\mathbf{m}}\mathbf{\ }$ as $\mathbf{T}_{\mathbf{l}} = \{ t_{1,l},\ldots,t_{m,l}\}$. For example, $t_{i,l}$ will be regimen $l's$ full efficacy for strains without resistance to drugs/drug classes included in the regimen, and less than full efficacy for strains with resistance to drugs/drug classes in the regimen.

  - This approach also allows us to vary efficacy for shorter regimens containing the same drugs/classes.

  - Currently, treatment in TBsim includes an adherence argument that can adjust the clinical efficacy of a regimen. During multistrain infection, we would like to be able to use this adherence parameter in such a way that it can induce agent-level correlation in treatment efficacy across strains, by allowing the user to change adherence from a regimen-level value to a regimen-level distribution that can vary by agent and would get applied across all strains during a given treatment course.

  - Successful treatment of a strain resets the strain counter to 0, regardless of the starting value of the strain counter (i.e., treatment clears all infections of a given strain if successful).

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

- For agents infected with multiple of the same strain (using the strain counter), same strains are coupled. For example, if an agent has 2 susceptible strains and 1 resistant strain, either both of the susceptible strains are cured or neither is cured, and if not cured, either both of the susceptible strains acquire the same type of resistance or neither acquires any resistance. Strain count does not affect the effectiveness of treatment nor the probability of resistance acquisition.

**TPT**

- For the most part, we can take a similar approach to TPT as TB treatment.

- Efficacy can be modeled very similarly, as varying by regimen and resistance/strain.

  - Because we will model superinfection, this means that TPT can clear a drug-susceptible strain, increasing the risk that a resistant strain goes on to <u>progress</u> to active TB (i.e., the dynamics in Ted's papers cited at the end of this document) and <u>get transmitted</u> to others. Note that if we set $p\_ multi$ above to 1, only the transmission dynamic and not the progression dynamic will be present.

- Acquisition can also be modeled similarly: some percentage of agents for whom TPT was not effective (provided neither clearance nor longer-term protection from progression) have a probability of acquiring resistance to the drugs/classes included in the TPT regimen. This probability should vary by TB state at time of failure (e.g., highest risk for ASYMPTOMATIC and SYMPTOMATIC agents, very low risk for INFECTED agents, low-medium risk for NONINFECTIOUS agents)

**Diagnostics & Treatment Modification:**

- DST:

  - (Likely a separate class from current diagnostics for TB state) Allow for a diagnostic test to create observed resistance profile per agent $\mathbf{X}_{\mathbf{obs,k}} = \left\{ \ x_{obs,1,k},\ \ldots x_{obs,n,k} \right\},\$

    - Testing follows logic of current diagnostic class, in that we define sensitivity/specificity for each $x_{i,k}$ to have user-defined outcomes $x_{obs,i,k}$ (likely defaults, positive, negative, indeterminant) for whether DST detected resistance to drug $i$ for agent $k$.

      - Estimating the probability of detecting a given phenotype in mixed infections requires consideration of the probability that all (or multiple) strains are observed, due to possible bottlenecks at time of sample collection, culture, or other sample processing.

      - First, we should apply DST sensitivity and specificity at the strain level to indicate whether each strain-specific phenotype is observed. Then, we will aggregate these observed profiles to determine whether any $x_{obs,i,k} = 1$. At this step, there should be an additional probability (e.g., p_strain_obs) that the profile of a given strain is observed at all. When p_strain_obs=1 (i.e. we assume all strains are observed in the test), this will create a higher probability that a given resistance phenotype is detected if it exists in more than one infecting strain. However, when p_strain_obs \< 1, overall DST sensitivity will be reduced due to this strain drop-out.

      - By default, set \`p_strain_obs\` equal to strain fitness, as a proxy for within-host bacillary loads and culture growth potential. Allow p_strain_obs to be re-defined in a strain agnostic way.

    - DST will not identify specific strains; that is, the agent resistance profile will have only *n* objects for the chosen drugs/classes

    - DST similarly does not behave any differently if agent has multiple of the same strain (strain counter does not affect the sensitivity or specificity of DST).

  - Eligibility for DST can either be immediate (i.e., immediately following diagnosis) or dependent on treatment failure.

    - Should track time since last treatment initiation to inform whether later treatment is managed as treatment failure, with need for DST/second-line treatments, or as a new case.

  - Treatment provision can be dependent on observed diagnostic profile

    - Similar logic to current intervention structure -- certain treatments only provided to eligible individuals that match a certain profile

- Treatment monitoring:

  - Eligibility for treatment monitoring diagnostic depends on time under treatment (i.e., track time since treatment initiation).

  - Use current diagnostic class to identify those who are still bacteriologically positive in a state-dependent manner.

  - For those who remain bacteriologically positive, treatment regimen can be extended and/or changed:

    - Any extensions or changes in regimen are to be handled as a new treatment product, with eligibility conditional on the results of the treatment monitoring diagnostic.

      - Means we may need a way to prematurely stop/change an ongoing treatment regimen if someone becomes eligible (through treatment monitoring diagnostic) for another regimen

    - Treatment efficacy and risk of resistance following failure are handled as described above. Treatment efficacy parameters may need to be adjusted (external to TBsim) to reflect the conditional efficacy given prior treatment (e.g., the conditional probability of clearance in months 3-4 of an extended regimen given treatment failure from months 1-2).

**Notation:** software should feel free to diverge from the notation specified here in whatever way makes the most sense from a code perspective. Some of the notation is currently very dependent on the drugs being a consistent order within different arrays, so we will need to figure out either how to maintain that consistent ordering or use naming/keys in an expedient way (such as via an intermediate naming system, e.g., user defines $X_{1}\  = \ \{ 1,0,0\}\  = \ "RIF"$, or position 1 = "RIF", etc.). We are open to suggestions here.

**Testing:** It would be helpful to see:

- Tests comparing key burden results output by models before vs. after resistance is added (not necessarily burden metrics that are focused on resistance, but overall TB disease prevalence per 100,000, annual incidence of new asymptomatic disease per 100,000, annual TB mortality per 100,000). We do not expect multistrain functionality to substantially shift overall burden of disease -- but it is possible that it will, in which case it would be helpful to interrogate why/under what conditions this occurs.

- Tests demonstrating that raising/lowering key resistance parameters (such as risk of acquisition during treatment, fitness costs, relative treatment efficacy against resistant strains, and overall treatment rate) has the anticipated effect on both overall disease burden and the % of active TB that is resistant.

- For these tests, you could use the best-fitting parameter set (and model configuration) from tb_LAI_TPT if it is helpful to have a starting parameter set that produces reasonable values: <https://github.com/starsimhub/tb_LAI_TPT/blob/main/run_one_sim.py>

**Sources:** the proposed approach is based off of these 4 papers, which may be helpful for additional context and examples:

1.  Cohen T, Lipsitch M, Walensky RP, Murray M. Beneficial and perverse effects of isoniazid preventive therapy for latent tuberculosis infection in HIV--tuberculosis coinfected populations. PNAS. 2006;103(18):7042--7047. <https://doi.org/10.1073/pnas.0600349103>

2.  Mills HL, Cohen T, Colijn C. Community-wide isoniazid preventive therapy drives drug-resistant tuberculosis: a model-based analysis. Science Translational Medicine. 2013;5(180):180ra49. <https://doi.org/10.1126/scitranslmed.3005260>

3.  Kunkel A, Crawford FW, Shepherd J, Cohen T. Benefits of continuous isoniazid preventive therapy may outweigh resistance risks in a declining tuberculosis/HIV coepidemic. AIDS. 2016;30(17):2715--2723. <https://doi.org/10.1097/QAD.0000000000001235>

4.  Kunkel A, Colijn C, Lipsitch M, Cohen T. How could preventive therapy affect the prevalence of drug resistance? Causes and consequences. Philosophical Transactions of the Royal Society B. 2015;370(1670):20140306. <https://doi.org/10.1098/rstb.2014.0306>
