# TBsim Resistance — Spec Updates Since Implementation

This document lists the changes in the technical specification **since the version captured in `tbsim-resistance-tech-spec.md` was implemented**. It is not a restatement of the full spec — only the deltas that require new or changed model behavior.

Every change below stems from a single reversal: the implemented spec **blocked** superinfection with two identical strains, whereas the updated spec **allows** it and introduces a per-strain **counter** (the number of instances of each strain an agent carries). The counter then propagates through progression, clearance, random acquisition, treatment, and DST.

## 1. Strain competition — allow identical-strain superinfection (reversal)

**Was (implemented):** "We do *not* allow superinfection with 2 identical strains." The model blocked re-exposure to a carried strain and only tracked a per-step `new_blocked_superinf` analyzer counting how often the block fired.

**Now:** "We **do** allow superinfection with 2 identical strains," tracked via a per-strain count.

New behavior to implement:

- **Add a per-strain counter.** For each agent, count how many instances of each strain they carry. A newly-infected agent always starts with a strain count of **1** for the founding strain, even if the infecting agent carries >1 copy of that strain.
- **Counter feeds the transmission multinomial.** The probability that a given strain is passed is proportional to `count × fitness` (counts and fitness costs both act as multipliers in the multinomial draw). The *overall* probability of transmitting any strain is still governed by the most-fit strain and is **not** affected by strain count.
- **Scope of the counter (new note).** The same-strain counter only affects (a) transmission and (b) — if $p\_multi < 1$ — the progression bottleneck for multi-strain agents. It does **not** affect transition rates (progression, clearance), the overall transmission probability from mono- or superinfected agents, DST results, treatment effectiveness, or additional resistance acquisition. For DST, treatment, and additional resistance acquisition, same-strains are always **coupled** (see sections below).
- **Clock reset (clarification).** Once time-varying progression is added, reinfection with *any* strain (same or different from the current profile) should reset the progression clock.

**Reworked example.** Agent $A$ with $\mathbf{Y}_A = \{\{0,0,0\}\}$ contacts superinfected $B$ with $\mathbf{Y}_B = \{\{0,0,0\}, \{1,0,1\}\}$, each strain count 1:

- If the passed strain is the one $A$ already has ($\{0,0,0\}$, probability $\frac{1}{1 + r_{RIF}r_{FQ}}$), $A$'s count for $\{0,0,0\}$ now moves from **1 → 2** (previously: no change).
- If $B$ instead had a count of **2** for the susceptible strain, the probability $A$ is infected with it becomes $\frac{2}{2 + r_{RIF}r_{FQ}}$, and the probability $A$ becomes superinfected becomes $\frac{r_{RIF}r_{FQ}}{2 + r_{RIF}r_{FQ}}$.

**Removed from spec:** the old rationale that blocking identical-strain superinfection biases toward low-prevalence strains, and the previous instruction to build a blocking analyzer "for now" (now reframed as a past request, no longer the active design).

## 2. Progression to disease — counter selects the progressing strain

**New note added:** when $p\_multi < 1$ and the draw is such that an agent does *not* retain all strains upon progression, the strain counter determines which strain progresses. Example: an agent with 1 susceptible + 1 resistant strain has a 50/50 chance of progressing with only the susceptible vs. only the resistant strain; an agent with 2 susceptible + 1 resistant strain has a 2/3 vs. 1/3 chance.

## 3. Clearance — reset counters

**Changed:** natural clearance/resolution (INFECTION→CLEARED, NON-INFECTIOUS→CLEARED) clears all strains **and resets all infection counters to 0** (previously: cleared all strains, with no counter to reset).

## 4. Random (de novo) acquisition — couple identical strains

**New bullet added:** if an agent carries multiple copies of the same strain (e.g., 2 susceptible + 1 resistant), the same-strains are **coupled** — either all copies of a strain acquire the same resistance or none do. Strain count does **not** affect the probability of resistance acquisition.

## 5. Treatment & selective acquisition — reset counters and couple identical strains

Two additions:

- **Successful treatment resets the strain counter to 0** for the treated strain, regardless of its starting count (i.e., treatment clears *all* copies of a strain if successful).
- **Same-strains are coupled** (using the counter): if an agent has 2 susceptible + 1 resistant strain, either both susceptible copies are cured or neither is; and if not cured, either both acquire the same resistance or neither does. Strain count does **not** affect treatment effectiveness or the probability of resistance acquisition.

## 6. DST — counter has no effect

**New bullet added:** DST behaves no differently when an agent carries multiple copies of the same strain — the strain counter does not affect DST sensitivity or specificity.
