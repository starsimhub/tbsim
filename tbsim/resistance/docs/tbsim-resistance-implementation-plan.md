# TBsim Drug-Resistance / Multi-Strain: Starsim Translation & Implementation Plan

**Status:** Design proposal for review. **Source:** `tbsim-resistance-tech-spec.docx` (Ryckman, Grantz, Cohen, et al.). **Target:** `tbsim` on Starsim ≥ 3.5.0. **Audience:** TB modelers + Starsim/TBsim software.

This document translates the resistance technical specification into concrete Starsim/TBsim constructs, proposes an implementation architecture, and lays out the decisions the team needs to make. It is deliberately explicit about where the spec maps cleanly onto existing TBsim patterns and where Starsim's single-strain assumptions force a design choice. Notation follows the spec but renames freely per the spec's own invitation ("software should feel free to diverge"). Each open decision is tagged **[D#]** and collected in §11.

---

## 1. Executive summary

The spec adds **phenotypic drug resistance** to TBsim by letting each agent carry **one or more strains**, where a strain is a binary vector of resistance/susceptibility to a small set of drugs/classes (starting with RIF and BDQ; later INH, FQ, companions). This touches five subsystems: (1) strain bookkeeping, (2) transmission (fitness costs, which strain is passed, superinfection), (3) natural history (strain retention on progression, clearance, de-novo acquisition), (4) treatment/TPT (strain-specific efficacy, selective clearing, acquisition-on-failure), and (5) diagnostics (DST, treatment monitoring, regimen switching).

**The single most consequential fact:** Starsim 3.5.1 has *no* native support for strains/variants/superinfection. `ss.Arr` states are strictly 1-D and single-dtype (one scalar per agent); there is no 2-D or variable-length per-agent state. Susceptibility is gated by a single boolean (`rel_sus = susceptible.raw * rel_sus.raw`), so an infected agent is *un-infectable by construction* — superinfection is blocked until we change that gating. And when multiple infectious neighbors expose the same target in one step, Starsim collapses them to a single source (first in edge order). Every architectural choice below flows from these three facts.

**Recommended architecture (one sentence):** keep the existing agent-level `TB` state machine unchanged, add a **strain-membership layer** stored as a single integer bitmask per agent (`ss.IntArr`, one bit per possible strain), and make the model strain-aware by (a) setting state-dependent `susceptible`/`rel_sus`/`rel_trans` so the *existing* CRN force-of-infection engine does the transmission arithmetic, and (b) overriding `set_prognoses()` to choose the transmitted strain and enforce superinfection rules. Diagnostics, treatment, and TPT extend the existing product/delivery pattern with strain-aware products.

**Effort:** roughly 4–6 focused weeks for a first complete, tested implementation (§9, §12), front-loaded on transmission and the strain data model, with treatment-monitoring/regimen-switching (§6.11) the largest genuinely new piece of machinery and the least-specified part of the spec.

---

## 2. The spec in plain language

- **Strain** = an `n`-bit resistance profile `x_j = (x_{1j},…,x_{nj})`, `x_{ij}∈{0,1}`. For `n` drugs there are `m = 2^n` possible strains (n=2 → 4 strains; n=4 → 16).
- **Agent strain profile** `Y_k` = the *set* of strains an agent currently carries; more than one ⇒ superinfection.
- **Transmission:** susceptibles can only acquire strains the infector already carries (no resistance emerges during transmission). Each resistant phenotype `i` carries an independent multiplicative fitness cost `r_i∈[0,1]` on the force of infection. A single-strain infector transmits at `β·fitness(strain)`. A superinfected infector transmits at `β·max fitness` over its strains, and *which* strain is passed is a draw ∝ each strain's fitness. Only one strain per transmission event.
- **Superinfection protection (strain competition):** an agent's susceptibility to a *new* strain depends on state — `rr_reinfection_inf` in INFECTION, `rr_reinfection_non` in NON_INFECTIOUS, and 0 in ASYMPTOMATIC/SYMPTOMATIC (no superinfection while diseased). Protection is strain-agnostic and count-agnostic. No superinfection with an identical strain (build an analyzer to count how often this blocks an event).
- **Progression:** natural history stays *agent-level*, not strain-level; rates don't depend on strain count. On INFECTION/NON_INFECTIOUS → ASYMPTOMATIC, keep all strains with probability `p_multi` (default 1); otherwise a single, equiprobable strain progresses.
- **Clearance:** natural clearance/resolution (→ CLEARED) removes *all* strains (immune-mediated, strain-agnostic).
- **De-novo (random) acquisition:** a one-time, per-strain, per-drug probability `p_rand_i` of gaining resistance at INFECTION → NON_INFECTIOUS/ASYMPTOMATIC (background mutation; likely 0 for RIF, small for BDQ). Superinfection-vs-replacement left to software (ODE showed no difference).
- **Treatment & selective acquisition:** regimen `l` has strain-specific clinical efficacy `t_{il}`; adherence becomes an agent-level draw from a regimen distribution applied across all of that agent's strains (correlated outcomes). Clearing only some strains leaves the agent in the same TB state with the surviving subset. On unsuccessful treatment, each surviving drug-susceptible strain independently acquires resistance with probability `q_{li}` (>0 only for drugs in the regimen), scaled by a state-dependent relative risk; acquisition is **replacement**.
- **TPT:** same structure as treatment — strain-specific efficacy and acquisition-on-failure. Clearing a susceptible strain can hand the resistant strain a competitive advantage (the Cohen/Mills/Kunkel dynamics).
- **Diagnostics:** a **DST** produces an *observed* resistance profile `X_obs,k` (n bits) with per-drug sensitivity/specificity applied at the strain level, then aggregated across strains, with a per-strain observation probability `p_strain_obs` (default = strain fitness) modeling within-host/culture bottlenecks. Treatment provision can be conditioned on `X_obs`. **Treatment monitoring** tracks time on treatment, re-tests bacteriological positivity, and can extend/switch a regimen mid-course.

---

## 3. The central architectural problem: Starsim is single-strain

Verified against the installed Starsim 3.5.1 source (`/home/cliffk/idm/starsim/starsim`, single-module layout: `diseases.py`, `networks.py`, `arrays.py`, `distributions.py`, `connectors.py`):

1. **No strain/variant/superinfection support anywhere.** Two forward-looking comments (`modules.py:685`, `people.py:310`) name "multiple genotypes/strains" as a *reason* the state registry is overridable — i.e., Starsim anticipates but does not implement this. We build it.
2. **`ss.Arr` is 1-D, single-dtype** (`arrays.py:171,241,582`). No 2-D `(n_agents×m)` state and no variable-length per-agent state. `FloatArr`/`IntArr`/`BoolArr`/`BoolState` all assume one scalar per agent. Storing "a set of strains per agent" is therefore *our* problem to solve (§4).
3. **Susceptibility gate blocks superinfection.** In `Infection.infect()` (`diseases.py:257`), `rel_sus = susceptible.raw * rel_sus.raw` and `rel_trans = infectious.raw * rel_trans.raw`; the per-edge Bernoulli success is `rel_trans[src]·rel_sus[trg]·beta_edge > u` (`diseases.py:229-248`). An already-infected agent has `susceptible=False`, so it can never be a transmission target. Superinfection requires making mid-history agents `susceptible=True` with a reduced `rel_sus`.
4. **`sources` is one-to-one with new cases but multi-source is collapsed.** `infect()` finalizes with `new_cases, inds = new_cases.unique(return_index=True); sources = concatenate(sources)[inds]` (`diseases.py:304-312`) — for a target exposed by several infectors in one step, only the *first in edge order* survives. So the infecting source (and thus the candidate strain) is available in `set_prognoses(uids, sources)`, but only one per target per step.
5. **No `ss.multinomial`; `ss.choice` won't take per-agent `p`.** `ss.choice(a=k, p=...)` (`distributions.py:1501`) is CRN-safe but requires constant `p` across the call. However, **TBsim already ships `tbsim.choice2d`** (`tb.py:509`) — a per-agent-probability categorical sampler via `ppf` — which is exactly the tool for "draw which strain, with per-event probabilities." The transmission RNG is `ss.multi_random('source','target')` (`diseases.py:135`); a dedicated strain stream can be `ss.multi_random('source','target','strain')`.

The good news: points (3)–(5) mean we can **reuse the existing CRN transmission engine** rather than reimplement it, by feeding it state-dependent `susceptible`/`rel_sus`/`rel_trans` and doing all strain logic in `set_prognoses()`. §6.2 shows this reproduces the spec's worked examples exactly.

---

## 4. Proposed data model for strains

### 4.1 Strain identity and the fitness/profile lookups

A strain *is* its resistance profile. Enumerate all `m = 2^n` profiles as integer IDs `0…m-1`, where bit `i` of the ID is resistance to drug `i` (drug order fixed by a user-supplied list). This makes strain ID ↔ profile a pure bit operation and needs no per-strain storage.

A small **`Strains` registry** (plain helper object, not a Starsim module) holds:
- `drugs`: ordered list, e.g. `['RIF','BDQ']` (index = bit position; resolves the spec's "consistent ordering" concern **[D-notation]** by using names → indices, so `strain_id` bit 0 = RIF, etc.).
- `profile_matrix`: `(m × n)` bool, precomputed from bit decomposition.
- `rel_fitness`: dict drug→cost `r_i`; `fitness_by_id`: length-`m` float = `∏_{i: resistant} r_i` (pan-susceptible strain 0 → fitness 1.0).
- Helpers: `resistant_to(id, drug)`, `add_resistance(id, drug) -> new_id` (set a bit), `phenotype_any(mask)` (OR of profiles over an agent's carried strains → the agent's aggregate n-bit phenotype).

### 4.2 Per-agent strain membership — **[D1]**

Because an agent carries a *subset* of the `m` possible strains, and `m` is small for realistic drug sets, the recommended representation is a **single bitmask integer per agent**:

- `ss.IntArr('strain_mask', default=0)` — bit `j` set ⇒ agent carries strain `j`. `strain_mask=0` ⇒ uninfected.

Why this wins:
- One Starsim-managed state → automatic grow/shrink on births/deaths/migration; no parallel bookkeeping.
- Superinfection = bitwise OR; "has strain j?" = `(mask >> j) & 1`; identical-strain blocking is automatic; clearing a strain = clearing a bit; clearing all = set to 0. Count-agnostic protection needs no extra state.
- Vectorized cross-strain math is cheap: decode to a `(n_agents × m)` bool view with `(mask[:,None] >> arange(m)) & 1`, then `× fitness_by_id` and reduce. For n≤4, `m≤16`; even n=6 (`m=64`) fits a uint64.

Alternatives considered:
- **`m` parallel `BoolState`s** (`has_strain_0…`). More conventional and gives free `n_has_strain_j` results, but clutters the state registry and is awkward to reduce across strains. Viable fallback if the team prefers explicit states. (Starsim's documented "multiple genotypes" escape hatch — a list of states + `state_list` override, `modules.py:677` — is the same idea.)
- **One `ss.Infection` module per strain** (`m` modules, cross-strain `Connector`). Starsim explicitly anticipates this (`people.py:310`), and it would reuse the FOI per strain — *but* it contradicts the spec's core assumption that **natural history is agent-level, not strain-level** (each module carries its own state machine), and `m=2^n` modules is heavy. **Reject**, but it's the natural design *if* the team ever wants strain-level natural histories.
- **Fixed "strain slots"** (`K_max` IntArrs holding strain IDs). Memory-optimal when superinfection is rare (`K_max≈2–3`), but membership tests and blocking require scanning slots. Only worth it if `n` grows large enough that `m` bits become unwieldy (n≳6).

**Recommendation: `strain_mask` bitmask** for n≤~6; revisit a sparse/slot representation only if the drug set grows beyond that.

### 4.3 What stays agent-level (unchanged)

`TB.state`, all transition rates, `rr_activation/clearance/death`, `rr_reinfection`, and `ti_infected` remain exactly as today. Resistance is an *overlay*: `strain_mask` says which strains; `TB.state` says where in the natural history the agent is. `infected ⇔ strain_mask != 0` becomes the single source of truth linking the two (replacing today's `infected = ~isin(state, [SUSCEPTIBLE, CLEARED, terminal])`, which stays equivalent for single-strain).

---

## 5. Parameter & notation mapping

| Spec symbol | Proposed name | Default | Where it lives | Notes |
|---|---|---|---|---|
| drugs/classes, order | `drugs` (list) | `['RIF','BDQ']` | `Strains` | name→bit-index; extend to INH/FQ later |
| `x_{ij}` | (bit `i` of strain id `j`) | — | derived | no storage |
| `m = 2^n` | `strains.m` | derived | `Strains` | |
| `r_i` | `rel_fitness` (dict) | e.g. `{RIF:0.5, BDQ:0.8}` | `Strains` | multiplicative, independent |
| `β` | `beta` | existing | `TB.pars` | unchanged; see beta-rescaling §8.3 |
| `Y_k` | `strain_mask[k]` | 0 | `TB` state | bitmask (§4.2) |
| `rr_reinfection_inf` | `rr_reinfection_inf` | `= rr_reinfection_rec` (0.21) | `TB.pars` | new; INFECTION superinfection |
| `rr_reinfection_non` | `rr_reinfection_non` | `= rr_reinfection_inf` | `TB.pars` | new; NON_INFECTIOUS superinfection |
| `rr_reinfection_asy/sym` | `rr_reinfection_asy/sym` | 0 | `TB.pars` | **[D4]** Option 1 (parametric) vs hard-code |
| `p_multi` | `p_multi` | 1.0 | `TB.pars` | strain retention on progression to ASY |
| `p_rand_i` | `p_rand_acq` (dict drug→p) | `{RIF:0, BDQ:small}` | `TB.pars` | one-time at INFECTION→NON/ASY |
| `t_{il}` | regimen `efficacy_by_strain` | derived/explicit | `TxRegimen` product | **[D7]** penalties vs table |
| adherence | `p_adherence` (regimen dist) | existing scalar → dist | `TxRegimen` | **[D8]** agent-level draw, all strains |
| `q_{li}` | `q_acq` (dict drug→p) | regimen-specific | `TxRegimen` | >0 only for drugs in regimen |
| state RR on `q` | `rr_acq_by_state` | `{ASY:1, SYM:1, else:0}` | `TxRegimen`/delivery | multiplier on `q`, strain/regimen-agnostic |
| DST sens/spec | `sens`, `spec` (per drug) | per drug | `DST` product | applied at strain level |
| `p_strain_obs` | `p_strain_obs` | `= fitness(strain)` | `DST` product | culture bottleneck; redefinable strain-agnostically |
| `X_obs,k` | `dst_profile[k]` (n-bit) + `dst_tested` | 0 / False | `DSTDelivery` | observed profile per agent |

---

## 6. Component-by-component design

Each subsection: **Spec → Starsim mapping → Pros/Cons/Decision.**

### 6.1 Strain registry & agent state

**Mapping.** Add the `Strains` registry (§4.1) and `strain_mask` (§4.2) to a resistance-aware TB class. Two ways to package it **[D-packaging]**:
- **(a) Subclass `TBResistant(TB)`** adding strain state + overriding the transmission/transition hooks. Keeps base `TB` pristine; single-strain users are unaffected.
- **(b) Fold into `TB`** with `drugs=None` meaning "single-strain" (all new code guarded by `if self.strains is not None`). Avoids a parallel class but complicates the base model everyone uses.

Either way, `TB.step()` is currently monolithic; strain logic must interleave with the existing transitions. **Recommend refactoring `TB.step()` into small overridable hooks first** — `step_transmission()`, `step_transitions()`, `step_bookkeeping()` — then implementing (a) `TBResistant`. This aligns with the current `perf-refactor` branch's spirit and keeps the diff reviewable. (Decision (a)+hooks recommended; (b) if the team wants resistance to be first-class in the one and only `TB`.)

### 6.2 Transmission: fitness, superinfection, which-strain, blocking — the crux

**Spec.** Overall per-contact rate uses the *fittest* strain of the infector; the transmitted strain is drawn ∝ each strain's fitness; only one strain per event; can't acquire a strain you already have; susceptibility to a new strain is state-dependent.

**Mapping — reuse the CRN engine, do strain logic in `set_prognoses`.** In `step_bookkeeping()` set, per agent:
- `rel_trans = kappa_factor · max_{j∈mask} fitness_by_id[j]`, where `kappa_factor` = `trans_asymp` for ASYMPTOMATIC else 1 (folding today's `trans_asymp` in). Non-infectious agents already contribute 0 via the `infectious` mask.
- `susceptible = state ∈ {SUSCEPTIBLE, CLEARED, INFECTION, NON_INFECTIOUS}` (the last two are new — they enable superinfection), and
- `rel_sus =` 1 (SUSCEPTIBLE), `rr_reinfection` (CLEARED, unchanged), `rr_reinfection_inf` (INFECTION), `rr_reinfection_non` (NON_INFECTIOUS), 0 (ASY/SYM under **[D4]** Option 1).

The base engine then produces `(new_cases, sources)` using per-edge prob `rel_trans[src]·rel_sus[trg]·beta_edge`. Override **`set_prognoses(uids, sources)`** to:
1. For each event, build the source's per-strain probability row = `fitness_by_id · (source carries strain)`, normalized to sum 1 (spec's `r_j / Σ r`); draw the transmitted strain with `tbsim.choice2d` (or a manual CRN `searchsorted`). *Note the overall event probability already used max-fitness via `rel_trans`, and the split uses sum-normalized fitness — together these reproduce the spec's per-strain rates.*
2. **Block identical strains:** drop events where `(strain_mask[target] >> drawn) & 1` (target already carries it); increment the blocked-event analyzer counter (§6.12); leave those agents unchanged.
3. For the rest: `strain_mask[target] |= (1<<drawn)`; if the target was truly susceptible (SUSCEPTIBLE/CLEARED) set `state=INFECTION` and the usual infected flags; if already infected (INFECTION/NON_INFECTIOUS) **keep the state** (pure superinfection). Reset the progression clock `ti_infected=ti` for all acquisitions (and, per spec, for blocked same-strain exposures too — see §8.5).

**Validation against the spec's own numbers** (confirms the mapping is exact):
- *Table 2, agent C carrying strains fitness 0.5 and 0.4:* `rel_trans=max=0.5` → overall `0.5β` ✓; split `0.5/0.9=56%`, `0.4/0.9=44%` ✓; per-strain `0.28β`/`0.22β` ✓.
- *A/B example:* `rel_sus(A)=rr_reinfection_inf`, `rel_trans(B)=max(1, r_RIF·r_FQ)=1` → event prob `rr_inf·β` ✓; split → `(0,0,0)` blocked (counted), `(1,0,1)` acquired w.p. `r_RIF·r_FQ/(1+r_RIF·r_FQ)` → net superinfection `rr_inf·β·[r_RIF·r_FQ/(1+r_RIF·r_FQ)]` ✓.

**Pros.** Reuses the audited CRN transmission (correct common-random-number behavior, network/mixing-pool support, beta handling) — we write only the strain choice + bookkeeping. Small, testable surface.

**Cons / decisions.**
- **[D2]** *Reuse vs full `infect()` override.* Reuse works and is recommended. A full override is only needed for the next point.
- **[D3]** *Multi-source collapse.* The base keeps one source per target per step (first in edge order), so when a target is exposed by two infectors carrying *different* strains in the same step, the strain isn't fitness-weighted across sources — it's whichever edge came first. This is an ordering artifact on top of the pre-existing "one infection per target per step" behavior. For v1, **accept it** (document it; effect is second-order at weekly/monthly `dt`). A faithful version overrides `infect()` to collect *all* exposing edges per target and draw source+strain jointly ∝ fitness — more code, worth it only if tests show material bias.
- **[D-transmit-RNG]** Use `tbsim.choice2d` (already CRN-safe via `ppf`) for the strain draw, or add `ss.multi_random('source','target','strain')` for a dedicated stream. Recommend `choice2d` first (no new RNG plumbing).
- **Reference-ODE confirmation & the Q5 boundary.** This design reproduces the reference ODE's *bottleneck* force of infection (§10.1) in expectation — an AB source with `rel_trans=r_max` splitting transmissions `g_A:g_B` gives the ODE's `λ` contributions `g_A·r_max·P_AB` / `g_B·r_max·P_AB`. It does **not** natively implement the ODE's *independent* transmission mode (Q5, AB emits both strains at full rate), which would require per-strain transmission channels; since Q5 is answered in the ODE, accept the bottleneck-only ABM (**[D-txmode]**).

### 6.3 Superinfection protection (state-dependent susceptibility)

**Spec.** Table 3: INFECTION → `rr_reinfection_inf`, NON_INFECTIOUS → `rr_reinfection_non`, ASY/SYM → 0. Strain- and count-agnostic. Rationale for allowing it in NON_INFECTIOUS (a ~2-year state substituting for a second latent compartment) is in the spec.

**Mapping.** Exactly the state-dependent `susceptible`/`rel_sus` assignment in §6.2. Today `TB.step()` sets `susceptible` only for SUSCEPTIBLE/CLEARED and `rel_sus=1` except CLEARED; we extend both. Count-agnosticism is automatic (susceptibility keys off `state`, not `strain_mask` popcount). The reference ODE's explicit `s ∈ {A, B, AB}` stratification of the latent/non-infectious compartments (§10.1) — where the co-infected `AB` compartment is reachable only from latent/non-infectious states, never from asymptomatic/symptomatic — independently confirms this superinfection state space and gating.

**Decision [D4].** ASY/SYM handling: **Option 1** — parameters `rr_reinfection_asy=rr_reinfection_sym=0` (flexible; lets one-way sensitivity analysis probe the "immediate progression of the new strain" bias the spec warns about). **Option 2** — hard-code `susceptible=False` in ASY/SYM. *Recommend Option 1* (parametric): it is barely more code, keeps all four states uniform, and preserves the spec's wish to "assess any biases these assumptions induce." Defaults reproduce Option 2's behavior.

### 6.4 Progression & strain retention (`p_multi`)

**Spec.** Rates unchanged by strain count. INFECTION-super → NON_INFECTIOUS: keep all strains. INFECTION/NON_INFECTIOUS → ASYMPTOMATIC: keep all with prob `p_multi`, else one equiprobable strain progresses (default `p_multi=1`).

**Mapping.** The existing `transition()` already tells us who moved where (it mutates `state`; the code already computes `newly_cleared` masks by diffing). Add a hook after the INFECTION and NON_INFECTIOUS transitions: for agents that entered ASYMPTOMATIC *and* carry ≥2 strains, draw `ss.bernoulli(p_multi)`; for the "no" draws, pick one strain uniformly from the mask (decode → random present bit) and set `strain_mask` to just that bit. Progression to NON_INFECTIOUS: no change to the mask.

**Pros.** Localized; `p_multi=1` short-circuits to a no-op (zero overhead in the default). **Cons/decision [D6].** `p_multi<1` is the only place a superinfected agent loses strains at progression; the spec flags it for one-way sensitivity analysis, not calibration. Keep default 1; expose the parameter.

**Reference-ODE alignment & optional knobs.** The reference progression operator (`model-tests.md` §5) fires the bottleneck **only at `→ASYMPTOMATIC`**; `INFECTION^AB→NON_INFECTIOUS^AB` and `ASYMPTOMATIC^AB→SYMPTOMATIC^AB` retain both strains. Two variants the ODE toggles and the ABM should expose: (a) **selection rule** when one strain progresses — random `(½,½)` default vs fitness-weighted `(g_A,g_B)` (Q2b), a one-line choice in the "pick one present strain" step; (b) **`rr_prog_super` (ψ, Q1 — the top-priority question):** a progression-rate multiplier for multi-strain (≥2-strain) agents, default 1, applied trivially by scaling the existing `rr_activation` (and `asy_sym`) for those agents — this is the spec's "do superinfections progress faster?" test. An optional `rr_clear_super` (ω) multiplier on AB clearance/recovery (default 1) mirrors the ODE's sensitivity knob. De-novo resistance (§6.6) is a separate mechanism riding the same `INFECTION`-exit transition.

### 6.5 Natural clearance

**Spec.** Natural clearance/resolution (INFECTION→CLEARED, NON_INFECTIOUS→CLEARED) removes **all** strains (immune-mediated, strain-agnostic — the reviewer thread with Enriquez/Grantz/Ryckman confirms this is a biological stance, not just a simplification: no evidence resistance phenotypes differ in antigenic presentation). Rates unaffected by superinfection.

**Mapping.** Wherever the current model sets `state=CLEARED` from a *natural* pathway (`inf_cle`, `non_rec` in `TB.step()`), also `strain_mask=0`. This must be distinguished from **treatment/TPT** clearance, which is *selective* (§6.7–6.8) — so the "clear all strains" action belongs only in the natural-history transitions, never in the treatment code paths. Trivial, but the two clearance semantics are the crux of the resistance dynamics, so keep them in clearly separate methods and comment the rationale (per Enriquez's review request).

### 6.6 De-novo (random) acquisition

**Spec.** One-time per-strain, per-drug probability `p_rand_i` at INFECTION → NON_INFECTIOUS/ASYMPTOMATIC. Strain-agnostic except can't re-acquire an existing resistance. Multiple acquisitions allowed. Superinfection-vs-replacement left to software.

**Mapping.** Hook the same INFECTION-exit transition (§6.4), i.e. at `L→N` and `L→A`, mono-A(-susceptible) strains only. For each such agent, for each drug the strain is *not* yet resistant to, draw `ss.bernoulli(p_rand_acq[drug])`; on success compute the new strain id (`add_resistance`). Because rates are tiny, this is cheap even looping drugs (n small). **[D5] mixed vs replacement (revised to match the reference).** The reference ODE's null is **mixed** — the mutation *adds* the resistant strain, yielding a superinfection (`L^A→AB`); replacement (`L^A→B`) is the Q4 variant. So default to **mixed/superinfection** (bit-OR the new strain in), and expose replacement as a mode switch. *This is distinct from treatment-failure acquisition (§6.7), which is **always replacement**.* (The tech-spec docx said the ODE showed "no dynamic difference" and left it to software; the reference resolves the default to mixed, and reviewer Ryckman's comment — "probably results in infection w/ 2 strains" — agrees.) Defaults: `p_rand=0` for all drugs except a small BDQ value (Cohen's note: naturally-occurring pre-CFZ/BDQ mutations exist; RIF ≈ 0). Doing this once at the state transition (not per-timestep) is exactly the spec's efficiency argument and needs no per-step loop over all TB cases.

### 6.7 Treatment: strain-specific efficacy, adherence correlation, selective acquisition

**Spec.** Regimen `l` has per-strain efficacy `t_{il}` (full for susceptible strains, reduced for resistant). Adherence → agent-level draw from a regimen distribution, applied across all the agent's strains this course. Clearing a subset leaves the agent in the same TB state with the surviving strains (no memory of cleared strains, but track that treatment happened). On unsuccessful outcome, each surviving *susceptible* strain independently acquires resistance to each in-regimen drug with prob `q_{li}·RR_state`; **replacement**.

**Mapping.** Extend the product/delivery pattern. New **`TxRegimen`** product carrying: `drugs` (which modeled drugs it contains), a way to get `t_{il}` per strain **[D7]** — either *computed* from a base efficacy × per-drug resistance penalties (parsimonious; e.g. `t = base · ∏_{i∈regimen, strain resistant} penalty_i`) or an *explicit* per-strain table (full control, also lets short regimens differ). *Recommend computed-from-penalties as default with an explicit-override option.* `q_acq` dict (drug→acquisition prob) and `rr_acq_by_state` (multiplier, default 1 for ASY/SYM, 0 else — a single state-indexed multiplier, not per-strain/per-regimen, per the Grantz/Ryckman thread).

Rework `TxDelivery.step_start_treatment` / outcome resolution to be per-strain:
1. **Adherence [D8]:** draw one per-agent adherence value from the regimen's distribution at initiation; store it; apply it to every strain's clearance this course (this is the spec's mechanism for agent-level correlation across strains — addresses Grantz's "don't treat strain successes as independent" concern).
2. **Per-strain clearance:** for each carried strain, roll cure with probability `t_{il}·adherence_factor`. Clear the bits that succeed.
3. **Resolve:** if `strain_mask==0` → success → CLEARED (existing reinfection-protection bookkeeping). If some strains remain → **failure**: return to `prior_state` with the surviving mask (existing failure path, but not all-or-nothing).
4. **Acquisition on failure:** among *surviving, in-regimen-drug-susceptible* strains, for each in-regimen drug roll `ss.bernoulli(q·rr_acq_by_state[state])`; on success **replace** that strain with its resistant version. Once per treatment episode (at resolution), matching the spec.

**Pros.** Reuses the pre-roll/deferred-resolution timing already in `TxDelivery` (treatment resolves at `ti_treatment_end`). Adherence-as-agent-draw is a clean, well-motivated addition the team already wants beyond resistance. **Cons/decisions.** The current `Tx.administer` returns success/failure/relapse *sets*; strain-level outcomes need it to return per-agent surviving masks — a real refactor of `TxDelivery`'s success/failure handling (the biggest change to existing intervention code). Relapse (`p_relapse`, `dur_relapse`) must be reconciled with partial clearance: recommend relapse remains an agent-level property of a fully-cured agent (unchanged), acquisition applies only to failures. **[D7]/[D8]** as above; also **[D-q-duration]** whether `q` varies by regimen duration vs. folding duration into efficacy (spec scopes for both; expose `q` per regimen and allow duration-dependent efficacy).

**Reference-ODE alignment.** `model-tests.md` §6 gives the explicit 2-strain outcome operator that our n-drug design must generalize: independent per-strain cure (`e_a`,`e_b`), failures return to the **origin** disease state (the `T^{s,a}`/`T^{s,y}` stratification — the ABM's existing `prior_state` carries this), and among A-surviving failures resistance is acquired w.p. `q` as **always replacement** (an `AB` failure that mutates reduces to mono-`B`). SA calibration: `eff_a=0.75`, `eff_b=0.25`, `q_treat=0.05`. Note the ODE models treatment as an exponential sojourn (exit rate `δ`) with a rate-based initiation from the ASYMPTOMATIC/SYMPTOMATIC states, whereas TBsim uses a fixed `dur_treatment` with pre-rolled outcomes and a full care cascade (HSB→Dx→Tx) — the *outcome probabilities* are identical, so for ABM↔ODE matching, configure the ABM's care cascade to approximate the ODE's state-specific initiation rates. The treatment-failure `q` (always replacement) is a fixed assumption, **not** a test variable — deliberately distinct from the de-novo `q_p` (§6.6, default mixed, the Q4 test).

### 6.8 TPT

**Spec.** Same structure as treatment: strain-specific efficacy and acquisition-on-failure (highest risk ASY/SYM, low INFECTION, low-medium NON_INFECTIOUS). Because superinfection exists, TPT clearing a susceptible strain can advantage a resistant one (Cohen 2006, Mills 2013, Kunkel 2015/2016). With `p_multi=1`, only the transmission advantage (not a progression advantage) is present.

**Mapping.** Extend `TPTTx` (already has sterilize/suppress mechanisms) to be strain-aware: **sterilization becomes per-strain** (clear susceptible strains, leave resistant ones; if resistant strains survive, the agent is *not* fully cleared — this is precisely the mechanism producing the Cohen/Mills/Kunkel effect). Suppression (the `rr_*` modifiers) stays agent-level (natural history is agent-level). Add TPT-failure acquisition with a state-dependent risk mirroring §6.7. The existing TPT delivery classes (`TPTSimple`, `TPTHousehold`, `HouseholdContactTracing`+`TPTDelivery`) need no structural change — only the product's clearance becomes selective.

**Pros.** Directly reproduces the target published dynamics; reuses TPT delivery. **Cons.** `TPTTx.update_roster`'s current sterilization only fires in INFECTION state; per-strain sterilization must generalize to whichever states TPT targets. **[D-tpt-acq]** confirm the state-specific TPT acquisition risks (defaults) with the modelers.

### 6.9 DST & observed resistance profiles

**Spec.** A DST produces `X_obs,k` (n bits) via per-drug sensitivity/specificity applied **at the strain level**, then aggregated (`x_obs,i=1` if any observed strain is resistant to i), with a per-strain observation probability `p_strain_obs` (default = strain fitness, a bacillary-load proxy) modeling sample/culture/sequencing bottlenecks. `p_strain_obs=1` ⇒ a phenotype in multiple strains is *more* likely detected; `<1` ⇒ overall sensitivity drops from strain dropout. DST doesn't identify strains (only the n-bit aggregate). The reviewer thread (Grantz/Ryckman) settled on this "Option 2 with a bottleneck" over a simpler per-agent test.

**Mapping.** New **`DST`** product (its own class, not the TB-state `Dx` — different logic). For selected agents: for each carried strain, first roll `ss.bernoulli(p_strain_obs(strain))` (observed at all?); for observed strains, for each drug roll sensitivity (if truly resistant) or 1−specificity (if susceptible) → observed strain phenotype; then OR observed strain phenotypes across observed strains → `dst_profile[k]` (n bits), with optional "indeterminate" outcome. New **`DSTDelivery`** intervention storing `dst_profile` (n `BoolArr`s, or an n-bit `IntArr`), `dst_tested`, and `ti_dst`. Eligibility: immediate post-diagnosis *or* on treatment failure **[D-dst-elig]** (both are just eligibility callables, like the existing `DxDelivery`).

**Pros.** Mirrors the existing `DxDelivery` lifecycle (eligibility → administer → set states). The bottleneck is a couple of Bernoulli draws. **Cons.** DST is genuinely different from the DataFrame-driven `ProductMulti` (per-drug sens/spec, not per-state result probabilities), so it's a new product class rather than a `Dx` subclass. `p_strain_obs=fitness` couples DST sensitivity to the fitness parameters — document this so calibration doesn't double-count.

### 6.10 Treatment provision conditioned on the observed profile

**Spec.** Treatments provided only to agents matching an observed profile (same spirit as the current eligibility-based delivery).

**Mapping.** Pure eligibility composition — no new machinery. A `TxDelivery` for a second-line regimen takes an `eligibility` callable that reads `DSTDelivery.dst_profile` (e.g., "RIF-resistant observed → eligible for the RIF-sparing regimen"). This is exactly how `TxDelivery.eligibility` already works. **Decision:** none structural; provide a couple of ready-made eligibility helpers (e.g., `resistant_observed(drug)`), and confirm the empiric-vs-DST-guided treatment sequencing with the modelers.

### 6.11 Treatment monitoring & regimen switching — the least-specified, largest new piece

**Spec.** Eligibility depends on time under treatment (track time since initiation). Use the diagnostic class to detect still-bacteriologically-positive agents (state-dependent). For those, **extend or switch** the regimen — handled as a *new* treatment product conditional on the monitoring result — which "may need a way to prematurely stop/change an ongoing treatment regimen." Also: track time since last treatment initiation to decide whether a later presentation is *failure* (→ DST/second-line) or a *new case*. The reviewer thread (Grantz/Ryckman) explicitly defers the mechanism to Cliff.

**Mapping (proposal).** This needs one genuinely new capability: **preempting an in-progress treatment course.** Today `TxDelivery` pre-rolls an outcome and resolves it at `ti_treatment_end`, with no interruption. Proposed design:
- Add a **`TreatmentMonitoring`** intervention that, for agents on treatment past a configurable time-on-treatment, runs a (state-dependent) `Dx` for bacteriological positivity. It already has everything it needs (`ti_treatment_start` exists on `TxDelivery`).
- Give `TxDelivery` an explicit **`interrupt(uids)`** method that cancels pending success/failure/relapse for those agents and returns them to a state where another delivery can pick them up (clear `pending_*`, reset `ti_treatment_end`). Monitoring calls `interrupt` on still-positive agents, then a second `TxDelivery` (the extended/switched regimen) takes them via its eligibility.
- Track `ti_last_treatment_start` (agent-level) so a re-presentation within a window is routed as failure (→ DST/second-line) vs. a new episode. This is a new `FloatArr` and an eligibility helper.
- Efficacy of the extended/switched regimen is set *externally* (conditional-on-prior-treatment values), as the spec notes.

**Pros.** Builds on existing timing/eligibility; `interrupt()` is a small, reusable addition. **Cons/decisions [D9].** This is the most open area. Sub-decisions: (i) monthly `dt` limits regimen-change granularity — the reviewers suggest a **finer `dt`** for monitoring-focused analyses; flag that resistance/monitoring studies may run weekly. (ii) Whether to model "finish current regimen, then switch" (simpler; Grantz's leaning) vs. true mid-course preemption (`interrupt()`, more faithful). *Recommend implementing `interrupt()` but defaulting deliveries to non-preemptive*, so both modes are available. (iii) Confirm the bacteriological-positivity `Dx` definitions per state with the modelers.

### 6.12 Analyzers & results

**Spec (explicit asks).** (1) An analyzer counting, per agent, how often an infection *would* have occurred but was blocked because the drawn strain was identical to one already carried (the low-prevalence-strain bias probe). (2) Burden metrics to compare pre/post-resistance and to show resistance-parameter effects: TB prevalence/100k, annual asymptomatic-incidence/100k, TB mortality/100k, and **% of active TB that is resistant** (overall and per drug).

**Mapping.** 
- The blocked-superinfection counter is incremented directly in `set_prognoses` (§6.2, step 2) into a new result series and/or a per-agent `IntArr` (`n_blocked_superinf`).
- A new **`ResistanceStats`** analyzer (following `HouseholdStats`) records per step: `n_strains` distribution, `prevalence_resistant[drug]` = fraction of active-TB agents whose aggregate phenotype (`phenotype_any(strain_mask)`) is resistant to each drug, `prevalence_mdr`, `n_superinfected`, and the blocked-event rate. The existing per-state `TB` results are untouched, so the pre/post-resistance burden comparison is just "run with `drugs=None` vs `drugs=[…]`."
- Mirror the reference ODE's observables (`model-tests.md` §11) so ABM and ODE report like-for-like: `frac_resist` (active-B / active), `frac_super` (active-AB / active), and — the key mechanistic output — the **resistance-origin flux decomposition**: (i) *de-novo* (mutation at `L^A` progression), (ii) *treatment-acquired* (the `q` replacements at failure), (iii) *transmitted* (new B infections from B-carrying sources). Attributing each new resistant case to one of these three is what distinguishes the model variants; track them as counters incremented where each event fires.
- Optionally attach the transmitted strain as edge data on Starsim's `InfectionLog` (`diseases.py:513`) when an infection-log analyzer is present — a clean provenance trail for who transmitted which strain (useful for validating transmission and for resistance-emergence attribution).

**Pros.** Additive; no change to existing results. **Cons.** `ResistanceStats` decodes `strain_mask` each step — cheap for small `m`, but if `n` grows, compute the aggregate phenotype incrementally.

---

## 7. Code map: new/modified files & class hierarchy

**New**
- `tbsim/resistance/strains.py` — `Strains` registry (enumeration, `fitness_by_id`, `profile_matrix`, bit helpers).
- `tbsim/resistance/tb_resistant.py` — `TBResistant(TB)`: `strain_mask` state, `step_bookkeeping` override (state-dependent `susceptible`/`rel_sus`/`rel_trans`), `set_prognoses` override (strain choice, blocking, superinfection), transition hooks (progression retention, de-novo acquisition, clear-all-on-natural-clearance).
- `tbsim/resistance/dst.py` — `DST` product + `DSTDelivery`.
- `tbsim/resistance/treatments.py` — `TxRegimen` (strain-aware efficacy/acquisition); helpers; `TreatmentMonitoring`; `TxDelivery.interrupt()`.
- `tbsim/resistance/analyzers.py` — `ResistanceStats`.
- `tests/test_resistance.py` — unit + dynamics tests (§10).

**Modified**
- `tbsim/tb.py` — refactor `step()` into `step_transmission/step_transitions/step_bookkeeping` hooks; add `rr_reinfection_inf/non/asy/sym`, `p_multi`, `p_rand_acq`, `rel_fitness` pars (inert unless `drugs` set); expose `infected ⇔ strain_mask!=0` when strain-aware.
- `tbsim/interventions/treatments.py` — make outcome resolution per-strain (surviving-mask instead of all-or-nothing); agent-level adherence draw; acquisition-on-failure hook; `interrupt()`.
- `tbsim/interventions/tpt.py` — per-strain sterilization; TPT-failure acquisition.
- `tbsim/interventions/products.py` — factor out shared per-agent-probability plumbing if `DST` can reuse it (else leave `ProductMulti` as-is).
- `tbsim/__init__.py` — export the resistance subpackage.
- `tbsim/sim.py` — optional `get_dst()` helper mirroring `get_dx()`; allow the flat-par router to pass resistance pars.

**Starsim override points (for reviewers), from the 3.5.1 source:** `Infection.set_prognoses` (`diseases.py:87`), `Infection.infect` (`diseases.py:257`, only if we implement multi-source fitness-weighting **[D3]**), `Infection.compute_transmission`/`Network.net_beta` (`diseases.py:250`/`networks.py:424`, not needed under the reuse design), `InfectionLog.append` (`diseases.py:513`, optional strain provenance). Reused as-is: `multi_random` transmission RNG (`diseases.py:135`), the numba per-edge kernel (`diseases.py:229`), and `tbsim.choice2d` (`tb.py:509`).

---

## 8. Cross-cutting concerns

### 8.1 Reproducibility / CRN
All strain draws must be CRN-safe: use `tbsim.choice2d` (UID-keyed `ppf`) for strain selection and `ss.bernoulli`/`ss.random` (UID-keyed) for acquisition/adherence — never raw `np.random`. Give each new stochastic step its own named distribution (as `TB` already does with `_rng_inf/_non/_asy/_sym`) so adding resistance doesn't perturb the RNG streams of a single-strain run. **Regression guarantee:** with `drugs=None` (or `strain_mask` all pan-susceptible and all fitness=1, `rr_*` at defaults), results must match the current `TB` bit-for-bit; make this an explicit test (§10).

### 8.2 Performance
`strain_mask` is one integer per agent; decoding to `(n_agents×m)` bool is `O(n_agents·m)` with `m≤16`, negligible next to network transmission. De-novo/treatment-acquisition loops are over `n` drugs (tiny) and only over agents making a specific transition. No per-timestep loop over all TB cases (the spec's efficiency requirement for acquisition is honored by hooking transitions, not stepping). Watch: avoid rebuilding the `(n_agents×m)` matrix multiple times per step — compute once in `step_bookkeeping` and reuse.

### 8.3 Beta rescaling & calibration
Fitness costs `r_i≤1` reduce transmission of resistant strains but, because superinfection uses `max` fitness and the pan-susceptible strain has fitness 1, **overall transmissibility of a pan-susceptible-dominated epidemic is essentially unchanged** — so the spec's expectation that burden shouldn't shift much is structurally supported. Still, seed the resistance-enabled runs from the **`tb_LAI_TPT` calibrated parameter set** (India; RandomNet+HouseholdNet; burn-in from 1950; treatment on from 1995; calibrated `beta, rel_beta_hh, trans_asymp, rr_reinfection_rec, inf_*, non_*, asy_*, sym_asy, p_adherence, p_relapse, sym_dead`) as the spec's testing section directs, and check whether adding strains requires any `beta` re-fit (Ryckman's comment flags a possible rescale if transmission is ever modeled strain-to-agent independently — which we are *not* doing in the recommended design). For the resistance layer specifically, the reference ODE's closed-form `R_0^{(i)}=β·r_i·(κ·E[t in A]+E[t in Y])` (§10.1) is the analytic anchor for setting `beta` and predicting which strain dominates; match its South-Africa 2-strain calibration for ABM↔ODE validation.

### 8.4 Backward compatibility
Recommended packaging (subclass + refactored hooks, §6.1) leaves `TB`, all existing interventions, `tbsim.Sim`, and every current test unchanged. Resistance is opt-in via `TBResistant`/`TxRegimen`/`DST`.

### 8.5 Time-varying progression clock
The spec asks that each *successful exposure* reset "time since infection" (including same-strain exposures that don't change the profile), to support time-varying progression if/when added. The current model uses time-invariant competing exponential rates, so this is latent — but cheap to honor now: set `ti_infected=ti` on every acquisition *and* on blocked same-strain events. Build it in; it's a no-op until time-varying progression exists. **[D-clock]** confirm whether to reset on blocked same-strain exposures (spec's "ideally yes").

---

## 9. Implementation phases

1. **Foundations & refactor (low risk).** Refactor `TB.step()` into hooks; add the `Strains` registry and `strain_mask`; add inert resistance pars. Regression test: identical to current `TB`.
2. **Transmission & superinfection (highest risk/value).** State-dependent `susceptible`/`rel_sus`/`rel_trans`; `set_prognoses` strain choice + blocking + superinfection; blocked-event counter. Validate against spec Table 2 and the A/B example numerically; validate two-strain competition against the team's reference ODE — reduction + symmetry + endemic invasion (§10.1).
3. **Natural history overlay.** Progression retention (`p_multi`), clear-all-on-natural-clearance, de-novo acquisition. Sensitivity tests on `p_multi`, `p_rand`.
4. **Treatment & TPT.** Per-strain efficacy, agent-level adherence, acquisition-on-failure (replacement), per-strain TPT sterilization. Reproduce the qualitative TPT→resistance dynamics of the cited papers.
5. **Diagnostics & monitoring.** `DST`/`DSTDelivery`, profile-conditioned treatment eligibility, `TreatmentMonitoring` + `TxDelivery.interrupt()`. (Largest new machinery; may want finer `dt`.)
6. **Analyzers, docs, calibration hooks.** `ResistanceStats`, optional `InfectionLog` strain provenance, docs/examples, seed from `tb_LAI_TPT`.

Phases 1–3 are the backbone; 4–6 layer on. Phases 2 and 5 carry the most risk.

---

## 10. Testing & validation plan

Driven by the spec's explicit asks, and leveraging TBsim's existing assets (full pytest suite; `tbsim/compartmental/lshtm_ode.py` with name-aligned `TB_ODE`/`TB_SS`) — and, crucially, the team's **two-strain reference ODE** (`ode.r`/`model-tests.md`, §10.1), which is the authoritative validation target.

- **Regression / equivalence.** `drugs=None` or single pan-susceptible strain (all fitness 1, default `rr_*`) reproduces current `TB` outputs exactly (same RNG seed) — burden metrics and per-state counts identical.
- **Transmission unit tests.** Assert the `set_prognoses` strain-choice probabilities and the `rel_trans=max·kappa` / split-∝-fitness / identical-blocking logic reproduce spec Table 2 (56/44%, 0.28β/0.22β) and the A/B superinfection probability, on tiny hand-built populations.
- **ODE cross-check (against the team's reference ODE).** Validate the strain-summed ABM against `ode.r` at matched parameters (its SA 2-strain calibration, §10.1): the three `model-tests.md` §8 checks — single-strain reduction, strain symmetry, and population conservation — plus endemic prevalence, resistant fraction (`frac_resist`), superinfected fraction (`frac_super`), and the resistance-origin flux split. Anchor `β` and the expected strain dominance with the closed-form `R_0` (§9). `ode.r`'s `two_strain_collapse()` gives the strain-summed view for compartment-for-compartment comparison. The Q1–Q5 *structural* comparisons live in the ODE; the ABM reproduces the chosen defaults (and, where feasible, the same switches). See §10.1.
- **Burden invariance.** Show overall TB prevalence/100k, annual asymptomatic incidence/100k, and mortality/100k are ~unchanged when resistance is added with realistic fitness costs (spec's expectation), and *explain* any shift.
- **Monotonicity / directional tests (spec's second ask).** Raising acquisition risk `q`, treatment rate, or relative efficacy against resistant strains ↑ the % of active TB that is resistant; raising fitness costs `r_i` (toward more cost) ↓ resistant prevalence — assert the sign and rough magnitude.
- **TPT dynamics.** Reproduce the qualitative Cohen/Mills/Kunkel result: community TPT that clears susceptible strains increases the resistant fraction (present via transmission even at `p_multi=1`).
- **DST behavior.** `p_strain_obs=1` → higher detection of a phenotype carried by multiple strains; `p_strain_obs<1` → reduced aggregate sensitivity; sens/spec recovered in the mono-infection limit.
- **Reproducibility.** Same seed → identical resistance trajectories; adding resistance modules doesn't perturb single-strain streams.
- **Starting parameters.** Two reference sets: the reference ODE's **South Africa** 2-strain calibration (§10.1) for ABM↔ODE matching, and the `tb_LAI_TPT` **India** best-fitting set (§8.3) for full-model burden tests.

### 10.1 Reference two-strain ODE (`ode.r` + `model-tests.md`) — the validation target

The team has a deterministic **two-strain reference ODE** (`ode.r`, built on `tbsim/compartmental/lshtm_ode.R`; its master system and "questions of interest" are specified in a companion `model-tests.md`). This is the authority the ABM's resistance behavior should be validated against, and its structure both confirms and sharpens the design above. Its state→symbol mapping and full 22-compartment system give:

- **Two strains: A = treatment-susceptible, B = treatment-resistant.** This is the minimal resistance system and maps onto the ABM's strain framework as `n=1` "drug" (the regimen), `m=2` strains: strain A ↔ strain id 0 (pan-susceptible, fitness 1), strain B ↔ strain id 1 (resistant). The general `n`-drug ABM (§4) reduces to exactly this when `drugs=['TX']`, so the reference is a clean special case to validate against before enabling more drugs.
- **Superinfection is explicit: latent/disease compartments are stratified `s ∈ {A, B, AB}`** (`L_*`, `N_*`, `AS_*`, `SY_*`). The `AB` compartment is the co-infected (both-strain) agent — the direct ODE analogue of an ABM `strain_mask` with both bits set. This independently validates the bitmask membership model (§4.2) and the state-dependent superinfection gating (§6.3).
- **Treatment is stratified by the state it was initiated from:** `TA_* = T^{s,a}` (initiated from ASYMPTOMATIC) vs `TY_* = T^{s,y}` (initiated from SYMPTOMATIC). This is the ODE's mechanism for the spec's state-dependent acquisition risk (`rr_acq_by_state`, §6.7) and state-dependent return-on-failure. The ABM already carries `prior_state` through `TxDelivery`, so it can reproduce this directly — the reference confirms acquisition risk and the failure-return state must key on the **initiating** state, not the current one.
- **CLEARED is split into C/R/W** (`CLE`/`REC`/`TRD`, i.e. cleared-from-latent / recovered-from-non-infectious / post-treatment), matching the existing `lshtm_ode` split for the three reinfection-pathway RRs (`rr_reinfection_cleared`/`rec`/`treat`) — consistent with the ABM. `DTH` is a cumulative-deaths accumulator outside the conserved population.

**The master system (full spec now in hand).** `model-tests.md` §1–7 specifies, and `ode.r` implements, a **22-compartment** system: 4 strain-agnostic pools (`S`,`C`,`R`,`W`), the 4 natural-history states × {A,B,AB} (12), and treatment × {A,B,AB} × {origin a,y} (6), plus a deaths accumulator. What the ABM must match:

- **Force of infection, two modes.** *Bottleneck (default):* `λ_A = (β/N)[r_a·P_A + g_A·r_max·P_AB]`, `λ_B = (β/N)[r_b·P_B + g_B·r_max·P_AB]`, with `g_A=r_a/(r_a+r_b)`, `r_max=max(r_a,r_b)`, `P_s=κ·A^s+Y^s`. This is exactly the ABM reuse design (§6.2): an AB source has `rel_trans=r_max` and its transmissions split `g_A:g_B` — reproduced *in expectation* by the per-event multinomial. *Independent (Q5):* each strain transmits at its own full rate from AB sources (`λ_A=(β/N)·r_a·(P_A+P_AB)`), yielding strictly more onward transmission — see the Q5 caveat below.
- **Superinfection hazards** (§4): mono→AB governed by state σ's; **`σ_L=σ_N=1`, `σ_A=σ_Y=0` by default**. Primary infection of `S/C/R/W` uses the `{1,ρ_C,ρ_R,ρ_W}` multipliers — the ρ (cleared/recovered/treated) reinfection factors are **distinct** from the σ (superinfection) factors.
- **Progression bottleneck** (§5): fires only at `→A`; `L^AB→N^AB` and `A^AB→Y^AB` retain both strains. Selection when one strain progresses: random `(½,½)` default or fitness-weighted `(g_A,g_B)` (Q2b). A superinfection progression multiplier `ψ=rr_prog_super` (default 1) scales AB progression (Q1); optional `ω=rr_clear_super` scales AB clearance (default 1).
- **De novo resistance** (§5.1): mono-A only, at `L^A→N` and `L^A→A`, one-time prob `q_p` — **mixed→AB (default)** vs replacement→B (Q4).
- **Treatment operator** (§6): independent per-strain cure (`e_a`,`e_b`); failures return to the **origin** disease state (asymptomatic/symptomatic); among A-surviving failures, resistance is acquired w.p. `q`, **always replacement** (so an AB that fails and mutates *reduces* to mono-B). Treatment initiation is a state-specific rate (`r_treat_asym`,`r_treat_sym`); the ODE has no diagnosis/DST/monitoring/TPT cascade — those are ABM-only (§6.8–6.11), validated by unit/behavioral tests rather than the ODE.

**Calibration & analytic anchor.** The reference is calibrated to **South Africa**: `β=16.45` (~400/100k/yr incidence), `fit_a=1.0`, `fit_b=0.575` (~43% fitness cost → resistant fraction ~4%), `r_treat_sym=1.204`, `eff_a=0.75`, `eff_b=0.25`, `q_treat=0.05`, `q_prog=1e-4`. Section 9 gives a closed-form next-generation `R_0^{(i)} = β·r_i·(κ·E[time in A] + E[time in Y])` per strain, which both sets `β` to a target prevalence and predicts which strain wins: fitness cost scales `R_0` linearly via `r_i`, while treatment shortens strain A's infectious sojourn and lowers its *effective* `R_0`; the sign of `R_0^{A,eff}−R_0^{B,eff}` predicts invasion. Use this SA 2-strain set for ABM↔ODE matching; use `tb_LAI_TPT` (India) for full-model burden tests.

**Questions of interest → who answers them.** Each is a switch on the one system: Q1 `ψ` (faster progression when superinfected — *top priority*), Q2a `p_multi`, Q2b progression-selection rule, Q3a/b superinfection eligibility (`σ`), Q4 de-novo mixed-vs-replacement, Q5 transmission bottleneck-vs-independent. **Q6 (identical-strain/repeat superinfection) is deprioritized and not implemented in the ODE** — but the ABM's blocked-superinfection counter (§6.12, the tech-spec's explicit ask) is precisely the per-agent capability the ODE lacks and the natural home for Q6 if ever revisited; the ABM default (block identical-strain draws, no effect) matches the ODE's implicit behavior. The ODE is the primary vehicle for the Q1–Q5 *structural* comparisons; the ABM implements the chosen defaults and exposes the same parameters for its own sensitivity analyses.

**Two default reconciliations for the team (parameters, not architecture).** (1) **`σ_L`/`σ_N` default** — the ODE null uses `1` (full susceptibility to a 2nd strain in latent/non-infectious), whereas the tech-spec docx default was `rr_reinfection_inf=rr_reinfection_non=rr_reinfection_rec` (0.21). Same parameter, different baseline; **[D-sigma]** pick one (the ODE is the newer artifact). (2) **De-novo mechanism default** — the reference defaults to **mixed**; §6.6 now matches this (previously recommended replacement) — see the flipped **[D5]**.

**Reduction/symmetry checks (§8) → ABM equivalence tests.** (i) *Single-strain reduction:* seed only A, `q=q_p=0`, `σ=0`, `p_multi=1` → the strain-summed ABM must reproduce single-strain `TB` (and `TB_ODE` with treatment off). (ii) *Strain symmetry:* `r_a=r_b`, `e_a=e_b`, `q=q_p=0`, symmetric seeding → A- and B-compartments identical for all `t`. (iii) *Population conservation.* `ode.r` provides `two_strain_collapse()` (strain-summed view for compartment-for-compartment ABM↔ODE comparison) and `two_strain_residence()`/`_reach()` (per-episode time-in-state anchors from the absorbing-chain fundamental matrix).

**One architectural caveat (Q5).** The recommended single-`strain_mask` + reuse-FOI design (§4.2, §6.2) implements the **bottleneck** natively (an AB source passes one strain per event ∝ fitness). It does *not* natively express the **independent** mode (Q5, AB transmits both strains at full rate) — that needs per-strain transmission channels (one-`Infection`-module-per-strain, or an `infect()` override emitting per-strain events). Since Q5 is the lowest-priority structural question and is answered *in the ODE*, this is an acceptable ABM limitation; flag it only if Q5 must be reproduced in the ABM.

---

## 11. Consolidated open decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Strain membership storage | bitmask `IntArr` / `m` `BoolState`s / per-strain modules / fixed slots | **Bitmask `IntArr`** (n≤~6); revisit for larger drug sets |
| D2 | Superinfection transmission | reuse FOI (state-dependent `rel_sus`) / full `infect()` override | **Reuse FOI** + strain logic in `set_prognoses` |
| D3 | Multi-source collapse (same-step, different strains) | accept base first-edge-order / override `infect()` for fitness-weighted source+strain | **Accept for v1**, document; override only if tests show bias |
| D4 | ASY/SYM superinfection | parametric `rr_reinfection_asy/sym=0` / hard-code block | **Parametric (Option 1)**; defaults reproduce the block |
| D5 | De-novo (progression) acquisition mode | mixed→AB / replacement→B | **Mixed** (matches reference ODE null; = the Q4 switch). *Treatment-failure* acquisition is separate and **always replacement** (§6.7) |
| D6 | `p_multi` | default value | **1.0**; expose for one-way sensitivity |
| D7 | Treatment efficacy spec | computed from per-drug penalties / explicit per-strain table | **Computed default**, explicit override allowed |
| D8 | Adherence | scalar / agent-level draw from regimen distribution applied to all strains | **Agent-level regimen draw** (induces correlation) |
| D9 | Treatment monitoring / regimen switching | finish-then-switch / mid-course `interrupt()`; `dt` granularity | Implement **`interrupt()`**, default non-preemptive; consider **finer `dt`** for monitoring studies |
| D-pkg | Packaging | `TBResistant(TB)` subclass / fold into `TB` | **Subclass + refactor `TB.step()` into hooks** |
| D-notation | Drug ordering | positional arrays / name→index map | **Name→index map** (`drugs=['RIF','BDQ']`) |
| D-clock | Progression clock reset | reset on acquisition only / also on blocked same-strain exposure | **Also on blocked** (spec's "ideally"); no-op until time-varying progression exists |
| D-dst-elig | DST eligibility | immediate post-diagnosis / on treatment failure / both | **Both**, via eligibility callables |
| D-q-dur | `q` vs regimen duration | vary `q` by duration / fold into efficacy | Expose both; default `q` per regimen |
| D-sigma | Superinfection-susceptibility default (`σ_L`,`σ_N`) | reference-ODE null `=1` / tech-spec `=rr_reinfection_rec` (0.21) | **Reconcile with team** (ODE null is the newer artifact); parameter only, not architecture |
| D-txmode | Transmission from AB sources | bottleneck (one strain/event ∝ fitness) / independent (both at full rate, Q5) | **Bottleneck** — native to the bitmask design; independent needs per-strain channels (defer; the ODE answers Q5) |
| D-progsuper | Faster progression when superinfected (`ψ=rr_prog_super`, Q1 — top priority) | default 1 / >1 for multi-strain agents | **Default 1**, expose (scale `rr_activation`/`asy_sym` for ≥2-strain agents) |

Questions the modelers should confirm regardless of software: default `rel_fitness` per drug; the `σ_L`/`σ_N` superinfection-susceptibility default (**[D-sigma]**: reference ODE uses 1, tech-spec used `rr_reinfection_rec`); whether to enable faster progression for superinfected agents (`ψ`, the top-priority Q1); default `p_rand_acq` (esp. BDQ vs RIF≈0); default `q` and `rr_acq_by_state` per regimen; state-specific TPT-failure acquisition risks; the bacteriological-positivity `Dx` definitions per state for monitoring; whether monitoring analyses warrant a weekly `dt`.

---

## 12. Risk & effort

- **Highest technical risk:** transmission/superinfection (§6.2, D2/D3) — mitigated by reusing the audited CRN engine and validating against the spec's own worked numbers and the team's two-strain reference ODE (reduction, symmetry, and endemic-invasion checks, §10.1).
- **Largest new machinery:** treatment monitoring / regimen switching (§6.11) — the least-specified part of the spec; the `interrupt()` capability is new to `TxDelivery`, and monitoring may push toward a finer `dt`.
- **Biggest refactor of existing code:** `TxDelivery` outcome resolution moving from all-or-nothing to per-strain surviving masks (§6.7).
- **Lowest risk:** natural-history overlay (§6.4–6.6), analyzers (§6.12), and profile-conditioned eligibility (§6.10), which are additive.
- **Rough effort:** ~4–6 focused weeks to a tested first implementation across all subsystems, with phases 1–3 (~2 weeks) delivering a usable multi-strain transmission model and 4–6 completing treatment/diagnostics/monitoring. The recommended architecture keeps single-strain TBsim fully backward-compatible throughout.

---

### Sources cited in the spec (for context on the target dynamics)
Cohen, Lipsitch, Walensky, Murray. *PNAS* 2006;103(18):7042. — Mills, Cohen, Colijn. *Sci Transl Med* 2013;5(180):180ra49. — Kunkel, Crawford, Shepherd, Cohen. *AIDS* 2016;30(17):2715. — Kunkel, Colijn, Lipsitch, Cohen. *Phil Trans R Soc B* 2015;370(1670):20140306.
