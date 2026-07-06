# Resistance implementation comparison: branch `resistance` vs. PR #430 (`mme-resistance`)

This document compares the two independent implementations of the TBsim drug-resistance / multi-strain extension against each other and against the technical specification (`tbsim-resistance-tech-spec.md`, Ryckman/Grantz/Cohen). It focuses on the *differences* and ends with a concrete recommendation about what to keep from each. Recommendations are anchored to two things only: fidelity to the original spec, and Starsim style/idiom.

- **"Current branch"** = branch `resistance` (this working tree): `tbsim/resistance/` — `strains.py`, `tb_resistant.py`, `treatments.py`, `dst.py`, `analyzers.py`. ~930 lines, plus a +90/−46 refactor of `tbsim/tb.py`.
- **"PR #430"** = branch `mme-resistance` (GitHub PR #430, commit `66f10ce`): `tbsim/resistance/` — `strains.py`, `connector.py`, `resolvers.py`, `regimens.py`, `tx.py`, `tpt.py`, `diagnostics.py`, `analyzers.py`. ~2060 lines, plus a +291/−15 change to `tbsim/tb.py`, small edits to `__init__.py`/`migration.py`/`interventions/tpt.py`, ~2480 lines of example scripts, and 346 lines of tests.

## Bottom line (recommendation in one paragraph)

Keep the **current branch's foundation** — the bitmask strain representation, the non-invasive `TBResistant(TB)` subclass, transmission/superinfection folded into the disease's own `step_bookkeeping`/`set_prognoses`, module-owned CRN distributions, and the ODE validation — because these are the most spec-faithful on the core model and the most idiomatic for Starsim. Then **port PR #430's feature-completeness onto that foundation**: strain-aware TPT, the DST→regimen router, treatment-monitoring eligibility, per-drug and state-varying acquisition (`q_{l,i}` with the state RR term), the `Regimen` abstraction, cascade-integrated treatment as an option alongside the rate-based path, the spec's default reinfection coupling, and input validation. PR #430 implements more of the spec; the current branch implements the core more correctly and more cleanly. The right end state is the current branch's architecture with PR #430's missing features grafted on.

## Implementation status (2026-07-06)

This merge plan has since been **implemented on branch `resistance`**, on the bitmask foundation. Ported/added: per-drug multi-strain de-novo acquisition (`p_rand`); per-drug treatment acquisition `q_{l,i}` with a per-TB-state RR (`acq_state_rr`) and regimen-scoped efficacy; the spec's default reinfection coupling (`rr_reinfection_inf` ← `rr_reinfection_rec`, `non` ← `inf`); strain-aware TPT (`TPTRx` — per-strain sterilization/unmasking plus per-state TPT-failure acquisition, via a small `_apply_neither_branch` hook on the base `TPTTx`); a DST→regimen router (`DSTDelivery.matches`); treatment monitoring with a *working* mid-course switch (`treatment_monitoring_eligibility` + `TxDeliveryR.interrupt`/`supersedes`); auto strain labels + registry input validation; and a per-strain `StrainResults` analyzer. `TBResistant` now defaults its module name to `'tb'` so the standard tbsim interventions (TPT, HSB, Dx) integrate cleanly. All 18 resistance tests (13 updated + 5 new) and the full 143-test suite pass, and the ODE self-checks still hold.

**Deliberate cleanliness-over-completeness deviations from the plan:** (1) the `Regimen` class and the cascade `StrainAwareTx`/`StrainAwareTxDelivery` were *not* ported — the ODE-validated rate-based `TxDeliveryR` plus `regimen_drugs` and eligibility/`supersedes` already deliver DST-routing and regimen-switching without PR #430's fragile per-UID cascade bookkeeping; (2) no separate `DuplicateStrainAnalyzer` — the identical-strain-block count is already a first-class `TBResistant` result (`new_blocked_superinf`); (3) the `tb_LAI_TPT` burden-per-100k before/after table remains a follow-up validation study rather than code. Adherence is still a single per-agent Bernoulli, not a full per-agent distribution.

## Side-by-side overview

| Dimension | Current branch (`resistance`) | PR #430 (`mme-resistance`) |
|----|----|----|
| Strain representation | Single-integer per-agent **bitmask** `ss.IntArr('strain_mask')`; registry auto-enumerates all `m=2ⁿ` strains from a drug list | **One `ss.BoolArr` per declared strain** (`carries_<uid>`); user declares an explicit list of `StrainSpec` |
| Registry | `Strains(drugs, rel_fitness)` — integer ids, `profile[j,i]`, name-keyed fitness | `StrainSpec` + `StrainRegistry` + `StrainProfile` — named uids, per-strain fitness/init_prev, validation |
| Disease class | `TBResistant(TB)` in `resistance/` subpackage; `tb.py` refactor is minimal and behavior-preserving; **no** core→overlay dependency | `MultiStrainTB(TB)` defined **inside `tb.py`**; `TB.__init__` rejects strain kwargs; `tb.py` **imports from `resistance/`** (circular dep, worked around with lazy `__getattr__`) |
| Transmission fitness | `rel_trans = max_fitness` set in `step_bookkeeping` (self-contained) | Separate **`ResistanceConnector`** multiplies `rel_trans`; **required** companion, warns if missing |
| Superinfection FOI reuse | Persistent state-dependent `susceptible`/`rel_sus` set each step | `infect()` override temporarily flips `susceptible`/`rel_sus` with per-state `alpha`, `try/finally` restore |
| Which-strain draw | `Strains.transmit_probs` ∝ fitness via CRN-safe `choice2d` | `StrainProfile.sample_transmitted_strain` ∝ fitness via ad-hoc `ss.random` |
| Treatment | `TxR`/`TxDeliveryR` — dedicated **rate-based** delivery, per-strain cure, replacement acquisition | `Regimen` + `StrainAwareTx`/`StrainAwareTxDelivery` — **subclass the existing cascade** (`Tx`/`TxDelivery`), per-drug efficacy, resolver-based acquisition, latent/relapse handling |
| TPT | **Not implemented** | `StrainAwareTPTTx` — per-strain sterilization + TPT-failure acquisition with state gradient |
| DST | `DST`/`DSTDelivery` — strain-level sens/spec + `p_strain_obs` bottleneck; integer profile | `DSTDx`/`DSTDelivery` — same core + `RegimenRouter` + `treatment_monitoring_eligibility` |
| De-novo acquisition | `_denovo` — scalar `q_prog`, mono-strain only, target always exists | `AcquisitionResolver.random_acquisition` — per-drug `p_random`, all strains, drops if target strain undeclared |
| Selective (treatment) acquisition | scalar `q_acq`, replacement, no state RR | per-drug `p_selective` + **state modifiers** (RR on `q`), replacement |
| Analyzers | `ResistanceStats` — resistant/super fractions + 3-way resistance-origin flux (de-novo / tx-acquired / transmitted), `to_df` for ODE comparison | `StrainResults` (per-strain carriers/active/new) + `DuplicateStrainAnalyzer` (blocked-superinfection counter) |
| Validation | Bit-identical Python port of `ode.r` + matched-IC transient ABM↔ODE comparison; 12 pytest tests on the spec's levers | 346 lines of construction/validation unit tests + a directional PASS/FAIL example harness (`run_resistance2.py`); **no** ODE validation |
| Public API to build products | `tb.strains` (public) | `tb._strain_registry` (private; examples reach into `strain_profile._tb`, `profile.names`) |
| RNG discipline | Module-owned dists discovered by `sc.search`; CRN-safe | Resolvers create `ss.random(strict=False)` and self-init outside module plumbing |

## The decisive architectural difference: strain representation

This is the single most consequential fork, and it is where the two implementations diverge most from each other and from the spec.

The spec is explicit in its first section: "*For `n` drugs/classes, there will be `m = 2ⁿ` possible strains*," and a strain is a bit-vector `X_j = {x_{1,j}, …, x_{n,j}}` whose integer id decodes to that vector (id 5 → `{1,0,1}` = RIF+FQ). The **current branch's bitmask is a literal encoding of this**: `Strains` auto-enumerates all `2ⁿ` strains, and the agent's `strain_mask` packs the *set* of carried strains into one integer. Because every one of the `2ⁿ` strains always exists, every resistance-emergence event — de-novo mutation or treatment acquisition — always has a well-defined target strain (`strain | drug_bit`). Nothing can be silently dropped.

PR #430 instead requires the user to **declare an explicit list of `StrainSpec`s**, one `ss.BoolArr` per strain. This has three consequences worth weighing:

- **It can silently drop resistance emergence.** `AcquisitionResolver._resistant_target_idx` searches the registry for the strain that differs by exactly one resistance bit; if that strain was not declared, "*the acquisition event is silently dropped*" (its own docstring). For a model whose entire purpose is to track the *emergence* of resistance, silently discarding emergence events is a genuine fidelity hazard, not a cosmetic one.
- **In practice its flexibility becomes busywork.** For the spec's realistic drug scope (RIF/BDQ, plus maybe INH/FQ — i.e. `n ≈ 2–4`), you want the full `2ⁿ` strain space so acquisition has somewhere to go. PR #430's own 3-drug demo (`run_resistance_demo.py`) declares all eight strains (`x1_pan … x8_bdq_fq`) **by hand** — i.e. it reconstructs the current branch's automatic `2ⁿ` enumeration manually, with an opportunity to omit one. The "declare only what you need" flexibility only pays off if you *deliberately* want a sparse strain set and accept dropped emergence, which the spec does not ask for.
- **It scales with the number of declared strains, not `2ⁿ`.** This is PR #430's real advantage: for a hypothetical high-`n` model where only a few strains circulate, its memory/compute is proportional to the declared set, whereas the bitmask decodes `n_agents × 2ⁿ` boolean matrices on the fly. This matters only beyond the spec's scope (roughly `n > 5`), and only when strains are genuinely sparse.

On the other side, PR #430's named strains (`rif_r`, `mdr`, with human-readable labels) are nicer for configuration, plotting, and results than the current branch's bare integer ids — and the spec's Notation section explicitly *invites* a naming layer ("*position 1 = 'RIF'*"). But naming and the `2ⁿ` bitmask are not mutually exclusive: the bitmask registry already uses name-keyed drug dicts, and could auto-generate a readable label per id (id 5 → `"RIF+FQ"`) trivially.

**Verdict: keep the bitmask (current branch).** It is the exact encoding the spec describes, it makes resistance emergence safe by construction, it is fully vectorized, and it removes an entire class of configuration error. Add auto-generated per-strain labels to recover PR #430's readability. Note the `2ⁿ` scaling limit honestly (fine to `n ≈ 5`; revisit only if the model ever needs many drugs with sparse strains — outside the current spec).

## Disease-class integration and Starsim style

The two implementations reuse Starsim's force-of-infection engine in the same spirit (set state-dependent `susceptible`/`rel_sus`/`rel_trans`, let the CRN engine do the transmission arithmetic, resolve which strain in `set_prognoses`) but package it very differently.

- **`tb.py` footprint.** The current branch makes a small, behavior-preserving refactor: it splits `TB.step()` into `step_transitions()` and `step_bookkeeping()` hooks and adds state-query properties (`latent`, `asymptomatic`, `active_tb`, …). `TBResistant` then lives entirely in `resistance/tb_resistant.py` and overrides those hooks. Crucially, **`tb.py` has no dependency on the resistance subpackage.** PR #430 instead introduces a `BaseTB`/`TB` split, makes `TB.__init__` raise `TypeError('Use MultiStrainTB …')`, and defines the ~260-line `MultiStrainTB` **inside `tb.py`**, which forces `tb.py` to import `StrainRegistry`/`StrainProfile`/`ProgressionResolver`/`AcquisitionResolver` from `resistance/`. That is a circular dependency (core → overlay → core), which PR #430 has to defuse with a lazy `__getattr__` in `resistance/__init__.py`. Putting the overlay disease in the shared core file, and coupling the core to the overlay, is the less clean and less Starsim-idiomatic arrangement. **Keep the current branch's separation.**
- **Fitness application.** The current branch sets `rel_trans = max_fitness` directly in `step_bookkeeping`, so the fitness cost is intrinsic to the disease and cannot be forgotten. PR #430 factors it into a separate `ResistanceConnector` that is *required* but easy to omit — hence its `_warn_if_missing_resistance_connector` guard. A connector is a legitimate Starsim pattern for cross-*module* coupling, but here the coupling is internal to the disease's own transmission, so the connector is over-separated and fragile (the warning is the tell). **Keep the current branch's baked-in fitness.**
- **Superinfection susceptibility.** The current branch sets `susceptible`/`rel_sus` persistently each step (the intended lifecycle for those arrays). PR #430 overrides `infect()` to temporarily set `susceptible=True` and scale `rel_sus` by per-state `alpha`, then restores in a `finally`. PR #430's version is more surgical (touches only infection targets) but more delicate. The current branch's persistent-set is the more idiomatic pattern. **Keep the current branch's approach**; PR #430's explicit per-state `alpha_act` dict is a nice way to *expose* the σ factors and is equivalent to the current branch's named `rr_reinfection_*` pars.
- **RNG discipline.** The current branch's distributions are module attributes (`choice2d`, `ss.bernoulli`) discovered and initialized through the normal Starsim plumbing, keeping draws CRN-safe. PR #430's resolvers construct `ss.random(name=…, strict=False)` and self-initialize them outside the owning module — a common-random-number smell that can undermine reproducibility across scenarios. **Keep the current branch's RNG ownership;** if the resolver pattern is adopted, give the resolvers RNGs owned by the disease module.
- **Public handle.** The current branch exposes `tb.strains` as the public object you pass to `TxR`/`DST`. PR #430's products need `tb._strain_registry` (nominally private), and its example/analysis code reaches further into `tb.strain_profile._tb` and `profile.names`. **Prefer a public accessor;** if PR #430's registry is adopted, expose it publicly.
- **Vectorization.** The current branch is fully vectorized over the bitmask. PR #430 has per-UID Python loops (e.g. the relapse strain snapshots keyed by `int(uid)` in `StrainAwareTxDelivery`), which will not scale well. If PR #430's treatment layer is adopted, vectorize those.

## Feature-by-feature fidelity to the spec

| Spec section | Current branch | PR #430 | More spec-faithful |
|----|----|----|----|
| Strain profiles `x_{i,j}`, `m=2ⁿ` | Exact (auto-enumerated bitmask) | Subset declared by hand; risks dropped emergence | **Current branch** |
| Multi-strain infection (set `Y_k`) | Bits of one integer | Multiple BoolArrs | Tie (both correct) |
| Transmission (`max` fitness, ∝-fitness split) | Reproduces worked table exactly | Reproduces same model | Tie |
| Reinfection ASY/SYM (Option 1 vs 2) | **Option 1** (explicit `rr_reinfection_asy/sym`, default 0) | **Option 1** (`alpha_act`, default 0) | Tie (same choice) |
| Reinfection *default coupling* (`inf=rec`, `non=inf`) | Diverges: defaults σ_L=σ_N=**1.0** (ODE null) | **Follows spec**: `alpha_super`←`rr_reinfection_rec`, `non`←`super` | **PR #430** |
| Identical-strain blocking + analyzer | `new_blocked_superinf` counter | `DuplicateStrainAnalyzer` + per-step counter | Tie |
| Progression bottleneck (`p_multi`, equal-prob) | `_bottleneck`, equal-prob default (+ optional fitness mode) | `ProgressionResolver`, equal-prob | Tie |
| Optional ψ (multi-strain progression rate) | `rr_prog_super` implemented | Not implemented | **Current branch** |
| Clearance clears all strains | Yes | Yes | Tie |
| Random acquisition: one-time at activation | Yes | Yes | Tie |
| Random acquisition: **per-drug `p_rand_i`**, each strain independently | Scalar `q_prog`, mono-strain only | **Per-drug dict, all strains** (but drops if target undeclared) | **PR #430** (mechanism) / **current branch** (no drops) |
| Random acquisition Option 1 vs 2 | Configurable, default mixed | Superinfection (adds variant) | Tie (spec says either) |
| Treatment: per-strain efficacy `T_l` | `base × ∏penalty` | `Regimen` per-drug + best-drug-drives-cure | Tie (both valid; PR #430 more mechanistic) |
| Treatment: adherence induces agent-level correlation | Yes (binary) | Yes (binary) | Tie (neither does the full per-agent *distribution*) |
| Treatment: partial cure leaves subset of strains | Yes | Yes | Tie |
| Selective acquisition **per-drug `q_{l,i}` + state RR** | Scalar `q_acq`, no state RR | **Per-drug + state modifiers** (RR default 1 for ASY/SYM, 0 else) | **PR #430** |
| Treatment delivery model | Rate-based (good for ODE validation) | Cascade-integrated (good for scenarios) | Tie (different, both useful) |
| **TPT** (strain-aware sterilization + failure acquisition, state gradient) | **Not implemented** | Implemented; state gradient (0.05/0.5/1/1) matches spec wording | **PR #430** |
| **DST** core (strain-level sens/spec, `p_strain_obs`=fitness) | Implemented | Implemented | Tie |
| **DST→treatment routing** by observed profile | `observed_resistant(drug)` callable (minimal) | `RegimenRouter.matches(DRUG=bool)` (composable) | **PR #430** |
| **Treatment monitoring** (time-under-treatment, regimen switch) | **Not implemented** | Eligibility factory implemented (mid-course *interrupt* likely incomplete — see caveats) | **PR #430** |
| Notation divergence (name/key based) | Name-keyed drug dicts | Named strains + name-keyed drugs | Tie (both invited) |
| Testing: before/after burden per 100k on `tb_LAI_TPT` | Not done (used SA ODE set) | Not done (directional-effects example only) | Tie (both incomplete) |
| Testing: **quantitative dynamical validation** | **ODE port + matched-IC comparison** | None | **Current branch** |
| Testing: parameter-effect checks | pytest tests on the levers | Directional PASS/FAIL example harness | Tie (different form) |
| Testing: construction/input validation | Minimal | Extensive | **PR #430** |

Two rows deserve emphasis because they reverse the "current branch is cleaner" theme and are honest wins for PR #430:

- **Default reinfection coupling.** The spec says to default `rr_reinfection_inf = rr_reinfection_rec` and `rr_reinfection_non = rr_reinfection_inf`. PR #430's `_finalize_alpha_defaults` does exactly this. The current branch instead defaults σ_L = σ_N = 1.0 (the reference-ODE null, chosen for validation convenience), so it silently *diverges* from the spec's intended epidemiological default. **Adopt PR #430's spec-faithful default** (while keeping the ability to set 1.0 for ODE runs).
- **Selective- and random-acquisition detail.** The spec asks for per-drug acquisition probabilities (`p_rand_i`, `q_{l,i}`) and an acquisition RR that varies by TB state ("*default 1 for ASYMPTOMATIC and SYMPTOMATIC, 0 for other states*"). PR #430 implements both precisely (per-drug dicts + `DEFAULT_STATE_MODIFIERS`/`DEFAULT_TPT_STATE_MODIFIERS`). The current branch collapses these to single scalars with no state RR. **Adopt PR #430's per-drug + state-varying acquisition** — but implement it on the bitmask so the target strain always exists.

## Where each implementation is stronger, summarized

**Current branch is stronger on:** spec fidelity of the core representation (`2ⁿ` bitmask, safe emergence); clean, non-circular `tb.py` refactor with the overlay disease in its own subpackage; self-contained transmission (no fragile required connector); idiomatic persistent `rel_sus`; correct module-owned CRN RNGs; public `tb.strains` handle; full vectorization; and rigorous **dynamical** validation against the reference ODE. It is the better *foundation*.

**PR #430 is stronger on:** breadth of spec coverage — it actually implements TPT, DST→regimen routing, treatment-monitoring eligibility, per-drug and state-varying acquisition, and a first-class `Regimen` abstraction; it integrates treatment with the existing HSB→Dx→Tx cascade (closer to the spec's care-cascade framing and composable with the rest of TBsim); it follows the spec's default reinfection coupling; and it has thorough input validation and a directional parameter-effects harness. It is the better *feature set*.

## Recommendation: what to keep from each

### Keep from the current branch (the foundation)

1. **Bitmask `strain_mask` + auto-enumerated `Strains` (`2ⁿ`).** The spec's literal model; guarantees emergence targets; compact and vectorized. Add auto-generated readable strain labels for output/plots.
2. **`TBResistant(TB)` in the `resistance/` subpackage + the minimal `step_transitions`/`step_bookkeeping` hook refactor of `tb.py`.** No circular dependency; the shared core stays overlay-agnostic.
3. **Fitness folded into `step_bookkeeping`** (drop the separate required `ResistanceConnector`).
4. **Persistent state-dependent `rel_sus`** for superinfection (not the `infect()` temporary-flip).
5. **Module-owned CRN distributions / `choice2d`** for the which-strain and bottleneck draws.
6. **The ODE validation** (`two_strain_ode.py` + `validate_resistance_abm_vs_ode.py`) as the correctness backbone, and the rate-based delivery path that makes clean ODE comparison possible.
7. **`ResistanceStats`' 3-way resistance-origin flux decomposition** (de-novo / treatment-acquired / transmitted) — it directly answers the spec's "where does resistance come from" question.

### Port in from PR #430 (the missing features), rebuilt on the bitmask

1. **Strain-aware TPT** (`StrainAwareTPTTx`): per-strain sterilization so clearing a susceptible strain lets a resistant one progress/transmit, plus TPT-failure acquisition with the state gradient (INFECTION ≪ NON_INFECTIOUS < ASY = SYM). Spec-required; currently absent.
2. **Per-drug + state-varying acquisition** for both de-novo (`p_rand_i`) and treatment failure (`q_{l,i}` + state RR). Replace the current scalar `q_prog`/`q_acq`. Because the bitmask always has a target strain, drop PR #430's silent-drop path.
3. **`Regimen` abstraction** (drugs + per-drug efficacy + combine mode) and **cascade-integrated treatment** (`StrainAwareTx`/`StrainAwareTxDelivery` subclassing `Tx`/`TxDelivery`) as the delivery path for real scenarios — while retaining the rate-based delivery for ODE validation. (Make treatment initiation pluggable: rate-based *or* eligibility/cascade.) Vectorize PR #430's per-UID relapse bookkeeping when porting.
4. **DST→regimen router** (`RegimenRouter.matches(DRUG=bool)`) and **treatment-monitoring eligibility** (`treatment_monitoring_eligibility`), which realize the spec's "treatment provision dependent on observed DST profile" and "treatment monitoring" sections. Verify the mid-course interrupt actually fires (see caveats) before claiming regimen-switching is complete.
5. **The spec's default reinfection coupling** (`rr_reinfection_inf = rr_reinfection_rec`, `rr_reinfection_non = rr_reinfection_inf`), overridable to 1.0 for ODE null runs.
6. **Input validation** on the strain registry and regimens (reject bad fitness/resistance/duplicate ids, etc.).
7. **`DuplicateStrainAnalyzer` and per-strain `StrainResults` channels** as complements to `ResistanceStats` (per-strain prevalence + the blocked-superinfection diagnostic the spec explicitly asks for).
8. **The directional parameter-effects harness** (`run_resistance2.py`'s PASS/FAIL checks on acquisition/fitness/treatment/DST) — promote it into the automated test suite alongside the ODE checks, and complete the spec's requested burden-per-100k before/after table using the `tb_LAI_TPT` parameter set (neither implementation did this yet).

### Do not carry over

- PR #430's `MultiStrainTB`-inside-`tb.py` and the resulting core→overlay circular import — use the current branch's subpackage subclass.
- The **required-but-separate `ResistanceConnector`** — bake fitness into the disease.
- The **ad-hoc `ss.random(strict=False)`** RNGs in the resolvers — use module-owned dists.
- Reliance on the **private `_strain_registry`** and `strain_profile._tb`/`names` internals — expose a public handle.
- The **scratch example scripts**: `run_resistance.py` has a live `NameError` (`plt.subplots` with only `pylab` imported), and `run_resistance_critical_paths.py` carries dead code. Keep `run_resistance2.py` (clean, spec-aligned, self-checking); trim `run_resistance_demo.py` down to a single supported demo that uses public accessors.
- The current branch's **over-simplifications** (scalar `q_prog`/`q_acq`, mono-strain-only de-novo) — superseded by item 2 above.

## Caveats to verify before merging

- **Treatment monitoring / mid-course interrupt.** PR #430 provides `treatment_monitoring_eligibility` (selects agents on treatment past a threshold) and a router to route them to a second-line `StrainAwareTxDelivery`. But `StrainAwareTxDelivery.step_start_treatment` acts on agents in active-disease states (`NON_INFECTIOUS`/`ASYMPTOMATIC`/`SYMPTOMATIC`), whereas a monitored agent is in `TREATMENT` state — so the switch may not actually fire mid-course. The spec explicitly flags the need to "*prematurely stop/change an ongoing treatment regimen*." Confirm whether PR #430's regimen switch is functional or only scaffolding before relying on it; the interrupt mechanism likely still needs to be built in either implementation.
- **Efficacy functional form.** PR #430's `Regimen` uses "best susceptible drug drives cure" (`combine='max'`/`'parallel'`); the current branch uses a multiplicative penalty per resistant drug. The spec mandates neither (it only says full efficacy for susceptible strains, reduced for resistant). Pick one deliberately; PR #430's is arguably more mechanistic for combination therapy.
- **Adherence distribution.** The spec asks for adherence as a per-agent *distribution* that correlates outcomes across strains. Both implementations currently use a single per-agent Bernoulli. Neither fully satisfies this; add the distribution when the treatment layer is consolidated.
- **`tb_LAI_TPT` burden validation.** Neither implementation produced the spec's requested before/after burden-per-100,000 table (prevalence, asymptomatic incidence, mortality) using the `tb_LAI_TPT` parameter set. This should be completed on the merged implementation.
