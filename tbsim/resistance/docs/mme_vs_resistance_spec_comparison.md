# `mme-resistance` vs `resistance` — capability comparison against the tech spec

Date: 2026-07-14. Reference spec: [`tbsim/resistance/docs/tbsim-resistance-tech-spec.md`](tbsim/resistance/docs/tbsim-resistance-tech-spec.md).

## Summary

`mme-resistance` and the accepted `resistance` branch (which now contains `resistance-uat` via PR #437) are **two independent implementations of the same tech spec**, not successive versions of one. They share a last common ancestor at `66f10ce` (2026-07-03) and have diverged ever since — `mme-resistance` added ~40 commits, the `resistance` lineage ~35, with **no merges between them** (the `resistance` side recorded the connection with a `-s ours` merge, `4b9a7d6`, that kept none of mme's content). Their `tbsim/resistance/` packages are architecturally disjoint: mme uses `multistrain_tb.py` / `connector.py` / `resolvers.py` / `regimens.py` / `spec.py` / `tx.py` / `diagnostics.py`; the accepted branch uses `tb_resistant.py` / `dst.py` / `treatments.py` plus a `devtests/` suite.

Judged against the spec's **normative requirements**, the accepted branch is the better implementation overall: it is correct on the two mechanisms the spec specifies most strictly (de-novo/selective resistance acquisition, and fitness-weighted transmission), where mme carries latent correctness bugs. mme's extra capabilities are mostly **out of the spec's scope** (relapse natural history, latent clearance by the main treatment product, DST lab-dropout stages) rather than gaps in the accepted branch. The one genuine, spec-backed improvement mme had that the accepted branch lacked — **strain-specific TPT protection-from-progression** — has now been ported to the accepted branch (see "Change applied" below).

## Framing caveat

The tech spec was authored alongside the accepted branch: every indented "**Implementation —**" block in the spec names the accepted branch's own modules (`tb_resistant.py`, `treatments.py`, `dst.py`, `TxR`, `TxDeliveryR`, `DST`, …). On the accepted branch the spec is therefore partly self-descriptive. This comparison judges both branches only against the **non-indented normative prose** (the actual requirements), not those implementation callouts.

## Verdict on each capability difference

| Difference (branch that has it) | Spec requirement | Verdict |
|---|---|---|
| Auto-enumerated 2ⁿ strains; acquisition target always exists (accepted) | §Random & §Selective Acquisition: acquisition *always* yields the resistant strain — all `m = 2ⁿ` strains exist; worked examples add/replace with a strain that must exist | **Accepted correct; mme has a latent spec violation** — mme silently drops an acquisition event when the resistant target is not pre-declared in its `catalog` (`resolvers.py`, `if target_idx is None: continue`). Resistance the spec says must emerge can fail to. |
| Fitness-weighted strain passing at transmission (both, differently) | §Transmission: the passed strain is drawn ∝ fitness (multinomial; worked example 56%/44%, 0.28β/0.22β) | Both are compliant *when configured*, but mme requires a bolt-on `ResistanceConnector` and **warns it degrades to unweighted `rel_trans` if the connector is absent** (`multistrain_tb.py`). The accepted branch gets fitness-weighting by construction (reuses Starsim's FOI: `rel_trans = max_fitness`, passed strain via `transmit_probs`). Accepted is the more robust match. |
| Strain-specific TPT protection-from-progression (mme; **now also accepted**) | §TPT: efficacy "varying by regimen and resistance/strain"; the resistance-unmasking dynamic (Cohen/Mills/Kunkel papers) | **mme was more faithful.** The accepted branch cleared susceptible strains per-strain (correct) but applied *whole-agent* `rr_*` progression-protection under suppression, so a surviving *resistant* strain was also shielded from progressing — muting the unmasking-via-progression channel. Ported to the accepted branch (below). *Caveat: at the default `p_multi = 1`, the spec itself notes the progression-unmasking channel is largely absent, so the dynamical impact is second-order.* |
| Relapse modeled as an unsuccessful outcome → acquisition (mme) | §Selective Acquisition: `q_{l,i}` applies at "failure, relapse, and LTFU (…right now it isn't [modeled])" | **mme implements the spec's wording more literally** (relapse is an explicit unsuccessful outcome that triggers `q`). The accepted branch applies `q_acq` on failure only. Whether this is a real gap depends on whether base tbsim TB emits relapse events the resistance layer should intercept — worth confirming, but relapse is named in-scope by the spec. mme's full relapse *engine* (scheduling, strain restore) is otherwise natural-history scope, not resistance-spec scope. |
| DST `p_sample` / `p_culture` lab-dropout stages (mme) | §Diagnostics: the sample-collection/culture bottleneck is **folded into the single `p_strain_obs`** parameter | **Accepted matches the spec as written**; mme decomposes the bottleneck into extra upstream stages. Compatible with the spirit, but not required — mme is "extra," not "more correct." |
| DST drug-panel subset (`drugs=`); DST delivery `coverage` (mme) | §Diagnostics: profile over "chosen drugs/classes"; eligibility immediate or failure-dependent | Not required by the spec; reasonable operational extras. Neutral. |
| Latent clearance by the *main* treatment product (mme) | Spec cleanly separates **Treatment (active TB)** from **TPT (latent/preventive)**; latent clearance is TPT's job | **Accepted is more spec-aligned** — it routes latent clearance through `TPTRx`, not the treatment product. mme treating INFECTION agents with the main Tx blurs the spec's Tx/TPT split. |
| `Regimen` abstraction: per-drug efficacy, `combine='max'|'parallel'`, `resistance_penalty` (mme) | §Treatment: per-strain efficacy `T_l = {t_{1,l}, …, t_{m,l}}`; §Notation invites divergence | Both satisfy `T_l`. mme is richer; the accepted branch's `base × ∏ resist_penalty` is sufficient. Neutral. |
| `prog_resist_mode='replacement'`, `prog_select='fitness'` (accepted) | §Progression: bottleneck strain choice = **equal probability**; §Random Acquisition: replacement *or* superinfection is "up to software" | Both defaults are spec-compliant (equal / mixed). The accepted branch merely exposes extra optional modes. Neutral. |

## Bottom line against the spec

1. The accepted branch is the spec-correct one on the two mechanisms the spec specifies most strictly. Acquisition that *always* produces its target strain, and transmission that is fitness-weighted by construction, are core worked-example requirements. mme can silently drop acquisitions (undeclared catalog target) and can silently fall back to unweighted transmission (missing connector) — latent correctness bugs, not stylistic choices.
2. mme was more faithful in exactly one place: TPT progression-protection should be strain-specific, and the accepted branch's whole-agent suppression over-protected surviving resistant strains. This was the one genuine, spec-backed improvement mme had — now ported.
3. Everything else mme has that the accepted branch lacks is either **beyond the spec's scope** (relapse natural history, `Regimen` combine-modes) or moves **away** from how the spec is written (latent clearance via the main Tx; DST lab-dropout stages that the spec deliberately folds into `p_strain_obs`).

## Change applied

Added a coverage-weighted `apply_protection` override to `TPTRx` in [`tbsim/resistance/tpt.py`](tbsim/resistance/tpt.py). Base `TPTTx` protects a suppressed agent as a whole (multiplying `rr_activation` / `rr_clearance` / `rr_death` by the sampled modifiers). The override scales each modifier by the fraction `w` of the agent's carried strains the regimen actually covers, blending each modifier toward 1 (no effect) by the uncovered fraction: `rr' = 1 − w·(1 − modifier)`.

- An agent carrying only regimen-resistant strains (`w = 0`) gets **no** protection — its resistant strain is free to progress (the spec's unmasking dynamic).
- An all-susceptible agent (`w = 1`) gets **full** protection, unchanged from base behavior.
- Mixed infections get **partial** protection, weighted by coverage.
- `w` is recomputed each step from the agent's current strains, so it tracks strain changes (bottleneck, de-novo acquisition).
- Non-strain `TB` (not `TBResistant`) falls back to the base whole-agent behavior.

Verified: all resistance devtests (`tbsim/resistance/devtests/`) and `tests/test_resistance.py` pass, and a direct check confirms a pan-susceptible carrier receives the full `activation_modifier` (0.4) while a RIF+INH-resistant-only carrier receives none (factor 1.0).
