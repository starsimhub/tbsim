# TBsim Resistance & Multi-Strain — User Manual

A practical guide for researchers using TBsim’s drug-resistance and multi-strain extension (`tbsim.resistance`). It focuses on **how to set up, run, and interpret** simulations. For the mathematical specification see [tbsim-resistance-tech-spec.md](tbsim-resistance-tech-spec.md); for implementation internals see [tbsim-resistance-implementation.md](tbsim-resistance-implementation.md).

**Audience:** epidemiologists and modelers comfortable with Python / Starsim.  
**Requirements:** Python ≥ 3.11, `tbsim` installed editable (`pip install -e .`), Starsim ≥ 3.5.

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Concepts in brief](#2-concepts-in-brief)
3. [Building a multi-strain simulation](#3-building-a-multi-strain-simulation)
   - [3.1 Defining strains](#31-defining-strains)
   - [3.2 Creating `TBResistant`](#32-creating-tbresistant)
   - [3.3 Seeding the epidemic](#33-seeding-the-epidemic)
   - [3.4 Reading results](#34-reading-results)
4. [Transmission and fitness](#4-transmission-and-fitness)
5. [Superinfection and competition](#5-superinfection-and-competition)
6. [De-novo resistance](#6-de-novo-resistance)
7. [Treatment and acquired resistance](#7-treatment-and-acquired-resistance)
8. [Drug-susceptibility testing (DST)](#8-drug-susceptibility-testing-dst)
9. [Treatment monitoring and regimen switching](#9-treatment-monitoring-and-regimen-switching)
10. [Strain-aware preventive therapy (TPT)](#10-strain-aware-preventive-therapy-tpt)
11. [Analyzing where resistance comes from](#11-analyzing-where-resistance-comes-from)
12. [Parameter reference (cheat sheet)](#12-parameter-reference-cheat-sheet)
13. [Known limitations](#13-known-limitations)
14. [Further reading](#14-further-reading)

---

## 1. Quick start

The shortest path from install to a multi-strain run:

```python
import numpy as np
import starsim as ss
import tbsim

# 1) Disease module with two drugs and fitness costs
tb = tbsim.TBResistant(
    drugs=['RIF', 'BDQ'],
    rel_fitness={'RIF': 0.9, 'BDQ': 0.85},
    pars=dict(
        beta=ss.permonth(0.35),
        init_prev=ss.bernoulli(0.10),
        init_strains=[0.85, 0.12, 0.03, 0.0],  # pan / RIF / BDQ / RIF+BDQ
    ),
)

# 2) Small contact network + sim
net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=8), dur=0))
sim = ss.Sim(
    n_agents=2000,
    networks=net,
    diseases=tb,
    dt=ss.days(30),
    start=ss.date('2000-01-01'),
    stop=ss.date('2030-12-31'),
    rand_seed=0,
    verbose=0,
)
sim.run()

# 3) Inspect resistance among active TB
res = sim.results.tb
print('Final active prevalence:', float(res['prevalence_active'][-1]))
print('Final any-resistance fraction:', float(res['frac_resist'][-1]))
print('Final RIF-resistant fraction:', float(res['frac_resist_RIF'][-1]))
```

**Reusable helper** used in later sections (copy once into a notebook or script):

```python
import starsim as ss
import tbsim


def build_sim(tb, interventions=None, analyzers=None, n_agents=2000,
              start='2000-01-01', stop='2035-12-31', seed=0):
    """Small multi-strain TB sim on a random contact network."""
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=8), dur=0))
    return ss.Sim(
        n_agents=n_agents,
        networks=net,
        diseases=tb,
        interventions=interventions,
        analyzers=analyzers,
        dt=ss.days(30),
        start=ss.date(start),
        stop=ss.date(stop),
        rand_seed=seed,
        verbose=0,
    )
```

---

## 2. Concepts in brief

| Idea | Meaning in TBsim |
|------|------------------|
| **Drug / class** | Named label you choose (`'RIF'`, `'BDQ'`, `'INH'`, …). Nothing is hard-coded. |
| **Strain** | One of `2ⁿ` binary resistance profiles over `n` drugs. Id `0` = pan-susceptible. |
| **Fitness cost `r_i`** | Multiplicative reduction in transmission for resistance to drug `i` (`∈ [0,1]`). Strain fitness = product of costs for drugs it resists. |
| **`strain_mask`** | Per-agent integer: bit `j` set means the agent carries strain `j`. Agents can carry several strains (superinfection). |
| **Transmission bottleneck** | A source transmits at the rate of its **fittest** carried strain; *which* strain is passed is drawn ∝ fitness. |
| **Product / delivery** | Same pattern as the rest of TBsim: products define *what* (efficacy, DST, TPT); deliveries define *who / when*. |

Natural history (SUSCEPTIBLE → INFECTION → …) remains **agent-level**. Strains are an overlay on that state machine.

---

## 3. Building a multi-strain simulation

### 3.1 Defining strains

```python
import tbsim

strains = tbsim.Strains(drugs=['RIF', 'BDQ'], rel_fitness={'RIF': 0.5, 'BDQ': 0.8})
print(f'{strains.n} drugs → {strains.m} strains')
for j in range(strains.m):
    print(f'  id {j}: {strains.labels[j]:9s} '
          f'profile={strains.profile[j].astype(int)} '
          f'fitness={strains.fitness[j]:.2f}')
```

Expected layout for two drugs:

| id | label | profile | fitness (example) |
|----|-------|---------|-------------------|
| 0 | pan | `{0,0}` | 1.00 |
| 1 | RIF | `{1,0}` | 0.50 |
| 2 | BDQ | `{0,1}` | 0.80 |
| 3 | RIF+BDQ | `{1,1}` | 0.40 |

Adding a third drug (e.g. `'FQ'`) only requires appending to `drugs`; `m` becomes `8`.

You usually do **not** construct `Strains` yourself for a sim — `TBResistant` builds it and exposes it as `tb.strains` for products (`TxR`, `DST`, `TPTRx`).

### 3.2 Creating `TBResistant`

`TBResistant` is a drop-in replacement for `tbsim.TB`:

```python
import starsim as ss
import tbsim

tb = tbsim.TBResistant(
    drugs=['RIF', 'BDQ'],
    rel_fitness={'RIF': 0.9, 'BDQ': 0.85},
    pars=dict(
        beta=ss.permonth(0.35),
        init_prev=ss.bernoulli(0.10),
        # Superinfection susceptibility (σ); defaults couple to rr_reinfection_rec
        rr_reinfection_inf=1.0,
        rr_reinfection_non=1.0,
        rr_reinfection_asy=0.0,   # no superinfection in active disease (default)
        rr_reinfection_sym=0.0,
        p_multi=1.0,              # keep all strains when progressing to ASY
    ),
)
print(tb.strains.labels)
```

The module name defaults to `'tb'`, so existing interventions that look up disease `'tb'` continue to work.

### 3.3 Seeding the epidemic

`init_strains` is a probability vector over strain ids for **seeded** infections (length `m`, need not be normalized — it is renormalized internally):

```python
import starsim as ss
import tbsim

# Two-drug space: [pan, RIF, BDQ, RIF+BDQ]
tb = tbsim.TBResistant(
    drugs=['RIF', 'BDQ'],
    rel_fitness={'RIF': 0.9, 'BDQ': 0.85},
    pars=dict(
        beta=ss.permonth(0.35),
        init_prev=ss.bernoulli(0.10),
        init_strains=[0.85, 0.12, 0.03, 0.0],
    ),
)
sim = build_sim(tb, stop='2010-12-31')  # uses helper from §1
sim.run()
print('Seeded strain labels:', tb.strains.labels)
print('Final frac_resist:', float(sim.results.tb['frac_resist'][-1]))
```

Default (if omitted): all seeds are pan-susceptible (`[1, 0, …, 0]`).

### 3.4 Reading results

Alongside standard TB outputs, `TBResistant` records:

| Result | Meaning |
|--------|---------|
| `frac_resist` | Fraction of active TB with any resistance |
| `frac_resist_<drug>` | Fraction resistant to that drug (aggregate phenotype) |
| `frac_super` | Fraction of active TB that is superinfected |
| `new_denovo_resistance` | De-novo acquisition events this step |
| `new_transmitted_resistance` | New acquisitions of a resistant strain via transmission |
| `new_blocked_superinf` | Identical-strain re-exposures that were blocked |

```python
import starsim as ss
import tbsim

tb = tbsim.TBResistant(
    drugs=['RIF', 'BDQ'],
    rel_fitness={'RIF': 0.9, 'BDQ': 0.85},
    pars=dict(
        beta=ss.permonth(0.35),
        init_prev=ss.bernoulli(0.10),
        init_strains=[0.85, 0.12, 0.03, 0.0],
    ),
)
sim = build_sim(tb, stop='2015-12-31')  # uses helper from §1
sim.run()
res = sim.results.tb
print('Mean frac_resist (last 5 years of series):',
      float(res['frac_resist'][-60:].mean()) if len(res['frac_resist']) >= 60
      else float(res['frac_resist'].mean()))
print('Cumulative blocked superinfections:', int(res['new_blocked_superinf'].sum()))
```

For origin decomposition and per-strain counts, see [§11](#11-analyzing-where-resistance-comes-from).

---

## 4. Transmission and fitness

**Rules (as implemented):**

1. Infectees only receive a strain the source already carries (no resistance emergence on transmission).
2. Overall infectiousness = fitness of the **fittest** carried strain.
3. Conditional on transmission, which strain is passed ∝ fitness among carried strains.

Reproduce the specification’s worked example (≈56% / 44% → 0.28β / 0.22β):

```python
import numpy as np
import tbsim

s = tbsim.Strains(['RIF', 'BDQ'], rel_fitness={'RIF': 0.5, 'BDQ': 0.8})
mask = np.array([(1 << 1) | (1 << 3)])  # carries {RIF} and {RIF,BDQ}
print('max fitness (relative infectiousness):', s.max_fitness(mask)[0])
tp = s.transmit_probs(mask)[0]
print(f'P(pass RIF)     = {tp[1]:.1%}')
print(f'P(pass RIF+BDQ) = {tp[3]:.1%}')
print(f'rate RIF        = {tp[1] * 0.5:.2f} β')
print(f'rate RIF+BDQ    = {tp[3] * 0.5:.2f} β')
```

Implication for research: **superinfection does not dilute** a source’s overall transmission risk relative to mono-infection with its fittest strain, but it does split which strain is passed.

---

## 5. Superinfection and competition

Already-infected agents can acquire a **second** (distinct) strain. Relative risk vs a fully susceptible person depends on disease state:

| State | Parameter | Default behavior |
|-------|-----------|------------------|
| INFECTION | `rr_reinfection_inf` | Defaults to `rr_reinfection_rec` |
| NON_INFECTIOUS | `rr_reinfection_non` | Defaults to `rr_reinfection_inf` |
| ASYMPTOMATIC | `rr_reinfection_asy` | `0` (closed) |
| SYMPTOMATIC | `rr_reinfection_sym` | `0` (closed) |

Protection is **strain-agnostic** and **count-agnostic** (a third distinct strain is not harder than a second). Identical-strain re-exposure is blocked and counted in `new_blocked_superinf`.

Fitness costs drive competition. Without treatment, a less-fit resistant strain tends to decline:

```python
import matplotlib.pyplot as plt
import starsim as ss
import tbsim


def resist_over_time(sigma):
    tb = tbsim.TBResistant(
        rel_fitness={'TX': 0.7},  # single drug, 30% fitness cost
        pars=dict(
            beta=ss.permonth(0.35),
            init_prev=ss.bernoulli(0.12),
            init_strains=[0.7, 0.3],
            rr_reinfection_inf=sigma,
            rr_reinfection_non=sigma,
        ),
    )
    sim = build_sim(tb, stop='2050-12-31')
    sim.run()
    return sim.results.timevec, sim.results.tb['frac_resist'], sim.results.tb['frac_super']


fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for sigma, label in [(0.0, 'σ = 0 (no superinfection)'), (1.0, 'σ = 1')]:
    t, fr, fs = resist_over_time(sigma)
    axes[0].plot(t, fr, label=label)
    axes[1].plot(t, fs, label=label)
axes[0].set(title='Resistant fraction', xlabel='year', ylabel='fraction')
axes[1].set(title='Superinfected fraction', xlabel='year', ylabel='fraction')
axes[0].legend(frameon=False)
axes[1].legend(frameon=False)
plt.tight_layout()
plt.show()
```

---

## 6. De-novo resistance

Resistance can arise endogenously at progression **out of INFECTION** (→ NON_INFECTIOUS or → ASYMPTOMATIC), as a one-time per-drug probability — not a per-timestep rate.

| Parameter | Role |
|-----------|------|
| `p_rand` | Dict `{drug: probability}` per not-yet-resistant drug |
| `prog_resist_mode` | `'mixed'` (add resistant variant → superinfection; default) or `'replacement'` |

```python
import starsim as ss
import tbsim

tb = tbsim.TBResistant(
    rel_fitness={'TX': 0.9},
    pars=dict(
        beta=ss.permonth(0.35),
        init_prev=ss.bernoulli(0.12),
        init_strains=[1.0, 0.0],       # start 100% pan-susceptible
        p_rand={'TX': 0.02},
        prog_resist_mode='mixed',
        rr_reinfection_inf=0.0,
        rr_reinfection_non=0.0,
    ),
)
sim = build_sim(tb, stop='2050-12-31')
sim.run()
print('Cumulative de-novo events:', int(sim.results.tb['new_denovo_resistance'].sum()))
print('Final resistant fraction:', float(sim.results.tb['frac_resist'][-1]))
print('Final superinfected fraction:', float(sim.results.tb['frac_super'][-1]))
```

Both modes create resistance; only `'mixed'` produces lasting AB (superinfected) agents from de-novo events.

---

## 7. Treatment and acquired resistance

Use the product/delivery pair:

- **`TxR`** — per-strain efficacy, adherence, acquisition-on-failure (`q_acq`)
- **`TxDeliveryR`** — who starts treatment and when (rates from ASY/SYM, or a custom `eligibility` callable)

Efficacy for strain `j` is `base_efficacy` × product of `resist_penalty` over **regimen** drugs that strain resists. Failed courses can acquire resistance to regimen drugs by **replacement**, once per episode, scaled by TB-state RR (`acq_state_rr`; default 1 for ASY/SYM, 0 elsewhere).

```python
import starsim as ss
import tbsim

tb = tbsim.TBResistant(
    rel_fitness={'TX': 0.6},
    pars=dict(
        beta=ss.permonth(0.35),
        init_prev=ss.bernoulli(0.12),
        init_strains=[0.95, 0.05],
        rr_reinfection_inf=1.0,
        rr_reinfection_non=1.0,
    ),
)
tx = tbsim.TxDeliveryR(
    name='tx',
    product=tbsim.TxR(
        strains=tb.strains,
        base_efficacy=0.8,
        resist_penalty={'TX': 0.2},
        adherence=0.9,
        q_acq={'TX': 0.04},
    ),
    rate_sym=ss.peryear(1.5),
    rate_asym=ss.peryear(0.1),
)
sim = build_sim(tb, interventions=tx, stop='2050-12-31')
sim.run()

print('Courses started:', int(sim.results['tx'].n_treated.sum()))
print('Acquisitions on failure:', int(sim.results['tx'].n_acquired.sum()))
print('Final resistant fraction:', float(sim.results.tb['frac_resist'][-1]))
```

**Research tip:** Lower `resist_penalty` values (stronger efficacy loss against resistant strains) and higher `q_acq` both tend to raise the resistant share of active TB — useful for sensitivity analysis.

Partial cure is supported: if only some strains clear, the agent returns to the pre-treatment TB state carrying the survivors.

---

## 8. Drug-susceptibility testing (DST)

`DST` produces an **observed n-drug profile** (not strain identities). Sensitivity/specificity are applied at the strain level; `p_strain_obs` (default = strain fitness) can drop strains from the sample. `DSTDelivery.matches(...)` turns the observed profile into eligibility callables for regimen routing.

```python
import starsim as ss
import tbsim

tb = tbsim.TBResistant(
    drugs=['RIF', 'BDQ'],
    rel_fitness={'RIF': 0.9, 'BDQ': 0.85},
    pars=dict(
        beta=ss.permonth(0.35),
        init_prev=ss.bernoulli(0.12),
        init_strains=[0.8, 0.2, 0.0, 0.0],
        rr_reinfection_inf=1.0,
        rr_reinfection_non=1.0,
    ),
)
dst = tbsim.DSTDelivery(
    name='dst',
    product=tbsim.DST(strains=tb.strains, sens=0.95, spec=0.98),
    eligibility=lambda sim: tbsim.get_tb(sim, which=tbsim.TBResistant).active_tb.uids,
)
first = tbsim.TxDeliveryR(
    name='first',
    eligibility=dst.matches(RIF=False),
    product=tbsim.TxR(
        strains=tb.strains,
        regimen_drugs=['RIF'],
        base_efficacy=0.85,
        resist_penalty={'RIF': 0.1},
    ),
)
second = tbsim.TxDeliveryR(
    name='second',
    eligibility=dst.matches(RIF=True),
    product=tbsim.TxR(
        strains=tb.strains,
        regimen_drugs=['BDQ'],
        base_efficacy=0.8,
    ),
)
sim = build_sim(tb, interventions=[dst, first, second], stop='2040-12-31')
sim.run()
print('DST tests:', int(sim.results['dst'].n_tested.sum()))
print('First-line courses:', int(sim.results['first'].n_treated.sum()))
print('Second-line (RIF-R→BDQ):', int(sim.results['second'].n_treated.sum()))
```

Also available: `dst.observed_resistant('RIF')` for a single-drug eligibility callable.

---

## 9. Treatment monitoring and regimen switching

To change regimen mid-course:

1. Name the first-line delivery.
2. Build a second-line delivery with `eligibility=treatment_monitoring_eligibility(...)` and `supersedes=['first']`.
3. Optionally pass `extra=` (e.g. `dst.matches(RIF=True)`) so the switch is contingent on a DST result or other filter.
4. The second line **interrupts** the ongoing course, then starts the new regimen.

```python
import starsim as ss
import tbsim

tb = tbsim.TBResistant(
    drugs=['INH', 'RIF'],
    rel_fitness={'INH': 0.95},
    pars=dict(
        beta=ss.permonth(0.3),
        init_prev=ss.bernoulli(0.10),
        init_strains=[0.5, 0.5, 0.0, 0.0],  # pan + INH-resistant
    ),
)
first = tbsim.TxDeliveryR(
    name='first',
    rate_sym=ss.peryear(2.0),
    product=tbsim.TxR(
        strains=tb.strains,
        regimen_drugs=['INH'],
        base_efficacy=0.8,
        resist_penalty={'INH': 0.1},
    ),
)
switch = tbsim.TxDeliveryR(
    name='switch',
    supersedes=['first'],
    # Time-on-treatment gate; add extra=dst.matches(...) to require a DST phenotype too.
    eligibility=tbsim.treatment_monitoring_eligibility('first', after_steps=2),
    product=tbsim.TxR(
        strains=tb.strains,
        regimen_drugs=['RIF'],
        base_efficacy=0.85,
    ),
)
sim = build_sim(tb, interventions=[first, switch], stop='2015-12-31')
sim.run()
print('First-line initiations:', int(sim.results['first'].n_treated.sum()))
print('Switched to second-line:', int(sim.results['switch'].n_treated.sum()))
```

`after_steps` is in **simulation timesteps** (with `dt=ss.days(30)`, `after_steps=2` ≈ 2 months).

---

## 10. Strain-aware preventive therapy (TPT)

`TPTRx` sterilizes **per strain**: only strains susceptible to every drug in the TPT regimen are cleared. A resistant strain in a co-infected agent can survive and later progress/transmit — the classic “TPT unmasks resistance” dynamic. Ineffective TPT can also select resistance (`p_tpt_acq`), scaled by TB state.

```python
import matplotlib.pyplot as plt
import starsim as ss
import tbsim


def run_tpt(with_tpt):
    tb = tbsim.TBResistant(
        drugs=['INH'],
        rel_fitness={'INH': 0.9},
        pars=dict(
            beta=ss.permonth(0.35),
            init_prev=ss.bernoulli(0.15),
            init_strains=[0.8, 0.2],
            rr_reinfection_inf=0.0,
            rr_reinfection_non=0.0,
        ),
    )
    ivs = None
    if with_tpt:
        ivs = tbsim.TPTSimple(
            product=tbsim.TPTRx(
                strains=tb.strains,
                regimen_drugs=['INH'],
                pars=dict(
                    efficacy=ss.bernoulli(0.9),
                    p_sterilize=ss.bernoulli(1.0),
                ),
            ),
            pars=dict(coverage=ss.bernoulli(0.5)),
        )
    sim = build_sim(tb, interventions=ivs, n_agents=4000, stop='2035-12-31')
    sim.run()
    return sim.results.timevec, sim.results.tb['frac_resist']


fig, ax = plt.subplots(figsize=(7, 4))
for with_tpt, label in [(False, 'no TPT'), (True, 'INH TPT')]:
    t, fr = run_tpt(with_tpt)
    ax.plot(t, fr, label=label)
ax.set(title='INH TPT can raise the resistant share of active TB',
       xlabel='year', ylabel='resistant fraction')
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
```

Set `p_sterilize > 0` so the strain-aware clearance path runs. With `p_multi=1` (default), the main harm pathway is **transmission unmasking**, not progression bottlenecking.

---

## 11. Analyzing where resistance comes from

`ResistanceStats` decomposes new resistance into **de-novo**, **treatment-acquired**, and **transmitted** fluxes. `StrainResults` records per-strain active-TB counts.

```python
import starsim as ss
import tbsim

tb = tbsim.TBResistant(
    rel_fitness={'TX': 0.9},
    pars=dict(
        beta=ss.permonth(0.35),
        init_prev=ss.bernoulli(0.12),
        init_strains=[0.9, 0.1],
        p_rand={'TX': 0.01},
        rr_reinfection_inf=1.0,
        rr_reinfection_non=1.0,
    ),
)
tx = tbsim.TxDeliveryR(
    product=tbsim.TxR(
        strains=tb.strains,
        base_efficacy=0.8,
        resist_penalty={'TX': 0.2},
        q_acq={'TX': 0.05},
    ),
    rate_sym=ss.peryear(1.0),
)
stats = tbsim.ResistanceStats()
strains_az = tbsim.StrainResults()
sim = build_sim(tb, interventions=tx, analyzers=[stats, strains_az], stop='2045-12-31')
sim.run()

df = stats.to_df(sim)
origins = {
    'de-novo': int(df.flux_denovo.sum()),
    'treatment-acquired': int(df.flux_txacq.sum()),
    'transmitted': int(df.flux_transmitted.sum()),
}
print(origins)
print(df.tail())
```

Once resistance is established, **transmission** usually dominates cumulative events; de-novo and treatment acquisition seed and top up the pool.

---

## 12. Parameter reference (cheat sheet)

### 12.1 `TBResistant` / `Strains`

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `drugs` | list[str] | `['TX']` | Ordered drug names; `m = 2**len(drugs)` |
| `rel_fitness` | dict | `{}` | Per-drug `r_i ∈ [0,1]`; missing → 1.0 |
| `rr_reinfection_inf` | float | → `rr_reinfection_rec` | σ for INFECTION |
| `rr_reinfection_non` | float | → `rr_reinfection_inf` | σ for NON_INFECTIOUS |
| `rr_reinfection_asy` / `_sym` | float | `0` | Superinfection in active disease |
| `p_multi` | float | `1` | Prob. keep all strains at →ASY |
| `prog_select` | str | `'random'` | Or `'fitness'` under bottleneck |
| `rr_prog_super` / `rr_clear_super` | float | `1` | Optional multi-strain rate multipliers |
| `p_rand` | dict | off | De-novo `{drug: p}` |
| `prog_resist_mode` | str | `'mixed'` | Or `'replacement'` |
| `init_strains` | array | pan only | Seed mix over strain ids |

### 12.2 `TxR` / `TxDeliveryR`

| Parameter | Notes |
|-----------|--------|
| `base_efficacy` | Cure prob. for fully susceptible strain |
| `resist_penalty` | Per-drug multiplier on efficacy for resistance to **regimen** drugs |
| `adherence` | Per-agent Bernoulli; non-adherent clears no strains that course |
| `q_acq` | Per-drug acquisition-on-failure (replacement) |
| `acq_state_rr` | Scale `q_acq` by TB state at failure |
| `regimen_drugs` | Which drugs the regimen contains |
| `rate_asym` / `rate_sym` | Initiation rates |
| `eligibility` | Optional `sim → uids` override |
| `supersedes` | Names of deliveries to interrupt before starting |

### 12.3 `DST` / `TPTRx`

| Parameter | Notes |
|-----------|--------|
| `sens` / `spec` | Scalar or per-drug dict |
| `p_strain_obs` | `None` → use fitness; or scalar/dict |
| `matches(**drugs)` | Eligibility from observed profile |
| `regimen_drugs` (TPT) | Drugs that must all be susceptible for sterilization |
| `p_tpt_acq` | Acquisition among ineffective TPT outcomes |
| `p_sterilize` | Must be > 0 to exercise strain-aware clearance |

---

## 13. Known limitations

These are specified or desired but **not fully implemented** yet. Acceptance criteria live in [tbsim-resistance-uat.md](tbsim-resistance-uat.md).

| Topic | Current behavior |
|-------|------------------|
| Adherence distribution | Single Bernoulli per course, not a full per-agent distribution |
| DST indeterminate | Binary observed profile only |
| Failure vs new case | No durable “time since last treatment” classifier |
| Time-varying progression hazard | `ti_infected` resets on exposure; full hazard not modeled |
| LAI_TPT burden table | ODE / directional tests exist; per-100k LAI_TPT table not produced |
| Strain carriage counts | Identical strains blocked + counted; no count-based carriage mode |
| LTFU outcome | Not a separate treatment outcome |

---

## 14. Further reading

| Document | What it is |
|----------|------------|
| [model-tests.md](model-tests.md) | ODE reference questions of interest |
| [validate_resistance_abm_vs_ode.py](validate_resistance_abm_vs_ode.py) | ABM ↔ ODE validation script |
| [Resistance tutorial (Quarto)](../../../docs/tutorials/resistance_tutorial.qmd) | Executable notebook-style tutorial in the docs site |

**Automated checks** (run from the repository root; the ODE validation can take several minutes):

```bash
# From the tbsim repository root (needs: pip install -e ".[dev]"):
python -m pytest tbsim/resistance/devtests/ -q
```

```bash
# Optional longer ABM ↔ ODE validation (several minutes):
python tbsim/resistance/docs/validate_resistance_abm_vs_ode.py
```

**Public API entry points** (all re-exported on `tbsim`):  
`Strains`, `TBResistant`, `TxR`, `TxDeliveryR`, `treatment_monitoring_eligibility`, `DST`, `DSTDelivery`, `TPTRx`, `ResistanceStats`, `StrainResults`.
