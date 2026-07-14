# What's new in TBsim resistance — a hands-on guide

This guide walks through the features added in the resistance update pass, with small runnable examples. It assumes you already know the basics of the multi-strain model (see [`tbsim-resistance-user-manual.md`](tbsim-resistance-user-manual.md)); here we focus only on **what changed** and **how to use it**.

The headline change is a **per-strain counter**: an agent can now be superinfected with *the same* strain more than once, and the model tracks how many copies of each strain each agent carries. Everything else in this guide builds on or sits beside that change. Design rationale for each item is in [`implementation-decisions.md`](implementation-decisions.md).

Every code block below is self-contained and runnable. A tiny helper builds the sims:

```python
import numpy as np
import starsim as ss
import tbsim

def demo_sim(tb, interventions=None, analyzers=None, n_agents=3000, stop='2020-12-31', lam=8, seed=0):
    """Build a small random-network sim around a TB(Resistant) module."""
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=lam), dur=0))
    return ss.Sim(n_agents=n_agents, networks=net, diseases=tb, interventions=interventions,
                  analyzers=analyzers, dt=ss.days(30), start=ss.date('2000-01-01'),
                  stop=ss.date(stop), rand_seed=seed, verbose=0)
```

---

## 1. Identical-strain superinfection and the per-strain counter

Previously, re-exposure to a strain an agent already carried was *blocked* (a no-op). Now it is **allowed**: the agent's **count** for that strain increments. The count starts at 1 for a new infection and only grows through repeated transmission (or seeding) of an already-carried strain.

The counter lives in `tb.strain_counts` — a list of one integer array per strain id (`strain_counts[j][uid]` = how many copies of strain `j` agent `uid` carries). The invariant is simple: **count > 0 exactly when the agent carries that strain**.

```python
tb = tbsim.TBResistant(
    rel_fitness={'TX': 1.0},
    pars=dict(beta=ss.permonth(0.35), init_prev=ss.bernoulli(0.15), init_strains=[0.5, 0.5],
              rr_reinfection_inf=1.0, rr_reinfection_non=1.0),  # allow superinfection
)
sim = demo_sim(tb); sim.run()
tb = tbsim.get_tb(sim, which=tbsim.TBResistant)

# How many identical-strain superinfection events occurred (count increments)?
print('identical-strain superinfections:', int(np.sum(sim.results.tb['new_identical_superinf'])))

# Peek at the multiplicity of strain 0 (pan-susceptible) across agents that carry it.
carriers = (tb.strain_counts[0].values > 0)
print('max copies of strain 0 in any agent:', int(tb.strain_counts[0].values.max()))
print('mean copies among carriers:', float(tb.strain_counts[0].values[carriers].mean()))
```

The counter has a deliberately **narrow scope**. It affects only two things: (a) the transmission multinomial and (b) the progression bottleneck (below). It does **not** change transition rates, DST, treatment efficacy, or resistance-acquisition probabilities — for those, identical strains are always *coupled* (all copies behave as one).

### Count-weighted transmission

When a superinfected agent transmits, the probability it passes a given strain is proportional to `count × fitness`. You can see this directly on the strain registry:

```python
strains = tbsim.Strains(['TX'], rel_fitness=None)  # neutral fitness
mask = np.array([0b11])                            # carries strain 0 and strain 1

print('counts 1,1 ->', strains.transmit_probs(mask, counts=np.array([[1, 1]]))[0])  # [0.5, 0.5]
print('counts 2,1 ->', strains.transmit_probs(mask, counts=np.array([[2, 1]]))[0])  # [0.667, 0.333]

# The overall infectiousness (max carried fitness) does NOT depend on count:
print('rel_trans (max fitness):', strains.max_fitness(mask)[0])  # 1.0 either way
```

---

## 2. Count-weighted progression bottleneck

When `p_multi < 1`, a multi-strain agent progressing to active disease may retain only **one** strain. Which strain survives is now weighted by `count` (and by fitness if `prog_select='fitness'`). An agent carrying 2 copies of the susceptible strain and 1 of the resistant strain has a 2/3 vs 1/3 chance of the survivor being susceptible vs resistant.

```python
tb = tbsim.TBResistant(rel_fitness={'TX': 1.0},
                       pars=dict(init_prev=ss.bernoulli(0.0), p_multi=0.0, prog_select='random'))
sim = demo_sim(tb, n_agents=9000, stop='2000-06-30'); sim.init()
tb = tbsim.get_tb(sim, which=tbsim.TBResistant)

u = ss.uids(np.arange(9000))
tb.strain_mask[u] = 0b11        # carries strain 0 and strain 1
tb.strain_counts[0][u] = 2      # ...2 copies of strain 0
tb.strain_counts[1][u] = 1      # ...1 copy of strain 1
tb._bottleneck(u)               # p_multi=0 → each agent keeps exactly one strain

print('survivor = strain 0:', float(np.mean(tb.strain_mask[u] == 1)))  # ~0.667
print('survivor = strain 1:', float(np.mean(tb.strain_mask[u] == 2)))  # ~0.333
```

> **Note.** Because the more-transmitted strain accumulates higher multiplicity, count-weighting *amplifies* competitive exclusion relative to the count-free reference ODE. A less-fit strain is driven out faster. This is expected, spec-mandated behavior.

---

## 3. DST results can expire, and treatment can have a refractory period (L1)

A stored DST result used to live forever, so a patient who failed treatment and stayed bacteriologically positive could be re-treated off the *same* stale test every step. Two independent, opt-in levers fix this (both default off, preserving old behavior):

- `DSTDelivery(result_validity=<ss.dur>)` — wipe a stored result older than the window so the agent must be re-tested.
- `TxDeliveryR(retreat_after=<ss.dur>)` — a refractory period after a course ends before the same agent can be re-treated.
- `matches(..., max_age=<ss.dur>)` / `observed_resistant(drug, max_age=...)` — require the DST result be *fresh* at the eligibility site.

```python
tb = tbsim.TBResistant(drugs=['RIF'],
                       pars=dict(init_prev=ss.bernoulli(0.05), init_prev_active=ss.bernoulli(0.05)))
dst = tbsim.DSTDelivery(
    name='dst',
    product=tbsim.DST(strains=tb.strains, sens=0.95, spec=0.98),
    result_validity=ss.months(12),                      # results go stale after a year → re-test
    eligibility=lambda sim: tbsim.get_tb(sim, which=tbsim.TBResistant).active_tb.uids,
)
tx = tbsim.TxDeliveryR(
    name='tx',
    eligibility=dst.matches(RIF=True, max_age=ss.months(12)),  # only act on a fresh RIF-resistant result
    retreat_after=ss.months(6),                         # don't re-treat within 6 months of a course
    product=tbsim.TxR(strains=tb.strains, base_efficacy=0.5),
)
sim = demo_sim(tb, interventions=[dst, tx]); sim.run()
print('DST tests:', int(sim.results['dst'].n_tested.sum()))
print('courses started:', int(sim.results['tx'].n_treated.sum()))
```

The recommended clinical setup is "a DST result is valid for N months" (`result_validity`), optionally combined with `retreat_after` as a hard refractory guard.

---

## 4. TPT-acquired resistance is now a visible origin channel (L2)

TPT drug pressure can *select* for resistance (a susceptible strain mutates under an ineffective TPT course). This resistance used to be created but not counted in the resistance-origin decomposition. `ResistanceStats` now has a fourth flux channel, `flux_tptacq`, alongside de-novo, treatment-acquired, and transmitted.

```python
tb = tbsim.TBResistant(drugs=['INH'], rel_fitness={'INH': 0.9},
                       pars=dict(beta=ss.permonth(0.3), init_prev=ss.bernoulli(0.2),
                                 init_strains=[1.0, 0.0], rr_reinfection_inf=0.0, rr_reinfection_non=0.0))
tpt = tbsim.TPTSimple(
    product=tbsim.TPTRx(strains=tb.strains, regimen_drugs=['INH'], p_tpt_acq={'INH': 1.0},
                        pars=dict(efficacy=ss.bernoulli(0.6), p_sterilize=ss.bernoulli(0.0))),
    pars=dict(coverage=ss.bernoulli(0.5)),
)
sim = demo_sim(tb, interventions=tpt, analyzers=tbsim.ResistanceStats()); sim.run()

df = sim.analyzers[0].to_df(sim)
print(df[['flux_denovo', 'flux_txacq', 'flux_tptacq', 'flux_transmitted']].sum())
# The *root* origin here can only be TPT → flux_tptacq > 0 while flux_denovo == flux_txacq == 0.
# (flux_transmitted is nonzero too: once TPT creates a resistant strain, it then spreads by
# transmission — that downstream spread is correctly attributed to the transmitted channel.)
```

---

## 5. Latent treatment: match base TBsim, or run a full course (L3)

If a custom (e.g. DST-routed) eligibility selects a **latent** (`INFECTION`) agent for treatment, what should happen? By default (`treat_latent=False`), latent agents are cleared immediately — exactly like single-strain `tbsim.TxDelivery` — so they never enter a `TREATMENT` course and cannot acquire resistance. Set `treat_latent=True` to route them through a full course instead.

```python
def run_latent(treat_latent):
    tb = tbsim.TBResistant(pars=dict(init_prev=ss.bernoulli(0.3), beta=ss.permonth(0.0),
                                     init_strains=[1.0, 0.0]))
    tx = tbsim.TxDeliveryR(
        name='tx', treat_latent=treat_latent, dur_treatment=ss.months(3),
        eligibility=lambda sim: tbsim.get_tb(sim, which=tbsim.TBResistant).latent.uids,
        product=tbsim.TxR(strains=tb.strains, base_efficacy=0.0, q_acq={'TX': 1.0},
                          acq_state_rr={int(tbsim.TBS.INFECTION): 1.0}),
    )
    sim = demo_sim(tb, interventions=tx, stop='2003-12-31'); sim.run()
    return sim.results['tx']

default, coursed = run_latent(False), run_latent(True)
print('default  → courses:', int(default.n_treated.sum()), ' acquired:', int(default.n_acquired.sum()))
print('coursed  → courses:', int(coursed.n_treated.sum()), ' acquired:', int(coursed.n_acquired.sum()))
# default: 0 courses, 0 acquired (latent cleared outright); coursed: >0 courses, >0 acquired.
```

The default rate-based eligibility never selects latent agents, so this only matters for custom or DST-routed eligibilities.

---

## 6. Which strain acquires resistance on failure (L4)

When a treatment (or TPT) course fails and selects for resistance, the model picks **one** carried strain that is susceptible to the hit drug and mutates it. Previously it always chose the lowest-id strain (a pan-leaning bias). Now it chooses **at random** among the carried susceptible strains (or `∝ fitness` with `acq_select='fitness'`), so acquisition can land on a strain that already carries other resistances.

```python
tb = tbsim.TBResistant(drugs=['RIF', 'FQ'], pars=dict(init_prev=ss.bernoulli(0.0)))
prod = tbsim.TxR(strains=tb.strains, base_efficacy=0.0, q_acq={'FQ': 1.0}, acq_select='random')
sim = demo_sim(tb, interventions=tbsim.TxDeliveryR(product=prod), n_agents=8000, stop='2001-12-31')
sim.init()
tb = tbsim.get_tb(sim, which=tbsim.TBResistant)
prod = sim.interventions[0].product

u = ss.uids(np.arange(8000))
tb.strain_mask[u] = (1 << 0) | (1 << 1)     # carries pan (id0) and RIF-resistant (id1); both FQ-susceptible
surv = prod.acquire(u, tb.strain_mask[u].copy(), states=np.full(len(u), int(tbsim.TBS.SYMPTOMATIC)))
print('acquired FQ on the pan strain  → carries strain 2 ({FQ}):   ', ((surv >> 2) & 1).mean())
print('acquired FQ on the RIF strain  → carries strain 3 ({RIF,FQ}):', ((surv >> 3) & 1).mean())
# Both ~0.5 with random selection; the old lowest-id rule produced strain 2 every time.
```

---

## 7. Composable treatment-monitoring eligibility (L5)

Treatment monitoring / regimen switching can now be made contingent on more than elapsed time. Three helpers make eligibility callables composable:

- `eligibility_all(*callables)` / `eligibility_any(*callables)` — intersect / union of `sim -> uids` callables.
- `treatment_monitoring_eligibility(..., require=<callable>)` — sugar that ANDs another callable into the time-based monitor.
- `will_fail(tx_name)` — an oracle selecting on-treatment agents whose pre-rolled course outcome is a failure (handy for building scenarios).

```python
tb = tbsim.TBResistant(drugs=['RIF', 'BDQ'], rel_fitness={'RIF': 0.9},
                       pars=dict(beta=ss.permonth(0.3), init_prev=ss.bernoulli(0.1),
                                 init_strains=[0.7, 0.3, 0.0, 0.0], rr_reinfection_inf=1.0, rr_reinfection_non=1.0))
dst = tbsim.DSTDelivery(name='dst', product=tbsim.DST(strains=tb.strains, sens=0.95, spec=0.98),
                        eligibility=lambda sim: tbsim.get_tb(sim, which=tbsim.TBResistant).active_tb.uids)
first = tbsim.TxDeliveryR(name='first', rate_sym=ss.peryear(2.0),
                          product=tbsim.TxR(strains=tb.strains, base_efficacy=0.8, regimen_drugs=['RIF'],
                                            resist_penalty={'RIF': 0.1}))

# Switch to a BDQ regimen only for agents who are BOTH ≥4 steps into first-line AND observed RIF-resistant.
switch_when = tbsim.treatment_monitoring_eligibility(
    'first', after_steps=4, require=dst.matches(RIF=True, exclude_on_treatment=False))
second = tbsim.TxDeliveryR(name='second', eligibility=switch_when, supersedes=['first'],
                           product=tbsim.TxR(strains=tb.strains, base_efficacy=0.8, regimen_drugs=['BDQ']))

sim = demo_sim(tb, interventions=[dst, first, second], stop='2030-12-31'); sim.run()
print('first-line courses: ', int(sim.results['first'].n_treated.sum()))
print('switched to BDQ:    ', int(sim.results['second'].n_treated.sum()))
```

---

## 8. Agnostic (single-strain) convenience mode (L7)

For strain-aware-vs-agnostic comparisons, `TBResistant.agnostic()` returns a module preconfigured to behave like single-strain `tbsim.TB`: one drug, all infections pan-susceptible, no de-novo resistance, no superinfection. No resistant strain ever arises.

```python
pars = dict(beta=ss.permonth(0.2), init_prev=ss.bernoulli(0.05))

sim_tb = demo_sim(tbsim.TB(name='tb', pars=pars), n_agents=5000, stop='2035-12-31', lam=5, seed=1); sim_tb.run()
sim_ag = demo_sim(tbsim.TBResistant.agnostic(pars=pars), n_agents=5000, stop='2035-12-31', lam=5, seed=1); sim_ag.run()

print('resistance ever (agnostic):', float(np.max(sim_ag.results.tb['frac_resist'])))  # 0.0
print('endemic prevalence  TB:', round(float(np.mean(sim_tb.results.tb['prevalence_active'][-24:])), 4))
print('endemic prevalence  agnostic:', round(float(np.mean(sim_ag.results.tb['prevalence_active'][-24:])), 4))
# The two track each other within stochastic noise.
```

This is a convenience recipe, not a true single-strain space (the bitmask needs `m = 2**n ≥ 2`).

---

## 9. Drug-name validation (L8)

Mistyped drug names used to be silently ignored (they resolved through `.get()` and did nothing). Now `TxR`, `TPTRx`, and `DST` validate every drug name against the strain registry and raise a clear error.

```python
tb = tbsim.TBResistant(drugs=['RIF'], pars=dict(init_prev=ss.bernoulli(0.0)))
try:
    tbsim.TxR(strains=tb.strains, regimen_drugs=['RIFF'])   # typo!
except ValueError as e:
    print('caught:', e)

# You can also validate names yourself:
tb.strains.validate_drugs(['RIF'], where='my check')       # OK, no error
```

---

## Deferred (not in this update)

Two spec items are intentionally left for a later pass because they need modeling-team input (see `implementation-decisions.md` D-DEFER):

- **TPT `resist_penalty`** for partial (per-drug) TPT efficacy — TPT sterilization stays all-or-nothing for now.
- **Time-varying progression hazard** — progression remains constant-hazard. The clock-reset hook (`ti_infected` reset on every exposure, including same-strain re-exposure) is already in place for when this lands.
