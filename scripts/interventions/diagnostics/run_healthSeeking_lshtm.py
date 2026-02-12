"""
Run health-seeking with the LSHTM TB model (TB_LSHTM).
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

# Higher init_prev and single_use=False so we see ongoing dynamics and care-seeking
disease = mtb.TB_LSHTM(pars={'init_prev': ss.bernoulli(0.05)})

sim = ss.Sim(
    people=ss.People(n_agents=1000),
    diseases=disease,
    interventions=[
        mtb.HealthSeekingBehavior(pars={
            'initial_care_seeking_rate': ss.perday(0.1),
            'single_use': False,
        }),
    ],
    networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
    pars=dict(start=ss.date(2000), stop=ss.date(2020), dt=ss.months(6)),
)
sim.run()

# TB (LSHTM) results from the disease module
tb_res = sim.diseases[0].results
t = np.asarray(tb_res['timevec']) if 'timevec' in tb_res else np.arange(len(tb_res['n_infectious']))

def _get_vals(r, key):
    v = r[key]
    return v.values if hasattr(v, 'values') else np.asarray(v)[:]

# Health-seeking results
hsb_key = 'healthseekingbehavior' if 'healthseekingbehavior' in sim.results else None
if hsb_key is None and hasattr(sim.results, 'keys'):
    for k in sim.results.keys():
        if hasattr(sim.results[k], '__contains__') and 'n_sought_care' in sim.results[k]:
            hsb_key = k
            break
if hsb_key is None:
    raise KeyError("Health-seeking results not found.")
hsb = sim.results[hsb_key]

# --- Figure 1: LSHTM TB states (active disease and treatment) ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Active states (eligible for care-seeking): UNCONFIRMED, ASYMPTOMATIC, SYMPTOMATIC
ax = axs[0, 0]
for key in ['n_UNCONFIRMED', 'n_ASYMPTOMATIC', 'n_SYMPTOMATIC']:
    if key in tb_res:
        ax.plot(t, _get_vals(tb_res, key), label=key.replace('n_', ''), marker='o', markersize=3)
ax.set_title('Active TB states (care-seeking eligible)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# Infectious and prevalence
ax = axs[0, 1]
if 'n_infectious' in tb_res:
    ax.plot(t, _get_vals(tb_res, 'n_infectious'), label='Infectious', color='C0')
if 'prevalence_active' in tb_res:
    ax2 = ax.twinx()
    ax2.plot(t, _get_vals(tb_res, 'prevalence_active'), label='Prevalence (active)', color='C1', linestyle='--')
    ax2.set_ylabel('Prevalence')
    ax2.legend(loc='upper right')
ax.set_title('Infectious and prevalence')
ax.set_ylabel('Infectious count')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Treatment states
ax = axs[1, 0]
for key in ['n_TREATMENT', 'n_TREATED']:
    if key in tb_res:
        ax.plot(t, _get_vals(tb_res, key), label=key.replace('n_', ''), marker='o', markersize=3)
ax.set_title('Treatment states')
ax.set_xlabel('Time')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# Incidence
ax = axs[1, 1]
if 'incidence_kpy' in tb_res:
    ax.plot(t, _get_vals(tb_res, 'incidence_kpy'), label='Incidence (per 1000 py)', color='C2')
ax.set_title('Incidence')
ax.set_xlabel('Time')
ax.set_ylabel('Incidence per 1000 person-years')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('TB_LSHTM: disease states and outcomes', fontsize=12)
plt.tight_layout()
plt.show()

# --- Figure 2: Health-seeking ---
fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))

timevec_h = hsb['n_sought_care'].timevec
n_sought = hsb['n_sought_care'].values
new_sought = np.diff(n_sought, prepend=0)
n_eligible = hsb['n_eligible'].values

axs2[0].plot(timevec_h, new_sought, label='New sought care (per step)', marker='o', markersize=4)
axs2[0].plot(timevec_h, n_sought, label='Cumulative sought care', linestyle='--')
axs2[0].set_xlabel('Time')
axs2[0].set_ylabel('People')
axs2[0].set_title('Health-seeking over time')
axs2[0].legend()
axs2[0].grid(True, alpha=0.3)

axs2[1].plot(timevec_h, n_eligible, label='Eligible (active, not yet sought)', color='C1', marker='s', markersize=3)
axs2[1].set_xlabel('Time')
axs2[1].set_ylabel('People')
axs2[1].set_title('Eligible for care-seeking')
axs2[1].legend()
axs2[1].grid(True, alpha=0.3)

fig2.suptitle('TB_LSHTM + health-seeking', fontsize=12)
plt.tight_layout()
plt.show()
