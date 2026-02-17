"""
Run health-seeking with the legacy TB model (TBS states).
Uses dt=1 month and a moderate care-seeking rate so care-seeking is spread over time
(not all in one step). With dt=6 months, a per-day rate becomes ~1 per step, so
everyone eligible seeks care in the first step and the rest of the run is flat.
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

sim = ss.Sim(
    people=ss.People(n_agents=1000),
    diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
    interventions=[
        mtb.HealthSeekingBehavior(pars={
            'initial_care_seeking_rate': ss.perday(0.05),
            'single_use': False,
        }),
    ],
    networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
    pars=dict(start=ss.date(2000), stop=ss.date(2020), dt=ss.months(1)),
)
sim.run()

# TB results (legacy model: key is 'tb')
tb = sim.results['tb']

def _get_vals(r, key):
    v = r[key]
    return v.values if hasattr(v, 'values') else np.asarray(v)[:]

t = tb['n_active'].timevec

# --- Figure 1: TB states (active disease) ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Active states (care-seeking eligible: smpos, smneg, exptb; presymp for context)
ax = axs[0, 0]
for key in ['n_active_presymp', 'n_active_smpos', 'n_active_smneg', 'n_active_exptb']:
    if key in tb:
        ax.plot(t, _get_vals(tb, key), label=key.replace('n_active_', '').replace('_', ' '), marker='o', markersize=3)
ax.set_title('Active TB states (care-seeking eligible: smpos, smneg, exptb)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# Combined active and prevalence
ax = axs[0, 1]
ax.plot(t, _get_vals(tb, 'n_active'), label='Active (combined)', color='C0')
ax2 = ax.twinx()
ax2.plot(t, _get_vals(tb, 'prevalence_active'), label='Prevalence (active)', color='C1', linestyle='--')
ax2.set_ylabel('Prevalence')
ax2.legend(loc='upper right')
ax.set_title('Active TB and prevalence')
ax.set_ylabel('Active count')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Infectious and incidence
ax = axs[1, 0]
ax.plot(t, _get_vals(tb, 'n_infectious'), label='Infectious', color='C0')
ax.set_title('Infectious')
ax.set_xlabel('Time')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axs[1, 1]
ax.plot(t, _get_vals(tb, 'incidence_kpy'), label='Incidence (per 1000 py)', color='C2')
ax.set_title('Incidence')
ax.set_xlabel('Time')
ax.set_ylabel('Incidence per 1000 person-years')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('TB (legacy): disease states and outcomes', fontsize=12)
plt.tight_layout()
plt.show()

# --- Figure 2: Health-seeking ---
hsb = sim.results['healthseekingbehavior']
timevec_h = hsb['n_sought_care'].timevec
n_sought = np.asarray(hsb['n_sought_care'].values)
new_sought = np.asarray(hsb['new_sought_care'].values)
n_eligible = np.asarray(hsb['n_eligible'].values)

fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))

axs2[0].plot(timevec_h, new_sought, label='New sought care (per step)', marker='o', markersize=3)
axs2[0].plot(timevec_h, n_sought, label='Cumulative sought care', linestyle='--')
axs2[0].set_xlabel('Time')
axs2[0].set_ylabel('People (count)')
axs2[0].set_title('Health-seeking over time')
axs2[0].legend()
axs2[0].grid(True, alpha=0.3)
axs2[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

axs2[1].plot(timevec_h, n_eligible, label='Eligible (active, not yet sought)', color='C1', marker='s', markersize=3)
axs2[1].set_xlabel('Time')
axs2[1].set_ylabel('People (count)')
axs2[1].set_title('Eligible for care-seeking')
axs2[1].legend()
axs2[1].grid(True, alpha=0.3)
axs2[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))

fig2.suptitle('TB + health-seeking', fontsize=12)
plt.tight_layout()
plt.show()
