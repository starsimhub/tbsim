import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

sim = ss.Sim(
    people=ss.People(n_agents=1000),
    diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
    networks=ss.RandomNet(),
    interventions=mtb.HealthSeekingBehavior(pars={
        'initial_care_seeking_rate': ss.perday(0.05),
        'single_use': False,
    }),
    pars=dict(start=ss.date(2000), stop=ss.date(2020), dt=ss.months(1)),
)
sim.run()

tb  = sim.results['tb']
hsb = sim.results['healthseekingbehavior']
t   = tb['n_active'].timevec

# TB states
fig, panels = plt.subplots(1, 2, figsize=(12, 4))

for key in ['n_active_smpos', 'n_active_smneg', 'n_active_exptb']:
    if key in tb:
        panels[0].plot(t, tb[key].values, label=key.replace('n_active_', ''))
panels[0].set_title('Active TB states')
panels[0].set_ylabel('Count')
panels[0].legend()

panels[1].plot(t, tb['n_infectious'].values, label='Infectious')
panels[1].set_title('Infectious')
panels[1].set_ylabel('Count')
panels[1].legend()

plt.tight_layout()
plt.show()

# Health-seeking
t_hsb      = hsb['n_sought_care'].timevec
n_sought   = hsb['n_sought_care'].values
new_sought = hsb['new_sought_care'].values
n_eligible = hsb['n_eligible'].values

fig, panels = plt.subplots(1, 2, figsize=(12, 4))

panels[0].plot(t_hsb, n_sought, label='Cumulative sought care')
panels[0].plot(t_hsb, new_sought, label='New per step', linestyle='--')
panels[0].set_title('Health-seeking')
panels[0].set_ylabel('People')
panels[0].legend()

panels[1].plot(t_hsb, n_eligible, label='Eligible (not yet sought)')
panels[1].set_title('Eligible for care-seeking')
panels[1].set_ylabel('People')
panels[1].legend()

plt.tight_layout()
plt.show()
