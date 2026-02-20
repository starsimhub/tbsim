import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

sim = ss.Sim(
    people=ss.People(n_agents=1000),
    diseases=mtb.TB_LSHTM(pars={'init_prev': ss.bernoulli(0.05)}),
    networks=ss.RandomNet(),
    interventions=mtb.HealthSeekingBehavior(pars={
        'initial_care_seeking_rate': ss.perday(0.1),
        'single_use': False,
    }),
    pars=dict(start=ss.date(2000), stop=ss.date(2020), dt=ss.months(6)),
)
sim.run()

tb  = sim.diseases[0].results
hsb = sim.results.healthseekingbehavior
t   = tb['timevec']

# TB states
fig, panel = plt.subplots(1, 2, figsize=(12, 4))

for key in ['n_UNCONFIRMED', 'n_ASYMPTOMATIC', 'n_SYMPTOMATIC']:
    if key in tb:
        panel[0].plot(t, tb[key].values, label=key.replace('n_', ''))
panel[0].set_title('Active TB states')
panel[0].set_ylabel('Count')
panel[0].legend()

if 'n_infectious' in tb:
    panel[1].plot(t, tb['n_infectious'].values, label='Infectious')
panel[1].set_title('Infectious')
panel[1].set_ylabel('Count')
panel[1].legend()

plt.tight_layout()
plt.show()

# Health-seeking
t_hsb     = hsb['n_sought_care'].timevec
n_sought  = hsb['n_sought_care'].values
n_eligible = hsb['n_eligible'].values

fig, panel = plt.subplots(1, 2, figsize=(12, 4))

panel[0].plot(t_hsb, n_sought, label='Cumulative sought care')
panel[0].plot(t_hsb, np.diff(n_sought, prepend=0), label='New per step', linestyle='--')
panel[0].set_title('Health-seeking')
panel[0].set_ylabel('People')
panel[0].legend()

panel[1].plot(t_hsb, n_eligible, label='Eligible (not yet sought)')
panel[1].set_title('Eligible for care-seeking')
panel[1].set_ylabel('People')
panel[1].legend()

plt.tight_layout()
plt.show()
