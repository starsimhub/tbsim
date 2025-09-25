
import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import sciris as sc
import numpy as np

import os, sys
scripts_dir = os.path.join(os.getcwd(), '..', '..', 'scripts')
sys.path.append(scripts_dir)
import tbsim.utils.plots as pl

# Create and run the sim
sim = ss.Sim(
    people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
    diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
    interventions=[
        # mtb.HealthSeekingBehavior(pars={'prob': ss.bernoulli(p=0.1)})  # For old code with probability
        mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.1)})  # For new code with initial care-seeking rate
    ],
    networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
    pars=dict(start=ss.date(2000), stop=ss.date(2020), dt=ss.months(6)),
)
sim.run()

# Flatten results into format expected by plot_results()
flat_results = {'TB + HSB': sim.results.flatten()}

# Plot all matching metrics (you can adjust keywords below)
pl.plot_results(
    flat_results,
    keywords=['active', 'sought', 'eligible', 'incidence'],
    exclude=(),
    n_cols=2,
    cmap='viridis',
    dark=False,
    heightfold=2,
    style='default',
    title='TB + Health-Seeking Behavior'
)

# Custom plot for new_sought_care + optional cumulative view
hsb = sim.results['healthseekingbehavior']
timevec = hsb['new_sought_care'].timevec
new_sought = hsb['new_sought_care'].values
cum_sought = np.cumsum(new_sought)

plt.figure(figsize=(10, 5))
plt.plot(timevec, new_sought, label='New sought care (this step)', linestyle='-', marker='o')
plt.plot(timevec, cum_sought, label='Cumulative sought care', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Number of People')
plt.title('New and Cumulative Health-Seeking Behavior Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
