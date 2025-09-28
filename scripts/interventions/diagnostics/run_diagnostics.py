import numpy as np
import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt


sim = ss.Sim(
    people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
    diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
    interventions=[
        mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),  # For new code with initial care-seeking rate
        mtb.TBDiagnostic(pars={
            'coverage': ss.bernoulli(0.8, strict=False),
            'sensitivity': 0.20,
            'specificity': 0.20,
            'care_seeking_multiplier': 2.0,
        }),
    ],
    networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
    pars=dict(start=ss.date(2000), stop=ss.date(2010), dt=ss.months(1)),  
)
sim.run()

tbdiag = sim.results['tbdiagnostic']
print(sim.results['tbdiagnostic'].keys())

# Plot incident diagnostic results
plt.figure(figsize=(10, 5))
plt.plot(tbdiag['n_tested'].timevec, tbdiag['n_tested'].values, label='Tested', marker='o')
plt.plot(tbdiag['n_test_positive'].timevec, tbdiag['n_test_positive'].values, label='Tested Positive', linestyle='--')
plt.plot(tbdiag['n_test_negative'].timevec, tbdiag['n_test_negative'].values, label='Tested Negative', linestyle=':')
plt.xlabel('Time')
plt.ylabel('People')
plt.title('TB Diagnostic Testing Outcomes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot cumulative diagnostic results
plt.figure(figsize=(10, 5))
plt.plot(tbdiag['cum_test_positive'].timevec, tbdiag['cum_test_positive'].values, label='Cumulative Positives', linestyle='--')
plt.plot(tbdiag['cum_test_negative'].timevec, tbdiag['cum_test_negative'].values, label='Cumulative Negatives', linestyle=':')
plt.xlabel('Time')
plt.ylabel('Cumulative Tests')
plt.title('Cumulative TB Diagnostic Results')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Pull people who were tested multiple times
n_retested = sim.people.n_times_tested
n_retested_int = np.array(n_retested, dtype=int)
retested_uids = np.where(n_retested_int > 1)[0]

# Plot histogram of repeat tests
plt.figure(figsize=(8, 4))
plt.hist(n_retested_int[retested_uids],
            bins=range(2, int(n_retested_int.max())+2),
            rwidth=0.6,
            align='left')
plt.xlabel("Number of times tested")
plt.ylabel("Number of people")
plt.title("Distribution of Repeat Testing (n_times_tested > 1)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional readouts
print("People with care-seeking multipliers > 1.0:", np.sum(sim.people.care_seeking_multiplier > 1.0))
print("Final mean care-seeking multiplier:", np.mean(sim.people.care_seeking_multiplier))

tb = sim.results['tb']
print("Average # of active TB cases:", np.mean(tb['n_active'].values))  # Confirm active TB prevalence

hsb = sim.results['healthseekingbehavior']
print("Max incident sought care:", np.max(hsb['new_sought_care'].values))
print("People who sought care:", np.sum(sim.people.sought_care))
print("People who were tested:", np.sum(sim.people.tested))

print(f"People who were retested: {len(retested_uids)}")
print(f"Max times tested: {n_retested.max()}")

def run_diagnostic_scenario(label, sensitivity, specificity):
    print(f"\nRunning scenario: {label}")
    
    sim = ss.Sim(
        people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
        diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
        interventions=[
            # mtb.HealthSeekingBehavior(pars={'prob': ss.bernoulli(p=0.25, strict=False)}),
            mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),  # For new code with initial care-seeking rate
            mtb.TBDiagnostic(pars={
                'coverage': ss.bernoulli(0.8, strict=False),
                'sensitivity': sensitivity,
                'specificity': specificity,
                'care_seeking_multiplier': 2.0,
            }),
        ],
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
        pars=dict(start=2000, stop=2010, dt=ss.days(1)/12),
    )
    sim.run()

    # Pull diagnostic results
    tbdiag = sim.results['tbdiagnostic']
    n_retested = np.array(sim.people.n_times_tested, dtype=int)
    retested_uids = np.where(n_retested > 1)[0]

    # Diagnostic plots
    plt.figure(figsize=(10, 5))
    plt.plot(tbdiag['n_tested'].timevec, tbdiag['n_tested'].values, label='Tested')
    plt.plot(tbdiag['n_test_positive'].timevec, tbdiag['n_test_positive'].values, label='Tested Positive', linestyle='--')
    plt.plot(tbdiag['n_test_negative'].timevec, tbdiag['n_test_negative'].values, label='Tested Negative', linestyle=':')
    plt.xlabel('Time')
    plt.ylabel('People')
    plt.title(f'TB Diagnostic Testing Outcomes – {label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Cumulative plot
    plt.figure(figsize=(10, 5))
    plt.plot(tbdiag['cum_test_positive'].timevec, tbdiag['cum_test_positive'].values, label='Cumulative Positives', linestyle='--')
    plt.plot(tbdiag['cum_test_negative'].timevec, tbdiag['cum_test_negative'].values, label='Cumulative Negatives', linestyle=':')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Tests')
    plt.title(f'Cumulative Diagnostic Results – {label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Histogram of repeat testing
    plt.figure(figsize=(8, 4))
    plt.hist(n_retested[retested_uids], bins=range(2, int(n_retested.max()) + 2), rwidth=0.6, align='left')
    plt.xlabel("Number of times tested")
    plt.ylabel("Number of people")
    plt.title(f'Repeat Testing (n_times_tested > 1) – {label}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Summary
    print(f"→ Scenario: {label}")
    print(f"People retested: {len(retested_uids)}")
    print(f"Max times tested: {n_retested.max()}")
    print(f"Final mean care-seeking multiplier: {np.mean(sim.people.care_seeking_multiplier)}")
    print(f"Total tested: {np.sum(sim.people.tested)}")
