import numpy as np
import starsim as ss
from tbsim import TBS

__all__ = ['TBDiagnostic']


class TBDiagnostic(ss.Intervention):
    """
    TB diagnostic triggered by health-seeking behavior.

    Parameters:
        coverage (float or Dist): Fraction of those who sought care who get tested.
        sensitivity (float): Probability test is positive if truly has TB.
        specificity (float): Probability test is negative if no TB.
        reset_flag (bool): Whether to reset `sought_care` flag after testing (optional).
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            coverage=1.0,
            sensitivity=0.9,
            specificity=0.95,
            reset_flag=False,
            care_seeking_multiplier=1.0,
        )
        self.update_pars(pars=pars, **kwargs)

        # Temporary state for update_results
        self.tested_this_step = []
        self.test_result_this_step = []


    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Find people who sought care but haven't been tested
        # eligible = ppl.sought_care & (~ppl.tested) & ppl.alive
        eligible = ppl.sought_care & (~ppl.diagnosed) & ppl.alive  # Avoids excluding once-tested people
        uids = eligible.uids
        if len(uids) == 0:
            return

        # # Confirm uids is non-empty
        # print(f"[t={self.sim.ti}] Eligible for testing: {len(uids)}")

        # Apply coverage filter
        if isinstance(self.pars.coverage, ss.Dist):
            selected = self.pars.coverage.filter(uids)
        else:
            selected = ss.bernoulli(self.pars.coverage).filter(uids)
        if len(selected) == 0:
            return

        # # Debug: Log type and value of coverage
        # print(f"[t={self.sim.ti}] Coverage type: {type(self.pars.coverage)}")
        # if hasattr(self.pars.coverage, 'p'):
        #     print(f"[t={self.sim.ti}] Coverage probability: {self.pars.coverage.p}")
        # print(f"[t={self.sim.ti}] Selected for testing: {len(selected)} / {len(uids)} eligible")

        # Determine TB status
        tb_states = tb.state[selected]
        has_tb = np.isin(tb_states, [TBS.ACTIVE_SMPOS,
                                     TBS.ACTIVE_SMNEG,
                                     TBS.ACTIVE_EXPTB])

        # Apply test logic
        rand = np.random.rand(len(selected))
        # test_positive = np.where(has_tb, rand < self.pars.sensitivity,
        #                          rand > (1 - self.pars.specificity))
        test_positive = ((has_tb & (rand < self.pars.sensitivity)) |
                         (~has_tb & (rand < self.pars.specificity)))

        # Update person state
        ppl.tested[selected] = True
        ppl.n_times_tested[selected] += 1
        ppl.test_result[selected] = test_positive
        ppl.diagnosed[selected[test_positive]] = True

        # Optional: reset the health-seeking flag
        if self.pars.reset_flag:
            ppl.sought_care[selected] = False

        # Handle false negatives: schedule another round of health-seeking
        false_negative_uids = selected[~test_positive & has_tb]

        if len(false_negative_uids):
            # print(f"[t={self.sim.ti}] {len(false_negative_uids)} false negatives → retry scheduled")
            pass

        # # Enable retry: reset care flag and allow re-test
        # ppl.sought_care[false_negative_uids] = False
        # ppl.tested[false_negative_uids] = False
        # mult = self.pars.care_seeking_multiplier
        # ppl.care_seeking_multiplier[false_negative_uids] *= mult

        # Filter only those who haven't had multiplier applied yet
        unboosted = false_negative_uids[~ppl.multiplier_applied[false_negative_uids]]

        # Apply multiplier only to them
        if len(unboosted):
            ppl.care_seeking_multiplier[unboosted] *= self.pars.care_seeking_multiplier
            ppl.multiplier_applied[unboosted] = True  # ✅ mark as boosted

        if len(unboosted):
            # print(f"[t={self.sim.ti}] Multiplier applied to {len(unboosted)} people")
            pass

        # Reset flags to allow re-care-seeking
        ppl.sought_care[false_negative_uids] = False
        ppl.tested[false_negative_uids] = False

        # Store for update_results
        self.tested_this_step = selected
        self.test_result_this_step = test_positive

    # def init_results(self):
    #     self.define_results(
    #         ss.Result('n_tested', dtype=int),
    #         ss.Result('n_test_positive', dtype=int),
    #         ss.Result('n_test_negative', dtype=int),
    #     )

    def init_results(self):
        self.define_results(
            ss.Result('n_tested', dtype=int),
            ss.Result('n_test_positive', dtype=int),
            ss.Result('n_test_negative', dtype=int),
            ss.Result('cum_test_positive', dtype=int),
            ss.Result('cum_test_negative', dtype=int),
        )

    # def update_results(self):
    #     self.results['n_tested'][self.ti] = len(self.tested_this_step)
    #     self.results['n_test_positive'][self.ti] = np.count_nonzero(self.test_result_this_step)
    #     self.results['n_test_negative'][self.ti] = len(self.tested_this_step) - np.count_nonzero(self.test_result_this_step)

    #     # Reset temporary storage
    #     self.tested_this_step = []
    #     self.test_result_this_step = []

    def update_results(self):
        # Per-step counts
        n_tested = len(self.tested_this_step)
        n_pos = np.count_nonzero(self.test_result_this_step)
        n_neg = n_tested - n_pos

        self.results['n_tested'][self.ti] = n_tested
        self.results['n_test_positive'][self.ti] = n_pos
        self.results['n_test_negative'][self.ti] = n_neg

        # Cumulative totals (add to previous step)
        if self.ti > 0:
            self.results['cum_test_positive'][self.ti] = self.results['cum_test_positive'][self.ti-1] + n_pos
            self.results['cum_test_negative'][self.ti] = self.results['cum_test_negative'][self.ti-1] + n_neg
        else:
            self.results['cum_test_positive'][self.ti] = n_pos
            self.results['cum_test_negative'][self.ti] = n_neg

        # Reset temporary storage
        self.tested_this_step = []
        self.test_result_this_step = []


# Sample calling function below
if __name__ == '__main__':

    import tbsim as mtb
    import starsim as ss
    import matplotlib.pyplot as plt

    sim = ss.Sim(
        people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
        diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
        interventions=[
            # mtb.HealthSeekingBehavior(pars={'prob': ss.bernoulli(p=0.25, strict=False)}),  # For old code with probability
            mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),  # For new code with initial care-seeking rate
            # mtb.TBDiagnostic(pars={
            #     'coverage': ss.bernoulli(0.8, strict=False)
            # })
            # mtb.TBDiagnostic(pars={
            #     'coverage': ss.bernoulli(0.8, strict=False),
            #     'care_seeking_multiplier': 2.0  # Encourages faster retries
            # }),
            mtb.TBDiagnostic(pars={
                'coverage': ss.bernoulli(0.8, strict=False),
                'sensitivity': 0.20,
                'specificity': 0.20,
                'care_seeking_multiplier': 2.0,
            }),
        ],
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
        pars=dict(start=2000, stop=2010, dt=1/12),  # dt=1/12 for a monthly timestep
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
    n_retested_int = n_retested.astype(int)
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
            pars=dict(start=2000, stop=2010, dt=1/12),
        )
        sim.run()

        # Pull diagnostic results
        tbdiag = sim.results['tbdiagnostic']
        n_retested = sim.people.n_times_tested.astype(int)
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
