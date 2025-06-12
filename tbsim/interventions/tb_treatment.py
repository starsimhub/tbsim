import numpy as np
import starsim as ss
from tbsim import TBS
import matplotlib.ticker as mticker

__all__ = ['TBTreatment']


class TBTreatment(ss.Intervention):
    """
    Starts TB treatment for diagnosed individuals and applies treatment success/failure logic.

    Parameters:
        treatment_success_rate (float or Dist): Probability of cure if treated.
        reseek_multiplier (float): Care-seeking multiplier applied after failure.
        reset_flags (bool): Whether to reset tested/diagnosed flags after failure.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            treatment_success_rate=0.85,
            reseek_multiplier=2.0,
            reset_flags=True,
        )
        self.update_pars(pars=pars, **kwargs)

        # Storage for results
        self.new_treated = []
        self.successes = []
        self.failures = []

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Select individuals diagnosed with TB and alive
        diagnosed = ppl.diagnosed & ppl.alive
        active_tb = np.isin(tb.state, [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
        uids = (diagnosed & active_tb).uids

        if len(uids) == 0:
            return

        # Start treatment
        started = tb.start_treatment(uids)

        # Treatment outcomes
        tx_uids = uids[tb.on_treatment[uids]]
        ppl.n_times_treated[tx_uids] += 1
        rand = np.random.rand(len(tx_uids))
        success_uids = tx_uids[rand < self.pars.treatment_success_rate]
        failure_uids = tx_uids[rand >= self.pars.treatment_success_rate]

        # Update success: instant clearance via TB logic
        tb.state[success_uids] = TBS.NONE
        tb.on_treatment[success_uids] = False
        tb.susceptible[success_uids] = True
        tb.infected[success_uids] = False
        tb.active_tb_state[success_uids] = TBS.NONE
        tb.ti_active[success_uids] = np.nan
        ppl.diagnosed[success_uids] = False
        ppl.tb_treatment_success[success_uids] = True

        # Update failure
        ppl.treatment_failure[failure_uids] = True

        if self.pars.reset_flags:
            ppl.diagnosed[failure_uids] = False
            ppl.tested[failure_uids] = False

        # Trigger renewed care-seeking for failures
        if len(failure_uids):
            ppl.sought_care[failure_uids] = False
            ppl.care_seeking_multiplier[failure_uids] *= self.pars.reseek_multiplier
            ppl.multiplier_applied[failure_uids] = True

        # Store
        self.new_treated = tx_uids
        self.successes = success_uids
        self.failures = failure_uids

    def init_results(self):
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_treatment_success', dtype=int),
            ss.Result('n_treatment_failure', dtype=int),
            ss.Result('cum_treatment_success', dtype=int),
            ss.Result('cum_treatment_failure', dtype=int),
        )

    def update_results(self):
        n_treated = len(self.new_treated)
        n_success = len(self.successes)
        n_failure = len(self.failures)

        self.results['n_treated'][self.ti] = n_treated
        self.results['n_treatment_success'][self.ti] = n_success
        self.results['n_treatment_failure'][self.ti] = n_failure

        if self.ti > 0:
            self.results['cum_treatment_success'][self.ti] = self.results['cum_treatment_success'][self.ti - 1] + n_success
            self.results['cum_treatment_failure'][self.ti] = self.results['cum_treatment_failure'][self.ti - 1] + n_failure
        else:
            self.results['cum_treatment_success'][self.ti] = n_success
            self.results['cum_treatment_failure'][self.ti] = n_failure

        # Reset for next step
        self.new_treated = []
        self.successes = []
        self.failures = []

if __name__ == '__main__':

    # import tbsim as mtb
    # import starsim as ss
    # import matplotlib.pyplot as plt
    # import numpy as np

    # sim = ss.Sim(
    #     people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
    #     diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
    #     interventions=[
    #         mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),
    #         mtb.TBDiagnostic(pars={
    #             'coverage': ss.bernoulli(0.8, strict=False),
    #             'sensitivity': 0.50,
    #             'specificity': 0.50,
    #             'care_seeking_multiplier': 2.0,
    #         }),
    #         mtb.TBTreatment(pars={
    #             'treatment_success_rate': 0.80,
    #             'reseek_multiplier': 2.0,
    #             'reset_flags': True,
    #         }),
    #     ],
    #     networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
    #     pars=dict(start=2000, stop=2010, dt=1/12),
    # )

    # sim.run()

    # # Plot TB prevalence (active cases) over time
    # tb = sim.results['tb']
    # timevec = tb['n_active'].timevec
    # n_active = tb['n_active'].values

    # plt.figure(figsize=(10, 5))
    # plt.plot(timevec, n_active, label='Active TB Cases', color='blue')
    # plt.xlabel('Time')
    # plt.ylabel('Number of People with Active TB')
    # plt.title('Active TB Prevalence Over Time')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    # # Results extraction
    # results = sim.results['tbtreatment']
    # timevec = results['n_treated'].timevec

    # # Plot new treated, successes, failures
    # plt.figure(figsize=(10, 5))
    # plt.plot(timevec, results['n_treated'].values, label='Treated', marker='o')
    # plt.plot(timevec, results['n_treatment_success'].values, label='Successes', linestyle='--')
    # plt.plot(timevec, results['n_treatment_failure'].values, label='Failures', linestyle=':')
    # plt.xlabel('Time')
    # plt.ylabel('People')
    # plt.title('TB Treatment Outcomes')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Summary stats
    # total_treated = np.sum(results['n_treated'].values)
    # total_success = np.sum(results['n_treatment_success'].values)
    # total_failure = np.sum(results['n_treatment_failure'].values)

    # print(f"Total treated: {total_treated}")
    # print(f"Total treatment successes: {total_success}")
    # print(f"Total treatment failures: {total_failure}")

    # # Behavior and diagnostic outputs
    # people = sim.people
    # print("Final care-seeking multiplier (mean):", np.mean(people.care_seeking_multiplier))
    # print("People who were tested:", np.sum(people.tested))
    # print("People who were diagnosed:", np.sum(people.diagnosed))
    # print("People with treatment success:", np.sum(people.tb_treatment_success))
    # print("People with treatment failure:", np.sum(people.treatment_failure))

    # # Cumulative treatment outcomes over time
    # tbtx = sim.results['tbtreatment']
    # timevec = tbtx['cum_treatment_success'].timevec

    # plt.figure(figsize=(10, 5))
    # plt.plot(timevec, tbtx['cum_treatment_success'].values, label='Cumulative Treatment Successes', linestyle='--', color='green')
    # plt.plot(timevec, tbtx['cum_treatment_failure'].values, label='Cumulative Treatment Failures', linestyle=':', color='red')
    # plt.xlabel('Time')
    # plt.ylabel('Cumulative Treatments')
    # plt.title('Cumulative TB Treatment Outcomes')
    # plt.legend()
    # plt.grid(True)
    # plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # plt.tight_layout()
    # plt.show()

    # Run two scenarios side by side with different treatment_success_rate values
    import tbsim as mtb
    import starsim as ss
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as mticker

    # Define a reusable function to run one scenario
    def run_scenario(label, treatment_success_rate):
        sim = ss.Sim(
            people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
            diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
            interventions=[
                mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),
                mtb.TBDiagnostic(pars={
                    'coverage': ss.bernoulli(0.8, strict=False),
                    'sensitivity': 0.20,
                    'specificity': 0.20,
                    'care_seeking_multiplier': 2.0,
                }),
                mtb.TBTreatment(pars={
                    'treatment_success_rate': treatment_success_rate,
                    'reseek_multiplier': 2.0,
                    'reset_flags': True,
                }),
            ],
            networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
            pars=dict(start=2000, stop=2010, dt=1/12),
        )
        sim.run()
        return sim

    # Run scenarios with different treatment success rates
    sim_low = run_scenario("Low Success (50%)", treatment_success_rate=0.5)
    sim_high = run_scenario("High Success (90%)", treatment_success_rate=0.9)

    # Plot TB prevalence for both scenarios
    plt.figure(figsize=(10, 5))
    plt.plot(sim_low.results['tb']['n_active'].timevec,
            sim_low.results['tb']['n_active'].values,
            label='Active TB – 50% Success', linestyle=':', color='red')
    plt.plot(sim_high.results['tb']['n_active'].timevec,
            sim_high.results['tb']['n_active'].values,
            label='Active TB – 90% Success', linestyle='-', color='green')
    plt.xlabel('Time')
    plt.ylabel('Number of People')
    plt.title('TB Prevalence Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot cumulative treatment outcomes for both scenarios
    plt.figure(figsize=(10, 5))
    plt.plot(sim_low.results['tbtreatment']['cum_treatment_success'].timevec,
            sim_low.results['tbtreatment']['cum_treatment_success'].values,
            label='Cumulative Success – 50%', linestyle='--', color='green')
    plt.plot(sim_high.results['tbtreatment']['cum_treatment_success'].timevec,
            sim_high.results['tbtreatment']['cum_treatment_success'].values,
            label='Cumulative Success – 90%', linestyle='-', color='green')

    plt.plot(sim_low.results['tbtreatment']['cum_treatment_failure'].timevec,
            sim_low.results['tbtreatment']['cum_treatment_failure'].values,
            label='Cumulative Failure – 50%', linestyle='--', color='red')
    plt.plot(sim_high.results['tbtreatment']['cum_treatment_failure'].timevec,
            sim_high.results['tbtreatment']['cum_treatment_failure'].values,
            label='Cumulative Failure – 90%', linestyle='-', color='red')

    plt.xlabel('Time')
    plt.ylabel('Cumulative Treatments')
    plt.title('Cumulative Treatment Outcomes')
    plt.grid(True)
    plt.legend()
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
