import numpy as np
import starsim as ss


class TPTInitiation(ss.Intervention):
    """
    Tuberculosis Preventive Therapy (TPT) intervention for entire eligible households.

    This intervention identifies households with at least one TB-treated individual, and offers TPT to all
    other members of those households who meet the following eligibility criteria:
    
    Eligibility criteria:
        - Must reside in a household where at least one member is on TB treatment
        - Must not already be on TB treatment themselves
        - Must be screen-negative or non-symptomatic
        - Optionally filtered by age and/or HIV status (logic can be extended)
    
    Treatment logic:
        - A Bernoulli trial (`p_tpt`) is used to determine which eligible individuals receive TPT
        - If initiated, individuals are marked as `on_tpt` and receive a fixed protection duration (`tpt_duration`)
        - After the specified `start` date, a proportion (`p_3HP`) receive the 3HP regimen
    
    Parameters:
        p_tpt (float or ss.Bernoulli): Probability of initiating TPT for an eligible individual
        tpt_duration (float): Duration of protection in years
        max_age (int): Optional filter for outcome reporting (default: 5)
        hiv_status_threshold (bool): Reserved for HIV-based filtering (default: False)
        p_3HP (float): Proportion of individuals initiated on 3HP after the `start` date
        start (date): Rollout date after which 3HP becomes available

    Results tracked:
        n_eligible (int): Number of individuals in eligible households meeting criteria
        n_tpt_initiated (int): Number of individuals actually started on TPT
        n_3HP_assigned (int): Subset of TPT individuals presumed to receive 3HP

    Notes:
        - Requires people to have a 'HHID' attribute (household ID).
        - Assumes states like 'on_tpt', 'received_tpt', 'screen_negative' are initialized.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            p_tpt=ss.bernoulli(1.0),
            tpt_duration=2.0,
            max_age=5,
            hiv_status_threshold=False,
            p_3HP=0.3,
            start=ss.date('2000-01-01'),
        )
        self.update_pars(pars, **kwargs)

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Identify households with at least one member currently on TB treatment
        treated = tb.on_treatment & ppl.alive
        eligible_hhids = np.unique(ppl['HHID'][treated])

        # Identify all members of those households
        in_eligible_households = np.isin(ppl['HHID'], eligible_hhids)
        eligible = in_eligible_households & (~tb.on_treatment) & (ppl['screen_negative'] | ppl['non_symptomatic'])

        tpt_candidates = self.pars.p_tpt.filter(eligible.uids)

        if len(tpt_candidates):
            use_3HP = sim.year >= self.pars.start.year
            assigned_3HP = np.random.rand(len(tpt_candidates)) < self.pars.p_3HP if use_3HP else np.zeros(len(tpt_candidates), dtype=bool)

            if not hasattr(tb, 'on_treatment_duration'):
                tb.define_states(ss.FloatArr('on_treatment_duration', default=0.0))

            tb.start_treatment(tpt_candidates)
            tb.on_treatment_duration[tpt_candidates] = self.pars.tpt_duration
            ppl['on_tpt'][tpt_candidates] = True
            ppl['received_tpt'][tpt_candidates] = True

            self.results['n_eligible'][self.ti] = np.count_nonzero(eligible)
            self.results['n_tpt_initiated'][self.ti] = len(tpt_candidates)
            self.results['n_3HP_assigned'][self.ti] = np.count_nonzero(assigned_3HP)

    def init_results(self):
        self.define_results(
            ss.Result('n_eligible', dtype=int),
            ss.Result('n_tpt_initiated', dtype=int),
            ss.Result('n_3HP_assigned', dtype=int),
        )

    def update_results(self):
        ppl = self.sim.people
        self.results['n_eligible'][self.ti] = np.count_nonzero(ppl['on_tpt'] | ppl['received_tpt'])
        self.results['n_tpt_initiated'][self.ti] = np.count_nonzero(ppl['on_tpt'])
        self.results['n_3HP_assigned'][self.ti] = np.count_nonzero(ppl['on_tpt'] & (ppl.age < self.pars.max_age))