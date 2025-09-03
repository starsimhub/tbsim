import numpy as np
import starsim as ss
from tbsim import TBS

__all__ = ['TBDiagnostic']


class TBDiagnostic(ss.Intervention):
    """
    TB diagnostic intervention that performs testing on individuals who seek care.
    
    This intervention simulates TB diagnostic testing with configurable sensitivity and specificity.
    It is triggered when individuals seek care and can handle false negatives by allowing
    retesting with increased care-seeking probability.
    
    Parameters:
        coverage (float or Dist): Fraction of those who sought care who get tested.
                                 Can be a fixed probability or a distribution.
        sensitivity (float): Probability that a test is positive given the person has TB.
                            Range: 0.0 to 1.0. Higher values mean fewer false negatives.
        specificity (float): Probability that a test is negative given the person does not have TB.
                            Range: 0.0 to 1.0. Higher values mean fewer false positives.
        reset_flag (bool): Whether to reset the `sought_care` flag after testing.
                          Default: False (allows for retesting).
        care_seeking_multiplier (float): Multiplier applied to care-seeking probability
                                       for individuals with false negative results.
                                       Default: 1.0 (no change).
    
    Results:
        n_tested (int): Number of people tested at each timestep.
        n_test_positive (int): Number of positive test results at each timestep.
        n_test_negative (int): Number of negative test results at each timestep.
        cum_test_positive (int): Cumulative number of positive test results.
        cum_test_negative (int): Cumulative number of negative test results.
    
    Person States Modified:
        tested (bool): Set to True for all tested individuals.
        n_times_tested (int): Incremented for each test.
        test_result (bool): True for positive results, False for negative.
        diagnosed (bool): Set to True for positive test results.
        care_seeking_multiplier (float): Multiplied by care_seeking_multiplier parameter
                                       for false negative cases.
        multiplier_applied (bool): Tracks whether multiplier has been applied.
        sought_care (bool): Reset to False for false negative cases to allow retesting.
    
    Example:
        >>> sim = ss.Sim(
        ...     interventions=[
        ...         mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.1)}),
        ...         mtb.TBDiagnostic(pars={
        ...             'coverage': 0.8,
        ...             'sensitivity': 0.85,
        ...             'specificity': 0.95,
        ...             'care_seeking_multiplier': 2.0
        ...         })
        ...     ]
        ... )
    """
    def __init__(self, pars=None, **kwargs):
        """
        Initialize the TB diagnostic intervention.
        
        Args:
            pars (dict, optional): Dictionary of parameters to override defaults.
            **kwargs: Additional keyword arguments passed to parent class.
        """
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
        """
        Execute one timestep of TB diagnostic testing.
        
        This method:
        1. Identifies eligible individuals (sought care, not diagnosed, alive)
        2. Applies coverage filter to select who gets tested
        3. Determines TB status and applies test sensitivity/specificity
        4. Updates person states based on test results
        5. Handles false negatives by allowing retesting with increased care-seeking probability
        """
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Find people who sought care but haven't been tested
        # eligible = ppl.sought_care & (~ppl.tested) & ppl.alive
        eligible = ppl.sought_care & (~ppl.diagnosed) & ppl.alive  # Avoids excluding once-tested people
        uids = eligible.uids
        if len(uids) == 0:
            return

        # Apply coverage filter to determine who actually gets tested
        if isinstance(self.pars.coverage, ss.Dist):
            selected = self.pars.coverage.filter(uids)
        else:
            selected = ss.bernoulli(self.pars.coverage).filter(uids)
        if len(selected) == 0:
            return

        # Determine TB status for selected individuals
        tb_states = tb.state[selected]
        has_tb = np.isin(tb_states, [TBS.ACTIVE_SMPOS,
                                     TBS.ACTIVE_SMNEG,
                                     TBS.ACTIVE_EXPTB])

        # Apply test sensitivity and specificity logic
        # For true TB cases: test positive with probability = sensitivity
        # For non-TB cases: test positive with probability = (1 - specificity)
        rand = np.random.rand(len(selected))
        test_positive = ((has_tb & (rand < self.pars.sensitivity)) |
                         (~has_tb & (rand < self.pars.specificity)))

        # Update person state
        ppl.tested[selected] = True
        ppl.n_times_tested[selected] += 1
        ppl.test_result[selected] = test_positive
        ppl.diagnosed[selected[test_positive]] = True

        # Optional: reset the health-seeking flag after testing
        if self.pars.reset_flag:
            ppl.sought_care[selected] = False

        # Handle false negatives: individuals with TB who tested negative
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

        if len(unboosted):
            # Increase care-seeking probability for future attempts
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



    def init_results(self):
        """
        Initialize result tracking for the diagnostic intervention.
        
        Defines the following metrics to track over time:
        - n_tested: Number of people tested at each timestep
        - n_test_positive: Number of positive test results at each timestep
        - n_test_negative: Number of negative test results at each timestep
        - cum_test_positive: Cumulative number of positive test results
        - cum_test_negative: Cumulative number of negative test results
        """
        super().init_results()
        self.define_results(
            ss.Result('n_tested', dtype=int),
            ss.Result('n_test_positive', dtype=int),
            ss.Result('n_test_negative', dtype=int),
            ss.Result('cum_test_positive', dtype=int),
            ss.Result('cum_test_negative', dtype=int),
        )


    def update_results(self):
        """
        Update result tracking for the current timestep.
        
        Records the number of people tested and their results for this timestep,
        and updates cumulative totals. Resets temporary storage for next timestep.
        """
        # Calculate per-step counts
        n_tested = len(self.tested_this_step)
        n_pos = np.count_nonzero(self.test_result_this_step)
        n_neg = n_tested - n_pos

        # Record current timestep results
        self.results['n_tested'][self.ti] = n_tested
        self.results['n_test_positive'][self.ti] = n_pos
        self.results['n_test_negative'][self.ti] = n_neg

        # Update cumulative totals (add to previous step's cumulative)
        if self.ti > 0:
            self.results['cum_test_positive'][self.ti] = self.results['cum_test_positive'][self.ti-1] + n_pos
            self.results['cum_test_negative'][self.ti] = self.results['cum_test_negative'][self.ti-1] + n_neg
        else:
            # First timestep: cumulative equals current
            self.results['cum_test_positive'][self.ti] = n_pos
            self.results['cum_test_negative'][self.ti] = n_neg

        # Reset temporary storage for next timestep
        self.tested_this_step = []
        self.test_result_this_step = []
