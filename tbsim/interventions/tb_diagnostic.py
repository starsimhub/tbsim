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
    
    The intervention models the diagnostic cascade in TB care, where individuals who seek
    healthcare are tested for TB using diagnostic tests with known sensitivity and specificity.
    The intervention handles both true positive/negative results and false positive/negative
    results, with special logic for false negatives to allow retesting.
    
    Use Cases:
    ----------
    - Simulating TB diagnostic programs in healthcare settings
    - Evaluating the impact of diagnostic test performance on TB case detection
    - Modeling diagnostic coverage and its effect on TB transmission
    - Assessing the role of false negatives in TB care-seeking behavior
    - Studying the effectiveness of different diagnostic strategies
    
    Inputs/Requirements:
    --------------------
    The intervention requires the following to be present in the simulation:
    - TB disease module (sim.diseases.tb) with active TB states
    - Health seeking behavior intervention that sets the 'sought_care' flag
    - Person states: 'sought_care', 'tested', 'diagnosed', 'n_times_tested', 
      'test_result', 'care_seeking_multiplier', 'multiplier_applied'
    - TB states: ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB for active TB cases
    
    Parameters:
    -----------
    coverage (float or Dist): Fraction of those who sought care who get tested.
                             Can be a fixed probability or a distribution.
                             Range: 0.0 to 1.0. Default: 1.0 (all eligible get tested).
    sensitivity (float): Probability that a test is positive given the person has TB.
                        Range: 0.0 to 1.0. Higher values mean fewer false negatives.
                        Default: 0.9 (90% sensitivity).
    specificity (float): Probability that a test is negative given the person does not have TB.
                        Range: 0.0 to 1.0. Higher values mean fewer false positives.
                        Default: 0.95 (95% specificity).
    reset_flag (bool): Whether to reset the `sought_care` flag after testing.
                      Default: False (allows for retesting).
    care_seeking_multiplier (float): Multiplier applied to care-seeking probability
                                   for individuals with false negative results.
                                   Default: 1.0 (no change).
    
    Outputs/Reports:
    ----------------
    The intervention generates the following results that can be accessed via sim.results:
    - n_tested (int): Number of people tested at each timestep
    - n_test_positive (int): Number of positive test results at each timestep
    - n_test_negative (int): Number of negative test results at each timestep
    - cum_test_positive (int): Cumulative number of positive test results
    - cum_test_negative (int): Cumulative number of negative test results
    
    Person States Modified:
    -----------------------
    The intervention modifies the following person states:
    - tested (bool): Set to True for all tested individuals
    - n_times_tested (int): Incremented for each test performed
    - test_result (bool): True for positive results, False for negative results
    - diagnosed (bool): Set to True for positive test results
    - care_seeking_multiplier (float): Multiplied by care_seeking_multiplier parameter
                                     for false negative cases to increase future care-seeking
    - multiplier_applied (bool): Tracks whether multiplier has been applied to prevent
                               multiple applications
    - sought_care (bool): Reset to False for false negative cases to allow retesting
    
    Algorithm:
    ----------
    1. Identify eligible individuals (sought care, not diagnosed, alive)
    2. Apply coverage filter to select who gets tested
    3. Determine TB status for selected individuals
    4. Apply test sensitivity and specificity logic:
       - True TB cases: test positive with probability = sensitivity
       - Non-TB cases: test positive with probability = (1 - specificity)
    5. Update person states based on test results
    6. Handle false negatives by:
       - Increasing care-seeking probability for future attempts
       - Resetting care-seeking and testing flags to allow retesting
    
    Example:
    --------
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
        
        This method sets up the diagnostic intervention with default parameters and
        initializes temporary storage for result tracking. The intervention is ready
        to be added to a simulation after initialization.
        
        Args:
            pars (dict, optional): Dictionary of parameters to override defaults.
                                 Valid keys: 'coverage', 'sensitivity', 'specificity',
                                 'reset_flag', 'care_seeking_multiplier'.
            **kwargs: Additional keyword arguments passed to parent class.
        
        Default Parameters:
        -------------------
        - coverage: 1.0 (all eligible individuals get tested)
        - sensitivity: 0.9 (90% sensitivity)
        - specificity: 0.95 (95% specificity)
        - reset_flag: False (allows retesting)
        - care_seeking_multiplier: 1.0 (no change to care-seeking probability)
        
        Raises:
        -------
        ValueError: If parameter values are outside valid ranges (0.0-1.0 for probabilities).
        
        Note:
        -----
        The intervention requires specific person states to be present in the simulation.
        These are typically initialized by the TB disease module and health seeking
        behavior intervention.
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
        
        This method performs the core diagnostic testing logic for each simulation timestep.
        It identifies eligible individuals, applies diagnostic tests with specified sensitivity
        and specificity, and handles the consequences of test results including false negatives.
        
        Process Flow:
        ------------
        1. Identify eligible individuals (sought care, not diagnosed, alive)
        2. Apply coverage filter to select who gets tested
        3. Determine TB status and apply test sensitivity/specificity
        4. Update person states based on test results
        5. Handle false negatives by allowing retesting with increased care-seeking probability
        
        Eligibility Criteria:
        --------------------
        - Individual must have sought care (sought_care = True)
        - Individual must not be already diagnosed (diagnosed = False)
        - Individual must be alive (alive = True)
        
        Test Logic:
        ----------
        - True TB cases: test positive with probability = sensitivity
        - Non-TB cases: test positive with probability = (1 - specificity)
        
        False Negative Handling:
        ------------------------
        - Individuals with TB who test negative are identified as false negatives
        - Their care-seeking probability is increased by the care_seeking_multiplier
        - Their care-seeking and testing flags are reset to allow retesting
        - The multiplier is only applied once per individual to prevent infinite boosting
        
        Side Effects:
        -------------
        - Updates person states: tested, n_times_tested, test_result, diagnosed
        - Modifies care_seeking_multiplier and multiplier_applied for false negatives
        - Resets sought_care and tested flags for false negatives
        - Stores results for update_results method
        
        Returns:
        --------
        None
        
        Note:
        -----
        This method is called automatically by the simulation framework at each timestep.
        The intervention must be added to the simulation's interventions list to be executed.
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
            # Use starsim bernoulli distribution for coverage selection
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
        test_positive = np.zeros(len(selected), dtype=bool)
        
        # Handle true TB cases with sensitivity
        tb_cases = selected[has_tb]
        if len(tb_cases) > 0:
            tb_test_positive = ss.bernoulli(self.pars.sensitivity).filter(tb_cases)
            test_positive[has_tb] = np.isin(selected[has_tb], tb_test_positive)
        
        # Handle non-TB cases with specificity (test positive with prob = 1-specificity)
        non_tb_cases = selected[~has_tb]
        if len(non_tb_cases) > 0:
            non_tb_test_positive = ss.bernoulli(1 - self.pars.specificity).filter(non_tb_cases)
            test_positive[~has_tb] = np.isin(selected[~has_tb], non_tb_test_positive)

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
            ppl.multiplier_applied[unboosted] = True  # mark as boosted 

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
        
        This method sets up the result tracking system for the diagnostic intervention.
        It defines the metrics that will be recorded at each timestep and makes them
        available for analysis and reporting.
        
        Result Metrics:
        --------------
        - n_tested (int): Number of people tested at each timestep
        - n_test_positive (int): Number of positive test results at each timestep
        - n_test_negative (int): Number of negative test results at each timestep
        - cum_test_positive (int): Cumulative number of positive test results
        - cum_test_negative (int): Cumulative number of negative test results
        
        Usage:
        ------
        Results can be accessed after simulation completion via:
        - sim.results['TBDiagnostic']['n_tested']
        - sim.results['TBDiagnostic']['n_test_positive']
        - etc.
        
        Note:
        -----
        This method is called automatically by the simulation framework during
        initialization. The results are updated each timestep by update_results().
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
        
        This method records the diagnostic testing results for the current timestep
        and updates cumulative totals. It processes the temporary storage from the
        step() method and stores the results in the intervention's result arrays.
        
        Process:
        -------
        1. Calculate per-step counts from temporary storage
        2. Record current timestep results
        3. Update cumulative totals (add to previous step's cumulative)
        4. Reset temporary storage for next timestep
        
        Calculations:
        -------------
        - n_tested: Total number of people tested this timestep
        - n_test_positive: Number of positive test results this timestep
        - n_test_negative: Number of negative test results this timestep
        - cum_test_positive: Cumulative positive results (previous + current)
        - cum_test_negative: Cumulative negative results (previous + current)
        
        Data Sources:
        -------------
        - self.tested_this_step: Array of UIDs tested this timestep
        - self.test_result_this_step: Array of test results (True/False)
        
        Note:
        -----
        This method is called automatically by the simulation framework after
        each timestep. The temporary storage is reset after processing to prepare
        for the next timestep.
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
