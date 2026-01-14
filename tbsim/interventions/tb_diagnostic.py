import numpy as np
import starsim as ss
from tbsim import TBS

__all__ = ['TBDiagnostic', 'EnhancedTBDiagnostic']


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
        
        Parameters
        ----------
        pars : dict, optional
            Dictionary of parameters to override defaults.
            Valid keys: 'coverage', 'sensitivity', 'specificity',
            'reset_flag', 'care_seeking_multiplier'.
        **kwargs
            Additional keyword arguments passed to parent class.
        
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
        selected = [] # reset to empty list
        if isinstance(self.pars.coverage, ss.Dist):
            selected = self.pars.coverage.filter(uids)
        else:
            selected = ss.bernoulli(self.pars.coverage, strict=False).filter(uids)

        
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
            tb_test_positive = ss.bernoulli(self.pars.sensitivity, strict=False).filter(tb_cases)
            test_positive[has_tb] = np.isin(selected[has_tb], tb_test_positive)
        
        # Handle non-TB cases with specificity (test positive with prob = 1-specificity)
        non_tb_cases = selected[~has_tb]
        if len(non_tb_cases) > 0:
            non_tb_test_positive = ss.bernoulli(1 - self.pars.specificity, strict=False).filter(non_tb_cases)
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


class EnhancedTBDiagnostic(ss.Intervention):
    """
    Enhanced TB diagnostic intervention with stratified diagnostic parameters.
    
    This intervention provides age, TB state, and HIV-stratified sensitivity/specificity
    parameters for multiple diagnostic methods, integrated with health-seeking behavior.
    
    Features:
    1. Age and TB state-specific sensitivity/specificity parameters
    2. HIV-stratified parameters for LAM testing
    3. Integration with health-seeking behavior
    4. False negative handling with care-seeking multipliers
    5. Comprehensive result tracking
    
    Diagnostic Methods:
    - Xpert MTB/RIF: Primary molecular diagnostic test
    - Oral Swab: Non-sputum based testing (optional)
    - FujiLAM: Urine-based test for HIV-positive individuals (optional)
    - CAD CXR: Computer-aided chest X-ray for pediatric cases (optional)
    
    Note: Diagnostic performance parameters are based on published literature values
    for each test method. Parameters can be customized via the pars argument.
    """
    
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        # Define comprehensive parameters
        self.define_pars(
            # Coverage and basic parameters
            coverage=1.0,
            reset_flag=False,
            care_seeking_multiplier=1.0,
            
            # Xpert MTB/RIF baseline parameters
            # Values based on published literature for Xpert MTB/RIF Ultra diagnostic test
            # Source: See Papers - Parameters folder in tb_peds repository for original references
            sensitivity_adult_smearpos=0.909,
            specificity_adult_smearpos=0.966,
            sensitivity_adult_smearneg=0.775,
            specificity_adult_smearneg=0.958,
            sensitivity_adult_eptb=0.775,
            specificity_adult_eptb=0.958,
            sensitivity_child=0.73,
            specificity_child=0.95,
            
            # Oral swab parameters (optional, literature-based)
            # Values: Adult smear-pos: sens 66-92%, spec 66-100% (using 80%, 90%)
            #         Adult smear-neg: sens 30% (95% CI: 11-59), spec 98% (95% CI: 88-100)
            #         Children: sens 8-42% (using 25%), spec 93-100% (using 95%)
            # Source: See Papers - Parameters folder in tb_peds repository for original references
            #         Based on interventions_updated.py comments
            use_oral_swab=False,
            sens_adult_smearpos_oral=0.80,
            spec_adult_smearpos_oral=0.90,
            sens_adult_smearneg_oral=0.30,
            spec_adult_smearneg_oral=0.98,
            sens_child_oral=0.25,
            spec_child_oral=0.95,
            
            # FujiLAM parameters (optional, literature-based)
            # Values: HIV+ adults (TB): sens 75% (95% CI: 72-79), spec 90% (95% CI: 88-82)
            #         HIV+ adults (EPTB): sens 75% (95% CI: 47.6-92.7), spec 73.9% (95% CI: 69.8-78.0)
            #         HIV- adults: sens 58% (95% CI: 51-66), spec 98% (95% CI: 97-99)
            #         HIV+ children: sens 57.9% (95% CI: 48.4-67.3), spec 87.7% (95% CI: 75.3-100)
            #         HIV- children: sens 51% (95% CI: 27.5-74.5), spec 89.5% (95% CI: 84.7-94.2)
            # Source: See Papers - Parameters folder in tb_peds repository for original references
            #         Based on interventions_updated.py comments
            use_fujilam=False,
            sens_hivpos_adult_tb=0.75,
            spec_hivpos_adult_tb=0.90,
            sens_hivpos_adult_eptb=0.75,
            spec_hivpos_adult_eptb=0.739,
            sens_hivneg_adult=0.58,
            spec_hivneg_adult=0.98,
            sens_hivpos_child=0.579,
            spec_hivpos_child=0.877,
            sens_hivneg_child=0.51,
            spec_hivneg_child=0.895,
            
            # CAD CXR parameters (optional, literature-based)
            # Values: Children: sens 66%, spec 79%
            # Note: Applied to children only (not EPTB)
            # Source: See Papers - Parameters folder in tb_peds repository for original references
            #         Based on interventions_updated.py comments
            use_cadcxr=False,
            cad_cxr_sensitivity=0.66,
            cad_cxr_specificity=0.79,
        )
        self.update_pars(pars=pars, **kwargs)

        # Temporary state for update_results
        self.tested_this_step = []
        self.test_result_this_step = []
        self.diagnostic_method_used = []  # Track which diagnostic was used

    def _get_diagnostic_parameters(self, uid, age, tb_state, hiv_state=None):
        """
        Get sensitivity and specificity based on individual characteristics.
        """
        is_child = age < 15
        hiv_positive = hiv_state is not None and hiv_state in [1, 2, 3]  # ACUTE, LATENT, AIDS
        
        # Determine which diagnostic method to use
        if self.pars.use_fujilam and hiv_state is not None:
            # FujiLAM for HIV-positive individuals
            if is_child:
                if hiv_positive:
                    return self.pars.sens_hivpos_child, self.pars.spec_hivpos_child
                else:
                    return self.pars.sens_hivneg_child, self.pars.spec_hivneg_child
            else:
                if hiv_positive:
                    if tb_state == TBS.ACTIVE_EXPTB:
                        return self.pars.sens_hivpos_adult_eptb, self.pars.spec_hivpos_adult_eptb
                    else:
                        return self.pars.sens_hivpos_adult_tb, self.pars.spec_hivpos_adult_tb
                else:
                    return self.pars.sens_hivneg_adult, self.pars.spec_hivneg_adult
        
        elif self.pars.use_cadcxr and is_child and tb_state != TBS.ACTIVE_EXPTB:
            # CAD CXR for children (not EPTB)
            return self.pars.cad_cxr_sensitivity, self.pars.cad_cxr_specificity
        
        elif self.pars.use_oral_swab:
            # Oral swab parameters
            if is_child:
                return self.pars.sens_child_oral, self.pars.spec_child_oral
            elif tb_state == TBS.ACTIVE_SMPOS:
                return self.pars.sens_adult_smearpos_oral, self.pars.spec_adult_smearpos_oral
            elif tb_state == TBS.ACTIVE_SMNEG:
                return self.pars.sens_adult_smearneg_oral, self.pars.spec_adult_smearneg_oral
            else:
                return self.pars.sens_adult_smearneg_oral, self.pars.spec_adult_smearneg_oral
        
        else:
            # Default Xpert baseline parameters
            if is_child:
                return self.pars.sensitivity_child, self.pars.specificity_child
            elif tb_state == TBS.ACTIVE_SMPOS:
                return self.pars.sensitivity_adult_smearpos, self.pars.specificity_adult_smearpos
            elif tb_state == TBS.ACTIVE_SMNEG:
                return self.pars.sensitivity_adult_smearneg, self.pars.specificity_adult_smearneg
            elif tb_state == TBS.ACTIVE_EXPTB:
                return self.pars.sensitivity_adult_eptb, self.pars.specificity_adult_eptb
            else:
                return 0.0, 1.0  # Default for unknown states

    def _determine_diagnostic_method(self, uid, age, tb_state, hiv_state=None):
        """
        Determine which diagnostic method to use and return method name.
        """
        is_child = age < 15
        hiv_positive = hiv_state is not None and hiv_state in [1, 2, 3]
        
        if self.pars.use_fujilam and hiv_state is not None:
            return "FujiLAM"
        elif self.pars.use_cadcxr and is_child and tb_state != TBS.ACTIVE_EXPTB:
            return "CAD_CXR"
        elif self.pars.use_oral_swab:
            return "Oral_Swab"
        else:
            return "Xpert_Baseline"

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Find people who sought care but haven't been diagnosed
        eligible = ppl.sought_care & (~ppl.diagnosed) & ppl.alive
        uids = eligible.uids
        if len(uids) == 0:
            return

        # Apply coverage filter
        if isinstance(self.pars.coverage, ss.Dist):
            selected = self.pars.coverage.filter(uids)
        else:
            selected = ss.bernoulli(self.pars.coverage, strict=False).filter(uids)
        if len(selected) == 0:
            return

        # Get TB and HIV states for selected individuals
        tb_states = tb.state[selected]
        ages = ppl.age[selected]
        
        # Get HIV state if HIV disease exists
        hiv_states = None
        if hasattr(sim.diseases, 'hiv'):
            hiv_states = sim.diseases.hiv.state[selected]

        # Determine TB status
        has_tb = np.isin(tb_states, [TBS.ACTIVE_SMPOS,
                                     TBS.ACTIVE_SMNEG,
                                     TBS.ACTIVE_EXPTB])

        # Apply diagnostic testing with individual-specific parameters
        test_positive = np.zeros(len(selected), dtype=bool)
        diagnostic_methods = []

        for i, uid in enumerate(selected):
            age_i = float(ages[i])
            tb_state_i = tb_states[i]
            hiv_state_i = hiv_states[i] if hiv_states is not None else None
            
            # Get sensitivity/specificity for this individual
            sensitivity, specificity = self._get_diagnostic_parameters(
                uid, age_i, tb_state_i, hiv_state_i
            )
            
            # Determine diagnostic method used
            method = self._determine_diagnostic_method(
                uid, age_i, tb_state_i, hiv_state_i
            )
            diagnostic_methods.append(method)
            
            # Apply test logic
            rand = np.random.rand()
            has_tbi = has_tb[i]
            
            if has_tbi:
                test_positive[i] = rand < sensitivity
            else:
                test_positive[i] = rand > (1 - specificity)

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
            # Filter only those who haven't had multiplier applied yet
            unboosted = false_negative_uids[~ppl.multiplier_applied[false_negative_uids]]

            # Apply multiplier only to them
            if len(unboosted):
                ppl.care_seeking_multiplier[unboosted] *= self.pars.care_seeking_multiplier
                ppl.multiplier_applied[unboosted] = True

            # Reset flags to allow re-care-seeking
            ppl.sought_care[false_negative_uids] = False
            ppl.tested[false_negative_uids] = False

        # Store for update_results
        self.tested_this_step = selected
        self.test_result_this_step = test_positive
        self.diagnostic_method_used = diagnostic_methods

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_tested', dtype=int),
            ss.Result('n_test_positive', dtype=int),
            ss.Result('n_test_negative', dtype=int),
            ss.Result('cum_test_positive', dtype=int),
            ss.Result('cum_test_negative', dtype=int),
            ss.Result('n_xpert_baseline', dtype=int),
            ss.Result('n_oral_swab', dtype=int),
            ss.Result('n_fujilam', dtype=int),
            ss.Result('n_cadcxr', dtype=int),
        )

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

        # Count diagnostic methods used
        if hasattr(self, 'diagnostic_method_used') and self.diagnostic_method_used:
            methods = np.array(self.diagnostic_method_used)
            self.results['n_xpert_baseline'][self.ti] = np.sum(methods == 'Xpert_Baseline')
            self.results['n_oral_swab'][self.ti] = np.sum(methods == 'Oral_Swab')
            self.results['n_fujilam'][self.ti] = np.sum(methods == 'FujiLAM')
            self.results['n_cadcxr'][self.ti] = np.sum(methods == 'CAD_CXR')
        else:
            self.results['n_xpert_baseline'][self.ti] = 0
            self.results['n_oral_swab'][self.ti] = 0
            self.results['n_fujilam'][self.ti] = 0
            self.results['n_cadcxr'][self.ti] = 0

        # Reset temporary storage
        self.tested_this_step = []
        self.test_result_this_step = []
        self.diagnostic_method_used = []


# Example usage function
def create_enhanced_diagnostic_scenarios():
    """
    Create different diagnostic scenarios using the enhanced intervention.
    """
    scenarios = {
        'baseline': {
            'use_oral_swab': False,
            'use_fujilam': False,
            'use_cadcxr': False,
        },
        'oral_swab': {
            'use_oral_swab': True,
            'use_fujilam': False,
            'use_cadcxr': False,
        },
        'fujilam': {
            'use_oral_swab': False,
            'use_fujilam': True,
            'use_cadcxr': False,
        },
        'cadcxr': {
            'use_oral_swab': False,
            'use_fujilam': False,
            'use_cadcxr': True,
        },
        'combo_all': {
            'use_oral_swab': True,
            'use_fujilam': True,
            'use_cadcxr': True,
        }
    }
    return scenarios



if __name__ == '__main__':
    import tbsim as mtb
    import starsim as ss
    import matplotlib.pyplot as plt

    # Example simulation with enhanced diagnostic
    sim = ss.Sim(
        people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
        diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
        interventions=[
            mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),
            EnhancedTBDiagnostic(pars={
                'coverage': ss.bernoulli(0.8, strict=False),
                'use_oral_swab': True,
                'use_fujilam': True,
                'care_seeking_multiplier': 2.0,
            }),
        ],
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
        pars=dict(start=ss.date(2000), stop=ss.date(2010), dt=ss.months(1)),
    )
    sim.run()

    # Plot results
    tbdiag = sim.results['enhancedtbdiagnostic']
    
    plt.figure(figsize=(12, 8))
    
    # Plot diagnostic methods used
    plt.subplot(2, 2, 1)
    plt.plot(tbdiag['n_xpert_baseline'].timevec, tbdiag['n_xpert_baseline'].values, label='Xpert Baseline')
    plt.plot(tbdiag['n_oral_swab'].timevec, tbdiag['n_oral_swab'].values, label='Oral Swab')
    plt.plot(tbdiag['n_fujilam'].timevec, tbdiag['n_fujilam'].values, label='FujiLAM')
    plt.plot(tbdiag['n_cadcxr'].timevec, tbdiag['n_cadcxr'].values, label='CAD CXR')
    plt.xlabel('Time')
    plt.ylabel('Number of Tests')
    plt.title('Diagnostic Methods Used')
    plt.legend()
    plt.grid(True)
    
    # Plot test outcomes
    plt.subplot(2, 2, 2)
    plt.plot(tbdiag['n_test_positive'].timevec, tbdiag['n_test_positive'].values, label='Positive')
    plt.plot(tbdiag['n_test_negative'].timevec, tbdiag['n_test_negative'].values, label='Negative')
    plt.xlabel('Time')
    plt.ylabel('Number of Tests')
    plt.title('Test Outcomes')
    plt.legend()
    plt.grid(True)
    
    # Plot cumulative results
    plt.subplot(2, 2, 3)
    plt.plot(tbdiag['cum_test_positive'].timevec, tbdiag['cum_test_positive'].values, label='Cumulative Positive')
    plt.plot(tbdiag['cum_test_negative'].timevec, tbdiag['cum_test_negative'].values, label='Cumulative Negative')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Tests')
    plt.title('Cumulative Results')
    plt.legend()
    plt.grid(True)
    
    # Plot total tested
    plt.subplot(2, 2, 4)
    plt.plot(tbdiag['n_tested'].timevec, tbdiag['n_tested'].values, label='Total Tested')
    plt.xlabel('Time')
    plt.ylabel('Number of People')
    plt.title('Total Tests Per Time Step')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 