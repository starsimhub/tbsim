"""TB diagnostic testing interventions"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['TBDiagnostic', 'EnhancedTBDiagnostic']

# Active TB states in the LSHTM model (used for diagnostic eligibility)
_ACTIVE_TB_STATES = [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]


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
    - TB disease module (accessed via tbsim.get_tb(sim)) with active TB states
    - Health seeking behavior intervention that sets the 'sought_care' flag
    - Person states: 'sought_care', 'tested', 'diagnosed', 'n_times_tested', 
      'test_result', 'care_seeking_multiplier', 'multiplier_applied'
    - TB states: SYMPTOMATIC, ASYMPTOMATIC, NON_INFECTIOUS for active TB cases
    
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
    ...         tbsim.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.1)}),
    ...         tbsim.TBDiagnostic(pars={
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

        # Define person-level states needed by this intervention
        self.define_states(
            ss.BoolState('sought_care',            default=False),
            ss.BoolState('diagnosed',              default=False),
            ss.BoolArr('tested',                     default=False),
            ss.IntArr('n_times_tested',            default=0),
            ss.BoolState('test_result',            default=False),
            ss.FloatArr('care_seeking_multiplier', default=1.0),
            ss.BoolState('multiplier_applied',     default=False),
        )

        # Create distributions once (avoid recreating every step)
        if not isinstance(self.pars.coverage, ss.Dist):
            self.dist_coverage = ss.bernoulli(p=self.pars.coverage)
        self.dist_sensitivity = ss.bernoulli(p=self.pars.sensitivity)
        self.dist_false_positive = ss.bernoulli(p=1 - self.pars.specificity)

        # Temporary state for update_results
        self.tested_this_step = []
        self.test_result_this_step = []
        return

    def step(self):
        """Execute one timestep of TB diagnostic testing."""
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        # Find people who sought care but haven't been diagnosed
        eligible = self.sought_care & (~self.diagnosed) & ppl.alive
        uids = eligible.uids
        if len(uids) == 0:
            return

        # Apply coverage filter to determine who actually gets tested
        if isinstance(self.pars.coverage, ss.Dist):
            selected = self.pars.coverage.filter(uids)
        else:
            selected = self.dist_coverage.filter(uids)
        if len(selected) == 0:
            return

        # Determine TB status for selected individuals
        tb_states = tb.state[selected]
        has_tb = np.isin(tb_states, _ACTIVE_TB_STATES)

        # Apply test sensitivity and specificity
        test_positive = np.zeros(len(selected), dtype=bool)

        # True TB cases: test positive with probability = sensitivity
        tb_cases = selected[has_tb]
        if len(tb_cases) > 0:
            tb_test_positive = self.dist_sensitivity.filter(tb_cases)
            test_positive[has_tb] = np.isin(selected[has_tb], tb_test_positive)

        # Non-TB cases: test positive with probability = 1 - specificity
        non_tb_cases = selected[~has_tb]
        if len(non_tb_cases) > 0:
            non_tb_test_positive = self.dist_false_positive.filter(non_tb_cases)
            test_positive[~has_tb] = np.isin(selected[~has_tb], non_tb_test_positive)

        # Update person state
        self.tested[selected] = True
        self.n_times_tested[selected] += 1
        self.test_result[selected] = test_positive
        self.diagnosed[selected[test_positive]] = True

        # Optional: reset the health-seeking flag after testing
        if self.pars.reset_flag:
            self.sought_care[selected] = False

        # Handle false negatives: individuals with TB who tested negative
        false_negative_uids = selected[~test_positive & has_tb]

        # Filter only those who haven't had multiplier applied yet
        unboosted = false_negative_uids[~self.multiplier_applied[false_negative_uids]]

        if len(unboosted):
            # Increase care-seeking probability for future attempts
            self.care_seeking_multiplier[unboosted] *= self.pars.care_seeking_multiplier
            self.multiplier_applied[unboosted] = True

        # Reset flags to allow re-care-seeking
        self.sought_care[false_negative_uids] = False
        self.tested[false_negative_uids] = False

        # Store for update_results
        self.tested_this_step = selected
        self.test_result_this_step = test_positive
        return

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
        return

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
        return


class EnhancedTBDiagnostic(ss.Intervention):
    """
    Enhanced TB diagnostic intervention that combines detailed parameter stratification
    from interventions_updated.py with health-seeking integration from tb_diagnostic.py.
    
    This intervention provides:
    1. Age and TB state-specific sensitivity/specificity parameters
    2. HIV-stratified parameters for LAM testing
    3. Integration with health-seeking behavior
    4. False negative handling with care-seeking multipliers
    5. Comprehensive result tracking
    """
    
    def __init__(self, pars=None, **kwargs):
        """Initialize with age/HIV-stratified sensitivity and specificity for multiple diagnostic methods."""
        super().__init__(**kwargs)

        # Define comprehensive parameters combining both approaches
        # Note: LSHTM state mapping: SYMPTOMATIC≈smear+, ASYMPTOMATIC≈smear-, NON_INFECTIOUS≈EPTB
        self.define_pars(
            # Coverage and basic parameters
            coverage=1.0,
            reset_flag=False,
            care_seeking_multiplier=1.0,

            # Xpert baseline parameters (stratified by LSHTM active-TB category)
            sensitivity_adult_symptomatic=0.909,       # was smearpos
            specificity_adult_symptomatic=0.966,
            sensitivity_adult_asymptomatic=0.775,      # was smearneg
            specificity_adult_asymptomatic=0.958,
            sensitivity_adult_noninfectious=0.775,     # was eptb
            specificity_adult_noninfectious=0.958,
            sensitivity_child=0.73,
            specificity_child=0.95,

            # Oral swab parameters (optional)
            use_oral_swab=False,
            sens_adult_symptomatic_oral=0.80,
            spec_adult_symptomatic_oral=0.90,
            sens_adult_asymptomatic_oral=0.30,
            spec_adult_asymptomatic_oral=0.98,
            sens_child_oral=0.25,
            spec_child_oral=0.95,

            # FujiLAM parameters (optional)
            use_fujilam=False,
            sens_hivpos_adult_tb=0.75,
            spec_hivpos_adult_tb=0.90,
            sens_hivpos_adult_noninfectious=0.75,
            spec_hivpos_adult_noninfectious=0.739,
            sens_hivneg_adult=0.58,
            spec_hivneg_adult=0.98,
            sens_hivpos_child=0.579,
            spec_hivpos_child=0.877,
            sens_hivneg_child=0.51,
            spec_hivneg_child=0.895,

            # CAD CXR parameters (optional)
            use_cadcxr=False,
            cad_cxr_sensitivity=0.66,
            cad_cxr_specificity=0.79,
        )
        self.update_pars(pars=pars, **kwargs)

        # Define person-level states needed by this intervention
        self.define_states(
            ss.BoolState('sought_care',            default=False),
            ss.BoolState('diagnosed',              default=False),
            ss.BoolArr('tested',                     default=False),
            ss.IntArr('n_times_tested',            default=0),
            ss.BoolState('test_result',            default=False),
            ss.FloatArr('care_seeking_multiplier', default=1.0),
            ss.BoolState('multiplier_applied',     default=False),
        )

        # Create distributions once
        if not isinstance(self.pars.coverage, ss.Dist):
            self.dist_coverage = ss.bernoulli(p=self.pars.coverage)
        self.dist_test_positive = ss.bernoulli(p=self.p_test_positive)

        # Temporary state for update_results
        self.tested_this_step = []
        self.test_result_this_step = []
        self.diagnostic_method_used = []
        return

    def _get_diagnostic_parameters(self, uid, age, tb_state, hiv_state=None):
        """Get sensitivity and specificity based on individual characteristics."""
        is_child = age < 15
        hiv_positive = hiv_state is not None and hiv_state in [1, 2, 3]

        if self.pars.use_fujilam and hiv_state is not None:
            if is_child:
                if hiv_positive:
                    return self.pars.sens_hivpos_child, self.pars.spec_hivpos_child
                else:
                    return self.pars.sens_hivneg_child, self.pars.spec_hivneg_child
            else:
                if hiv_positive:
                    if tb_state == TBSL.NON_INFECTIOUS:
                        return self.pars.sens_hivpos_adult_noninfectious, self.pars.spec_hivpos_adult_noninfectious
                    else:
                        return self.pars.sens_hivpos_adult_tb, self.pars.spec_hivpos_adult_tb
                else:
                    return self.pars.sens_hivneg_adult, self.pars.spec_hivneg_adult

        elif self.pars.use_cadcxr and is_child and tb_state != TBSL.NON_INFECTIOUS:
            return self.pars.cad_cxr_sensitivity, self.pars.cad_cxr_specificity

        elif self.pars.use_oral_swab:
            if is_child:
                return self.pars.sens_child_oral, self.pars.spec_child_oral
            elif tb_state == TBSL.SYMPTOMATIC:
                return self.pars.sens_adult_symptomatic_oral, self.pars.spec_adult_symptomatic_oral
            else:
                return self.pars.sens_adult_asymptomatic_oral, self.pars.spec_adult_asymptomatic_oral

        else:
            if is_child:
                return self.pars.sensitivity_child, self.pars.specificity_child
            elif tb_state == TBSL.SYMPTOMATIC:
                return self.pars.sensitivity_adult_symptomatic, self.pars.specificity_adult_symptomatic
            elif tb_state == TBSL.ASYMPTOMATIC:
                return self.pars.sensitivity_adult_asymptomatic, self.pars.specificity_adult_asymptomatic
            elif tb_state == TBSL.NON_INFECTIOUS:
                return self.pars.sensitivity_adult_noninfectious, self.pars.specificity_adult_noninfectious
            else:
                return 0.0, 1.0

    def _determine_diagnostic_method(self, uid, age, tb_state, hiv_state=None):
        """Determine which diagnostic method to use and return method name."""
        is_child = age < 15
        if self.pars.use_fujilam and hiv_state is not None:
            return "FujiLAM"
        elif self.pars.use_cadcxr and is_child and tb_state != TBSL.NON_INFECTIOUS:
            return "CAD_CXR"
        elif self.pars.use_oral_swab:
            return "Oral_Swab"
        else:
            return "Xpert_Baseline"

    @staticmethod
    def p_test_positive(self, sim, uids):
        """Calculate per-individual probability of a positive test result."""
        tb = tbsim.get_tb(sim)
        ppl = sim.people
        tb_states = tb.state[uids]
        ages = ppl.age[uids]
        has_tb = np.isin(tb_states, _ACTIVE_TB_STATES)

        hiv_states = None
        if hasattr(sim.diseases, 'hiv'):
            hiv_states = sim.diseases.hiv.state[uids]

        p = np.zeros(len(uids), dtype=float)
        for i in range(len(uids)):
            age_i = float(ages[i])
            tb_state_i = tb_states[i]
            hiv_state_i = hiv_states[i] if hiv_states is not None else None
            sensitivity, specificity = self._get_diagnostic_parameters(uids[i], age_i, tb_state_i, hiv_state_i)
            if has_tb[i]:
                p[i] = sensitivity
            else:
                p[i] = 1 - specificity
        return p

    def step(self):
        """Test care-seekers using the appropriate diagnostic method and record results."""
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        # Find people who sought care but haven't been diagnosed
        eligible = self.sought_care & (~self.diagnosed) & ppl.alive
        uids = eligible.uids
        if len(uids) == 0:
            return

        # Apply coverage filter
        if isinstance(self.pars.coverage, ss.Dist):
            selected = self.pars.coverage.filter(uids)
        else:
            selected = self.dist_coverage.filter(uids)
        if len(selected) == 0:
            return

        # Get TB and HIV states for selected individuals
        tb_states = tb.state[selected]
        ages = ppl.age[selected]

        hiv_states = None
        if hasattr(sim.diseases, 'hiv'):
            hiv_states = sim.diseases.hiv.state[selected]

        has_tb = np.isin(tb_states, _ACTIVE_TB_STATES)

        # Apply diagnostic testing — bernoulli with per-individual probabilities
        test_positive_uids = self.dist_test_positive.filter(selected)
        test_positive = np.isin(selected, test_positive_uids)

        # Determine diagnostic methods used (for result tracking)
        diagnostic_methods = []
        for i, uid in enumerate(selected):
            age_i = float(ages[i])
            tb_state_i = tb_states[i]
            hiv_state_i = hiv_states[i] if hiv_states is not None else None
            method = self._determine_diagnostic_method(uid, age_i, tb_state_i, hiv_state_i)
            diagnostic_methods.append(method)

        # Update person state
        self.tested[selected] = True
        self.n_times_tested[selected] += 1
        self.test_result[selected] = test_positive
        self.diagnosed[selected[test_positive]] = True

        if self.pars.reset_flag:
            self.sought_care[selected] = False

        # Handle false negatives
        false_negative_uids = selected[~test_positive & has_tb]

        if len(false_negative_uids):
            unboosted = false_negative_uids[~self.multiplier_applied[false_negative_uids]]
            if len(unboosted):
                self.care_seeking_multiplier[unboosted] *= self.pars.care_seeking_multiplier
                self.multiplier_applied[unboosted] = True
            self.sought_care[false_negative_uids] = False
            self.tested[false_negative_uids] = False

        # Store for update_results
        self.tested_this_step = selected
        self.test_result_this_step = test_positive
        self.diagnostic_method_used = diagnostic_methods
        return

    def init_results(self):
        """Define result channels for test counts by outcome and diagnostic method."""
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
        return

    def update_results(self):
        """Record per-step and cumulative test counts by outcome and method."""
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
        return


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
    import tbsim
    import starsim as ss
    import matplotlib.pyplot as plt

    # Example simulation with enhanced diagnostic
    sim = ss.Sim(
        people=ss.People(n_agents=1000),
        diseases=tbsim.TB_LSHTM(pars={'init_prev': 0.25}),
        interventions=[
            tbsim.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),
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
