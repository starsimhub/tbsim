__all__ = ['TBDiagnosticErrors']


class TBDiagnosticErrors:
    """Error messages and validation for TBDiagnostic intervention"""
    
    # Required attributes error messages
    SOUGHT_CARE_MISSING = (
        "TBDiagnostic requires 'sought_care' attribute on people object. "
        "This is typically provided by a HealthSeekingBehavior intervention. "
        "Please add a health seeking behavior intervention to your simulation."
    )
    
    DIAGNOSED_MISSING = (
        "TBDiagnostic requires 'diagnosed' attribute on people object. "
        "This should be automatically created by the TB disease module. "
        "Please ensure TB disease is properly configured in your simulation."
    )
    
    ALIVE_MISSING = (
        "TBDiagnostic requires 'alive' attribute on people object. "
        "This should be automatically created by the simulation framework. "
        "Please check your simulation setup."
    )
    
    TESTED_MISSING = (
        "TBDiagnostic requires 'tested' attribute on people object. "
        "This should be automatically created by the TB disease module. "
        "Please ensure TB disease is properly configured in your simulation."
    )
    
    N_TIMES_TESTED_MISSING = (
        "TBDiagnostic requires 'n_times_tested' attribute on people object. "
        "This should be automatically created by the TB disease module. "
        "Please ensure TB disease is properly configured in your simulation."
    )
    
    TEST_RESULT_MISSING = (
        "TBDiagnostic requires 'test_result' attribute on people object. "
        "This should be automatically created by the TB disease module. "
        "Please ensure TB disease is properly configured in your simulation."
    )
    
    CARE_SEEKING_MULTIPLIER_MISSING = (
        "TBDiagnostic requires 'care_seeking_multiplier' attribute on people object. "
        "This should be automatically created by the TB disease module. "
        "Please ensure TB disease is properly configured in your simulation."
    )
    
    MULTIPLIER_APPLIED_MISSING = (
        "TBDiagnostic requires 'multiplier_applied' attribute on people object. "
        "This should be automatically created by the TB disease module. "
        "Please ensure TB disease is properly configured in your simulation."
    )
    
    # Simulation structure error messages
    DISEASES_MISSING = (
        "TBDiagnostic requires diseases to be present in simulation. "
        "Please add TB disease to your simulation: sim = ss.Sim(diseases=[tb], ...)"
    )
    
    TB_DISEASE_MISSING = (
        "TBDiagnostic requires TB disease module. "
        "Please add TB disease to your simulation: sim = ss.Sim(diseases=[tb], ...)"
    )
    
    TB_STATE_MISSING = (
        "TBDiagnostic requires TB disease to have 'state' attribute. "
        "Please ensure TB disease is properly initialized in your simulation."
    )
    
    # Parameter validation error messages
    @staticmethod
    def coverage_invalid(value):
        return (
            f"TBDiagnostic coverage parameter must be between 0.0 and 1.0, "
            f"got {value}. Please set a valid coverage value."
        )
    
    @staticmethod
    def sensitivity_invalid(value):
        return (
            f"TBDiagnostic sensitivity parameter must be between 0.0 and 1.0, "
            f"got {value}. Please set a valid sensitivity value."
        )
    
    @staticmethod
    def specificity_invalid(value):
        return (
            f"TBDiagnostic specificity parameter must be between 0.0 and 1.0, "
            f"got {value}. Please set a valid specificity value."
        )
    
    @staticmethod
    def care_seeking_multiplier_invalid(value):
        return (
            f"TBDiagnostic care_seeking_multiplier must be >= 1.0, "
            f"got {value}. Please set a multiplier >= 1.0 to increase care-seeking probability."
        )
    
    # Coverage filter error messages
    @staticmethod
    def coverage_filter_failed(error, coverage_value):
        return (
            f"TBDiagnostic failed to apply coverage filter: {error}. "
            f"Coverage parameter value: {coverage_value}. "
            f"Please check that coverage is a valid probability or distribution."
        )
    
    @staticmethod
    def coverage_filter_invalid_selection(eligible_count, selected_count):
        return (
            f"TBDiagnostic coverage filter returned more people than eligible. "
            f"Eligible: {eligible_count}, Selected: {selected_count}. "
            f"This indicates a problem with the coverage filter implementation."
        )
    
    @staticmethod
    def coverage_filter_wrong_selection():
        return (
            f"TBDiagnostic coverage filter selected people not in eligible list. "
            f"This indicates a problem with the coverage filter implementation."
        )
    
    # TB status error messages
    @staticmethod
    def tb_status_failed(error):
        return (
            f"TBDiagnostic failed to determine TB status: {error}. "
            f"Please ensure TB disease module is properly initialized and has 'state' attribute."
        )
    
    @staticmethod
    def tb_state_length_mismatch(selected_count, tb_states_count):
        return (
            f"TBDiagnostic TB state length mismatch. "
            f"Selected people: {selected_count}, TB states: {tb_states_count}. "
            f"This indicates a problem with TB disease state tracking."
        )
    
    # Test application error messages
    @staticmethod
    def sensitivity_test_failed(error, sensitivity):
        return (
            f"TBDiagnostic failed to apply sensitivity test to TB cases: {error}. "
            f"Sensitivity parameter: {sensitivity}. "
            f"Please check that sensitivity is a valid probability between 0.0 and 1.0."
        )
    
    @staticmethod
    def specificity_test_failed(error, specificity):
        return (
            f"TBDiagnostic failed to apply specificity test to non-TB cases: {error}. "
            f"Specificity parameter: {specificity}. "
            f"Please check that specificity is a valid probability between 0.0 and 1.0."
        )
    
    @staticmethod
    def test_result_length_mismatch(selected_count, test_results_count):
        return (
            f"TBDiagnostic test result length mismatch. "
            f"Selected people: {selected_count}, Test results: {test_results_count}. "
            f"This indicates a problem with test application logic."
        )
    
    @staticmethod
    def test_result_wrong_type(dtype):
        return (
            f"TBDiagnostic test results must be boolean array, "
            f"got {dtype}. This indicates a problem with test application logic."
        )
    
    # Person state update error messages
    @staticmethod
    def person_state_update_failed(error):
        return (
            f"TBDiagnostic failed to update person states: {error}. "
            f"Please ensure all required person attributes are properly initialized. "
            f"Required attributes: tested, n_times_tested, test_result, diagnosed."
        )
    
    @staticmethod
    def sought_care_reset_failed(error):
        return (
            f"TBDiagnostic failed to reset sought_care flag: {error}. "
            f"Please ensure 'sought_care' attribute is properly initialized."
        )
    
    @staticmethod
    def multiplier_filter_failed(error):
        return (
            f"TBDiagnostic failed to filter unboosted false negatives: {error}. "
            f"Please ensure 'multiplier_applied' attribute is properly initialized."
        )
    
    @staticmethod
    def multiplier_application_failed(error):
        return (
            f"TBDiagnostic failed to apply care-seeking multiplier: {error}. "
            f"Please ensure 'care_seeking_multiplier' and 'multiplier_applied' attributes are properly initialized."
        )
    
    @staticmethod
    def false_negative_reset_failed(error):
        return (
            f"TBDiagnostic failed to reset flags for false negatives: {error}. "
            f"Please ensure 'sought_care' and 'tested' attributes are properly initialized."
        )
    
    # False negative logic validation
    @staticmethod
    def false_negative_count_invalid(selected_count, false_negative_count):
        return (
            f"TBDiagnostic false negative count exceeds selected people. "
            f"Selected: {selected_count}, False negatives: {false_negative_count}. "
            f"This indicates a problem with false negative identification logic."
        )
    
    @staticmethod
    def unboosted_count_invalid(false_negative_count, unboosted_count):
        return (
            f"TBDiagnostic unboosted count exceeds false negatives. "
            f"False negatives: {false_negative_count}, Unboosted: {unboosted_count}. "
            f"This indicates a problem with multiplier application logic."
        )
    
    # Results initialization error messages
    @staticmethod
    def results_init_failed(error):
        return (
            f"TBDiagnostic failed to initialize results: {error}. "
            f"Please ensure the intervention is properly added to the simulation and "
            f"that the simulation framework is correctly initialized."
        )
    
    RESULTS_ATTR_MISSING = (
        "TBDiagnostic results attribute not created. "
        "This indicates a problem with result initialization in the simulation framework."
    )
    
    @staticmethod
    def result_missing(result_name, available_results):
        return (
            f"TBDiagnostic missing result '{result_name}'. "
            f"Available results: {available_results}. "
            f"This indicates a problem with result definition."
        )
    
    # Results update error messages
    RESULTS_NOT_INITIALIZED = (
        "TBDiagnostic results not initialized. "
        "Please ensure init_results() was called before update_results()."
    )
    
    TI_MISSING = (
        "TBDiagnostic timestep index (ti) not available. "
        "This indicates a problem with the simulation framework."
    )
    
    TESTED_THIS_STEP_MISSING = (
        "TBDiagnostic tested_this_step not initialized. "
        "This indicates a problem with intervention initialization."
    )
    
    TEST_RESULT_THIS_STEP_MISSING = (
        "TBDiagnostic test_result_this_step not initialized. "
        "This indicates a problem with intervention initialization."
    )
    
    @staticmethod
    def data_inconsistency(tested_length, test_result_length):
        return (
            f"TBDiagnostic data inconsistency: "
            f"tested_this_step length ({tested_length}) != "
            f"test_result_this_step length ({test_result_length}). "
            f"This indicates a problem with the step() method."
        )
    
    @staticmethod
    def result_calculation_failed(error):
        return (
            f"TBDiagnostic failed to calculate result counts: {error}. "
            f"Please check that tested_this_step and test_result_this_step are properly set."
        )
    
    @staticmethod
    def negative_tested_count(count):
        return (
            f"TBDiagnostic negative tested count: {count}. "
            f"This indicates a problem with result calculation."
        )
    
    @staticmethod
    def negative_positive_count(count):
        return (
            f"TBDiagnostic negative positive count: {count}. "
            f"This indicates a problem with result calculation."
        )
    
    @staticmethod
    def negative_negative_count(count):
        return (
            f"TBDiagnostic negative negative count: {count}. "
            f"This indicates a problem with result calculation."
        )
    
    @staticmethod
    def result_count_mismatch(positive_count, negative_count, tested_count):
        return (
            f"TBDiagnostic result count mismatch: "
            f"positive ({positive_count}) + negative ({negative_count}) != tested ({tested_count}). "
            f"This indicates a problem with result calculation."
        )
    
    @staticmethod
    def current_timestep_record_failed(error):
        return (
            f"TBDiagnostic failed to record current timestep results: {error}. "
            f"Please ensure results are properly initialized and ti is valid."
        )
    
    @staticmethod
    def cumulative_update_failed(error):
        return (
            f"TBDiagnostic failed to update cumulative totals: {error}. "
            f"Please ensure results are properly initialized and ti is valid."
        )
    
    @staticmethod
    def cumulative_positive_decreased(previous, current):
        return (
            f"TBDiagnostic cumulative positive decreased: "
            f"previous {previous} -> current {current}. "
            f"This indicates a problem with cumulative calculation."
        )
    
    @staticmethod
    def cumulative_negative_decreased(previous, current):
        return (
            f"TBDiagnostic cumulative negative decreased: "
            f"previous {previous} -> current {current}. "
            f"This indicates a problem with cumulative calculation."
        )
