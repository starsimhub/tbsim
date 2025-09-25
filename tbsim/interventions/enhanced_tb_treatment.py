"""
Enhanced TB Treatment for TBsim

This module implements enhanced TB treatment with configurable drug types and treatment protocols.

It provides a flexible framework for modeling different TB treatment strategies including DOTS,
improved DOTS, and first-line combination therapies.

The module extends the base Starsim intervention system to provide TB-specific treatment
functionality with configurable success rates, drug parameters, and outcome tracking.
"""

import numpy as np
import starsim as ss
from tbsim import TBS
from .tb_drug_types import TBDrugType, TBDrugParameters, TBDrugTypeParameters
from typing import Dict, Optional

__all__ = ['EnhancedTBTreatment']


class EnhancedTBTreatment(ss.Intervention):
    """
    Enhanced TB treatment intervention that implements configurable drug types and treatment protocols.
    
    This intervention manages the treatment of diagnosed TB cases by:
    - Selecting eligible individuals for treatment
    - Applying drug-specific treatment protocols
    - Tracking treatment outcomes (success/failure)
    - Managing post-treatment care-seeking behavior
    - Recording comprehensive treatment statistics
    
    Parameters
    ----------
    drug_type : TBDrugType
        The type of drug regimen to use for treatment. Options include:
        - DOTS: Standard DOTS protocol
        - DOTS_IMPROVED: Enhanced DOTS with better adherence
        - FIRST_LINE_COMBO: First-line combination therapy
    treatment_success_rate : float, default 0.85
        Base treatment success rate (can be overridden by drug-specific parameters)
    reseek_multiplier : float, default 2.0
        Multiplier applied to care-seeking probability after treatment failure
    reset_flags : bool, default True
        Whether to reset diagnosis and testing flags after treatment failure
        
    Attributes
    ----------
    drug_parameters : TBDrugTypeParameters
        Drug-specific parameters including cure rates and side effects
    new_treated : list
        List of UIDs of individuals who started treatment in the current timestep
    successes : list
        List of UIDs of individuals who successfully completed treatment
    failures : list
        List of UIDs of individuals who failed treatment
    drug_type_assignments : dict
        Mapping of individual UIDs to assigned drug types
        
    Examples
    --------
    Create a standard DOTS treatment intervention:
    
    >>> treatment = EnhancedTBTreatment(drug_type=TBDrugType.DOTS)
    
    Create a first-line combination treatment with custom parameters:
    
    >>> treatment = EnhancedTBTreatment(
    ...     drug_type=TBDrugType.FIRST_LINE_COMBO,
    ...     reseek_multiplier=3.0,
    ...     reset_flags=False
    ... )
    
    Track treatment outcomes over time:
    
    >>> treatment = EnhancedTBTreatment()
    >>> # After running simulation
    >>> cumulative_success = treatment.results['cum_treatment_success']
    >>> treatment_failures = treatment.results['n_treatment_failure']
    """
    
    def __init__(self, pars=None, **kwargs):
        """
        Initialize the enhanced TB treatment intervention.
        
        Parameters
        ----------
        pars : dict, optional
            Dictionary of parameters to override defaults
        **kwargs
            Additional keyword arguments passed to parent class
        """
        super().__init__(**kwargs)
        
        # Default to DOTS treatment
        self.define_pars(
            drug_type=TBDrugType.DOTS,
            treatment_success_rate=0.85,
            reseek_multiplier=2.0,
            reset_flags=True,
        )
        self.update_pars(pars=pars, **kwargs)
        
        # Get drug parameters based on selected drug type
        self.drug_parameters = TBDrugTypeParameters.create_parameters_for_type(self.pars.drug_type)
        
        # Storage for results tracking
        self.new_treated = []
        self.successes = []
        self.failures = []
        self.drug_type_assignments = {}
    
    def step(self):
        """
        Execute one timestep of enhanced TB treatment.
        
        This method performs the following operations:
        1. Identifies eligible individuals (diagnosed with active TB)
        2. Initiates treatment for eligible cases
        3. Determines treatment outcomes based on drug-specific success rates
        4. Updates individual states and flags based on outcomes
        5. Manages post-treatment care-seeking behavior
        6. Records treatment statistics for analysis
        
        The treatment process follows standard TB treatment protocols where:
        - Successful treatment clears TB infection and restores susceptibility
        - Failed treatment triggers renewed care-seeking with increased probability
        - Treatment history is tracked for epidemiological analysis
        """
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb
        
        # Select individuals diagnosed with TB and alive
        diagnosed_uids = (ppl.diagnosed & ppl.alive).uids
        active_tb = np.isin(tb.state, [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
        active_tb_uids = np.where(active_tb)[0]
        # Find intersection of diagnosed and active TB
        uids = np.intersect1d(diagnosed_uids, active_tb_uids)
        
        if len(uids) == 0:
            return
        
        # Start treatment for eligible individuals
        started = tb.start_treatment(uids)
        
        # Treatment outcomes based on drug parameters
        tx_uids = uids[tb.on_treatment[uids]]
        ppl.n_times_treated[tx_uids] += 1
        
        # Use drug-specific success rate for outcome determination
        success_rate = self.drug_parameters.cure_rate
        rand = np.random.rand(len(tx_uids))
        success_uids = tx_uids[rand < success_rate]
        failure_uids = tx_uids[rand >= success_rate]
        
        # Update successful treatment outcomes
        # Clear TB infection and restore susceptibility
        tb.state[success_uids] = TBS.NONE
        tb.on_treatment[success_uids] = False
        tb.susceptible[success_uids] = True
        tb.infected[success_uids] = False
        tb.active_tb_state[success_uids] = TBS.NONE
        tb.ti_active[success_uids] = np.nan
        ppl.diagnosed[success_uids] = False
        ppl.tb_treatment_success[success_uids] = True
        
        # Update failed treatment outcomes
        ppl.treatment_failure[failure_uids] = True
        
        # Reset diagnosis and testing flags for failures if configured
        if self.pars.reset_flags:
            ppl.diagnosed[failure_uids] = False
            ppl.tested[failure_uids] = False
        
        # Trigger renewed care-seeking for treatment failures
        # This models the increased likelihood of seeking care after treatment failure
        if len(failure_uids):
            ppl.sought_care[failure_uids] = False
            ppl.care_seeking_multiplier[failure_uids] *= self.pars.reseek_multiplier
            ppl.multiplier_applied[failure_uids] = True
        
        # Store results for current timestep
        self.new_treated = tx_uids
        self.successes = success_uids
        self.failures = failure_uids
    
    def init_results(self):
        """
        Initialize results tracking for the intervention.
        
        This method sets up the data structures needed to track:
        - Number of individuals treated per timestep
        - Treatment success and failure counts
        - Cumulative treatment outcomes over time
        - Drug type used in each timestep
        
        The results are designed to facilitate epidemiological analysis and
        intervention effectiveness evaluation.
        """
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_treatment_success', dtype=int),
            ss.Result('n_treatment_failure', dtype=int),
            ss.Result('cum_treatment_success', dtype=int),
            ss.Result('cum_treatment_failure', dtype=int),
            ss.Result('drug_type_used', dtype=str, scale=False),  # Don't scale string results
        )
    
    def update_results(self):
        """
        Update results for the current timestep.
        
        This method records the treatment outcomes from the current timestep
        and maintains running totals for cumulative analysis. It processes:
        - Current timestep treatment counts
        - Success and failure outcomes
        - Drug type information
        - Cumulative totals for epidemiological analysis
        
        The results are stored in the intervention's results dictionary and
        can be accessed for post-simulation analysis and visualization.
        """
        n_treated = len(self.new_treated)
        n_success = len(self.successes)
        n_failure = len(self.failures)
        
        # Record current timestep results
        self.results['n_treated'][self.ti] = n_treated
        self.results['n_treatment_success'][self.ti] = n_success
        self.results['n_treatment_failure'][self.ti] = n_failure
        self.results['drug_type_used'][self.ti] = self.pars.drug_type.name
        
        # Update cumulative totals
        if self.ti > 0:
            self.results['cum_treatment_success'][self.ti] = self.results['cum_treatment_success'][self.ti - 1] + n_success
            self.results['cum_treatment_failure'][self.ti] = self.results['cum_treatment_failure'][self.ti - 1] + n_failure
        else:
            self.results['cum_treatment_success'][self.ti] = n_success
            self.results['cum_treatment_failure'][self.ti] = n_failure
        
        # Reset tracking lists for next timestep
        self.new_treated = []
        self.successes = []
        self.failures = []


# Convenience functions for common treatment configurations

def create_dots_treatment(pars=None, **kwargs) -> EnhancedTBTreatment:
    """
    Create a standard DOTS treatment intervention.
    
    This function provides a convenient way to create a DOTS treatment
    intervention with standard parameters. DOTS (Directly Observed Treatment,
    Short-course) is the WHO-recommended strategy for TB control.
    
    Parameters
    ----------
    pars : dict, optional
        Additional parameters to override defaults
    **kwargs
        Additional keyword arguments passed to EnhancedTBTreatment
        
    Returns
    -------
    EnhancedTBTreatment
        Configured DOTS treatment intervention
        
    Examples
    --------
    >>> dots_treatment = create_dots_treatment()
    >>> dots_treatment.pars.drug_type
    TBDrugType.DOTS
    """
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.DOTS, **(pars or {})}, **kwargs)

def create_dots_improved_treatment(pars=None, **kwargs) -> EnhancedTBTreatment:
    """
    Create an improved DOTS treatment intervention.
    
    This function creates a DOTS treatment intervention with enhanced
    parameters that may include better adherence monitoring, improved
    drug formulations, or enhanced patient support systems.
    
    Parameters
    ----------
    pars : dict, optional
        Additional parameters to override defaults
    **kwargs
        Additional keyword arguments passed to EnhancedTBTreatment
        
    Returns
    -------
    EnhancedTBTreatment
        Configured improved DOTS treatment intervention
        
    Examples
    --------
    >>> improved_dots = create_dots_improved_treatment(
    ...     reseek_multiplier=2.5,
    ...     reset_flags=False
    ... )
    """
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.DOTS_IMPROVED, **(pars or {})}, **kwargs)

def create_first_line_treatment(pars=None, **kwargs) -> EnhancedTBTreatment:
    """
    Create a first-line combination treatment intervention.
    
    This function creates a treatment intervention using first-line
    combination therapy, which typically includes multiple drugs
    to prevent resistance development and improve treatment outcomes.
    
    Parameters
    ----------
    pars : dict, optional
        Additional parameters to override defaults
    **kwargs
        Additional keyword arguments passed to EnhancedTBTreatment
        
    Returns
    -------
    EnhancedTBTreatment
        Configured first-line combination treatment intervention
        
    Examples
    --------
    >>> first_line = create_first_line_treatment(
    ...     treatment_success_rate=0.90,
    ...     reseek_multiplier=1.5
    ... )
    """
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.FIRST_LINE_COMBO, **(pars or {})}, **kwargs)
