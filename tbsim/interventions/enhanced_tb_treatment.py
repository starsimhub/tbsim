"""
Enhanced TB Treatment for TBsim

This module implements enhanced TB treatment with drug types similar to EMOD-Generic's approach.
"""

import numpy as np
import starsim as ss
from tbsim import TBS
from .tb_drug_types import TBDrugType, TBDrugParameters, TBDrugTypeParameters
from typing import Dict, Optional

__all__ = ['EnhancedTBTreatment']


class EnhancedTBTreatment(ss.Intervention):
    """
    Enhanced TB treatment intervention that implements drug types similar to EMOD-Generic.
    """
    
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        # Default to DOTS treatment
        self.define_pars(
            drug_type=TBDrugType.DOTS,
            treatment_success_rate=0.85,
            reseek_multiplier=2.0,
            reset_flags=True,
        )
        self.update_pars(pars=pars, **kwargs)
        
        # Get drug parameters
        self.drug_parameters = TBDrugTypeParameters.create_parameters_for_type(self.pars.drug_type)
        
        # Storage for results
        self.new_treated = []
        self.successes = []
        self.failures = []
        self.drug_type_assignments = {}
    
    def step(self):
        """Execute one timestep of enhanced TB treatment."""
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
        
        # Start treatment
        started = tb.start_treatment(uids)
        
        # Treatment outcomes based on drug parameters
        tx_uids = uids[tb.on_treatment[uids]]
        ppl.n_times_treated[tx_uids] += 1
        
        # Use drug-specific success rate
        success_rate = self.drug_parameters.cure_rate
        rand = np.random.rand(len(tx_uids))
        success_uids = tx_uids[rand < success_rate]
        failure_uids = tx_uids[rand >= success_rate]
        
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
        
        # Store results
        self.new_treated = tx_uids
        self.successes = success_uids
        self.failures = failure_uids
    
    def init_results(self):
        """Initialize results tracking."""
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_treatment_success', dtype=int),
            ss.Result('n_treatment_failure', dtype=int),
            ss.Result('cum_treatment_success', dtype=int),
            ss.Result('cum_treatment_failure', dtype=int),
            ss.Result('drug_type_used', dtype=str, scale=False),  # Don't scale string results
        )
    
    def update_results(self):
        """Update results for this timestep."""
        n_treated = len(self.new_treated)
        n_success = len(self.successes)
        n_failure = len(self.failures)
        
        self.results['n_treated'][self.ti] = n_treated
        self.results['n_treatment_success'][self.ti] = n_success
        self.results['n_treatment_failure'][self.ti] = n_failure
        self.results['drug_type_used'][self.ti] = self.pars.drug_type.name
        
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


# Convenience functions
def create_dots_treatment(pars=None, **kwargs) -> EnhancedTBTreatment:
    """Create a DOTS treatment intervention."""
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.DOTS, **(pars or {})}, **kwargs)

def create_dots_improved_treatment(pars=None, **kwargs) -> EnhancedTBTreatment:
    """Create an improved DOTS treatment intervention."""
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.DOTS_IMPROVED, **(pars or {})}, **kwargs)

def create_first_line_treatment(pars=None, **kwargs) -> EnhancedTBTreatment:
    """Create a first-line combination treatment intervention."""
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.FIRST_LINE_COMBO, **(pars or {})}, **kwargs)
