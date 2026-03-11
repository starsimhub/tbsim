"""TB treatment intervention for diagnosed individuals"""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL
from .tb_drug_types import TBDrugType, TBDrugParameters, TBDrugTypeParameters

__all__ = ['TBTreatment']

# Active TB states in the LSHTM model
_ACTIVE_TB_STATES = [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]


class TBTreatment(ss.Intervention):
    """
    Starts TB treatment for diagnosed individuals and applies treatment success/failure logic.

    Requires ``HealthSeekingBehavior`` and ``TBDiagnostic`` to run upstream in the
    same simulation; see ``tbsim_examples/run_tb_interventions.py`` for a full example.

    Parameters:
        treatment_success_prob (float or Dist): Probability of cure if treated.
        reseek_multiplier (float): Care-seeking multiplier applied after failure.
        reset_flags (bool): Whether to reset tested/diagnosed flags after failure.
    """
    def __init__(self, pars=None, **kwargs):
        """Initialize with treatment success probability and failure-handling parameters."""
        super().__init__(**kwargs)
        self.define_pars(
            treatment_success_prob=0.85,
            reseek_multiplier=2.0,
            reset_flags=True,
        )
        self.update_pars(pars=pars, **kwargs)

        # Define person-level states needed by this intervention
        self.define_states(
            ss.BoolState('sought_care',            default=False),
            ss.BoolState('diagnosed',              default=False),
            ss.BoolArr('tested',                     default=False),
            ss.IntArr('n_times_treated',           default=0),
            ss.BoolState('tb_treatment_success',   default=False),
            ss.BoolArr('treatment_failure',          default=False),
            ss.FloatArr('care_seeking_multiplier', default=1.0),
            ss.BoolState('multiplier_applied',     default=False),
        )

        self.dist_treatment_success = ss.bernoulli(p=self.pars.treatment_success_prob)

        # Storage for results
        self.new_treated = []
        self.successes = []
        self.failures = []
        return

    def step(self):
        """Treat diagnosed active-TB individuals."""
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        # Select individuals diagnosed with TB and alive
        diagnosed_uids = (self.diagnosed & ppl.alive).uids
        active_tb_uids = np.where(np.isin(np.asarray(tb.state), _ACTIVE_TB_STATES))[0]
        uids = ss.uids(np.intersect1d(diagnosed_uids, active_tb_uids))

        if len(uids) == 0:
            return

        # Start treatment (moves active → TREATMENT state in TB_LSHTM)
        tb.start_treatment(uids)

        # Treatment outcomes
        tx_uids = uids[tb.on_treatment[uids]]
        self.n_times_treated[tx_uids] += 1
        success_uids, failure_uids = self.dist_treatment_success.filter(tx_uids, both=True)

        # Successful treatment clears infection
        tb.state[success_uids] = TBSL.CLEARED
        tb.on_treatment[success_uids] = False
        tb.susceptible[success_uids] = True
        tb.infected[success_uids] = False
        self.diagnosed[success_uids] = False
        self.tb_treatment_success[success_uids] = True

        # Update failure
        self.treatment_failure[failure_uids] = True

        if self.pars.reset_flags:
            self.diagnosed[failure_uids] = False
            self.tested[failure_uids] = False

        # Trigger renewed care-seeking for failures
        if len(failure_uids):
            self.sought_care[failure_uids] = False
            self.care_seeking_multiplier[failure_uids] *= self.pars.reseek_multiplier
            self.multiplier_applied[failure_uids] = True

        # Store
        self.new_treated = tx_uids
        self.successes = success_uids
        self.failures = failure_uids
        return

    def init_results(self):
        """Define result channels for treatment counts and outcomes."""
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_treatment_success', dtype=int),
            ss.Result('n_treatment_failure', dtype=int),
            ss.Result('cum_treatment_success', dtype=int),
            ss.Result('cum_treatment_failure', dtype=int),
        )
        return

    def update_results(self):
        """Record per-step and cumulative treatment success/failure counts."""
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
        return


__all__ += ['EnhancedTBTreatment']


class EnhancedTBTreatment(ss.Intervention):
    """
    Enhanced TB treatment intervention that implements configurable drug types and treatment protocols.

    Args:
        drug_type (TBDrugType): The type of drug regimen to use for treatment.
        treatment_success_prob (float, default 0.85): Base treatment success rate.
        reseek_multiplier (float, default 2.0): Multiplier applied to care-seeking probability after treatment failure.
        reset_flags (bool, default True): Whether to reset diagnosis and testing flags after treatment failure.
    """

    def __init__(self, pars=None, **kwargs):
        """Initialize the enhanced TB treatment intervention."""
        super().__init__(**kwargs)

        self.define_pars(
            drug_type=TBDrugType.DOTS,
            treatment_success_prob=0.85,
            reseek_multiplier=2.0,
            reset_flags=True,
        )
        self.update_pars(pars=pars, **kwargs)

        # Define person-level states needed by this intervention
        self.define_states(
            ss.BoolState('sought_care',            default=False),
            ss.BoolState('diagnosed',              default=False),
            ss.BoolArr('tested',                     default=False),
            ss.IntArr('n_times_treated',           default=0),
            ss.BoolState('tb_treatment_success',   default=False),
            ss.BoolArr('treatment_failure',          default=False),
            ss.FloatArr('care_seeking_multiplier', default=1.0),
            ss.BoolState('multiplier_applied',     default=False),
        )

        # Get drug parameters based on selected drug type
        self.drug_parameters = TBDrugTypeParameters.create_parameters_for_type(self.pars.drug_type)
        self.dist_treatment_success = ss.bernoulli(p=self.drug_parameters.cure_prob)

        # Storage for results tracking
        self.new_treated = []
        self.successes = []
        self.failures = []
        self.drug_type_assignments = {}
        return

    def step(self):
        """Execute one timestep of enhanced TB treatment."""
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        # Select individuals diagnosed with TB and alive
        diagnosed_uids = (self.diagnosed & ppl.alive).uids
        active_tb = np.isin(np.asarray(tb.state), _ACTIVE_TB_STATES)
        active_tb_uids = np.where(active_tb)[0]
        uids = ss.uids(np.intersect1d(diagnosed_uids, active_tb_uids))

        if len(uids) == 0:
            return

        # Start treatment (moves active → TREATMENT state in TB_LSHTM)
        tb.start_treatment(uids)

        # Treatment outcomes based on drug parameters
        tx_uids = uids[tb.on_treatment[uids]]
        self.n_times_treated[tx_uids] += 1

        success_uids, failure_uids = self.dist_treatment_success.filter(tx_uids, both=True)

        # Successful treatment clears infection
        tb.state[success_uids] = TBSL.CLEARED
        tb.on_treatment[success_uids] = False
        tb.susceptible[success_uids] = True
        tb.infected[success_uids] = False
        self.diagnosed[success_uids] = False
        self.tb_treatment_success[success_uids] = True

        # Update failed treatment outcomes
        self.treatment_failure[failure_uids] = True

        if self.pars.reset_flags:
            self.diagnosed[failure_uids] = False
            self.tested[failure_uids] = False

        # Trigger renewed care-seeking for treatment failures
        if len(failure_uids):
            self.sought_care[failure_uids] = False
            self.care_seeking_multiplier[failure_uids] *= self.pars.reseek_multiplier
            self.multiplier_applied[failure_uids] = True

        # Store results for current timestep
        self.new_treated = tx_uids
        self.successes = success_uids
        self.failures = failure_uids
        return

    def init_results(self):
        """Initialize results tracking for the intervention."""
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_treatment_success', dtype=int),
            ss.Result('n_treatment_failure', dtype=int),
            ss.Result('cum_treatment_success', dtype=int),
            ss.Result('cum_treatment_failure', dtype=int),
            ss.Result('drug_type_used', dtype=str, scale=False),
        )
        return

    def update_results(self):
        """Update results for the current timestep."""
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

        # Reset tracking lists for next timestep
        self.new_treated = []
        self.successes = []
        self.failures = []
        return


# Convenience functions for common treatment configurations

def create_dots_treatment(pars=None, **kwargs):
    """Create a standard DOTS treatment intervention."""
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.DOTS, **(pars or {})}, **kwargs)

def create_dots_improved_treatment(pars=None, **kwargs):
    """Create an improved DOTS treatment intervention."""
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.DOTS_IMPROVED, **(pars or {})}, **kwargs)

def create_first_line_treatment(pars=None, **kwargs):
    """Create a first-line combination treatment intervention."""
    return EnhancedTBTreatment(pars={'drug_type': TBDrugType.FIRST_LINE_COMBO, **(pars or {})}, **kwargs)
