import numpy as np
import starsim as ss
from tbsim.utils.probabilities import Probability
from tbsim.wrappers import Agents


__all__ = ['BCGProtection']

class BCGProtection(ss.Intervention):
    """
    Simulates BCG-like vaccination for tuberculosis prevention in children under a specified age.

    This intervention identifies children below a configurable age limit who have not yet 
    been vaccinated. At each timestep, a proportion of these eligible individuals are 
    selected based on the `coverage` parameter to receive simulated BCG protection.

    Once vaccinated, individuals are considered protected for a fixed number of years 
    (`duration`). While protected, their TB-related risk modifiers — activation, clearance, 
    and death — are adjusted using scaled and sampled values from a BCG-specific probability 
    model (`BCGProb`).

    Parameters:
        pars (dict, optional): Dictionary of parameters. Supported keys:
            - 'coverage' (float): Fraction of eligible individuals vaccinated per timestep (default: 0.9).
            - 'start' (int): Year when the intervention starts (default: 1900).
            - 'stop' (int): Year when the intervention stops (default: 2100).
            - 'efficacy' (float): Scaling factor applied to sampled risk modifiers (default: 0.8).
            - 'duration' (int): Duration (in years) for which BCG protection remains effective (default: 10).
            - 'age_limit' (int): Maximum age (in years) to be considered eligible for vaccination (default: 5).
            - 'prob_file' (str): Optional path to a JSON or CSV file defining probability distributions.

    Attributes:
        bcg_probs (BCGProb): Instance that stores and samples from the defined probability ranges.
        vaccinated (ss.State): Boolean array indicating vaccination status.
        ti_bcgvaccinated (ss.State): Array storing the timestep at which individuals were vaccinated.
        n_eligible (int): Number of individuals eligible for vaccination in the current step.
        eligible (np.ndarray): Boolean mask of currently eligible individuals.

    States:
        vaccinated (bool): Indicates whether an individual has received the BCG vaccine.
        ti_bcgvaccinated (float): Timestep at which the individual was vaccinated.

    Methods:
        check_eligibility(): Identify and randomly select eligible individuals for vaccination.
        is_protected(uids, current_time): Return boolean mask indicating protected individuals.
        step(): Apply BCG protection and adjust TB risk modifiers accordingly.
        init_results(): Define simulation result metrics.
        update_results(): Record the number of vaccinated and eligible individuals each timestep.

    Notes:
        This intervention assumes the presence of a TB disease model attached to the simulation 
        and modifies its rr_activation, rr_clearance, and rr_death arrays.
    """

    def __init__(self, pars={}, **kwargs):
        super().__init__(**kwargs)
        self.coverage = pars.get('coverage', 0.6)
        self.start = pars.get('start', 1900)            
        self.stop = pars.get('stop', 2100)
        self.efficacy = pars.get('efficacy', 0.8)      # BCGProb of protection
        self.duration = pars.get('duration', 10)       # Duration of protection in years
        self.age_limit = pars.get('age_limit', 5)      # Max age for eligibility
        self.n_eligible = 0
        self.eligible = []
        self.probs = BCGProb()

        self.define_states(
            ss.BoolArr('vaccinated', default=False),
            ss.FloatArr('ti_bcgvaccinated'), 
        )
        print(self.pars)
        
    def check_eligibility(self):
        ages = self.sim.people.age
        under_age = ages <= self.age_limit

        eligible = under_age & ~self.vaccinated
        eligible_uids = np.where(eligible)[0]
        n_to_vaccinate = int(len(eligible_uids) * self.coverage)
        if n_to_vaccinate > 0:
            chosen = np.random.choice(eligible_uids, size=n_to_vaccinate, replace=False)
            self.eligible = np.zeros_like(eligible)
            self.eligible[chosen] = True
        else:
            chosen = np.array([], dtype=int)
            self.eligible = np.zeros_like(eligible)
        self.n_eligible = len(chosen)
        return ss.uids(chosen)

    def is_protected(self, uids, current_time):
        """Return boolean array: True if still protected (within duration), else False."""
        return (self.vaccinated[uids]) & ((current_time - self.ti_bcgvaccinated[uids]) <= self.duration)

    def step(self):
        # Check if now is the right time to vaccinate
        if self.sim.now < self.start or self.sim.now > self.stop:
            return
        
        current_time = self.ti  # Assuming sim.t is in years
        eligible = self.check_eligibility()
        if len(eligible) == 0:
            return
        # Vaccinate
        self.vaccinated[eligible] = True
        self.ti_bcgvaccinated[eligible] = self.ti
        tb = self.sim.diseases.tb
        # Only apply effect to those who are protected (within duration)
        protected = self.is_protected(eligible, current_time)
        protected_uids = eligible[protected]
        if len(protected_uids) > 0:
            tb.rr_activation[protected_uids] *= self.efficacy * self.probs.activation()
            tb.rr_clearance[protected_uids] *= self.probs.clearance()
            tb.rr_death[protected_uids] *= self.probs.death()

    def init_results(self):
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )

    def update_results(self):
        self.results['n_vaccinated'][self.ti] = np.count_nonzero(self.vaccinated)
        self.results['n_eligible'][self.ti] = self.n_eligible


class BCGProb(Probability):
    """
    Specialized `Probability` class for BCG-related tuberculosis risk modifiers.

    This class predefines three common parameters relevant to BCG vaccination impact
    on tuberculosis outcomes:

    - **activation**: Modifier for TB activation risk (default: uniform between 0.5 and 0.65).
    - **clearance**: Modifier for bacterial clearance probability (default: uniform between 1.3 and 1.5).
    - **death**: Modifier for TB-related mortality risk (default: uniform between 0.05 and 0.15).

    Additional distributions may be loaded or overridden via JSON or CSV.

    Parameters
    ----------
    from_file : str, optional
        Path to a `.json` or `.csv` file containing custom probability definitions.
        If provided, the file is used to override or supplement the default values.

    Attributes
    ----------
    values : RangeDict
        Container for all named `Range` objects, supporting both dict- and dot-access.

    Methods
    -------
    activation(size=None)
        Sample from the activation modifier distribution.

    clearance(size=None)
        Sample from the clearance modifier distribution.

    death(size=None)
        Sample from the TB mortality modifier distribution.

    from_json(filename)
        Load distributions from a JSON file with structure: {name: {min, max, dist}}.

    from_csv(filename)
        Load distributions from a CSV file with columns: name, min, max [,dist].

    from_dict(data)
        Load distributions from a dictionary (same structure as JSON).

    sample(name, size=None)
        Sample from any named distribution using the specified distribution type.

    to_json(filename)
        Export all current distributions to a JSON file.

    to_csv(filename)
        Export all current distributions to a CSV file.
    """

    def __init__(self, from_file=None):
        super().__init__()
        self.from_dict({
            "activation": {"min": 0.5, "max": 0.65},
            "clearance": {"min": 1.3, "max": 1.5},
            "death": {"min": 0.05, "max": 0.15}
        })
        if from_file:
            if from_file.endswith('.json'):
                self.from_json(from_file)
            elif from_file.endswith('.csv'):
                self.from_csv(from_file)
            else:
                raise ValueError("Unsupported file format. Use .json or .csv.")

    def activation(self, size=None):
        """Sample from the activation risk modifier distribution."""
        return self.sample("activation", size)

    def clearance(self, size=None):
        """Sample from the clearance modifier distribution."""
        return self.sample("clearance", size)

    def death(self, size=None):
        """Sample from the TB mortality modifier distribution."""
        return self.sample("death", size)
