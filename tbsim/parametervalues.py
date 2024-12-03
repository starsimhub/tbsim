import starsim as ss
import numpy as np

class RatesByAge:

    def __init__(self, unit, dt, override=None):
        self.unit = unit
        self.dt = dt
        self.RATES_DICT = {
            'rate_LS_to_presym': {
                0: ss.perday(3e-5, unit, dt),   
                15: ss.perday(2.0548e-6, unit, dt),  
                25: ss.perday(3e-5, unit, dt),
                np.inf: ss.perday(3e-5, unit, dt),
            },
            'rate_LF_to_presym': {
                0: ss.perday(6e-3, unit, dt),
                15: ss.perday(4.5e-3, unit, dt),
                25: ss.perday(6e-3, unit, dt),
                np.inf: ss.perday(6e-3, unit, dt),
            },
            'rate_presym_to_active': {
                0: ss.perday(3e-2, unit, dt),
                15: ss.perday(5.48e-3, unit, dt),
                25: ss.perday(3e-2, unit, dt),
                np.inf: ss.perday(6e-3, unit, dt),
            },
            'rate_active_to_clear': {
                0: ss.perday(2.4e-4, unit, dt),
                15: ss.perday(2.74e-4, unit, dt),
                25: ss.perday(2.4e-4, unit, dt),
                np.inf: ss.perday(2.4e-4, unit, dt),
            },
            'rate_smpos_to_dead': {
                0: ss.perday(4.5e-4, unit, dt),
                15: ss.perday(6.85e-4, unit, dt),
                25: ss.perday(4.5e-4, unit, dt),
                np.inf: ss.perday(4.5e-4, unit, dt),
            },
            'rate_smneg_to_dead': {
                0: ss.perday(0.3 * 4.5e-4, unit, dt),
                15: ss.perday(2.74e-4, unit, dt),
                25: ss.perday(0.3 * 4.5e-4, unit, dt),
                np.inf: ss.perday(0.3 * 4.5e-4, unit, dt),
            },
            'rate_exptb_to_dead': {
                0: ss.perday(0.15 * 4.5e-4, unit, dt),
                15: ss.perday(2.74e-4, unit, dt),
                25: ss.perday(0.15 * 4.5e-4, unit, dt),
                np.inf: ss.perday(0.15 * 4.5e-4, unit, dt),
            },
            'rate_treatment_to_clear': {
                0: ss.peryear(12/2, unit, dt),
                15: ss.peryear(2, unit, dt),
                25: ss.peryear(12/2, unit, dt),
                np.inf: ss.perday(12/2, unit, dt),
            },
        }
        
        self.override_rates(override)

        self.RATES  = self.RATES = {key: self.arr(key) for key in self.RATES_DICT}
        self.AGE_CUTOFFS = np.array([ 0, 15, 25, np.inf])
            
    def arr(self, name):
        arr = np.array(list(self.RATES_DICT[name].values()))
        return arr
    
    def override_rates(self, override):
        if override:
            for rate_name, value in override.items():
                if rate_name not in self.RATES_DICT:
                    raise ValueError(f"Rate '{rate_name}' is not recognized.")

                if isinstance(value, dict):
                    # Validate and update age-specific rates
                    if set(value.keys()) != set(self.RATES_DICT[rate_name].keys()):
                        raise ValueError(f"Age values for '{rate_name}' must match {list(self.RATES_DICT[rate_name].keys())}.")
                    
                    for age, rate in value.items():
                        if isinstance(rate, (int, float)):
                            rate = ss.perday(rate, self.unit, self.dt)  # Convert numeric to ss.perday
                        elif not isinstance(rate, ss.rate):
                            raise ValueError(f"Rate for age {age} in '{rate_name}' must be a numeric value or ss.rate.")
                        self.RATES_DICT[rate_name][age] = rate
                else:
                    # Validate and update global rate
                    if isinstance(value, (int, float)):
                        value = ss.perday(value, self.unit, self.dt)  # Convert numeric to ss.perday
                    elif not isinstance(value, ss.rate):
                        raise ValueError(f"Global value for '{rate_name}' must be a numeric value or ss.rate.")
                    
                    # Apply the same global rate to all age groups
                    for age in self.RATES_DICT[rate_name]:
                        self.RATES_DICT[rate_name][age] = value
