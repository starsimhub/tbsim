import starsim as ss
import numpy as np

class RatesByAge:

    def __init__(self, unit, dt, override=None, use_globals=False):
        self.unit = unit
        self.dt = dt
        self.override = override
        self.use_globals = use_globals
        
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
        
        self.override_rates()

        self.RATES  = self.RATES = {key: self.arr(key) for key in self.RATES_DICT}
        self.AGE_CUTOFFS = self.generate_age_cutoffs()
        return
    
    def arr(self, name):
        arr = np.array(list(self.RATES_DICT[name].values()))
        return arr

    def override_rates(self):
        override = self.override
        
        if self.use_globals:
            for rate_name in self.RATES_DICT:
                self.RATES_DICT[rate_name] = {np.inf : self.RATES_DICT[rate_name][np.inf]} # Override all rates with the global value
            return   # Makes sure only default rates are used
            
        if override:
            for rate_name, value in override.items():
                if rate_name not in self.RATES_DICT:
                    raise ValueError(f"Rate '{rate_name}' is not recognized.")
                
                # If the value is a dictionary, validate, sort, and merge it with the existing rates
                if isinstance(value, dict):
                    # Sort the dictionary by age keys
                    sorted_value = {k: v for k, v in sorted(value.items())}
                    
                    for age, rate in sorted_value.items():
                        if isinstance(rate, (int, float)):
                            rate = ss.perday(rate, self.unit, self.dt)  # Convert to ss.perday
                        elif not isinstance(rate, ss.rate):
                            raise ValueError(f"Rate for age {age} in '{rate_name}' must be numeric or ss.rate.")
                        
                        # Update or add the new rate for the specified age
                        self.RATES_DICT[rate_name][age] = rate
                
                # If the value is a scalar and a key of np.inf is passed, replace the entire rate dictionary
                elif isinstance(value, (int, float)):
                    self.RATES_DICT[rate_name] = {np.inf: ss.perday(value, self.unit, self.dt)}
                
                # If the value is a scalar without np.inf key, override all rates for this rate name
                elif isinstance(value, ss.rate):
                    self.RATES_DICT[rate_name] = {np.inf: value}
                
                else:
                    raise ValueError(f"Value for '{rate_name}' must be a dictionary, scalar, or ss.rate.")
            
            # Ensure keys are sorted for consistency
            for rate_name in self.RATES_DICT:
                self.RATES_DICT[rate_name] = dict(sorted(self.RATES_DICT[rate_name].items()))
                        
    def generate_age_cutoffs(self):
            return {rate_name: np.array(sorted(rates.keys())) for rate_name, rates in self.RATES_DICT.items()}
        
    def print_all(self):
        import pprint
        pprint.pprint(self.RATES_DICT)
        pprint.pprint(self.RATES)
        return
    
    def save(self, filename):
        import json
        with open(filename, 'w') as f:
            json.dump(self.RATES_DICT, f)
            json.dump(self.AGE_CUTOFFS, f)
        return
    
