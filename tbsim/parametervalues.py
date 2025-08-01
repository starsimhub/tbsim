import starsim as ss
import numpy as np

class RatesByAge:

    def __init__(self, unit, dt, override=None, use_globals=False):
        self.unit = unit
        self.dt = dt
        self.override = override
        self.use_globals = use_globals
        
        self.rates_dict = {
            'rate_LS_to_presym': {
                0: ss.perday(3e-5),   
                15: ss.perday(2.0548e-6),  
                25: ss.perday(3e-5),
                np.inf: ss.perday(3e-5),
            },
            'rate_LF_to_presym': {
                0: ss.perday(6e-3),
                15: ss.perday(4.5e-3),
                25: ss.perday(6e-3),
                np.inf: ss.perday(6e-3),
            },
            'rate_presym_to_active': {
                0: ss.perday(3e-2),
                15: ss.perday(5.48e-3),
                25: ss.perday(3e-2),
                np.inf: ss.perday(6e-3),
            },
            'rate_active_to_clear': {
                0: ss.perday(2.4e-4),
                15: ss.perday(2.74e-4),
                25: ss.perday(2.4e-4),
                np.inf: ss.perday(2.4e-4),
            },
            'rate_smpos_to_dead': {
                0: ss.perday(4.5e-4),
                15: ss.perday(6.85e-4),
                25: ss.perday(4.5e-4),
                np.inf: ss.perday(4.5e-4),
            },
            'rate_smneg_to_dead': {
                0: ss.perday(0.3 * 4.5e-4),
                15: ss.perday(2.74e-4),
                25: ss.perday(0.3 * 4.5e-4),
                np.inf: ss.perday(0.3 * 4.5e-4),
            },
            'rate_exptb_to_dead': {
                0: ss.perday(0.15 * 4.5e-4),
                15: ss.perday(2.74e-4),
                25: ss.perday(0.15 * 4.5e-4),
                np.inf: ss.perday(0.15 * 4.5e-4),
            },
            'rate_treatment_to_clear': {
                0: ss.rateperyear(12/2),
                15: ss.rateperyear(2),
                25: ss.rateperyear(12/2),
                np.inf: ss.rateperyear(12/2),
            },
        }
        
        self.override_rates()
        self.rates = {key: self.arr(key) for key in self.rates_dict}
        self.age_cutoffs = self.generate_age_cutoffs()
        return
    
    def arr(self, name):
        arr = np.array(list(self.rates_dict[name].values()))
        return arr

    def override_rates(self):
        override = self.override
        
        if self.use_globals:
            for rate_name in self.rates_dict:
                self.rates_dict[rate_name] = {np.inf : self.rates_dict[rate_name][np.inf]} # Override all rates with the global value
            return   # Makes sure only default rates are used
            
        if override:
            for rate_name, value in override.items():
                if rate_name not in self.rates_dict:
                    raise ValueError(f"Rate '{rate_name}' is not recognized.")
                
                # If the value is a dictionary, validate, sort, and merge it with the existing rates
                if isinstance(value, dict):
                    # Sort the dictionary by age keys
                    sorted_value = {k: v for k, v in sorted(value.items())}
                    
                    for age, rate in sorted_value.items():
                        if isinstance(rate, (int, float)):
                            rate = ss.perday(rate)  # Convert to ss.perday
                        elif not isinstance(rate, ss.rate):
                            raise ValueError(f"Rate for age {age} in '{rate_name}' must be numeric or ss.rate.")
                        
                        # Update or add the new rate for the specified age
                        self.rates_dict[rate_name][age] = rate
                
                # If the value is a scalar and a key of np.inf is passed, replace the entire rate dictionary
                elif isinstance(value, (int, float)):
                    self.rates_dict[rate_name] = {np.inf: ss.perday(value)}
                
                # If the value is a scalar without np.inf key, override all rates for this rate name
                elif isinstance(value, ss.rate):
                    self.rates_dict[rate_name] = {np.inf: value}
                
                else:
                    raise ValueError(f"Value for '{rate_name}' must be a dictionary, scalar, or ss.rate.")
            
            # Ensure keys are sorted for consistency
            for rate_name in self.rates_dict:
                self.rates_dict[rate_name] = dict(sorted(self.rates_dict[rate_name].items()))
                        
    def generate_age_cutoffs(self):
            return {rate_name: np.array(sorted(rates.keys())) for rate_name, rates in self.rates_dict.items()}

