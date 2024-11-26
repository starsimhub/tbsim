import starsim as ss
import numpy as np

class RatesByAge:

    def __init__(self, unit, dt, override=None):
        self.unit = unit
        self.dt = dt    

        # Key tuberculosis natural history parameters.
        self.AGE_SPECIFIC_RATES = {
            (0, None): {  # Default values for all ages
                'rate_LS_to_presym':        ss.perday(3e-5, parent_unit=self.unit, parent_dt=self.dt),  # Latent Slow to Active Pre-Symptomatic (per day)
                'rate_LF_to_presym':        ss.perday(6e-3, parent_unit=self.unit, parent_dt=self.dt),  # Latent Fast to Active Pre-Symptomatic (per day)
                'rate_presym_to_active':    ss.perday(3e-2, parent_unit=self.unit, parent_dt=self.dt),  # Pre-symptomatic to symptomatic (per day)
                'rate_active_to_clear':     ss.perday(2.4e-4, parent_unit=self.unit, parent_dt=self.dt),  # Active infection to natural clearance (per day)
                'rate_exptb_to_dead':       ss.perday(0.15 * 4.5e-4, parent_unit=self.unit, parent_dt=self.dt),  # Extra-Pulmonary TB to Dead (per day)
                'rate_smpos_to_dead':       ss.perday(4.5e-4, parent_unit=self.unit, parent_dt=self.dt),  # Smear Positive Pulmonary TB to Dead (per day)
                'rate_smneg_to_dead':       ss.perday(0.3 * 4.5e-4, parent_unit=self.unit, parent_dt=self.dt),  # Smear Negative Pulmonary TB to Dead (per day)
                'rate_treatment_to_clear':  ss.peryear(12/2, parent_unit=self.unit, parent_dt=self.dt)  # Treatment to natural clearance (per year)
            },
            (0,15): {
                'rate_LS_to_presym': ss.perday(2.0548e-06, parent_unit=self.unit, parent_dt=self.dt),   # 0.00075/365  
                'rate_LF_to_presym': ss.perday(4.5e-3, parent_unit=self.unit, parent_dt=self.dt),       # 1.64e-3/365 
                'rate_presym_to_active': ss.perday(5.48e-3, parent_unit=self.unit, parent_dt=self.dt),  # 2/365 
                'rate_active_to_clear': ss.perday(2.74e-4, parent_unit=self.unit, parent_dt=self.dt),   # 0.1/365 
                'rate_smpos_to_dead': ss.perday(6.85e-4, parent_unit=self.unit, parent_dt=self.dt),     # 0.25/365 
                'rate_smneg_to_dead': ss.perday(2.74e-4, parent_unit=self.unit, parent_dt=self.dt),     # 0.1/365 
                'rate_exptb_to_dead': ss.perday(2.74e-4, parent_unit=self.unit, parent_dt=self.dt),     # 0.1/365 
                'rate_treatment_to_clear': ss.peryear(2, parent_unit=self.unit, parent_dt=self.dt)      # 2 per year
            },
            (15,25): {     # For now, using the same values as for adults but could be different
                'rate_LS_to_presym': ss.perday(3e-5, parent_unit=self.unit, parent_dt=self.dt),
                'rate_LF_to_presym': ss.perday(6e-3, parent_unit=self.unit, parent_dt=self.dt),
                'rate_presym_to_active': ss.perday(3e-2, parent_unit=self.unit, parent_dt=self.dt),
                'rate_active_to_clear': ss.perday(2.4e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_smpos_to_dead': ss.perday(4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_smneg_to_dead': ss.perday(0.3 * 4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_exptb_to_dead': ss.perday(0.15 * 4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_treatment_to_clear': ss.peryear(12/2, parent_unit=self.unit, parent_dt=self.dt) 
                },
            (25,150): {
                'rate_LS_to_presym': ss.perday(3e-5, parent_unit=self.unit, parent_dt=self.dt),
                'rate_LF_to_presym': ss.perday(6e-3, parent_unit=self.unit, parent_dt=self.dt),
                'rate_presym_to_active': ss.perday(3e-2, parent_unit=self.unit, parent_dt=self.dt),
                'rate_active_to_clear': ss.perday(2.4e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_smpos_to_dead': ss.perday(4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_smneg_to_dead': ss.perday(0.3 * 4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_exptb_to_dead': ss.perday(0.15 * 4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_treatment_to_clear': ss.peryear(12/2, parent_unit=self.unit, parent_dt=self.dt)
            } 
        }
        
        self.RATES_DICT_RESOLVED = {
            'rate_LS_to_presym': {
                0: ss.perday(0, unit, dt),
                1: ss.perday(3e-5, unit, dt),
                2: ss.perday(2.0548e-6, unit, dt),
                3: ss.perday(3e-5, unit, dt),
                4: ss.perday(3e-5, unit, dt),
            },
            'rate_LF_to_presym': {
                0: ss.perday(0, unit, dt),
                1: ss.perday(6e-3, unit, dt),
                2: ss.perday(4.5e-3, unit, dt),
                3: ss.perday(6e-3, unit, dt),
                4: ss.perday(6e-3, unit, dt),
            },
            'rate_presym_to_active': {
                0: ss.perday(0, unit, dt),
                1: ss.perday(3e-2, unit, dt),
                2: ss.perday(5.48e-3, unit, dt),
                3: ss.perday(3e-2, unit, dt),
                4: ss.perday(3e-2, unit, dt),
            },
            'rate_active_to_clear': {
                0: ss.perday(0, unit, dt),
                1: ss.perday(2.4e-4, unit, dt),
                2: ss.perday(2.74e-4, unit, dt),
                3: ss.perday(2.4e-4, unit, dt),
                4: ss.perday(2.4e-4, unit, dt),
            },
            'rate_smpos_to_dead': {
                0: ss.perday(0, unit, dt),
                1: ss.perday(4.5e-4, unit, dt),
                2: ss.perday(6.85e-4, unit, dt),
                3: ss.perday(4.5e-4, unit, dt),
                4: ss.perday(4.5e-4, unit, dt),
            },
            'rate_smneg_to_dead': {
                0: ss.perday(0, unit, dt),
                1: ss.perday(0.3 * 4.5e-4, unit, dt),
                2: ss.perday(2.74e-4, unit, dt),
                3: ss.perday(0.3 * 4.5e-4, unit, dt),
                4: ss.perday(0.3 * 4.5e-4, unit, dt),
            },
            'rate_exptb_to_dead': {
                0: ss.perday(0, unit, dt),
                1: ss.perday(0.15 * 4.5e-4, unit, dt),
                2: ss.perday(2.74e-4, unit, dt),
                3: ss.perday(0.15 * 4.5e-4, unit, dt),
                4: ss.perday(0.15 * 4.5e-4, unit, dt),
            },
            'rate_treatment_to_clear': {
                0: ss.peryear(0, unit, dt),
                1: ss.peryear(6, unit, dt),
                2: ss.peryear(2, unit, dt),
                3: ss.peryear(6, unit, dt),
                4: ss.peryear(6, unit, dt),
            },
        }

        # Convert raw rate values to starsim rates using the helper functions

        data = {
            'age_cutoffs' : np.array([ -1, 0, 15, 25, 200]),
            'rate_LS_to_presym': self.arr('rate_LS_to_presym'),
            'rate_LF_to_presym': self.arr('rate_LF_to_presym'),
            'rate_presym_to_active': self.arr('rate_presym_to_active'),
            'rate_active_to_clear': self.arr('rate_active_to_clear'),
            'rate_exptb_to_dead': self.arr('rate_exptb_to_dead'),
            'rate_smpos_to_dead': self.arr('rate_smpos_to_dead'),
            'rate_smneg_to_dead': self.arr('rate_smneg_to_dead'),
            'rate_treatment_to_clear': self.arr('rate_treatment_to_clear')
        }
        self.RESOLVED = data
        # Apply overrides
        if override is not None:
            self.apply_overrides(override)
            
    def arr(self, name):
        result = np.array(list(self.RATES_DICT_RESOLVED[name].values()))
        return result
    
    def apply_overrides(self, override): # optional method
        for age_range, rates in override.items():
            if age_range in self.AGE_SPECIFIC_RATES:
                # Update only the specific rates provided in the override for each age range
                self.AGE_SPECIFIC_RATES[age_range].update(rates)
            else:
                # Add new age ranges if not in the defaults
                self.AGE_SPECIFIC_RATES[age_range] = rates

    def get_rates(self, age):
        for age_range, rates in self.AGE_SPECIFIC_RATES.items():
            min_age, max_age = age_range
            if max_age is None:  # Handle "all ages" bin
                max_age = float('inf')
            if min_age <= age < max_age:
                return rates
        return None

    def get_rate(self, age, rate):  
        # Attempt to get rates for the specific age bin
        rates = self.get_rates(age)
        if rates is not None and rate in rates:
            return rates[rate]
        
        # Fallback to the "all ages" bin
        if (0, None) in self.AGE_SPECIFIC_RATES and rate in self.AGE_SPECIFIC_RATES[(0, None)]:
            return self.AGE_SPECIFIC_RATES[(0, None)][rate]
        return None

    
    def get_groups(self):
        return list(self.AGE_SPECIFIC_RATES.keys())

    def age_bins(self):
        # Extract the minimum age from each age range tuple and append np.inf as the upper bound
        age_bins = np.array([min_age for min_age, _ in self.AGE_SPECIFIC_RATES.keys()] + [np.inf])
        return age_bins
    
    def get_map(self, rate):
        # Dynamically create the mapping based on the defined age groups
        mapping = {}
        for idx, age_range in enumerate(self.AGE_SPECIFIC_RATES.keys()):
            if rate in self.AGE_SPECIFIC_RATES[age_range]:
                mapping[idx] = self.AGE_SPECIFIC_RATES[age_range][rate]
            else:
                raise KeyError(f"Rate '{rate}' not found for age range {age_range}")
        # Optionally handle default or special cases, like -1
        mapping[-1] = mapping.get(0, None)
        return mapping
    
    