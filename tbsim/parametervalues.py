import starsim as ss
import numpy as np


def get_rates( unit, dt ):
    unit = unit
    dt = dt    
    # Abbreviated helper functions for rate conversion
    def pd(value):
        # return ss.perday(value, parent_unit=unit, parent_dt=dt)
        return value
    
    def py(value):
        # return ss.peryear(value, parent_unit=unit, parent_dt=dt)
        return value

    # Define age cutoffs (lower limits)
    age_cutoffs = np.array([ -1, 15, 25, 200])

    # Define rates per age group using numpy arrays
    rates_dict = {
        'rate_LS_to_presym': np.array([     # Latent Slow to Pre-Symptomatic
            pd(3e-5),          
            pd(2.0548e-6),     
            pd(3e-5),          
            pd(3e-5),          
        ]),
        'rate_LF_to_presym': np.array([     # Latent Fast to Pre-Symptomatic
            pd(6e-3),
            pd(4.5e-3),
            pd(6e-3),
            pd(6e-3),
        ]),
        'rate_presym_to_active': np.array([ # Pre-Symptomatic to Active
            pd(3e-2),
            pd(5.48e-3),
            pd(3e-2),
            pd(3e-2),
        ]),
        'rate_active_to_clear': np.array([  # Active infection to clearance (Question: Not sure if we said natural clearance or intervention induced clearance)
            pd(2.4e-4),
            pd(2.74e-4),
            pd(2.4e-4),
            pd(2.4e-4),  
        ]),
        'rate_smpos_to_dead': np.array([    # Smear Positive to Dead
            pd(4.5e-4),
            pd(6.85e-4),
            pd(4.5e-4),
            pd(4.5e-4),
        ]),
        'rate_smneg_to_dead': np.array([    # Smear Negative to Dead
            pd(0.3 * 4.5e-4),
            pd(2.74e-4),  
            pd(0.3 * 4.5e-4),
            pd(0.3 * 4.5e-4),
        ]),
        'rate_exptb_to_dead': np.array([    # Extra-Pulmonary TB to Dead
            pd(0.15 * 4.5e-4),
            pd(2.74e-4), 
            pd(0.15 * 4.5e-4),
            pd(0.15 * 4.5e-4),
        ]),
        'rate_treatment_to_clear': np.array([   # Per treatment to clear
            py(6),
            py(2), 
            py(6), 
            py(6),
        ])}

    # Convert raw rate values to starsim rates using the helper functions

    data = {
        'age_cutoffs': age_cutoffs,
        'rate_LS_to_presym': rates_dict['rate_LS_to_presym'],
        'rate_LF_to_presym': rates_dict['rate_LF_to_presym'],
        'rate_presym_to_active': rates_dict['rate_presym_to_active'],
        'rate_active_to_clear': rates_dict['rate_active_to_clear'],
        'rate_exptb_to_dead': rates_dict['rate_exptb_to_dead'],
        'rate_smpos_to_dead': rates_dict['rate_smpos_to_dead'],
        'rate_smneg_to_dead': rates_dict['rate_smneg_to_dead'],
        'rate_treatment_to_clear': rates_dict['rate_treatment_to_clear']
    }

    return data

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
        # Apply overrides
        if override is not None:
            self.apply_overrides(override)
            
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
    
    def get_dict(self):
        unit = self.unit
        dt = self.dt
        # Abbreviated helper functions for rate conversion
        def pd(value):
            # v =  ss.perday(value, parent_unit=unit, parent_dt=dt)  BUG - can't get the units from parent
            return value
        
        def py(value):
            # v = ss.peryear(value, parent_unit=unit, parent_dt=dt)   BUG - can't get the units from parent
            return value 

        # Define age cutoffs (lower limits)
        age_cutoffs = np.array([ -1, 0, 15, 25, 200])
        age_cutoffs = np.array(age_cutoffs, dtype=np.float64)

        # Define rates per age group using numpy arrays
        rates_dict = {
            'rate_LS_to_presym': np.array([
                pd(0),
                pd(3e-5),          
                pd(2.0548e-6),     
                pd(3e-5),          
                pd(3e-5),          
            ]),
            'rate_LF_to_presym': np.array([
                pd(0),
                pd(6e-3),          
                pd(4.5e-3),        
                pd(6e-3),          
                pd(6e-3),          
            ]),
            'rate_presym_to_active': np.array([
                pd(0),
                pd(3e-2),          
                pd(5.48e-3),       
                pd(3e-2),          
                pd(3e-2),          
            ]),
            'rate_active_to_clear': np.array([
                pd(0),
                pd(2.4e-4),        
                pd(2.74e-4),       
                pd(2.4e-4),        
                pd(2.4e-4),        
            ]),
            'rate_smpos_to_dead': np.array([
                pd(0),
                pd(4.5e-4),        
                pd(6.85e-4),       
                pd(4.5e-4),        
                pd(4.5e-4),        
            ]),
            'rate_smneg_to_dead': np.array([
                pd(0),
                pd(0.3 * 4.5e-4),  
                pd(2.74e-4),       
                pd(0.3 * 4.5e-4),  
                pd(0.3 * 4.5e-4),  
            ]),
            'rate_exptb_to_dead': np.array([   
                pd(0),
                pd(0.15 * 4.5e-4), 
                pd(2.74e-4),       
                pd(0.15 * 4.5e-4), 
                pd(0.15 * 4.5e-4), 
            ]),
            'rate_treatment_to_clear': np.array([   # Per year
                py(0),
                py(6),
                py(2), 
                py(6), 
                py(6),
            ])}

        prognoses_age = {
            'age_cutoffs': age_cutoffs,
            'rates': rates_dict
        }

        return prognoses_age    