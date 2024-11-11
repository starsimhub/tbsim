import starsim as ss 

class RatesByAge:
    
    
    def __init__(self, unit, dt):
        self.unit = unit
        self.dt = dt    

        # Key tuberculosis natural history parameters.
        self.AGE_SPECIFIC_RATES = {
            '0,15': {           
                'rate_LS_to_presym': ss.perday(2.0548e-06, parent_unit=self.unit, parent_dt=self.dt), # 0.00075/365  
                'rate_LF_to_presym': ss.perday(4.5e-3, parent_unit=self.unit, parent_dt=self.dt), # 1.64e-3/365 
                'rate_presym_to_active': ss.perday(5.48e-3, parent_unit=self.unit, parent_dt=self.dt),  # 2/365 
                'rate_active_to_clear': ss.perday(2.74e-4, parent_unit=self.unit, parent_dt=self.dt),  # 0.1/365 
                'rate_smpos_to_dead': ss.perday(6.85e-4, parent_unit=self.unit, parent_dt=self.dt),  #0.25/365 
                'rate_smneg_to_dead': ss.perday(2.74e-4, parent_unit=self.unit, parent_dt=self.dt),  # 0.1/365 
                'rate_exptb_to_dead': ss.perday(2.74e-4, parent_unit=self.unit, parent_dt=self.dt),  # 0.1/365 
                'rate_treatment_to_clear': ss.peryear(2, parent_unit=self.unit, parent_dt=self.dt)   # 2 per year
            },
            '15,25': {     # For now, using the same values as for adults but could be different
                'rate_LS_to_presym': ss.perday(3e-5, parent_unit=self.unit, parent_dt=self.dt),
                'rate_LF_to_presym': ss.perday(6e-3, parent_unit=self.unit, parent_dt=self.dt),
                'rate_presym_to_active': ss.perday(3e-2, parent_unit=self.unit, parent_dt=self.dt),
                'rate_active_to_clear': ss.perday(2.4e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_smpos_to_dead': ss.perday(4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_smneg_to_dead': ss.perday(0.3 * 4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_exptb_to_dead': ss.perday(0.15 * 4.5e-4, parent_unit=self.unit, parent_dt=self.dt),
                'rate_treatment_to_clear': ss.peryear(12/2, parent_unit=self.unit, parent_dt=self.dt) 
                },
            '25,150': {
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

    def get_rates(self, age):
        for age_range, rates in self.AGE_SPECIFIC_RATES.items():
            age_range = age_range.split(',')
            if int(age) >= int(age_range[0]) and int(age) < int(age_range[1]):
                return rates
        return None
    
    def get_rate(self, age, rate):
        rates = self.get_rates(age)
        if rates is not None:
            return rates[rate]
        return None
    
    def get_groups(self):
        return list(self.AGE_SPECIFIC_RATES.keys())
