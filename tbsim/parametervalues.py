import starsim as ss
import numpy as np

class RatesByAge:

    def __init__(self, unit, dt, override=None):

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

        self.RATES  = {
            'rate_LS_to_presym': self.arr('rate_LS_to_presym'),
            'rate_LF_to_presym': self.arr('rate_LF_to_presym'),
            'rate_presym_to_active': self.arr('rate_presym_to_active'),
            'rate_active_to_clear': self.arr('rate_active_to_clear'),
            'rate_exptb_to_dead': self.arr('rate_exptb_to_dead'),
            'rate_smpos_to_dead': self.arr('rate_smpos_to_dead'),
            'rate_smneg_to_dead': self.arr('rate_smneg_to_dead'),
            'rate_treatment_to_clear': self.arr('rate_treatment_to_clear')
        }
        self.AGE_CUTOFFS = np.array([ 0, 15, 25, np.inf])
            
    def arr(self, name):
        arr = np.array(list(self.RATES_DICT[name].values()))
        return arr
    