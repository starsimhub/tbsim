import starsim as ss
import numpy as np

class RatesByAge:

    def __init__(self, unit, dt):
        self.unit = unit
        self.dt = dt
        self.rates_dict = {
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
        return 
 

class RateVec:
    def __init__(self, cutoffs, values, interpolation="stair"):
        """
        Initialize a RateVec instance.
        Args:
            cutoffs (list): List of determinant cutoffs (e.g., ages).
            values (list): Corresponding rates for each range.
            interpolation (str): Method of interpolation ('stair' or 'linear').
        """
        self.cutoffs = np.array(cutoffs)
        self.values = np.array(values)
        self.interpolation = interpolation

        if len(self.cutoffs) + 1 != len(self.values):
            raise ValueError("Number of values must be one more than the number of cutoffs.")

    def digitize(self, inputs):
        """
        Assign rates based on cutoffs using the selected interpolation method.
        """
        indices = np.digitize(inputs, self.cutoffs, right=True)
        if self.interpolation == "stair":
            return self.values[indices]
        elif self.interpolation == "linear":
            return self.linear_interpolate(inputs)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

    def linear_interpolate(self, inputs):
        """
        Linearly interpolate rates based on inputs.
        """
        rates = np.zeros_like(inputs, dtype=float)
        for i, input_val in enumerate(inputs):
            if input_val < self.cutoffs[0]:
                rates[i] = self.values[0]
            elif input_val > self.cutoffs[-1]:
                rates[i] = self.values[-1]
            else:
                idx = np.searchsorted(self.cutoffs, input_val) - 1
                x0, x1 = self.cutoffs[idx], self.cutoffs[idx + 1]
                y0, y1 = self.values[idx], self.values[idx + 1]
                rates[i] = y0 + (y1 - y0) * (input_val - x0) / (x1 - x0)
        return rates

    def __call__(self, inputs):
        return self.digitize(inputs)

    def __repr__(self):
        return self.__str__()
    
    def __summary__(self):
        return f"RateVec(cutoffs={self.cutoffs}, values={self.values}, interpolation={self.interpolation})"

