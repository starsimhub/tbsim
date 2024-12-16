import starsim as ss
import numpy as np


class RateVec:
    def __init__(self, cutoffs, values, interpolation="stair", off_value=None):
        """
        Initialize a RateVec instance.
        Args:
            cutoffs (list): List of determinant cutoffs (e.g., ages).
            values (list): Corresponding rates for each range.
            interpolation (str): Method of interpolation ('stair' or 'linear').
        """
        self.cutoffs = np.array(cutoffs)
        # Assume units of day for all rates
        rates = [ss.perday(v) for v in values if not isinstance(v, ss.TimePar)]
        self.values = np.array(rates)
        self.interpolation = interpolation
        self.off_value=ss.perday(off_value) if off_value is not None else None # Default value for turning age off

        if len(self.cutoffs) + 1 != len(self.values):
            raise ValueError("Number of values must be one more than the number of cutoffs.")

    def init(self, parent):
        """ Initialize the rate vector and age stratificaiton off value. """
        for v in self.values:
            v.init(parent)
        
        if self.off_value is not None:
            self.off_value.init(parent)

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
        
    def turn_age_off(self, new_off_value=None):
        """
        Turn age-specific rate functionality off by setting all values to a single value.
        Args:
            new_off_value (float, optional):  The value to set for all rates. 
                                              If not provided, the default `off_value` will be used.
        Raises:
            ValueError: If `new_off_value` is not a valid number and `off_value` is not set.
        """
        if new_off_value is not None:
            if not isinstance(new_off_value, (int, float)):
                raise ValueError(f"Invalid value for new_off_value: {new_off_value}. Must be a number.")
            self.values = np.array([new_off_value, new_off_value])
        elif hasattr(self, 'off_value') and self.off_value is not None:
            self.values = np.array([self.off_value, self.off_value])
        else:
            raise ValueError("No valid value provided for turning age off, and 'off_value' is not set.")

        # Simplify cutoffs to cover all ages
        self.cutoffs = [0]  # Single cutoff to apply the same value to everyone


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

