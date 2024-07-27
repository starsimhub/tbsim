"""
Define non-communicable disease (Malnutrition) model
"""

import numpy as np
from enum import IntEnum, auto


__all__ = ['MacroNutrients', 'MicroNutrients', 'TBS', 'StudyArm']

class MacroNutrients(IntEnum):
    STANDARD_OR_ABOVE = auto()
    SLIGHTLY_BELOW_STANDARD = auto()
    MARGINAL = auto()
    UNSATISFACTORY = auto()

class MicroNutrients(IntEnum):
    NORMAL = auto()
    DEFICIENT = auto()
    
class StudyArm(IntEnum):
    CONTROL = auto()
    VITAMIN = auto()

class TBS():           # Enum
    NONE            = np.nan # No TB
    LATENT_SLOW     = 0.0    # Latent TB, slow progression
    LATENT_FAST     = 1.0    # Latent TB, fast progression
    ACTIVE_PRESYMP  = 2.0    # Active TB, pre-symptomatic
    ACTIVE_SMPOS    = 3.0    # Active TB, smear positive
    ACTIVE_SMNEG    = 4.0    # Active TB, smear negative
    ACTIVE_EXPTB    = 5.0    # Active TB, extra-pulmonary
    CURE            = 6.0    # Being cured
    DEAD            = 7.0    # TB death
