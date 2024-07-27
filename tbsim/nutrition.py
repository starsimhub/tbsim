from enum import IntEnum, auto

__all__ = ['nMacroNutrients', 'nMicroNutrients', 'BasicStatus', 'SpecificStatus', 'BmiStatus']

class nMacroNutrients(IntEnum):
    STANDARD_OR_ABOVE = auto()
    SLIGHTLY_BELOW_STANDARD = auto()
    MARGINAL = auto()
    UNSATISFACTORY = auto()

class nMicroNutrients(IntEnum):
    NORMAL = auto()
    DEFICIENT = auto()

# class NutritionStatus(IntEnum):
class BasicStatus(IntEnum):
    NORMAL = auto()                   # Normal nutrition
    UNDERNUTRITION = auto()           # malnutrition, this includes stunting, wasting, underweight
    OVERNUTRITION = auto()            # overweight, obesity, and the extension of this would be the related NCD
    MICRONUTRIENT_DEFICIENCY = auto() # micronutrient-imbalance: allows use of the disease state without being too specific

class SpecificStatus(IntEnum):
    # Subforms of malnutrition Nutrition States
    STUNTING = auto()                    # undernutrition sub-form: low height for age
    WASTING = auto()                     # undernutrition sub-form: low weight for height
    UNDERWEIGHT = auto()                 # undernutrition sub-form: low weight for age
    MICRONUTRIENT_DEFICIENCY = auto()    # micronutrient deficiencies, a lack of important vitamins and minerals
    MACRONUTRIENT_DEFICIENCY = auto()    # macronutrient deficiencies
    MICRONUTRIENT_EXCESS = auto()        # micronutrient excess
    MACRONUTRIENT_EXCESS = auto()        # macronutrient excess
    OVERWEIGHT = auto()                  # BMI of 30 or more
    OBESITY = auto()                     # BMI of 30 or more

class BmiStatus(IntEnum):
    SEVERE_THINNESS = auto()             # BMI < 16.0
    MODERATE_THINNESS = auto()           # BMI 16.0 - 16.99
    MILD_THINNESS = auto()               # BMI 17.0 - 18.49
    NORMAL_WEIGHT = auto()               # BMI 18.5 - 24.99
    ABOVE_NORMAL_WEIGHT = auto()         # BMI 25.0 - 29.99
    
    # Obesity
    OVERWEIGHT = auto()                  # BMI 25.0 - 29.9
    OBESE_CLASS_I_MODERATE = auto()      # BMI 30.0 - 34.9
    OBESE_CLASS_II_SEVERE = auto()       # BMI 35.0 - 39.9
    OBESE_CLASS_III_VERY_SEVERE = auto() # BMI ≥ 40.0

class descriptions():
    BMI_RANGES = {
        "SEVERE_THINNESS": "< 16.0",
        "MODERATE_THINNESS": "16.0 - 16.9",
        "MILD_THINNESS": "17.0 - 18.4",
        "NORMAL_WEIGHT": "18.5 - 24.9",
        "OVERWEIGHT": "25.0 - 29.9",
        "OBESE_CLASS_I_MODERATE": "30.0 - 34.9",
        "OBESE_CLASS_II_SEVERE": "35.0 - 39.9",
        "OBESE_CLASS_III_VERY_SEVERE": "≥ 40.0"
    }
    