from enum import IntEnum, auto

__all__ = ['eStudyArm', 'eMacroNutrients', 'eMicroNutrients', 'eHighlevelStatus', 'eDetailedStatus', 'eBmiStatus']

class eStudyArm(IntEnum):
    CONTROL = auto()
    VITAMIN = auto()


class eMacroNutrients(IntEnum):
    STANDARD_OR_ABOVE = auto()
    SLIGHTLY_BELOW_STANDARD = auto()
    MARGINAL = auto()
    UNSATISFACTORY = auto()

class eMicroNutrients(IntEnum):
    NORMAL = auto()
    DEFICIENT = auto()

# class NutritionStatus(IntEnum):
class eHighlevelStatus(IntEnum):
    NORMAL = auto()                   # Normal nutrition
    UNDERNUTRITION = auto()           # malnutrition, this includes stunting, wasting, underweight
    OVERNUTRITION = auto()            # overweight, obesity, and the extension of this would be the related NCD
    MICRONUTRIENT_DEFICIENCY = auto() # micronutrient-imbalance: allows use of the disease state without being too specific

class eDetailedStatus(IntEnum):
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

class eBmiStatus(IntEnum):
    NORMAL_WEIGHT = auto()               # BMI 18.5 - 24.99
    MILD_THINNESS = auto()               # BMI 17.0 - 18.49
    MODERATE_THINNESS = auto()           # BMI 16.0 - 16.99
    SEVERE_THINNESS = auto()             # BMI < 16.0

    # Obesity
    OVERWEIGHT = auto()                  # BMI 25.0 - 29.9
    OBESE_CLASS_I_MODERATE = auto()      # BMI 30.0 - 34.9
    OBESE_CLASS_II_SEVERE = auto()       # BMI 35.0 - 39.9
    OBESE_CLASS_III_VERY_SEVERE = auto() # BMI ≥ 40.0

class descriptions():
    SEVERE_THINNESS = "Severe Thinness: \n< 16.0",
    MODERATE_THINNESS= "Moderate Thiness: \n16.0 - 16.9",
    MILD_THINNESS="Mild Thinness: \n17.0 - 18.4",
    NORMAL_WEIGHT= "Normal Weight: \n18.5 - 24.9",
    OVERWEIGHT= "Overweight: \n25.0 - 29.9",
    OBESE_CLASS_I_MODERATE= "Obese class I Moderate: \n30.0 - 34.9",
    OBESE_CLASS_II_SEVERE= "Obese Class II Severe: \n35.0 - 39.9",
    OBESE_CLASS_III_VERY_SEVERE= "Obese Class III Very Severe \n≥ 40.0"
    