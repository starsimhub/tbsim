from enum import IntEnum

__all__ = ['TBSX']

class TBSX(IntEnum):

    # --- Clinical symptoms (observable in SYMPTOMATIC individuals) -----------
    # These are not disease-progression states; they flag which symptoms an
    # agent is presenting.  They can be used downstream by diagnostics or
    # care-seeking interventions to condition on specific symptom profiles.
    SX_COUGH        = 200   # Persistent cough (>2-3 weeks)
    SX_FEVER        = 201   # Fever and/or night sweats
    SX_WEIGHT_LOSS  = 202   # Weight loss / fatigue
    SX_HEMOPTYSIS   = 203   # Hemoptysis (coughing blood)
    SX_CHEST_PAIN   = 204   # Chest pain