import starsim as ss
import sciris as sc
import pandas as pd
import json
from scripts.interventions.constants import START, STOP


class DotDict(dict):
    """Simple dict subclass that allows dot notation access."""
    def __getattr__(self, key):
        return self[key]
    
    def to_json(self, indent=2):
        return json.dumps(self, indent=indent, default=str)


Scenarios = DotDict({
        'baseline': {
            'name': 'Baseline',
            'tbpars': dict(start=START, stop=STOP),
        },

        'Baseline_BetaByYear': {
            'name': 'No interventions',
            'tbpars': dict(start=START, stop=STOP),
            'betabyyear':dict(years=[1990, 2000], x_beta=[0.5, 1.4])
        },
        'TPT_Household_Network': {
            'name': 'TPT intervention with optimized household network',
            'tbpars': dict(start=START, stop=STOP),
            'tptintervention': dict(
                p_tpt=0.8,
                age_range=[0, 100],
                hiv_status_threshold=False,
                tpt_treatment_duration=ss.peryear(0.25),  # 3 months
                tpt_protection_duration=ss.peryear(2.0),  # 2 years
                start=ss.date('1980-01-01'),
                stop=ss.date('2020-12-31'),
            ),
        },
        
        'Single_BCG': {
            'name': 'Single BCG intervention',
            'tbpars': dict(start=START, stop=STOP),
            'bcgintervention': dict(
                coverage=0.8,
                start=ss.date('1980-01-01'),
                stop=ss.date('2020-12-31'),
                age_range=[1, 5],
            ),
        },
        
        'Multiple_BCG': {
            'name': 'Multiple BCG interventions',
            'tbpars': dict(start=START, stop=STOP),
            'bcgintervention': [
                dict(
                    coverage=0.9,
                    start=ss.date('1980-01-01'),
                    stop=ss.date('2020-12-31'),
                    age_range=[0, 2],           # For children
                ),
                dict(
                    coverage=0.3,
                    start=ss.date('1985-01-01'),
                    stop=ss.date('2015-12-31'),
                    age_range=[25, 40],         # For adults
                ),
            ],
        }
})
    
