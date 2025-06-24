import numpy as np
import starsim as ss
import tbsim as mtb
from collections import namedtuple
from enum import Enum

__all__ = ['TPTInitiation', 'TPTRegimes']


class TPTInitiation(ss.Intervention):
    """
    Tuberculosis Preventive Therapy (TPT) intervention for entire eligible households.

    This intervention identifies households with at least one TB-treated individual, and offers TPT to all
    other members of those households who meet the following eligibility criteria:
    
    Eligibility criteria:
        - Must reside in a household where at least one member is on TB treatment
        - Must not already be on TB treatment themselves
        - Must be screen-negative or non-symptomatic
        - Optionally filtered by age and/or HIV status (logic can be extended)
    
    Treatment logic:
        - A Bernoulli trial (`p_tpt`) is used to determine which eligible individuals receive TPT
        - If initiated, individuals are marked as `on_tpt` and receive a fixed protection duration (`tpt_regime_duration`)
        - After the specified `start` date, a proportion (`p_on_tpt`) receive the 3HP regimen
    
    Parameters:
        p_tpt (float or ss.Bernoulli): Probability of initiating TPT for an eligible individual
        p_on_tpt (float): Proportion of individuals initiated on 3HP after the `start` date
        max_age (int): Optional filter for outcome reporting (default: 5)
        hiv_status_threshold (bool): Reserved for HIV-based filtering (default: False)
        tpt_regime_duration (float): Duration of protection in years
        start (date): Intervention start date, this is when TPT can begin being offered
        stop (date): Intervention end date, after which no new TPT is initiated
ß
    Results tracked:
        n_eligible (int): Number of individuals in eligible households meeting criteria
        n_tpt_initiated (int): Number of individuals actually started on TPT
        n_3HP_assigned (int): Subset of TPT individuals presumed to receive 3HP

    Notes:
        - Requires people to have a 'hhid' attribute (household ID).
        - Assumes states like 'on_tpt', 'received_tpt', 'screen_negative' are initialized.
        - Requires HouseHoldNet or similar to define household structure.   
        
        - tpt_regime_duration is set to 0.5 years by default (6 months for isoniazid monotherapy), 
          however is expected to be provided during the simulation definition. i.e.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            p_tpt=ss.bernoulli(p=1.0),
            p_on_tpt=ss.bernoulli(p=0.3),      # Proportion of individuals on TPT who receive the selected regimen
            max_age=5,
            hiv_status_threshold=False,
            tpt_regime_duration=TPTRegimes.cdc_3HP,      # duration of the TPT regimen in years, defaulting to 3HP (3 months)
            start=ss.date('2000-01-01'),              # Date when the intervention starts
            stop=ss.date('2100-12-31'),               # Date when the intervention stops   
        )
        self.update_pars(pars=pars, **kwargs)

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Identify households with at least one member currently on TB treatment
        treated = tb.on_treatment 
        eligible_hhids = np.unique(ppl['hhid'][treated])

        # Identify all members of those households
        in_eligible_households = np.isin(ppl['hhid'], eligible_hhids)
        eligible = in_eligible_households & (~tb.on_treatment) & (ppl['screen_negative'] | ppl['non_symptomatic'])

        tpt_candidates = self.pars.p_tpt.filter(eligible)

        if len(tpt_candidates):
            use_3HP = sim.year >= self.pars.start.year
            assigned_3HP = np.random.rand(len(tpt_candidates)) < self.pars.p_on_tpt if use_3HP else np.zeros(len(tpt_candidates), dtype=bool)

            if not hasattr(tb, 'on_treatment_duration'):
                tb.define_states(ss.FloatArr('on_treatment_duration', default=0.0))

            tb.start_treatment(tpt_candidates)
            tb.on_treatment_duration[tpt_candidates] = self.pars.tpt_regime_duration
            ppl['on_tpt'][tpt_candidates] = True
            ppl['received_tpt'][tpt_candidates] = True

            self.results['n_eligible'][self.ti] = np.count_nonzero(eligible)
            self.results['n_tpt_initiated'][self.ti] = len(tpt_candidates)
            self.results['n_3HP_assigned'][self.ti] = np.count_nonzero(assigned_3HP)

    def init_results(self):
        self.define_results(
            ss.Result('n_eligible', dtype=int),
            ss.Result('n_tpt_initiated', dtype=int),
            ss.Result('n_3HP_assigned', dtype=int),
        )

    def update_results(self):
        ppl = self.sim.people
        self.results['n_eligible'][self.ti] = np.count_nonzero(ppl['on_tpt'] | ppl['received_tpt'])
        self.results['n_tpt_initiated'][self.ti] = np.count_nonzero(ppl['on_tpt'])
        self.results['n_3HP_assigned'][self.ti] = np.count_nonzero(ppl['on_tpt'] & (ppl.age < self.pars.max_age))
        

class TPTRegimes():
    """
    Tuberculosis Preventive Therapy (TPT) Regimens - CDC 2024 Recommendations

    This class defines latent TB infection (LTBI) treatment regimens and their
    standard durations using `ss.peryear(...)`.

    Regimens:
        - 3HP: Isoniazid + Rifapentine, once weekly for 3 months
            → ss.peryear(0.25)
            Recommended for ages 2+ and people with HIV if ART-compatible.

        - 4R: Rifampin, daily for 4 months
            → ss.peryear(1/3)
            Recommended for HIV-negative individuals and INH-intolerant cases.

        - 3HR: Isoniazid + Rifampin, daily for 3 months
            → ss.peryear(0.25)
            Recommended for all ages, including some on ART.

        - 6H: Isoniazid, daily for 6 months
            → ss.peryear(0.5)
            Used when rifamycins are not feasible.

        - 9H: Isoniazid, daily for 9 months
            → ss.peryear(0.75)
            Alternative when shorter regimens are not suitable.

    Source:
        https://www.cdc.gov/tb/hcp/treatment/latent-tuberculosis-infection.html
    """
    cdc_3HP = ss.peryear(0.25)
    cdc_4R  = ss.peryear(1/3)
    cdc_3HR = ss.peryear(0.25)
    cdc_6H  = ss.peryear(0.5)
    cdc_9H  = ss.peryear(0.75)

