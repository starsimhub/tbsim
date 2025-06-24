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
        - A Bernoulli trial (`p_tpt`) is used to determine which eligible individuals initiate TPT
        - If initiated, individuals are flagged as `on_tpt` for the treatment duration (`tpt_treatment_duration`)
        - After completing treatment, individuals become `protected_from_tb` for a fixed duration (`tpt_protection_duration`)
        - Protection starts at `tpt_treatment_until` and ends at `tpt_protection_until`
    
    Parameters:
        p_tpt (float or ss.Bernoulli): Probability of initiating TPT for an eligible individual
        max_age (int): Optional filter for outcome reporting (default: 5)
        hiv_status_threshold (bool): Reserved for HIV-based filtering (default: False)
        tpt_treatment_duration (float): Duration of TPT administration (e.g., 3 months)
        tpt_protection_duration (float): Duration of post-treatment protection (e.g., 2 years)
        start (date): Start date for offering TPT
        stop (date): Stop date for offering TPT

    Results tracked:
        n_eligible (int): Number of individuals meeting criteria
        n_tpt_initiated (int): Number of individuals who started TPT

    Notes:
        - Requires 'hhid', 'on_tpt', 'received_tpt', 'screen_negative' attributes on people
        - Assumes household structure is available (e.g., via HouseHoldNet)
        - TB disease model must support 'protected_from_tb', 'tpt_treatment_until', and 'tpt_protection_until' states
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            p_tpt=ss.bernoulli(p=1.0),
            max_age=5,
            hiv_status_threshold=False,
            tpt_treatment_duration=ss.peryear(0.25),   # 3 months of treatment
            tpt_protection_duration=ss.peryear(2.0),   # 2 years of protection
            start=ss.date('2000-01-01'),
            stop=ss.date('2100-12-31'),
        )
        self.update_pars(pars=pars, **kwargs)

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Initialize states if not already present
        if not hasattr(tb, 'protected_from_tb'):
            tb.define_states(
                ss.BoolArr('protected_from_tb', default=False),
                ss.Arr('tpt_treatment_until', default=None),
                ss.Arr('tpt_protection_until', default=None),
            )

        # Households with TB cases
        treated = tb.on_treatment
        eligible_hhids = np.unique(ppl['hhid'][treated])

        # Eligibility screening
        in_eligible_households = np.isin(ppl['hhid'], eligible_hhids)
        eligible = in_eligible_households & (~tb.on_treatment) & (ppl['screen_negative'] | ppl['non_symptomatic'])

        # Bernoulli selection
        tpt_candidates = self.pars.p_tpt.filter(eligible)

        if len(tpt_candidates):
            # Treatment phase
            ppl['on_tpt'][tpt_candidates] = True
            ppl['received_tpt'][tpt_candidates] = True

            # Track treatment and future protection schedule
            treatment_end = sim.date + self.pars.tpt_treatment_duration
            protection_end = treatment_end + self.pars.tpt_protection_duration

            tb.tpt_treatment_until[tpt_candidates] = treatment_end
            tb.tpt_protection_until[tpt_candidates] = protection_end

            # Result tracking
            self.results['n_eligible'][self.ti] = np.count_nonzero(eligible)
            self.results['n_tpt_initiated'][self.ti] = len(tpt_candidates)

    def init_results(self):
        self.define_results(
            ss.Result('n_eligible', dtype=int),
            ss.Result('n_tpt_initiated', dtype=int),
        )

    def update_results(self):
        ppl = self.sim.people
        self.results['n_eligible'][self.ti] = np.count_nonzero(ppl['on_tpt'] | ppl['received_tpt'])
        self.results['n_tpt_initiated'][self.ti] = np.count_nonzero(ppl['on_tpt'])

        

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

