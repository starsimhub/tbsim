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
    
    Requirements:
        - Must have been in a previous intervention which updates screen_negative and non_symptomatic attributes
    
    Eligibility criteria:
        - Must reside in a household where at least one member is on TB treatment
        - Must not already be on TB treatment themselves
        - Must be screen-negative or non-symptomatic
        - Must be within the specified age_range (list: [min_age, max_age], default: [0, 100])
        - Must meet HIV status threshold (if set and hiv_positive attribute present)
    
    Treatment logic:
        - A Bernoulli trial (`p_tpt`) is used to determine which eligible individuals initiate TPT
        - If initiated, individuals are flagged as `on_tpt` for the treatment duration (`tpt_treatment_duration`)
        - After completing treatment, individuals become protected (tb.state = TBS.PROTECTED) for a fixed duration (`tpt_protection_duration`)
        - Protection is tracked using `tpt_protection_until` and is automatically removed when expired (tb.state set to TBS.NONE)
    
    Parameters:
        p_tpt (float or ss.Bernoulli): Probability of initiating TPT for an eligible individual
        age_range (list): [min_age, max_age] for eligibility (default: [0, 100])
        hiv_status_threshold (bool): If True, only HIV-positive individuals are eligible (requires 'hiv_positive' attribute)
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
        - Protection logic mirrors BCG: tb.state is set to TBS.PROTECTED for the immunity period, then reset to TBS.NONE
    """

    def __init__(self, pars=None, **kwargs):
        """
        Initialize the TPTInitiation intervention.

        Args:
            pars (dict, optional): Dictionary containing intervention parameters.
            - 'p_tpt' (float or ss.Bernoulli): Probability of initiating TPT
            - 'age_range' (list): [min_age, max_age] for eligibility (default: [0, 100])
            - 'hiv_status_threshold' (bool): If True, only HIV-positive individuals are eligible
            - 'tpt_treatment_duration' (float): Duration of TPT administration
            - 'tpt_protection_duration' (float): Duration of post-treatment protection
            - 'start' (date): Start date for offering TPT
            - 'stop' (date): Stop date for offering TPT
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.define_pars(
            p_tpt=ss.bernoulli(p=1.0),
            age_range=[0, 100],  # Default: all ages 0-99
            hiv_status_threshold=False,
            tpt_treatment_duration=ss.peryear(0.25),   # 3 months of treatment
            tpt_protection_duration=ss.peryear(2.0),   # 2 years of protection
            start=ss.date('2000-01-01'),
            stop=ss.date('2100-12-31'),
        )
        self.update_pars(pars=pars, **kwargs)

        # --- Parameter validation ---
        try:
            if not isinstance(self.pars.age_range, (list, tuple)) or len(self.pars.age_range) != 2:
                raise ValueError("age_range must be a list or tuple of length 2 (min_age, max_age).")
            min_age, max_age = self.pars.age_range
            if not (isinstance(min_age, (int, float)) and isinstance(max_age, (int, float))):
                raise ValueError("age_range values must be numeric.")
            if min_age < 0 or max_age <= min_age:
                raise ValueError("age_range must have 0 <= min_age < max_age.")
            if not (0 <= self.pars.p_tpt.p <= 1):
                raise ValueError("p_tpt must be a probability between 0 and 1.")
            if not hasattr(self.pars, 'tpt_treatment_duration') or self.pars.tpt_treatment_duration <= 0:
                raise ValueError("tpt_treatment_duration must be positive.")
            if not hasattr(self.pars, 'tpt_protection_duration') or self.pars.tpt_protection_duration < 0:
                raise ValueError("tpt_protection_duration must be non-negative.")
        except Exception as e:
            print(f"[TPTInitiation] Parameter validation error: {e}")
            raise
    
    def step(self):
        """
        Execute the TPT intervention step, applying TPT and protection logic at each timestep.
        """
        try:
            sim = self.sim
            ppl = sim.people
            tb = sim.diseases.tb

            # --- Start/stop window check ---
            now_date = sim.date
            if now_date < self.pars.start or now_date > self.pars.stop:
                return

            # Initialize states if not already present
            if not hasattr(tb, 'protected_from_tb'):
                tb.define_states(
                    ss.BoolArr('protected_from_tb', default=False),
                    ss.Arr('tpt_treatment_until', default=None),
                    ss.Arr('tpt_protection_until', default=None),
                )

            # --- Remove protection if expired ---
            if hasattr(tb, 'tpt_protection_until'):
                expired = (tb.state == mtb.TBS.PROTECTED) & (sim.date >= tb.tpt_protection_until)
                if np.any(expired):
                    tb.state[expired] = mtb.TBS.NONE
                    tb.tpt_protection_until[expired] = None

            # Households with TB cases
            if not hasattr(tb, 'on_treatment'):
                print("[TPTInitiation] Error: tb.on_treatment attribute missing.")
                return
            if not hasattr(ppl, 'hhid'):
                print("[TPTInitiation] Error: people.hhid attribute missing.")
                return
            treated = tb.on_treatment
            eligible_hhids = np.unique(ppl['hhid'][treated])

            # Eligibility screening
            if not hasattr(ppl, 'screen_negative') or not hasattr(ppl, 'non_symptomatic'):
                print("[TPTInitiation] Error: people must have 'screen_negative' and 'non_symptomatic' attributes.")
                return
            in_eligible_households = np.isin(ppl['hhid'], eligible_hhids)
            eligible = in_eligible_households & (~tb.on_treatment) & (ppl['screen_negative'] | ppl['non_symptomatic'])

            # Age range filter (if age_range is set and present)
            if hasattr(ppl, 'age') and self.pars.age_range is not None:
                age_range = list(self.pars.age_range)
                min_age, max_age = age_range[0], age_range[1]
                eligible = eligible & (ppl['age'] >= min_age) & (ppl['age'] < max_age)

            # HIV status filter (if hiv_status_threshold is set True and attribute present)
            if self.pars.hiv_status_threshold:
                if not hasattr(ppl, 'hiv_positive'):
                    print("[TPTInitiation] Error: hiv_status_threshold is True but people.hiv_positive attribute is missing.")
                    return
                eligible = eligible & (ppl['hiv_positive'] == True)

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

                # Set TB state to PROTECTED after treatment ends
                # (Assume protection starts immediately after treatment for simplicity)
                tb.state[tpt_candidates] = mtb.TBS.PROTECTED

                # Result tracking
                self.results['n_eligible'][self.ti] = np.count_nonzero(eligible)
                self.results['n_tpt_initiated'][self.ti] = len(tpt_candidates)
        except Exception as e:
            print(f"[TPTInitiation] Runtime error in step: {e}")
            raise

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

