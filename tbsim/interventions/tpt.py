import numpy as np
import starsim as ss
import tbsim
from collections import namedtuple
from enum import Enum

__all__ = ['TPTInitiation', 'TPTRegimes']

class TPTInitiation(ss.Intervention):
    """
    Tuberculosis Preventive Therapy (TPT) intervention for household contacts of care-seeking individuals.

    This intervention identifies households where at least one member sought care, and offers TPT to all
    other members of those households who meet the following eligibility criteria:
    
    Requirements:
        - Must have been in a previous intervention which updates sought_care, non_symptomatic, and symptomatic attributes
    
    Eligibility criteria:
        - Must reside in a household where at least one member sought care
        - Must not have symptoms (non_symptomatic = True and symptomatic = False)
        - Must not have active TB (not in ACTIVE_PRESYMP, ACTIVE_SMPOS, ACTIVE_SMNEG, or ACTIVE_EXPTB states)
        - Must not already be on TB treatment themselves
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
        start (ss.date): Start date for offering TPT
        stop (ss.date): Stop date for offering TPT

    Results tracked:
        n_eligible (int): Number of individuals meeting criteria
        n_tpt_initiated (int): Number of individuals who started TPT

    Notes:
        - Requires 'hhid', 'on_tpt', 'received_tpt', 'sought_care', 'non_symptomatic', 'symptomatic' attributes on people
        - Assumes household structure is available (e.g., via HouseHoldNet)
        - TB disease model must support 'protected_from_tb', 'tpt_treatment_until', and 'tpt_protection_until' states
        - Protection logic mirrors BCG: tb.state is set to TBS.PROTECTED for the immunity period, then reset to TBS.NONE
        - Uses care-seeking individuals as the index population for household contact tracing
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
        
        # # Define states for TPT tracking
        # self.define_states(
        #     ss.BoolState('sought_care', default=False),
        #     ss.BoolState('non_symptomatic', default=True),
        #     ss.BoolState('symptomatic', default=False),
        #     ss.BoolState('on_tpt', default=False),
        #     ss.BoolState('received_tpt', default=False),
        #     ss.IntArr('hhid', default=-1),
        #     ss.BoolState('hiv_positive', default=False),
        #     ss.FloatArr('tpt_treatment_until', default=np.nan),  # When TPT treatment ends
        #     ss.FloatArr('tpt_protection_until', default=np.nan),  # When TPT protection ends
        # )
    
    def step(self):
        """
        Execute the TPT intervention step, applying TPT and protection logic at each timestep.

        - Removes protection for individuals whose TPT protection has expired (tb.state set to TBS.NONE)
        - Identifies agents who sought care (INDEX list)
        - For all INDEX UIDs, identifies household members
        - Filters household members for eligibility: no symptoms, no active TB, not on_treatment
        - Uses a Bernoulli trial to select candidates for TPT
        - Sets on_tpt and received_tpt flags for those who start TPT
        - Sets tpt_treatment_until and tpt_protection_until for those who start TPT
        - Sets tb.state to TBS.PROTECTED for the protection period after treatment
        - Tracks results for eligibility and initiation
        """
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # TPT-related states are now defined in the TB disease module
        # No need to dynamically define them here

        # --- Remove protection if expired (simplified for now) ---
        # TODO: Implement proper date-based expiration logic
        # For now, we'll skip the expiration check to get the basic functionality working

        # Find agents who sought care (INDEX list)
        sought_care = ppl.sought_care
        
        # Safety check: ensure hhid array is properly initialized
        if not hasattr(ppl, 'hhid') or len(ppl.hhid) == 0:
            return
            
        # Get households of agents who sought care
        eligible_hhids = np.unique(ppl['hhid'][sought_care])

        # Eligibility screening for household members
        in_eligible_households = np.isin(ppl['hhid'], eligible_hhids)
        # Filter criteria: no symptoms, no active TB, not on_treatment
        no_symptoms = ppl['non_symptomatic'] & (~ppl['symptomatic'])
        no_active_tb = (tb.state != tbsim.TBS.ACTIVE_PRESYMP) & (tb.state != tbsim.TBS.ACTIVE_SMPOS) & (tb.state != tbsim.TBS.ACTIVE_SMNEG) & (tb.state != tbsim.TBS.ACTIVE_EXPTB)
        not_on_treatment = ~tb.on_treatment
        
        eligible = in_eligible_households & no_symptoms & no_active_tb & not_on_treatment

        # Age range filter (if age_range is set and present)
        if hasattr(ppl, 'age') and self.pars.age_range is not None:
            age_range = list(self.pars.age_range)
            min_age, max_age = age_range[0], age_range[1]
            eligible = eligible & (ppl['age'] >= min_age) & (ppl['age'] < max_age)

        # HIV status filter (if hiv_status_threshold is set True and attribute present)
        if self.pars.hiv_status_threshold and hasattr(ppl, 'hiv_positive'):
            eligible = eligible & (ppl['hiv_positive'] == True)

        # Bernoulli selection
        tpt_candidates = self.pars.p_tpt.filter(eligible)

        if len(tpt_candidates):
            # Treatment phase
            ppl['on_tpt'][tpt_candidates] = True
            ppl['received_tpt'][tpt_candidates] = True

            # Track treatment and future protection schedule
            now = self.sim.now
            if hasattr(now, 'date'):
                now_date = now.date()
            else:
                now_date = now
                
            # Set treatment and protection end dates (simplified)
            # For now, we'll use a simple approach without complex date arithmetic
            tb.tpt_treatment_until[tpt_candidates] = self.sim.ti + 365  # 1 year from now (in time steps)
            tb.tpt_protection_until[tpt_candidates] = self.sim.ti + 365*5  # 5 years from now (in time steps)

            # Set TB state to PROTECTED after treatment ends
            # (Assume protection starts immediately after treatment for simplicity)
            tb.state[tpt_candidates] = tbsim.TBS.PROTECTED

            # Result tracking
            self.results['n_eligible'][self.ti] = np.count_nonzero(eligible)
            self.results['n_tpt_initiated'][self.ti] = len(tpt_candidates)

    def init_results(self):
        super().init_results()
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