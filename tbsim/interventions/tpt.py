import numpy as np
import starsim as ss
import tbsim as mtb
from collections import namedtuple
from enum import Enum

__all__ = ['TPTInitiation']


class eTPTRegimens(Enum):
    """
    Minimal Enum of Tuberculosis Preventive Therapy (TPT) durations
    based on CDC 2024 guidelines. Each value is the duration in years
    using `ss.peryear(...)` format.
    
    Usage:
        TPTRegimens._3HP.value  # => ss.peryear(0.25)
    """
    _3HP = ss.peryear(0.25)
    _4R  = ss.peryear(1/3)
    _3HR = ss.peryear(0.25)
    _6H  = ss.peryear(0.5)
    _9H  = ss.peryear(0.75)

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
        - If initiated, individuals are marked as `on_tpt` and receive a fixed protection duration (`tpt_duration`)
        - After the specified `start` date, a proportion (`p_on_tpt_regime`) receive the 3HP regimen
    
    Parameters:
        p_tpt (float or ss.Bernoulli): Probability of initiating TPT for an eligible individual
        max_age (int): Optional filter for outcome reporting (default: 5)
        hiv_status_threshold (bool): Reserved for HIV-based filtering (default: False)
        tpt_duration (float): Duration of protection in years
        p_on_tpt_regime (float): Proportion of individuals initiated on 3HP after the `start` date
        start (date): Rollout date after which 3HP becomes available

    Results tracked:
        n_eligible (int): Number of individuals in eligible households meeting criteria
        n_tpt_initiated (int): Number of individuals actually started on TPT
        n_3HP_assigned (int): Subset of TPT individuals presumed to receive 3HP

    Notes:
        - Requires people to have a 'hhid' attribute (household ID).
        - Assumes states like 'on_tpt', 'received_tpt', 'screen_negative' are initialized.
        - Requires HouseHoldNet or similar to define household structure.   
        
        - tpt_duration is set to 0.5 years by default (6 months for isoniazid monotherapy), 
          however is expected to be provided during the simulation definition. i.e.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            p_tpt=ss.bernoulli(p=1.0),
            max_age=5,
            hiv_status_threshold=False,
            p_on_tpt_regime=ss.bernoulli(p=0.3),      # Proportion of individuals on TPT who receive the 3HP regimen
            tpt_duration=eTPTRegimens._3HP.value,  # duration of the TPT regimen in years, defaulting to 3HP (3 months)
            tpt_duration_years=ss.peryear(3/12),  # duration of TPT in years, defaulting to 3HP (3 months)
            start=ss.date('2000-01-01'),
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
            assigned_3HP = np.random.rand(len(tpt_candidates)) < self.pars.p_on_tpt_regime if use_3HP else np.zeros(len(tpt_candidates), dtype=bool)

            if not hasattr(tb, 'on_treatment_duration'):
                tb.define_states(ss.FloatArr('on_treatment_duration', default=0.0))

            tb.start_treatment(tpt_candidates)
            tb.on_treatment_duration[tpt_candidates] = self.pars.tpt_duration
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
        



# Define a named tuple for clarity
TPTRegimen = namedtuple("TPTRegimen", ["description", "duration"])

class TPTRegimens:
    """
    Tuberculosis Preventive Therapy (TPT) Regimens - CDC 2024 Recommendations

    This class defines each latent TB infection (LTBI) treatment regimen as a class-level attribute.
    Each regimen includes a description, simulation duration using `ss.peryear()`.

    Usage:
        TPTRegimens.apply_3HP.duration

    Source: https://www.cdc.gov/tb/hcp/treatment/latent-tuberculosis-infection.html
    """

    # 3HP: Isoniazid + Rifapentine weekly for 3 months
    apply_3HP = TPTRegimen(
        description=(
            "3 months of once-weekly Isoniazid + Rifapentine. "
            "Recommended for ages 2+, including people with HIV if on ART with acceptable interactions."
        ),
        duration=ss.peryear(0.25),
    )

    # 4R: Rifampin daily for 4 months
    apply_4R = TPTRegimen(
        description=(
            "4 months of daily Rifampin. Recommended for HIV-negative people of all ages, "
            "and those who cannot tolerate isoniazid or have exposure to isoniazid-resistant TB."
        ),
        duration=ss.peryear(1/3),
    )

    # 3HR: Isoniazid + Rifampin daily for 3 months
    apply_3HR = TPTRegimen(
        description=(
            "3 months of daily Isoniazid + Rifampin. Recommended for all ages, "
            "including people with HIV on compatible ART."
        ),
        duration=ss.peryear(0.25),
    )

    # 6H: Isoniazid daily for 6 months
    apply_6H = TPTRegimen(
        description=(
            "6 months of daily Isoniazid monotherapy. Recommended for people of all ages, "
            "especially where rifamycin-based regimens aren't feasible."
        ),
        duration=ss.peryear(0.5),
    )

    # 9H: Isoniazid daily for 9 months
    apply_9H = TPTRegimen(
        description=(
            "9 months of daily Isoniazid monotherapy. Alternative option for people of all ages "
            "if shorter regimens are not suitable."
        ),
        duration=ss.peryear(0.75),
    )

    # Alias to enable dot access using conventional names
    _registry = {
        "3HP": apply_3HP,
        "4R":  apply_4R,
        "3HR": apply_3HR,
        "6H":  apply_6H,
        "9H":  apply_9H
    }

    # Dot-access properties
    __getattr__ = classmethod(lambda cls, key: cls._registry[key])

    @classmethod
    def list_all(cls):
        """
        List all available TPT regimens.
        """
        return list(cls._registry.keys())



