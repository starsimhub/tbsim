import numpy as np
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd

from enum import IntEnum

__all__ = ['TB', 'TBS']


class TBS(IntEnum):
    """
    Tuberculosis disease states enumeration.
    
    with respect to tuberculosis infection and disease progression.
    It is based on the tb natural history model from https://www.pnas.org/doi/full/10.1073/pnas.0901720106
    and updated with rates from TB natural history literature (e.g., PMID: 32766718, 9363017)
    
    Enumeration:
        | State           | Value | Description                                           |
        |-----------------|-------|-------------------------------------------------------|
        | NONE            | -1    | No TB infection (susceptible)                        |
        | LATENT_SLOW     | 0     | Latent TB with slow progression to active disease   |
        | LATENT_FAST     | 1     | Latent TB with fast progression to active disease   |
        | ACTIVE_PRESYMP  | 2     | Active TB in pre-symptomatic phase                  |
        | ACTIVE_SMPOS    | 3     | Active TB, smear positive (most infectious)         |
        | ACTIVE_SMNEG    | 4     | Active TB, smear negative (moderately infectious)   |
        | ACTIVE_EXPTB    | 5     | Active TB, extra-pulmonary (least infectious)       |
        | DEAD            | 8     | Death from TB                                         |
        | PROTECTED       | 100   | Protected from TB (e.g., from BCG vaccination)      |
        
    Methods:
        all(): Get all TB states including special states
        all_active(): Get all active TB states
        all_latent(): Get all latent TB states
        all_infected(): Get all states that represent TB infection (excluding NONE, DEAD, and PROTECTED)
    """
    NONE            = -1    # No TB
    LATENT_SLOW     = 0     # Latent TB, slow progression
    LATENT_FAST     = 1     # Latent TB, fast progression
    ACTIVE_PRESYMP  = 2     # Active TB, pre-symptomatic
    ACTIVE_SMPOS    = 3     # Active TB, smear positive
    ACTIVE_SMNEG    = 4     # Active TB, smear negative
    ACTIVE_EXPTB    = 5     # Active TB, extra-pulmonary
    DEAD            = 8     # TB death
    PROTECTED       = 100

    # region Convenience methods for all states
  
    @staticmethod
    def all():
        """
        Get all TB states including special states.
        
        Returns:
            numpy.ndarray: All TB states including NONE, LATENT_SLOW, LATENT_FAST, 
                          ACTIVE_PRESYMP, ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB, 
                          DEAD, and PROTECTED.
        """
        return np.array([TBS.NONE, TBS.LATENT_SLOW, TBS.LATENT_FAST, TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB, TBS.DEAD, TBS.PROTECTED])
    
    @staticmethod
    def all_active():
        """
        Get all active TB states.
        
        Returns:
            numpy.ndarray: Active TB states including ACTIVE_PRESYMP, ACTIVE_SMPOS, 
                          ACTIVE_SMNEG, and ACTIVE_EXPTB.
        """
        return np.array([TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
    
    @staticmethod
    def all_latent():
        """
        Get all latent TB states.
        
        Returns:
            numpy.ndarray: Latent TB states including LATENT_SLOW and LATENT_FAST.
        """
        return np.array([TBS.LATENT_SLOW, TBS.LATENT_FAST])
    
    @staticmethod
    def all_infected():
        """
        Get all states that represent TB infection (excluding NONE, DEAD, and PROTECTED).
        
        Returns:
            numpy.ndarray: TB infection states including LATENT_SLOW, LATENT_FAST, 
                          ACTIVE_PRESYMP, ACTIVE_SMPOS, ACTIVE_SMNEG, and ACTIVE_EXPTB.
        """
        return np.array([TBS.LATENT_SLOW, TBS.LATENT_FAST, TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
    
    @staticmethod
    def care_seeking_eligible():
        """Get active TB states eligible for care-seeking (excludes ACTIVE_PRESYMP)."""
        return np.array([TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
    # endregion
    

class TB(ss.Infection):
    """
    Tuberculosis disease model for TBsim simulations.
    
    This class implements a comprehensive tuberculosis simulation model that tracks
    individuals through various disease states including latent infection, active disease,
    and treatment. The model includes:
    
    - Multiple latent TB states (slow vs. fast progression)
    - Pre-symptomatic active TB phase 
    - Different active TB forms (smear positive/negative, extra-pulmonary)
    - Treatment effects on transmission and mortality
    - Age-specific reporting (15+ years)
    
    The model follows a state transition approach where individuals move between
    states based on probabilistic transitions that depend on their current state,
    treatment status, and individual risk factors.
    
    Attributes:
        state: Current TB state for each individual
        latent_tb_state: Specific latent state (slow/fast) for each individual
        active_tb_state: Specific active state (smear pos/neg, extra-pulmonary) for each individual
        on_treatment: Boolean indicating if individual is currently on treatment
        ever_infected: Boolean indicating if individual has ever been infected
        rr_activation: Relative risk multiplier for latent-to-active transition
        rr_clearance: Relative risk multiplier for active-to-clearance transition
        rr_death: Relative risk multiplier for active-to-death transition
        reltrans_het: Individual-level heterogeneity in infectiousness
    """
    
    def __init__(self, pars=None, **kwargs):
        """
        Initialize the TB disease model.
        
        Args:
            pars: Dictionary of parameters to override defaults
            **kwargs: Additional keyword arguments for parameters
        """
        super().__init__()

        self.define_pars(
            # Initial conditions
            init_prev = ss.bernoulli(0.01),                            # Initial seed infections
            
            # Transmission parameters
            beta = ss.peryear(0.025),                                  # Infection probability per year
            
            # Latent progression parameters
            p_latent_fast = ss.bernoulli(0.1),                         # Probability of latent fast vs slow progression
            
            # State transition rates
            rate_LS_to_presym       = ss.perday(3e-5),                 # Latent Slow to Active Pre-Symptomatic (per day)            
            rate_LF_to_presym       = ss.perday(6e-3),                 # Latent Fast to Active Pre-Symptomatic (per day)
            rate_presym_to_active   = ss.perday(3e-2),                 # Pre-symptomatic to symptomatic (per day)
            rate_active_to_clear    = ss.perday(2.4e-4),               # Active infection to natural clearance (per day)
            rate_exptb_to_dead      = ss.perday(0.15 * 4.5e-4),        # Extra-Pulmonary TB to Dead (per day)
            rate_smpos_to_dead      = ss.perday(4.5e-4),               # Smear Positive Pulmonary TB to Dead (per day)
            rate_smneg_to_dead      = ss.perday(0.3 * 4.5e-4),         # Smear Negative Pulmonary TB to Dead (per day)
            rate_treatment_to_clear = ss.peryear(6),                    # Treatment clearance rate (6 per year = 2 months duration)

            # Active state distribution
            active_state = ss.choice(a=TBS.all_active(), p=[0.1, 0.1, 0.60, 0.20]),

            # Relative transmissibility of each state
            rel_trans_presymp   = 0.1,                                 # Pre-symptomatic relative transmissibility
            rel_trans_smpos     = 1.0,                                 # Smear positive relative transmissibility (baseline)
            rel_trans_smneg     = 0.3,                                 # Smear negative relative transmissibility
            rel_trans_exptb     = 0.05,                                # Extra-pulmonary relative transmissibility
            rel_trans_treatment = 0.5,                                 # Treatment effect on transmissibility (multiplicative) - Multiplicative on smpos, smneg, or exptb rel_trans


            # Susceptibility parameters
            rel_sus_latentslow = 0.20,                                 # Relative susceptibility of reinfection for slow progressors
            
            # Diagnostic parameters
            cxr_asymp_sens = 1.0,                                      # Sensitivity of chest x-ray for screening asymptomatic cases

            # Heterogeneity parameters
            reltrans_het = ss.constant(v=1.0),                         # Individual-level transmission heterogeneity
        )
        self.update_pars(pars, **kwargs) 

        # Validate rates
        for k, v in self.pars.items():
            if k[:5] == 'rate_':
                assert isinstance(v, ss.TimePar), 'Rate parameters for TB must be TimePars, e.g. ss.perday(x)'

        self.define_states(
            # Core TB states
            ss.FloatArr('state', default=TBS.NONE),                    # Current TB state
            ss.FloatArr('latent_tb_state', default=TBS.NONE),          # Form of latent TB (Slow or Fast)
            ss.FloatArr('active_tb_state', default=TBS.NONE),          # Form of active TB (SmPos, SmNeg, or ExpTB)
            
            # Risk modifiers
            ss.FloatArr('rr_activation', default=1.0),                 # Multiplier on latent-to-presymp rate
            ss.FloatArr('rr_clearance', default=1.0),                  # Multiplier on active-to-clearance rate
            ss.FloatArr('rr_death', default=1.0),                      # Multiplier on active-to-death rate
            
            # Treatment and infection flags
            ss.BoolState('on_treatment', default=False),                # Currently on treatment
            ss.BoolState('ever_infected', default=False),               # Flag for ever infected
            
            # Timing variables
            ss.FloatArr('ti_presymp'),                                 # Time index of transition to pre-symptomatic
            ss.FloatArr('ti_active'),                                  # Time index of transition to active
            ss.FloatArr('ti_cur', default=0),                          # Time index of transition to current state
            
            # Transmission heterogeneity
            ss.FloatArr('reltrans_het', default=1.0),                  # Individual-level heterogeneity on infectiousness
        )

        # Initialize probability objects for state transitions
        self._init_transition_probabilities()

        # AI, and ME insist that these get overridden by the dynamic probabilities
        # self.p_latent_to_presym = ss.bernoulli(p=self.p_latent_to_presym)
        # self.p_presym_to_clear = ss.bernoulli(p=self.p_presym_to_clear)
        # self.p_presym_to_active = ss.bernoulli(p=self.p_presym_to_active)
        # self.p_active_to_clear = ss.bernoulli(p=self.p_active_to_clear)
        # self.p_active_to_death = ss.bernoulli(p=self.p_active_to_death)
        # return

    def _init_transition_probabilities(self):
        """Initialize probability objects for state transitions.
        
        This method is a placeholder for future dynamic probability initialization.
        Currently, all transition probabilities are calculated on-demand in the
        individual probability methods (p_latent_to_presym, p_presym_to_clear, etc.).
        
        The method exists to maintain consistency with the original design pattern
        and can be extended in the future to pre-calculate or cache probability
        objects for performance optimization.
        """
        # These will be calculated dynamically in the transition methods
        pass

    def p_latent_to_presym(self, sim, uids):
        """
        Calculate probability of transition from latent to pre-symptomatic TB.
        
        This method computes the daily probability that individuals in latent TB states
        will progress to active pre-symptomatic TB. The calculation uses different
        base rates for slow vs. fast progressors and applies individual risk modifiers.
        
        **Mathematical Details:**
        - Slow progressors: rate_LS_to_presym = 3e-5 per day
        - Fast progressors: rate_LF_to_presym = 6e-3 per day  
        - Individual risk: rate *= rr_activation[uids]
        - Final probability: 1 - exp(-rate * dt)
        
        **State Requirements:**
        All individuals must be in TBS.LATENT_SLOW or TBS.LATENT_FAST states.
        
        Args:
            sim (starsim.Sim): Simulation object containing time step information
            uids (numpy.ndarray): Array of individual IDs to evaluate (must be in latent states)
            
        Returns:
            numpy.ndarray: Daily transition probabilities for each individual (0.0 to 1.0)
            
        Raises:
            AssertionError: If any individuals are not in latent states (LATENT_SLOW or LATENT_FAST)
            
        Example:
            >>> probs = tb.p_latent_to_presym(sim, latent_uids)
            >>> transition_mask = np.random.random(len(probs)) < probs
            >>> new_presymp_uids = latent_uids[transition_mask]
        """
        # Validate input states
        assert np.isin(self.state[uids], TBS.all_latent()).all(), "All individuals must be in latent states"
        
        # Get base rates and ensure consistent units
        unit = self.pars.rate_LS_to_presym.unit
        ls_rate_val = self.pars.rate_LS_to_presym.rate      # 3e-5 per day
        lf_rate_val = ss.per(self.pars.rate_LF_to_presym, unit).rate # 6e-3 per day
        
        # Initialize rate array with slow rate for all individuals
        ratevals_arr = np.full(len(uids), ls_rate_val)
        
        # Apply fast rate to individuals in latent fast state
        fast_arr = self.state[uids] == TBS.LATENT_FAST
        ratevals_arr[fast_arr] = lf_rate_val
        
        # Apply individual relative risk modifiers
        ratevals_arr *= self.rr_activation[uids]
        
        # Convert to Starsim rate object and calculate probability
        rate = ss.per(ratevals_arr, unit=unit)
        prob = rate.to_prob()
        return prob

    def p_presym_to_clear(self, sim, uids):
        """
        Calculate probability of transition from pre-symptomatic to clearance.
        
        This method computes the daily probability that individuals in pre-symptomatic
        TB will clear the infection. The probability is zero for untreated individuals
        and follows the treatment clearance rate for those on treatment.
        
        **Mathematical Details:**
        - Untreated individuals: probability = 0.0 (no spontaneous clearance)
        - Treated individuals: rate_treatment_to_clear = 6 per year (2-month treatment duration)
        - Final probability: 1 - exp(-rate * dt) for treated individuals
        
        **State Requirements:**
        All individuals must be in TBS.ACTIVE_PRESYMP state.
        
        **Treatment Dependency:**
        This transition only occurs for individuals with on_treatment = True.
        
        Args:
            sim (starsim.Sim): Simulation object containing time step information
            uids (numpy.ndarray): Array of individual IDs to evaluate (must be pre-symptomatic)
            
        Returns:
            numpy.ndarray: Daily clearance probabilities for each individual (0.0 for untreated, >0.0 for treated)
            
        Raises:
            AssertionError: If any individuals are not in pre-symptomatic state (ACTIVE_PRESYMP)
            
        Example:
            >>> probs = tb.p_presym_to_clear(sim, presym_uids)
            >>> clear_mask = np.random.random(len(probs)) < probs
            >>> cleared_uids = presym_uids[clear_mask]  # Only treated individuals will clear
        """
        # Validate input states
        assert (self.state[uids] == TBS.ACTIVE_PRESYMP).all(), "All individuals must be in pre-symptomatic state"
        
        # Get treatment rate and unit
        base_rate = self.pars.rate_treatment_to_clear.rate
        unit = self.pars.rate_treatment_to_clear.unit
        
        # Create rate array - zero for untreated, treatment rate for treated
        ratevals_arr = np.zeros(len(uids))
        ratevals_arr[self.on_treatment[uids]] = base_rate
        
        # Convert to Starsim rate object and calculate probability
        rate = ss.per(ratevals_arr, unit=unit)
        prob = rate.to_prob()
        return prob

    def p_presym_to_active(self, sim, uids):
        """
        Calculate probability of transition from pre-symptomatic to active TB.
        
        This method computes the daily probability that individuals in pre-symptomatic
        TB will progress to symptomatic active TB. The probability is uniform for all
        individuals in this state, representing the natural progression of disease.
        
        **Mathematical Details:**
        - Base rate: rate_presym_to_active = 3e-2 per day
        - No individual risk modifiers applied (uniform progression)
        - Final probability: 1 - exp(-rate * dt)
        
        **State Requirements:**
        All individuals must be in TBS.ACTIVE_PRESYMP state.
        
        **Active State Assignment:**
        Upon transition, individuals are assigned to specific active states based on
        their pre-determined active_tb_state (SMPOS, SMNEG, or EXPTB).
        
        Args:
            sim (starsim.Sim): Simulation object containing time step information
            uids (numpy.ndarray): Array of individual IDs to evaluate (must be pre-symptomatic)
            
        Returns:
            numpy.ndarray: Daily progression probabilities for each individual (uniform values)
            
        Raises:
            AssertionError: If any individuals are not in pre-symptomatic state (ACTIVE_PRESYMP)
            
        Example:
            >>> probs = tb.p_presym_to_active(sim, presym_uids)
            >>> active_mask = np.random.random(len(probs)) < probs
            >>> new_active_uids = presym_uids[active_mask]
            >>> # Assign specific active states based on active_tb_state[new_active_uids]
        """
        # Validate input states
        assert (self.state[uids] == TBS.ACTIVE_PRESYMP).all(), "All individuals must be in pre-symptomatic state"
        
        # Get rate and unit
        rate_val = self.pars.rate_presym_to_active.rate
        unit = self.pars.rate_presym_to_active.unit
        
        # Create rate array with same rate for all individuals
        ratevals_arr = np.full(len(uids), rate_val)
        
        # Convert to Starsim rate object and calculate probability
        rate = ss.per(ratevals_arr, unit=unit)
        prob = rate.to_prob()
        return prob

    def p_active_to_clear(self, sim, uids):
        """
        Calculate probability of transition from active TB to clearance.
        
        This method computes the daily probability that individuals with active TB
        will clear the infection. The probability depends on treatment status and
        includes individual risk modifiers for natural clearance.
        
        **Mathematical Details:**
        - Natural clearance: rate_active_to_clear = 2.4e-4 per day
        - Treatment clearance: rate_treatment_to_clear = 6 per year (2-month duration)
        - Individual risk: rate *= rr_clearance[uids]
        - Final probability: 1 - exp(-rate * dt)
        
        **State Requirements:**
        All individuals must be in active TB states (ACTIVE_SMPOS, ACTIVE_SMNEG, or ACTIVE_EXPTB).
        
        **Treatment Effect:**
        Treated individuals use the higher treatment clearance rate instead of natural clearance.
        
        **Post-Clearance State:**
        Upon clearance, individuals return to TBS.NONE (susceptible) and can be reinfected.
        
        Args:
            sim (starsim.Sim): Simulation object containing time step information
            uids (numpy.ndarray): Array of individual IDs to evaluate (must be in active states)
            
        Returns:
            numpy.ndarray: Daily clearance probabilities for each individual (varies by treatment status)
            
        Raises:
            AssertionError: If any individuals are not in active states (SMPOS, SMNEG, or EXPTB)
            
        Example:
            >>> probs = tb.p_active_to_clear(sim, active_uids)
            >>> clear_mask = np.random.random(len(probs)) < probs
            >>> cleared_uids = active_uids[clear_mask]
            >>> # These individuals will return to susceptible state
        """
        # Validate input states
        assert np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]).all(), "All individuals must be in active states"
        
        # Get base rates and unit
        base_rate_val = self.pars.rate_active_to_clear.rate
        treatment_rate_val = self.pars.rate_treatment_to_clear.rate
        unit = self.pars.rate_active_to_clear.unit
        
        # Create rate array with base rate for all individuals
        ratevals_arr = np.full(len(uids), base_rate_val)
        
        # Apply treatment rate to those on treatment
        ratevals_arr[self.on_treatment[uids]] = treatment_rate_val
        
        # Apply individual relative risk modifiers
        ratevals_arr *= self.rr_clearance[uids]
        
        # Convert to Starsim rate object and calculate probability
        rate = ss.per(ratevals_arr, unit=unit)
        prob = rate.to_prob()
        return prob

    def p_active_to_death(self, sim, uids):
        """
        Calculate probability of transition from active TB to death.
        
        This method computes the daily probability that individuals with active TB
        will die from the disease. The probability varies significantly by active TB type
        and includes individual risk modifiers.
        
        **Mathematical Details:**
        - Smear positive: rate_smpos_to_dead = 4.5e-4 per day (highest mortality)
        - Smear negative: rate_smneg_to_dead = 0.3 * 4.5e-4 per day (30% of smear positive)
        - Extra-pulmonary: rate_exptb_to_dead = 0.15 * 4.5e-4 per day (15% of smear positive)
        - Individual risk: rate *= rr_death[uids]
        - Final probability: 1 - exp(-rate * dt)
        
        **State Requirements:**
        All individuals must be in active TB states (ACTIVE_SMPOS, ACTIVE_SMNEG, or ACTIVE_EXPTB).
        
        **Treatment Effect:**
        Treated individuals have rr_death = 0, effectively preventing TB mortality.
        
        **Death Process:**
        Upon death, individuals are marked as TBS.DEAD and removed from transmission.
        
        Args:
            sim (starsim.Sim): Simulation object containing time step information
            uids (numpy.ndarray): Array of individual IDs to evaluate (must be in active states)
            
        Returns:
            numpy.ndarray: Daily death probabilities for each individual (varies by TB type and treatment)
            
        Raises:
            AssertionError: If any individuals are not in active states (SMPOS, SMNEG, or EXPTB)
            
        Example:
            >>> probs = tb.p_active_to_death(sim, active_uids)
            >>> death_mask = np.random.random(len(probs)) < probs
            >>> death_uids = active_uids[death_mask]
            >>> # These individuals will be marked for death by the simulation framework
        """
        # Validate input states
        assert np.isin(self.state[uids], [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]).all(), "All individuals must be in active states"
        
        # Get death rates and unit
        smpos_rate_val = self.pars.rate_smpos_to_dead.rate
        smneg_rate_val = self.pars.rate_smneg_to_dead.rate
        exptb_rate_val = self.pars.rate_exptb_to_dead.rate
        unit = self.pars.rate_exptb_to_dead.unit
        
        # Create rate array with extra-pulmonary rate as default
        ratevals_arr = np.full(len(uids), exptb_rate_val)
        
        # Apply appropriate rates based on active TB type
        ratevals_arr[self.state[uids] == TBS.ACTIVE_SMPOS] = smpos_rate_val
        ratevals_arr[self.state[uids] == TBS.ACTIVE_SMNEG] = smneg_rate_val
        
        # Apply individual relative risk modifiers
        ratevals_arr *= self.rr_death[uids]
        
        # Convert to Starsim rate object and calculate probability
        rate = ss.per(ratevals_arr, unit=unit)
        prob = rate.to_prob()
        return prob

    @property
    def infectious(self):
        """
        Determine which individuals are currently infectious.
        
        An individual is considered infectious if they are in any active TB state
        or currently on treatment. This property is used by the Starsim transmission
        system to calculate transmission rates and for contact tracing.
        
        **Infectious States:**
        - ACTIVE_PRESYMP: Pre-symptomatic active TB (low transmissibility)
        - ACTIVE_SMPOS: Smear positive pulmonary TB (highest transmissibility)
        - ACTIVE_SMNEG: Smear negative pulmonary TB (moderate transmissibility)
        - ACTIVE_EXPTB: Extra-pulmonary TB (lowest transmissibility)
        - on_treatment: Currently receiving treatment (reduced transmissibility)
        
        **Transmission Rates:**
        The relative transmissibility of each state is defined by:
        - rel_trans_presymp = 0.1 (10% of baseline)
        - rel_trans_smpos = 1.0 (baseline, 100%)
        - rel_trans_smneg = 0.3 (30% of baseline)
        - rel_trans_exptb = 0.05 (5% of baseline)
        - rel_trans_treatment = 0.5 (50% reduction for treated individuals)
        
        Returns:
            numpy.ndarray: Boolean array indicating infectious status for each individual
            
        Example:
            >>> infectious_uids = tb.infectious.uids
            >>> print(f"Number of infectious individuals: {len(infectious_uids)}")
        """
        return (self.on_treatment) | (self.state==TBS.ACTIVE_PRESYMP) | (self.state==TBS.ACTIVE_SMPOS) | (self.state==TBS.ACTIVE_SMNEG) | (self.state==TBS.ACTIVE_EXPTB)

    def set_prognoses(self, uids, sources=None):
        """
        Set initial prognoses for newly infected individuals.
        
        This method is called by the Starsim framework when individuals first become
        infected with TB. It determines their latent TB progression type, sets their
        initial state, and configures individual risk factors and heterogeneity.
        
        **Progression Type Assignment:**
        - Fast progressors (10%): p_latent_fast = 0.1, progress rapidly to active disease
        - Slow progressors (90%): p_latent_slow = 0.9, progress slowly or remain latent
        
        **State Initialization:**
        - Fast progressors: Set to TBS.LATENT_FAST, become non-susceptible
        - Slow progressors: Set to TBS.LATENT_SLOW, remain susceptible to reinfection
        
        **Individual Risk Factors:**
        - rr_activation: Multiplier for latent-to-active transition (default 1.0)
        - rr_clearance: Multiplier for active-to-clearance transition (default 1.0)
        - rr_death: Multiplier for active-to-death transition (default 1.0)
        - reltrans_het: Individual transmission heterogeneity (default 1.0)
        
        **Active State Pre-assignment:**
        - active_tb_state: Pre-determined active state for future progression
        - Distribution: 10% SMPOS, 10% SMNEG, 60% SMNEG, 20% EXPTB
        
        **Reinfection Handling:**
        - Tracks new infections vs. reinfections separately
        - Slow progressors maintain susceptibility for reinfection
        - Fast progressors become non-susceptible after infection
        
        Args:
            uids (numpy.ndarray): Array of individual IDs to set prognoses for
            sources (numpy.ndarray, optional): Source of infection (not used in current implementation)
            
        Example:
            >>> # Called automatically by Starsim when new infections occur
            >>> tb.set_prognoses(new_infected_uids)
            >>> # Individuals are now assigned to latent states with risk factors
        """
        super().set_prognoses(uids, sources)

        p = self.pars

        # Decide which agents go to latent fast vs slow
        fast_uids, slow_uids = p.p_latent_fast.filter(uids, both=True)
        self.latent_tb_state[fast_uids] = TBS.LATENT_FAST
        self.latent_tb_state[slow_uids] = TBS.LATENT_SLOW
        self.state[slow_uids] = TBS.LATENT_SLOW
        self.state[fast_uids] = TBS.LATENT_FAST
        self.ti_cur[uids] = self.ti

        # Identify new vs. reinfected individuals
        new_uids = uids[~self.infected[uids]] # Previously uninfected
        reinfected_uids = uids[(self.infected[uids]) & (self.state[uids] == TBS.LATENT_FAST)]
        self.results['n_reinfected'][self.ti] = len(reinfected_uids)

        # Carry out state changes upon new infection
        self.susceptible[fast_uids] = False # N.B. Slow progressors remain susceptible!
        self.infected[uids] = True # Not needed, but useful for reporting
        self.rel_sus[slow_uids] = self.pars.rel_sus_latentslow

        # Determine active TB state for future progression
        # Filter out invalid UIDs (like -1)
        valid_uids = uids[uids >= 0]
        if len(valid_uids) > 0:
            self.active_tb_state[valid_uids] = self.pars.active_state.rvs(valid_uids)

        # Set base transmission heterogeneity
        # Filter out invalid UIDs (like -1)
        valid_uids = uids[uids >= 0]
        if len(valid_uids) > 0:
            self.reltrans_het[valid_uids] = p.reltrans_het.rvs(valid_uids)

        # Update result count of new infections 
        self.ti_infected[new_uids] = self.ti # Only update ti_infected for new...
        self.ti_infected[reinfected_uids] = self.ti # ... and reinfection uids
        self.ever_infected[uids] = True

        return

    def step(self):
        """
        Execute one simulation time step for TB disease progression.
        
        This method is called by the Starsim framework at each simulation time step
        and handles all TB-specific state transitions and updates. It processes state
        transitions in a specific order to ensure proper disease progression.
        
        **State Transition Sequence:**
        1. **Latent → Pre-symptomatic**: Slow/fast progressors advance to active disease
        2. **Pre-symptomatic → Active/Clear**: Progression to symptomatic or treatment clearance
        3. **Active → Clear**: Natural recovery or treatment-induced clearance
        4. **Active → Death**: TB mortality (varies by active state type)
        5. **Transmission Updates**: Update relative transmission rates for all states
        6. **Risk Reset**: Reset individual risk modifiers for next time step
        
        **Key Features:**
        - **Age-specific tracking**: Separate counts for individuals 15+ years old
        - **Treatment effects**: Treated individuals have reduced transmission and zero death risk
        - **State-specific rates**: Different progression rates for each TB state
        - **Individual heterogeneity**: Applies individual risk modifiers and transmission heterogeneity
        - **Result tracking**: Updates all epidemiological metrics for analysis
        
        **Transmission Rate Updates:**
        - Resets all relative transmission rates to 1.0
        - Applies state-specific transmission rates (presymp, smpos, smneg, exptb)
        - Applies individual transmission heterogeneity
        - Reduces transmission for treated individuals
        
        **Result Updates:**
        - new_active, new_active_15+: New cases of active TB
        - new_deaths, new_deaths_15+: New TB deaths
        - new_notifications_15+: New treatment initiations (15+ years)
        
        This method is the core of the TB disease model and must be called
        at each simulation time step by the Starsim framework.
        """
        # Make all the updates from the base class
        super().step()
        p = self.pars
        ti = self.ti

        # Latent --> active pre-symptomatic
        latent_uids = (((self.state == TBS.LATENT_SLOW) | (self.state == TBS.LATENT_FAST))).uids
        if len(latent_uids):
            probs = self.p_latent_to_presym(self.sim, latent_uids)
            # Convert probabilities to boolean array for indexing
            transition_mask = np.random.random(len(probs)) < probs
            new_presymp_uids = latent_uids[transition_mask]
        else:
            new_presymp_uids = np.array([], dtype=int)
        if len(new_presymp_uids):
            self.state[new_presymp_uids] = TBS.ACTIVE_PRESYMP
            self.ti_cur[new_presymp_uids] = ti
            self.ti_presymp[new_presymp_uids] = ti
            self.susceptible[new_presymp_uids] = False # No longer susceptible regardless of the latent form
        self.results['new_active'][ti] = len(new_presymp_uids)
        self.results['new_active_15+'][ti] = np.count_nonzero(self.sim.people.age[new_presymp_uids] >= 15)

        # Pre-symptomatic --> Active or Clear
        presym_uids = (self.state == TBS.ACTIVE_PRESYMP).uids
        new_clear_presymp_uids = np.array([], dtype=int)
        new_active_uids = np.array([], dtype=int)
        if len(presym_uids):
            # Pre-symptomatic --> Clear (if on treatment)
            clear_probs = self.p_presym_to_clear(self.sim, presym_uids)
            clear_mask = np.random.random(len(clear_probs)) < clear_probs
            new_clear_presymp_uids = presym_uids[clear_mask]

            # Pre-symptomatic --> Active
            active_probs = self.p_presym_to_active(self.sim, presym_uids)
            active_mask = np.random.random(len(active_probs)) < active_probs
            new_active_uids = presym_uids[active_mask]
            if len(new_active_uids):
                active_state = self.active_tb_state[new_active_uids] 
                self.state[new_active_uids] = active_state
                self.ti_cur[new_active_uids] = ti
                self.ti_active[new_active_uids] = ti

        # Active --> Susceptible via natural recovery or treatment (clear)
        active_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMNEG) | (self.state == TBS.ACTIVE_EXPTB))).uids
        if len(active_uids):
            clear_probs = self.p_active_to_clear(self.sim, active_uids)
            clear_mask = np.random.random(len(clear_probs)) < clear_probs
            new_clear_active_uids = active_uids[clear_mask]
        else:
            new_clear_active_uids = np.array([], dtype=int)
        new_clear_uids = ss.uids.cat(new_clear_presymp_uids, new_clear_active_uids)
        if len(new_clear_uids):
            # Set state and reset timers
            self.susceptible[new_clear_uids] = True
            self.infected[new_clear_uids] = False
            self.state[new_clear_uids] = TBS.NONE
            self.ti_cur[new_clear_uids] = ti
            self.active_tb_state[new_clear_uids] = TBS.NONE
            self.ti_presymp[new_clear_uids] = np.nan
            self.ti_active[new_clear_uids] = np.nan
            self.on_treatment[new_clear_uids] = False

        # Active --> Death
        active_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMNEG) | (self.state == TBS.ACTIVE_EXPTB))).uids # Recompute after clear
        if len(active_uids):
            death_probs = self.p_active_to_death(self.sim, active_uids)
            death_mask = np.random.random(len(death_probs)) < death_probs
            new_death_uids = active_uids[death_mask]
        else:
            new_death_uids = np.array([], dtype=int)
        if len(new_death_uids):
            self.sim.people.request_death(new_death_uids)
            self.state[new_death_uids] = TBS.DEAD
            self.ti_cur[new_death_uids] = ti
        self.results['new_deaths'][ti] = len(new_death_uids)
        self.results['new_deaths_15+'][ti] = np.count_nonzero(self.sim.people.age[new_death_uids] >= 15)

        # Update transmission rates based on current states
        self.rel_trans[:] = 1 # Reset

        state_reltrans = [
            (TBS.ACTIVE_PRESYMP, p.rel_trans_presymp),
            (TBS.ACTIVE_EXPTB, p.rel_trans_exptb),
            (TBS.ACTIVE_SMPOS, p.rel_trans_smpos),
            (TBS.ACTIVE_SMNEG, p.rel_trans_smneg),
        ]

        for state, reltrans in state_reltrans:
            uids = self.state == state
            self.rel_trans[uids] *= reltrans

        # Apply transmission heterogeneity
        uids = self.infectious
        self.rel_trans[uids] *= self.reltrans_het[uids]

        # Treatment reduces transmissibility
        uids = self.on_treatment
        self.rel_trans[uids] *= self.pars.rel_trans_treatment

        # Reset relative rates for the next time step
        uids = self.sim.people.auids
        self.rr_activation[uids] = 1
        self.rr_clearance[uids] = 1
        self.rr_death[uids] = 1

        return

    def start_treatment(self, uids):
        """
        Start treatment for individuals with active TB.
        
        This method initiates TB treatment for specified individuals, implementing
        the key effects of treatment on disease progression and transmission.
        
        **Treatment Effects:**
        - **Mortality Prevention**: Sets rr_death = 0 (treatment prevents TB death)
        - **Transmission Reduction**: Applies rel_trans_treatment = 0.5 (50% reduction)
        - **Clearance Enhancement**: Uses rate_treatment_to_clear = 6/year (2-month duration)
        - **State Tracking**: Marks individuals as on_treatment = True
        
        **Eligibility Requirements:**
        Only individuals in active TB states can start treatment:
        - ACTIVE_PRESYMP: Pre-symptomatic active TB
        - ACTIVE_SMPOS: Smear positive pulmonary TB
        - ACTIVE_SMNEG: Smear negative pulmonary TB
        - ACTIVE_EXPTB: Extra-pulmonary TB
        
        **Notification Tracking:**
        - Tracks new_notifications_15+ for individuals 15+ years old
        - Used for epidemiological reporting and program evaluation
        
        **Treatment Duration:**
        - Treatment continues until clearance (rate_treatment_to_clear)
        - Average treatment duration: 2 months (6 per year)
        - Treatment effects persist throughout the treatment period
        
        Args:
            uids (numpy.ndarray): Array of individual IDs to start treatment for
            
        Returns:
            int: Number of individuals who actually started treatment (only active TB cases)
            
        Example:
            >>> # Start treatment for detected active TB cases
            >>> n_treated = tb.start_treatment(detected_uids)
            >>> print(f"Started treatment for {n_treated} individuals")
        """
        if len(uids) == 0:
            return 0  # No one to treat

        rst = self.state[uids]

        # Find individuals with active TB
        is_active = np.isin(rst, [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])

        # Get the corresponding UIDs that match the active state
        tx_uids = uids[is_active]

        # Track notifications for individuals 15+
        self.results['new_notifications_15+'][self.ti] = np.count_nonzero(self.sim.people.age[tx_uids] >= 15)

        if len(tx_uids) == 0:
            return 0  # No one to treat

        # Mark individuals as being on treatment
        self.on_treatment[tx_uids] = True

        # Adjust death and clearance rates for those starting treatment
        self.rr_death[tx_uids] = 0  # People on treatment have zero death rate

        # Reduce transmission rates for people on treatment
        self.rel_trans[tx_uids] *= self.pars.rel_trans_treatment

        # Return the number of individuals who started treatment
        return len(tx_uids)

    def step_die(self, uids):
        """
        Handle death events for TB-infected individuals.
        
        This method is called by the Starsim framework when individuals die from
        any cause (TB-related or other). It ensures that deceased individuals
        cannot transmit TB or become infected after death.
        
        **State Cleanup:**
        - **Susceptibility**: Sets susceptible = False (cannot be infected)
        - **Infection Status**: Sets infected = False (no longer infected)
        - **Transmission**: Sets rel_trans = 0 (cannot transmit to others)
        
        **Death Sources:**
        - **TB Deaths**: Individuals who died from active TB (TBS.DEAD state)
        - **Other Deaths**: Individuals who died from other causes while infected
        
        **Framework Integration:**
        - Calls super().step_die(uids) to handle base class death processing
        - Ensures proper cleanup of TB-specific states
        - Maintains consistency with Starsim death handling
        
        **Important Notes:**
        - This method handles ALL deaths, not just TB deaths
        - TB-specific death tracking is handled in the step() method
        - Deceased individuals are removed from all TB calculations
        
        Args:
            uids (numpy.ndarray): Array of individual IDs who have died
            
        Example:
            >>> # Called automatically by Starsim when deaths occur
            >>> tb.step_die(deceased_uids)
            >>> # Deceased individuals are now removed from TB transmission
        """
        if len(uids) == 0:
            return # Nothing to do

        super().step_die(uids)
        
        # Ensure deceased individuals cannot transmit or get infected
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.rel_trans[uids] = 0
        return

    def init_results(self):
        """
        Initialize result tracking variables for TB epidemiological outcomes.
        
        This method sets up all the result variables used to track TB-specific
        epidemiological outcomes throughout the simulation. It defines both
        basic state counts and derived epidemiological indicators.
        
        **State Count Results:**
        - n_latent_slow, n_latent_fast: Counts of individuals in latent states
        - n_active_presymp, n_active_smpos, n_active_smneg, n_active_exptb: Active state counts
        - n_active: Combined count of all active TB cases
        - n_infectious: Total number of infectious individuals
        
        **Age-Specific Results (15+ years):**
        - All active state counts have 15+ variants for adult-focused analysis
        - new_active_15+, new_deaths_15+: Age-specific incidence and mortality
        - new_notifications_15+: Treatment initiations for adults
        - n_detectable_15+: Detectable cases including CXR screening
        
        **Incidence and Cumulative Measures:**
        - new_active, new_deaths: Daily incidence of new cases and deaths
        - cum_active, cum_deaths: Cumulative counts over simulation period
        - cum_active_15+, cum_deaths_15+: Age-specific cumulative measures
        
        **Derived Epidemiological Indicators:**
        - prevalence_active: Active TB prevalence (active cases / total population)
        - incidence_kpy: Incidence per 1,000 person-years
        - deaths_ppy: Death rate per person-year
        - n_reinfected: Number of reinfection events
        
        **Detection and Treatment Metrics:**
        - new_notifications_15+: New TB notifications (treatment initiations)
        - n_detectable_15+: Detectable cases (smear pos/neg + CXR-screened pre-symptomatic)
        
        This method is called automatically by the Starsim framework during
        simulation initialization and should not be called manually.
        """
        super().init_results()
        
        self.define_results(
            # State counts
            ss.Result('n_latent_slow',         dtype=int, label='Latent Slow'),
            ss.Result('n_latent_fast',         dtype=int, label='Latent Fast'),
            ss.Result('n_active',              dtype=int, label='Active (Combined)'),
            ss.Result('n_active_presymp',      dtype=int, label='Active Pre-Symptomatic'),
            ss.Result('n_active_presymp_15+',  dtype=int, label='Active Pre-Symptomatic, 15+'),
            ss.Result('n_active_smpos',        dtype=int, label='Active Smear Positive'),
            ss.Result('n_active_smpos_15+',    dtype=int, label='Active Smear Positive, 15+'),
            ss.Result('n_active_smneg',        dtype=int, label='Active Smear Negative'),
            ss.Result('n_active_smneg_15+',    dtype=int, label='Active Smear Negative, 15+'),
            ss.Result('n_active_exptb',        dtype=int, label='Active Extra-Pulmonary'),
            ss.Result('n_active_exptb_15+',    dtype=int, label='Active Extra-Pulmonary, 15+'),
            
            # Incidence and cumulative measures
            ss.Result('new_active',            dtype=int, label='New Active'),
            ss.Result('new_active_15+',        dtype=int, label='New Active, 15+'),
            ss.Result('cum_active',            dtype=int, label='Cumulative Active'),
            ss.Result('cum_active_15+',        dtype=int, label='Cumulative Active, 15+'),
            
            # Mortality measures
            ss.Result('new_deaths',            dtype=int, label='New Deaths'),
            ss.Result('new_deaths_15+',        dtype=int, label='New Deaths, 15+'),
            ss.Result('cum_deaths',            dtype=int, label='Cumulative Deaths'),
            ss.Result('cum_deaths_15+',        dtype=int, label='Cumulative Deaths, 15+'),
            
            # Transmission and detection measures
            ss.Result('n_infectious',          dtype=int, label='Number Infectious'),
            ss.Result('n_infectious_15+',      dtype=int, label='Number Infectious, 15+'),
            ss.Result('prevalence_active',     dtype=float, scale=False, label='Prevalence (Active)'),
            ss.Result('incidence_kpy',         dtype=float, scale=False, label='Incidence per 1,000 person-years'),
            ss.Result('deaths_ppy',            dtype=float, label='Death per person-year'), 
            ss.Result('n_reinfected',          dtype=int, label='Number reinfected'), 
            ss.Result('new_notifications_15+', dtype=int, label='New TB notifications, 15+'),
            ss.Result('n_detectable_15+',      dtype=float, label='Detectable cases (Sm+ + Sm- + CXR-screened pre-symptomatic)'),
        )
        return

    def update_results(self):
        """
        Update result tracking variables for the current time step.
        
        This method is called by the Starsim framework at each time step to
        calculate and store all TB-specific epidemiological metrics. It updates
        both basic state counts and derived epidemiological indicators.
        
        **State Count Updates:**
        - Counts individuals in each TB state (latent, active, infectious)
        - Calculates age-specific counts for individuals 15+ years old
        - Updates total active TB and infectious case counts
        
        **Detection Metrics:**
        - n_detectable_15+: Calculates detectable cases including:
          * All smear positive and negative cases (100% detectable)
          * CXR-screened pre-symptomatic cases (cxr_asymp_sens sensitivity)
        
        **Derived Epidemiological Indicators:**
        - **prevalence_active**: Active TB prevalence = n_active / total_alive_population
        - **incidence_kpy**: Incidence per 1,000 person-years = 1000 * new_infections / (alive_pop * dt_year)
        - **deaths_ppy**: Death rate per person-year = new_deaths / (alive_pop * dt_year)
        
        **Age-Specific Calculations:**
        - All 15+ metrics use age >= 15 filter for adult-focused analysis
        - Important for TB epidemiology as adult cases are typically more severe
        
        **Population Scaling:**
        - Uses sim.people.alive to get current living population
        - Accounts for deaths and births during simulation
        - Ensures rates are calculated relative to current population
        
        **Time Step Integration:**
        - Uses sim.t.dt_year for time-based rate calculations
        - Ensures consistent units across different time step sizes
        
        This method is called automatically by the Starsim framework at each
        time step and should not be called manually.
        """
        super().update_results()
        res = self.results
        ti = self.ti
        ti_infctd = self.ti_infected
        dty = self.sim.t.dt_year
        n_alive = np.count_nonzero(self.sim.people.alive)

        # Update state counts
        res.n_latent_slow[ti]       = np.count_nonzero(self.state == TBS.LATENT_SLOW)
        res.n_latent_fast[ti]       = np.count_nonzero(self.state == TBS.LATENT_FAST)
        res.n_active_presymp[ti]    = np.count_nonzero(self.state == TBS.ACTIVE_PRESYMP)
        res['n_active_presymp_15+'][ti] = np.count_nonzero((self.sim.people.age>=15) & (self.state == TBS.ACTIVE_PRESYMP))
        res.n_active_smpos[ti]      = np.count_nonzero(self.state == TBS.ACTIVE_SMPOS) 
        res['n_active_smpos_15+'][ti] = np.count_nonzero((self.sim.people.age>=15) & (self.state == TBS.ACTIVE_SMPOS))
        res.n_active_smneg[ti]      = np.count_nonzero(self.state == TBS.ACTIVE_SMNEG)
        res['n_active_smneg_15+'][ti] = np.count_nonzero((self.sim.people.age>=15) & (self.state == TBS.ACTIVE_SMNEG))
        res.n_active_exptb[ti]      = np.count_nonzero(self.state == TBS.ACTIVE_EXPTB)
        res['n_active_exptb_15+'][ti] = np.count_nonzero((self.sim.people.age>=15) & (self.state == TBS.ACTIVE_EXPTB))
        res.n_active[ti]            = np.count_nonzero(np.isin(self.state, TBS.all_active()))
        res.n_infectious[ti]        = np.count_nonzero(self.infectious)
        res['n_infectious_15+'][ti] = np.count_nonzero(self.infectious & (self.sim.people.age>=15))

        # Calculate detectable cases (including CXR screening)
        res['n_detectable_15+'][ti] = np.dot( self.sim.people.age >= 15,
            np.isin(self.state, [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG]) + \
                self.pars.cxr_asymp_sens * (self.state == TBS.ACTIVE_PRESYMP) )

        # Calculate rates if population is alive
        if n_alive > 0:
            res.prevalence_active[ti] = res.n_active[ti] / n_alive 
            res.incidence_kpy[ti]     = 1_000 * np.count_nonzero(ti_infctd == ti) / (n_alive * dty)
            res.deaths_ppy[ti]        = res.new_deaths[ti] / (n_alive * dty)

        return

    def finalize_results(self):
        """
        Finalize result calculations after simulation completion.
        
        This method is called by the Starsim framework after the simulation
        has completed to calculate cumulative measures and finalize any results
        that depend on the complete simulation history.
        
        **Cumulative Calculations:**
        - **cum_deaths**: Cumulative TB deaths = cumsum(new_deaths)
        - **cum_deaths_15+**: Cumulative TB deaths for adults 15+ = cumsum(new_deaths_15+)
        - **cum_active**: Cumulative active TB cases = cumsum(new_active)
        - **cum_active_15+**: Cumulative active TB cases for adults 15+ = cumsum(new_active_15+)
        
        **Purpose:**
        - Provides total burden measures over the entire simulation period
        - Enables calculation of cumulative incidence and mortality rates
        - Supports long-term epidemiological analysis and program evaluation
        
        **Usage:**
        - Called automatically by Starsim at simulation end
        - Results are available in self.results after completion
        - Used for final analysis and reporting
        
        **Integration:**
        - Calls super().finalize_results() to handle base class finalization
        - Ensures compatibility with Starsim result system
        
        This method is called automatically by the Starsim framework at the
        end of simulation and should not be called manually.
        """
        super().finalize_results()
        res = self.results
        
        # Calculate cumulative measures
        res['cum_deaths']     = np.cumsum(res['new_deaths'])
        res['cum_deaths_15+'] = np.cumsum(res['new_deaths_15+'])
        res['cum_active']     = np.cumsum(res['new_active'])
        res['cum_active_15+'] = np.cumsum(res['new_active_15+'])
        
        return

    def plot(self):
        """
        Create a basic plot of all TB simulation results.
        
        This method generates a matplotlib figure showing the time series of all
        tracked TB epidemiological measures over the simulation period. It provides
        a quick visual overview of the simulation outcomes.
        
        **Plot Contents:**
        - **Time Series**: All result variables plotted against simulation time
        - **Multiple Metrics**: Includes state counts, incidence, mortality, and prevalence
        - **Legend**: Each result variable is labeled in the legend
        - **Automatic Scaling**: Matplotlib handles axis scaling automatically
        
        **Excluded Variables:**
        - timevec: Excluded as it represents the x-axis (time)
        - All other result variables are included in the plot
        
        **Plot Features:**
        - **Line Plot**: Each result variable plotted as a separate line
        - **Legend**: Automatic legend with result variable names
        - **Time Axis**: X-axis represents simulation time steps
        - **Value Axis**: Y-axis represents result variable values
        
        **Usage:**
        - Useful for quick visual inspection of simulation results
        - Can be customized by modifying the returned figure
        - Suitable for basic analysis and presentation
        
        **Limitations:**
        - All variables on same plot may have different scales
        - No automatic subplot organization
        - Basic styling without customization
        
        Returns:
            matplotlib.figure.Figure: Figure containing the time series plot of all TB results
            
        Example:
            >>> fig = tb.plot()
            >>> fig.show()  # Display the plot
            >>> fig.savefig('tb_results.png')  # Save to file
        """
        fig = plt.figure()
        for rkey in self.results.keys():
            if rkey == 'timevec':
                continue
            plt.plot(self.results['timevec'], self.results[rkey], label=rkey.title())
        plt.legend()
        return fig

