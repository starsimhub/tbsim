import numpy as np
import starsim as ss
from tbsim.interventions.base import TBIntervention

__all__ = ['HealthSeekingBehavior']

#TODO: what is the best way to handle elegibility for care-seeking? 
# maybe we should allow the researcher to specify the symptoms and let the
# intervention handle the identification of eligible agents?

class HealthSeekingBehavior(TBIntervention):
    """
    Identifies individuals with active TB who seek healthcare.

    This intervention models the **patient delay** component of TB diagnostic delay,
    representing the time from symptom recognition to initial healthcare contact.
    It identifies which symptomatic individuals seek care and makes them available
    for downstream diagnostic interventions. At each time step it:
    
    1. Identifies agents who have just become care-seeking-eligible (e.g., newly
       symptomatic) and assigns clinical symptoms based on configured prevalence rates
    2. Samples which eligible agents seek care this step (based on probability or rate)
    3. Sets their ``sought_care`` state to True, making them eligible for diagnostic
       testing by downstream interventions (e.g., ``TBDiagnostic``)

    **Important:** This intervention does NOT perform diagnosis or initiate treatment.
    It ONLY identifies care-seeking individuals. Diagnosis and treatment must be handled
    by separate interventions (e.g., ``TBDiagnostic`` for testing, sensitivity/specificity,
    and treatment initiation).

    **Workflow:** HealthSeekingBehavior → TBDiagnostic → Treatment
    
    The intervention works with all TB model variants (``TB``, ``TB_LSHTM``,
    ``TB_LSHTM_Acute``), automatically detecting the model and identifying the
    appropriate care-seeking-eligible disease states.

    Parameters (pars)
    ----------------
    prob : float or ss.Dist, default 0.1
        Per-time-step probability that an eligible agent seeks care.
        Ignored when ``initial_care_seeking_rate`` is provided.
    initial_care_seeking_rate : ss.Rate, optional
        Annual or daily care-seeking rate as an ``ss.Rate`` object
        (e.g., ``ss.perday(0.005)`` or ``ss.peryear(2.0)``).
        Internally converted to per-step probability.
        When provided, ``prob`` is ignored.
    single_use : bool, default True
        If True, intervention expires after the first care-seeking event
        (one-time screening scenario). Set to False for continuous care-seeking
        behavior over time (e.g., modeling patient-initiated care).
    start : date-like, optional
        Simulation date before which the intervention is inactive.
    stop : date-like, optional
        Simulation date after which the intervention is inactive.
    cough_rate : float or ss.Dist or None, default None
        Prevalence of persistent cough (>2-3 weeks) among care-seeking-eligible
        agents. Can be a float (e.g., 0.85 for 85%), an ss.Dist object for
        variable rates, or None to skip symptom assignment. Typical TB literature
        values: 70-90% (Drain et al., Clin Infect Dis 2018).
    fever_rate : float or ss.Dist or None, default None
        Prevalence of fever/night sweats among care-seeking-eligible agents.
        Can be float, ss.Dist, or None. Typical values: 40-60%.
    weight_loss_rate : float or ss.Dist or None, default None
        Prevalence of weight loss/fatigue among care-seeking-eligible agents.
        Can be float, ss.Dist, or None. Typical values: 60-80%.
    hemoptysis_rate : float or ss.Dist or None, default None
        Prevalence of hemoptysis (coughing blood) among care-seeking-eligible
        agents. Can be float, ss.Dist, or None. Typical values: 10-20%
        (more common with cavitary disease).
    chest_pain_rate : float or ss.Dist or None, default None
        Prevalence of chest pain among care-seeking-eligible agents.
        Can be float, ss.Dist, or None. Typical values: 30-50%.

    States (agent-level)
    --------------------
    sought_care : bool
        Whether each agent has previously sought care.
    symptoms_initialized : bool
        Whether symptoms have been assigned to each agent (set when agent
        becomes care-seeking-eligible).
    has_cough : bool
        Whether each agent presents persistent cough symptom.
    has_fever : bool
        Whether each agent presents fever/night sweats symptom.
    has_weight_loss : bool
        Whether each agent presents weight loss/fatigue symptom.
    has_hemoptysis : bool
        Whether each agent presents hemoptysis symptom.
    has_chest_pain : bool
        Whether each agent presents chest pain symptom.

    Results (recorded each time step)
    ----------------------------------
    new_sought_care : int
        Number of agents who sought care this step.
    n_sought_care : int
        Cumulative number of agents who have ever sought care.
    n_eligible : int
        Number of agents currently care-seeking-eligible who have not yet
        sought care.

    Notes
    -----
    **Intervention Scope:**
    
    This intervention models ONLY the **patient delay** (symptom recognition to healthcare
    contact). It does NOT model:
    
    - Health system delay (time from facility contact to diagnosis)
    - Diagnostic testing (sensitivity/specificity)
    - Treatment initiation
    
    These components must be handled by downstream interventions. The typical workflow is:
    
    1. **HealthSeekingBehavior**: Sets ``sought_care=True`` for care-seeking agents
    2. **TBDiagnostic**: Tests agents with ``sought_care=True``, applies diagnostic
       accuracy, sets ``diagnosed=True``, and initiates treatment for positive cases
    3. **TB disease model**: Manages treatment outcomes via ``start_treatment()``
    
    **Patient delay (symptomatic → care-seeking):**
    
    The gap from entering a symptomatic state to presenting for care is modeled
    implicitly. Each time step, eligible agents who have not yet sought care are
    sampled with a per-step probability (from ``prob`` or from ``initial_care_seeking_rate``).
    The delay for an agent is therefore the number of steps until they are first
    sampled as seeking care—i.e. the time to first "success" in a repeated
    Bernoulli (or rate-based) process. This implies a geometric (discrete-time)
    or exponential (continuous-time) distribution of delays: mean delay ≈ 1/p
    in steps (or 1/rate in time units). The intervention does *not* assign a
    per-agent delay (e.g. from a log-normal or Gamma); all eligible agents
    share the same per-step probability until they seek care.
    
    **Care-seeking-eligible states** vary by TB model. State and rate names refer
    to the model in use and thus to one of two status transition diagrams; see
    the linked implementations for the full diagrams and parameter lists:
    
    - **LSHTM diagram**: :class:`tbsim.tb_lshtm.TB_LSHTM`, :class:`tbsim.tb_lshtm.TB_LSHTM_Acute`.
      Care-seeking-eligible is :class:`tbsim.tb_lshtm.TBSL` SYMPTOMATIC only. Rate names
      (e.g. ``asy_sym``, ``theta``) are in the ``pars`` of those classes.
    - **Legacy TB diagram**: :class:`tbsim.tb.TB`. Care-seeking-eligible states are
      :class:`tbsim.tb.TBS` ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB. Rate names
      (e.g. ``rate_presym_to_active``, ``rate_smpos_to_dead``) are in ``TB.pars``.
    
    **Symptom assignment** occurs when agents transition into care-seeking-eligible
    states, representing symptom onset that triggers health-seeking. Symptom states
    persist throughout the simulation and can be used by downstream interventions
    (e.g., diagnostic algorithms with symptom-dependent sensitivity).
    
    **Burn-in**: Symptoms are assigned to agents already in eligible states at
    simulation start, allowing equilibrium simulations with pre-existing symptom
    distributions.

    See Also
    --------
    :class:`tbsim.interventions.tb_diagnostic.TBDiagnostic` : Diagnostic testing and accuracy
    :class:`tbsim.tb_lshtm.TB_LSHTM` : LSHTM TB model (implementation and state diagram)
    :class:`tbsim.tb.TB` : Legacy TB model (implementation and state diagram)

    Examples
    --------
    Complete diagnostic pathway with care-seeking and testing::

        from tbsim.interventions import HealthSeekingBehavior, TBDiagnostic
        
        sim = ss.Sim(
            diseases=TB_LSHTM(),
            interventions=[
                HealthSeekingBehavior(pars={
                    'initial_care_seeking_rate': ss.peryear(2.0),
                    'single_use': False,
                    'cough_rate': 0.85,
                    'fever_rate': 0.55,
                }),
                TBDiagnostic(pars={
                    'sensitivity': 0.85,
                    'specificity': 0.95,
                }),
            ],
            ...
        )
        sim.run()

    Care-seeking only (for theoretical scenarios with perfect diagnosis)::

        # Note: Without TBDiagnostic, care-seekers won't be diagnosed or treated
        sim = ss.Sim(
            diseases=TB_LSHTM(),
            interventions=HealthSeekingBehavior(pars={
                'initial_care_seeking_rate': ss.peryear(2.0),
                'single_use': False,
            }),
            ...
        )
    
    Active case-finding campaign::
    
        sim = ss.Sim(
            diseases=TB_LSHTM(),
            interventions=[
                HealthSeekingBehavior(pars={
                    'prob': 0.8,  # 80% participate
                    'single_use': True,
                    'start': '2025-06-01',
                }),
                TBDiagnostic(pars={'sensitivity': 0.90}),
            ],
            ...
        )
    """

    _state_method = 'care_seeking_eligible'

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            prob=0.1,
            initial_care_seeking_rate=None,
            single_use=True,
            start=None,
            stop=None,
            cough_rate=None,
            fever_rate=None,
            weight_loss_rate=None,
            hemoptysis_rate=None,
            chest_pain_rate=None,
        )
        self.update_pars(pars=pars, **kwargs)
        self.define_states(
            ss.BoolArr('sought_care', default=False),
            ss.BoolArr('symptoms_initialized', default=False),  # Track symptom initialization
            ss.BoolArr('has_cough', default=False),
            ss.BoolArr('has_fever', default=False),
            ss.BoolArr('has_weight_loss', default=False),
            ss.BoolArr('has_hemoptysis', default=False),
            ss.BoolArr('has_chest_pain', default=False),
        )
        self._new_seekers_count = 0
        self._care_seeking_dist = None
        self._symptom_dists = {}  # Store symptom distributions for reuse

    def init_post(self):
        """Validate parameters and build the sampling distribution.

        Called automatically after the simulation is initialised.
        If ``initial_care_seeking_rate`` is set, it is validated here and
        converted to a probability each step.  Otherwise a Bernoulli
        distribution is created from ``prob`` and reused every step.
        
        Also initializes symptom distributions and assigns symptoms to 
        any agents already in care-seeking-eligible states (burn-in).
        """
        super().init_post()
        
        # Prepare symptom distributions for reuse
        symptom_params = [
            ('cough_rate', 'has_cough'),
            ('fever_rate', 'has_fever'),
            ('weight_loss_rate', 'has_weight_loss'),
            ('hemoptysis_rate', 'has_hemoptysis'),
            ('chest_pain_rate', 'has_chest_pain'),
        ]
        
        for rate_param, state_name in symptom_params:
            rate_val = self.pars[rate_param]
            if rate_val is None:
                self._symptom_dists[state_name] = None
                continue
            
            # Handle both percentages (float) and distributions (ss.Dist)
            if isinstance(rate_val, ss.Dist):
                # It's already a distribution, initialize and store it
                trace = f'{self.name or "HealthSeekingBehavior"}.{state_name}'
                rate_val.init(trace=trace, sim=self.sim, module=self)
                self._symptom_dists[state_name] = rate_val
            elif isinstance(rate_val, (int, float)):
                # It's a percentage, convert to bernoulli distribution
                trace = f'{self.name or "HealthSeekingBehavior"}.{state_name}'
                dist = ss.bernoulli(p=rate_val)
                dist.init(trace=trace, sim=self.sim, module=self)
                self._symptom_dists[state_name] = dist
            else:
                raise TypeError(
                    f"{rate_param} must be a float (percentage), ss.Dist, or None; "
                    f"got {type(rate_val).__name__}"
                )
        
        # Assign symptoms to any agents already in care-seeking-eligible states (burn-in)
        eligible_uids = np.where(np.isin(self.tb.state, self.states))[0]
        if len(eligible_uids) > 0:
            self._assign_symptoms(eligible_uids)
        
        # Initialize care-seeking distribution
        rate = self.pars.initial_care_seeking_rate
        if rate is not None:
            if not isinstance(rate, ss.Rate):
                raise TypeError(
                    f"initial_care_seeking_rate must be an ss.Rate (e.g. ss.perday(0.1), ss.peryear(0.1)), got {type(rate).__name__}"
                )
            return
        trace = self.name or 'HealthSeekingBehavior.care_seeking'
        if isinstance(self.pars.prob, ss.Dist):
            dist = self.pars.prob
        else:
            dist = ss.bernoulli(p=self.pars.prob)
        dist.init(trace=trace, sim=self.sim, module=self)
        self._care_seeking_dist = dist
    
    def _assign_symptoms(self, uids):
        """Assign symptoms to agents based on symptom rate distributions.
        
        Parameters
        ----------
        uids : array
            Agent UIDs to assign symptoms to
        """
        if len(uids) == 0:
            return
        
        for state_name, dist in self._symptom_dists.items():
            if dist is None:
                continue  # Skip if no distribution for this symptom
            
            symptom_state = getattr(self, state_name)
            symptom_state[uids] = dist.rvs(uids)
        
        # Mark these agents as having symptoms initialized
        self.symptoms_initialized[uids] = True

    def step(self):
        """Identify eligible agents, assign symptoms to new ones, and sample who seeks care this step."""
        sim = self.sim
        t = sim.now
        if self.pars.start is not None and t < self.pars.start:
            return
        if self.pars.stop is not None and t > self.pars.stop:
            return

        active_uids = np.where(np.isin(self.tb.state, self.states))[0]
        
        # Assign symptoms to newly eligible agents (those who just became symptomatic)
        newly_eligible = active_uids[~self.symptoms_initialized[active_uids]]
        if len(newly_eligible) > 0:
            self._assign_symptoms(newly_eligible)
        
        not_yet_sought = active_uids[~self.sought_care[active_uids]]
        self._new_seekers_count = 0

        if len(not_yet_sought) == 0:
            return

        if self.pars.initial_care_seeking_rate is not None:
            rate = self.pars.initial_care_seeking_rate
            if not isinstance(rate, ss.Rate):
                raise TypeError(
                    f"initial_care_seeking_rate must be an ss.Rate (e.g. ss.perday(0.1)), got {type(rate).__name__}"
                )
            p = rate.to_prob()
            dist = ss.bernoulli(p=p)
            trace = self.name or 'HealthSeekingBehavior.care_seeking'
            dist.init(trace=trace, sim=sim, module=self)
            seeking_uids = dist.filter(not_yet_sought)
        else:
            seeking_uids = self._care_seeking_dist.filter(not_yet_sought)

        if len(seeking_uids) == 0:
            return
        self._new_seekers_count = len(seeking_uids)
        self.sought_care[seeking_uids] = True
        # Note: Agents are now available for diagnostic intervention (e.g., TBDiagnostic)
        # This intervention does NOT initiate treatment - that happens after diagnosis
        if self.pars.single_use:
            self.expired = True

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_sought_care', dtype=int),
            ss.Result('n_sought_care', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )

    def update_results(self):
        """Record care-seeking counts and remaining eligible agents."""
        self.results['new_sought_care'][self.ti] = self._new_seekers_count
        self.results['n_sought_care'][self.ti] = np.count_nonzero(self.sought_care)
        active_uids = np.where(np.isin(self.tb.state, self.states))[0]
        not_yet_sought = active_uids[~self.sought_care[active_uids]]
        self.results['n_eligible'][self.ti] = len(not_yet_sought)
