import numpy as np
import starsim as ss
from tbsim import TBS, TB
from tbsim.tb_lshtm import TB_LSHTM, TBSL, TB_LSHTM_Acute

__all__ = ['HealthSeekingBehavior']

_DISEASE_ENUM = {
    TB_LSHTM_Acute: TBSL,
    TB_LSHTM:       TBSL,
    TB:             TBS,
}

class TBIntervention(ss.Intervention):
    _state_method = None

    def init_post(self):
        """Detect the TB variant in the simulation and resolve the state enum"""
        super().init_post()
        self.tb = (
            getattr(self.sim.diseases, 'tb', None)
            or getattr(self.sim.diseases, 'tb_lshtm', None)
            or self.sim.diseases[0]
        )
        for cls in type(self.tb).__mro__:
            if cls in _DISEASE_ENUM:
                self.state_enum = _DISEASE_ENUM[cls]
                break
        if self._state_method is not None:
            self.states = getattr(self.state_enum, self._state_method)()

class HealthSeekingBehavior(TBIntervention):
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
            ss.BoolArr('symptoms_initialized', default=False),
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
        self._ensure_tb_resolved()
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
        if len(uids) == 0:
            return
        
        for state_name, dist in self._symptom_dists.items():
            if dist is None:
                continue  # Skip if no distribution for this symptom
            
            symptom_state = getattr(self, state_name)
            symptom_state[uids] = dist.rvs(uids)
        
        # Mark these agents as having symptoms initialized
        self.symptoms_initialized[uids] = True

    def _ensure_tb_resolved(self):
        """Resolve tb and states if init_post ran before diseases were ready."""
        if hasattr(self, 'tb') and self.tb is not None:
            return
        sim = self.sim
        self.tb = (
            getattr(sim.diseases, 'tb', None)
            or getattr(sim.diseases, 'tb_lshtm', None)
            or (sim.diseases[0] if len(sim.diseases) else None)
        )
        if self.tb is None:
            raise TypeError('HealthSeekingBehavior requires TB or TB_LSHTM disease module')
        for cls in type(self.tb).__mro__:
            if cls in _DISEASE_ENUM:
                self.state_enum = _DISEASE_ENUM[cls]
                break
        else:
            raise TypeError(f'HealthSeekingBehavior requires TB or TB_LSHTM, got {type(self.tb).__name__}')
        self.states = getattr(self.state_enum, self._state_method)()

    def step(self):
        """Identify eligible agents, assign symptoms to new ones, and sample who seeks care this step."""
        self._ensure_tb_resolved()
        sim = self.sim
        ppl = sim.people
        t = sim.now
        if self.pars.start is not None and t < self.pars.start:
            return
        if self.pars.stop is not None and t > self.pars.stop:
            return

        active_uids = np.where(np.isin(self.tb.state, self.states))[0]
        active_uids = active_uids[ppl.alive[active_uids]]

        # Assign symptoms to newly eligible agents (those who just became symptomatic)
        newly_eligible = active_uids[~self.symptoms_initialized[active_uids]]
        if len(newly_eligible) > 0:
            self._assign_symptoms(newly_eligible)

        not_yet_sought = active_uids[~ppl.sought_care[active_uids]]
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
        ppl.sought_care[seeking_uids] = True
        self.tb.start_treatment(seeking_uids)
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
        self._ensure_tb_resolved()
        ppl = self.sim.people
        self.results['new_sought_care'][self.ti] = self._new_seekers_count
        self.results['n_sought_care'][self.ti] = np.count_nonzero(ppl.sought_care)
        active_uids = np.where(np.isin(self.tb.state, self.states))[0]
        active_uids = active_uids[ppl.alive[active_uids]]
        not_yet_sought = active_uids[~ppl.sought_care[active_uids]]
        self.results['n_eligible'][self.ti] = len(not_yet_sought)
        
