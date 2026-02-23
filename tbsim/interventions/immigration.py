
import numpy as np
import starsim as ss
import sciris as sc

__all__ = ['Immigration']

# Placeholder for default parameters
_ = None

class Immigration(ss.Demographics):
    """
    Add new people to the simulation population over time.
    
    This module simulates immigration by adding new agents to the population
    at specified rates. New immigrants can have different characteristics
    (age, TB natural-history state, etc.) than the existing population.
    
    This implementation supports only the LSHTM-spectrum TB natural history models:
    :class:`tbsim.TB_LSHTM` and :class:`tbsim.TB_LSHTM_Acute`.

    Parameters:
    -----------
    immigration_rate : float or ss.TimePar
        Immigration intensity in people per year (e.g. ``100`` or ``ss.peryear(100)``).
    age_distribution : dict or array-like
        Age distribution for new immigrants
    tb_state_distribution : dict
        Distribution of TB natural-history states for new immigrants when using
        ``tbsim.TB_LSHTM`` or ``tbsim.TB_LSHTM_Acute``. Keys are ``TBSL`` names
        (e.g. ``"SUSCEPTIBLE"``, ``"INFECTION"``, ``"ASYMPTOMATIC"``) and values
        are probabilities that sum to 1.
    rel_immigration : float
        Relative immigration rate multiplier
    """
    
    def __init__(self, pars=None, immigration_rate=_, rel_immigration=_, 
                 age_distribution=_, tb_state_distribution=_, **kwargs):
        super().__init__()
        
        self.define_pars(
            immigration_rate=ss.peryear(10),  # immigrants per year by default
            rel_immigration=1.0,  # Relative immigration rate multiplier
            age_distribution=None,  # Will use population age distribution if None
            # Default LSHTM-spectrum distribution for arriving migrants.
            # This is intentionally conservative on active TB (<<1%), and leaves most
            # of the burden in INFECTION (latent infection) rather than active disease.
            # Users should override this to match their setting and target migrant group.
            tb_state_distribution=dict(
                SUSCEPTIBLE=0.6517,
                INFECTION=0.33,
                CLEARED=0.007,
                RECOVERED=0.005,
                NON_INFECTIOUS=0.0008,
                ASYMPTOMATIC=0.0015,
                SYMPTOMATIC=0.0010,
                TREATED=0.003,
                TREATMENT=0.0,
            ),
        )
        self.update_pars(pars, **kwargs)
        
        # Define states for immigration tracking
        self.define_states(
            ss.IntArr('hhid', default=-1),  # Household ID for immigrants
            ss.BoolState('is_immigrant', default=False),
            ss.FloatArr('immigration_time', default=np.nan),  # When they immigrated
            ss.FloatArr('age_at_immigration', default=np.nan),  # Age when they immigrated
            # TB state at immigration. Stored as an int code (TBSL.*).
            # Note: Starsim does not provide a portable StrArr across versions.
            ss.IntArr('immigration_tb_status', default=-1),
        )
        
        # Initialize tracking variables
        self.n_immigrants = 0
        
        return
    
    def init_pre(self, sim):
        """Initialize with simulation information."""
        super().init_pre(sim)
        
        # Set up age distribution for immigrants
        if self.pars.age_distribution is None:
            # Use a default age distribution if none provided
            self.pars.age_distribution = {
                0: 0.15,   # 15% children 0-4
                5: 0.20,   # 20% children 5-14
                15: 0.25,  # 25% young adults 15-29
                30: 0.20,  # 20% adults 30-49
                50: 0.15,  # 15% middle-aged 50-64
                65: 0.05,  # 5% elderly 65+
            }
        
        return
    
    def init_results(self):
        """Initialize results tracking."""
        super().init_results()
        self.define_results(
            ss.Result('n_immigrants', dtype=int, label='Number of immigrants'),
        )
        return
    
    def _dt_years(self):
        """Return timestep length in years."""
        # Prefer Starsim's dt_year if available
        dt_year = getattr(self.sim.t, 'dt_year', None)
        if dt_year is not None:
            return float(dt_year)
        # Fallback to days
        return float(self.t.dt) / 365.25

    def _rate_per_year(self):
        """Return immigration intensity in people/year (scalar float)."""
        r = self.pars.immigration_rate
        if r is None:
            return 0.0

        # Starsim time parameters usually have .value and .unit
        if hasattr(r, 'value') and hasattr(r, 'unit'):
            # Convert whatever unit is used into per-year using its unit
            # Example: ss.peryear(10) has unit=ss.years(1), so factor=1
            try:
                factor = ss.years(1) / r.unit
                return float(r.value) * float(factor) * float(self.pars.rel_immigration)
            except Exception:
                return float(r.value) * float(self.pars.rel_immigration)

        # Plain numeric
        return float(r) * float(self.pars.rel_immigration)

    def expected_immigrants_per_timestep(self):
        """Return expected number of immigrants in the current timestep."""
        lam = self._rate_per_year() * self._dt_years()
        if not np.isfinite(lam) or lam < 0:
            lam = 0.0
        return lam

    def _sample_ages(self, n):
        """Sample ages (years) for n immigrants."""
        if n <= 0:
            return np.empty(0, dtype=float)

        ad = self.pars.age_distribution
        if not isinstance(ad, dict) or len(ad) == 0:
            return np.random.uniform(0, 85, n)

        # Keys are treated as lower bounds; upper bound is next key (or 85 for last)
        keys = sorted(ad.keys())
        probs = np.array([ad[k] for k in keys], dtype=float)
        if probs.sum() <= 0:
            return np.random.uniform(0, 85, n)
        probs = probs / probs.sum()

        # Allocate counts exactly using multinomial (avoids rounding bias)
        counts = np.random.multinomial(n, probs)
        ages = np.empty(n, dtype=float)
        idx = 0
        for i, (k, c) in enumerate(zip(keys, counts)):
            if c == 0:
                continue
            lo = float(k)
            hi = float(keys[i + 1]) if i + 1 < len(keys) else 85.0
            ages[idx:idx + c] = np.random.uniform(lo, hi, c)
            idx += c

        # Shuffle to avoid any ordering artifacts
        np.random.shuffle(ages)
        return ages

    def _get_immigrant_characteristics(self, n_immigrants):
        """Generate characteristics for new immigrants."""
        if n_immigrants == 0:
            return {}
        return {'ages': self._sample_ages(n_immigrants)}

    def _get_tb_module(self):
        """Return the TB_LSHTM(_Acute) disease module if available."""
        diseases = getattr(self.sim, 'diseases', None)
        if diseases is None:
            return None
        # Starsim sometimes stores diseases in a dict-like container keyed by module name
        if hasattr(diseases, 'tb'):  # older convention; still must be LSHTM
            mod = diseases.tb
            try:
                from tbsim.tb_lshtm import TB_LSHTM
                return mod if isinstance(mod, TB_LSHTM) else None
            except Exception:
                return None

        # Prefer selecting by type rather than by key name (e.g. 'tb_lshtm_acute')
        try:
            from tbsim.tb_lshtm import TB_LSHTM
        except Exception:
            TB_LSHTM = None

        try:
            items = list(diseases.items())
        except Exception:
            items = []

        for _k, mod in items:
            if TB_LSHTM is not None and isinstance(mod, TB_LSHTM):
                return mod

        return None

    def _init_tb_lshtm(self, tb, new_uids):
        """Initialize TB_LSHTM / TB_LSHTM_Acute state and bookkeeping for new agents."""
        from tbsim.tb_lshtm import TBSL, TB_LSHTM, TB_LSHTM_Acute

        if not isinstance(tb, TB_LSHTM):
            raise TypeError('LSHTM initializer called with non-LSHTM TB module')

        dist = sc.dcp(self.pars.tb_state_distribution) or {}
        if len(dist) == 0:
            raise ValueError('tb_state_distribution must be provided for TB_LSHTM(_Acute)')

        # Convert keys to TBSL values (accept keys as strings or TBSL)
        state_vals = []
        probs = []
        for k, v in dist.items():
            if v is None:
                continue
            if isinstance(k, str):
                if not hasattr(TBSL, k):
                    raise KeyError(f'Unknown LSHTM state "{k}" in tb_state_distribution')
                sv = getattr(TBSL, k)
            else:
                sv = k
            state_vals.append(int(sv))
            probs.append(float(v))

        probs = np.array(probs, dtype=float)
        if probs.sum() <= 0:
            raise ValueError('tb_state_distribution probabilities must sum to > 0')
        probs = probs / probs.sum()

        # Disallow ACUTE unless using TB_LSHTM_Acute
        has_acute = int(TBSL.ACUTE) in state_vals
        if has_acute and not isinstance(tb, TB_LSHTM_Acute):
            raise ValueError('tb_state_distribution includes ACUTE but TB module is not TB_LSHTM_Acute')

        n = len(new_uids)
        sampled = np.random.choice(state_vals, size=n, p=probs)
        tb.state[new_uids] = sampled
        tb.state_next[new_uids] = sampled
        tb.ti_next[new_uids] = np.inf

        # Flags and timing
        infected_states = np.array([int(TBSL.INFECTION), int(TBSL.NON_INFECTIOUS), int(TBSL.ASYMPTOMATIC),
                                    int(TBSL.SYMPTOMATIC), int(TBSL.TREATMENT)])
        if isinstance(tb, TB_LSHTM_Acute):
            infected_states = np.append(infected_states, int(TBSL.ACUTE))

        reinfectable_states = np.array([int(TBSL.SUSCEPTIBLE), int(TBSL.CLEARED), int(TBSL.RECOVERED), int(TBSL.TREATED)])

        is_inf = np.isin(sampled, infected_states)
        tb.infected[new_uids] = is_inf
        tb.susceptible[new_uids] = np.isin(sampled, reinfectable_states)
        tb.ever_infected[new_uids] = sampled != int(TBSL.SUSCEPTIBLE)

        tb.ti_infected[new_uids] = np.where(is_inf, self.ti, -np.inf)
        tb.on_treatment[new_uids] = sampled == int(TBSL.TREATMENT)

        # rel_sus: defaults to 1; RECOVERED/TREATED have modifiers
        tb.rel_sus[new_uids] = 1.0
        tb.rel_sus[new_uids[sampled == int(TBSL.RECOVERED)]] = float(tb.pars.rr_rec)
        tb.rel_sus[new_uids[sampled == int(TBSL.TREATED)]] = float(tb.pars.rr_treat)

        # rel_trans: defaults to 1; ASYMPTOMATIC uses kappa; ACUTE uses alpha; DEAD uses 0
        tb.rel_trans[new_uids] = 1.0
        tb.rel_trans[new_uids[sampled == int(TBSL.ASYMPTOMATIC)]] = float(tb.pars.trans_asymp)
        if isinstance(tb, TB_LSHTM_Acute):
            tb.rel_trans[new_uids[sampled == int(TBSL.ACUTE)]] = float(tb.pars.trans_acute)
        tb.rel_trans[new_uids[sampled == int(TBSL.DEAD)]] = 0.0

        return sampled
    
    def step(self):
        """Add immigrants to the population."""
        lam = self.expected_immigrants_per_timestep()
        n_immigrants = np.random.poisson(lam) if lam > 0 else 0
        
        if n_immigrants == 0:
            self.n_immigrants = 0
            return []
        
        # Get characteristics for new immigrants
        characteristics = self._get_immigrant_characteristics(n_immigrants)
        
        # Add new people to the population
        new_uids = self.sim.people.grow(n_immigrants)
        
        # Set ages for new immigrants
        self.sim.people.age[new_uids] = characteristics['ages']
        
        # Initialize TB state for new immigrants (LSHTM-spectrum models only)
        tb = self._get_tb_module()
        if tb is None:
            raise RuntimeError(
                'Immigration requires an LSHTM TB disease module (TB_LSHTM or TB_LSHTM_Acute) '
                'to be present in sim.diseases.'
            )
        sampled_states = self._init_tb_lshtm(tb, new_uids)
        self.immigration_tb_status[new_uids] = sampled_states
        
        # Update household assignments for new immigrants
        # Assign them to existing households or create new ones
        self.assign_immigrants_to_households(new_uids)

        # Populate immigrant tracking fields
        self.is_immigrant[new_uids] = True
        self.immigration_time[new_uids] = float(self.ti)
        self.age_at_immigration[new_uids] = self.sim.people.age[new_uids]
        
        self.n_immigrants = n_immigrants
        return new_uids
    
    def assign_immigrants_to_households(self, new_uids):
        """Assign new immigrants to households."""
        # Get household network
        hh_net = None
        for net in self.sim.networks.values():
            if hasattr(net, 'hhid'):
                hh_net = net
                break
        
        if hh_net is None:
            return  # No household network found
        
        # For simplicity, assign immigrants to existing households
        # In a more sophisticated model, you might create new households
        existing_hhids = np.unique(hh_net.hhid[hh_net.hhid >= 0])
        
        if len(existing_hhids) > 0:
            # Assign each immigrant to a random existing household
            assigned_hhids = np.random.choice(existing_hhids, size=len(new_uids))
            hh_net.hhid[new_uids] = assigned_hhids
            self.hhid[new_uids] = assigned_hhids
        else:
            # Create new households for immigrants
            for i, uid in enumerate(new_uids):
                hh_net.hhid[uid] = i
                self.hhid[uid] = i
    
    def update_results(self):
        """Update results tracking."""
        super().update_results()
        if hasattr(self, 'results') and self.results is not None:
            self.results['n_immigrants'][self.ti] = int(self.n_immigrants)
        return



