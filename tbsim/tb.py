import numpy as np
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd

from enum import IntEnum

__all__ = ['TB', 'TBS']

class TBS(IntEnum):
    NONE            = -1    # No TB
    LATENT_SLOW     = 0     # Latent TB, slow progression
    LATENT_FAST     = 1     # Latent TB, fast progression
    ACTIVE_PRESYMP  = 2     # Active TB, pre-symptomatic
    ACTIVE_SMPOS    = 3     # Active TB, smear positive
    ACTIVE_SMNEG    = 4     # Active TB, smear negative
    ACTIVE_EXPTB    = 5     # Active TB, extra-pulmonary
    DEAD            = 8     # TB death
    PROTECTED       = 100


class TB(ss.Infection):
    def __init__(self, pars=None, **kwargs):
        super().__init__()

        self.define_pars(
            init_prev = ss.bernoulli(0.01),                            # Initial seed infections
            beta = ss.peryear(0.025),                                      # Infection probability
            p_latent_fast = ss.bernoulli(0.1),                         # Probability of latent fast as opposed to latent slow
            rate_LS_to_presym       = ss.perday(3e-5),                 # Latent Slow to Active Pre-Symptomatic (per day)            
            rate_LF_to_presym       = ss.perday(6e-3),                 # Latent Fast to Active Pre-Symptomatic (per day)
            rate_presym_to_active   = ss.perday(3e-2),                 # Pre-symptomatic to symptomatic (per day)
            rate_active_to_clear    = ss.perday(2.4e-4),               # Active infection to natural clearance (per day)
            rate_exptb_to_dead      = ss.perday(0.15 * 4.5e-4),        # Extra-Pulmonary TB to Dead (per day)
            rate_smpos_to_dead      = ss.perday(4.5e-4),               # Smear Positive Pulmonary TB to Dead (per day)
            rate_smneg_to_dead      = ss.perday(0.3 * 4.5e-4),         # Smear Negative Pulmonary TB to Dead (per day)
            rate_treatment_to_clear = ss.peryear(12/2),                # 2 months is the duration treatment implies 6 per year

            active_state = ss.choice(a=[TBS.ACTIVE_EXPTB, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG], p=[0.1, 0.65, 0.25]),

            # Relative transmissibility of each state
            rel_trans_presymp   = 0.1, # 0.0274
            rel_trans_smpos     = 1.0,
            rel_trans_smneg     = 0.3, # 0.25
            rel_trans_exptb     = 0.05,
            rel_trans_treatment = 0.5, # Multiplicative on smpos, smneg, or exptb rel_trans

            rel_sus_latentslow = 0.20, # Relative susceptibility of reinfection for slow progressors
            
            cxr_asymp_sens = 1.0, # Sensitivity of chest x-ray for screening asymptomatic cases

            reltrans_het = ss.constant(v=1.0),
        )
        self.update_pars(pars, **kwargs) 

        # Validate rates
        for k, v in self.pars.items():
            if k[:5] == 'rate_':
                assert isinstance(v, ss.TimePar), 'Rate parameters for TB must be TimePars, e.g. ss.perday(x)'

        self.define_states(
            # Initialize states specific to TB:
            ss.FloatArr('state', default=TBS.NONE),             # One state to rule them all?
            ss.FloatArr('latent_tb_state', default=TBS.NONE),   # Form of latent TB (Slow or Fast)
            ss.FloatArr('active_tb_state', default=TBS.NONE),   # Form of active TB (SmPos, SmNeg, or ExpTB)
            ss.FloatArr('rr_activation', default=1.0),          # Multiplier on the latent-to-presymp rate
            ss.FloatArr('rr_clearance', default=1.0),           # Multiplier on the active-to-susceptible rate
            ss.FloatArr('rr_death', default=1.0),               # Multiplier on the active-to-dead rate
            ss.BoolState('on_treatment', default=False),
            ss.BoolState('ever_infected', default=False),           # Flag for ever infected

            ss.FloatArr('ti_presymp'),
            ss.FloatArr('ti_active'),
            ss.FloatArr('ti_cur', default=0),                   # Time index of transition into the current state

            ss.FloatArr('reltrans_het', default=1.0),           # Individual-level heterogeneity on infectiousness, acts in addition to stage-based rates
        )

        self.p_latent_to_presym_dist = ss.bernoulli(p=self.p_latent_to_presym)
        self.p_presym_to_clear_dist = ss.bernoulli(p=self.p_presym_to_clear)
        self.p_presym_to_active_dist = ss.bernoulli(p=self.p_presym_to_active)
        self.p_active_to_clear_dist = ss.bernoulli(p=self.p_active_to_clear)
        self.p_active_to_death_dist = ss.bernoulli(p=self.p_active_to_death)

        return
    
    def p_latent_to_presym(self, sim, uids):
        """
        Calculate the probability of progression from latent TB infection to pre-symptomatic active disease.
        
        This method models the critical transition where latent TB infection becomes active but not yet
        symptomatic. The progression rate differs significantly between fast and slow progressors:
        - Fast progressors (typically 10% of infections) progress rapidly to active disease
        - Slow progressors (90% of infections) have much lower progression rates and may remain
          latent for years or decades
        
        The progression probability is influenced by individual risk factors (rr_activation) which
        can represent factors like age, immune status, or comorbidities.
        """
        if len(uids) == 0:
            return np.array([])
            
        # Validate that all agents are in valid latent states
        valid_states = [TBS.LATENT_FAST, TBS.LATENT_SLOW]
        invalid_states = ~np.isin(self.state[uids], valid_states)
        if invalid_states.any():
            invalid_uids = uids[invalid_states]
            invalid_state_values = self.state[invalid_uids]
            raise ValueError(f"Agents {invalid_uids} are not in valid latent states. "
                           f"Expected {valid_states}, got {invalid_state_values}")
        
        # Initialize probability array
        prob = np.zeros(len(uids))
        
        # Calculate probabilities for each latent state type
        slow_mask = self.state[uids] == TBS.LATENT_SLOW
        fast_mask = self.state[uids] == TBS.LATENT_FAST
        
        # Apply slow progression rate to slow progressors
        if slow_mask.any():
            slow_uids = uids[slow_mask]
            prob[slow_mask] = self.pars.rate_LS_to_presym.to_prob(
                self.dt, scale=self.rr_activation[slow_uids]
            )
        
        # Apply fast progression rate to fast progressors
        if fast_mask.any():
            fast_uids = uids[fast_mask]
            prob[fast_mask] = self.pars.rate_LF_to_presym.to_prob(
                self.dt, scale=self.rr_activation[fast_uids]
            )
        
        return prob

    def p_presym_to_clear(self, sim, uids):
        """
        Calculate the probability of spontaneous clearance from pre-symptomatic TB without treatment.
        
        This method models the natural resolution of TB infection during the pre-symptomatic phase,
        where the immune system successfully controls the infection before symptoms develop. This
        represents the body's natural defense mechanisms against TB, which can occur in individuals
        with strong immune responses.
        
        Note that this clearance rate is typically very low, as most pre-symptomatic cases will
        progress to symptomatic disease rather than clear spontaneously.
        """
        if len(uids) == 0:
            return np.array([])
            
        # Validate that all agents are in pre-symptomatic state
        invalid_states = self.state[uids] != TBS.ACTIVE_PRESYMP
        if invalid_states.any():
            invalid_uids = uids[invalid_states]
            invalid_state_values = self.state[invalid_uids]
            raise ValueError(f"Agents {invalid_uids} are not in pre-symptomatic state. "
                           f"Expected {TBS.ACTIVE_PRESYMP}, got {invalid_state_values}")
        
        prob = np.zeros(len(uids))
        on_treatment_mask = self.on_treatment[uids]
        
        if on_treatment_mask.any():
            treated_uids = uids[on_treatment_mask]
            prob[on_treatment_mask] = self.pars.rate_treatment_to_clear.to_prob(self.dt)
        
        return prob

    def p_presym_to_active(self, sim, uids):
        """
        Calculate the probability of progression from pre-symptomatic to symptomatic active TB.
        
        This method models the development of clinical symptoms in individuals with active TB infection.
        During the pre-symptomatic phase, the infection is active and potentially infectious but the
        individual has not yet developed noticeable symptoms. This transition represents the onset of
        clinical disease, which typically occurs within weeks to months after the infection becomes active.
        
        The progression to symptomatic disease is a key event in TB natural history, as it often
        triggers health-seeking behavior and diagnosis.
        """
        if len(uids) == 0:
            return np.array([])
            
        # Validate that all agents are in pre-symptomatic state
        invalid_states = self.state[uids] != TBS.ACTIVE_PRESYMP
        if invalid_states.any():
            invalid_uids = uids[invalid_states]
            invalid_state_values = self.state[invalid_uids]
            raise ValueError(f"Agents {invalid_uids} are not in pre-symptomatic state. "
                           f"Expected {TBS.ACTIVE_PRESYMP}, got {invalid_state_values}")
        
        return self.pars.rate_presym_to_active.to_prob(self.dt)

    def p_active_to_clear(self, sim, uids):
        """
        Calculate the probability of TB clearance (cure) from active symptomatic disease.
        
        This method models both natural recovery and treatment-mediated cure from active TB disease.
        Natural clearance represents spontaneous resolution of the infection, which is rare but can
        occur in individuals with strong immune responses. Treatment-mediated clearance represents
        successful completion of anti-TB therapy, which is the primary mechanism for TB cure.
        
        The clearance probability is influenced by individual factors (rr_clearance) which can represent
        treatment adherence, drug resistance, or underlying health conditions that affect treatment
        effectiveness.
        """
        if len(uids) == 0:
            return np.array([])
            
        # Validate that all agents are in valid active states
        valid_states = [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]
        invalid_states = ~np.isin(self.state[uids], valid_states)
        if invalid_states.any():
            invalid_uids = uids[invalid_states]
            invalid_state_values = self.state[invalid_uids]
            raise ValueError(f"Agents {invalid_uids} are not in valid active TB states. "
                           f"Expected {valid_states}, got {invalid_state_values}")
        
        # Initialize with natural clearance rate
        prob = self.pars.rate_active_to_clear.to_prob(self.dt, scale=self.rr_clearance[uids])
        
        # Override with treatment rate for those on treatment
        on_treatment_mask = self.on_treatment[uids]
        if on_treatment_mask.any():
            treated_uids = uids[on_treatment_mask]
            prob[on_treatment_mask] = self.pars.rate_treatment_to_clear.to_prob(
                self.dt, scale=self.rr_clearance[treated_uids]
            )
        
        return prob

    def p_active_to_death(self, sim, uids):
        """
        Calculate the probability of TB-related death from active symptomatic disease.
        
        This method models TB mortality, which varies significantly by disease form and severity.
        Smear-positive pulmonary TB has the highest mortality rate due to high bacterial burden and
        severe lung damage. Smear-negative pulmonary TB has moderate mortality, while extra-pulmonary
        TB mortality depends on the specific site of infection and accessibility to treatment.
        
        The death probability is influenced by individual risk factors (rr_death) which can represent
        age, immune status, comorbidities, or delays in diagnosis and treatment.
        """
        if len(uids) == 0:
            return np.array([])
            
        # Validate that all agents are in valid active states
        valid_states = [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]
        invalid_states = ~np.isin(self.state[uids], valid_states)
        if invalid_states.any():
            invalid_uids = uids[invalid_states]
            invalid_state_values = self.state[invalid_uids]
            raise ValueError(f"Agents {invalid_uids} are not in valid active TB states. "
                           f"Expected {valid_states}, got {invalid_state_values}")
        
        # Initialize probability array
        prob = np.zeros(len(uids))
        
        # Calculate death probabilities for each active state type
        state_rate_pairs = [
            (TBS.ACTIVE_SMPOS, self.pars.rate_smpos_to_dead),
            (TBS.ACTIVE_SMNEG, self.pars.rate_smneg_to_dead),
            (TBS.ACTIVE_EXPTB, self.pars.rate_exptb_to_dead)
        ]
        
        for state, rate in state_rate_pairs:
            state_mask = self.state[uids] == state
            if state_mask.any():
                state_uids = uids[state_mask]
                prob[state_mask] = rate.to_prob(self.dt, scale=self.rr_death[state_uids])
        
        return prob

    @property
    def infectious(self):
        """
        Infectious if in any of the active states
        """
        return (self.on_treatment) | (self.state==TBS.ACTIVE_PRESYMP) | (self.state==TBS.ACTIVE_SMPOS) | (self.state==TBS.ACTIVE_SMNEG) | (self.state==TBS.ACTIVE_EXPTB)

    def set_prognoses(self, uids, from_uids=None, sources=None):
        super().set_prognoses(uids, from_uids)

        p = self.pars

        # Decide which agents go to latent fast vs slow
        fast_uids, slow_uids = p.p_latent_fast.filter(uids, both=True)
        self.latent_tb_state[fast_uids] = TBS.LATENT_FAST
        self.latent_tb_state[slow_uids] = TBS.LATENT_SLOW
        self.state[slow_uids] = TBS.LATENT_SLOW
        self.state[fast_uids] = TBS.LATENT_FAST
        self.ti_cur[uids] = self.ti

        new_uids = uids[~self.infected[uids]] # Previously uninfected

        # Only consider as "reinfected" if slow --> fast
        reinfected_uids = uids[(self.infected[uids]) & (self.state[uids] == TBS.LATENT_FAST) ]
        self.results['n_reinfected'][self.ti] = len(reinfected_uids)

        # Carry out state changes upon new infection
        self.susceptible[fast_uids] = False # N.B. Slow progressors remain susceptible!
        self.infected[uids] = True # Not needed, but useful for reporting
        self.rel_sus[slow_uids] = self.pars.rel_sus_latentslow

        # Determine active TB state
        self.active_tb_state[uids] = self.pars.active_state.rvs(uids)

        # Set base transmission heterogeneity
        self.reltrans_het[uids] = p.reltrans_het.rvs(uids)

        # Update result count of new infections 
        self.ti_infected[new_uids] = self.ti # Only update ti_infected for new...
        self.ti_infected[reinfected_uids] = self.ti # ... and reinfection uids
        self.ever_infected[uids] = True

        return

    def step(self):
        # Make all the updates from the SIR model 
        super().step()
        p = self.pars
        ti = self.ti

        # Latent --> active pre-symptomatic
        latent_uids = (((self.state == TBS.LATENT_SLOW) | (self.state == TBS.LATENT_FAST))).uids
        new_presymp_uids = self.p_latent_to_presym_dist.filter(latent_uids)
        if len(new_presymp_uids):
            self.state[new_presymp_uids] = TBS.ACTIVE_PRESYMP
            self.ti_cur[new_presymp_uids] = ti
            self.ti_presymp[new_presymp_uids] = ti
            self.susceptible[new_presymp_uids] = False # No longer susceptible regardless of the latent form
        self.results['new_active'][ti] = len(new_presymp_uids)
        self.results['new_active_15+'][ti] = np.count_nonzero(self.sim.people.age[new_presymp_uids] >= 15)

        # Pre symp --> Active
        presym_uids = (self.state == TBS.ACTIVE_PRESYMP).uids
        new_clear_presymp_uids = ss.uids()
        if len(presym_uids):
            # Pre symp --> Clear
            new_clear_presymp_uids = self.p_presym_to_clear_dist.filter(presym_uids)

            new_active_uids = self.p_presym_to_active_dist.filter(presym_uids)
            if len(new_active_uids):
                active_state = self.active_tb_state[new_active_uids] 
                self.state[new_active_uids] = active_state
                self.ti_cur[new_active_uids] = ti
                self.ti_active[new_active_uids] = ti

        # Active --> Susceptible via natural recovery or as accelerated by treatment (clear)
        active_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMNEG) | (self.state == TBS.ACTIVE_EXPTB))).uids
        new_clear_active_uids = self.p_active_to_clear_dist.filter(active_uids)
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
        new_death_uids = self.p_active_to_death_dist.filter(active_uids)
        if len(new_death_uids):
            self.sim.people.request_death(new_death_uids)
            self.state[new_death_uids] = TBS.DEAD
            self.ti_cur[new_death_uids] = ti
        self.results['new_deaths'][ti] = len(new_death_uids)
        self.results['new_deaths_15+'][ti] = np.count_nonzero(self.sim.people.age[new_death_uids] >= 15)

        # Set rel_trans
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

        # Transmission heterogeneity
        uids = self.infectious
        self.rel_trans[uids] *= self.reltrans_het[uids]

        # Treatment can reduce transmissibility
        uids = self.on_treatment
        self.rel_trans[uids] *= self.pars.rel_trans_treatment

        # Reset relative rates for the next time step, they will be recalculated
        uids = self.sim.people.auids
        self.rr_activation[uids] = 1
        self.rr_clearance[uids] = 1
        self.rr_death[uids] = 1

        return

    def start_treatment(self, uids):
        """ Start treatment for active TB """
        if len(uids) == 0:
            return 0  # No one to treat

        rst = self.state[uids]

        #find individuals with active TB
        is_active = np.isin(rst, [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])

        # Get the corresponding UIDs that match the active state
        tx_uids = uids[is_active]

        self.results['new_notifications_15+'][self.ti] = np.count_nonzero(self.sim.people.age[tx_uids] >= 15)

        if len(tx_uids) == 0:
            return 0  # No one to treat

        # Mark the individuals as being on treatment
        self.on_treatment[tx_uids] = True

        # Adjust death and clearance rates for those starting treatment
        self.rr_death[tx_uids] = 0  # People on treatment have zero death rate

        # Reduce transmission rates for people on treatment
        self.rel_trans[tx_uids] *= self.pars.rel_trans_treatment

        # Return the number of individuals who started treatment
        return len(tx_uids)

    def step_die(self, uids):
        if len(uids) == 0:
            return # Nothing to do

        super().step_die(uids)
        # Make sure these agents do not transmit or get infected after death
        self.susceptible[uids] = False
        self.infected[uids] = False
        #self.state[uids] = TBS.NONE
        self.rel_trans[uids] = 0
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        
        self.define_results(
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
            ss.Result('new_active',            dtype=int, label='New Active'),
            ss.Result('new_active_15+',        dtype=int, label='New Active, 15+'),
            ss.Result('cum_active',            dtype=int, label='Cumulative Active'),
            ss.Result('cum_active_15+',        dtype=int, label='Cumulative Active, 15+'),
            ss.Result('new_deaths',            dtype=int, label='New Deaths'),
            ss.Result('new_deaths_15+',        dtype=int, label='New Deaths, 15+'),
            ss.Result('cum_deaths',            dtype=int, label='Cumulative Deaths'),
            ss.Result('cum_deaths_15+',        dtype=int, label='Cumulative Deaths, 15+'),
            ss.Result('n_infectious',          dtype=int, label='Number Infectious'),
            ss.Result('n_infectious_15+',      dtype=int, label='Number Infectious, 15+'),
            ss.Result('prevalence_active',     dtype=float, scale=False, label='Prevalence (Active)'),
            ss.Result('incidence_kpy',         dtype=float, scale=False, label='Incidence per 1,000 person-years'),
            ss.Result('deaths_ppy',            dtype=float, label='Death per person-year'), 
            ss.Result('n_reinfected',          dtype=int, label='Number reinfected'), 
            ss.Result('new_notifications_15+', dtype=int, label='New TB notifications, 15+'),
            ss.Result('n_detectable_15+',      dtype=float, label='Sm+ plus SM- plus cxr_asymp_sens * pre-symptomatic'),  # Move to analyzer?
        )
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.ti
        ti_infctd = self.ti_infected
        dty = self.sim.t.dt_year
        n_alive = np.count_nonzero(self.sim.people.alive)

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
        res.n_active[ti]            = np.count_nonzero(np.isin(self.state, [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]))
        res.n_infectious[ti]        = np.count_nonzero(self.infectious)
        res['n_infectious_15+'][ti] = np.count_nonzero(self.infectious & (self.sim.people.age>=15))

        res['n_detectable_15+'][ti] = np.dot( self.sim.people.age >= 15,
            np.isin(self.state, [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG]) + \
                self.pars.cxr_asymp_sens * (self.state == TBS.ACTIVE_PRESYMP) )

        if n_alive > 0:
            res.prevalence_active[ti] = res.n_active[ti] / n_alive 
            res.incidence_kpy[ti]     = 1_000 * np.count_nonzero(ti_infctd == ti) / (n_alive * dty)
            res.deaths_ppy[ti]        = res.new_deaths[ti] / (n_alive * dty)

        return

    def finalize_results(self):
        super().finalize_results()
        res = self.results
        res['cum_deaths']     = np.cumsum(res['new_deaths'])
        res['cum_deaths_15+'] = np.cumsum(res['new_deaths_15+'])
        res['cum_active']     = np.cumsum(res['new_active'])
        res['cum_active_15+'] = np.cumsum(res['new_active_15+'])
        
        return

    def plot(self):
        fig = plt.figure()
        for rkey in self.results.keys(): #['latent_slow', 'latent_fast', 'active', 'active_presymp', 'active_smpos', 'active_smneg', 'active_exptb']:
            if rkey == 'timevec':
                continue
            plt.plot(self.results['timevec'], self.results[rkey], label=rkey.title())
        plt.legend()
        return fig
