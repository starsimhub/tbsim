import numpy as np
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
from tb_acf import expon_LTV

from enum import IntEnum

__all__ = ['TB_LSHTM', 'TB_LSHTM_Acute', 'TBSL']

class TBSL(IntEnum):
    SUSCEPTIBLE  = -1    # No TB
    INFECTION    = 0     # Has latent infection
    CLEARED      = 1     # Cleared infection
    UNCONFIRMED  = 2     # Unconfirmed TB
    RECOVERED    = 3     # Recovered from TB
    ASYMPTOMATIC = 4     # Asymptomatic TB
    SYMPTOMATIC  = 5     # Symptomatic TB
    TREATMENT    = 6     # On treatment
    TREATED      = 7     # Treated
    DEAD         = 8     # TB death
    ACUTE        = 9     # Acute infection ~ (only in TB_LSHTM_ACUTE)


class TB_LSHTM(ss.Infection):
    def __init__(self, pars=None, **kwargs):
        super().__init__(name=kwargs.pop('name', None), label=kwargs.pop('label', None))

        self.define_pars(
            init_prev = ss.bernoulli(0.01), # Initial seed infections
            beta = ss.beta(0.25, unit='month'), # Infection probability
            kappa = 0.82,                   # Relative transmission from asymptomatic TB
            pi    = 0.21,                   # Relative risk of reinfection after recovery from unconfirmed TB
            rho   = 3.15,                   # Relative risk of reinfection after treatment completion
            infcle    = ss.years(ss.expon(1/1.90)),   # Rate of clearance from infection per year
            infunc    = ss.years(ss.expon(1/0.16)),   # Rate of progression from infection to unconfirmed TB per year
            infasy    = ss.years(ss.expon(1/0.06)),   # Rate of progression from infection to asymptomatic TB per year
            uncrec    = ss.years(ss.expon(1/0.18)),   # Rate of recovery from unconfirmed TB per year
            uncasy    = ss.years(ss.expon(1/0.25)),   # Rate of progression from unconfirmed TB to asymptomatic TB per year
            asyunc    = ss.years(ss.expon(1/1.66)),   # Rate of recovery from asymptomatic to unconfirmed TB per year
            asysym    = ss.years(ss.expon(1/0.88)),   # Rate of progression from asymptomatic to symptomatic TB per year
            symasy    = ss.years(ss.expon(1/0.54)),   # Rate of recovery from symptomatic to asymptomatic TB per year
            #theta     = ss.years(expon_LTV(ss.date('1800-01-01'), 0.46, ss.date('2050-01-01'), 0.71)),   # Rate of treatment initiation from symptomatic TB per year
            theta     = ss.years(ss.expon(1/0.46)),   # Rate of treatment initiation from symptomatic TB per year, initial value
            delta     = ss.years(ss.expon(1/2.00)),   # Rate of treatment completion per year 
            #phi       = ss.years(expon_LTV(ss.date('1800-01-01'), 0.63, ss.date('2050-01-01'), 0.09)),   # Rate of treatment failure per year
            phi       = ss.years(ss.expon(1/0.63)),   # Rate of treatment failure per year, initial value
            #mutb      = ss.years(expon_LTV(ss.date('1800-01-01'), 0.34, ss.date('2050-01-01'), 0.17)),   # TB-specific mortality rate per year
            mutb      = ss.years(ss.expon(1/0.34)),   # TB-specific mortality rate per year, initial value
            mu        = ss.years(ss.expon(1/0.014)),  # Background mortality rate per year

            cxr_asymp_sens = 1.0, # Sensitivity of chest x-ray for screening asymptomatic cases
        )
        self.update_pars(pars, **kwargs) 

        self.define_states(
            # Initialize states specific to TB:
            ss.FloatArr('state', default=TBSL.SUSCEPTIBLE),      # One state to rule them all?
            ss.FloatArr('state_next', default=TBSL.INFECTION),   # Next state
            ss.FloatArr('ti_next', default=np.inf),             # Time of next transition
            ss.State('on_treatment', default=False),
            ss.State('ever_infected', default=False),
            ss.FloatArr('ti_infected', default=-np.inf),         # Time of infection
        )

        return

    @property
    def infectious(self):
        """
        Infectious if in any of the active states
        """
        return (self.state==TBSL.ASYMPTOMATIC) | (self.state==TBSL.SYMPTOMATIC)

    def set_prognoses(self, uids, from_uids=None):
        super().set_prognoses(uids, from_uids)
        if len(uids) == 0:
            return # Nothing to do

        p = self.pars

        # Carry out state changes upon new infection
        self.susceptible[uids] = False
        self.infected[uids] = True # Not needed, but useful for reporting
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti

        self.state[uids] = TBSL.INFECTION

        # INFECTION --> CLEARED, UNCONFIRMED, or ASYMPTOMATIC
        self.state_next[uids], self.ti_next[uids] = self.transition(uids, to={
            TBSL.CLEARED: self.pars.infcle,
            TBSL.UNCONFIRMED: self.pars.infunc,
            TBSL.ASYMPTOMATIC: self.pars.infasy
        })

        return

    def transition(self, uids, to):
        """ Transition between states """
        if len(uids) == 0:
            return np.array([]), np.array([])

        state_next = np.full(len(uids), fill_value=TBSL.SUSCEPTIBLE)
        ti_next = np.full(len(uids), fill_value=np.inf)
        ti = self.ti

        # TODO: Consider SSA algorithm
        ti_state = np.zeros((len(to), len(uids)))
        for idx, rate in enumerate(to.values()):
            ti_state[idx,:] = ti + rate.rvs(uids)

        state_next_idx = ti_state.argmin(axis=0)

        state_next = np.array(list(to.keys()))[state_next_idx]
        ti_next = ti_state.min(axis=0)

        # Potentially faster compared to doing min when we already have argmin:
        #entries = zip(state_next_idx, range(len(uids)))
        #ti_next = ti_state[entries]

        return state_next, ti_next

    def step(self):
        super().step()
        p = self.pars
        ti = self.ti

        uids = ss.uids(ti >= self.ti_next)
        if len(uids) == 0:
            return # Nothing to do

        # Reporting of new_active
        new_asymp_uids = uids[self.state_next[uids] == TBSL.ASYMPTOMATIC]
        self.results['new_active'][ti] = len(new_asymp_uids)
        self.results['new_active_15+'][ti] = np.count_nonzero(self.sim.people.age[new_asymp_uids] >= 15)

        # Update infected flag
        new_inf_uids = uids[self.state_next[uids] == TBSL.INFECTION]
        self.infected[new_inf_uids] = True
        new_clr_uids = uids[np.isin(self.state_next[uids], [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])]
        self.infected[new_clr_uids] = False

        # Update state
        self.state[uids] = self.state_next[uids]
        self.ti_next[uids] = np.inf # Reset to avoid accidental transitions
        self.on_treatment[uids] = self.state[uids] == TBSL.TREATMENT # Set treatment flag

        # Cleared, recovered, and treated are all susceptible to infection
        self.susceptible[uids] = np.isin(self.state[uids], [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])

        # Handle deaths
        new_death_uids = uids[self.state_next[uids] == TBSL.DEAD]
        self.sim.people.request_death(new_death_uids)
        self.results['new_deaths'][ti] = len(new_death_uids)
        self.results['new_deaths_15+'][ti] = np.count_nonzero(self.sim.people.age[new_death_uids] >= 15)

        # Set rel_sus
        self.rel_sus[uids] = 1 # Reset
        #self.rel_sus[self.state == TBSL.CLEARED] = 1.0 # Nothing to do for now
        self.rel_sus[uids[self.state[uids] == TBSL.RECOVERED]] = self.pars.pi
        self.rel_sus[uids[self.state[uids] == TBSL.TREATED]] = self.pars.rho

        # Set rel_trans
        self.rel_trans[uids] = 1 # Reset
        self.rel_trans[uids[self.state[uids] == TBSL.ASYMPTOMATIC]] = self.pars.kappa

        # INFECTION --> CLEARED, UNCONFIRMED, or ASYMPTOMATIC
        u = uids[self.state[uids] == TBSL.INFECTION]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.CLEARED: self.pars.infcle,
            TBSL.UNCONFIRMED: self.pars.infunc,
            TBSL.ASYMPTOMATIC: self.pars.infasy
        })

        # CLEARED --> INFECTION [Happens via transmission]

        # UNCONFIRMED to RECOVERED or ASYMPTOMATIC
        u = uids[self.state[uids] == TBSL.UNCONFIRMED]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.RECOVERED: self.pars.uncrec,
            TBSL.ASYMPTOMATIC: self.pars.uncasy
        })

        # RECOVERED to INFECTION [Happens via transmission, modified by pi]

        # ASYMPTOMATIC to UNCONFIRMED or SYMPTOMATIC
        u = uids[self.state[uids] == TBSL.ASYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.UNCONFIRMED: self.pars.asyunc,
            TBSL.SYMPTOMATIC: self.pars.asysym
        })

        # SYMPTOMATIC to ASYMPTOMATIC, TREATMENT, or TB DEATH
        u = uids[self.state[uids] == TBSL.SYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.ASYMPTOMATIC: self.pars.symasy,
            TBSL.TREATMENT: self.pars.theta,
            TBSL.DEAD: self.pars.mutb
        })

        # TREATMENT to SYMPTOMATIC or TREATED
        u = uids[self.state[uids] == TBSL.TREATMENT]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.SYMPTOMATIC: self.pars.phi,
            TBSL.TREATED: self.pars.delta
        })

        # TREATED to INFECTION [Happens via transmission, modified by rho]

        return

    def start_treatment(self, uids):
        """ Start treatment for active TB """
        if len(uids) == 0:
            return 0  # No one to treat

        # INFECTION --> CLEARED
        u = uids[self.state[uids] == TBSL.INFECTION]
        self.state_next[u] = TBSL.CLEARED
        self.ti_next[u] = self.ti

        # UNCONFIRMED, ASYMPTOMATIC, SYMPTOMATIC --> TREATMENT
        u = uids[np.isin(self.state[uids], [TBSL.UNCONFIRMED, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        self.state_next[u] = TBSL.TREATMENT # Schwalb paper shows TREATED here
        self.ti_next[u] = self.ti

        self.results['new_notifications_15+'][self.ti] = np.count_nonzero(self.sim.people.age[u] >= 15)

        return

    def step_die(self, uids):
        if len(uids) == 0:
            return # Nothing to do

        super().step_die(uids)
        # Make sure these agents do not transmit or get infected after death
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.state[uids] = TBSL.DEAD
        self.ti_next[uids] = np.inf # Ensure no more transitions
        self.rel_trans[uids] = 0 # Ensure no disease transmission
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()

        results = []
        for state in TBSL:
            results.append(ss.Result(f'n_{state.name}', dtype=int, label=state.name) )
            results.append(ss.Result(f'n_{state.name}_15+', dtype=int, label=f'{state.name} (15+)') )

        self.define_results(*results)
        self.define_results(
            ss.Result('n_infectious',      dtype=int, label='Number Infectious'),
            ss.Result('n_infectious_15+',  dtype=int, label='Number Infectious, 15+'),
            ss.Result('new_active',        dtype=int, label='New Active'),
            ss.Result('new_active_15+',    dtype=int, label='New Active, 15+'),
            ss.Result('cum_active',        dtype=int, label='Cumulative Active'),
            ss.Result('cum_active_15+',    dtype=int, label='Cumulative Active, 15+'),
            ss.Result('new_deaths',        dtype=int, label='New Deaths'),
            ss.Result('new_deaths_15+',    dtype=int, label='New Deaths, 15+'),
            ss.Result('cum_deaths',        dtype=int, label='Cumulative Deaths'),
            ss.Result('cum_deaths_15+',    dtype=int, label='Cumulative Deaths, 15+'),
            ss.Result('prevalence_active', dtype=float, scale=False, label='Prevalence (Active)'),
            ss.Result('incidence_kpy',     dtype=float, scale=False, label='Incidence per 1,000 person-years'),
            ss.Result('deaths_ppy',        dtype=float, label='Death per person-year'), 
            ss.Result('new_notifications_15+', dtype=int, label='New TB notifications, 15+'), 

            ss.Result('n_detectable_15+',  dtype=int, label='Symptomatic plus cxr_asymp_sens * Asymptomatic'),  # Move to analyzer?
        )
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.ti
        ti_infctd = self.ti_infected
        dty = self.sim.t.dt_year

        for state in TBSL:
            res[f'n_{state.name}'][ti] = np.count_nonzero(self.state == state)
            res[f'n_{state.name}_15+'][ti] = np.count_nonzero((self.sim.people.age>=15) & (self.state == state))

        #res.n_infectious[ti]       = np.count_nonzero(np.isin(self.state, [TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]))
        res.n_infectious[ti]        = np.count_nonzero(self.infectious)
        res['n_infectious_15+'][ti] = np.count_nonzero(self.infectious & (self.sim.people.age>=15))
        res.prevalence_active[ti]   = res.n_infectious[ti] / np.count_nonzero(self.sim.people.alive)
        res.incidence_kpy[ti]       = 1_000 * np.count_nonzero(ti_infctd == ti) / (np.count_nonzero(self.sim.people.alive) * dty)
        res.deaths_ppy[ti]          = res.new_deaths[ti] / (np.count_nonzero(self.sim.people.alive) * dty)

        res['n_detectable_15+'][ti] = np.dot( self.sim.people.age >= 15, (self.state == TBSL.SYMPTOMATIC) + self.pars.cxr_asymp_sens * (self.state == TBSL.ASYMPTOMATIC) )

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


# Make a version of the TB_LSHTM class that includes one additional state representing acute infection immediately after infection
class TB_LSHTM_Acute(TB_LSHTM):
    def __init__(self, pars=None, **kwargs):
        super().__init__()

        # Where the current INFECTION state is... split that into ACUTE and INFECTION
        # We had infunc and infasy, but now there's a brief ACUTE state prior to INFECTION
        # The infunc rate is not 

        self.define_pars(
            acuinf = ss.years(ss.expon(1/4.0)),   # Rate of transition from acute to infection per year
            alpha = 0.9,                   # Relative transmission from acute TB
        )
        self.update_pars(pars, **kwargs) 
        return

    @property
    def infectious(self):
        return (self.state==TBSL.ACUTE) | (self.state==TBSL.ASYMPTOMATIC) | (self.state==TBSL.SYMPTOMATIC)

    def set_prognoses(self, uids, from_uids=None):
        super().set_prognoses(uids, from_uids)
        if len(uids) == 0:
            return # Nothing to do

        p = self.pars

        # Carry out state changes upon new infection
        self.susceptible[uids] = False
        self.infected[uids] = True # Not needed, but useful for reporting
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti

        self.state[uids] = TBSL.ACUTE # Instead of INFECTION

        # ACUTE --> INFECTION
        self.state_next[uids], self.ti_next[uids] = self.transition(uids, to={
            TBSL.INFECTION: self.pars.acuinf,
        })

        return

    def step(self):
        super(TB_LSHTM, self).step() # Performs transmission
        p = self.pars
        ti = self.ti

        uids = ss.uids(ti >= self.ti_next)
        if len(uids) == 0:
            return # Nothing to do

        # Reporting of new_active
        new_asymp_uids = uids[self.state_next[uids] == TBSL.ASYMPTOMATIC]
        self.results['new_active'][ti] = len(new_asymp_uids)
        self.results['new_active_15+'][ti] = np.count_nonzero(self.sim.people.age[new_asymp_uids] >= 15)

        # Update infected flag
        new_inf_uids = uids[self.state_next[uids] == TBSL.ACUTE]
        self.infected[new_inf_uids] = True
        new_clr_uids = uids[np.isin(self.state_next[uids], [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])]
        self.infected[new_clr_uids] = False

        # Update state
        self.state[uids] = self.state_next[uids]
        self.ti_next[uids] = np.inf # Reset to avoid accidental transitions
        self.on_treatment[uids] = self.state[uids] == TBSL.TREATMENT # Set treatment flag

        # Cleared, recovered, and treated are all susceptible to infection
        self.susceptible[uids] = np.isin(self.state[uids], [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])

        # Handle deaths
        new_death_uids = uids[self.state_next[uids] == TBSL.DEAD]
        self.sim.people.request_death(new_death_uids)
        self.results['new_deaths'][ti] = len(new_death_uids)
        self.results['new_deaths_15+'][ti] = np.count_nonzero(self.sim.people.age[new_death_uids] >= 15)

        # Set rel_sus
        self.rel_sus[uids] = 1 # Reset
        #self.rel_sus[self.state == TBSL.CLEARED] = 1.0 # Nothing to do for now
        self.rel_sus[uids[self.state[uids] == TBSL.RECOVERED]] = self.pars.pi
        self.rel_sus[uids[self.state[uids] == TBSL.TREATED]] = self.pars.rho

        # Set rel_trans
        self.rel_trans[uids] = 1 # Reset
        self.rel_trans[uids[self.state[uids] == TBSL.ACUTE]] = self.pars.alpha
        self.rel_trans[uids[self.state[uids] == TBSL.ASYMPTOMATIC]] = self.pars.kappa

        # ACUTE --> INFECTION
        u = uids[self.state[uids] == TBSL.ACUTE]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.INFECTION: self.pars.acuinf,
        })


        # INFECTION --> CLEARED, UNCONFIRMED, or ASYMPTOMATIC
        u = uids[self.state[uids] == TBSL.INFECTION]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.CLEARED: self.pars.infcle,
            TBSL.UNCONFIRMED: self.pars.infunc,
            TBSL.ASYMPTOMATIC: self.pars.infasy
        })

        # CLEARED --> ACUTE [Happens via transmission]

        # UNCONFIRMED to RECOVERED or ASYMPTOMATIC
        u = uids[self.state[uids] == TBSL.UNCONFIRMED]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.RECOVERED: self.pars.uncrec,
            TBSL.ASYMPTOMATIC: self.pars.uncasy
        })

        # RECOVERED to ACUTE [Happens via transmission, modified by pi]

        # ASYMPTOMATIC to UNCONFIRMED or SYMPTOMATIC
        u = uids[self.state[uids] == TBSL.ASYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.UNCONFIRMED: self.pars.asyunc,
            TBSL.SYMPTOMATIC: self.pars.asysym
        })

        # SYMPTOMATIC to ASYMPTOMATIC, TREATMENT, or TB DEATH
        u = uids[self.state[uids] == TBSL.SYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.ASYMPTOMATIC: self.pars.symasy,
            TBSL.TREATMENT: self.pars.theta,
            TBSL.DEAD: self.pars.mutb
        })

        # TREATMENT to SYMPTOMATIC or TREATED
        u = uids[self.state[uids] == TBSL.TREATMENT]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.SYMPTOMATIC: self.pars.phi,
            TBSL.TREATED: self.pars.delta
        })

        # TREATED to ACUTE [Happens via transmission, modified by rho]

        return

    def start_treatment(self, uids):
        """ Start treatment for active TB """
        if len(uids) == 0:
            return 0  # No one to treat

        # ACUTE or INFECTION --> CLEARED
        u = uids[(self.state[uids] == TBSL.ACUTE) | (self.state[uids] == TBSL.INFECTION)]
        self.state_next[u] = TBSL.CLEARED
        self.ti_next[u] = self.ti

        # UNCONFIRMED, ASYMPTOMATIC, SYMPTOMATIC --> TREATMENT
        u = uids[np.isin(self.state[uids], [TBSL.UNCONFIRMED, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        self.state_next[u] = TBSL.TREATMENT # Schwalb paper shows TREATED here
        self.ti_next[u] = self.ti

        self.results['new_notifications_15+'][self.ti] = np.count_nonzero(self.sim.people.age[u] >= 15)

        return