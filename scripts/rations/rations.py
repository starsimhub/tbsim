#### -------------- HELBER CLASS FOR CREATING RATIONS SIMULATIONS -------------------
import starsim as ss
import tbsim as mtb
import sciris as sc
import numpy as np
import pandas as pd
from enum import IntEnum, auto

__all__ = ['RATIONSTrial', 'RATIONS', 'Arm']

class Arm(IntEnum):
    CONTROL      = 0
    INTERVENTION = 1

class Cluster(IntEnum):
    DTC_SARAIKELA           = auto() #1
    RAJNAGAR                = auto() #2
    CHANDIL                 = auto() #3
    ADITYAPUR               = auto() #4
    GAMHARIA                = auto() #5
    KHARSWAN                = auto() #6
    ICHAGARH                = auto() #7
    ANGARHA_RATU            = auto() #8
    BUNDU                   = auto() #9
    DORANDA                 = auto() #10
    ITKI                    = auto() #11
    MANDAR_BURMU            = auto() #12
    ORMANJHI                = auto() #13
    SADAR                   = auto() #14
    JAGANNATHPUR            = auto() #15
    DTC_CHAIBASA_URBAN      = auto() #16
    DTC_CHAIBASA_RURAL      = auto() #17
    CHAKRADHARPUR           = auto() #18
    JHINKPANI               = auto() #19
    MANJHARI                = auto() #20
    TANTNAGAR               = auto() #21
    DHALBHUMGARH            = auto() #22
    SADAR2                  = auto() #23
    MUSABONI                = auto() #24
    MANGO                   = auto() #25
    BAHRAGORA               = auto() #26
    JUGSALAI                = auto() #27
    POTKA                   = auto() #28


cluster_to_arm = {k:Arm.CONTROL for k in Cluster.__members__.values()}
intv_clusters = [Cluster.RAJNAGAR, Cluster.CHANDIL, Cluster.ICHAGARH, Cluster.BUNDU, Cluster.DORANDA, Cluster.ITKI, Cluster.SADAR, Cluster.JAGANNATHPUR, Cluster.DTC_CHAIBASA_URBAN, Cluster.TANTNAGAR, Cluster.DHALBHUMGARH, Cluster.SADAR2, Cluster.MUSABONI, Cluster.POTKA]
for k in intv_clusters:
    assert k in Cluster.__members__.values()
    cluster_to_arm[k] = Arm.INTERVENTION

cdf = pd.DataFrame({ # Cluster data frame
    'Cluster': Cluster.__members__.keys(),
    'ACF': [539, 255, 209, 272, 335, 152, 241, 124, 273, 332, 800, 207, 124, 2843, 265, 775, 775, 564, 204, 193, 121, 268, 451, 289, 300, 300, 512, 251],
    'Population': [93759, 136600, 157949, 349065, 309072, 88642, 83099, 112759, 82975, 597044, 50058, 218474, 94137, 437178, 99169, 69565, 69565, 56531, 53792, 68450, 63910, 61932, 631364, 107084, 223805, 153051, 49660, 199612]
}, index=pd.Index([x.value for x in Cluster.__members__.values()], dtype=int, name='cid'))
cdf['Pulmonary Case Incidence Rate per 100,000 PY'] = 100_000 * cdf['ACF'] / cdf['Population']

class RATIONSTrial(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        self.define_states( # For individual people
            ss.FloatArr('cid'),  # Cluster ID
            ss.FloatArr('hhid'), # Household id
            ss.BoolArr('arm', default=Arm.CONTROL),
            ss.BoolArr('is_index'),
            ss.FloatArr('ti_dx'), # Only used for index cases, but easier here
            ss.FloatArr('ti_prev_visit'), # Used for tracking person-years of follow-up
            ss.FloatArr('ti_treatment'),
            ss.FloatArr('ti_enrolled'),
        )

        self.define_pars(
            n_clusters = 28,
            n_hhs = 2_800,

            # Multiplier on community incidence rate, set to 0 to disable community incidence
            x_community_incidence_rate = 1,

            p_sm_pos = ss.bernoulli(0.72), # SmPos vs SmNeg for active pulmonary TB of index patients

            dur_active_to_dx = ss.weibull(c=2, scale=3/12), # Sensitive parameter - determines how much HH transmission BEFORE the trial
            #dur_active_to_dx = ss.constant(v=0.01),

            dur_dx_to_first_visit = ss.uniform(low=0, high=1/12),

            dur_visit_to_tx = ss.weibull(c=2, scale=3 * 7/365), # for secondary cases, same as "dur_active_to_dx"?
            
            #hhsize = ss.histogram(
            #    values=[186832, 489076, 701325, 1145585, 1221054, 951857, 1_334_629, 160_132, 46657][1:],
            #    bins=np.array([1,2,3,4,5,6,7, 11, 15, 16][1:]),
            #    density=False),
        )
        self.update_pars(pars, **kwargs)

        # States for index cases / households
        self.index_uids = np.full(self.pars.n_hhs, fill_value=np.nan)
        self.ti_first_visit = np.full(self.pars.n_hhs, fill_value=np.nan)
        self.ti_visit = np.full(self.pars.n_hhs, fill_value=np.nan)
        self.uids_by_hhid = []
        self.hhs = None

        # Distributions
        self.community_acq = ss.bernoulli(p = self.p_community_acq)

        return

    @staticmethod
    def p_community_acq(self, sim, uids):
        '''
        Compute the bernoulli probability of each agent becoming infected from
        the community. Uses RATIONS cluster incidence data from the "cdf" dataframe.
        '''
        p = np.ones(len(uids))
        tb = sim.diseases['tb']
        dt = self.dt 
        frac_pulmonary = 0.65 + 0.25

        years = dt
        for cid, cdata in cdf.iterrows():
            cids = self.cid[uids] == cid
            person = np.count_nonzero(cids)
            expected_ptb_cases = cdata['Pulmonary Case Incidence Rate per 100,000 PY'] * person*years / 100_000
            # all infections eventually become active, but not all are pulmonary
            ptb_cases_to_seed = expected_ptb_cases / frac_pulmonary
            in_cluster_and_sus = cids & tb.susceptible[uids]
            p[in_cluster_and_sus] = self.pars.x_community_incidence_rate * ptb_cases_to_seed / len(in_cluster_and_sus)
        return p

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result(name='new_hhs_enrolled_ctrl', dtype=int, label='New households enrolled (control)'),
            ss.Result(name='incident_cases_ctrl', dtype=int, label='Incident cases (control)'),
            ss.Result(name='coprevalent_cases_ctrl', dtype=int, label='Coprevalent cases (control)'),
            ss.Result(name='person_years_ctrl', dtype=int, label='Person-years (control)'),
            ss.Result(name='new_hhs_enrolled_intv', dtype=int, label='New households enrolled (intervention)'),
            ss.Result(name='incident_cases_intv', dtype=int, label='Incident cases (intervention)'),
            ss.Result(name='coprevalent_cases_intv', dtype=int, label='Coprevalent cases (intervention)'),
            ss.Result(name='person_years_intv', dtype=int, label='Person-years (intervention)'),
        )
        return

    def init_post(self):
        super().init_post()

        ppl = self.sim.people
        tb = self.sim.diseases['tb']

        # Pick one adult to be the source. At time of diagnosis, must be 18+
        # with microbiologically confirmed pulminary TB
        over18 = ppl.age >= 18
        self.index_uids = ss.uids(np.random.choice(a=ppl.uid, p=over18/np.count_nonzero(over18), size=self.pars.n_hhs, replace=False))
        self.is_index[self.index_uids] = True
        non_seeds = ss.uids(np.setdiff1d(ppl.uid, self.index_uids))

        # Map index seeds to clusters
        self.cid[self.is_index] = np.repeat([x.value for x in Cluster.__members__.values()], int(self.pars.n_hhs / self.pars.n_clusters))

        # Map index seeds to arm (via cluster)
        self.arm[self.index_uids] = [cluster_to_arm[Cluster(int(k))] for k in self.cid[self.is_index]]

        # Map people to households
        self.hhid[ self.index_uids] = np.arange(self.pars.n_hhs)
        self.hhid[~self.is_index] = np.random.choice(np.arange(self.pars.n_hhs), len(non_seeds), replace=True)

        # Now that we know how agents map to hhs, we can update some things...
        hhn = self.sim.networks['householdnet']
        for hhid in np.arange(self.pars.n_hhs):
            uids_in_hh = (self.hhid == hhid).uids
            self.uids_by_hhid.append(uids_in_hh)

            # Set the cluster and arm for the other HH members
            self.cid[uids_in_hh] = self.cid[self.index_uids[hhid]]
            self.arm[uids_in_hh] = self.arm[self.index_uids[hhid]]

            # Add this household to the network
            hhn.add_hh(uids_in_hh)

        # Initialize the TB infection
        tb.set_prognoses(self.index_uids)

        # All RATIONS index cases are pulmonary, choose SmPos vs SmNeg
        smpos = self.pars.p_sm_pos(self.index_uids)
        tb.active_tb_state[self.index_uids[smpos]] = mtb.TBS.ACTIVE_SMPOS
        tb.active_tb_state[self.index_uids[~smpos]] = mtb.TBS.ACTIVE_SMNEG

        # The individual should be shedding active pulmonary TB for some period
        # of time before seeking care, distribution from input

        # On seeking care, the individual will be diagnosed and start treatment.
        # Only at this point does RATIONS learn of this individual/household.

        # At some additional delay, the household receives its first visit +
        # food basket for either just the index (control) or index + HH members
        # (intervention).
        return

    def step(self):
        super().step()

        tb = self.sim.diseases['tb']
        nut = self.sim.diseases['malnutrition']
        ti, dt = self.ti, self.dt
        
        # INCIDENCE FROM COMMUNITY
        if self.pars.x_community_incidence_rate > 0:
            uids = tb.susceptible.uids
            incident_uids = self.community_acq.filter(uids)
            if len(incident_uids): tb.set_prognoses(incident_uids)

        # After set_prognoses, index_uids will be in latent slow or fast.  Latent
        # phase doesn't matter, not transmissible during that period.  So fast
        # forward to end of latent, beginning of active PRE-SYMPTOMATIC stage.
        # Change to ACTIVE_PRESYMP and set time of activation to current time
        # step.
        tb.rr_activation[self.index_uids] = 100000 # Increase the rate to individuals activate on the next time step
        tb.rr_clearance[self.index_uids] = 0 # Consider resetting to 1 after diagnosis
        tb.rr_death[self.index_uids] = 0 # Consider resetting to 1 after diagnosis

        # INDEX CASES: Pre symp --> Active (state change has already happend in TB on this timestep)
        new_active_uids = self.index_uids[ np.isin(tb.state[self.index_uids], [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG]) & (tb.ti_active[self.index_uids] == ti-1) ] # Activated on previous step
        if len(new_active_uids):
            # Newly active, figure out time to care seeking
            dur_untreated = self.pars.dur_active_to_dx(new_active_uids)
            self.ti_dx[new_active_uids] = np.ceil(ti + dur_untreated / dt)

        # INDEX CASES: Active --> Diagnosed and beginning immediate treatment
        dx_uids = self.index_uids[self.ti_dx[self.index_uids] == ti]
        if len(dx_uids):
            # Index cases can be infected & diagnosed multiple times, only keep the first
            hhids = self.hhid[dx_uids].astype(int)
            first = np.isnan(self.ti_first_visit[hhids])

            # Filter to first
            dx_uids = dx_uids[first]
            hhids = hhids[first]

        if len(dx_uids): # (now filtered)
            # Newly diagnosed. Start treatment and determine when the first RATIONS visit will occur.
            tb.start_treatment(dx_uids)

            dur_dx_to_first_visit = self.pars.dur_dx_to_first_visit(dx_uids)
            self.ti_first_visit[hhids] = np.ceil(ti + dur_dx_to_first_visit / dt)
            self.ti_visit[hhids] = self.ti_first_visit[hhids]

        # Remember when the first visit occurs
        first_visit_hhids = np.where(self.ti_first_visit == ti)[0]
        if len(first_visit_hhids):
            first_visit_uids = ss.uids.cat([self.uids_by_hhid[h] for h in first_visit_hhids])
            ti_first_visit_byuid = np.concatenate([np.full(len(self.uids_by_hhid[h]), fill_value=self.ti_first_visit[h]) for h in first_visit_hhids])
            self.ti_enrolled[first_visit_uids] = ti_first_visit_byuid
            first_visit_index_uids = ss.uids(np.intersect1d(self.index_uids, first_visit_uids))
            self.results['new_hhs_enrolled_ctrl'][ti] += np.count_nonzero(self.arm[first_visit_index_uids] == Arm.CONTROL)
            self.results['new_hhs_enrolled_intv'][ti] += np.count_nonzero(self.arm[first_visit_index_uids] == Arm.INTERVENTION)

        # Visit households
        visit_hhids = np.where(self.ti_visit == ti)[0]
        if len(visit_hhids):
            # TODO: During RATIONS, were newborns added to the trial populations? If not, we could just set the birth rate to 0.

            visit_uids = ss.uids.cat([self.uids_by_hhid[h] for h in visit_hhids])

            # Have visited previously?
            have_visited_uids = visit_uids[~np.isnan(self.ti_prev_visit[visit_uids])]
            new_py = dt * (ti - self.ti_prev_visit[have_visited_uids])
            self.results['person_years_ctrl'][ti] = np.sum(new_py[self.arm[have_visited_uids] == Arm.CONTROL])
            self.results['person_years_intv'][ti] = np.sum(new_py[self.arm[have_visited_uids] == Arm.INTERVENTION])
            self.ti_prev_visit[visit_uids] = ti

            # Check for new active cases
            active = np.isin(tb.state[visit_uids], [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB]) # Include extra-pulmonary here?
            not_dx = np.isnan(self.ti_dx[visit_uids])
            new_active_uids = visit_uids[ active & not_dx]

            if len(new_active_uids):
                # Record cases
                within_2m = (ti - self.ti_enrolled[new_active_uids]) * dt < 2/12 # within first 2 months
                self.results['coprevalent_cases_ctrl'][ti] = np.count_nonzero(self.arm[new_active_uids[within_2m]] == Arm.CONTROL )
                self.results['coprevalent_cases_intv'][ti] = np.count_nonzero(self.arm[new_active_uids[within_2m]] == Arm.INTERVENTION)
                self.results['incident_cases_ctrl'][ti] = np.count_nonzero(self.arm[new_active_uids[~within_2m]] == Arm.CONTROL)
                self.results['incident_cases_intv'][ti] = np.count_nonzero(self.arm[new_active_uids[~within_2m]] == Arm.INTERVENTION)

                # SET TIME TO TREATMENT
                dur_visit_to_tx = self.pars.dur_visit_to_tx(new_active_uids)
                # Maybe could get treatment independently, outside the trial
                self.ti_treatment[new_active_uids] = np.ceil(ti + dur_visit_to_tx / dt)

            # PLAN NEXT VISIT
            # The basket was provided to the participants for the duration of
            # treatment—6 months for drug susceptible tuberculosis and 12 months
            # for multidrug-resistant tuberculosis. The intervention was extended
            # if the patient had a BMI of lower than 18·5 kg/m² or any household
            # contact in the intervention group fulfilled the following: an adult
            # household contact with a BMI of lower than 16 kg/m²; children (aged
            # <10 years) with a weight-for-age Z-score of lower than –2SD and
            # adolescents (aged 10–18 years) with BMI-for-age Z-scores of lower
            # than –2SD. This extension was for a period of 12 months or until
            # improvements above these cutoffs, whichever was shorter.
            uids = visit_uids[self.is_index[visit_uids] | (self.arm[visit_uids] == Arm.INTERVENTION)] # Index or intervention arm
            nut.receiving_macro[uids] = True
            nut.receiving_micro[uids] = True

            # Set timer to next visit, end visits at 6, 12, OR 6-12mo
            # "monthly for the first year and every 3 months thereafter"
            within_1y = (ti - self.ti_first_visit[visit_hhids]) * dt < 1 # year
            self.ti_visit[visit_hhids[within_1y]] = np.ceil(ti + 1/12 / dt)
            self.ti_visit[visit_hhids[~within_1y]] = np.ceil(ti + 3/12 / dt)

            over_2y = (ti - self.ti_first_visit[visit_hhids]) * dt >= 2 # years
            self.ti_visit[visit_hhids[over_2y]] = np.nan # Do not visit again

            # 6mo intervention for starters
            # TODO: Extend some households up to 12 months based on conditions noted above
            over_6m = (ti - self.ti_first_visit[visit_hhids]) * dt >= 6/12 # 6 months
            over6m_uids_byhh = [self.uids_by_hhid[h] for h in visit_hhids[over_6m]]
            if len(over6m_uids_byhh) > 0:
                uids = ss.uids.cat(over6m_uids_byhh)
                nut.receiving_macro[uids] = False
                nut.receiving_micro[uids] = False

        treatment_uids = (self.ti_treatment == ti).uids
        if len(treatment_uids):
            # Start treatment for those diagnosed during a household visit
            tb.start_treatment(treatment_uids)

        return
