#### -------------- HELBER CLASS FOR CREATING RATIONS SIMULATIONS -------------------
import starsim as ss
import tbsim as mtb
import sciris as sc
import numpy as np
import pandas as pd
from enum import IntEnum, auto

__all__ = ['RATIONSTrial', 'RATIONS']

# mtb.StudyArm?
class Arm(IntEnum):
    CONTROL = 0
    INTERVENTION = 1


class RATIONSTrial(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        self.add_states( # For individual people
            ss.FloatArr('hhid'),
            ss.BoolArr('intv_arm'),
            ss.BoolArr('is_index'),
            ss.FloatArr('ti_dx'), # Only used for index cases, but easier here
            ss.FloatArr('ti_treatment'),
            ss.FloatArr('ti_enrolled'),
        )

        self.default_pars(
            n_hhs = 2_800,
            p_intv = ss.bernoulli(0.5), # 50% randomization

            p_sm_pos = ss.bernoulli(0.72), # SmPos vs SmNeg for active pulmonary TB of index patients

            dur_active_to_dx = ss.weibull(c=2, scale=3/12), # Sensitive parameter - determines how much HH transmission BEFORE the trial
            #dur_active_to_dx = ss.constant(v=0.01),

            dur_dx_to_first_visit = ss.uniform(low=0, high=1/12),

            dur_visit_to_tx = ss.weibull(c=2, scale=3 * 7/365), # for secondary cases, same as "dur_active_to_dx"?
            
            #hhsize_ctrl = ss.histogram(
            #    values=[186832, 489076, 701325, 1145585, 1221054, 951857, 1_334_629, 160_132, 46657][1:],
            #    bins=np.array([1,2,3,4,5,6,7, 11, 15, 16][1:]),
            #    density=False),
            #hhsize_intv = ss.histogram(
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

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        for arm in ['ctrl', 'intv']:
            self.results += ss.Result(self.name, f'new_hhs_enrolled_{arm}', self.sim.npts, dtype=int)
            self.results += ss.Result(self.name, f'incident_cases_{arm}', self.sim.npts, dtype=int)
            self.results += ss.Result(self.name, f'coprevalent_cases_{arm}', self.sim.npts, dtype=int)
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

        self.intv_arm[self.index_uids] = self.pars.p_intv.rvs(self.index_uids)

        # Map people to households
        self.hhid[ self.is_index] = np.arange(self.pars.n_hhs)
        self.hhid[~self.is_index] = np.random.choice(np.arange(self.pars.n_hhs), len(non_seeds), replace=True)

        # Now that we know how agents map to hhs, we can update some things...
        hhn = self.sim.networks['householdnet']
        for hhid in np.arange(self.pars.n_hhs):
            uids_in_hh = (self.hhid == hhid).uids
            self.uids_by_hhid.append(uids_in_hh)

            # Set the arm for the other HH members
            self.intv_arm[uids_in_hh] = self.intv_arm[self.index_uids[hhid]]
            
            # Add this household to the network
            hhn.add_hh(uids_in_hh)

        # Initialize the TB infection
        tb.set_prognoses(self.index_uids)

        # After set_prognoses, index_uids will be in latent slow or fast.  Latent
        # phase doesn't matter, not transmissible during that period.  So fast
        # forward to end of latent, beginning of active PRE-SYMPTOMATIC stage.
        # Change to ACTIVE_PRESYMP and set time of activation to current time
        # step.
        tb.rr_activation[self.index_uids] = 1000 # Increase the rate to individuals activate on the next time step

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


    def apply(self, sim):
        super().apply(sim)

        tb = self.sim.diseases['tb']
        nut = self.sim.diseases['malnutrition']
        ti, dt = self.sim.ti, self.sim.dt

        # INDEX CASES: Pre symp --> Active (state change has already happend in TB on this timestep)
        active_uids = self.index_uids[ np.isin(tb.state[self.index_uids], [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG]) & (tb.ti_active[self.index_uids] == ti) ]
        if len(active_uids):
            # Newly active, figure out time to care seeking
            dur_untreated = self.pars.dur_active_to_dx(active_uids)
            self.ti_dx[active_uids] = np.ceil(ti + dur_untreated / dt)

            # Individual could self cure (an exponential) prior to being diagnosed, hmm!
            # Let's say that the index cases don't cure, instead they'll go on treatment soon enough and clear that way.
            tb.rr_clearance[active_uids] = np.nan

            # And let's make it so index cases do not die from TB
            tb.rr_death[active_uids] = np.nan

        # INDEX CASES: Active --> Diagnosed and beginning immediate treatment
        dx_uids = self.index_uids[self.ti_dx[self.index_uids] == ti]
        if len(dx_uids):
            # Index cases can be infected & diagnosed multiple times, only keep the first
            hhids = self.hhid[dx_uids].astype(int)
            first = np.isnan(self.ti_first_visit[hhids])

            # Filter to first
            dx_uids = dx_uids[first]
            hhids = hhids[first]

            # Newly diagnosed. Start treatment and determine when the first RATIONS visit will occur.
            tb.start_treatment(dx_uids)

            dur_dx_to_first_visit = self.pars.dur_dx_to_first_visit(dx_uids)
            self.ti_first_visit[hhids] = np.ceil(ti + dur_dx_to_first_visit / dt)
            self.ti_visit[hhids] = self.ti_first_visit[hhids]

        # Remember when the first visit occurs
        first_visit_hhids = np.where(self.ti_first_visit == ti)[0]
        if len(first_visit_hhids):
            first_visit_uids = ss.uids(np.concatenate([self.uids_by_hhid[h] for h in first_visit_hhids]))
            ti_first_visit_byuid = np.concatenate([np.full(len(self.uids_by_hhid[h]), fill_value=self.ti_first_visit[h]) for h in first_visit_hhids])
            self.ti_enrolled[first_visit_uids] = ti_first_visit_byuid
            first_visit_index_uids = ss.uids(np.intersect1d(self.index_uids, first_visit_uids))
            self.results['new_hhs_enrolled_ctrl'][ti] += np.count_nonzero(~self.intv_arm[first_visit_index_uids])
            self.results['new_hhs_enrolled_intv'][ti] += np.count_nonzero( self.intv_arm[first_visit_index_uids])

        # Visit households
        visit_hhids = np.where(self.ti_visit == ti)[0]
        if len(visit_hhids):
            visit_uids = ss.uids(np.concatenate([self.uids_by_hhid[h] for h in visit_hhids]))

            # Check for new active cases
            active = np.isin(tb.state[visit_uids], [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB]) # Include extra-pulmonary here?
            not_dx = np.isnan(self.ti_dx[visit_uids]) # TODO: Doesn't allow for reinfection?
            new_active_uids = visit_uids[ active & not_dx]

            if len(new_active_uids):
                # Record cases
                within_2m = (ti - self.ti_enrolled[new_active_uids]) * dt < 2/12 # within first 2 months
                self.results['coprevalent_cases_ctrl'][ti] = np.count_nonzero(~self.intv_arm[new_active_uids[within_2m]] )
                self.results['coprevalent_cases_intv'][ti] = np.count_nonzero( self.intv_arm[new_active_uids[within_2m]] )
                self.results['incident_cases_ctrl'][ti] = np.count_nonzero(~self.intv_arm[new_active_uids[~within_2m]] )
                self.results['incident_cases_intv'][ti] = np.count_nonzero( self.intv_arm[new_active_uids[~within_2m]] )

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
            uids = visit_uids[self.is_index[visit_uids] | self.intv_arm[visit_uids]] # Index or intervention arm
            nut.receiving_macro[uids] = True
            nut.receiving_micro[uids] = True

            # Set timer to next visit, end visits at 6, 12, OR 6-12mo
            # "monthly for the first year and every 3 months thereafter"
            within_1y = (ti - self.ti_first_visit[visit_hhids]) * dt < 1 # year
            self.ti_visit[visit_hhids[within_1y]] = np.ceil(ti + 1/12 / dt)
            self.ti_visit[visit_hhids[~within_1y]] = np.ceil(ti + 3/12 / dt)

            over_2y = (ti - self.ti_first_visit[visit_hhids]) * dt >= 2 # years
            self.ti_visit[visit_hhids[over_2y]] = np.nan # Do not visit again

            over2y_uids_byhh = [self.uids_by_hhid[h] for h in visit_hhids[over_2y]]
            if len(over2y_uids_byhh) > 0:
                uids = ss.uids(np.concatenate(over2y_uids_byhh))
                nut.receiving_macro[uids] = False
                nut.receiving_micro[uids] = False

        treatment_uids = (self.ti_treatment == ti).uids
        if len(treatment_uids):
            # Start treatment for those diagnosed during a household visit
            tb.start_treatment(treatment_uids)

        return

