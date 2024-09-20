import starsim as ss
import pandas as pd
from run_rations import build_RATIONS

import numpy as np

class RATIONS_Calibration(ss.Calibration):

    @staticmethod
    def translate_pars(sim=None, calib_pars=None):
        sim.pars['verbose'] = 0

        spec = calib_pars.pop('tb_hh_beta', None)
        if spec is not None:
            beta = spec['value']
            sim.diseases['tb'].pars['beta']['householdnet'] = [beta, beta]

        sim = ss.Calibration.translate_pars(sim, calib_pars)
        return sim

    def compute_fit(self, sim):
        fit = 0
        df_res = sim.export_df()
        df_res['year'] = np.floor(np.round(df_res.index, 1)).astype(int)
        for skey in self.sim_result_list:
            if skey in ['rationstrial.incident_cases_ctrl', 'rationstrial.incident_cases_intv']:
                py_key = 'rationstrial.person_years_intv' if 'intv' in skey else 'rationstrial.person_years_ctrl'
                model_output = df_res[skey].sum() / df_res[py_key].sum()

                if model_output == 0:
                    fit = np.nan # Nan indicates that a trial should be ignored, otherwise we get inf in the log2
                    return fit

                data = self.target_data[skey].values[0]
                mismatch = np.abs(np.log2(model_output / data)) # absolute log2 difference (number of 2-folds away from the data)
                print(skey, data, model_output, mismatch) # TEMP
                fit += mismatch
                continue

            if 'prevalence' in skey:
                model_output = df_res.groupby(by='year')[skey].mean()
            else:
                model_output = df_res.groupby(by='year')[skey].sum()

            py = None
            if skey in ['rationstrial.incident_cases_ctrl', 'rationstrial.incident_cases_intv']:
                # Normalize by person-years
                if 'ctrl' in skey:
                    py_key = 'rationstrial.person_years_ctrl'
                else:
                    py_key = 'rationstrial.person_years_intv'
                py = df_res.groupby(by='year')[py_key].sum()

            data = self.target_data[skey]

            if len(data) == 1 and (-1 in data.index):
                # -1 is a flag to sum across years
                model_output = pd.Series([df_res[skey].sum()], index=pd.Index([-1], name='year'), name=skey)
                if py is not None:
                    py = py.sum() # Sum over years

            if py is not None: # Normalize by person-years
                model_output /= py

            combined = pd.merge(data, model_output, how='left', on='year')
            combined['diffs'] = combined[skey+'_x'] - combined[skey+'_y']
            gofs = ss.calibration.compute_gof(combined.dropna()[skey+'_x'], combined.dropna()[skey+'_y'])

            losses = gofs  #* self.weights[skey] # TODO: Where are weights applied?
            mismatch = losses.sum()
            fit += mismatch

        return fit

def run_calib():

    # Define the calibration parameters
    calib_pars = dict(
        tb_hh_beta = dict(low=0.01, high=1.0, guess=0.5, log=True),
        x_community_ir = dict(low=0.8, high=1.2, guess=1.0, path=('interventions', 'rationstrial', 'x_community_incidence_rate')),
    )

    # Define weights for the data
    weights = {
        'rationstrial.incident_cases_ctrl':  1.0,
        #'rationstrial.incident_cases_intv':  1.0,
    }

    skey = 'Calib'
    scen = None

    sim = build_RATIONS(skey, scen, rand_seed=0)
    
    data = pd.DataFrame({
        'year': [-1], # Required by Starsim
        'rationstrial.incident_cases_ctrl': [90 / 9_557],
        #'rationstrial.incident_cases_intv': [62 / 12_208],
    })

    # Make the calibration
    calib = RATIONS_Calibration(
        calib_pars = calib_pars,
        sim = sim,
        data = data,
        weights = weights,
        total_trials = 250,
        n_workers = 9,
        die = True,
        #estimator='root_mean_squared_log_error',
    )

     # Perform the calibration
    print('\nPeforming calibration...')
    calib.calibrate(confirm_fit=True)

    return sim, calib


if __name__ == '__main__':
    sim, calib = run_calib()

    print(calib.df.head(25))