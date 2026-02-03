import os
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

import tb_acf as acf


class expon_LTV(ss.Dist):
    """
    Exponential distribution with linear time varying rate parameter
    """
    def __init__(self, date0, rate0, date1, rate1, **kwargs):
        super().__init__(**kwargs)

        # Assuming date0 and date1 as ss.date(date str)
        self.rate0 = rate0
        self.date0 = date0.to_year()
        self.rate1 = rate1
        self.date1 = date1.to_year()
        
        self.m = (self.rate1 - self.rate0) / (self.date1 - self.date0)
        self.b = self.rate0 - self.m * self.date0

        self.pars = sc.objdict(
            m = self.m,
            b = self.b,
        )

        return

    def make_rvs(self):
        """ Specified here because uniform() doesn't take a dtype argument """
        p = self._pars

        # Hazard function:
        #   H(t) = int_{t0}^{t} h(u) du
        #   h(u) = b + m * u
        #   H(t) = b * t + 0.5 * m * t^2 - H0
        #   H0   = b * t0 + 0.5 * m * t0^2

        # Survival function:
        #   S(t) = exp(-H(t))

        # Inverse transform sampling:
        #   S(t) = u ~ U(0, 1)
        #   S(t) = exp(-H(t))
        #   H(t) = -ln(u)
        #   b * t + 0.5 * m * t^2 - H0 = -ln(u)
        #   (0.5 * m) * t^2 + (b) * t + (ln(u) - H0) = 0
        
        # Solution from quadaratic formula with a = 0.5 * m, b = b, c = ln(u) - H0
        #   Take the positive root:
        #   t = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a)
        #   t = (-b + sqrt(b^2 - 2 * m * (ln(u)-H0))) / m

        t0 = self.sim.t.now('year')
        H0 = p.b * t0 + 0.5 * p.m * t0**2
        u = self.rand(self._size) + np.finfo(float).eps # Avoid log(0) as rand is [0,1)
        ln_u = np.log(u)
        
        # Floating point error can cause negative square root
        disc = np.clip(p.b**2 - 2 * p.m * (ln_u - H0), a_min=0, a_max=None)
        t = (-p.b + np.sqrt(disc)) / p.m
        return t


def get_intv(sim, name):
    if sim.initialized:
        interventions = sim.interventions.values()
    else:
        interventions = sim.pars.interventions

    for intv in interventions:
        if intv.name == name:
            return intv

    raise ValueError(f'Intervention {name} not found')

def save_info(resdir):
    sc.metadata(outfile=os.path.join(resdir, 'metadata.json'))

    # Save GIT hash to file
    info = sc.gitinfo()
    sc.savejson(os.path.join(resdir, 'tb_acf.json'), info)

    # Save the GIT hash of Starsim to file
    info = sc.gitinfo('starsim')
    sc.savejson(os.path.join(resdir, 'starsim.json'), info)

    # Save the GIT hash of TBsim to file
    info = sc.gitinfo('tbsim')
    sc.savejson(os.path.join(resdir, 'tbsim.json'), info)

    # Save GIT diff to file
    import git

    # Open the repository
    git_dir = os.path.join(os.path.dirname(__file__), '..')
    repo = git.Repo(git_dir)
    git_diff = repo.git.diff(repo.head.commit.tree)
    with open(os.path.join(resdir, 'git_diff.txt'), 'w') as f:
        f.write(git_diff)

def produce_param_grid(n_sample=10, seed=616, model='tbsim'):
    """ Produce a grid of parameter values"""
    # set the seed
    np.random.seed(seed)

    if model == 'tbsim':
        # define the parameter values
        params_grid = pd.DataFrame({
            "params_beta" : np.random.uniform(0.0, 1, n_sample),
            "params_init_prev" : np.random.uniform(0.0, 1.0, n_sample),
            "params_rate_presym_to_active" : np.power(10, np.random.uniform(-4, -1, n_sample)),
            "params_rel_trans_presymp" : np.random.uniform(0.0, 1.0, n_sample),
            "params_preymp_test_sens": np.random.uniform(0.0, 1.0, n_sample),
            })
    elif model == 'lshtm':
        params_grid = pd.DataFrame({
            "params_beta" : np.random.uniform(0.0, 1, n_sample),
            "params_init_prev" : np.random.uniform(0.0, 1.0, n_sample),
            "params_asysym" : np.power(10, np.random.uniform(-4, -1, n_sample)),
            "params_kappa" : np.random.uniform(0.0, 1.0, n_sample),
            "params_preymp_test_sens": np.random.uniform(0.0, 1.0, n_sample),
            })
    
    return params_grid

def process_param_grid(param_grid, best_fit):
    """ Process the parameter grid to give to run ACF"""

    # create a copy of the best fit
    best_fit_c = best_fit.copy()

    # Columns that are unique to each DataFrame
    columns_param_grid_only = set(param_grid.columns) - set(best_fit.columns)
    columns_best_fit_only = set(best_fit.columns) - set(param_grid.columns)

    # Add missing columns to param_grid with values from the first row of param_grid
    for col in columns_best_fit_only:
        param_grid[col] = best_fit[col].iloc[0]

    # Add missing columns to best_fit with values from the first row of param_grid
    for col in columns_param_grid_only:
        best_fit_c[col] = param_grid.iloc[0][col]

    # Concatenate the two DataFrames
    final_param_grid = pd.concat([best_fit_c, param_grid], ignore_index=True)

    # arrange all the columns in the final_param_grid
    # select all the columns that start with "params_"
    params_cols = [col 
                   for col in final_param_grid.columns 
                   if col.startswith("params_")
                   ]
    # select the remainder of the columns
    non_params_cols = [col 
                   for col in final_param_grid.columns 
                   if not col.startswith("params_")
                   ]

    return final_param_grid[non_params_cols + params_cols]
