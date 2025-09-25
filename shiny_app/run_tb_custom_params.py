import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt


def build_tbsim(sim_pars=None):
    optional_pars = dict(
        dt = ss.days(7), 
        start = ss.date('1940-01-01'), 
        stop = ss.date('2010'),
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
        active_state = ss.choice(a=mtb.TBS.all_active(), p=[0.1, 0.1, 0.60, 0.20]),

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
        reltrans_het = ss.constant(v=1.0),           
    )
    if sim_pars is not None:
        optional_pars.update(sim_pars)
    sim = ss.Sim(diseases=mtb.TB(pars=optional_pars),pars=dict(dt = ss.days(7), start = ss.date('1940-01-01'), stop = ss.date('2010')))
    return sim

if __name__ == '__main__':
    sim = ss.Sim()
    
    sim = build_tbsim()
    sim.run()
    print(sim.pars)
    results = {'TB DEFAULTS  ': sim.results.flatten()}
    mtb.plot_combined(results, title='TB MODEL WITH DEFAULT PARAMETERS', dark=False, heightfold=1.5)
    
    plt.show()