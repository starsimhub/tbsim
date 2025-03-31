import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
def build_hivsim():
    # -------- simulation -------
    sim_pars = dict(
        unit = 'day',
        dt=7,
        start=sc.date('2013-01-01'),
        stop=sc.date('2020-12-31'),
        )
    
    # --------- Disease ----------
    hiv_pars = dict(
        init_prev=0.30,  # Initial prevalence of HIV
        p_ATRISK_to_ACUTE=0.0001,  # Probability of transitioning from ATRISK to ACUTE
        p_ACUTE_to_LATENT=1-np.exp(-1/8),  # Probability of transitioning from HIV to LATENT
        p_LATENT_to_AIDS=1-np.exp(-1/416),  # Probability of transitioning from LATENT to AIDS
        p_AIDS_to_DEAD=1-np.exp(-1/104),  # Probability of transitioning from AIDS to DEAD
        art_progression_factor=0.5,  # Progression factor when on ART
        on_ART = 0.50) # Percxentage of people on ART (infected)
    # Create the HIV disease model with the specified parameters
    hiv = mtb.HIV(pars=hiv_pars)

    # --------- People ----------
    n_agents = 1000
    extra_states = [   # People additional attributes - Cross simulation and diseases
        ss.FloatArr('SES', default= ss.bernoulli(p=0.3)), # SES example: ~30% get 0, ~70% get 1 (TODO)
        ss.Arr(name="CustomField", dtype=str, default="Any Value"),  # Custom field for each agent
        ss.Arr(name="FavoriteColor", dtype=str, default="Blue"),
    ]
    pop = ss.People(n_agents=n_agents, extra_states=extra_states)
    
    # --------- Network ---------
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    
    # --------- Demographics -----
    births = ss.Births(pars=dict(birth_rate=5))
    deaths = ss.Deaths(pars=dict(death_rate=5))
    
    # --------- Simulation -------
    sim = ss.Sim(people=pop, 
                 diseases=[hiv], 
                 demographics=[deaths,births],
                 networks=net,
                 pars=sim_pars)
    
    sim.pars.verbose = 30/365
    return sim


if __name__ == '__main__':
    # Make Malnutrition simulation
    sim = build_hivsim()
    sim.run()
    sim.plot()
    plt.show()
