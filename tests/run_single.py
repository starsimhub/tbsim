import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt



def make_tb_nut():
    # --------- People ----------
    n_agents = 1000
    extra_states = [
        ss.FloatArr('SES', default= ss.bernoulli(p=0.3)), # SES example: ~30% get 0, ~70% get 1 (TODO)
    ]
    pop = ss.People(n_agents=n_agents, extra_states=extra_states)

    # ------- TB disease --------
    tb_pars = dict(
        beta = 0.01, 
        init_prev = 0.25,
        )
    tb = mtb.TB(tb_pars)

    # ---------- Malnutrition --------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)

    # -------- Network ---------
    net_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0,  # End after one timestep
        )
    net = ss.RandomNet(net_pars)
    
    # ---------Demographics---------
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
    ]
    
    # --------- Connector ---------
    cn_pars = dict()
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # --------- Interventions ------
    
    # TODO: Add a set of common TB interventions
    # Create a TB vaccine product   
    tb_vaccine = None # ss.Product(name="TB Vaccine")
    # Create a routine TB vaccination intervention
    routine_vx = None # mtb.TBVaccinationCampaign(year=1997, target_state=1.0, product=tb_vaccine, prob=0.9)
    
    # TODO: Add a set of common nutrition interventions - we may need to remove dependencies from Harlem scenarios. 

    # --------- Analyzers ----------
    # TODO: Add a set of common post processing analyzers

    # --------- simulation ---------
    sim_pars = dict(
        dt = 7/365,
        start = 1980,
        stop = 1995
        )
    sim = ss.Sim(people=pop, networks=net, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn, interventions=routine_vx)
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps

    #

    return sim


if __name__ == '__main__':  
    sim_tbn = make_tb_nut()
    sim_tbn.run()
    sim_tbn.diseases['tb'].plot()
    plt.show()
    plt.close()
