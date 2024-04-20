import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

def make_harlem():

    np.random.seed(0) # TODO

    # --------- Harlem ----------
    harlem = mtb.Harlem()

    # --------- People ----------
    pop = harlem.people()

    # -------- Network ---------
    harlemnet = harlem.net()

    # Network parameters
    randnet_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
    )
    # Initialize a random network
    randnet = ss.RandomNet(randnet_pars)

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = dict(harlem=0.1, random=0.001),
        init_prev = 0, # Infections seeded by Harlem class
        rel_trans_smpos     = 1.0,
        rel_trans_smneg     = 0.3,
        rel_trans_exptb     = 0.05,
        rel_trans_presymp   = 0.10,
    )
    # Initialize
    tb = mtb.TB(tb_pars)

    # ---------- Nutrition --------
    nut_pars = dict()
    nut = mtb.Nutrition(nut_pars)

    # Add demographics
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
    ]

    # -------- Connector -------
    cn_pars = dict()
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- Interventions -------
    vs = mtb.VitaminSupplementation(year=[1942, 1943], rate=[2.0, 0.25])
    m = mtb.MacroNutrients
    lsff0 = mtb.LargeScaleFoodFortification(year=[1942, 1944], rate=[1.25, 0], from_state=m.UNSATISFACTORY, to_state=m.MARGINAL)
    lsff1 = mtb.LargeScaleFoodFortification(year=[1942, 1944], rate=[1.75, 0], from_state=m.MARGINAL, to_state=m.SLIGHTLY_BELOW_STANDARD)
    lsff2 = mtb.LargeScaleFoodFortification(year=[1942, 1944], rate=[1.75, 0], from_state=m.SLIGHTLY_BELOW_STANDARD, to_state=m.STANDARD_OR_ABOVE)
    intvs = [vs, lsff0, lsff1, lsff2]

    # -------- Analyzer -------
    az = mtb.HarlemAnalyzer()

    # -------- Simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt = 7/365,
        start = 1940, # 2y burn-in
        #start = 1942,
        end = 1947,
        )
    # initialize the simulation
    sim = ss.Sim(people=pop, networks=[harlemnet, randnet], diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn, interventions=intvs, analyzers=az)
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps

    sim.initialize()

    harlem.set_states(sim)
    seed_uids = harlem.choose_seed_infections(sim, p_hh=0.835) #83% or 84%
    tb.set_prognoses(sim, seed_uids)

    return sim


if __name__ == '__main__':
    sim = make_harlem()
    sim.run()
    sim.diseases['tb'].log.line_list.to_csv('linelist.csv')
    #sim.diseases['tb'].plot()
    sim.plot()

    sim.analyzers['harlemanalyzer'].plot()

    plt.show()
