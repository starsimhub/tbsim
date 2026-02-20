import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt


def build_tbsim(sim_pars=None):

    sim = ss.Sim(diseases=mtb.TB(),pars=dict(dt = ss.days(7), start = ss.date('1940-01-01'), stop = ss.date('2010')))
    return sim

if __name__ == '__main__':
    sim = ss.Sim()
    
    sim = build_tbsim()
    sim.run()
    print(sim.pars)
    results = {'TB DEFAULTS  ': sim.results.flatten()}
    mtb.plot_combined(results, title='TB MODEL WITH DEFAULT PARAMETERS', dark=False, heightfold=1.5)
    
    plt.show()