"""
Simplest TB example: run the default TB model and plot results.
"""

import tbsim
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sim = tbsim.Sim(start='1940', stop='2010')
    sim.run()
    print(sim.pars)
    results = {'TB DEFAULTS': sim.results.flatten()}
    tbsim.plot(results, title='TB MODEL WITH DEFAULT PARAMETERS', row_height=1.5)
    plt.show()
