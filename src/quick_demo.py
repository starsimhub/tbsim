"""
This is an example of the use of STARSim to simulate an SIR model. 
"""
import numpy as np
import sciris as sc
import scipy.stats as sps
from matplotlib import pyplot as plt

import starsim as ss

class SIR(ss.diseases.Disease):
    """
    Example SIR model: 
    This class implements a basic SIR model with states for susceptible,
    infected/infectious, and recovered. It also includes deaths, and basic
    results.

    """

    def __init__(self, pars=None, *args, **kwargs):
        """
        Initializes an instance of the Environment_Setup_Test class.

        Parameters:
        - pars (dict): Optional dictionary of parameters to override the default parameters.
        - args: Additional positional arguments.
        - kwargs: Additional keyword arguments.

        Returns:
        None
        """
        default_pars = {
            #'dur_inf': sps.weibull_min(c=lambda self, sim, uids: sim.people.age[uids], scale=10),#, seed='Duration of SIR Infection'),
            #'dur_inf': sps.norm(loc=lambda self, sim, uids: sim.people.age[uids], scale=2),
            'dur_inf': sps.lognorm(s=1, loc=10),
            'seed_infections': sps.bernoulli(p=0.1),
            'death_given_infection': sps.bernoulli(p=0.2),
            'beta': None,
        }

        super().__init__(pars=ss.omerge(default_pars, pars), *args, **kwargs)

        self.susceptible = ss.State('susceptible', bool, True)
        self.infected = ss.State('infected', bool, False)
        self.recovered = ss.State('recovered', bool, False)
        self.t_infected = ss.State('t_infected', float, np.nan)
        self.t_recovered = ss.State('t_recovered', float, np.nan)
        self.t_dead = ss.State('t_dead', float, np.nan)

        return

    def init_results(self, sim):
        """
        Initialize results for the simulation.

        Args:
            sim (Simulation): The simulation object.

        Returns:
            None
        """
        super().init_results(sim)
        self.results += ss.Result(self.name, 'prevalence', sim.npts, dtype=float)
        self.results += ss.Result(self.name, 'new_infections', sim.npts, dtype=int)
        return

    def update_pre(self, sim):
        """
        Update the state of the environment before the simulation progresses.

        Args:
            sim (Simulation): The simulation object.

        Returns:
            int: The number of deaths triggered during the update.
        """
        # Progress infectious -> recovered
        recovered = ss.true(self.infected & (self.t_recovered <= sim.year))
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = ss.true(self.t_dead <= sim.year)
        if len(deaths):
            sim.people.request_death(deaths)
        return len(deaths)

    def update_death(self, sim, uids):
        """
        Updates the flags for infected and recovered agents who have died.
        (Reset infected/recovered flags for dead agents)

        Parameters:
        sim (Simulation): The simulation object.
        uids (list): The unique IDs of the dead agents.

        Returns:
        None
        
        PLEASE NOTE:
        This is an optional step. Implementing this function means that in `SIR.update_results()` the prevalence
        calculation does not need to filter the infected agents by the alive agents. An alternative would be
        to omit implementing this function, and instead filter by the alive agents when calculating prevalence
        """
        super().update_death(sim, uids)
        self.infected[uids] = False
        self.recovered[uids] = False
        return

    def validate_pars(self, sim):
        """
        Validates the parameters for the simulation.

        Args:
            sim: The simulation object.

        Returns:
            None
        """
        if self.pars.beta is None:
            self.pars.beta = sc.objdict({k: 1 for k in sim.people.networks})
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.

        Parameters:
        - sim: The simulation object.

        Returns:
        None
        """
        alive_uids = ss.true(sim.people.alive)
        initial_cases = self.pars['seed_infections'].filter(alive_uids)
        self.infect(sim, initial_cases, None)
        return

    def infect(self, sim, uids, from_uids):
        """
        Infects individuals with a disease.

        Args:
            sim (Simulation): The simulation object.
            uids (array-like): The unique identifiers of the individuals to infect.
            from_uids (array-like): The unique identifiers of the individuals from whom the infection originated.

        Returns:
            None
        """
        super().set_prognoses(sim, uids, from_uids)

        # Carry out state changes associated with infection
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.t_infected[uids] = sim.year

        # Calculate and schedule future outcomes for recovery/death
        dur_inf = self.pars['dur_inf'].rvs(uids)
        will_die = self.pars['death_given_infection'].rvs(uids)
        self.t_recovered[uids[~will_die]] = sim.year + dur_inf[~will_die]
        self.t_dead[uids[will_die]] = sim.year + dur_inf[will_die]

        # Update result count of new infections - important to use += because
        # infect() may be called multiple times per timestep
        self.results['new_infections'][sim.ti] += len(uids)
        return

    def make_new_cases(self, sim):
        """
        Generates new cases of infection based on the given simulation object.

        Args:
            sim (Simulation): The simulation object containing the network and population information.

        Returns:
            None
        """
        for k, layer in sim.people.networks.items():
            if k in self.pars['beta']:
                rel_trans = (self.infected & sim.people.alive).astype(float)
                rel_sus = (self.susceptible & sim.people.alive).astype(float)
                for a, b, beta in [[layer.contacts['p1'], layer.contacts['p2'], self.pars['beta'][k]],
                                   [layer.contacts['p2'], layer.contacts['p1'], self.pars['beta'][k]]]:
                    # probability of a->b transmission
                    p_transmit = rel_trans[a] * rel_sus[b] * layer.contacts['beta'] * beta * sim.dt
                    new_cases = np.random.random(len(a)) < p_transmit # As this class is not common-random-number safe anyway, calling np.random is perfectly fine!
                    if new_cases.any():
                        self.infect(sim, b[new_cases], a[new_cases])
        return

    def update_results(self, sim):
        super().update_results(sim)
        self.results['prevalence'][sim.ti] = self.results.n_infected[sim.ti] / np.count_nonzero(sim.people.alive)
        return


def New_SIR_Model():
    """
    Function to create and run a new SIR model.

    Returns:
        None
    """
    ppl = ss.People(10000)
    ppl.networks = ss.ndict(ss.RandomNetwork(n_contacts=sps.poisson(mu=4)))

    sir_pars = {
        'dur_inf': sps.norm(loc=10), # Override the default distribution
    }
    sir = SIR(sir_pars)

    # You can also change the parameters of the default lognormal distribution directly!
    sir.pars['dur_inf'].kwds['loc'] = 5 

    # Or why not put a lambda here for fun!
    sir.pars['dur_inf'].kwds['loc'] = lambda self, sim, uids: sim.people.age[uids]/10

    sir.pars['beta'] = {'randomnetwork': 0.1}
    sim = ss.Sim(people=ppl, diseases=sir)
    sim.run()

    # CK: parameters changed
    # assert len(sir.log.out_edges(np.nan)) == sir.pars.initial # Log should match initial infections
    df = sir.log.line_list # Check generation of line-list
    # assert df.source.isna().sum() == sir.pars.initial # Check seed infections in line list

    plt.figure()
    plt.stackplot(
        sim.yearvec,
        sir.results.n_susceptible,
        sir.results.n_infected,
        sir.results.n_recovered,
        sim.results.new_deaths.cumsum(),
        colors=plt.cm.viridis(np.linspace(0, 1, 4))
    )
    
    plt.legend(['Susceptible', 'Infected', 'Recovered', 'Dead'])
    plt.ylabel('Number of People')
    plt.xlabel('Year')
    plt.title('SIR sample model')
    return


if __name__ == '__main__':
    ss.options(multirng=False)
    sim1 = New_SIR_Model()
    plt.show()