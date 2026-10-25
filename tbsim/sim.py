"""
TB simulation class: `ss.Sim` with a TB model and TB defaults
"""
import sciris as sc
import starsim as ss
import tbsim

__all__ = ['Sim', 'demo']


class Sim(ss.Sim):
    """
    An `ss.Sim` with a TB model (`tbsim.TB`) and TB defaults

    Parameters of the TB model, and its other arguments (`init`, `drugs`, `max_strains`, `strains`,
    `immigrants`), can be given directly (`tbsim.Sim(beta=12, init=dict(infection=0.1))`), and go to it.
    With no network, people mix homogeneously, so the sim runs as an ODE unless asked otherwise
    (`solver='abm'`, or a network such as `ss.RandomNet()`, makes it agent-based).

    Args:
        tb (`ss.Disease`): the TB model (default: `tbsim.TB()` with the parameters given)
        pars (dict): sim parameters and TB parameters, by name
        kwargs (dict): the same, as keywords

    Examples::

        sim = tbsim.Sim(beta=12).run()                       # An ODE
        sim = tbsim.Sim(networks=ss.RandomNet(), beta=2)     # Agents, on a network (beta per contact)
        sim = tbsim.Sim(tb=tbsim.TB(**tbsim.regimen('dots')), interventions=...)
    """
    def __init__(self, tb=None, pars=None, **kwargs):
        pars = sc.mergedicts(pars, kwargs)
        tb_keys = [key for key in pars if key in tbsim.tb_pars() or key in ['init', 'drugs', 'max_strains', 'strains', 'immigrants']] # Parameters and arguments of tbsim.TB
        tb_kw = {key:pars.pop(key) for key in tb_keys}
        if tb is None:
            tb = tbsim.TB(**tb_kw)
        elif tb_kw:
            errormsg = f'TB parameters ({sc.strjoin(tb_kw.keys())}) can only be given here if the TB model is not: give them to it instead'
            raise ValueError(errormsg)
        diseases = [tb] + sc.tolist(pars.pop('diseases', None))
        defaults = dict(start=ss.date('2000-01-01'), stop=ss.date('2010-01-01'), dt=ss.weeks(1), total_pop=100_000)
        super().__init__(diseases=diseases, **sc.mergedicts(defaults, pars))
        return

    def get_tb(self):
        """ The TB module of this sim """
        return tbsim.get_tb(self)

    def plot(self, key=None, **kwargs):
        """ Plot the results: by default, a selection of the TB results """
        if key is None:
            name = self.get_tb().name
            keys = [f'{name}_{k}' for k in ['infectious', 'prevalence_active', 'transmission', 'tb_death', 'treatment']]
            key = [k for k in keys if k in self.results.flatten()] or None
        return super().plot(key=key, **kwargs)


def demo(run=True, plot=True, **kwargs):
    """
    A demo TB simulation

    Args:
        run (bool): whether to run the sim
        plot (bool): whether to plot results (if run)
        kwargs (dict): passed to `tbsim.Sim()`

    Examples::

        tbsim.demo()
        sim = tbsim.demo(run=False, beta=12)
    """
    sim = Sim(**kwargs)
    if run:
        sim.run()
        if plot:
            sim.plot()
    return sim
