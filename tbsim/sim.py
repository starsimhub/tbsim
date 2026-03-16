"""
TB simulation class -- a convenience wrapper around ss.Sim with TB-specific defaults.
"""

import sciris as sc
import starsim as ss
import tbsim

__all__ = ['Sim', 'demo']


class Sim(ss.Sim):
    """
    A subclass of ss.Sim specifically designed for TB simulations.

    Parses input parameters among the sim and the TB module, and provides
    default demographics (births and deaths), networks (random), and disease
    (TB_LSHTM).

    Args:
        pars (dict): Flat parameter dict; keys are auto-routed to sim vs TB pars.
        sim_pars (dict): Explicit sim-level parameter overrides.
        tb_pars (dict): Explicit TB parameter overrides.
        tb_model (str/Disease): Which TB model to use. Options: 'lshtm' (default),
            'lshtm_acute', 'emod', or a pre-built Disease instance.
        location (str): Placeholder for future location-based data.
        **kwargs: Additional parameters (auto-routed like ``pars``).

    Examples::

        import tbsim

        # Simplest usage -- all defaults
        sim = tbsim.Sim()
        sim.run()
        sim.plot()

        # Override TB and sim parameters via flat dict
        sim = tbsim.Sim(n_agents=2000, beta=ss.peryear(0.3), start=1990, stop=2020)

        # Use a different TB model
        sim = tbsim.Sim(tb_model='lshtm_acute')

        # Pass a pre-built TB instance
        tb = tbsim.TB_LSHTM(pars=dict(beta=ss.peryear(0.5)))
        sim = tbsim.Sim(tb_model=tb)
    """

    # Map of string names to TB model classes
    _tb_models = {
        'lshtm':       'TB_LSHTM',
        'lshtm_acute': 'TB_LSHTM_Acute',
        'emod':        'TB_EMOD',
    }

    def __init__(self, pars=None, sim_pars=None, tb_pars=None, tb_model=None,
                 location=None, **kwargs):

        if location is not None:
            raise NotImplementedError('Location-based sim creation is not implemented yet')

        # Merge all user inputs into a single dict
        pars = sc.mergedicts(pars, kwargs)
        sim_pars = sc.mergedicts(sim_pars)
        tb_pars = sc.mergedicts(tb_pars)

        # Pull modules out for special processing
        modules = sc.objdict()
        for mod_type in ['diseases', 'networks', 'demographics']:
            modules[mod_type] = sc.mergelists(pars.pop(mod_type, None))

        # Determine the TB model class and its default parameter keys
        if isinstance(tb_model, ss.Disease):
            tb = tb_model
            default_tb_keys = set()  # Don't route pars -- user gave a pre-built instance
        else:
            tb_cls = self._resolve_tb_class(tb_model)
            default_tb_keys = set(tb_cls().pars.keys())

            # Route flat pars: keys matching TB parameter names go to tb_pars
            default_sim_keys = set(ss.SimPars().keys())
            for key in list(pars.keys()):
                if key in default_tb_keys:
                    if key in default_sim_keys:
                        val = pars[key]  # In both: copy (don't pop)
                    else:
                        val = pars.pop(key)  # TB-only: pop
                    tb_pars[key] = val

            tb = tb_cls(pars=tb_pars)

        modules.diseases.insert(0, tb)

        # Handle demographics
        if not modules.demographics:
            modules.demographics = [ss.Births(), ss.Deaths()]

        # Handle networks
        if not modules.networks:
            modules.networks = [
                ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0)),
            ]

        # Apply default sim parameters, letting user values override
        default_sim_pars = dict(
            start    = ss.date('2000-01-01'),
            stop     = ss.date('2010-01-01'),
            dt       = ss.days(7),
            n_agents = 5000,
        )
        sim_pars = sc.mergedicts(default_sim_pars, sim_pars)

        super().__init__(
            pars          = sim_pars,
            demographics  = modules.demographics,
            networks      = modules.networks,
            diseases      = modules.diseases,
            **pars,
        )
        return

    @classmethod
    def _resolve_tb_class(cls, tb_model):
        """Resolve a string name or None to a TB disease class."""
        if tb_model is None:
            tb_model = 'lshtm'
        if isinstance(tb_model, str):
            key = tb_model.lower().replace('-', '_').replace(' ', '_')
            if key not in cls._tb_models:
                available = ', '.join(cls._tb_models.keys())
                raise ValueError(f"Unknown TB model '{tb_model}'. Available: {available}")
            return getattr(tbsim, cls._tb_models[key])
        raise TypeError(f"tb_model must be a string, Disease instance, or None; got {type(tb_model)}")

    def get_tb(self, which=None):
        """
        Get the TB disease module from this sim.

        Args:
            which (type, optional): Class of TB module to find (e.g. TB_LSHTM).
                If None, returns the first BaseTB subclass found.

        Returns:
            The TB disease module instance.
        """
        return tbsim.get_tb(self, which=which)

    def get_dx(self, name=None, result_state=None):
        """Get a DxDelivery instance. Returns None if not found."""
        for intv in self.interventions.values():
            if isinstance(intv, tbsim.DxDelivery):
                if name is not None and intv.name != name:
                    continue
                if result_state is not None and intv.result_state != result_state:
                    continue
                return intv
        return None

    def get_hsb(self):
        """Get the HealthSeekingBehavior instance. Returns None if not found."""
        for intv in self.interventions.values():
            if isinstance(intv, tbsim.HealthSeekingBehavior):
                return intv
        return None

    def plot(self, key=None, **kwargs):
        """
        Plot sim results. If key is 'tb', shows a curated panel of TB results.
        Otherwise falls back to the standard starsim plot.

        Args:
            key (str/list): Result key(s) to plot, or 'tb' for curated TB panel.
            **kwargs: Passed to ss.Sim.plot().

        Returns:
            matplotlib.figure.Figure
        """
        if isinstance(key, str) and key == 'tb':
            try:
                tb = self.get_tb()
                return tb.plot(**kwargs)
            except Exception:
                pass  # Fall through to default plot

        if key is None:
            # Default: plot curated TB keys if available
            try:
                tb = self.get_tb()
                tb_name = tb.name
                key = [
                    f'{tb_name}_n_infectious',
                    f'{tb_name}_prevalence_active',
                    f'{tb_name}_incidence_kpy',
                    f'{tb_name}_new_deaths',
                    'n_alive',
                ]
                # Filter to keys that actually exist
                available = set(self.results.keys())
                key = [k for k in key if k in available]
                if not key:
                    key = None  # Fall back to default
            except Exception:
                pass

        return super().plot(key=key, **kwargs)


def demo(run=True, plot=True, **kwargs):
    """
    Create a demo TB simulation.

    Args:
        run (bool): Whether to run the sim.
        plot (bool): Whether to plot results (only if run=True).
        **kwargs: Passed to Sim().

    Returns:
        Sim: Configured (and optionally run) simulation.

    Examples::

        import tbsim
        tbsim.demo()                                  # Run simple default demo
        sim = tbsim.demo(run=False, n_agents=500)     # Just create it
    """
    sim = Sim(**kwargs)

    if run:
        sim.run()
        if plot:
            sim.plot()

    return sim
