"""ResistanceConnector: applies strain fitness to TB.rel_trans each timestep."""

import numpy as np
import starsim as ss
import tbsim

__all__ = ['ResistanceConnector']


class ResistanceConnector(ss.Connector):
    """
    Apply strain fitness to TB.rel_trans for infectious agents.

    Implements the "fittest strain" transmission model: an infectious agent's
    relative transmissibility is multiplied by the maximum fitness across
    strains the agent carries. Agents carrying no strain (e.g. legacy
    infections seeded before the resistance overlay was attached) get a
    multiplier of 1.0 (no change).

    This connector does *not* own strain state and does *not* sample which
    strain is transmitted; both responsibilities belong to
    :class:`StrainProfile` and :class:`tbsim.TB.set_prognoses`.

    Args:
        disease (str): Name of the TB disease module. Default ``'tb'``.

    Example:
        ::

            import starsim as ss
            import tbsim
            from tbsim.resistance import StrainSpec, ResistanceConnector

            strains = [
                StrainSpec('pan',   {'INH': 0, 'RIF': 0, 'BDQ': 0}, init_prev=0.04),
                StrainSpec('rif_r', {'INH': 0, 'RIF': 1, 'BDQ': 0}, fitness=0.9,
                           init_prev=0.005),
            ]
            tb = tbsim.TB(strains=strains)
            sim = ss.Sim(diseases=tb, connectors=ResistanceConnector(),
                         pars=dict(start='2000', stop='2010'))
            sim.run()
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(label=kwargs.pop('label', 'TB-Resistance'))
        self.define_pars(
            disease='tb',
        )
        self.update_pars(pars, **kwargs)
        return

    def _get_tb(self):
        return self.sim.diseases[self.pars.disease]

    def step(self):
        """Apply fittest-strain fitness multiplier to TB.rel_trans for infectious agents."""
        tb = self._get_tb()
        profile = getattr(tb, 'strain_profile', None)
        if profile is None:
            return  # no strain overlay configured; nothing to do

        infectious_uids = tb.infectious.uids
        if len(infectious_uids) == 0:
            return

        fitness = profile.effective_rel_trans(infectious_uids)
        # Agents with no carried strain get fitness 0 — keep them at the prior
        # rel_trans value so legacy infections still transmit normally.
        no_strain = fitness == 0
        if no_strain.any():
            fitness[no_strain] = 1.0

        tb.rel_trans[infectious_uids] *= fitness
        return
