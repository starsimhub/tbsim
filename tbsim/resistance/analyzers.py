"""Analyzers for multi-strain (drug-resistance) TB."""

import numpy as np
import sciris as sc
import starsim as ss

from ..tb import get_tb
from .tb_resistant import TBResistant
from .treatments import TxDeliveryR

__all__ = ['ResistanceStats']


class ResistanceStats(ss.Analyzer):
    """
    Records the resistance observables of the reference ODE (``model-tests.md`` §11):
    resistant and superinfected fractions of active TB, and the three-way
    **resistance-origin flux decomposition** — new resistant cases arising from
    (i) de-novo mutation, (ii) treatment-acquired resistance, and (iii) transmission.

    ``TBResistant`` already logs ``frac_resist``/``frac_super`` and the de-novo and
    transmitted fluxes in its own results; this analyzer collates them with the
    treatment-acquired flux (summed over any ``TxDeliveryR`` interventions) into one
    place and exposes ``to_df()`` for like-for-like comparison against the ODE.
    """

    def init_pre(self, sim):
        super().init_pre(sim)
        self.tb = get_tb(sim, which=TBResistant)
        self.tx = [iv for iv in sim.interventions.values() if isinstance(iv, TxDeliveryR)]
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('frac_resist', dtype=float, scale=False, label='Resistant fraction of active TB'),
            ss.Result('frac_super', dtype=float, scale=False, label='Superinfected fraction of active TB'),
            ss.Result('flux_denovo', dtype=int, label='New resistance: de-novo mutation'),
            ss.Result('flux_txacq', dtype=int, label='New resistance: treatment-acquired'),
            ss.Result('flux_transmitted', dtype=int, label='New resistance: transmitted'),
        )
        return

    def step(self):
        ti = self.ti
        tbr = self.tb.results
        res = self.results
        res.frac_resist[ti] = tbr['frac_resist'][ti]
        res.frac_super[ti] = tbr['frac_super'][ti]
        res.flux_denovo[ti] = tbr['new_denovo_resistance'][ti]
        res.flux_transmitted[ti] = tbr['new_transmitted_resistance'][ti]
        res.flux_txacq[ti] = sum(int(iv.results.n_acquired[ti]) for iv in self.tx)
        return

    def to_df(self, sim):
        """Return the recorded observables as a tidy DataFrame indexed by time (pass the finished sim)."""
        res = sim.results[self.name]  # results are collated here after the run
        return sc.dataframe(
            time=sim.results.timevec,
            frac_resist=res.frac_resist,
            frac_super=res.frac_super,
            flux_denovo=res.flux_denovo,
            flux_txacq=res.flux_txacq,
            flux_transmitted=res.flux_transmitted,
        )
