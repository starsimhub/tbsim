"""Treatment products for TB."""

import starsim as ss
from .drug_types import drug_params

__all__ = ['Tx', 'DOTS', 'DOTSImproved', 'FirstLine', 'SecondLine']


class Tx(ss.Product):
    """
    TB treatment product that encapsulates drug efficacy.

    Args:
        efficacy: Probability of treatment success (0-1). Default 0.85.
        drug_type: If provided (e.g. 'dots', 'first_line_combo'), overrides efficacy with drug-specific cure probability.
    """

    def __init__(self, efficacy=0.85, drug_type=None, **kwargs):
        super().__init__()
        if drug_type is not None:
            self.efficacy = drug_params[drug_type]['cure_prob']
        else:
            self.efficacy = efficacy

        self.define_pars(
            p_success = ss.bernoulli(self.efficacy)
        )
        self.update_pars(**kwargs)
        return

    def administer(self, sim, uids):
        """
        Administer treatment to agents.

        Returns:
            dict with 'success' and 'failure' UIDs.
        """
        success_uids, failure_uids = self.pars.p_success.filter(uids, both=True)
        return {'success': success_uids, 'failure': failure_uids}


class DOTS(Tx):
    """Standard DOTS (85% cure)."""

    def __init__(self, **kwargs):
        super().__init__(drug_type='dots', **kwargs)


class DOTSImproved(Tx):
    """Enhanced DOTS (90% cure)."""

    def __init__(self, **kwargs):
        super().__init__(drug_type='dots_improved', **kwargs)


class FirstLine(Tx):
    """First-line combination therapy (95% cure)."""

    def __init__(self, **kwargs):
        super().__init__(drug_type='first_line_combo', **kwargs)


class SecondLine(Tx):
    """Second-line therapy for MDR-TB (75% cure)."""

    def __init__(self, **kwargs):
        super().__init__(drug_type='second_line_combo', **kwargs)
