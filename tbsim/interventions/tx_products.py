"""Treatment products for TB."""

import starsim as ss
from .drug_types import TBDrugType, TBDrugTypeParameters

__all__ = ['Tx', 'dots', 'dots_improved', 'first_line', 'second_line']


class Tx(ss.Product):
    """
    TB treatment product that encapsulates drug efficacy.

    Args:
        efficacy: Probability of treatment success (0-1). Default 0.85.
        drug_type: If provided, overrides efficacy with drug-specific cure probability.
    """

    def __init__(self, efficacy=0.85, drug_type=None, **kwargs):
        super().__init__()
        if drug_type is not None:
            params = TBDrugTypeParameters.create_parameters_for_type(drug_type)
            self.efficacy = params.cure_prob
        else:
            self.efficacy = efficacy
            
        self.define_pars(
            p_success = ss.bernoulli(self.efficacy) # TODO: think about if there's a more efficient way
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


def dots():
    """Standard DOTS (85% cure)."""
    return Tx(drug_type=TBDrugType.DOTS)


def dots_improved():
    """Enhanced DOTS (90% cure)."""
    return Tx(drug_type=TBDrugType.DOTS_IMPROVED)


def first_line():
    """First-line combination therapy (95% cure)."""
    return Tx(drug_type=TBDrugType.FIRST_LINE_COMBO)


def second_line():
    """Second-line therapy for MDR-TB (75% cure)."""
    return Tx(drug_type=TBDrugType.SECOND_LINE_COMBO)
