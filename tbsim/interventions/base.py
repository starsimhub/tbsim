"""Base class for TB interventions.

Subclass ``TBIntervention`` instead of ``ss.Intervention`` when your
intervention needs the TB module and its disease states. It auto-detects
the TB variant and exposes ``self.tb`` and ``self.states``. Set
``_state_method`` (e.g. ``'care_seeking_eligible'``) to pick which states
your intervention uses.
"""

import starsim as ss
from tbsim.tb import TBS, TB
from tbsim.tb_lshtm import TBSL, TB_LSHTM, TB_LSHTM_Acute

__all__ = ['TBIntervention']

# Maps each TB disease class to the enum that defines its states.
# When a new TB variant is added to tbsim, add a corresponding entry here.
_DISEASE_ENUM = {
    TB_LSHTM_Acute: TBSL,
    TB_LSHTM:       TBSL,
    TB:             TBS,
}


class TBIntervention(ss.Intervention):
    """Base for interventions that need the TB module and its disease states.

    Provides ``self.tb`` (the TB disease module), ``self.state_enum`` (TBS or
    TBSL), and ``self.states`` (if ``_state_method`` is set).
    """

    _state_method = None

    def init_post(self):
        """Detect the TB variant in the simulation and resolve the state enum
        and (if ``_state_method`` is set) the relevant disease states."""
        super().init_post()
        self.tb = getattr(self.sim.diseases, 'tb', None) or self.sim.diseases[0]
        for cls in type(self.tb).__mro__:
            if cls in _DISEASE_ENUM:
                self.state_enum = _DISEASE_ENUM[cls]
                break
        if self._state_method is not None:
            self.states = getattr(self.state_enum, self._state_method)()
