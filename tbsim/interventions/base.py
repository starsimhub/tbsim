"""
Base class for TB interventions.

TB interventions need access to the TB disease module running in the
simulation and to the set of disease states relevant to the intervention
(e.g. which states count as "eligible for care-seeking").  Because tbsim
supports multiple TB model variants (``TB``, ``TB_LSHTM``, ``TB_LSHTM_Acute``),
each with its own state definitions (``TBS`` or ``TBSL``), every intervention
would otherwise have to figure out which variant is in use and look up the
right states.

``TBIntervention`` removes that burden.  It automatically detects the TB
variant present in the simulation and resolves the corresponding state
definitions, so subclasses can focus on their own logic.

Example — creating a new TB intervention::

    from tbsim.interventions.base import TBIntervention

    class MyIntervention(TBIntervention):

        # Name of a static method on the state enum (TBS or TBSL) that
        # returns the array of states this intervention cares about.
        _state_method = 'care_seeking_eligible'

        def step(self):
            # These are available after initialisation:
            #   self.tb     — the TB disease module (e.g. a TB_LSHTM instance)
            #   self.states — the relevant state array (e.g. [ACUTE, NON_INFECTIOUS, ...])
            ...
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
    """
    Base class for interventions that operate on a TB disease module.

    Subclass this instead of ``ss.Intervention`` when your intervention needs
    to know which TB variant is running and which disease states are relevant.

    What the base class does for you (during simulation initialisation):

        self.tb
            Reference to the TB disease module in the simulation
            (e.g. an instance of ``TB``, ``TB_LSHTM``, or ``TB_LSHTM_Acute``).

        self.state_enum
            The state-definition enum that matches the TB variant
            (``TBS`` for the legacy model, ``TBSL`` for the LSHTM models).

        self.states
            An array of the specific states your intervention cares about,
            obtained by calling ``self.state_enum.<_state_method>()``.
            Only populated if the subclass sets ``_state_method``.

    To use: set ``_state_method`` to the name of a static method defined on
    both ``TBS`` and ``TBSL`` that returns the relevant states for your
    intervention (e.g. ``'care_seeking_eligible'``).
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
