"""Drug resistance overlay for TB.

Import from this subpackage explicitly — resistance is not re-exported on
``tbsim``::

    from tbsim.resistance import MultiStrainTB, StrainSpec, ResistanceConnector

Package layout:

- :class:`MultiStrainTB` — strain-aware subclass of base :class:`~tbsim.tb.TB`
- :class:`StrainSpec` / :class:`StrainCatalog` / :class:`AgentStrains` — data model
- :class:`ResistanceConnector` — fitness-weighted transmission
- Resolvers, regimens, strain-aware Tx/TPT/DST, analyzers

Architecture and tests: ``tbsim/resistance/docs/``.
"""

from .analyzers import DuplicateStrainAnalyzer, ResistanceStats, StrainResults
from .connector import ResistanceConnector
from .diagnostics import (
    DSTDelivery,
    DSTDx,
    RegimenRouter,
    treatment_monitoring_eligibility,
)
from .multistrain_tb import MultiStrainTB
from .regimens import Regimen
from .resolvers import AcquisitionResolver, ProgressionResolver
from .strains import AgentStrains, StrainCatalog, StrainSpec
from .tpt import StrainAwareTPTTx
from .tx import StrainAwareTx, StrainAwareTxDelivery

__all__ = [
    'MultiStrainTB',
    'StrainSpec', 'StrainCatalog', 'AgentStrains', 'ResistanceConnector',
    'Regimen', 'ProgressionResolver', 'AcquisitionResolver',
    'StrainAwareTx', 'StrainAwareTxDelivery', 'StrainAwareTPTTx',
    'DSTDx', 'DSTDelivery', 'RegimenRouter', 'treatment_monitoring_eligibility',
    'StrainResults', 'DuplicateStrainAnalyzer', 'ResistanceStats',
]
