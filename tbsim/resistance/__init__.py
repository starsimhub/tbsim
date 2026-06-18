"""
Drug resistance overlay for TB.

This subpackage adds a multi-strain layer on top of the agent-level TB
natural-history state. It is opt-in: TB defaults to a single-strain model
unless `strains=` is provided.

Public objects:
    StrainSpec        — declarative strain specification
    StrainRegistry    — catalog of strains with phenotype and fitness lookup
    StrainProfile     — per-agent strain presence state (owned by TB)
    ResistanceConnector — applies strain fitness to TB.rel_trans
"""

from .strains import StrainSpec, StrainRegistry
from .profile import StrainProfile
from .connector import ResistanceConnector
from .regimens import Regimen
from .resolvers import ProgressionResolver, AcquisitionResolver
from .tx import StrainAwareTx, StrainAwareTxDelivery
from .tpt import StrainAwareTPTTx
from .diagnostics import DSTDx, DSTDelivery
from .analyzers import StrainResults, DuplicateStrainAnalyzer

__all__ = [
    'StrainSpec', 'StrainRegistry', 'StrainProfile', 'ResistanceConnector',
    'Regimen', 'ProgressionResolver', 'AcquisitionResolver',
    'StrainAwareTx', 'StrainAwareTxDelivery', 'StrainAwareTPTTx',
    'DSTDx', 'DSTDelivery', 'StrainResults', 'DuplicateStrainAnalyzer',
]
