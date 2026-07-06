"""
Drug resistance overlay for TB.

This subpackage supports :class:`tbsim.MultiStrainTB`, which adds a
multi-strain layer on top of the agent-level TB natural-history state.

Public objects:
    StrainSpec        — declarative strain specification
    StrainCatalog    — catalog of strains with phenotype and fitness lookup
    AgentStrains     — per-agent strain presence state (owned by MultiStrainTB)
    ResistanceConnector — applies strain fitness to TB.rel_trans
"""

from importlib import import_module

__all__ = [
    'StrainSpec', 'StrainCatalog', 'AgentStrains', 'ResistanceConnector',
    'Regimen', 'ProgressionResolver', 'AcquisitionResolver',
    'StrainAwareTx', 'StrainAwareTxDelivery', 'StrainAwareTPTTx',
    'DSTDx', 'DSTDelivery', 'RegimenRouter', 'treatment_monitoring_eligibility',
    'StrainResults', 'DuplicateStrainAnalyzer',
]

_exports = {
    'StrainSpec':                         'strains',
    'StrainCatalog':                     'strains',
    'AgentStrains':                      'strains',
    'ResistanceConnector':                'connector',
    'Regimen':                            'regimens',
    'ProgressionResolver':                'resolvers',
    'AcquisitionResolver':                'resolvers',
    'StrainAwareTx':                      'tx',
    'StrainAwareTxDelivery':              'tx',
    'StrainAwareTPTTx':                   'tpt',
    'DSTDx':                              'diagnostics',
    'DSTDelivery':                        'diagnostics',
    'RegimenRouter':                      'diagnostics',
    'treatment_monitoring_eligibility':   'diagnostics',
    'StrainResults':                      'analyzers',
    'DuplicateStrainAnalyzer':            'analyzers',
}


def __getattr__(name):
    """Lazily load public resistance objects to keep package imports acyclic."""
    if name not in _exports:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f'.{_exports[name]}', __name__)
    obj = getattr(module, name)
    globals()[name] = obj
    return obj
