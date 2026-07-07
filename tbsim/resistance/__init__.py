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
from .sim import (
    ResistanceSim,
    STRAIN_PRESETS,
    build_care_cascade,
    strain_preset_standard,
    strain_preset_two,
)
from .spec import (
    SPEC_SCENARIO_LABELS,
    SPEC_SCENARIO_META,
    build_spec_sim,
    compute_spec_directional_checks,
    format_spec_report,
    get_spec_scenario_configs,
    save_spec_report,
    strain_preset_spec,
    summarize_spec_sim,
)
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
    'ResistanceSim', 'build_care_cascade',
    'build_spec_sim', 'get_spec_scenario_configs', 'summarize_spec_sim',
    'compute_spec_directional_checks', 'format_spec_report', 'save_spec_report',
    'SPEC_SCENARIO_LABELS', 'SPEC_SCENARIO_META',
    'strain_preset_two', 'strain_preset_standard', 'strain_preset_spec', 'STRAIN_PRESETS',
]
