"""Convenience wrapper and presets for resistance-enabled TB simulations."""

import sciris as sc
import starsim as ss
import tbsim
from tbsim.sim import Sim

from ..tb import TBS, get_tb

from .analyzers import DuplicateStrainAnalyzer, StrainResults
from .connector import ResistanceConnector
from .diagnostics import DSTDx, DSTDelivery, RegimenRouter
from .multistrain_tb import MultiStrainTB
from .regimens import Regimen
from .strains import StrainSpec
from .tx import StrainAwareTx, StrainAwareTxDelivery

# Re-export spec symbols for backward compat (they now live in spec.py)
from .spec import (  # noqa: F401
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

__all__ = [
    'ResistanceSim',
    'build_care_cascade',
    'build_spec_sim',
    'compute_spec_directional_checks',
    'format_spec_report',
    'get_spec_scenario_configs',
    'save_spec_report',
    'summarize_spec_sim',
    'strain_preset_spec',
    'strain_preset_two',
    'strain_preset_standard',
    'STRAIN_PRESETS',
    'SPEC_SCENARIO_LABELS',
    'SPEC_SCENARIO_META',
]


def strain_preset_two(init_prev_pan=0.05):
    """Pan-susceptible + INH-resistant (two-strain ODE / tutorial preset)."""
    return [
        StrainSpec('pan', {'INH': 0, 'RIF': 0}, fitness=1.0, init_prev=init_prev_pan),
        StrainSpec('inh_r', {'INH': 1, 'RIF': 0}, fitness=0.95, init_prev=0.0),
    ]


def strain_preset_standard(init_prev_pan=0.04):
    """Five-strain INH / RIF / BDQ catalog (``run_resistance.py`` preset)."""
    return [
        StrainSpec('pan',   {'INH': 0, 'RIF': 0, 'BDQ': 0}, fitness=1.00, init_prev=init_prev_pan),
        StrainSpec('inh_r', {'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95, init_prev=0.010),
        StrainSpec('rif_r', {'INH': 0, 'RIF': 1, 'BDQ': 0}, fitness=0.90, init_prev=0.005),
        StrainSpec('mdr',   {'INH': 1, 'RIF': 1, 'BDQ': 0}, fitness=0.85, init_prev=0.003),
        StrainSpec('bdq_r', {'INH': 0, 'RIF': 0, 'BDQ': 1}, fitness=0.90, init_prev=0.001),
    ]


STRAIN_PRESETS = {
    'two_strain': strain_preset_two,
    'standard': strain_preset_standard,
}


def _resolve_strains(strains, strain_preset):
    if strains is not None:
        return list(strains)
    preset = STRAIN_PRESETS.get(strain_preset)
    if preset is None:
        available = ', '.join(sorted(STRAIN_PRESETS))
        raise ValueError(f"Unknown strain_preset {strain_preset!r}; available: {available}")
    return preset()


def _active_tb_eligibility():
    """Eligible: alive active TB not already on treatment."""

    def _elig(sim):
        tb_local = get_tb(sim)
        active = (
            (tb_local.state == TBS.NON_INFECTIOUS)
            | (tb_local.state == TBS.ASYMPTOMATIC)
            | (tb_local.state == TBS.SYMPTOMATIC)
        ).uids
        return active.intersect(sim.people.alive.uids).intersect(tb_local.on_treatment.false())

    return _elig


def _diagnosed_eligibility(result_state='diagnosed'):
    """Eligible: diagnosed alive agents not already on treatment."""

    def _elig(sim):
        tb_local = get_tb(sim)
        dx_local = sim.get_dx(result_state=result_state)
        if dx_local is None:
            return ss.uids()
        return dx_local.diagnosed.uids.intersect(sim.people.alive.uids).intersect(
            tb_local.on_treatment.false()
        )

    return _elig


def build_care_cascade(tb, mode='basic', **kwargs):
    """
    Build a strain-aware care cascade for *tb* (:class:`MultiStrainTB`).

    Modes:

    - ``'basic'``: HSB → confirm Dx → DST → single first-line
      :class:`StrainAwareTxDelivery` (diagnosed active TB).
    - ``'routed'``: same front end, then DST-routed first/second-line Tx tiers
      via :class:`RegimenRouter` on ``routing_drug`` (default ``INH``).
    - ``'uniform_no_dst'``: HSB → confirm Dx → uniform INH short-course Tx
      without DST (spec scenario F1).
    - ``'dst_routed_inh'``: HSB → confirm Dx → INH-only DST → INH first-line /
      RIF second-line routing (spec scenario F2, two-strain catalog).
    - ``'tx_pressure'``: single low-efficacy INH :class:`StrainAwareTxDelivery`
      on active TB without diagnostics (spec scenario E2).

    Args:
        tb (MultiStrainTB): Initialized disease module (needs ``_strain_catalog``).
        mode (str): Cascade mode (see above).
        **kwargs: See source for per-mode keyword arguments.

    Returns:
        list: Intervention modules ready for ``tbsim.Sim(interventions=...)``.
    """
    valid_modes = ('basic', 'routed', 'uniform_no_dst', 'dst_routed_inh', 'tx_pressure')
    if mode not in valid_modes:
        raise ValueError(f"build_care_cascade mode must be one of {valid_modes}; got {mode!r}")

    catalog = tb._strain_catalog
    is_inh = mode == 'dst_routed_inh'

    if mode == 'tx_pressure':
        tx_efficacy = kwargs.get('tx_efficacy', 0.05)
        regimen = Regimen('inh_first_line', drugs=['INH'], per_drug_efficacy={'INH': tx_efficacy})
        product = StrainAwareTx(
            regimen=regimen, catalog=catalog,
            p_selective_acquisition=kwargs.get('p_selective_acquisition') or {'INH': 0.0},
            acq_state_modifiers=kwargs.get('acq_state_modifiers') or {
                'infection': 1.0, 'non_infectious': 1.0, 'asymptomatic': 1.0,
                'symptomatic': 1.0, 'treatment': 0.0, 'cleared': 0.0,
            },
            adherence=kwargs.get('tx_adherence', 1.0),
        )
        return [StrainAwareTxDelivery(product=product, name='tx_first', eligibility=_active_tb_eligibility())]

    confirm = tbsim.DxDelivery(
        name='confirm', product=tbsim.Xpert(),
        coverage=kwargs.get('confirm_coverage', 0.95 if mode in ('uniform_no_dst', 'dst_routed_inh') else 0.80),
        result_state='diagnosed',
    )
    hsb = tbsim.HealthSeekingBehavior()

    if mode == 'uniform_no_dst':
        regimen = Regimen('uniform_short_pan_tb', drugs=['INH'],
                          per_drug_efficacy={'INH': kwargs.get('uniform_efficacy', 0.95)})
        tx = StrainAwareTx(regimen=regimen, catalog=catalog,
                           p_selective_acquisition=kwargs.get('first_line_acq') or {'INH': 0.0},
                           adherence=kwargs.get('tx_adherence', 1.0))
        return [hsb, confirm,
                StrainAwareTxDelivery(product=tx, name='uniform_tx', eligibility=_diagnosed_eligibility())]

    drugs = list(kwargs.get('drugs') or (['INH'] if is_inh else catalog.drugs))
    dst = DSTDelivery(
        name='dst',
        product=DSTDx(
            catalog, drugs=drugs,
            sensitivity=kwargs.get('dst_sensitivity', 0.98 if is_inh else 0.95),
            specificity=kwargs.get('dst_specificity', 0.99),
            p_strain_obs=kwargs.get('p_strain_obs', 1.0 if is_inh else None),
            p_sample=kwargs.get('p_sample', 1.0),
            p_culture=kwargs.get('p_culture', 1.0),
        ),
        coverage=kwargs.get('dst_coverage', 0.95 if is_inh else 0.85),
    )

    first_drugs = kwargs.get('first_line_drugs') or (['INH'] if is_inh else ['INH', 'RIF'])
    first_regimen = Regimen('first_line', drugs=first_drugs,
                            per_drug_efficacy=kwargs.get('first_line_efficacy') or {d: 0.95 for d in first_drugs})
    first_product = StrainAwareTx(
        regimen=first_regimen, catalog=catalog,
        p_selective_acquisition=kwargs.get('first_line_acq') or dict(INH=0.05, RIF=0.02),
        adherence=kwargs.get('tx_adherence', 0.85 if not is_inh else 1.0),
    )

    interventions = [hsb, confirm, dst]

    if mode == 'basic':
        interventions.append(StrainAwareTxDelivery(product=first_product, name='first_line_tx'))
        return interventions

    # routed / dst_routed_inh
    routing_drug = kwargs.get('routing_drug', 'INH')
    second_drugs = kwargs.get('second_line_drugs') or (['RIF'] if is_inh else ['RIF', 'BDQ'])
    second_regimen = Regimen('second_line', drugs=second_drugs,
                             per_drug_efficacy=kwargs.get('second_line_efficacy') or {d: 0.90 for d in second_drugs})
    second_product = StrainAwareTx(
        regimen=second_regimen, catalog=catalog,
        p_selective_acquisition=kwargs.get('second_line_acq') or dict(RIF=0.03, BDQ=0.02),
        adherence=kwargs.get('second_line_adherence', 0.85 if not is_inh else 1.0),
    )
    router = RegimenRouter(dst, diagnosed_state='diagnosed', require_dst_tested=True)
    interventions.extend([
        StrainAwareTxDelivery(product=second_product, name='second_line_tx',
                              eligibility=router.matches(**{routing_drug: True})),
        StrainAwareTxDelivery(product=first_product, name='first_line_tx',
                              eligibility=router.matches(**{routing_drug: False})),
    ])
    return interventions


class ResistanceSim(Sim):
    """
    TB simulation wrapper with the drug-resistance overlay pre-wired.

    Extends :class:`tbsim.Sim` to build :class:`MultiStrainTB`,
    :class:`ResistanceConnector`, and optional strain analyzers by default.
    Optionally attaches a full strain-aware care cascade via
    :func:`build_care_cascade`.

    Example::

        from tbsim.resistance import ResistanceSim

        sim = ResistanceSim(n_agents=2000, strain_preset='two_strain')
        sim.run()

        sim = ResistanceSim(
            n_agents=5000,
            cascade='routed',
            resistance_pars=dict(p_multi=0.8),
        )
        sim.run()
        sim.get_dst()
    """

    _RESISTANCE_KEYS = frozenset({
        'strains', 'progression_mode', 'p_multi', 'p_random_acquisition',
        'alpha_super', 'alpha_act', 'alpha_non_infectious',
    })

    def __init__(
        self, pars=None, sim_pars=None, tb_pars=None, resistance_pars=None,
        strains=None, strain_preset='standard', cascade=None, cascade_pars=None,
        connector=True, resistance=True, analyzers='default', **kwargs,
    ):
        pars = sc.mergedicts(pars, kwargs)
        sim_pars = sc.mergedicts(sim_pars)
        tb_pars = sc.mergedicts(tb_pars)
        resistance_pars = sc.mergedicts(resistance_pars)
        cascade_pars = sc.mergedicts(cascade_pars)

        extra_interventions = sc.mergelists(pars.pop('interventions', None))
        extra_connectors = sc.mergelists(pars.pop('connectors', None))
        extra_analyzers = sc.mergelists(pars.pop('analyzers', None))
        networks = sc.mergelists(pars.pop('networks', None))
        demographics = sc.mergelists(pars.pop('demographics', None))

        if not resistance:
            if cascade:
                raise ValueError('resistance=False cannot be combined with a strain-aware cascade.')
            tb = tbsim.TB(pars=tb_pars)
            connectors_list = list(extra_connectors or [])
            analyzer_list = sc.mergelists(analyzers) if analyzers not in (False, None, 'default') else []
            analyzer_list.extend(extra_analyzers or [])
            init_kw = dict(sim_pars=sim_pars, tb_pars=tb_pars, tb_model=tb,
                           interventions=extra_interventions or None,
                           connectors=connectors_list or None,
                           analyzers=analyzer_list or None)
            if networks is not None:
                init_kw['networks'] = networks
            if demographics is not None:
                init_kw['demographics'] = demographics
            init_kw.update(pars)
            super().__init__(**init_kw)
            return

        # Pull resistance kwargs out of flat pars / resistance_pars
        resistance_kw = dict(progression_mode='bottleneck', p_multi=1.0)
        for src in (resistance_pars, pars):
            for key in list(src.keys()):
                if key in self._RESISTANCE_KEYS:
                    resistance_kw[key] = src.pop(key)

        if strains is None and 'strains' in resistance_kw:
            strains = resistance_kw.pop('strains')
        strain_list = _resolve_strains(strains, strain_preset)

        # Route remaining flat pars to tb_pars
        default_tb_keys = set(tbsim.TB().pars.keys())
        default_sim_keys = set(ss.SimPars().keys())
        for key in list(pars.keys()):
            if key in default_tb_keys:
                val = pars[key] if key in default_sim_keys else pars.pop(key)
                tb_pars[key] = val

        tb = MultiStrainTB(strains=strain_list, pars=tb_pars, **resistance_kw)

        connectors_list = []
        if connector:
            connectors_list.append(ResistanceConnector())
        connectors_list.extend(extra_connectors or [])

        interventions = []
        if cascade:
            mode = 'basic' if cascade is True else str(cascade)
            interventions.extend(build_care_cascade(tb, mode=mode, **cascade_pars))
        interventions.extend(extra_interventions or [])

        analyzer_list = []
        if analyzers is True or analyzers == 'default':
            analyzer_list = [StrainResults(), DuplicateStrainAnalyzer()]
        elif analyzers not in (False, None):
            analyzer_list = sc.mergelists(analyzers)
        analyzer_list.extend(extra_analyzers or [])

        init_kw = dict(sim_pars=sim_pars, tb_pars=tb_pars, tb_model=tb,
                       interventions=interventions or None,
                       connectors=connectors_list or None,
                       analyzers=analyzer_list or None)
        if networks is not None:
            init_kw['networks'] = networks
        if demographics is not None:
            init_kw['demographics'] = demographics
        init_kw.update(pars)
        super().__init__(**init_kw)

    def get_multistrain_tb(self):
        """Return the :class:`MultiStrainTB` module, or ``None`` if resistance is off."""
        try:
            return self.get_tb(MultiStrainTB)
        except (ValueError, KeyError, TypeError):
            return None

    def get_dst(self, name='dst'):
        """Return a :class:`DSTDelivery` by name, or None."""
        for intv in self.interventions.values():
            if isinstance(intv, DSTDelivery) and intv.name == name:
                return intv
        return None

    def get_strain_results(self):
        """Return :class:`StrainResults` if present."""
        for analyzer in self.analyzers.values():
            if isinstance(analyzer, StrainResults):
                return analyzer
        return None
