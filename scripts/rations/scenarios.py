from functools import partial
import tbsim as mtb
import starsim as ss
import numpy as np

def clearance_rr_func(tb, mn, uids, rate_ratio=2):
    rr = np.ones_like(uids)
    rr[mn.receiving_macro[uids] & mn.receiving_micro[uids]] = rate_ratio
    return rr

scens = {
    'Baseline': {
        'Skip': False,
    },

    'Lönnroth Nutrition-->TB activation link': {
        'Skip': False,
        'Connector': dict(
            rr_activation_func = partial(mtb.TB_Nutrition_Connector.lonnroth_bmi_rr, scale=3, slope=3, bmi50=20),
            rr_clearance_func = mtb.TB_Nutrition_Connector.ones_rr,
        ),
    },

    'Rel trans het + Nutrition-->TB activation': {
        'Skip': False,
        'TB': dict(
            reltrans_het = ss.gamma(a=0.5, scale=2), # mean = a*scale (keep the product equal to 1)
        ),
        'Connector': dict(
            rr_activation_func = partial(mtb.TB_Nutrition_Connector.supplementation_rr, rate_ratio=0.1),
        ),
    },
    'Nutrition-->TB activation link': {
        'Skip': False,
        'Connector': dict(
            rr_activation_func = partial(mtb.TB_Nutrition_Connector.supplementation_rr, rate_ratio=0.1),
            rr_clearance_func = mtb.TB_Nutrition_Connector.ones_rr,
        ),
    },
    'Nutrition-->TB clearance link': {
        'Skip': False,
        'Connector': dict(
            rr_activation_func = mtb.TB_Nutrition_Connector.ones_rr,
            rr_clearance_func = partial(clearance_rr_func, rate_ratio=10),
            ),
    },
    'Increase index treatment seeking delays': {
        'Skip': False,
        'TB': None,
        'Malnutrition': None,
        'Connector': None,
        'RATIONS': dict(dur_active_to_dx = ss.years(ss.weibull(c=2, scale=6/12))),
        'Simulation': None,
    },
}
