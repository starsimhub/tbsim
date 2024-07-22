import sciris as sc
import tbsim as mtb
from tests.e2e.harlem.functionparams import compute_rel_prog, compute_rel_prog_alternate, run_scen, p_micro_recovery_alt, p_cure_func

__all__ = ['Scenarios', 'Arms' ]
class Scenarios():
    # Default scenario settings
    @staticmethod
    def defaults():
        defaults = {
            'relsus_microdeficient': 1,
            'n_hhs': 194 / 2,
        }
        return defaults

    # Define scenarios
    @staticmethod
    def get_scenarios(scen_filter=None):
        scenarios_dict = sc.odict(
            Base={
                'beta': 0.12,
                'active': run_scen('Base', scen_filter),
            },
            MoreMicroDeficient={
                'beta': 0.10,
                'p_microdeficient_given_macro': {
                    mtb.MacroNutrients.UNSATISFACTORY: 1.0,
                    mtb.MacroNutrients.MARGINAL: 1.0,
                    mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD: 0.75,
                    mtb.MacroNutrients.STANDARD_OR_ABOVE: 0.5,
                },
                'active': run_scen('MoreMicroDeficient', scen_filter),
            },
            LatentSeeding={
                'beta': 0.05,
                'init_prev': 0.33,
                'active': run_scen('LatentSeeding', scen_filter),
            },
            RelSus={
                'relsus_microdeficient': 5,
                'beta': 0.04,
                'active': run_scen('RelSus', scen_filter),
            },
            LSProgAlt={
                'beta': 0.09,
                'rel_LS_prog_func': compute_rel_prog_alternate,
                'active': run_scen('LSProgAlt', scen_filter),
            },
            LatentFast={
                'beta': 0.12,
                'rel_LF_prog_func': compute_rel_prog,
                'active': run_scen('LatentFast', scen_filter),
            },
            FastSlowAlt={
                'beta': 0.09,
                'rel_LF_prog_func': compute_rel_prog_alternate,
                'rel_LS_prog_func': compute_rel_prog_alternate,
                'active': run_scen('LatentFast', scen_filter),
            },
            AllVitamin={
                'beta': 0.11,
                'p_micro_recovery_func': p_micro_recovery_alt,
                'active': run_scen('AllVitamin', scen_filter),
            },
            LatentClearance={
                'beta': 0.16,
                'active': run_scen('LatentClearance', scen_filter),
                'p_clearance_func': p_cure_func,
            },
            NoSecular={
                'beta': 0.12,
                'secular_trend': False,
                'active': False,
            },
            SecularMicro={
                'beta': 0.14,
                'p_new_micro': 0.5,
                'active': False,
            },
        )
        return scenarios_dict
    
class Arms(Scenarios):
    
    @staticmethod
    def create_matching_arms(vitamin_year_rate=[(1942, 10.0), (1943, 3.0)], calib=False, scen_filter=None):
    # Create matching CONTROL and VITAMIN arms for each scenario
        scens = {}
        for skey, scn in Scenarios.get_scenarios(scen_filter).items():
            if 'active' in scn and not scn['active']:
                continue
            control = Scenarios.defaults().copy() | scn.copy()
            control['p_control'] = 1
            control['vitamin_year_rate'] = None
            control['skey'] = skey
            control['arm'] = 'CONTROL'
            scens[f'{skey} CONTROL'] = control

            if not calib:
                vitamin = control.copy()
                vitamin['p_control'] = 0
                vitamin['vitamin_year_rate'] = vitamin_year_rate
                vitamin['arm'] = 'VITAMIN'
                vitamin['ref'] = f'{skey} CONTROL'
                scens[f'{skey} VITAMIN'] = vitamin
                
        return scens