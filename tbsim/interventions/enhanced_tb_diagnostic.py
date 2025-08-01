import numpy as np
import starsim as ss
from tbsim import TBS

__all__ = ['EnhancedTBDiagnostic']


class EnhancedTBDiagnostic(ss.Intervention):
    """
    Enhanced TB diagnostic intervention that combines detailed parameter stratification
    from interventions_updated.py with health-seeking integration from tb_diagnostic.py.
    
    This intervention provides:
    1. Age and TB state-specific sensitivity/specificity parameters
    2. HIV-stratified parameters for LAM testing
    3. Integration with health-seeking behavior
    4. False negative handling with care-seeking multipliers
    5. Comprehensive result tracking
    """
    
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        # Define comprehensive parameters combining both approaches
        self.define_pars(
            # Coverage and basic parameters (from tb_diagnostic.py)
            coverage=1.0,
            reset_flag=False,
            care_seeking_multiplier=1.0,
            
            # Xpert baseline parameters (from interventions_updated.py)
            sensitivity_adult_smearpos=0.909,
            specificity_adult_smearpos=0.966,
            sensitivity_adult_smearneg=0.775,
            specificity_adult_smearneg=0.958,
            sensitivity_adult_eptb=0.775,
            specificity_adult_eptb=0.958,
            sensitivity_child=0.73,
            specificity_child=0.95,
            
            # Oral swab parameters (optional)
            use_oral_swab=False,
            sens_adult_smearpos_oral=0.80,
            spec_adult_smearpos_oral=0.90,
            sens_adult_smearneg_oral=0.30,
            spec_adult_smearneg_oral=0.98,
            sens_child_oral=0.25,
            spec_child_oral=0.95,
            
            # FujiLAM parameters (optional)
            use_fujilam=False,
            sens_hivpos_adult_tb=0.75,
            spec_hivpos_adult_tb=0.90,
            sens_hivpos_adult_eptb=0.75,
            spec_hivpos_adult_eptb=0.739,
            sens_hivneg_adult=0.58,
            spec_hivneg_adult=0.98,
            sens_hivpos_child=0.579,
            spec_hivpos_child=0.877,
            sens_hivneg_child=0.51,
            spec_hivneg_child=0.895,
            
            # CAD CXR parameters (optional)
            use_cadcxr=False,
            cad_cxr_sensitivity=0.66,
            cad_cxr_specificity=0.79,
        )
        self.update_pars(pars=pars, **kwargs)

        # Temporary state for update_results
        self.tested_this_step = []
        self.test_result_this_step = []
        self.diagnostic_method_used = []  # Track which diagnostic was used

    def _get_diagnostic_parameters(self, uid, age, tb_state, hiv_state=None):
        """
        Get sensitivity and specificity based on individual characteristics.
        """
        is_child = age < 15
        hiv_positive = hiv_state is not None and hiv_state in [1, 2, 3]  # ACUTE, LATENT, AIDS
        
        # Determine which diagnostic method to use
        if self.pars.use_fujilam and hiv_state is not None:
            # FujiLAM for HIV-positive individuals
            if is_child:
                if hiv_positive:
                    return self.pars.sens_hivpos_child, self.pars.spec_hivpos_child
                else:
                    return self.pars.sens_hivneg_child, self.pars.spec_hivneg_child
            else:
                if hiv_positive:
                    if tb_state == TBS.ACTIVE_EXPTB:
                        return self.pars.sens_hivpos_adult_eptb, self.pars.spec_hivpos_adult_eptb
                    else:
                        return self.pars.sens_hivpos_adult_tb, self.pars.spec_hivpos_adult_tb
                else:
                    return self.pars.sens_hivneg_adult, self.pars.spec_hivneg_adult
        
        elif self.pars.use_cadcxr and is_child and tb_state != TBS.ACTIVE_EXPTB:
            # CAD CXR for children (not EPTB)
            return self.pars.cad_cxr_sensitivity, self.pars.cad_cxr_specificity
        
        elif self.pars.use_oral_swab:
            # Oral swab parameters
            if is_child:
                return self.pars.sens_child_oral, self.pars.spec_child_oral
            elif tb_state == TBS.ACTIVE_SMPOS:
                return self.pars.sens_adult_smearpos_oral, self.pars.spec_adult_smearpos_oral
            elif tb_state == TBS.ACTIVE_SMNEG:
                return self.pars.sens_adult_smearneg_oral, self.pars.spec_adult_smearneg_oral
            else:
                return self.pars.sens_adult_smearneg_oral, self.pars.spec_adult_smearneg_oral
        
        else:
            # Default Xpert baseline parameters
            if is_child:
                return self.pars.sensitivity_child, self.pars.specificity_child
            elif tb_state == TBS.ACTIVE_SMPOS:
                return self.pars.sensitivity_adult_smearpos, self.pars.specificity_adult_smearpos
            elif tb_state == TBS.ACTIVE_SMNEG:
                return self.pars.sensitivity_adult_smearneg, self.pars.specificity_adult_smearneg
            elif tb_state == TBS.ACTIVE_EXPTB:
                return self.pars.sensitivity_adult_eptb, self.pars.specificity_adult_eptb
            else:
                return 0.0, 1.0  # Default for unknown states

    def _determine_diagnostic_method(self, uid, age, tb_state, hiv_state=None):
        """
        Determine which diagnostic method to use and return method name.
        """
        is_child = age < 15
        hiv_positive = hiv_state is not None and hiv_state in [1, 2, 3]
        
        if self.pars.use_fujilam and hiv_state is not None:
            return "FujiLAM"
        elif self.pars.use_cadcxr and is_child and tb_state != TBS.ACTIVE_EXPTB:
            return "CAD_CXR"
        elif self.pars.use_oral_swab:
            return "Oral_Swab"
        else:
            return "Xpert_Baseline"

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = sim.diseases.tb

        # Find people who sought care but haven't been diagnosed
        eligible = ppl.sought_care & (~ppl.diagnosed) & ppl.alive
        uids = eligible.uids
        if len(uids) == 0:
            return

        # Apply coverage filter
        if isinstance(self.pars.coverage, ss.Dist):
            selected = self.pars.coverage.filter(uids)
        else:
            selected = ss.bernoulli(self.pars.coverage).filter(uids)
        if len(selected) == 0:
            return

        # Get TB and HIV states for selected individuals
        tb_states = tb.state[selected]
        ages = ppl.age[selected]
        
        # Get HIV state if HIV disease exists
        hiv_states = None
        if hasattr(sim.diseases, 'hiv'):
            hiv_states = sim.diseases.hiv.state[selected]

        # Determine TB status
        has_tb = np.isin(tb_states, [TBS.ACTIVE_SMPOS,
                                     TBS.ACTIVE_SMNEG,
                                     TBS.ACTIVE_EXPTB])

        # Apply diagnostic testing with individual-specific parameters
        test_positive = np.zeros(len(selected), dtype=bool)
        diagnostic_methods = []

        for i, uid in enumerate(selected):
            age_i = float(ages[i])
            tb_state_i = tb_states[i]
            hiv_state_i = hiv_states[i] if hiv_states is not None else None
            
            # Get sensitivity/specificity for this individual
            sensitivity, specificity = self._get_diagnostic_parameters(
                uid, age_i, tb_state_i, hiv_state_i
            )
            
            # Determine diagnostic method used
            method = self._determine_diagnostic_method(
                uid, age_i, tb_state_i, hiv_state_i
            )
            diagnostic_methods.append(method)
            
            # Apply test logic
            rand = np.random.rand()
            has_tbi = has_tb[i]
            
            if has_tbi:
                test_positive[i] = rand < sensitivity
            else:
                test_positive[i] = rand > (1 - specificity)

        # Update person state
        ppl.tested[selected] = True
        ppl.n_times_tested[selected] += 1
        ppl.test_result[selected] = test_positive
        ppl.diagnosed[selected[test_positive]] = True

        # Optional: reset the health-seeking flag
        if self.pars.reset_flag:
            ppl.sought_care[selected] = False

        # Handle false negatives: schedule another round of health-seeking
        false_negative_uids = selected[~test_positive & has_tb]

        if len(false_negative_uids):
            # Filter only those who haven't had multiplier applied yet
            unboosted = false_negative_uids[~ppl.multiplier_applied[false_negative_uids]]

            # Apply multiplier only to them
            if len(unboosted):
                ppl.care_seeking_multiplier[unboosted] *= self.pars.care_seeking_multiplier
                ppl.multiplier_applied[unboosted] = True

            # Reset flags to allow re-care-seeking
            ppl.sought_care[false_negative_uids] = False
            ppl.tested[false_negative_uids] = False

        # Store for update_results
        self.tested_this_step = selected
        self.test_result_this_step = test_positive
        self.diagnostic_method_used = diagnostic_methods

    def init_results(self):
        self.define_results(
            ss.Result('n_tested', dtype=int),
            ss.Result('n_test_positive', dtype=int),
            ss.Result('n_test_negative', dtype=int),
            ss.Result('cum_test_positive', dtype=int),
            ss.Result('cum_test_negative', dtype=int),
            ss.Result('n_xpert_baseline', dtype=int),
            ss.Result('n_oral_swab', dtype=int),
            ss.Result('n_fujilam', dtype=int),
            ss.Result('n_cadcxr', dtype=int),
        )

    def update_results(self):
        # Per-step counts
        n_tested = len(self.tested_this_step)
        n_pos = np.count_nonzero(self.test_result_this_step)
        n_neg = n_tested - n_pos

        self.results['n_tested'][self.ti] = n_tested
        self.results['n_test_positive'][self.ti] = n_pos
        self.results['n_test_negative'][self.ti] = n_neg

        # Cumulative totals (add to previous step)
        if self.ti > 0:
            self.results['cum_test_positive'][self.ti] = self.results['cum_test_positive'][self.ti-1] + n_pos
            self.results['cum_test_negative'][self.ti] = self.results['cum_test_negative'][self.ti-1] + n_neg
        else:
            self.results['cum_test_positive'][self.ti] = n_pos
            self.results['cum_test_negative'][self.ti] = n_neg

        # Count diagnostic methods used
        if hasattr(self, 'diagnostic_method_used') and self.diagnostic_method_used:
            methods = np.array(self.diagnostic_method_used)
            self.results['n_xpert_baseline'][self.ti] = np.sum(methods == 'Xpert_Baseline')
            self.results['n_oral_swab'][self.ti] = np.sum(methods == 'Oral_Swab')
            self.results['n_fujilam'][self.ti] = np.sum(methods == 'FujiLAM')
            self.results['n_cadcxr'][self.ti] = np.sum(methods == 'CAD_CXR')
        else:
            self.results['n_xpert_baseline'][self.ti] = 0
            self.results['n_oral_swab'][self.ti] = 0
            self.results['n_fujilam'][self.ti] = 0
            self.results['n_cadcxr'][self.ti] = 0

        # Reset temporary storage
        self.tested_this_step = []
        self.test_result_this_step = []
        self.diagnostic_method_used = []


# Example usage function
def create_enhanced_diagnostic_scenarios():
    """
    Create different diagnostic scenarios using the enhanced intervention.
    """
    scenarios = {
        'baseline': {
            'use_oral_swab': False,
            'use_fujilam': False,
            'use_cadcxr': False,
        },
        'oral_swab': {
            'use_oral_swab': True,
            'use_fujilam': False,
            'use_cadcxr': False,
        },
        'fujilam': {
            'use_oral_swab': False,
            'use_fujilam': True,
            'use_cadcxr': False,
        },
        'cadcxr': {
            'use_oral_swab': False,
            'use_fujilam': False,
            'use_cadcxr': True,
        },
        'combo_all': {
            'use_oral_swab': True,
            'use_fujilam': True,
            'use_cadcxr': True,
        }
    }
    return scenarios


if __name__ == '__main__':
    import tbsim as mtb
    import starsim as ss
    import matplotlib.pyplot as plt

    # Example simulation with enhanced diagnostic
    sim = ss.Sim(
        people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
        diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
        interventions=[
            mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),
            EnhancedTBDiagnostic(pars={
                'coverage': ss.bernoulli(0.8, strict=False),
                'use_oral_swab': True,
                'use_fujilam': True,
                'care_seeking_multiplier': 2.0,
            }),
        ],
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
        pars=dict(start=2000, stop=2010, dt=1/12),
    )
    sim.run()

    # Plot results
    tbdiag = sim.results['enhancedtbdiagnostic']
    
    plt.figure(figsize=(12, 8))
    
    # Plot diagnostic methods used
    plt.subplot(2, 2, 1)
    plt.plot(tbdiag['n_xpert_baseline'].timevec, tbdiag['n_xpert_baseline'].values, label='Xpert Baseline')
    plt.plot(tbdiag['n_oral_swab'].timevec, tbdiag['n_oral_swab'].values, label='Oral Swab')
    plt.plot(tbdiag['n_fujilam'].timevec, tbdiag['n_fujilam'].values, label='FujiLAM')
    plt.plot(tbdiag['n_cadcxr'].timevec, tbdiag['n_cadcxr'].values, label='CAD CXR')
    plt.xlabel('Time')
    plt.ylabel('Number of Tests')
    plt.title('Diagnostic Methods Used')
    plt.legend()
    plt.grid(True)
    
    # Plot test outcomes
    plt.subplot(2, 2, 2)
    plt.plot(tbdiag['n_test_positive'].timevec, tbdiag['n_test_positive'].values, label='Positive')
    plt.plot(tbdiag['n_test_negative'].timevec, tbdiag['n_test_negative'].values, label='Negative')
    plt.xlabel('Time')
    plt.ylabel('Number of Tests')
    plt.title('Test Outcomes')
    plt.legend()
    plt.grid(True)
    
    # Plot cumulative results
    plt.subplot(2, 2, 3)
    plt.plot(tbdiag['cum_test_positive'].timevec, tbdiag['cum_test_positive'].values, label='Cumulative Positive')
    plt.plot(tbdiag['cum_test_negative'].timevec, tbdiag['cum_test_negative'].values, label='Cumulative Negative')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Tests')
    plt.title('Cumulative Results')
    plt.legend()
    plt.grid(True)
    
    # Plot total tested
    plt.subplot(2, 2, 4)
    plt.plot(tbdiag['n_tested'].timevec, tbdiag['n_tested'].values, label='Total Tested')
    plt.xlabel('Time')
    plt.ylabel('Number of People')
    plt.title('Total Tests Per Time Step')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 