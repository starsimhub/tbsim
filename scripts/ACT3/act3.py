# Do we still need this file? It seems to be a duplicate of interevention ACF?

# from tbsim import TBS
# import sciris as sc

# class ACT3(tbsim.ActiveCaseFinding):

#     def __init__(self, pars=None, **kwargs):
#         '''
#         self.define_pars(
#             age_min = 15,
#             p_contact_and_consent = ss.bernoulli(),
#             sensitivity = {
#                 TBS.ACTIVE_PRESYMP: 0,
#                 TBS.ACTIVE_SMPOS: 1,
#                 TBS.ACTIVE_SMNEG: 0,
#                 TBS.ACTIVE_EXPTB: 0,
#             },
#             trial_visit_dates = [{
#                 sc.date('2014-06-01'): 0.6,
#                 sc.date('2015-06-01'): 0.7,
#                 sc.date('2016-06-01'): 0.64
#             }]
#         )
#         self.update_pars()

#         self.define_state(
#             # STATES
#         )
#         '''
#     return

#     def init_results(self):
#         self.define_results(
#             ss.Result('n_tested_positive', type=int, scale=True)
#         )
#         pass

#     def step(self):
#         num_pos_found, num_eligible = super().step()
#         # Update results here
#         self.results['n_tested_positive'][self.ti] = num_pos_found
#         return