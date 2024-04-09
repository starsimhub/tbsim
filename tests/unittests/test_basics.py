import unittest
import numpy as np
import starsim as ss
import tbsim as tbs


class TestTB(unittest.TestCase):

    def test_initial_states(self):
        # Test the initial states of the TB class.
        tb =tbs.TB()
        self.assertIsInstance(tb.latent_fast, ss.State)
        self.assertIsInstance(tb.latent_slow, ss.State)
        self.assertIsInstance(tb.active_pre_symptomatic, ss.State)
        self.assertIsInstance(tb.smear_positive, ss.State)
        self.assertIsInstance(tb.smear_negative, ss.State)
        self.assertIsInstance(tb.extra_pulmonary, ss.State)
        self.assertIsInstance(tb.dead, ss.State)
        # Test the initial states of the TB class.
        self.assertIsInstance(tb.ti_exposed, ss.State)
        self.assertIsInstance(tb.ti_latent_slow, ss.State)
        self.assertIsInstance(tb.ti_latent_fast, ss.State)
        self.assertIsInstance(tb.ti_active_presymp, ss.State)
        self.assertIsInstance(tb.tx_smpos, ss.State)
        self.assertIsInstance(tb.tx_smneg, ss.State)
        self.assertIsInstance(tb.ti_exptb, ss.State)
        
    def test_default_parameters(self):
        tb = tbs.TB()
        print(tb.pars)
        self.assertIsInstance(tb.pars['tb_latent_cure_rate'], float)
        self.assertIsInstance(tb.pars['tb_fast_progressor_rate'], float)
        self.assertIsInstance(tb.pars['tb_slow_progressor_rate'], float)
        self.assertIsInstance(tb.pars['tb_active_cure_rate'], float)
        self.assertIsInstance(tb.pars['tb_inactivation_rate'], float)
        self.assertIsInstance(tb.pars['tb_active_mortality_rate'], float)
        self.assertIsInstance(tb.pars['tb_extrapulmonary_mortality_multiplier'], float)
        self.assertIsInstance(tb.pars['tb_smear_negative_mortality_multiplier'], float)

    def test_update_pre(self):
        pass
    def test_make_new_cases(self):
        pass
    def test_set_prognoses(self):
        tb = tbs.TB()
        pass

if __name__ == '__main__':
    unittest.main()
    