import unittest
import numpy as np
import starsim as ss
import tbsim as mtb

def make_tb_simplified(agents=20, start=2000, end=2020, dt=7/365):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(pars={'beta':0.01, 'init_prev':0.25})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur = 0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=dict(dt = dt, start = start, end = end), demographics=dems)
    sim.pars.verbose = sim.pars.dt / 5
    return sim

class TestTB(unittest.TestCase):

    def test_initial_states(self):
        # TODO: Verify why is failing:  AttributeError: 'TB' object has no attribute 'latent_fast'
        # Test the initial states of the TB class.
        tb =mtb.TB()
        print(tb.states)
        self.assertIsInstance(tb.state, ss.FloatArr)
        self.assertIsInstance(tb.active_tb_state, ss.FloatArr)
        self.assertIsInstance(tb.rel_LS_prog, ss.FloatArr)
        self.assertIsInstance(tb.rel_LF_prog, ss.FloatArr)
        self.assertIsInstance(tb.ppf_LS_to_presymp, ss.FloatArr)
        self.assertIsInstance(tb.ppf_LF_to_presymp, ss.FloatArr)
        # Test the initial states of the TB class.
        self.assertIsInstance(tb.ti_latent, ss.FloatArr)
        self.assertIsInstance(tb.ti_presymp, ss.FloatArr)
        self.assertIsInstance(tb.ti_active, ss.FloatArr)
        self.assertIsInstance(tb.ti_dead, ss.FloatArr)
        self.assertIsInstance(tb.ti_cure, ss.FloatArr)
        pass
        
    def test_default_parameters(self):
        tb = mtb.TB()
        print(tb)
        self.assertIsNotNone(tb.pars['init_prev'])
        self.assertIsInstance(tb.pars['rate_LS_to_presym'], float)
        self.assertIsInstance(tb.pars['rate_LF_to_presym'], float)
        self.assertIsInstance(tb.pars['rate_active_to_cure'], float)
        self.assertIsInstance(tb.pars['rate_exptb_to_dead'], float)
        self.assertIsInstance(tb.pars['rate_smpos_to_dead'], float)
        self.assertIsInstance(tb.pars['rate_smneg_to_dead'], float)
        self.assertIsInstance(tb.pars['rel_trans_smpos'], float)


    def test_infectious(self):
        self.assertFalse(self.tb.infectious)

        self.tb.state = mtb.TBS.ACTIVE_PRESYMP
        self.assertTrue(self.tb.infectious)

        self.tb.state = mtb.TBS.ACTIVE_SMPOS
        self.assertTrue(self.tb.infectious)

        self.tb.state = mtb.TBS.ACTIVE_SMNEG
        self.assertTrue(self.tb.infectious)

        self.tb.state = mtb.TBS.ACTIVE_EXPTB
        self.assertTrue(self.tb.infectious)

        
    def test_set_prognoses(self):
        sim = make_tb_simplified()
        sim.run()
        tb = sim.diseases['tb']
        before = tb.state
        uids = ss.uids([1, 2, 3, 7, 9])
        tb.set_prognoses(uids)
        after = tb.state
        self.assertNotEqual(before, after)
        print("Before: ", before)
        print("After: ", after)
        
    def test_update_pre(self):
        sim = make_tb_simplified(agents=300)
        sim.run()
        tb = sim.diseases['tb']
        self.assertEqual(len(tb.state[tb.state == mtb.TBS.NONE]), 0)
        self.assertGreater(len(tb.state[tb.state == mtb.TBS.LATENT_SLOW]), 0)
        self.assertGreater(len(tb.state[tb.state == mtb.TBS.ACTIVE_SMNEG]), 0)
        
        print("none", tb.state[tb.state == mtb.TBS.NONE])
        print("Slow:", tb.state[tb.state == mtb.TBS.LATENT_SLOW])
        print("Fast:" ,tb.state[tb.state == mtb.TBS.LATENT_FAST])
        print("Active Presymp:", tb.state[tb.state == mtb.TBS.ACTIVE_PRESYMP])
        print("Active ExpTB:", tb.state[tb.state == mtb.TBS.ACTIVE_EXPTB])
        print("Active Smear Negative: ", tb.state[tb.state == mtb.TBS.ACTIVE_SMNEG])
        print("Active Smear Positive: ",tb.state[tb.state == mtb.TBS.ACTIVE_SMPOS])
        


if __name__ == '__main__':
    unittest.main()

    