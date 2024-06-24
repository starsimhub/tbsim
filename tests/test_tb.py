import unittest
import tbsim as mtb
import starsim as ss
import numpy as np

def make_tb_simplified(agents=10, start=2000, end=2020, dt=7/365):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(pars={'beta':0.01, 'init_prev':0.25})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur = 0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=dict(dt = dt, start = start, end = end), demographics=dems)
    sim.pars.verbose = sim.pars.dt / 5
    return sim
    
class TestTB(unittest.TestCase):
    

    def setUp(self):
        self.tb = mtb.TB()

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
        uids = ss.uids([1, 2, 3])
        tb.set_prognoses(uids)

        self.assertFalse(all(tb.susceptible[uids]))
        self.assertTrue(all(tb.infected[uids]))
        self.assertTrue(all(tb.ti_infected[uids] == tb.sim.ti))
        self.assertTrue(all(tb.ti_latent[uids] == tb.sim.ti))

        self.assertEqual(tb.state[uids[0]], mtb.TBS.LATENT_SLOW)
        self.assertEqual(tb.state[uids[1]], mtb.TBS.LATENT_FAST)
        self.assertEqual(tb.state[uids[2]], mtb.TBS.LATENT_FAST)

        self.assertEqual(tb.active_tb_state[uids[0]], mtb.TBS.NONE)
        self.assertEqual(tb.active_tb_state[uids[1]], mtb.TBS.NONE)
        self.assertEqual(tb.active_tb_state[uids[2]], mtb.TBS.NONE)

    def test_update_pre(self):
        sim = make_tb_simplified(agents=300)
        sim.run()
        tb = sim.diseases['tb']
        self.assertEqual(len(tb.state[tb.state == mtb.TBS.NONE]), 0)
        self.assertGreater(len(tb.state[tb.state == mtb.TBS.LATENT_SLOW]), 0)
        self.assertGreater(len(tb.state[tb.state == mtb.TBS.ACTIVE_SMNEG]), 0)
        
        print("none", tb.state[tb.state == mtb.TBS.NONE])
        print("Slow:", tb.state[tb.state == mtb.TBS.LATENT_SLOW])
        print("Fast: ,tb.state[tb.state == mtb.TBS.LATENT_FAST])
        print(tb.state[tb.state == mtb.TBS.ACTIVE_PRESYMP])
        print(tb.state[tb.state == mtb.TBS.ACTIVE_EXPTB])
        print(tb.state[tb.state == mtb.TBS.ACTIVE_SMNEG])
        print(tb.state[tb.state == mtb.TBS.ACTIVE_SMPOS])
        
        
    def test_update_death(self):
        sim = make_tb_simplified()
        sim.run()
        tb = sim.diseases['tb']
        uids = ss.uids([1, 2, 3])
        tb.update_death(uids)
        
        self.assertFalse(all(tb.susceptible[uids]))
        self.assertFalse(all(tb.infected[uids]))
        self.assertEqual(all(tb.rel_trans[uids]), 0)

if __name__ == '__main__':
    unittest.main()
