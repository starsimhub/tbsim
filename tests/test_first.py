import unittest
import numpy as np
import starsim as ss

from tbsim.tbmodel import TBModel


class TestTBModel(unittest.TestCase):
    def setUp(self):
        self.tb_model = TBModel()

    def test_init(self):
        self.assertIsInstance(self.tb_model, TBModel)
        self.assertIsInstance(self.tb_model.latent_fast, ss.State)
        self.assertIsInstance(self.tb_model.latent_slow, ss.State)
        self.assertIsInstance(self.tb_model.active_pre_symptomatic, ss.State)
        self.assertIsInstance(self.tb_model.smear_positive, ss.State)
        self.assertIsInstance(self.tb_model.smear_negative, ss.State)
        self.assertIsInstance(self.tb_model.extra_pulmonary, ss.State)
        self.assertIsInstance(self.tb_model.dead, ss.State)

    def test_update_pre(self):
        sim = ss.Sim()  # Assuming Simulation is a class in starsim
        self.tb_model.update_pre(sim)
        # Add assertions here based on expected changes in the model states

    def test_set_prognoses(self):
        sim = ss.Sim()  # Assuming Simulation is a class in starsim
        uids = np.array([1, 2, 3])
        from_uids = np.array([1, 2, 3])
        self.tb_model.set_prognoses(sim, uids, from_uids)
        # Add assertions here based on expected changes in the model states

if __name__ == '__main__':
    unittest.main()