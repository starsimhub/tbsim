"""
Tests for malnutrition comorbidity module

This module tests the Malnutrition disease model, its effects on TB, and the effect of HIV on TB
"""

import pytest
import numpy as np
import pandas as pd
import starsim as ss
import tbsim


class TestMalnutrition:
    """Test the Malnutrition disease model"""
    
    def test_malnutrition_creation(self):
        """Test that Malnutrition disease can be created"""
        nut = tbsim.Malnutrition()
        assert isinstance(nut, tbsim.Malnutrition)
        assert isinstance(nut, ss.Disease)
    
    def test_malnutrition_states(self):
        """Test that Malnutrition states are defined correctly"""
        nut = tbsim.Malnutrition()
        # Check that state arrays exist
        assert hasattr(nut, 'receiving_macro')
        assert hasattr(nut, 'receiving_micro')
        assert hasattr(nut, 'height_percentile')
        assert hasattr(nut, 'weight_percentile')
        assert hasattr(nut, 'micro')
    
    def test_malnutrition_lms_data_loading(self):
        """Test that LMS data is loaded correctly"""
        nut = tbsim.Malnutrition()
        assert hasattr(nut, 'LMS_data')
        assert isinstance(nut.LMS_data, pd.DataFrame)
        # Check that data contains expected keys
        assert 'Female' in nut.LMS_data.index
        assert 'Male' in nut.LMS_data.index
    
    def test_malnutrition_dweight_scale(self):
        """The weight random walk takes steps of the same size throughout the run, scaled by the square root of the step"""
        sim = ss.Sim(diseases=tbsim.Malnutrition(weight_sd=0.05), n_agents=100, start=2000, stop=2010, dt=ss.month, verbose=0).init()
        mn = sim.diseases.malnutrition
        uids = sim.people.auids
        expected = 0.05*np.sqrt(1/12)
        assert np.isclose(mn.dweight_scale(mn, sim, uids), expected, rtol=0.01)
        sim.run()
        assert np.isclose(mn.dweight_scale(mn, sim, uids), expected, rtol=0.01)
    
    def test_malnutrition_lms(self):
        """Weight and height from the LMS method are plausible, including for children under 2 (who have a length, not a height)"""
        sim = ss.Sim(diseases=tbsim.Malnutrition(), n_agents=1000, start=2000, stop=2001, verbose=0).init()
        mn = sim.diseases.malnutrition
        uids = sim.people.auids
        weight, height = mn.weight(uids), mn.height(uids)
        assert np.all(np.isfinite(weight)) and np.all(np.isfinite(height))
        adults = sim.people.age[uids] >= 20
        assert 40 < np.median(weight[adults]) < 100 and 140 < np.median(height[adults]) < 190


class TestTBNutritionConnector:
    """Test the effects of nutrition on TB (tbsim.nutrition_effects) and the risk-ratio functions"""

    def test_effects_are_modifiers(self):
        """nutrition_effects returns modifiers of the TB transitions and susceptibility"""
        tb = tbsim.TB()
        mods = tbsim.nutrition_effects(tb, rr_activation=tbsim.lonnroth_bmi_rr)
        assert all(isinstance(m, ss.modify) for m in mods)
        assert [m.target_name or m.target_kind for m in mods] == ['latent_asy', 'latent_non', 'recovery', 'rel_sus']

    def test_ones_rr_function(self):
        """ones_rr returns neutral risk ratios"""
        rr = tbsim.ones_rr(None, None, np.arange(5))
        assert len(rr) == 5 and np.all(rr == 1.0)

    def test_compute_relsus_function(self):
        """compute_relsus doubles susceptibility where micronutrient status is below 0.2"""
        class MockMalnutrition:
            micro = np.array([0.1, 0.3, 0.5, 0.2, 0.4])
        rel_sus = tbsim.compute_relsus(None, MockMalnutrition(), np.arange(5))
        assert np.array_equal(rel_sus, [2, 1, 1, 1, 1])

    def test_supplementation_rr_function(self):
        """supplementation_rr is the rate ratio for those receiving both supplements, and 1 otherwise"""
        class MockMalnutrition:
            receiving_macro = np.array([True, False, True, False, True])
            receiving_micro = np.array([True, True, False, False, True])
        rr = tbsim.supplementation_rr(None, MockMalnutrition(), np.arange(5), rate_ratio=0.5)
        assert np.array_equal(rr, [0.5, 1, 1, 1, 0.5])

    def test_lonnroth_bmi_rr_function(self):
        """lonnroth_bmi_rr is positive and at most the scale"""
        class MockMalnutrition:
            def weight(self, uids): return np.array([20.0, 25.0, 30.0, 18.0, 35.0])
            def height(self, uids): return np.array([100.0, 110.0, 105.0, 95.0, 120.0])
        rr = tbsim.lonnroth_bmi_rr(None, MockMalnutrition(), np.arange(5), scale=2.0, slope=3.0, bmi50=25.0)
        assert len(rr) == 5 and np.all(rr > 0) and np.all(rr <= 2.0)


class TestHIV:
    """Test HIV, and its effect on TB (tbsim.hiv_effects)"""

    @staticmethod
    def make(solver='ode', mult=2.0, drugs=None, n_agents=5000, **kwargs):
        tb = tbsim.TB(init=dict(infection=0.3), drugs=drugs, strains=dict(DS=0.9, RIF=0.1) if drugs else None)
        hiv = tbsim.HIV(incidence=0.01, init_prev=0.1, art_start=0.2, **kwargs)
        return ss.Sim(tb, diseases=hiv, connectors=tbsim.hiv_effects(tb, hiv, acute=mult, latent=mult, aids=mult), solver=solver,
                      n_agents=n_agents, total_pop=5000, start=2000, stop=2010, dt=ss.weeks(1), verbose=0, rand_seed=1)

    def test_hiv_speeds_progression(self):
        """With HIV, more latent infections progress to active TB, as an ODE and as agents"""
        for solver in ['ode', 'abm']:
            progressions = []
            for mult in [1, 5]:
                res = self.make(solver, mult).run().results
                progressions.append((res.tb.latent_asy + res.tb.latent_non).sum())
            assert progressions[1] > 1.3*progressions[0]

    def test_tb_hiv_jointly(self):
        """As compartments, TB (with resistance) and HIV are tracked jointly: they have the same people, and agree with agents"""
        ode = self.make('ode', drugs=['RIF']).run()
        abm = self.make('abm', drugs=['RIF'], n_agents=20e3).run() # At 5,000 agents, active TB varies too much between seeds
        res = ode.results
        n_tb = sum(res.tb[s.name] for s in ode.diseases.tb.graph.people_states)
        n_hiv = sum(res.hiv[s.name] for s in ode.diseases.hiv.graph.people_states)
        assert np.allclose(n_tb, n_hiv)
        assert np.isclose(res.hiv.infected[-1], abm.results.hiv.infected[-1], rtol=0.1)
        assert np.isclose(res.tb.active[-1], abm.results.tb.active[-1], rtol=0.25)
        assert res.hiv.infected.by('art').values[-1][1] > 0 # Some on ART

    def test_art_delivery(self):
        """ART started by a delivery, with no ART otherwise"""
        tb = tbsim.TB()
        hiv = tbsim.HIV(incidence=0.01, init_prev=0.1)
        arts = ss.Delivery(move=dict(art=True), coverage=0.5, at=[2005], where=hiv.acute | hiv.latent | hiv.aids)
        sim = ss.Sim(tb, diseases=hiv, connectors=tbsim.hiv_effects(tb, hiv), interventions=arts, start=2000, stop=2010, verbose=0).run()
        on_art = sim.results.hiv.infected.by('art').values[:, 1]
        assert on_art[5] == 0 and on_art[6] > 0 # Nobody on ART until the round in 2005, which is between the numbers at 2005 and 2006

    def test_art_slows_progression(self):
        """ART slows progression to AIDS"""
        aids = [self.make(art_progression=f).run().results.hiv.aids[-1] for f in [1.0, 0.1]]
        assert aids[1] < aids[0]


class TestMalnutritionIntegration:
    """Test integration between TB and Malnutrition"""
    
    def test_tb_with_malnutrition(self):
        """TB runs as agents with malnutrition affecting it"""
        tb = tbsim.TB(init=dict(infection=0.3))
        sim = ss.Sim(tb, diseases=tbsim.Malnutrition(), connectors=tbsim.nutrition_effects(tb, rr_activation=tbsim.lonnroth_bmi_rr),
                     solver='abm', n_agents=1000, start=2000, stop=2002, verbose=0).run()
        assert sim.results.tb.latent_asy.sum() > 0
    
if __name__ == "__main__":
    pytest.main([__file__])
