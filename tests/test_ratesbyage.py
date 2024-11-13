import pytest
import starsim as ss
import numpy as np
from tbsim.parametervalues import RatesByAge

@pytest.fixture
def rates_by_age():
    return RatesByAge(unit='days', dt=1)

def test_get_rates(rates_by_age):
    rates = rates_by_age.get_rates(10)
    assert rates is not None
    assert 'rate_LS_to_presym' in rates
    assert 'rate_LF_to_presym' in rates

def test_get_rate(rates_by_age):
    rate = rates_by_age.get_rate(10, 'rate_LS_to_presym')
    assert rate == ss.perday(2.0548e-06, parent_unit='days', parent_dt=1)

def test_apply_overrides():
    overrides = {
        '0,15': {
            'rate_LS_to_presym': ss.perday(1e-6, parent_unit='days', parent_dt=1)
        }
    }
    rates_by_age = RatesByAge(unit='days', dt=1, override=overrides)
    rate = rates_by_age.get_rate(10, 'rate_LS_to_presym')
    assert rate == ss.perday(1e-6, parent_unit='days', parent_dt=1)

def test_get_groups(rates_by_age):
    groups = rates_by_age.get_groups()
    assert groups == ['0,15', '15,25', '25,150']

def test_age_bins(rates_by_age):
    age_bins = rates_by_age.age_bins()
    expected_bins = np.array([0, 15, 25, np.inf])
    assert np.array_equal(age_bins, expected_bins)