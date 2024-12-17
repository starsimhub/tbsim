# tests/test_parametervalues.py

import pytest
import numpy as np
import starsim as ss
from tbsim import RateVec

@pytest.fixture
def rate_vec():
    cutoffs = [10, 20, 30]
    values = [0.1, 0.2, 0.3, 0.4]
    return RateVec(cutoffs, values, interpolation="stair", off_value=0.05)

def test_initialization(rate_vec):
    assert np.array_equal(rate_vec.cutoffs, np.array([10, 20, 30]))
    assert np.array_equal(rate_vec.values, np.array([ss.perday(0.1), ss.perday(0.2), ss.perday(0.3), ss.perday(0.4)]))
    assert rate_vec.interpolation == "stair"
    assert rate_vec.off_value == ss.perday(0.05)

def test_digitize_stair(rate_vec):
    inputs = np.array([5, 15, 25, 35])
    expected_rates = np.array([ss.perday(0.1), ss.perday(0.2), ss.perday(0.3), ss.perday(0.4)])
    rates = rate_vec.digitize(inputs)
    np.testing.assert_array_equal(rates, expected_rates)

def test_turn_age_off(rate_vec):
    rate_vec.turn_age_off(new_off_value=0.1)
    assert np.array_equal(rate_vec.values, np.array([0.1, 0.1]))
    assert rate_vec.cutoffs == [0]

    rate_vec.turn_age_off()
    assert np.array_equal(rate_vec.values, np.array([ss.perday(0.05), ss.perday(0.05)]))
    assert rate_vec.cutoffs == [0]

def test_turn_age_off_invalid_value(rate_vec):
    with pytest.raises(ValueError):
        rate_vec.turn_age_off(new_off_value="invalid")

def test_summary(rate_vec):
    summary = rate_vec.__summary__()
    assert summary == f"RateVec(cutoffs={rate_vec.cutoffs}, values={rate_vec.values}, interpolation={rate_vec.interpolation})"

if __name__ == "__main__":
    pytest.main()