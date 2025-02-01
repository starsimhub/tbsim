import pytest
import numpy as np
from tbsim.parametervalues import RatesByAge
import starsim as ss

def test_initial_rates():
    unit = 'days'
    dt = 1
    rates_by_age = RatesByAge(unit, dt)
    
    assert rates_by_age.unit == unit
    assert rates_by_age.dt == dt
    assert rates_by_age.override is None
    assert rates_by_age.use_globals is False

    expected_keys = [
        'rate_LS_to_presym', 'rate_LF_to_presym', 'rate_presym_to_active',
        'rate_active_to_clear', 'rate_smpos_to_dead', 'rate_smneg_to_dead',
        'rate_exptb_to_dead', 'rate_treatment_to_clear'
    ]
    assert set(rates_by_age.rates_dict.keys()) == set(expected_keys)

def test_override_rates():
    unit = 'days'
    dt = 1
    override = {
        'rate_LS_to_presym': {0: 1e-5, 15: 2e-5},
        'rate_LF_to_presym': 1e-3
    }
    rates_by_age = RatesByAge(unit, dt, override=override)
    
    assert rates_by_age.rates_dict['rate_LS_to_presym'][0] == ss.perday(1e-5, unit, dt)
    assert rates_by_age.rates_dict['rate_LS_to_presym'][15] == ss.perday(2e-5, unit, dt)
    assert rates_by_age.rates_dict['rate_LF_to_presym'][np.inf] == ss.perday(1e-3, unit, dt)

def test_use_globals():
    unit = 'days'
    dt = 1
    rates_by_age = RatesByAge(unit, dt, use_globals=True)
    
    for rate_name in rates_by_age.rates_dict:
        assert list(rates_by_age.rates_dict[rate_name].keys()) == [np.inf]

def test_generate_age_cutoffs():
    unit = 'days'
    dt = 1
    rates_by_age = RatesByAge(unit, dt)
    
    age_cutoffs = rates_by_age.generate_age_cutoffs()
    for rate_name, cutoffs in age_cutoffs.items():
        assert np.array_equal(cutoffs, np.array(sorted(rates_by_age.rates_dict[rate_name].keys())))

def test_arr_method():
    unit = 'days'
    dt = 1
    rates_by_age = RatesByAge(unit, dt)
    
    for rate_name in rates_by_age.rates_dict:
        arr = rates_by_age.arr(rate_name)
        expected_arr = np.array(list(rates_by_age.rates_dict[rate_name].values()))
        assert np.array_equal(arr, expected_arr)