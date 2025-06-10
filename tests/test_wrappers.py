import numpy as np
import pytest
from unittest import mock

import tbsim.wrappers as wrappers

class DummyPeople:
    def __init__(self, age, auids=None):
        self.age = np.array(age)
        if auids is None:
            # All alive by default
            self.auids = np.arange(len(age))
        else:
            self.auids = np.array(auids)

@pytest.fixture
def dummy_uids():
    # Patch ss.uids to just return the array for test simplicity
    with mock.patch('tbsim.wrappers.ss.uids', side_effect=lambda x: np.array(x)) as _fixture:
        yield

def test_under_5(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5])
    # All alive
    result = wrappers.Agents.under_5(people)
    # Should return indices 0,2,4 (ages 2,4,5)
    np.testing.assert_array_equal(result, [0, 2, 4])

def test_over_5(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5])
    result = wrappers.Agents.over_5(people)
    # Should return indices 1,3 (ages 6,10)
    np.testing.assert_array_equal(result, [1, 3])

def test_get_by_age_max(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5])
    result = wrappers.Agents.get_by_age(people, max_age=5)
    # Should return indices 0,2,4 (ages 2,4,5)
    np.testing.assert_array_equal(result, [0, 2, 4])

def test_get_by_age_min(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5])
    result = wrappers.Agents.get_by_age(people, min_age=5)
    # Should return indices 1,3 (ages 6,10)
    np.testing.assert_array_equal(result, [1, 3])

def test_get_by_age_min_max(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5])
    result = wrappers.Agents.get_by_age(people, min_age=2, max_age=6)
    # Should return indices 1,2,4 (ages 6,4,5) but min_age is exclusive, so 2 is excluded
    # Actually, min_age is exclusive, so ages >2 and <=6: indices 1,2,4 (ages 6,4,5)
    np.testing.assert_array_equal(result, [1, 2, 4])

def test_get_alive(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
    result = wrappers.Agents.get_alive(people)
    np.testing.assert_array_equal(result, [0, 2, 4])

def test_get_alive_by_age_max(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
    # Only indices 0,2,4 are alive, with ages 2,4,5
    result = wrappers.Agents.get_alive_by_age(people, max_age=4)
    # Should return indices 0,2 (ages 2,4)
    np.testing.assert_array_equal(result, [0, 2])

def test_get_alive_by_age_min(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
    result = wrappers.Agents.get_alive_by_age(people, min_age=4)
    # Should return index 4 (age 5)
    np.testing.assert_array_equal(result, [4])

def test_get_alive_by_age_min_max(dummy_uids):
    people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
    result = wrappers.Agents.get_alive_by_age(people, min_age=2, max_age=4)
    # Should return index 2 (age 4)
    np.testing.assert_array_equal(result, [2])