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

# def test_of_age(dummy_uids):
#     """Test filtering by exact age."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5])
#     result = wrappers.Agents.of_age(people, age=5)
#     # Should return index 4 (age 5)
#     np.testing.assert_array_equal(result, [4])

# def test_under_5(dummy_uids):
#     """Test filtering for individuals under 5 years old."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5])
#     # All alive
#     result = wrappers.Agents.under_5(people)
#     # Should return indices 0,2,4 (ages 2,4,5)
#     np.testing.assert_array_equal(result, [0, 2, 4])

# def test_over_5(dummy_uids):
#     """Test filtering for individuals over 5 years old."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5])
#     result = wrappers.Agents.over_5(people)
#     # Should return indices 1,3 (ages 6,10)
#     np.testing.assert_array_equal(result, [1, 3])

# def test_get_by_age_max_only(dummy_uids):
#     """Test filtering by maximum age only."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5])
#     result = wrappers.Agents.get_by_age(people, max_age=5)
#     # Should return indices 0,2,4 (ages 2,4,5)
#     np.testing.assert_array_equal(result, [0, 2, 4])

# def test_get_by_age_min_only(dummy_uids):
#     """Test filtering by minimum age only."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5])
#     result = wrappers.Agents.get_by_age(people, min_age=5)
#     # Should return indices 1,3 (ages 6,10)
#     np.testing.assert_array_equal(result, [1, 3])

# def test_get_by_age_min_max(dummy_uids):
#     """Test filtering by both minimum and maximum age."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5])
#     result = wrappers.Agents.get_by_age(people, min_age=2, max_age=6)
#     # Should return indices 1,2,4 (ages 6,4,5) - min_age is exclusive, max_age is inclusive
#     np.testing.assert_array_equal(result, [1, 2, 4])

def test_get_by_age_invalid_params(dummy_uids):
    """Test that get_by_age raises ValueError with invalid parameters."""
    people = DummyPeople(age=[2, 6, 4, 10, 5])
    
    # Test with no parameters
    with pytest.raises(ValueError, match="At least one of max_age or min_age must be specified"):
        wrappers.Agents.get_by_age(people)
    
    # Test with max_age <= min_age
    with pytest.raises(ValueError, match="max_age must be greater than min_age"):
        wrappers.Agents.get_by_age(people, min_age=5, max_age=5)
    
    with pytest.raises(ValueError, match="max_age must be greater than min_age"):
        wrappers.Agents.get_by_age(people, min_age=10, max_age=5)

def test_get_alive(dummy_uids):
    """Test getting all alive individuals."""
    people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
    result = wrappers.Agents.get_alive(people)
    np.testing.assert_array_equal(result, [0, 2, 4])

# def test_get_alive_by_age_max_only(dummy_uids):
#     """Test filtering alive individuals by maximum age only."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
#     # Only indices 0,2,4 are alive, with ages 2,4,5
#     result = wrappers.Agents.get_alive_by_age(people, max_age=4)
#     # Should return indices 0,2 (ages 2,4)
#     np.testing.assert_array_equal(result, [0, 2])

# def test_get_alive_by_age_min_only(dummy_uids):
#     """Test filtering alive individuals by minimum age only."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
#     result = wrappers.Agents.get_alive_by_age(people, min_age=4)
#     # Should return index 4 (age 5)
#     np.testing.assert_array_equal(result, [4])

# def test_get_alive_by_age_min_max(dummy_uids):
#     """Test filtering alive individuals by both minimum and maximum age."""
#     people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
#     result = wrappers.Agents.get_alive_by_age(people, min_age=2, max_age=4)
#     # Should return index 2 (age 4)
#     np.testing.assert_array_equal(result, [2])

def test_get_alive_by_age_no_filters(dummy_uids):
    """Test getting all alive individuals without age filters."""
    people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
    result = wrappers.Agents.get_alive_by_age(people)
    # Should return all alive individuals
    np.testing.assert_array_equal(result, [0, 2, 4])

def test_get_alive_by_age_invalid_params(dummy_uids):
    """Test that get_alive_by_age raises ValueError with invalid parameters."""
    people = DummyPeople(age=[2, 6, 4, 10, 5], auids=[0, 2, 4])
    
    # Test with max_age <= min_age
    with pytest.raises(ValueError, match="max_age must be greater than min_age"):
        wrappers.Agents.get_alive_by_age(people, min_age=5, max_age=5)
    
    with pytest.raises(ValueError, match="max_age must be greater than min_age"):
        wrappers.Agents.get_alive_by_age(people, min_age=10, max_age=5)

# def test_where_valid_condition(dummy_uids):
#     """Test the where method with a valid condition."""
#     # Mock ss.uids() to return a fixed length array
#     with mock.patch('tbsim.wrappers.ss.uids', side_effect=lambda x=None: np.arange(5) if x is None else np.array(x)):
#         condition = np.array([True, False, True, False, True])
#         result = wrappers.Agents.where(condition)
#         np.testing.assert_array_equal(result, [0, 2, 4])

def test_where_invalid_condition_type(dummy_uids):
    """Test that where raises ValueError with invalid condition type."""
    with pytest.raises(ValueError, match="Condition must be a numpy array or list"):
        wrappers.Agents.where("not an array")

def test_where_condition_length_mismatch(dummy_uids):
    """Test that where raises ValueError when condition length doesn't match agent count."""
    with mock.patch('tbsim.wrappers.ss.uids', side_effect=lambda x=None: np.arange(5) if x is None else np.array(x)):
        condition = np.array([True, False, True])  # Length 3, but 5 agents
        with pytest.raises(ValueError, match="Condition length must match the number of agents"):
            wrappers.Agents.where(condition)

# def test_edge_cases(dummy_uids):
#     """Test edge cases for age filtering."""
#     people = DummyPeople(age=[0, 1, 5, 10, 15])
    
#     # Test exact boundary conditions
#     result = wrappers.Agents.get_by_age(people, min_age=0, max_age=5)
#     # Should return indices 1,2 (ages 1,5) - min_age is exclusive
#     np.testing.assert_array_equal(result, [1, 2])
    
#     # Test with only one individual in range
#     result = wrappers.Agents.get_by_age(people, min_age=4, max_age=6)
#     # Should return index 2 (age 5)
#     np.testing.assert_array_equal(result, [2])
    
#     # Test with no individuals in range
#     result = wrappers.Agents.get_by_age(people, min_age=20, max_age=25)
#     # Should return empty array
#     np.testing.assert_array_equal(result, [])