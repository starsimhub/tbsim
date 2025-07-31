"""
Scientific Tests (SFT) Package

This package contains scientific tests for validating TB simulation interventions and methodologies.
"""

from .test_tpt_household_intervention import TestTPTHouseholdIntervention, run_scientific_test

__all__ = ['TestTPTHouseholdIntervention', 'run_scientific_test'] 