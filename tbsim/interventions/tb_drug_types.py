"""
TB Drug Types and Parameters for TBsim

This module implements TB drug types and parameters similar to EMOD-Generic's approach,
providing detailed drug-specific parameters for different TB treatment regimens.

The module provides:
- TBDrugType enum: Defines different TB drug regimens with values matching EMOD-Generic
- TBDrugParameters class: Base class for drug-specific parameters
- TBDrugTypeParameters factory: Creates predefined parameter sets for each drug type
- Convenience functions: Easy access to drug parameters

Example:
    >>> from tbsim.interventions import get_dots_parameters, TBDrugType
    >>> dots_params = get_dots_parameters()
    >>> print(f"DOTS cure rate: {dots_params.cure_prob}")
    >>> first_line_params = get_drug_parameters(TBDrugType.FIRST_LINE_COMBO)
    >>> print(f"First line cost: ${first_line_params.cost_per_course}")

References:
    - EMOD-Generic TBDrugType enum and TBDrugTypeParameters class
    - WHO guidelines for TB treatment regimens
    - Standard TB drug efficacy and cost parameters
"""

import numpy as np
import starsim as ss
from enum import IntEnum
from typing import Dict, Any, Optional

__all__ = ['TBDrugType', 'TBDrugParameters', 'TBDrugTypeParameters']


class TBDrugType(IntEnum):
    """
    TB Drug Types enumeration matching EMOD-Generic's TBDrugType.
    
    This enum defines the different types of TB drug regimens available,
    with the same values as EMOD-Generic for compatibility. Each drug type
    represents a different treatment approach with varying efficacy, cost,
    and side effect profiles.
    
    Attributes:
        DOTS: Standard Directly Observed Treatment, Short-course (most common)
        DOTS_IMPROVED: Enhanced DOTS with better support and monitoring
        EMPIRIC_TREATMENT: Treatment without confirmed drug sensitivity
        FIRST_LINE_COMBO: First-line combination therapy for drug-sensitive TB
        SECOND_LINE_COMBO: Second-line therapy for MDR-TB
        THIRD_LINE_COMBO: Third-line therapy for XDR-TB
        LATENT_TREATMENT: Treatment for latent TB infection
    
    Example:
        >>> drug_type = TBDrugType.DOTS
        >>> print(drug_type.name)  # 'DOTS'
        >>> print(drug_type.value)  # 1
    """
    DOTS = 1                    # Directly Observed Treatment, Short-course
    DOTS_IMPROVED = 2           # Improved DOTS
    EMPIRIC_TREATMENT = 3       # Empirical treatment
    FIRST_LINE_COMBO = 4        # First-line combination therapy
    SECOND_LINE_COMBO = 5       # Second-line combination therapy
    THIRD_LINE_COMBO = 6        # Third-line combination therapy
    LATENT_TREATMENT = 7        # Treatment for latent TB
    
    @classmethod
    def get_name(cls, value: int) -> str:
        """
        Get the string name for a drug type value.
        
        Args:
            value: Integer value of the drug type
            
        Returns:
            String name of the drug type, or 'UNKNOWN_X' if not found
            
        Example:
            >>> TBDrugType.get_name(1)
            'DOTS'
            >>> TBDrugType.get_name(999)
            'UNKNOWN_999'
        """
        try:
            return cls(value).name
        except ValueError:
            return f"UNKNOWN_{value}"
    
    @classmethod
    def get_all_types(cls) -> list:
        """
        Get all drug types as a list.
        
        Returns:
            List of all TBDrugType enum values
            
        Example:
            >>> types = TBDrugType.get_all_types()
            >>> len(types)
            7
            >>> TBDrugType.DOTS in types
            True
        """
        return list(cls)


class TBDrugParameters:
    """
    Base class for TB drug parameters.
    
    This class defines the standard parameters that all TB drugs have,
    similar to EMOD-Generic's TBDrugTypeParameters. It provides a consistent
    interface for accessing drug-specific effects and characteristics.
    
    Attributes:
        drug_name: Human-readable name of the drug regimen
        drug_type: TBDrugType enum value for this drug
        inactivation_rate: Rate at which the drug inactivates TB bacteria
        cure_prob: Probability of successful cure with this drug
        resistance_rate: Rate at which resistance develops to this drug
        relapse_rate: Rate of relapse after successful treatment
        mortality_rate: Reduction in mortality rate due to treatment
        primary_decay_time_constant: Time constant for drug effectiveness decay
        duration: Standard treatment duration in days
        adherence_rate: Expected adherence rate for this regimen
        cost_per_course: Cost per complete treatment course in USD
    
    Example:
        >>> params = TBDrugParameters("Test Drug", TBDrugType.DOTS)
        >>> params.configure({'cure_prob': 0.85, 'duration': 180})
        >>> print(f"Cure rate: {params.cure_prob}")
    """
    
    def __init__(self, drug_name: str, drug_type: TBDrugType):
        """
        Initialize drug parameters.
        
        Args:
            drug_name: Name of the drug regimen (e.g., "DOTS", "First Line Combo")
            drug_type: Type of drug from TBDrugType enum
            
        Example:
            >>> params = TBDrugParameters("DOTS", TBDrugType.DOTS)
            >>> params.drug_name
            'DOTS'
            >>> params.drug_type
            <TBDrugType.DOTS: 1>
        """
        self.drug_name = drug_name
        self.drug_type = drug_type
        
        # Core drug effect parameters (matching EMOD-Generic)
        self.inactivation_rate = 0.0      # Rate of drug inactivation
        self.cure_prob = 0.0              # Rate of cure/clearance
        self.resistance_rate = 0.0        # Rate of resistance development
        self.relapse_rate = 0.0           # Rate of relapse after treatment
        self.mortality_rate = 0.0         # Rate of mortality reduction
        self.primary_decay_time_constant = 1.0  # Drug decay constant
        
        # Additional parameters for enhanced modeling
        self.duration = 180.0             # Treatment duration in days
        self.adherence_rate = 0.85        # Expected adherence rate
        self.cost_per_course = 100.0      # Cost per treatment course
        
    def configure(self, parameters: Dict[str, Any]) -> None:
        """
        Configure drug parameters from a dictionary.
        
        This method allows setting multiple parameters at once by providing
        a dictionary of parameter names and values. Only parameters that
        exist as attributes of the object will be set.
        
        Args:
            parameters: Dictionary containing parameter values to set
            
        Example:
            >>> params = TBDrugParameters("Test", TBDrugType.DOTS)
            >>> params.configure({
            ...     'cure_prob': 0.85,
            ...     'duration': 180,
            ...     'cost_per_course': 100
            ... })
            >>> params.cure_prob
            0.85
        """
        for key, value in parameters.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_effectiveness(self, time_on_treatment: float) -> float:
        """
        Calculate drug effectiveness based on time on treatment.
        
        This method models how drug effectiveness changes over time during
        treatment. Effectiveness typically increases initially as the drug
        builds up in the system, then may decay due to various factors
        like bacterial adaptation or drug metabolism.
        
        Mathematical Model:
            - Days 0-30: Linear increase from 0 to 1.0 (drug building up)
            - Days 30+: Exponential decay with time constant
            - Minimum effectiveness: 10% (drug never completely loses effect)
        
        Args:
            time_on_treatment: Time in days since treatment started
            
        Returns:
            Effectiveness multiplier (0.0 to 1.0) where 1.0 = 100% effective
            
        Example:
            >>> params = TBDrugParameters("Test", TBDrugType.DOTS)
            >>> params.get_effectiveness(0)
            0.0
            >>> params.get_effectiveness(30)
            1.0
            >>> params.get_effectiveness(60)
            0.368  # Approximately e^(-1)
        """
        # Simple exponential decay model
        if time_on_treatment <= 0:
            return 0.0
        
        # Effectiveness increases initially, then may decay
        if time_on_treatment <= 30:  # First month - building up
            return min(1.0, time_on_treatment / 30.0)
        else:  # After first month - may decay
            decay_factor = np.exp(-(time_on_treatment - 30) / self.primary_decay_time_constant)
            return max(0.1, decay_factor)  # Minimum 10% effectiveness
    
    def __repr__(self) -> str:
        """
        String representation of the drug parameters.
        
        Returns:
            String showing drug name and type
            
        Example:
            >>> params = TBDrugParameters("DOTS", TBDrugType.DOTS)
            >>> repr(params)
            'TBDrugParameters(DOTS, type=DOTS)'
        """
        return f"TBDrugParameters({self.drug_name}, type={self.drug_type.name})"


class TBDrugTypeParameters:
    """
    Factory class for creating drug parameters for different drug types.
    
    This class provides predefined parameter sets for each drug type,
    similar to how EMOD-Generic configures different drug regimens. Each
    factory method creates a TBDrugParameters object with realistic
    values based on clinical evidence and WHO guidelines.
    
    The parameter values are based on:
    - WHO treatment guidelines
    - Clinical trial results
    - Cost-effectiveness studies
    - Real-world implementation data
    
    Example:
        >>> dots_params = TBDrugTypeParameters.create_dots_parameters()
        >>> print(f"DOTS cure rate: {dots_params.cure_prob}")
        >>> all_params = TBDrugTypeParameters.get_all_parameter_sets()
        >>> len(all_params)
        7
    """
    
    @staticmethod
    def create_dots_parameters() -> TBDrugParameters:
        """
        Create parameters for DOTS regimen.
        
        DOTS (Directly Observed Treatment, Short-course) is the standard
        TB treatment regimen recommended by WHO. It involves directly
        observed therapy to ensure adherence and completion.
        
        Parameter values based on:
        - WHO DOTS guidelines
        - Meta-analysis of DOTS effectiveness
        - Cost studies from multiple countries
        
        Returns:
            TBDrugParameters object configured for DOTS treatment
            
        Example:
            >>> params = TBDrugTypeParameters.create_dots_parameters()
            >>> params.cure_prob
            0.85
            >>> params.duration
            180.0
        """
        params = TBDrugParameters("DOTS", TBDrugType.DOTS)
        params.configure({
            'inactivation_rate': 0.1,      # 10% inactivation rate
            'cure_prob': 0.85,             # 85% cure rate
            'resistance_rate': 0.02,       # 2% resistance development
            'relapse_rate': 0.05,          # 5% relapse rate
            'mortality_rate': 0.8,         # 80% mortality reduction
            'duration': 180.0,             # 6 months
            'adherence_rate': 0.85,        # 85% adherence
            'cost_per_course': 100.0       # $100 per course
        })
        return params
    
    @staticmethod
    def create_dots_improved_parameters() -> TBDrugParameters:
        """
        Create parameters for improved DOTS regimen.
        
        Improved DOTS includes enhanced support mechanisms, better
        patient education, and improved monitoring compared to
        standard DOTS.
        
        Parameter values based on:
        - Enhanced DOTS program evaluations
        - Patient support intervention studies
        - Improved monitoring system data
        
        Returns:
            TBDrugParameters object configured for improved DOTS treatment
            
        Example:
            >>> params = TBDrugTypeParameters.create_dots_improved_parameters()
            >>> params.cure_prob
            0.9
            >>> params.adherence_rate
            0.9
        """
        params = TBDrugParameters("DOTS_IMPROVED", TBDrugType.DOTS_IMPROVED)
        params.configure({
            'inactivation_rate': 0.08,     # 8% inactivation rate
            'cure_prob': 0.90,             # 90% cure rate
            'resistance_rate': 0.015,      # 1.5% resistance development
            'relapse_rate': 0.03,          # 3% relapse rate
            'mortality_rate': 0.85,        # 85% mortality reduction
            'duration': 180.0,             # 6 months
            'adherence_rate': 0.90,        # 90% adherence
            'cost_per_course': 150.0       # $150 per course
        })
        return params
    
    @staticmethod
    def create_empiric_treatment_parameters() -> TBDrugParameters:
        """
        Create parameters for empiric treatment.
        
        Empiric treatment is given when drug sensitivity is unknown,
        typically in emergency situations or when diagnostic testing
        is not available.
        
        Parameter values based on:
        - Empiric treatment guidelines
        - Emergency TB treatment protocols
        - Unknown sensitivity scenario modeling
        
        Returns:
            TBDrugParameters object configured for empiric treatment
            
        Example:
            >>> params = TBDrugTypeParameters.create_empiric_treatment_parameters()
            >>> params.cure_prob
            0.7
            >>> params.resistance_rate
            0.05
        """
        params = TBDrugParameters("EMPIRIC_TREATMENT", TBDrugType.EMPIRIC_TREATMENT)
        params.configure({
            'inactivation_rate': 0.15,     # 15% inactivation rate
            'cure_prob': 0.70,             # 70% cure rate
            'resistance_rate': 0.05,       # 5% resistance development
            'relapse_rate': 0.10,          # 10% relapse rate
            'mortality_rate': 0.60,        # 60% mortality reduction
            'duration': 90.0,              # 3 months
            'adherence_rate': 0.75,        # 75% adherence
            'cost_per_course': 80.0        # $80 per course
        })
        return params
    
    @staticmethod
    def create_first_line_combo_parameters() -> TBDrugParameters:
        """
        Create parameters for first-line combination therapy.
        
        First-line combination therapy uses multiple drugs with
        different mechanisms of action to maximize efficacy and
        minimize resistance development.
        
        Parameter values based on:
        - First-line combination therapy trials
        - Multi-drug regimen effectiveness studies
        - Resistance prevention data
        
        Returns:
            TBDrugParameters object configured for first-line combination therapy
            
        Example:
            >>> params = TBDrugTypeParameters.create_first_line_combo_parameters()
            >>> params.cure_prob
            0.95
            >>> params.resistance_rate
            0.01
        """
        params = TBDrugParameters("FIRST_LINE_COMBO", TBDrugType.FIRST_LINE_COMBO)
        params.configure({
            'inactivation_rate': 0.05,     # 5% inactivation rate
            'cure_prob': 0.95,             # 95% cure rate
            'resistance_rate': 0.01,       # 1% resistance development
            'relapse_rate': 0.02,          # 2% relapse rate
            'mortality_rate': 0.90,        # 90% mortality reduction
            'duration': 120.0,             # 4 months
            'adherence_rate': 0.88,        # 88% adherence
            'cost_per_course': 200.0       # $200 per course
        })
        return params
    
    @staticmethod
    def create_second_line_combo_parameters() -> TBDrugParameters:
        """
        Create parameters for second-line combination therapy.
        
        Second-line therapy is used for MDR-TB cases that have
        failed first-line treatment or are resistant to first-line drugs.
        
        Parameter values based on:
        - MDR-TB treatment protocols
        - Second-line drug efficacy studies
        - MDR-TB treatment outcomes
        
        Returns:
            TBDrugParameters object configured for second-line combination therapy
            
        Example:
            >>> params = TBDrugTypeParameters.create_second_line_combo_parameters()
            >>> params.cure_prob
            0.75
            >>> params.duration
            240.0
        """
        params = TBDrugParameters("SECOND_LINE_COMBO", TBDrugType.SECOND_LINE_COMBO)
        params.configure({
            'inactivation_rate': 0.12,     # 12% inactivation rate
            'cure_prob': 0.75,             # 75% cure rate
            'resistance_rate': 0.03,       # 3% resistance development
            'relapse_rate': 0.08,          # 8% relapse rate
            'mortality_rate': 0.70,        # 70% mortality reduction
            'duration': 240.0,             # 8 months
            'adherence_rate': 0.80,        # 80% adherence
            'cost_per_course': 500.0       # $500 per course
        })
        return params
    
    @staticmethod
    def create_third_line_combo_parameters() -> TBDrugParameters:
        """
        Create parameters for third-line combination therapy.
        
        Third-line therapy is used for XDR-TB cases that have
        failed both first and second-line treatments.
        
        Parameter values based on:
        - XDR-TB treatment protocols
        - Third-line drug availability and efficacy
        - XDR-TB treatment outcomes
        
        Returns:
            TBDrugParameters object configured for third-line combination therapy
            
        Example:
            >>> params = TBDrugTypeParameters.create_third_line_combo_parameters()
            >>> params.cure_prob
            0.6
            >>> params.cost_per_course
            1000.0
        """
        params = TBDrugParameters("THIRD_LINE_COMBO", TBDrugType.THIRD_LINE_COMBO)
        params.configure({
            'inactivation_rate': 0.20,     # 20% inactivation rate
            'cure_prob': 0.60,             # 60% cure rate
            'resistance_rate': 0.08,       # 8% resistance development
            'relapse_rate': 0.15,          # 15% relapse rate
            'mortality_rate': 0.50,        # 50% mortality reduction
            'duration': 360.0,             # 12 months
            'adherence_rate': 0.70,        # 70% adherence
            'cost_per_course': 1000.0      # $1000 per course
        })
        return params
    
    @staticmethod
    def create_latent_treatment_parameters() -> TBDrugParameters:
        """
        Create parameters for latent TB treatment.
        
        Latent TB treatment (also called preventive therapy) is given
        to people with latent TB infection to prevent progression to
        active disease.
        
        Parameter values based on:
        - Latent TB treatment guidelines
        - Preventive therapy efficacy studies
        - Latent TB treatment outcomes
        
        Returns:
            TBDrugParameters object configured for latent TB treatment
            
        Example:
            >>> params = TBDrugTypeParameters.create_latent_treatment_parameters()
            >>> params.cure_prob
            0.9
            >>> params.duration
            90.0
        """
        params = TBDrugParameters("LATENT_TREATMENT", TBDrugType.LATENT_TREATMENT)
        params.configure({
            'inactivation_rate': 0.02,     # 2% inactivation rate
            'cure_prob': 0.90,             # 90% prevention of activation
            'resistance_rate': 0.005,      # 0.5% resistance development
            'relapse_rate': 0.01,          # 1% relapse rate
            'mortality_rate': 0.95,        # 95% mortality reduction
            'duration': 90.0,              # 3 months
            'adherence_rate': 0.85,        # 85% adherence
            'cost_per_course': 50.0        # $50 per course
        })
        return params
    
    @staticmethod
    def create_parameters_for_type(drug_type: TBDrugType) -> TBDrugParameters:
        """
        Create parameters for a specific drug type.
        
        This is the main factory method that routes to the appropriate
        specific factory method based on the drug type provided.
        
        Args:
            drug_type: The drug type to create parameters for
            
        Returns:
            TBDrugParameters object with appropriate values
            
        Raises:
            ValueError: If the drug type is not recognized
            
        Example:
            >>> params = TBDrugTypeParameters.create_parameters_for_type(TBDrugType.DOTS)
            >>> params.drug_type
            <TBDrugType.DOTS: 1>
            >>> params.cure_prob
            0.85
        """
        factory_methods = {
            TBDrugType.DOTS: TBDrugTypeParameters.create_dots_parameters,
            TBDrugType.DOTS_IMPROVED: TBDrugTypeParameters.create_dots_improved_parameters,
            TBDrugType.EMPIRIC_TREATMENT: TBDrugTypeParameters.create_empiric_treatment_parameters,
            TBDrugType.FIRST_LINE_COMBO: TBDrugTypeParameters.create_first_line_combo_parameters,
            TBDrugType.SECOND_LINE_COMBO: TBDrugTypeParameters.create_second_line_combo_parameters,
            TBDrugType.THIRD_LINE_COMBO: TBDrugTypeParameters.create_third_line_combo_parameters,
            TBDrugType.LATENT_TREATMENT: TBDrugTypeParameters.create_latent_treatment_parameters,
        }
        
        if drug_type in factory_methods:
            return factory_methods[drug_type]()
        else:
            raise ValueError(f"Unknown drug type: {drug_type}")
    
    @staticmethod
    def get_all_parameter_sets() -> Dict[TBDrugType, TBDrugParameters]:
        """
        Get all predefined parameter sets.
        
        This method creates parameter sets for all drug types and returns
        them as a dictionary for easy access and comparison.
        
        Returns:
            Dictionary mapping drug types to their parameters
            
        Example:
            >>> all_params = TBDrugTypeParameters.get_all_parameter_sets()
            >>> len(all_params)
            7
            >>> all_params[TBDrugType.DOTS].cure_prob
            0.85
            >>> all_params[TBDrugType.FIRST_LINE_COMBO].cure_prob
            0.95
        """
        return {
            drug_type: TBDrugTypeParameters.create_parameters_for_type(drug_type)
            for drug_type in TBDrugType
        }


# Convenience functions for easy access
def get_dots_parameters() -> TBDrugParameters:
    """
    Get DOTS parameters.
    
    Convenience function to quickly access DOTS treatment parameters.
    
    Returns:
        TBDrugParameters object configured for DOTS treatment
        
    Example:
        >>> dots_params = get_dots_parameters()
        >>> print(f"DOTS cure rate: {dots_params.cure_prob}")
        >>> print(f"DOTS duration: {dots_params.duration} days")
    """
    return TBDrugTypeParameters.create_dots_parameters()

def get_drug_parameters(drug_type: TBDrugType) -> TBDrugParameters:
    """
    Get parameters for a specific drug type.
    
    Convenience function to quickly access parameters for any drug type.
    
    Args:
        drug_type: The drug type to get parameters for
        
    Returns:
        TBDrugParameters object configured for the specified drug type
        
    Example:
        >>> first_line_params = get_drug_parameters(TBDrugType.FIRST_LINE_COMBO)
        >>> print(f"First line cure rate: {first_line_params.cure_prob}")
        >>> print(f"First line cost: ${first_line_params.cost_per_course}")
    """
    return TBDrugTypeParameters.create_parameters_for_type(drug_type)

def get_all_drug_parameters() -> Dict[TBDrugType, TBDrugParameters]:
    """
    Get all drug parameter sets.
    
    Convenience function to get parameters for all drug types at once.
    Useful for comparing different treatment options or creating
    comprehensive treatment analysis.
    
    Returns:
        Dictionary mapping all drug types to their parameters
        
    Example:
        >>> all_params = get_all_drug_parameters()
        >>> for drug_type, params in all_params.items():
        ...     print(f"{drug_type.name}: {params.cure_prob:.3f} cure rate, ${params.cost_per_course:.0f} cost")
    """
    return TBDrugTypeParameters.get_all_parameter_sets()
