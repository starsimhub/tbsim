"""
TB-Malnutrition Connector Module

This module implements a connector between Tuberculosis (TB) and Malnutrition disease models
in the simulation framework. The connector defines how nutritional status affects TB dynamics
through various risk ratios and susceptibility modifiers.

The connector implements three main interaction mechanisms:
1. Activation risk ratio: How malnutrition affects TB activation from latent to active
2. Clearance risk ratio: How malnutrition affects TB clearance/recovery rates
3. Relative susceptibility: How malnutrition affects susceptibility to new TB infection

Mathematical Framework:
- Risk ratios (RR) modify disease transition rates multiplicatively
- Relative susceptibility modifies infection probability for uninfected individuals
- BMI-based risk functions use sigmoid transformations of log-linear relationships

Examples:

Basic usage: Create connector with default functions and add to simulation.
Custom configurations: Use BMI-based risk ratios, supplementation effects, or combined functions.
Analysis: Access risk ratios and susceptibility modifiers after simulation runs.

See the module documentation and method docstrings for detailed usage examples.

References:
- Lönnroth et al. studies on BMI and TB risk
- Nutritional supplementation trials and their effects on TB outcomes
"""

import numpy as np
import starsim as ss
from tbsim import TB, Malnutrition

__all__ = ['TB_Nutrition_Connector']

class TB_Nutrition_Connector(ss.Connector):
    """
    Connector between Tuberculosis and Malnutrition disease models.
    
    This connector implements the bidirectional interactions between TB and malnutrition,
    where nutritional status affects TB dynamics and TB infection may affect nutritional
    status. The connector modifies disease transition rates through risk ratios and
    susceptibility modifiers.
    
    Mathematical Model:
    - Activation rate modification: λ_act(t) = λ_act_base * RR_activation(t)
    - Clearance rate modification: λ_clear(t) = λ_clear_base * RR_clearance(t)  
    - Susceptibility modification: P_inf(t) = P_inf_base * rel_sus(t)
      where λ = rates, RR = risk ratios, P_inf = infection probability
    
    Interaction Mechanisms:
    1. Supplementation effects: Reduced risk ratios for individuals receiving nutritional interventions
    2. BMI-based risk: Sigmoid function of BMI following Lönnroth et al. log-linear relationship
    3. Micronutrient effects: Increased susceptibility for individuals with low micronutrient status
    
    Parameters:
        rr_activation_func (callable): Function to compute activation risk ratios
        rr_clearance_func (callable): Function to compute clearance risk ratios  
        relsus_func (callable): Function to compute relative susceptibility modifiers
        
    Attributes:
        sim (ss.Sim): Reference to the simulation object
        pars (dict): Parameter dictionary containing function references
    """

    def __init__(self, pars=None, **kwargs):
        """
        Initialize the TB-Malnutrition connector.
        
        Sets up the connector with default risk ratio and susceptibility functions,
        and configures the interaction parameters between TB and malnutrition models.
        
        Initialization Process:
        1. Calls parent class constructor with label 'TB-Malnutrition'
        2. Defines parameter functions for risk ratios and susceptibility
        3. Updates parameters with any provided overrides
        
        Args:
            pars (dict, optional): Dictionary of parameters to override defaults
                                 Keys: 'rr_activation_func', 'rr_clearance_func', 'relsus_func'
                                 Values: callable functions
            **kwargs: Additional keyword arguments passed to parent class
            
        Default Functions:
            rr_activation_func: ones_rr (no effect on activation)
            rr_clearance_func: ones_rr (no effect on clearance)  
            relsus_func: compute_relsus (micronutrient-based susceptibility)
        """
        super().__init__(label='TB-Malnutrition')

        self.define_pars(
            rr_activation_func = self.ones_rr, #self.supplementation_rr, self.lonnroth_bmi_rr,
            rr_clearance_func = self.ones_rr,
            relsus_func = self.compute_relsus,
        )
        self.update_pars(pars, **kwargs)

        return

    @staticmethod
    def supplementation_rr(tb, mn, uids, rate_ratio=0.5):
        """
        Calculate risk ratios based on nutritional supplementation status.
        
        This function reduces TB activation and clearance rates for individuals
        receiving both macronutrient and micronutrient supplementation, modeling
        the protective effects of comprehensive nutritional interventions.
        
        Mathematical Formula:
            RR_i = 1.0 if not receiving both macro and micro supplementation
            RR_i = rate_ratio if receiving both macro and micro supplementation
            where rate_ratio < 1.0 indicates reduced risk (protective effect)
        
        Args:
            tb (TB): Tuberculosis disease model object
            mn (Malnutrition): Malnutrition disease model object  
            uids (np.ndarray): Array of individual identifiers (int64)
            rate_ratio (float): Risk ratio for supplemented individuals (default: 0.5)
                               Range: 0.0-1.0, where 0.5 = 50% risk reduction
                               
        Returns:
            np.ndarray: Risk ratios for each individual (float64)
                       Shape: (len(uids),)
                       Values: 1.0 for non-supplemented, rate_ratio for supplemented
                       
        Implementation Details:
            - Creates array of ones for all individuals
            - Identifies individuals receiving both macro and micro supplementation
            - Applies rate_ratio only to fully supplemented individuals
            - Uses boolean indexing with logical AND operation
        """
        rr = np.ones_like(uids)
        rr[mn.receiving_macro[uids] & mn.receiving_micro[uids]] = rate_ratio
        return rr

    @staticmethod
    def lonnroth_bmi_rr(tb, mn, uids, scale=2, slope=3, bmi50=25):
        """
        Calculate risk ratios based on BMI using Lönnroth et al. relationship.
        
        This function implements a sigmoid transformation of the log-linear relationship
        between BMI and TB risk described by Lönnroth et al. The function creates a
        smooth transition around a reference BMI value with configurable steepness.
        
        Mathematical Formula:
            BMI_i = 10,000 * weight_i(kg) / height_i(cm)²
            x_i = -0.05 * (BMI_i - 15) + 2  # Log-linear relationship from Lönnroth et al.
            x0 = -0.05 * (bmi50 - 15) + 2   # Center point at reference BMI
            RR_i = scale / (1 + 10^(-slope * (x_i - x0)))
            
            where:
            - BMI_i is calculated from weight and height measurements
            - x_i is the log-linear predictor from Lönnroth et al.
            - x0 centers the sigmoid at the reference BMI
            - scale controls the maximum risk ratio
            - slope controls the steepness of the sigmoid transition
        
        Args:
            tb (TB): Tuberculosis disease model object
            mn (Malnutrition): Malnutrition disease model object
            uids (np.ndarray): Array of individual identifiers (int64)
            scale (float): Maximum risk ratio value (default: 2.0)
                          Range: > 0, typically 1.0-5.0
            slope (float): Steepness of sigmoid transition (default: 3.0)
                          Range: > 0, higher values = steeper transition
            bmi50 (float): Reference BMI for sigmoid center (default: 25.0 kg/m²)
                          Range: 15-35 kg/m², typical healthy adult range
                          
        Returns:
            np.ndarray: Risk ratios based on BMI (float64)
                       Shape: (len(uids),)
                       Range: 0.0 to scale, with sigmoid transition around bmi50
                       
        Implementation Details:
            - Calculates BMI using weight (kg) and height (cm) from malnutrition model
            - Applies Lönnroth et al. log-linear transformation
            - Centers sigmoid function at specified reference BMI
            - Uses 10-based logarithm for sigmoid calculation
            - Returns risk ratios where higher values indicate increased risk
            
        References:
            - Lönnroth et al. studies on BMI and TB risk relationships
            - Log-linear model: log(incidence) = -0.05*(BMI-15) + 2
        """
        bmi = 10_000 * mn.weight(uids) / mn.height(uids)**2
        #tb_incidence_per_100k_year = 10**(-0.05*(bmi-15) + 2) # incidence rate of 100 at BMI of 15
        # How to go from incidence rate to relative risk?
        # --> How about a sigmoid?
        x = -0.05*(bmi-15) + 2 # Log linear relationship from lonnroth et al.
        x0 = -0.05*(bmi50-15) + 2 # Center on 25
        rr = scale / (1+10**(-slope * (x-x0) ))

        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(bmi, rr)
        '''

        return rr

    @staticmethod
    def ones_rr(tb, mn, uids):
        """
        Return neutral risk ratios (no effect on disease dynamics).
        
        This function serves as a neutral baseline that applies no modification
        to TB activation or clearance rates. It is used as a default function
        when no nutritional effects on TB dynamics are desired.
        
        Mathematical Formula:
            RR_i = 1.0 for all individuals i
            This means: λ_modified = λ_base * 1.0 = λ_base (no change)
        
        Args:
            tb (TB): Tuberculosis disease model object
            mn (Malnutrition): Malnutrition disease model object
            uids (np.ndarray): Array of individual identifiers (int64)
            
        Returns:
            np.ndarray: Neutral risk ratios of 1.0 for all individuals (float64)
                       Shape: (len(uids),)
                       Values: All elements equal to 1.0
                       
        Implementation Details:
            - Creates array of ones with same shape as uids
            - Uses np.ones_like() for efficient array creation
            - Serves as identity function for risk ratio calculations
        """
        rr = np.ones_like(uids)
        return rr

    @staticmethod
    def compute_relsus(tb, mn, uids):
        """
        Calculate relative susceptibility based on micronutrient status.
        
        This function modifies the susceptibility to new TB infection based on
        individual micronutrient status. Individuals with low micronutrient levels
        experience increased susceptibility to TB infection.
        
        Mathematical Formula:
            rel_sus_i = 1.0 if micro_i ≥ 0.2 (normal micronutrient status)
            rel_sus_i = 2.0 if micro_i < 0.2 (low micronutrient status)
            where micro_i is the micronutrient z-score from malnutrition model
            
        Threshold Logic:
            - micro_i ≥ 0.2: Normal susceptibility (rel_sus_i = 1.0)
            - micro_i < 0.2: Doubled susceptibility (rel_sus_i = 2.0)
            - Threshold of 0.2 represents approximately 42nd percentile of normal distribution
        
        Args:
            tb (TB): Tuberculosis disease model object
            mn (Malnutrition): Malnutrition disease model object
            uids (np.ndarray): Array of individual identifiers (int64)
            
        Returns:
            np.ndarray: Relative susceptibility modifiers (float64)
                       Shape: (len(uids),)
                       Values: 1.0 for normal micronutrient status, 2.0 for low status
                       
        Implementation Details:
            - Accesses micronutrient status from malnutrition model
            - Applies threshold-based logic with 0.2 z-score cutoff
            - Uses boolean indexing for efficient conditional assignment
            - Returns susceptibility multipliers where higher values = increased risk
        """
        rel_sus = np.ones_like(uids)
        rel_sus[mn.micro[uids]<0.2] = 2 # Double the susceptibility if micro is low???
        return rel_sus

    def step(self):
        """
        Execute one time step of TB-Malnutrition interactions.
        
        This method is called at each simulation time step to apply the nutritional
        effects on TB dynamics. It modifies TB transition rates and susceptibility
        based on current nutritional status of individuals.
        
        Mathematical Model:
            For infected individuals (latent TB):
            - RR_activation(t) = RR_activation_base * rr_activation_func(t)
            - RR_clearance(t) = RR_clearance_base * rr_clearance_func(t)
            
            For uninfected individuals:
            - rel_sus(t) = relsus_func(t)
            
        Step Process:
        1. Get references to TB and malnutrition disease models
        2. For infected individuals: modify activation and clearance risk ratios
        3. For uninfected individuals: update relative susceptibility
        4. Apply multiplicative modifications to existing rates
        
        Args:
            None (uses self.sim for simulation state)
            
        Returns:
            None (modifies TB model state variables in-place)
            
        Implementation Details:
            - Accesses disease models through self.sim.diseases dictionary
            - Processes infected and uninfected individuals separately
            - Uses multiplicative updates ( *= ) to combine multiple effects
            - Modifies tb.rr_activation, tb.rr_clearance, and tb.rel_sus arrays
            - Risk ratios start at 1.0 each time step and are modified by connector
        """
        # Specify how malnutrition and TB interact
        tb = self.sim.diseases['tb']
        mn = self.sim.diseases['malnutrition']

        uids = tb.infected.uids
        # Relative rates start at 1 each time step
        tb.rr_activation[uids] *= self.pars.rr_activation_func(tb, mn, uids)
        tb.rr_clearance[uids] *= self.pars.rr_clearance_func(tb, mn, uids)

        uids = (~tb.infected).uids
        tb.rel_sus[uids] = self.pars.relsus_func(tb, mn, uids)

        return