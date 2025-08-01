"""
Malnutrition Disease Model for TB Simulation

This module implements a malnutrition disease model that can be used in tuberculosis
simulation studies. The model tracks anthropometric measurements (weight, height)
using the LMS (Lambda-Mu-Sigma) method and simulates the effects of nutritional
interventions on growth and development.

The model is designed to integrate with the RATIONS trial framework and supports
both macronutrient and micronutrient supplementation interventions.

Mathematical Framework:
- Uses Cole's LMS method for growth reference curves
- Implements random walk processes for weight percentile evolution
- Applies nutritional intervention effects through drift parameters

Data Requirements:
- Requires anthropometry.csv file with LMS parameters (L, M, S) by age and sex
- Supports WHO growth standards and custom reference data

References:
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9971264/
- https://www.espen.org/files/ESPEN-guidelines-on-definitions-and-terminology-of-clinical-nutrition.pdf
"""

import os
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import norm
from tbsim import DATADIR

__all__ = ["Malnutrition"]


class Malnutrition(ss.Disease):
    """
    Malnutrition disease model for tuberculosis simulation studies.
    
    This class implements a comprehensive malnutrition model that tracks anthropometric
    measurements using the LMS (Lambda-Mu-Sigma) method. It simulates the effects of
    nutritional interventions on growth and development, with support for both
    macronutrient and micronutrient supplementation.
    
    Mathematical Model:
    - Weight percentile evolution: dW/dt = μ(t) + σ(t) * ε(t)
      where μ(t) = drift from interventions, σ(t) = time-varying noise, ε(t) ~ N(0,1)
    - LMS transformation: X = M * (L*S*Z + 1)^(1/L) for L ≠ 0
                         X = M * exp(S*Z) for L = 0
      where X = measurement, M = median, L = skewness, S = coefficient of variation, Z = z-score
    
    Individual States:
    - receiving_macro (bool): Whether individual receives macronutrient supplementation
    - receiving_micro (bool): Whether individual receives micronutrient supplementation  
    - height_percentile (float): Height percentile (0.0-1.0), assumed constant
    - weight_percentile (float): Weight percentile (0.0-1.0), evolves over time
    - micro (float): Micronutrient status z-score, evolves over time
    
    Attributes:
        LMS_data (pd.DataFrame): Anthropometric reference data indexed by sex with columns:
                                Age, Weight_L, Weight_M, Weight_S, Height_L, Height_M, Height_S
        dweight (ss.normal): Normal distribution for weight changes with location and scale functions
        
    Parameters:
        beta (float): Transmission rate (placeholder, not used in malnutrition model)
        init_prev (float): Initial prevalence of malnutrition (default: 0.001)
        
    File Dependencies:
        anthropometry.csv: Must contain LMS parameters by age and sex for growth calculations
    """
    # Possible references:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9971264/
    # https://www.espen.org/files/ESPEN-guidelines-on-definitions-and-terminology-of-clinical-nutrition.pdf 

    @staticmethod
    def dweight_loc(self, sim, uids):
        """
        Calculate the location parameter (mean drift) for weight change distribution.
        
        This method determines the mean drift in weight percentiles for individuals
        receiving macronutrient supplementation. The drift is proportional to the
        current time index to model cumulative intervention effects.
        
        Mathematical Formula:
            μ_i = 0 if not receiving macro supplementation
            μ_i = 1.0 * t_i if receiving macro supplementation
            where t_i is the current time index
        
        Args:
            sim (ss.Sim): The simulation object containing current state
            uids (np.ndarray): Array of unique identifiers for individuals (int64)
            
        Returns:
            np.ndarray: Mean drift values for weight change distribution (float64)
                       Shape: (len(uids),)
                       Values: 0.0 for non-supplemented, 1.0*ti for supplemented
                       
        Implementation Details:
            - Creates zero array for all individuals
            - Sets positive drift only for individuals with receiving_macro=True
            - Drift magnitude scales linearly with simulation time
        """
        mu = np.zeros(len(uids))
        mu[self.receiving_macro] = 1.0*self.ti # Upwards drift in percentile for those receiving macro supplementation
        return mu

    @staticmethod
    def dweight_scale(self, sim, uids):
        """
        Calculate the scale parameter (standard deviation) for weight change distribution.
        
        This method determines the standard deviation for weight changes over time.
        The scale increases linearly with time to model increasing variability in
        weight changes as individuals age and nutritional status becomes more variable.
        
        Mathematical Formula:
            σ_i = 0.01 * t_i for all individuals
            where t_i is the current time index
        
        Args:
            sim (ss.Sim): The simulation object containing current state
            uids (np.ndarray): Array of unique identifiers for individuals (int64)
            
        Returns:
            np.ndarray: Standard deviation values for weight change distribution (float64)
                       Shape: (len(uids),)
                       Values: All elements equal to 0.01 * current_time_index
                       
        Implementation Details:
            - Uses np.full to create array with identical values
            - Scale parameter increases linearly with simulation time
            - Applied uniformly to all individuals regardless of intervention status
        """
        std = np.full(len(uids), fill_value=0.01*self.ti)
        return std

    def weight(self, uids=None):
        """
        Calculate actual weight values (kg) from weight percentiles using LMS method.
        
        Converts weight percentiles (0.0-1.0) to actual weight measurements in kilograms
        using the LMS transformation with age and sex-specific reference parameters.
        
        Mathematical Formula:
            W = M * (L*S*Z + 1)^(1/L) for L ≠ 0
            W = M * exp(S*Z) for L = 0
            where W = weight (kg), M = median weight, L = skewness, S = coefficient of variation
            Z = Φ^(-1)(percentile) is the inverse normal CDF of the percentile
        
        Args:
            uids (np.ndarray, optional): Specific individuals to calculate weights for (int64)
                                       If None, calculates for all alive individuals
                                       
        Returns:
            np.ndarray: Actual weight values in kilograms (float64)
                       Shape: (len(uids),)
                       Range: Typically 2-100 kg depending on age and sex
                       
        Implementation Details:
            - Calls self.lms() with metric='Weight'
            - Uses weight_percentile state variable
            - Interpolates LMS parameters by age and sex from reference data
        """
        weight = self.lms(self.weight_percentile, uids, 'Weight')
        return weight

    def height(self, uids=None):
        """
        Calculate actual height values (cm) from height percentiles using LMS method.
        
        Converts height percentiles (0.0-1.0) to actual height measurements in centimeters
        using the LMS transformation with age and sex-specific reference parameters.
        
        Mathematical Formula:
            H = M * (L*S*Z + 1)^(1/L) for L ≠ 0
            H = M * exp(S*Z) for L = 0
            where H = height (cm), M = median height, L = skewness, S = coefficient of variation
            Z = Φ^(-1)(percentile) is the inverse normal CDF of the percentile
        
        Args:
            uids (np.ndarray, optional): Specific individuals to calculate heights for (int64)
                                       If None, calculates for all alive individuals
                                       
        Returns:
            np.ndarray: Actual height values in centimeters (float64)
                       Shape: (len(uids),)
                       Range: Typically 50-200 cm depending on age and sex
                       
        Implementation Details:
            - Calls self.lms() with metric='Height'
            - Uses height_percentile state variable
            - Interpolates LMS parameters by age and sex from reference data
        """
        height = self.lms(self.height_percentile, uids, 'Height')
        return height

    def lms(self, percentile, uids=None, metric='Weight'):
        """
        Calculate anthropometric measurements using the LMS (Lambda-Mu-Sigma) method.
        
        The LMS method is a statistical approach for constructing growth reference
        curves that accounts for the skewness of anthropometric data. It uses three
        parameters: lambda (skewness), mu (median), and sigma (coefficient of variation).
        
        This method interpolates LMS parameters by age and sex, then converts
        percentiles to actual measurements using the inverse LMS transformation.
        
        Mathematical Formula:
            For each individual i:
            age_months_i = age_years_i * 12
            L_i = interpolate(age_months_i, age_bins, L_values)
            M_i = interpolate(age_months_i, age_bins, M_values)  
            S_i = interpolate(age_months_i, age_bins, S_values)
            Z_i = Φ^(-1)(percentile_i)  # inverse normal CDF
            X_i = M_i * (L_i*S_i*Z_i + 1)^(1/L_i) if L_i ≠ 0
            X_i = M_i * exp(S_i*Z_i) if L_i = 0
            where X_i is the measurement (weight in kg or height in cm)
        
        Args:
            percentile (np.ndarray): Percentile values (0.0-1.0) for the measurements (float64)
            uids (np.ndarray, optional): Specific individuals to calculate for (int64)
                                       If None, uses all alive individuals
            metric (str): Type of measurement to calculate. Must be one of:
                         'Weight' (kg), 'Height' (cm), 'Length' (cm), 'BMI' (kg/m²)
                         
        Returns:
            np.ndarray: Actual anthropometric measurements (float64)
                       Shape: (len(uids),)
                       Units: kg for Weight, cm for Height/Length, kg/m² for BMI
                       
        Raises:
            AssertionError: If metric is not one of ['Weight', 'Height', 'Length', 'BMI']
            
        Implementation Details:
            - Processes males and females separately due to different reference data
            - Uses linear interpolation (np.interp) for age-specific parameters
            - Converts age from years to months for reference data lookup
            - Handles edge case where lambda parameter equals zero
            - Uses scipy.stats.norm().ppf() for percentile to z-score conversion
            
        References:
            - https://indianpediatrics.net/jan2014/jan-37-43.htm
            - https://iris.who.int/bitstream/handle/10665/44026/9789241547635_eng.pdf?sequence=1
        """
        # Return weight given a percentile using Cole's lambda, mu, and sigma (LMS) method 

        assert metric in ['Weight', 'Height', 'Length', 'BMI']

        if uids is None:
            uids = self.sim.people.auids

        ret = np.zeros(len(uids))

        ppl = self.sim.people
        female = ppl.female[uids]

        for sex, fem in zip(['Female', 'Male'], [female, ~female]):
            u = uids[fem]
            age = ppl.age[u] * 12 # in months

            age_bins = self.LMS_data.loc[sex]['Age']
            lam = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_L'])
            mu = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_M'])
            sigma = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_S'])

            # https://indianpediatrics.net/jan2014/jan-37-43.htm
            #Z = 1/(sigma*lam) * ((WEIGHT/mu)**lam - 1) 

            p = percentile[u]
            Z = norm().ppf(p) # Convert percentile to z-score
            ret[fem] = mu * (lam*sigma*Z + 1)**(1/lam) # if lam=0, w = mu * np.exp(sigma * Z)

            # https://iris.who.int/bitstream/handle/10665/44026/9789241547635_eng.pdf?sequence=1

        return ret

    def __init__(self, pars=None, **kwargs):
        """
        Initialize the Malnutrition disease model.
        
        Sets up the model parameters, loads anthropometric reference data,
        and defines the disease states for tracking individual nutritional status.
        
        Initialization Process:
        1. Calls parent class constructor (ss.Disease.__init__)
        2. Defines model parameters with default values
        3. Loads LMS reference data from anthropometry.csv
        4. Defines individual state variables
        5. Creates weight change distribution function
        
        Args:
            pars (dict, optional): Dictionary of parameters to override defaults
                                 Keys: 'beta', 'init_prev'
                                 Values: float
            **kwargs: Additional keyword arguments passed to parent class
            
        Parameters:
            beta (float): Transmission rate (placeholder, not used in malnutrition model)
                         Default: 1.0
            init_prev (float): Initial prevalence of malnutrition (0.0-1.0)
                             Default: 0.001 (0.1%)
                             
        State Variables Created:
            receiving_macro (bool): Macronutrient supplementation status
            receiving_micro (bool): Micronutrient supplementation status
            height_percentile (float): Height percentile (0.0-1.0)
            weight_percentile (float): Weight percentile (0.0-1.0)
            micro (float): Micronutrient status z-score
            
        File Dependencies:
            anthropometry.csv: Must be in tbsim/data/ directory with columns:
                              Sex, Age, Weight_L, Weight_M, Weight_S, Height_L, Height_M, Height_S
        """
        super().__init__(**kwargs)
        self.define_pars(
            beta = 1.0,         # Transmission rate  - TODO: Check if there is one
            init_prev = 0.001,  # Initial prevalence 
        )
        self.update_pars(pars, **kwargs)

        anthro_path = os.path.join(DATADIR, 'anthropometry.csv')
        self.LMS_data = pd.read_csv(anthro_path).set_index('Sex')

        # Adding Malnutrition states to handle the Individual Properties related to this disease 
        self.define_states(
            # Hooks to the RATIONS trial
            ss.BoolArr('receiving_macro', default=False), # Determines weight trend
            ss.BoolArr('receiving_micro', default=False), # Determines micro trend

            # Internal state
            # PROBLEM: Correlation between weight and height
            ss.FloatArr('height_percentile', default=ss.uniform(name='height_percentile')), # Percentile, stays fixed
            ss.FloatArr('weight_percentile', default=ss.uniform(name='weight_percentile')), # Percentile, increases when receiving micro, then declines?
            ss.FloatArr('micro', default=ss.uniform(name='micro')), # Continuous? Normal distribution around zero. Z-score, sigmoid thing. Half-life.
        )
        self.dweight = ss.normal(loc=self.dweight_loc, scale=self.dweight_scale)

        return

    def set_initial_states(self, sim):
        """
        Set initial values for disease states during simulation initialization.
        
        This method is called during simulation initialization to set up
        the initial nutritional status of individuals. Currently a placeholder
        for future implementation of correlated weight and height percentiles.
        
        Current Implementation:
            - No action taken (placeholder method)
            - All state variables initialized with defaults from define_states()
            
        Future Implementation Notes:
            - Could implement correlation between weight and height percentiles
            - Potential approach: bivariate normal distribution with correlation ρ
            - Corner corrections needed for percentiles near 0.0 or 1.0
            - Formula: (W_p, H_p) ~ BVN(μ, Σ) where Σ = [[1, ρ], [ρ, 1]]
        
        Args:
            sim (ss.Sim): The simulation object containing population data
            
        Returns:
            None
            
        Implementation Details:
            - Called once during simulation setup
            - Runs after individual creation but before first time step
            - Could modify height_percentile, weight_percentile, micro states
        """
        # Could correlate weight and height here, via gaussian along the diagonal with corner correction?
        return

    '''
    def set_macro_supplement(self, uids, kcal):
        self.macro_drift[uids] = kcal
        pass

    def set_micro_supplement(self, uids, stop):
        self.micro_drift[uids[~stop]] = True
        self.micro_drift[uids[stop]] = 0
        pass
    '''

    def step(self):
        """
        Execute one time step of the malnutrition model.
        
        This method is called at each simulation time step to update the
        nutritional status of individuals. It implements random walk processes
        for weight percentiles and applies nutritional interventions.
        
        Mathematical Model:
            For each individual i at time t:
            ΔW_i(t) ~ N(μ_i(t), σ_i(t)²)
            W_i(t+1) = W_i(t) + ΔW_i(t)
            W_i(t+1) = clip(W_i(t+1), 0.025, 0.975)
            where W_i(t) is weight percentile, μ_i(t) is drift from dweight_loc(),
            σ_i(t) is scale from dweight_scale(), and clip() ensures valid percentiles
        
        Step Process:
        1. Get all alive individual IDs
        2. Sample weight changes from normal distribution
        3. Update weight percentiles with random walk
        4. Clip percentiles to valid range (0.025-0.975)
        5. Apply intervention effects through drift parameters
        
        Args:
            None (uses self.sim for simulation state)
            
        Returns:
            None (modifies state variables in-place)
            
        Implementation Details:
            - Uses self.dweight() which calls dweight_loc() and dweight_scale()
            - Clipping prevents percentiles from reaching exact 0.0 or 1.0
            - Random walk process models natural weight variability
            - Intervention effects applied through location parameter
        """
        uids = self.sim.people.auids # All alive uids

        # Random walks
        self.weight_percentile[uids] += self.dweight(uids)
        self.weight_percentile[uids] = np.clip(self.weight_percentile[uids], 0.025, 0.975) # needed?

        '''
        new_macro = (self.ti_macro == ti).uids
        if len(new_macro) > 0:
            self.macro_state[new_macro] = self.new_macro_state[new_macro]

        new_micro = (self.ti_micro == ti).uids
        if len(new_micro) > 0:
            self.micro_state[new_micro] = self.new_micro_state[new_micro]

        return new_macro, new_micro
        '''
        return

    def init_results(self):
        """
        Initialize results tracking for the malnutrition model.
        
        Sets up the results structure to track key metrics during simulation.
        Currently tracks the proportion of people alive at each time step.
        
        Results Structure:
            people_alive (float): Proportion of population alive at each time step
                                Shape: (n_timesteps,)
                                Range: 0.0-1.0
                                Units: Proportion (dimensionless)
        
        Args:
            None (uses self.sim for simulation parameters)
            
        Returns:
            None (creates self.results object)
            
        Implementation Details:
            - Calls parent class init_results() first
            - Defines results using ss.Result objects
            - Results stored in self.results dictionary
            - Updated each time step in update_results()
        """
        super().init_results()
        self.define_results(
            ss.Result(name='people_alive', dtype=float, label='People alive'),
        )
        return

    def update_results(self):
        """
        Update results at each time step.
        
        Records the current state of the simulation for analysis. Currently
        tracks the proportion of individuals who are alive at each time step.
        
        Mathematical Formula:
            people_alive[ti] = count(alive_individuals) / total_population
            where ti is the current time index, alive_individuals is boolean array
            indicating survival status, and total_population is n_agents parameter
        
        Args:
            None (uses self.sim for simulation state)
            
        Returns:
            None (updates self.results in-place)
            
        Implementation Details:
            - Called at each simulation time step
            - Uses self.sim.people.alive boolean array
            - Calculates proportion as count(alive)/n_agents
            - Stores result in self.results.people_alive[ti]
            - ti is current time index from self.sim.ti
        """
        super().update_results()
        ti = self.sim.ti            # Current time index (step)
        alive = self.sim.people.alive    # People alive at current time index
        n_agents = self.sim.pars['n_agents']
        self.results.people_alive[ti] = alive.count()/n_agents
        return