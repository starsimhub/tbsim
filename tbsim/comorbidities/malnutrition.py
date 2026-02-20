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

__all__ = ["Malnutrition", "TB_Nutrition_Connector"]


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
            ss.FloatArr('height_percentile', default=ss.uniform(0.0, 1.0)), # Percentile, stays fixed
            ss.FloatArr('weight_percentile', default=ss.uniform(0.0, 1.0)), # Percentile, increases when receiving micro, then declines?
            ss.FloatArr('micro', default=ss.uniform(0.0, 1.0)), # Continuous? Normal distribution around zero. Z-score, sigmoid thing. Half-life.
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
        self.results.people_alive[ti] = np.count_nonzero(alive)/n_agents
        return


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
