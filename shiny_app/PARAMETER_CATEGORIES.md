# TBsim Shiny App - Parameter Categories

This document outlines all the parameter categories available in the TBsim Shiny application, organized by their function in the tuberculosis transmission model.

## 📊 **Simulation Parameters**
- **Population Size**: Number of individuals in the simulation (100-10,000)
- **Start Date**: Simulation start date (default: 1940-01-01)
- **End Date**: Simulation end date (default: 2010-12-31)
- **Time Step**: Simulation time step in days (1-30 days)
- **Random Seed**: Random number generator seed (1-10,000)

## 🦠 **TB Disease Parameters**
- **Initial Prevalence**: Proportion of population initially infected (0-1)
- **Transmission Rate**: Base transmission rate per year (0-0.1)
- **Probability of Fast Latent TB**: Proportion developing fast latent TB (0-1)

## 🔄 **TB State Transition Rates**
- **Latent Slow → Pre-symptomatic**: Rate per day (0-0.001)
- **Latent Fast → Pre-symptomatic**: Rate per day (0-0.1)
- **Pre-symptomatic → Active**: Rate per day (0-1)
- **Active → Clearance**: Natural clearance rate per day (0-0.01)
- **Treatment → Clearance**: Treatment clearance rate per year (0-50)

## 💀 **TB Mortality Rates**
- **Extra-Pulmonary TB → Death**: Death rate per day (0-0.001)
- **Smear Positive → Death**: Death rate per day (0-0.001)
- **Smear Negative → Death**: Death rate per day (0-0.001)

## 🦠 **TB Transmissibility**
- **Pre-symptomatic Relative Transmissibility**: Multiplier (0-1)
- **Smear Positive Relative Transmissibility**: Multiplier (0-2)
- **Smear Negative Relative Transmissibility**: Multiplier (0-1)
- **Extra-Pulmonary Relative Transmissibility**: Multiplier (0-1)
- **Treatment Effect on Transmissibility**: Multiplier (0-1)

## 🛡️ **TB Susceptibility**
- **Latent Slow Relative Susceptibility**: Reinfection susceptibility (0-1)

## 🔬 **TB Diagnostics**
- **Chest X-ray Sensitivity**: Sensitivity for asymptomatic cases (0-1)

## 🎲 **TB Heterogeneity**
- **Transmission Heterogeneity**: Individual variation in transmission (0.1-5)

## 👥 **Demographics**
- **Birth Rate**: Births per 1000 population per year (0-100)
- **Death Rate**: Deaths per 1000 population per year (0-100)

## 🌐 **Social Network**
- **Average Contacts per Person**: Mean number of contacts (1-50)

## 🎯 **Parameter Usage in Model**

### **Transmission Dynamics**
- Base transmission rate × Active cases × Transmissibility factors
- Different transmissibility for different TB states
- Heterogeneity in individual transmission potential

### **Disease Progression**
- **S → L**: Susceptible to Latent (based on transmission rate)
- **L → A**: Latent to Active (different rates for fast/slow)
- **A → S**: Active to Susceptible (natural clearance + treatment)

### **Mortality**
- TB-related deaths based on disease state
- Different mortality rates for different TB forms

### **Diagnostics**
- Screening sensitivity affects case detection
- Influences treatment initiation and outcomes

## 🔧 **Technical Implementation**

All parameters are implemented as **slider inputs** for easy adjustment:
- **Real-time updates**: Changes immediately affect simulation
- **Reset functionality**: One-click return to default values
- **Validation**: Parameter ranges prevent invalid inputs
- **Documentation**: Clear labels and units for each parameter

## 📈 **Model Outputs**

The enhanced model provides:
- **Susceptible population** (green)
- **Total infected** (red)
- **Latent TB** (orange)
- **Active TB** (dark red)
- **Time series** in years
- **Summary statistics** table

## 🚀 **Usage Tips**

1. **Start with defaults**: Use "Reset to Defaults" for baseline
2. **Adjust gradually**: Change one parameter at a time to see effects
3. **Use realistic ranges**: Parameters are set to epidemiologically plausible values
4. **Compare scenarios**: Run multiple simulations with different parameter sets
5. **Check results**: Use summary statistics to understand outcomes

## 📚 **References**

Parameters are based on:
- TBsim package documentation
- Epidemiological literature
- WHO TB guidelines
- Clinical trial data
- Mathematical modeling best practices
