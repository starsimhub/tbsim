# User Guide

TBsim is a comprehensive tuberculosis simulation framework built on the Starsim platform, designed for modeling TB transmission, natural history, and intervention impacts in realistic populations.

## Core Framework

TBsim leverages the Starsim individual-based modeling framework with the following key capabilities:

**Individual-Based Modeling (IBM)**  
Each person in the simulation is represented as an individual agent with unique characteristics, health states, and behaviors.

**Network-Based Transmission**  
Social connections between individuals are modeled as networks, enabling realistic transmission dynamics through household and community structures.

**Modular Architecture**  
Components can be mixed and matched to create custom simulation scenarios with flexible parameterization.

## TB Natural History and Disease States

TBsim implements a comprehensive TB natural history model with the following disease states:

**Core Disease States**

- **Susceptible (NONE)**: No TB infection
- **Latent TB (LATENT_SLOW/FAST)**: Two progression rates for latent infection
- **Active Pre-symptomatic (ACTIVE_PRESYMP)**: Early active disease before symptoms
- **Active Symptomatic Smear+ (ACTIVE_SMPOS)**: Most infectious form of TB
- **Active Symptomatic Smear- (ACTIVE_SMNEG)**: Moderately infectious TB
- **Extra-pulmonary TB (ACTIVE_EXPTB)**: TB outside the lungs
- **Protected (PROTECTED)**: Protected state from vaccines or preventive therapy
- **TB Death (DEAD)**: TB-related mortality

**Transmission Dynamics**

- Airborne transmission through social networks
- Smear-based infectiousness (smear+ > smear- > extra-pulmonary)
- Age-dependent transmission rates
- Individual transmission heterogeneity
- Household and community network structures

## Diagnostic Capabilities

TBsim includes comprehensive diagnostic modeling with multiple test types:

**Available Diagnostic Tests**

- **Xpert MTB/RIF**: Molecular testing for TB and rifampin resistance
- **FujiLAM**: Urine-based testing for HIV+ individuals
- **CAD CXR**: Computer-aided chest X-ray interpretation
- **Oral Swab**: Non-sputum based testing

**Diagnostic Features**

- Age and HIV-specific sensitivity/specificity parameters
- TB state-specific test performance
- False negative handling with care-seeking multipliers
- Enhanced diagnostic interventions with comprehensive result tracking

## Treatment and Interventions

**Drug Treatment**

- **DOTS**: Directly Observed Treatment, Short-course implementation
- **Latent Treatment**: TB Preventive Therapy (TPT) with multiple regimens
- **Treatment Success/Failure**: Configurable treatment outcomes
- **Treatment Relapse**: Post-treatment disease recurrence modeling

**Preventive Interventions**

- **BCG Vaccination**: Comprehensive BCG protection with age targeting
- **TB Preventive Therapy (TPT)**: Household-based and individual TPT
- **Vaccine Campaigns**: Mass vaccination strategies

**Health Seeking Behavior**

- Configurable care-seeking rates
- Care-seeking multipliers for different populations
- Re-seeking behavior after treatment failure
- Age-dependent health seeking patterns

## Disease Connectors and Co-infections

TBsim includes sophisticated connectors to link TB with other diseases:

**TB-HIV Co-infection**

- **HIV State Integration**: Uses HIV disease states (ACUTE, LATENT, AIDS) to modify TB progression
- **CD4-dependent Effects**: TB progression varies with HIV disease stage
- **ART Effects**: Antiretroviral therapy impacts on TB natural history
- **Risk Multipliers**: HIV increases TB activation risk (ACUTE: 1.22x, LATENT: 1.90x, AIDS: 2.60x)

**TB-Malnutrition Integration**

- **Comprehensive Malnutrition Modeling**: BMI-based nutritional status effects
- **Nutrition-TB Connector**: Nutritional status impacts TB progression and treatment
- **Supplementation Effects**: Nutritional intervention modeling

## Networks and Social Structure

**Network Types**

- **Household Networks**: Family and household connections for transmission
- **Trial-based Networks**: Framework for intervention trials
- **Contact Networks**: Detailed contact tracing capabilities

**Transmission Networks**

- Household-based transmission modeling
- Age-stratified contact patterns
- Network-based intervention targeting

## Analysis and Visualization Tools

TBsim provides comprehensive analysis capabilities through the DWT (Dwell Time) Analyzer system:

**Dwell Time Analysis**

- **Real-time Tracking**: Records time spent in each TB state during simulation
- **State Transition Analysis**: Detailed progression pathway tracking
- **Statistical Analysis**: Comprehensive dwell time statistics

**Visualization Capabilities**

- **Sankey Diagrams**: Interactive state transition flows
- **Network Graphs**: Transmission network visualization
- **Interactive Plots**: Plotly-based interactive visualizations
- **Kaplan-Meier Curves**: Survival analysis for TB progression
- **Histogram Analysis**: Dwell time distribution analysis
- **Reinfection Analysis**: Multiple infection episode tracking

**Reporting Features**

- **Treatment Events**: Comprehensive treatment outcome tracking
- **Diagnostic Events**: Test result and diagnostic pathway analysis
- **Mortality Events**: TB-related death tracking
- **Age-stratified Reports**: Age-specific analysis capabilities
- **TB-HIV Reports**: Co-infection specific reporting
- **Household Analysis**: Household-level intervention impact assessment

## Advanced Features

**Multi-Scenario Analysis**

- Run multiple parameter combinations simultaneously
- Comparative analysis across intervention scenarios

**Time-varying Parameters**

- Dynamic parameter adjustment during simulation
- Seasonal and temporal variation modeling

**Trial Framework Support**

- Household-based intervention trials
- Randomized controlled trial simulation capabilities

**Calibration and Optimization**

- Automated parameter fitting to observed data
- Intervention strategy optimization
- Sensitivity analysis for parameter uncertainty

## Running Simulations

Basic simulation workflow:

1. **Setup**: Configure TB model, networks, and interventions
2. **Initialize**: Create population with demographic data
3. **Run**: Execute simulation with real-time analysis
4. **Analyze**: Process results with comprehensive visualization tools
5. **Export**: Save results for further analysis and reporting

**Example Usage**

```python
import starsim as ss
import tbsim

# Create simulation with TB model and analyzer
sim = ss.Sim(
    diseases=[tbsim.TB_EMOD()],
    networks=tbsim.HouseholdNet(),
    analyzers=[tbsim.DwellTime(scenario_name="Baseline")]
)
sim.run()

# Generate comprehensive analysis plots
analyzer = sim.analyzers[0]
analyzer.plot('sankey')
analyzer.plot('network')
analyzer.plot('kaplan_meier')
```

For specific examples and tutorials, see the [examples](examples.md) and [tutorials](tutorials.md) sections.
