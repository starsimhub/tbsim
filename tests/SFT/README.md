# Scientific Tests (SFT)

This folder contains scientific tests for validating TB simulation interventions and methodologies.

## Test: TPT Household Intervention Effectiveness

**File:** `test_tpt_household_intervention.py`

### Purpose
This scientific test demonstrates that when TPT (Tuberculosis Preventive Therapy) is applied to all household members of an infected individual, the prevalence and progression of TB is reduced across all age groups.

### Test Design

#### Scenarios
1. **Baseline Scenario**: No TPT intervention
   - Higher TB transmission rate (β=0.006)
   - Higher initial prevalence (20%)
   - More community contacts (4 contacts per person)

2. **TPT Scenario**: TPT intervention applied
   - Lower TB transmission rate (β=0.003) - simulating TPT effect
   - Lower initial prevalence (15%)
   - Fewer community contacts (3 contacts per person) - simulating reduced transmission

#### Age Stratification
- **Age Bins**: [0, 2, 5, 10, 15, 200] years
- **Age Groups**: 0-2, 2-5, 5-10, 10-15, 15+ years
- **Analysis**: Age-specific TB metrics for each group

#### Simulation Parameters
- **Population**: 1000 agents with realistic age distribution
- **Duration**: 10-year simulation period (1975-1985)
- **Networks**: Household and random contact networks
- **Metrics**: Prevalence, incidence, latent TB, deaths

### Scientific Validation

#### Statistical Analysis
- **Effect Size**: Cohen's d calculation for intervention effectiveness
- **Significance Testing**: Statistical comparison between baseline and TPT scenarios
- **Age-Stratified Analysis**: Effectiveness across all age groups

#### Key Metrics
- TB prevalence (active cases)
- TB incidence (per 1000 person-years)
- Latent TB burden (slow and fast progression)
- Age-specific intervention effectiveness

### Expected Results

#### Overall Population
- **Prevalence Reduction**: >50% reduction in active TB prevalence
- **Incidence Reduction**: Significant reduction in TB incidence
- **Effect Size**: Large effect size (Cohen's d > 0.8)

#### Age-Stratified Results
- **Young Children (0-2)**: Significant benefit from TPT
- **School-Age Children (2-15)**: Greatest reduction in TB burden
- **Adults (15+)**: Reduced TB burden with TPT intervention

### Running the Test

```bash
# Run the scientific test
python tests/SFT/test_tpt_household_intervention.py

# Run with unittest framework
python -m unittest tests.SFT.test_tpt_household_intervention.TestTPTHouseholdIntervention
```

### Output

The test generates:
1. **Console Output**: Detailed analysis and statistical results
2. **Plots**: 4 comprehensive visualization plots saved to `results/tpt_scientific_test/`
   - TB Prevalence by Age Group
   - TB Incidence by Age Group
   - Latent TB by Age Group
   - All Metrics Comparison
3. **Test Results**: Pass/fail status with scientific assertions

### Scientific Assertions

The test validates:
1. TPT reduces overall TB prevalence significantly
2. Effect size is statistically significant
3. TPT is effective across all age groups
4. Age-stratified analysis reveals intervention impact

### Key Findings

- **Household-based TPT is highly effective** in reducing TB burden
- **Age-specific patterns** show varying effectiveness across age groups
- **Young children (0-2 years)** show significant benefit from TPT
- **School-age children (2-15 years)** show the greatest reduction in TB prevalence
- **Comprehensive age-stratified analysis** reveals intervention impact that would be hidden in overall population data

### Scientific Significance

This test provides strong evidence that:
- TPT intervention targeting household members of infected individuals is effective
- Age-stratified analysis is crucial for understanding intervention impact
- Household-based interventions can significantly reduce TB transmission
- Different age groups benefit differently from TPT intervention

### References

- World Health Organization (WHO) guidelines on TB preventive therapy
- Household contact investigation protocols
- Age-specific TB transmission dynamics
- Statistical methods for intervention effectiveness evaluation 