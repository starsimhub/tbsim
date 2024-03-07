Source of information:  https://www.pnas.org/doi/full/10.1073/pnas.0901720106 

States:
- **Susceptible(S):**
- **Latent Fast (LF):**
- **Latent Slow (LS):**
- **Active Pre-Symtomatic (APS):**
- **Smear-Positive (SP):**
- **Smear-Negative (SN):**
- **Extra-Pulmonary (EP):**
- **Dead (D):**


< in progress ... >

*** Information  above is based on https://www.pnas.org/doi/full/10.1073/pnas.0901720106  and EMOD TB model schematic.


---
# Additional Information:

Modeling tuberculosis (TB) using an agent-based technique involves creating a simulation where individual agents (representing people) interact within a defined environment. This approach can capture the complexities of TB spread, including social interactions, movement patterns, and individual health status. 

## States
The eight states commonly used in TB models are:

1. **Susceptible (S):** Individuals who can contract TB.
2. **Latent Infection (L):** Individuals who have contracted TB but do not show symptoms and are not infectious.
3. **Primary Active TB (P):** Individuals who develop active TB soon after the initial infection.
4. **Secondary Active TB (I):** Individuals who develop active TB after a period of latency.
5. **Recovered (R):** Individuals who have recovered from TB and may have immunity.
6. **Failed Treatment (F):** Individuals whose treatment for TB was not successful.
7. **Relapse (Re):** Recovered individuals who relapse back to active TB.
8. **Dead (D):** Individuals who have died from TB or other causes.


## High level tasks
Overall the tasks associated to this project should cover some of all of the points listed below.
1. **Agent Attributes:**
   - Agents represent individuals in the population.
   - Each agent should have a state attribute indicating their current TB status (S, L, P, I, R, F, Re, D).
   - Other attributes include demographic details (gender, age, etc.), health status, social behavior patterns, and history of TB exposure or treatment. 

1. **Transition Rules:**
   - Transitions between states, based on interaction with infectious agents, time since infection, treatment status, and individual health factors have been already defined by Stewart and is available as powerpoint presentation here: (**TODO:** add link).
   - For example, susceptible individuals (S) become latently infected (L) upon exposure to primary (P) or secondary (I) active TB cases.

1. **Environment and Interaction Dynamics:**
   - To model how agents interact in different settings (e.g., home, work, public places) and how these interactions lead to TB transmission.
   - To consider varying transmission rates in different settings and for different types of active TB (as defined in Stewart's documentation) - (TODO: Add more context)
   - TB transmission dynamics should be based on epidemiological data. This includes the likelihood of transmission upon contact, the incubation period, and the duration of infectiousness.
   - Consider different strains of TB, such as drug-resistant strains, if relevant.

1. **Disease Progression and Treatment:**
   - Simulate disease progression
   - To model the progression from latent infection to active TB, considering factors like immune response and co-morbidities.
   - Include treatment protocols for active TB and their outcomes, influencing transitions to recovered (R) or failed treatment (F).
   - Each agent follows a disease progression model, which could lead to recovery or worsening of the disease.
   - Include the effects of treatment, such as shortened infectious periods or cure.

1. **Incorporating Social Interactions and Movements:**
   - Define how agents interact and move within the environment. This could include daily routines, social gatherings, and travel patterns.
   - Social networks can be modeled to understand how close contacts influence the spread of TB.

1. **Incorporating Healthcare System Interactions:**
   - Model the healthcare system's role, including diagnosis, treatment, and public health interventions.
   - Include treatment adherence and the possibility of treatment failure.

1. **Public Health Interventions:**
   - Implement interventions like vaccination, improved diagnosis, treatment strategies, and public health campaigns.
   - Evaluate the impact of these interventions on the different states and overall TB dynamics.
   - Model additional interventions, like vaccination, contact tracing, quarantine, or improved healthcare access.
   - Assess the impact of these interventions on the spread and control of TB.

1. **Incorporate Relapse and Mortality:**
   - Include the probability of relapse (Re) for recovered individuals.
   - Model mortality (D) due to TB and other causes, impacting the population dynamics.

1. **Simulation Execution:**
   - Run simulations with varied parameters to explore different scenarios, such as increased treatment efficacy or changes in social behavior.
   - Track the prevalence of each state over time and under different intervention strategies.
   - Run the simulation multiple times with varying parameters to understand different scenarios.
   - Analyze the results to identify patterns, potential hotspots, and the effectiveness of interventions.

1. **Data Analysis and Model Validation:**
   - Analyze the distribution and transitions of states over time and compare with real-world data.
   - Validate the model by ensuring it can replicate known TB epidemiology trends and respond realistically to interventions.

1. **Calibration and Validation:**
   - Calibrate your model using real-world data to ensure its accuracy.
   - Validate the model by comparing its predictions with independent data sets or historical data.

1. **Sensitivity Analysis:**
   - Test the sensitivity of the outcomes to changes in key parameters to understand the robustness of the model's predictions.
   - Perform sensitivity analysis to understand how changes in parameters affect the outcomes. This is crucial for understanding the robustness of your model's predictions.


---- 
EMOD:

```
#pragma once
#include "InfectionAirborne.h"

#include "TBInterventionsContainer.h"
#include "SusceptibilityTB.h"
#include "Infection.h"

namespace Kernel
{
    // find a home for these...  TBEnums.h?
    ENUM_DEFINE(TBInfectionDrugResistance,
        ENUM_VALUE_SPEC(DrugSensitive           , 0)
        ENUM_VALUE_SPEC(FirstLineResistant      , 1))
    class IIndividualHumanCoInfection;

    class IInfectionTB : public ISupports
    {
    public:
        virtual bool IsSmearPositive() const = 0;
        virtual bool IsMDR() const = 0 ; 
        virtual float GetLatentCureRate() const = 0;
        virtual bool IsSymptomatic() const = 0;
        virtual bool IsActive() const = 0;
        virtual bool IsExtrapulmonary() const = 0; 
        virtual bool IsFastProgressor() const = 0;
        virtual float GetDurationSinceInitialInfection() const = 0; 
        virtual bool EvolvedResistance() const = 0;
        virtual bool IsPendingRelapse() const = 0;
        virtual void ExogenousLatentSlowToFast() = 0;
        virtual void LifeCourseLatencyTimerUpdate() = 0;
    };

    class InfectionTBConfig : public InfectionAirborneConfig
    {
        friend class IndividualTB;
        GET_SCHEMA_STATIC_WRAPPER(InfectionTBConfig)
        IMPLEMENT_DEFAULT_REFERENCE_COUNTING()
        DECLARE_QUERY_INTERFACE()

    public:
        virtual bool Configure( const Configuration* config ) override;
        std::map <float,float> GetCD4Map();
        InfectionStateChange::_enum TB_event_type_associated_with_infectious_timer;
        
    protected:
        friend class InfectionTB;
        
        static float TB_latent_cure_rate;
        static float TB_fast_progressor_rate;
        static float TB_slow_progressor_rate;
        static float TB_active_cure_rate;
        static float TB_inactivation_rate;
        static float TB_active_mortality_rate;
        static float TB_extrapulmonary_mortality_multiplier;
        static float TB_smear_negative_mortality_multiplier;
        static float TB_active_presymptomatic_infectivity_multiplier;
        static float TB_presymptomatic_rate;
        static float TB_presymptomatic_cure_rate;
        static float TB_smear_negative_infectivity_multiplier;
        static float TB_Drug_Efficacy_Multiplier_MDR;
        static float TB_Drug_Efficacy_Multiplier_Failed;
        static float TB_Drug_Efficacy_Multiplier_Relapsed;
        static float TB_MDR_Fitness_Multiplier;
        static std::map <float,float> CD4_map;
        static float TB_relapsed_to_active_rate;
        
        static DistributionFunction::Enum TB_active_period_distribution;
        static float TB_active_period_std_dev;

        static vector <float> TB_cd4_activation_vec;
        static vector <float> CD4_strata_act_vec;
        static IDistribution* p_infectious_timer_distribution;
    };

    //---------------------------- InfectionTB ----------------------------------------
    class InfectionTB : public InfectionAirborne, public IInfectionTB
    {
        IMPLEMENT_DEFAULT_REFERENCE_COUNTING()
        DECLARE_QUERY_INTERFACE()

    public:
        virtual ~InfectionTB(void);
        static InfectionTB *CreateInfection(IIndividualHumanContext *context, suids::suid _suid);

        virtual void SetParameters(IStrainIdentity* infstrain=nullptr, int incubation_period_override = -1) override;
        virtual void Update(float dt, ISusceptibilityContext* immunity = nullptr) override;
        virtual void InitInfectionImmunology(ISusceptibilityContext* _immunity) override;
        virtual void SetContextTo(IIndividualHumanContext * context) override;
        
       // Inherited from base class
        virtual bool IsActive() const override;

        //TB-specific
        virtual bool IsSmearPositive() const override;
        virtual bool IsExtrapulmonary() const override;
        virtual bool IsFastProgressor() const override;
        virtual bool IsMDR() const override;
        virtual bool EvolvedResistance() const override;
        virtual bool IsPendingRelapse() const override;
        virtual bool IsSymptomatic() const override;
        virtual float GetLatentCureRate() const override;
        virtual float GetDurationSinceInitialInfection() const override; 
        virtual void LifeCourseLatencyTimerUpdate() override;

        // Exogenous re-infection
        virtual void ModifyInfectionStrain(IStrainIdentity * exog_strain_id);
        virtual void ExogenousLatentSlowToFast();

    protected:
        InfectionTB();
        InfectionTB(IIndividualHumanContext *context);

        // For disease progression and MDR evolution, virtual functions are inherited from base class Infection
        virtual void Initialize(suids::suid _suid);
        void  InitializeLatentInfection(ISusceptibilityContext* immunity);
        void  InitializeActivePresymptomaticInfection(ISusceptibilityContext* immunity);
        void  InitializeActiveInfection(ISusceptibilityContext* immunity);
        void  InitializePendingRelapse(ISusceptibilityContext* immunity);
        bool  ApplyDrugEffects(float dt, ISusceptibilityContext* immunity = nullptr);
        virtual void EvolveStrain(ISusceptibilityContext* _immunity, float dt) override;
        TBDrugEffects_t GetTotalDrugEffectsForThisInfection();
        float CalculateTimerAgeDepSlowProgression(ISusceptibilityContext* immunity);

        // additional TB infection members
        // This chunk gets serialized.

        IIndividualHumanCoInfection* human_coinf;
        bool  m_is_active;
        float m_recover_fraction;
        float m_death_fraction;
        bool  m_is_smear_positive;
        bool  m_is_extrapulmonary;
        bool  m_is_fast_progressor;
        bool  m_evolved_resistance;
        bool  m_is_pending_relapse;
        bool  m_shows_symptoms;
        float m_duration_since_init_infection; //for reporting only

        DECLARE_SERIALIZABLE(InfectionTB);
    };
}

```
