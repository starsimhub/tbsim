# API Reference

This section provides comprehensive API documentation for all TBsim modules and components, automatically generated from Python docstrings.

## Core Modules

- [api/tbsim](api/tbsim.md)
- [api/tbsim.tb](api/tbsim.tb.md)
- [api/tbsim.networks](api/tbsim.networks.md)

## Analysis and Visualization

- [api/tbsim.analyzers](api/tbsim.analyzers.md)
- [api/tbsim.plots](api/tbsim.plots.md)

## Interventions

- [api/tbsim.interventions](api/tbsim.interventions.md)
- [api/tbsim.interventions.bcg](api/tbsim.interventions.bcg.md)
- [api/tbsim.interventions.beta](api/tbsim.interventions.beta.md)
- [api/tbsim.interventions.enhanced_tb_diagnostic](api/tbsim.interventions.enhanced_tb_diagnostic.md)
- [api/tbsim.interventions.enhanced_tb_treatment](api/tbsim.interventions.enhanced_tb_treatment.md)
- [api/tbsim.interventions.healthseeking](api/tbsim.interventions.healthseeking.md)
- [api/tbsim.interventions.interventions](api/tbsim.interventions.interventions.md)
- [api/tbsim.interventions.tb_diagnostic](api/tbsim.interventions.tb_diagnostic.md)
- [api/tbsim.interventions.tb_drug_types](api/tbsim.interventions.tb_drug_types.md)
- [api/tbsim.interventions.tb_health_seeking](api/tbsim.interventions.tb_health_seeking.md)
- [api/tbsim.interventions.tb_treatment](api/tbsim.interventions.tb_treatment.md)
- [api/tbsim.interventions.tpt](api/tbsim.interventions.tpt.md)

## Comorbidities

- [api/tbsim.hiv](api/tbsim.hiv.md)
- [api/tbsim.malnutrition](api/tbsim.malnutrition.md)

## Configuration and Support

- [api/tbsim.version](api/tbsim.version.md)

## Data and Utilities

- [api/tbsim.data](api/tbsim.data.md)

## Module Overview

**Core TB Model** (`tbsim.tb`)  
Main tuberculosis simulation module with disease dynamics, transmission, and state transitions. Implements the TBS state enumeration and TB disease class.

**Networks** (`tbsim.networks`)  
Social network structures for modeling transmission patterns, including household networks and RATIONS trial specific implementations.

**Analyzers** (`tbsim.analyzers`)  
Comprehensive data analysis tools including dwell time analysis (DWT), visualization, and post-processing capabilities for simulation results.

**Interventions** (`tbsim.interventions.*`)  
Various intervention modules for TB control and prevention, including DOTS implementation, BCG vaccination, enhanced diagnostics, and treatment protocols.

**Comorbidities** (`tbsim.comorbidities.hiv`, `tbsim.comorbidities.malnutrition`)  
Modeling of HIV, malnutrition, and other co-occurring conditions with bidirectional interactions with TB dynamics.

**Plots** (`tbsim.plots`)  
Plotting and visualization tools for simulation results.

**Data** (`tbsim.data`)  
Anthropometric reference data for malnutrition modeling.

For detailed information about each module, click on the links above or use the search functionality. All documentation is automatically generated from Python docstrings to ensure accuracy and completeness.
