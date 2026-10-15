# User guide

TBsim is a tuberculosis model built on Starsim. Its model of TB is declared as states and transitions, so the same model runs as an ODE, as a stochastic compartmental model, or as agents on contact networks, and its parameters can be fitted to data.

## The model

`tbsim.TB` is the LSHTM "spectrum of TB disease" model ([Schwalb et al. 2025](https://journals.plos.org/globalpublichealth/article?id=10.1371/journal.pgph.0005050)). Its states are:

- **`susceptible`**: never infected
- **`infection`**: latent infection
- **`cleared`**, **`recovered`**, and **`treated`**: no longer infected, after clearing a latent infection, recovering from non-infectious TB, or completing treatment; each has its own risk of reinfection
- **`non_infectious`**: subclinical TB that is not infectious
- **`asymptomatic`**: infectious TB without symptoms
- **`symptomatic`**: infectious TB with symptoms, from which people can die of TB
- **`treatment`**: on treatment, which ends in completion or failure

Its parameters (`tbsim.tb_pars()`) are rates per year. Optional parts are front-loaded progression (`k_latent`, a remote latent stage), relapse after treatment (`p_relapse`), waning protection after clearance (`dur_reinfection_protection`), and migration (`immigration`, `emigration`). Births replace deaths unless `mu=0`.

```python
import tbsim

sim = tbsim.Sim(beta=12, start=2000, stop=2030)  # An ODE
sim.run()
sim.plot()
```

With no network, people mix homogeneously and the sim runs as an ODE; `solver='abm'`, or a network such as `ss.RandomNet()`, runs it as agents.

## Diagnosis and treatment

- **Tests**: `tbsim.Xpert()`, `tbsim.OralSwab()`, `tbsim.FujiLAM()`, and `tbsim.CAD()` are `ss.Test` products whose sensitivity depends on the TB state and on age; `tbsim.test()` makes one from a table of your own.
- **The care cascade**: `tbsim.care(tb)` has people with symptomatic TB seek care, be tested, and start treatment if they test positive.
- **Regimens**: `tbsim.regimen('dots')` gives the model's treatment parameters (completion, failure, and relapse) for a regimen.

## Prevention

- **BCG**: `tbsim.BCG()` is an `ss.Vaccine` that slows progression, speeds recovery, and lowers TB mortality, delivered with `ss.vaccinate()`.
- **Preventive therapy**: `tbsim.TPT()` is an `ss.Treatment` that slows progression from latent infection while it protects, delivered with `ss.treat()`.

## Comorbidities

- **HIV**: `tbsim.HIV()`, with HIV infection at a given (possibly time-varying) incidence and ART started at a rate or by `ss.transfer()`, and `tbsim.hiv_effects(tb, hiv)`, modifiers that raise progression to active TB by HIV state (as an ODE or as agents).
- **Malnutrition**: `tbsim.Malnutrition()` with `tbsim.nutrition_effects(tb)`, modifiers on susceptibility, progression, and clearance, and `tbsim.supplementation_rr()` for nutritional supplements.

## Drug resistance

`tbsim.TB(drugs=['RIF', 'INH'])` has a strain for each set of drugs it is resistant to, and people can carry several strains at once. Resistant strains can be less fit, are acquired on progression and during treatment, and are cured less often; results are read by strain (`sim.results.tb.n_symptomatic.by('resistance')`, `tbsim.resistant_share(sim)`). `tbsim.LSHTMResistant` is the reference two-strain model. See the [drug resistance user guide](resistance.qmd) and the [drug resistance tutorial](../tutorials/resistance_tutorial.qmd).

## Analysis

- **Dwell times**: `tbsim.DwellTime` records how long agents spend in each state, with histograms, Kaplan-Meier curves, and state-transition networks (`analyzer.plot('kaplan_meier')`).
- **Households**: `tbsim.HouseholdStats` summarizes transmission within households.
- **Fitting**: every parameter is an `ss.par`, so `sim.calibrate()` can fit those with bounds to data.

For examples and tutorials, see the [examples](examples.qmd) and [tutorials](../tutorials/index.md).
