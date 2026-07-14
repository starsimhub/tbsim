# Resistance technical-spec compliance review

Date: 2026-07-14  
Spec reviewed: `tbsim/resistance/docs/tbsim-resistance-tech-spec-new.md`  
Implementation reviewed: `tbsim/resistance/*.py`, `tests/test_resistance.py`, and
`tbsim/resistance/devtests/`

## Executive summary

The implementation passes 10 of the 13 auditable sections. The core model—strain
enumeration, multistrain carriage and counts, transmission, superinfection,
progression, clearance, endogenous acquisition, and strain-aware TPT—is present
and well tested. The three FAIL verdicts concern incomplete product/integration
interfaces and missing requested validation evidence, rather than the central
transmission or natural-history mechanics.

| # | Spec section | Verdict |
|---:|---|:---:|
| 1 | Individual strain resistance profiles | PASS |
| 2 | Multi-strain infections | PASS |
| 3 | Transmission | PASS |
| 4 | Strain competition and protection against reinfection | PASS |
| 5 | Progression to disease | PASS |
| 6 | Clearance | PASS |
| 7 | Random/endogenous acquisition | PASS |
| 8 | Treatment and selective acquisition | **FAIL** |
| 9 | TPT | PASS |
| 10 | Diagnostics and treatment modification | **FAIL** |
| 11 | Treatment monitoring | PASS |
| 12 | Notation/API | PASS |
| 13 | Requested testing | **FAIL** |

## Detailed verdicts

### 1. Individual strain resistance profiles — PASS

`Strains.__init__` accepts an arbitrary ordered drug list, creates `m = 2**n`,
and builds the binary `(m,n)` phenotype matrix (`strains.py:42-65`). Drug names
are not hard-coded. For example, `Strains(['RIF','BDQ','FQ']).profile[5]` is
`[True, False, True]`, the spec's RIF+FQ profile.

### 2. Multi-strain infections — PASS

Each agent has a `strain_mask`, while one `strain_count_j` state per phenotype
stores repeated identical infections (`tb_resistant.py:89-95`). A mask can carry
any subset of the enumerated strains, and `Strains.carried()` decodes it
(`strains.py:99-110`). There is no two-strain cap.

### 3. Transmission — PASS

Fitness is the product of costs for all resisted drugs (`strains.py:63-65`). The
overall relative transmissibility is the maximum fitness among carried strains
(`tb_resistant.py:421-424`), while the transmitted phenotype is drawn in
proportion to `count * fitness` (`strains.py:116-136`,
`tb_resistant.py:228-232`). The target receives exactly that founding strain with
count one (`tb_resistant.py:247-264`), so transmission cannot create a phenotype
absent from the source. Existing evidence includes
`devtests/test_transmission.py::test_transmission_split_matches_spec_table` and
`::test_superinfection_does_not_reduce_infectiousness`.

### 4. Strain competition and protection against reinfection — PASS

The default coupling is correct: `rr_reinfection_inf` inherits
`rr_reinfection_rec`, and `rr_reinfection_non` inherits it
(`tb_resistant.py:67-87`). INFECTED and NON_INFECTIOUS are eligible with their
state-specific relative risks; ASYMPTOMATIC and SYMPTOMATIC default to zero and
TREATMENT is ineligible (`tb_resistant.py:382-418`). Identical-strain exposure
increments its count and resets `ti_infected` (`tb_resistant.py:234-245`). Counts
weight only transmission and the `p_multi < 1` bottleneck; DST, treatment, and
acquisition operate on phenotype membership, coupling identical copies as the
spec requires. The counter behavior is covered throughout
`devtests/test_counter.py`.

### 5. Progression to disease — PASS

Natural history remains one agent-level `TBS` state. Default progression and
clearance multipliers for multistrain agents are one, so rates do not change with
strain number (`tb_resistant.py:72-74`, `step_transitions`). All strains persist
on transition to NON_INFECTIOUS. On entry to ASYMPTOMATIC, `_bottleneck()` keeps
all strains with `p_multi` (default one), otherwise selects one in proportion to
the identical-strain counts (`tb_resistant.py:314-348`). Every successful
exposure resets `ti_infected`. The spec makes a time-varying hazard conditional
(“If we implement”); no hazard shape is specified, so its absence is not scored
as a failure of this document.

### 6. Clearance — PASS

Natural INFECTION→CLEARED and NON_INFECTIOUS→CLEARED transitions zero the mask and
all counts (`tb_resistant.py:285-311`). Strain number does not alter clearance at
the defaults (`rr_clear_super=1`). Death also clears all strain state. Covered by
`devtests/test_natural_history.py::test_natural_clearance_removes_all_strains`
and counter reset tests.

### 7. Random/endogenous acquisition — PASS

`_denovo()` runs once on INFECTION→NON_INFECTIOUS or
INFECTION→ASYMPTOMATIC, independently by carried phenotype and susceptible drug
(`tb_resistant.py:270-321,350-386`). Multiple drugs and multiple strains can
mutate in the same event. Both spec-permitted modes are exposed (`mixed` and
`replacement`), and identical copies are coupled because the draw is per distinct
phenotype, not per count. Existing tests cover mode choice, per-drug specificity,
multiple strains, and count independence in `devtests/test_acquisition.py`.

### 8. Treatment and selective acquisition — FAIL

Much is correct: cure is evaluated per distinct carried strain, one agent-level
Bernoulli adherence draw correlates outcomes, successful phenotype cure removes
all identical copies, partially cured agents retain state and surviving strains,
and failure-time acquisition is per regimen drug, state-scaled, and replacement
(`treatments.py:34-135,300-329`).

Two specified interfaces remain incomplete:

1. The spec defines an arbitrary efficacy vector `T_l={t_1,l,...,t_m,l}`. `TxR`
   instead derives the vector as `base_efficacy * product(resist_penalty)`
   (`treatments.py:74-82`). This cannot express every vector (with two drugs,
   `t_11` is constrained by the two single-resistance penalties), nor is an
   explicit vector accepted.
2. The requested regimen-level adherence *distribution* is not supported as a
   source of per-agent adherence values that modify efficacy across all strains.
   The API accepts one probability and turns it into a binary completion draw
   (`treatments.py:85-101`); it has no agent-varying adherence-value interface.

Failing demonstrations:
`test_spec_gaps.py::test_treatment_accepts_an_explicit_per_strain_efficacy_vector`
and `::test_treatment_accepts_an_agent_varying_adherence_distribution`.

### 9. TPT — PASS

`TPTRx` clears only regimen-susceptible phenotypes and leaves resistant strains
able to progress/transmit (`tpt.py:73-77,131-154`). Progression protection is
coverage-weighted (`tpt.py:164-191`). Ineffective TPT can acquire resistance by
regimen drug with a configurable state gradient (`tpt.py:52-71,106-129`). Tests
cover unmasking, acquisition, and the state gradient in
`devtests/test_diagnostics_tpt.py`.

### 10. Diagnostics and treatment modification — FAIL

DST itself is substantially compliant: sensitivity/specificity are applied at
the strain level, each phenotype has an observation bottleneck (default fitness),
results aggregate to one drug-bit profile rather than strain identities, and
counts do not affect calls (`dst.py:35-87`). Results route regimens through
`observed_resistant()`/`matches()` (`dst.py:138-188`). Eligibility can be immediate
or supplied by any callable.

The explicit requirement to track time since prior treatment so later treatment
can be classified as failure versus a new case is not implemented. A
`TxDeliveryR` records start/end arrays local to that delivery, but there is no
durable, cross-regimen treatment-history state or eligibility/classification API.
The existing `will_fail()` is an oracle for a pre-rolled *ongoing* outcome, not a
later failure-versus-new-case classifier. DST outcomes are also binary only; the
suggested indeterminate outcome is absent, though that wording is less normative.

Failing demonstration:
`test_spec_gaps.py::test_treatment_history_can_classify_failure_versus_new_case`.

### 11. Treatment monitoring — PASS

`treatment_monitoring_eligibility()` gates on time under treatment and composes
with diagnostics (`treatments.py:385-420`). `TxDeliveryR.interrupt()` and
`supersedes` allow a new product to stop and replace an ongoing regimen. Tests
`test_treatment_monitoring_switches_regimen` and
`test_monitoring_require_and_will_fail` demonstrate the workflow.

### 12. Notation/API — PASS

The name-keyed `drugs`/`drug_idx` mapping maintains stable correspondence while
bit positions provide compact profiles (`strains.py:42-50`). Public products
validate drug names and fail fast on typos. This is an allowed divergence under
the spec's notation section.

### 13. Requested testing — FAIL

The suite has extensive mechanistic, directional, and ABM-versus-ODE tests. It
tests acquisition, fitness, relative treatment efficacy, treatment selection,
single-strain reduction, and resistance fractions. However, the specifically
requested resistance-off versus resistance-on comparison of active prevalence,
new asymptomatic incidence, and TB mortality per 100,000 using a plausible or
`tb_LAI_TPT`-derived configuration has not been produced. There is no automated
table/artifact that permits review of whether resistance materially shifts those
three burden outcomes, and the directional suite does not comprehensively report
both overall burden and resistance percentage for every named lever (including
overall treatment rate).

Failing demonstration:
`test_spec_gaps.py::test_requested_lai_tpt_burden_validation_artifact_exists`.

## Verification commands and interpretation

The normal regression suite should pass. The gap file should fail until these
features/artifacts exist. Keeping the two commands separate prevents known-gap
acceptance tests from making the established regression suite unusable.
