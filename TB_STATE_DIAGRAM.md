# TB State Transition Diagram

## State Definitions
```
TBS.NONE = -1           # No TB infection (susceptible)
TBS.LATENT_SLOW = 0     # Latent TB with slow progression
TBS.LATENT_FAST = 1     # Latent TB with fast progression  
TBS.ACTIVE_PRESYMP = 2  # Active TB in pre-symptomatic phase
TBS.ACTIVE_SMPOS = 3    # Active TB, smear positive (most infectious)
TBS.ACTIVE_SMNEG = 4    # Active TB, smear negative (moderately infectious)
TBS.ACTIVE_EXPTB = 5    # Active TB, extra-pulmonary (least infectious)
TBS.DEAD = 8            # Death from TB
TBS.PROTECTED = 100     # Protected from TB (e.g., from BCG vaccination)
```

## State Transition Flow

```
                    ┌─────────────────┐
                    │   SUSCEPTIBLE   │
                    │   (TBS.NONE)    │
                    └─────────┬───────┘
                              │
                              │ Infection
                              │ (beta parameter)
                              ▼
                    ┌─────────────────┐
                    │   LATENT TB     │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │ LATENT_SLOW │ │ ──┐
                    │ │ (rate_LS)   │ │   │
                    │ └─────────────┘ │   │
                    │                 │   │
                    │ ┌─────────────┐ │   │
                    │ │ LATENT_FAST │ │ ──┼──┐
                    │ │ (rate_LF)   │ │   │  │
                    │ └─────────────┘ │   │  │
                    └─────────────────┘   │  │
                              │           │  │
                              │           │  │
                              ▼           │  │
                    ┌─────────────────┐   │  │
                    │  PRE-SYMPTOMATIC│   │  │
                    │ (ACTIVE_PRESYMP)│   │  │
                    └─────────┬───────┘   │  │
                              │           │  │
                              │           │  │
                    ┌─────────┴───────┐   │  │
                    │                 │   │  │
                    ▼                 ▼   │  │
            ┌─────────────┐   ┌─────────────┐ │  │
            │   CLEAR     │   │   ACTIVE    │ │  │
            │ (TBS.NONE)  │   │    TB       │ │  │
            │             │   │             │ │  │
            │ (if treated)│   │ ┌─────────┐ │ │  │
            └─────────────┘   │ │ SMPOS   │ │ │  │
                              │ │ (rate)  │ │ │  │
                              │ └─────────┘ │ │  │
                              │             │ │  │
                              │ ┌─────────┐ │ │  │
                              │ │ SMNEG   │ │ │  │
                              │ │ (rate)  │ │ │  │
                              │ └─────────┘ │ │  │
                              │             │ │  │
                              │ ┌─────────┐ │ │  │
                              │ │ EXPTB   │ │ │  │
                              │ │ (rate)  │ │ │  │
                              │ └─────────┘ │ │  │
                              └─────────────┘ │  │
                                      │       │  │
                                      │       │  │
                              ┌───────┴───────┴──┴──┐
                              │                     │
                              ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │   CLEAR         │   │     DEATH       │
                    │ (TBS.NONE)      │   │   (TBS.DEAD)    │
                    │                 │   │                 │
                    │ (natural or     │   │ (from active    │
                    │  treatment)     │   │  TB states)     │
                    └─────────────────┘   └─────────────────┘
```

## Transition Rates and Probabilities

### 1. Latent to Pre-symptomatic
- **Slow Progressors**: `rate_LS_to_presym` (3e-5 per day)
- **Fast Progressors**: `rate_LF_to_presym` (6e-3 per day)
- **Method**: `p_latent_to_presym()`

### 2. Pre-symptomatic Transitions
- **To Clear**: `p_presym_to_clear()` (only if on treatment)
- **To Active**: `rate_presym_to_active` (3e-2 per day)
- **Method**: `p_presym_to_active()`

### 3. Active TB Transitions
- **To Clear**: 
  - Natural: `rate_active_to_clear` (2.4e-4 per day)
  - Treatment: `rate_treatment_to_clear` (6 per year)
- **To Death**:
  - Smear Positive: `rate_smpos_to_dead` (4.5e-4 per day)
  - Smear Negative: `rate_smneg_to_dead` (0.3 * 4.5e-4 per day)
  - Extra-pulmonary: `rate_exptb_to_dead` (0.15 * 4.5e-4 per day)

## Key Features

### Individual Risk Factors
- `rr_activation`: Multiplier for latent-to-active transition
- `rr_clearance`: Multiplier for active-to-clearance transition
- `rr_death`: Multiplier for active-to-death transition

### Treatment Effects
- Reduces transmission: `rel_trans_treatment` (0.5)
- Prevents death: `rr_death = 0` for treated individuals
- Increases clearance: Uses `rate_treatment_to_clear`

### Transmission Heterogeneity
- `reltrans_het`: Individual-level heterogeneity in infectiousness
- Applied to all infectious states

## Recent Changes Impact

The recent refactoring changed how these transitions are calculated:

**Before**: Used Starsim's `filter()` method
```python
new_presymp_uids = self.p_latent_to_presym.filter(latent_uids)
```

**After**: Explicit probability calculation
```python
probs = self.p_latent_to_presym(self.sim, latent_uids)
transition_mask = np.random.random(len(probs)) < probs
new_presymp_uids = latent_uids[transition_mask]
```

This change provides:
1. More explicit control over probability calculations
2. Better handling of individual risk factors
3. More robust error handling for empty arrays
4. Consistent pattern across all transitions
