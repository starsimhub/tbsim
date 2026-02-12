# TB Disease Module and Intervention Module: Shared Interface (Issue #10)

**Reference:** [GitHub Issue #10 – Diagnostic interventions check TB states directly](https://github.com/starsimhub/tbsimV2/issues/10)  
**Complements:** [TB_DIAGNOSTICS_IMPLEMENTATION_PLAN.md](TB_DIAGNOSTICS_IMPLEMENTATION_PLAN.md) (test classifications and DiagnosticStateMapper)

---

## 1. Problem Summary

- **Diagnostic interventions** (e.g. `EnhancedTBDiagnostic`, `CaseFinding`, `TBDiagnostic`) currently read `sim.diseases.tb.state[uids]` and branch on **TBS** enum values (e.g. `TBS.ACTIVE_SMPOS`, `TBS.ACTIVE_SMNEG`) to choose test sensitivity.
- That **ties interventions to one TB model** and **does not work with TB_LSHTM**, which uses a different state set (**TBSL**: UNCONFIRMED, ASYMPTOMATIC, SYMPTOMATIC, etc.).
- The need is a **solution that works for both the TB disease module and the intervention module**: the disease module exposes the information interventions need, and interventions use that instead of reading internal state enums.

---

## 2. Recommended Design: Disease Module Exposes Biophysical Queries; Intervention Applies Test Sensitivity

- **TB disease module** provides a small set of **queries** that describe "biology" and do **not** depend on which test is used:
  - e.g. "Is this person smear-positive (by this model's definition)?", "Has active TB?", "Extra-pulmonary?"
- **Intervention module** holds test-specific sensitivity/specificity and applies them to those answers (e.g. P(positive) = sensitivity if truly positive, 1−specificity if truly negative).

So: **disease module = biology; intervention module = test characteristics.** Both sides stay consistent whether you use the current TB model or TB_LSHTM.

---

## 3. What the TB Module Should Provide (for Interventions to Use)

Any TB disease module that is used with these interventions should provide the same set of **methods** so interventions can work without knowing TBS or TBSL. Two ways to do this:

- **Option A – Duck typing:** Document the required methods; TB and TB_LSHTM implement them. No new base class.
- **Option B – Shared base:** e.g. a base class with these methods; TB and TB_LSHTM subclass it.

Suggested method set:

```python
# e.g. in tbsim/models/natural_history.py and tbsim/models/lshtm.py
# (or a small shared module that defines the contract)

def is_active(self, uids) -> np.ndarray:
    """
    True for agents with active TB (eligible for active TB tests / treatment).
    Shape: (len(uids),), dtype bool.
    """

def is_smear_positive(self, uids) -> np.ndarray:
    """
    True for agents who are "smear positive" in this model's definition.
    Used by interventions to apply smear-based test sensitivity.
    Shape: (len(uids),), dtype bool.
    """

def is_extra_pulmonary(self, uids) -> np.ndarray:
    """
    True for agents with extra-pulmonary TB. Used for EPTB-specific sensitivity.
    Shape: (len(uids),), dtype bool.
    """

def get_diagnostic_group(self, uids) -> np.ndarray:
    """
    Optional: one integer per agent (e.g. 0=no TB, 1=presymp, 2=smear+, 3=smear-, 4=EPTB).
    Interventions can map group -> sensitivity without depending on TBS/TBSL.
    dtype int.
    """
```

- **`is_active(uids)`** – Used by diagnostics ("has active TB") and by treatment/TPT (eligibility). Replaces direct checks on `TBS.ACTIVE_*` / TBSL active states.
- **`is_smear_positive(uids)`** – Intervention uses this plus its own test sensitivity (e.g. 0.7 for LED FM). Replaces `tb_state == TBS.ACTIVE_SMPOS`.
- **`is_extra_pulmonary(uids)`** – For EPTB-specific sensitivity. Replaces `tb_state == TBS.ACTIVE_EXPTB`.
- **`get_diagnostic_group(uids)`** – Optional: one integer per agent so interventions can do `sensitivity[group]` without knowing TBS/TBSL. Fits with DiagnosticGroup in the existing diagnostics plan.

**Design choice:** Test-specific sensitivity stays in the **intervention**. The **disease module** does not need to know about "Xpert" or "FujiLAM"; it only exposes biology (active, smear pos/neg, EPTB, and optionally diagnostic group).

---

## 4. Implementation per Disease Model

### 4.1 Natural history TB model (`tbsim/models/natural_history.py`)

- **`is_active(uids)`**  
  `(state == ACTIVE_SMPOS) | (state == ACTIVE_SMNEG) | (state == ACTIVE_EXPTB) | (state == ACTIVE_PRESYMP)` (or whatever "active" means for diagnostics; can exclude PRESYMP if needed).
- **`is_smear_positive(uids)`**  
  `state[uids] == TBS.ACTIVE_SMPOS`.
- **`is_extra_pulmonary(uids)`**  
  `state[uids] == TBS.ACTIVE_EXPTB`.
- **`get_diagnostic_group(uids)`**  
  Map TBS to a small integer (e.g. NONE=0, PRESYMP=1, SMPOS=2, SMNEG=3, EXPTB=4). Same semantics as current `_get_diagnostic_parameters(tb_state)` but implemented inside the disease module.

### 4.2 TB_LSHTM (`tbsim/models/lshtm.py`)

- **`is_active(uids)`**  
  `state in (TBSL.UNCONFIRMED, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC)` (and optionally TREATMENT if "active" includes on-treatment).
- **`is_smear_positive(uids)`**  
  LSHTM has no literal "smear" state. Map e.g. SYMPTOMATIC → smear-positive, UNCONFIRMED/ASYMPTOMATIC → smear-negative (or a parameterised proportion if needed).
- **`is_extra_pulmonary(uids)`**  
  LSHTM may not distinguish EPTB; return `np.zeros(len(uids), dtype=bool)` or a parameter-driven proportion if added later.
- **`get_diagnostic_group(uids)`**  
  Map TBSL to the **same** integer diagnostic groups as the current TB model so one sensitivity map in the intervention works for both disease modules.

---

## 5. Interventions to Update (Scope)

All of these currently depend on `tb.state` (and sometimes TBS) for diagnostics or eligibility. They should be refactored to use the methods above.

| Intervention | File | Current use of TB state | Change |
|-------------|------|-------------------------|--------|
| **EnhancedTBDiagnostic** | `tb_diagnostic.py` | `_get_diagnostic_parameters(uid, age, tb_state, …)`; `has_tb = np.isin(tb_states, [TBS.ACTIVE_SMPOS, ...])` | Use `tb.is_active`, `tb.is_smear_positive`, `tb.is_extra_pulmonary` (or `tb.get_diagnostic_group`); compute sensitivity per agent from that and the intervention's test parameters. |
| **TBDiagnostic** | `tb_diagnostic.py` | `has_tb = np.isin(tb_states, [TBS.ACTIVE_SMPOS, ...])` | Use `tb.is_active(selected)` (and optionally smear/EPTB for stratified sensitivity). |
| **CaseFinding** | `interventions_london_model.py` | `test_sens = {TBS.ACTIVE_SMPOS: 1, ...}`; `tb_state = sim.diseases.tb.state[uids]` in `p_pos_test`; result breakdown by presymp/smpos/smneg/exp | Use `tb.get_diagnostic_group(uids)` (or smear/EPTB) to index into a sensitivity array; result breakdown can use the same methods or a TB-model-specific helper if needed for reporting. |
| **TBTreatment** / **EnhancedTBTreatment** | `tb_treatment.py` | `active_tb = (tb.state == TBS.ACTIVE_SMPOS) \| ...` | Use `tb.is_active(uids)` for eligibility. |
| **TPT** | `tpt.py` | `no_active_tb = (tb.state != TBS.ACTIVE_*) & ...` | Use `~tb.is_active(uids)`. |
| **BCG** | `bcg.py` | Sets `tb.state[...] = TBS.PROTECTED`; analytics use raw `tb_states` | Can keep state write as-is (or add e.g. `set_protected(uids)` on the disease module later). Analytics can use the same methods for "active" counts if desired. |

**Order of work (suggestion):**

1. Add the methods on **TB** (current model).
2. Refactor **TBDiagnostic** and **EnhancedTBDiagnostic** to use these methods (no TBS in diagnostic logic).
3. Refactor **CaseFinding** to use `get_diagnostic_group` (or smear/EPTB) for sensitivity and, if needed, a small result-breakdown helper from the disease module.
4. Refactor **TBTreatment** and **TPT** to use `is_active`.
5. Implement the same methods on **TB_LSHTM** (with the chosen smear/EPTB mapping).
6. Optionally add **BCG** support (e.g. `set_protected`) and use the same methods for analytics.

---

## 6. Backward Compatibility and Fallback

- **Strict:** Require that the TB module implements the new methods; if not, raise a clear error when a diagnostic (or other) intervention uses them.
- **Fallback:** If the TB module does not implement the new methods, interventions can fall back to the current behavior (read `tb.state` and check TBS) and emit a deprecation warning. This allows gradual migration.

Recommendation: implement **fallback with deprecation** for one release, then require the new methods so TB_LSHTM and future models work with the same interventions.

---

## 7. Where Knowledge Lives

- **TB disease module:** only "biology" – who is active, smear positive, EPTB, and optionally a diagnostic group index. No test names, no test sensitivities.
- **Intervention module:** test-specific sensitivity and specificity. The intervention calls `tb.is_smear_positive(uids)`, `tb.is_extra_pulmonary(uids)`, and/or `tb.get_diagnostic_group(uids)`, then applies its own sensitivity/specificity.

This keeps the solution simple and works for both the current TB model and TB_LSHTM.

---

## 8. Summary Checklist

- [ ] Add `is_active`, `is_smear_positive`, `is_extra_pulmonary`, and optionally `get_diagnostic_group` on **TB**.
- [ ] Implement the same methods on **TB_LSHTM** (with defined smear/EPTB mapping).
- [ ] Refactor **EnhancedTBDiagnostic** and **TBDiagnostic** to use these methods only.
- [ ] Refactor **CaseFinding** to use them for sensitivity and, if needed, for result breakdown.
- [ ] Refactor **TBTreatment** and **TPT** to use `is_active`.
- [ ] Add tests: (1) TB and TB_LSHTM both provide the methods; (2) diagnostic and case-finding logic match current behavior when using the methods (then remove fallback); (3) TB_LSHTM runs with the same interventions without touching TBS.
- [ ] Document the contract in the docs and in [TB_DIAGNOSTICS_IMPLEMENTATION_PLAN.md](TB_DIAGNOSTICS_IMPLEMENTATION_PLAN.md) as the "disease–intervention interface" for diagnostics.

This addresses Issue #10: a solution that works for both the TB disease module and the intervention module, without interventions reading internal TB states directly.
