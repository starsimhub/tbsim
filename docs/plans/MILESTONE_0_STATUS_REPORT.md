# Milestone 0 Status Report

**Source:** [StarsimHub Project 24](https://github.com/orgs/starsimhub/projects/24) — **M0: Streamlined model for Research Team to build on**  
**Tracking repo:** [starsimhub/tbsimV2](https://github.com/starsimhub/tbsimV2) (milestone [M0](https://github.com/starsimhub/tbsimV2/milestone/2), due 2026-02-27)  
**Report date:** 2026-02-10  
**Repository context:** v2tbsim (tbsim / TB-ACF work)

---

## 1. Milestone 0 task list (tbsimV2)

All issues below are in [tbsimV2](https://github.com/starsimhub/tbsimV2) and, per API, **open** (M0: 29 open, 0 closed). Implementation notes reflect the **v2tbsim** codebase where applicable.

| # | Task | Issue | Status | Implementation notes |
|---|------|--------|--------|----------------------|
| 46 | Move the London School module into TBsim | [tbsimV2#46](https://github.com/starsimhub/tbsimV2/issues/46) | Open | **In progress (v2tbsim):** `tbsim/models/tb_lshtm.py` contains TB_LSHTM and TB_LSHTM_Acute; merge/PR into tbsim/tbsimV2 pending. |
| 47 | Set-up project document repository (Sharepoint) | [tbsimV2#47](https://github.com/starsimhub/tbsimV2/issues/47) | Open | Org/project setup. |
| 32 | Organize/conduct design bake-off | [tbsimV2#32](https://github.com/starsimhub/tbsimV2/issues/32) | Open | Process. |
| 45 | Identify current best docs | [tbsimV2#45](https://github.com/starsimhub/tbsimV2/issues/45) | Open | Documentation audit. |
| 33 | Verify constants are not conflicting (ref 2.2.1) | [tbsimV2#33](https://github.com/starsimhub/tbsimV2/issues/33) | Open | Constants/parameter review. |
| 34 | Implement age stratification (ref 2.2.6) | [tbsimV2#34](https://github.com/starsimhub/tbsimV2/issues/34) | Open | Age-stratified rates/logic. |
| 35 | Verify priority and design (ref 2.2.7) | [tbsimV2#35](https://github.com/starsimhub/tbsimV2/issues/35) | Open | Priority/design verification. |
| 36 | Determine which network fits requirements – follow-up w/Jamie (ref 3.1.1) | [tbsimV2#36](https://github.com/starsimhub/tbsimV2/issues/36) | Open | Network choice; external follow-up. |
| 38 | Add Symptomatic TB infectiousness to LSHTM (ref REQ-3.2.2) | [tbsimV2#38](https://github.com/starsimhub/tbsimV2/issues/38) | Open | **Done in v2tbsim:** SYMPTOMATIC and ASYMPTOMATIC are infectious; `rel_trans` = 1 for symptomatic, `kappa` for asymptomatic. |
| 39 | Add modification by smear status to LSHTM (ref REQ-3.2.3) | [tbsimV2#39](https://github.com/starsimhub/tbsimV2/issues/39) | Open | **Not in codebase:** No explicit smear+/smear− modifier on infectiousness yet (only ASYMPTOMATIC vs SYMPTOMATIC). |
| 40 | Add modification by cough presence to LSHTM (ref REQ-3.2.4) | [tbsimV2#40](https://github.com/starsimhub/tbsimV2/issues/40) | Open | **Not in codebase:** No cough attribute or infectiousness modifier. |
| 37 | Review current implementation w/Jamie. Meets reqs? (ref 3.2.5) | [tbsimV2#37](https://github.com/starsimhub/tbsimV2/issues/37) | Open | Review with Jamie. |
| 5 | tb_lshtm lacks core rates (rr_* parameters) | [tbsimV2#5](https://github.com/starsimhub/tbsimV2/issues/5) | Open | **Done in v2tbsim:** `rr_activation`, `rr_clearance`, `rr_death` in `tb_lshtm.py`; BCG/malnutrition can use same interface. |
| 7 | Refactor TBTreatment for compatibility with multiple TB model variants | [tbsimV2#7](https://github.com/starsimhub/tbsimV2/issues/7) | Open | Design in `TB_INTERVENTION_INTERFACE_ISSUE_10.md`; refactor not yet done. |
| 8 | Duplicate HealthSeekingBehavior implementations with np.random anti-pattern | [tbsimV2#8](https://github.com/starsimhub/tbsimV2/issues/8) | Open | Use Starsim RNG; consolidate duplicates. |
| 9 | HealthSeekingBehavior hardcodes TB module states – breaks TB_LSHTM compatibility | [tbsimV2#9](https://github.com/starsimhub/tbsimV2/issues/9) | Open | Same direction as #10: biophysical API. |
| 10 | Diagnostic interventions check TB states directly – need biophysical state API | [tbsimV2#10](https://github.com/starsimhub/tbsimV2/issues/10) | Open | Design in `TB_INTERVENTION_INTERFACE_ISSUE_10.md`; API not yet implemented. |
| 11 | Improve state organization: move diagnostic tracking from person to intervention level | [tbsimV2#11](https://github.com/starsimhub/tbsimV2/issues/11) | Open | Design in `TB_DIAGNOSTICS_IMPLEMENTATION_PLAN.md`. |
| 48 | Update README to reference the new disease model | [tbsimV2#48](https://github.com/starsimhub/tbsimV2/issues/48) | Open | Pending LSHTM merge. |
| 49 | Setup design bake off to discuss how models / diseases are structured in the codebase | [tbsimV2#49](https://github.com/starsimhub/tbsimV2/issues/49) | Open | Overlaps #32; structure of models/diseases. |
| 50 | Ensure we have documentation for current state of TB history | [tbsimV2#50](https://github.com/starsimhub/tbsimV2/issues/50) | Open | Assignee: gfmosoti. |
| 51 | REQ-2.2.7a: Verify nutrition stratification coverage in TB disease progression | [tbsimV2#51](https://github.com/starsimhub/tbsimV2/issues/51) | Open | Verification; blocks #52. |
| 52 | REQ-2.2.7b: Resolve priority and design for nutrition-stratified TB progression | [tbsimV2#52](https://github.com/starsimhub/tbsimV2/issues/52) | Open | Blocked by #51. |

---

## 2. Repository context

- **tbsimV2** milestone **M0** ([milestone/2](https://github.com/starsimhub/tbsimV2/milestone/2)): *M0: Streamlined model for Research Team to build on* — due **2026-02-27**; 29 open, 0 closed.
- **v2tbsim** (this repo): branch `london-model`; `tbsim/models/tb_lshtm.py` modified; `docs/plans/` contains TB diagnostics and intervention-interface plans.

---

## 3. Summary by theme

| Theme | Issues | Status |
|-------|--------|--------|
| **London/LSHTM in TBsim** | 46, 48 | Module in v2tbsim; merge + README pending. |
| **LSHTM requirements (infectiousness)** | 38, 39, 40, 37 | #38 done (symptomatic infectiousness); #39 smear, #40 cough not implemented. |
| **Core rates (rr_*)** | 5 | Implemented in v2tbsim `tb_lshtm.py`. |
| **Interventions & diagnostics API** | 7, 8, 9, 10, 11 | Design docs exist; refactor and biophysical API not implemented. |
| **Documentation & process** | 45, 47, 50, 32, 49 | Open. |
| **Parameters & stratification** | 33, 34, 35, 51, 52 | Constants, age, nutrition verification/design open. |
| **Network & review** | 36, 37 | External (Jamie). |

---

## 4. Suggested next steps

1. **Merge London module (#46):** PR from v2tbsim into tbsim/tbsimV2; then close #5, #38 in tbsimV2 when merged, and update README (#48).
2. **LSHTM reqs:** Implement smear (#39) and cough (#40) modifiers if required; schedule review with Jamie (#37).
3. **Intervention API:** Implement biophysical queries (`is_active`, `is_smear_positive`, etc.) and refactor TBTreatment (#7), HealthSeekingBehavior (#9), and diagnostics (#10, #11).
4. **Process:** Run design bake-off (#32 / #49); set up Sharepoint doc repo (#47); identify best docs (#45).
5. **Stratification & constants:** Verify constants (#33), age stratification (#34), priority/design (#35), nutrition (#51, #52).
