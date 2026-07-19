# Codex resistance-spec review artifacts

- `spec_compliance_review.md` is the section-by-section audit against
  `docs/tbsim-resistance-tech-spec-new.md`.
- `test_spec_gaps.py` contains intentionally failing acceptance tests for every
  section receiving a FAIL verdict. They are deliberately not marked `xfail`, so
  they become ordinary passing regression tests when the gaps are implemented.

Run only the gap demonstrations with:

```bash
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba-cache \
python -m pytest tbsim/resistance/codex_review/test_spec_gaps.py -q
```

Run the existing resistance regression suite with:

```bash
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba-cache \
python -m pytest tests/test_resistance.py tbsim/resistance/devtests -q
```

