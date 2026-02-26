# Harmonisation Year Extension to 2022 — Bug Fixes

## Background

The pipeline supports a `harmonize_eea_data_until` parameter that controls how far
historical data locks model pathways (e.g. `2020` or `2022`). Despite setting this
parameter to `2022`, outputs were still harmonised to 2020. This document describes
the six bugs identified and fixed to make non-5-year harmonisation years work correctly.

---

## Fix 1 — Interpolation range excluded the right anchor

**File:** `downscaler/utils.py` — line 16262
**Function:** `fun_add_variables_and_harmonize`

### Problem

Before harmonising, the pipeline calls `fun_interpolate` to fill in non-5-year columns
(e.g. 2022 between 2020 and 2025). The interpolation range was built with a half-open
`range`, so the right anchor (2025) was excluded from the list of columns passed to
pandas — meaning pandas had no right-hand endpoint and left 2022 as `NaN`.

### Fix

```python
# Before
range(int(interp_range[0]), int(interp_range[1])),

# After
range(int(interp_range[0]), int(interp_range[1]) + 1),
```

---

## Fix 2 — Step 5e CSV output stripped the harmonisation year

**File:** `downscaler/Step_5e_historical_harmo.py` — lines 508–512
**Function:** `fun_add_variables_and_harmonize` (output saving block)

### Problem

After harmonisation, the output columns were filtered to 5-year intervals only
(`range(2010, 2105, 5)`). A harmonisation year of 2022 — which falls between model
steps — was therefore dropped from the CSV. All downstream steps (5g, 5h, 5f) read
this CSV and never saw 2022.

### Fix

```python
selcols = [x for x in selcols if x in range(2010, 2105, 5)]
# Always include the harmonisation year in the output (even if it falls
# between two 5-year model steps, e.g. 2022 between 2020 and 2025)
if harmonize_eea_data_until not in selcols:
    selcols = sorted(selcols + [harmonize_eea_data_until])
```

---

## Fix 3 — `fun_append_missing_time_index` re-added years already present

**Files:**
- `downscaler/utils.py` — line 18758
- `downscaler/utils_emissions.py` — line 883

**Functions:** `fun_append_missing_time_index` (both files)

### Problem

This function is meant to add IAM time steps that are *absent* from the country data
so the harmonisation loop can iterate over them. It used a symmetric difference (`^`)
between the data's time index and the IAM columns. When 2022 was present in the data
(after Fix 2), the symmetric difference treated it as "extra" and included it in
`time_missing`, causing it to be appended a second time — resulting in duplicate TIME
index entries and a crash:

```
ValueError: Index contains duplicate entries, cannot reshape
```

### Fix

Change from symmetric difference to set difference (IAM columns minus data columns):

```python
# Before
time_missing = list(set([t for t in num.index]) ^ (set(df_iam.columns)))

# After
time_missing = list(set(df_iam.columns) - set([t for t in num.index]))
```

Applied identically in both `utils.py` and `utils_emissions.py`.

---

## Fix 4 — IAM ratio was `NaN` for non-5-year years

**Files:**
- `downscaler/utils.py` — line 18709
- `downscaler/utils_emissions.py` — line 835

**Functions:** `fun_harmonize_df_with_IAM` (both files)

### Problem

The IAM data only has values at 5-year intervals (2020, 2025, …). When computing the
scaling ratio for country data, years like 2022 that are present in the country
DataFrame but absent from the IAM produce `NaN` ratios. Multiplying country values by
`NaN` wiped them out.

### Fix

Fill `NaN` ratios with 1 so that non-IAM years pass through unchanged:

```python
ratio = ratio.replace(np.inf, np.nan)
# For years not present in df_iam (e.g. 2022 when IAM has 5-year intervals),
# ratio is NaN — keep original values by filling with 1 (no adjustment).
ratio = ratio.fillna(1)
```

Applied identically in both `utils.py` and `utils_emissions.py`.

---

## Fix 5 — Step 5h hydrogen ratio shape mismatch

**File:** `downscaler/Step_5h_SE_Hydrogen.py` — lines 239–245

### Problem

`IAM_hydrogen_ratio` is derived from IAM data (5-year columns only). After Fix 2,
country DataFrames include 2022, giving them one extra column. Multiplying via
`.values` relied on shape alignment — with mismatched column counts this raised:

```
ValueError: Unable to coerce to DataFrame, shape must be (28, 20): given (1, 19)
```

### Fix

Reindex the IAM ratio to match the country DataFrame's columns and interpolate to fill
any gaps before using `.values`:

```python
df1_slice = fun_xs(df1, {'VARIABLE': total_var}).droplevel("VARIABLE")
IAM_hydrogen_ratio_aligned = (
    IAM_hydrogen_ratio
    .reindex(columns=df1_slice.columns)
    .interpolate(axis=1)
)
df1_hydrogen = df1_slice * IAM_hydrogen_ratio_aligned.values
```

---

## Summary table

| # | File | Line | Root cause | Fix |
|---|------|------|-----------|-----|
| 1 | `utils.py` | 16262 | `range(a, b)` excludes `b` → 2022 stays `NaN` after interpolation | `range(a, b + 1)` |
| 2 | `Step_5e_historical_harmo.py` | 508–512 | 5-year filter drops 2022 from output CSV | Append `harmonize_eea_data_until` to `selcols` if missing |
| 3 | `utils.py`, `utils_emissions.py` | 18758, 883 | Symmetric diff `^` re-adds 2022 → duplicate TIME index | Set difference `-` (IAM minus data) |
| 4 | `utils.py`, `utils_emissions.py` | 18709, 835 | NaN ratio for 2022 wipes country values | `ratio.fillna(1)` |
| 5 | `Step_5h_SE_Hydrogen.py` | 239–245 | `.values` shape mismatch when IAM ratio lacks 2022 column | `.reindex(columns=...).interpolate(axis=1)` before `.values` |

---

## Affected pipeline steps

- **Step 5e** — produces the harmonised country CSV; Fixes 1, 2 apply here
- **Step 5g** — reads step 5e output, applies country→IAM harmonisation; Fixes 3, 4 (`utils.py`) apply here
- **Step 5h** — reads step 5e output, computes hydrogen shares; Fix 5 applies here
- **Step 5f** — reads step 5e output, applies emissions harmonisation; Fixes 3, 4 (`utils_emissions.py`) apply here
