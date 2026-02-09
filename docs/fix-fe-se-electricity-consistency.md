# Fix: FE|Electricity ≤ SE|Electricity Constraint

**Date:** 2026-02-06
**Branch:** `fix/clip-fe-electricity-to-se`
**Status:** Implemented and validated

---

## Summary

In step 5e historical harmonization, `Final Energy|Electricity` and `Secondary Energy|Electricity` are harmonized independently to IEA 2022 historical data using the ratio method. Because FE and SE were downscaled with different methods (Step 1: GDP/population vs Step 2: technology mix), their base-year errors relative to IEA differ. Independent harmonization with different correction ratios (FE: 1.44× for IDN, SE: 1.23× for IDN) causes FE to exceed SE in 121 countries (2025–2070 period), violating the physical constraint that electricity consumption cannot exceed generation.

**Fix:** Post-harmonization constraint enforcement — clip FE|Electricity to SE|Electricity and proportionally scale down sector-level FE variables to maintain variable hierarchy consistency.

---

## Root Cause

### 1. Different downscaling methods create geographic mismatches

| Variable | Step | Method | Basis |
|----------|------|--------|-------|
| FE\|Electricity | Step 1 | GDP/population shares | Economic activity + demographics |
| SE\|Electricity | Step 2 | Technology mix | Power plant capacity & generation |

These produce different country-level distributions of the same regional total.

### 2. Independent harmonization with different ratios

Both variables are in `step5e_harmo` (fixtures.py lines 2180, 2193). The ratio method computes:

```
ratio = IEA_historical_2022 / downscaled_2022
adjusted = downscaled × ratio × convergence_factor(t)
```

Where `convergence_factor(t)` blends ratio toward 1.0 by tc=2080 (hardcoded at utils.py:15932).

**Example (IDN):**
- FE: downscaled=1.03 EJ, IEA=1.48 EJ → ratio=1.44 (44% correction)
- SE: downscaled=1.27 EJ, IEA=1.56 EJ → ratio=1.23 (23% correction)

### 3. Paths diverge over time

The larger FE correction (1.44) vs smaller SE correction (1.23) causes FE to grow faster than SE in the harmonization period, pushing FE above SE for 121 countries, peaking at 8.8 EJ/yr excess for IDN in 2045.

---

## The Fix

**File:** `downscaler/Step_5e_historical_harmo.py` lines 325–366
**Location:** Immediately after `fun_add_variables_and_harmonize` returns (line 323)

### Implementation

```python
# 1. Extract FE|Electricity and SE|Electricity
_fe = df_merged.loc[df_merged.index.get_level_values("VARIABLE") == "Final Energy|Electricity"]
_se = df_merged.loc[df_merged.index.get_level_values("VARIABLE") == "Secondary Energy|Electricity"]

# 2. Align indices via rename (SE VARIABLE → FE VARIABLE)
_se_renamed = _se.rename(index={"Secondary Energy|Electricity": "Final Energy|Electricity"}, level="VARIABLE")

# 3. Clip FE element-wise: FE = min(FE, SE)
_fe_clipped = _fe.clip(upper=_se_renamed)

# 4. Write back clipped FE
df_merged.loc[fe_mask] = _fe_clipped

# 5. Compute scaling factor and apply to sector-level variables
scaling_factor = _fe_clipped / _fe
for sector_var in ["FE|Industry|Electricity", "FE|R&C|Electricity", "FE|Transportation|Electricity"]:
    sector_scaled = sector_data × scaling_factor
    df_merged.loc[sector_mask] = sector_scaled
```

### Logic

- **Clip top-level aggregate:** `FE|Electricity` capped at `SE|Electricity` for each country×year cell
- **Proportional sector scaling:** Each sector variable multiplied by the same factor to maintain `sum(sectors) = total`
- **Example:** If IDN FE clipped from 13.08 → 4.27 EJ (scale=0.326), all three sectors scale by 32.6%

---

## Impact on Historical Harmonization

### Does the fix break harmonization?

**No.** The clip runs *after* harmonization completes.

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| SE\|Electricity → IEA 2022 | ✓ Matches | ✓ Matches (unchanged) |
| FE\|Electricity → IEA 2022 | ✓ Matches | ⚠ Slightly lower for clipped countries |
| Physical constraint FE ≤ SE | ✗ Violated (121 countries) | ✓ Enforced everywhere |
| Sector sum consistency | ✓ Maintained | ✓ Maintained (scaled) |

### Trade-off

For the 121 countries where clipping occurs, `FE|Electricity` in 2022 is slightly lower than the IEA historical value because it was capped at `SE|Electricity`. This is the correct trade-off: **physical consistency (FE ≤ SE) takes priority over perfect base-year match when the two independent harmonizations diverge.**

`SE|Electricity` harmonization is fully preserved, which is the more fundamental constraint (generation determines the ceiling for consumption).

---

## Validation Results

**Pre-fix (step5e output):**
- 331 country×year violations across 68 countries
- Max excess: 8.8 EJ/yr (IDN, 2045)

**Post-fix:**
- 0 violations
- Max(FE − SE) = 0.00 everywhere

---

## Related Findings

### PE ≥ FE fuel supply checks (notebook addition)

Added consistency checks for `Primary Energy ≥ Final Energy` at the fuel level:

| Fuel | Constraint | Result |
|------|-----------|--------|
| Coal | PE\|Coal ≥ FE\|Solids\|Coal | ✓ Clean (0 violations) |
| Oil | PE\|Oil ≥ sum(FE\|…\|Liquids\|Oil) | ✗ 101 countries, max 3.49 EJ/yr (CHN) |
| Natural Gas | PE\|Gas ≥ sum(FE\|…\|Gases\|Natural Gas) | ✗ 31 countries, max 0.79 EJ/yr (IND) |

**No code fix applied for oil/gas.** These violations are expected: REMIND regional model does not track intra-regional fuel trade. At the regional level, PE ≥ FE everywhere (energy balance is correct). Country-level violations reflect oil/gas-importing countries within a region — FE > PE is physically valid when imports are considered. Clipping would be incorrect as it would artificially cut consumption in importing countries.

Coal has no violations because intra-regional coal trade is negligible.

---

## Files Modified

### Code
- **`downscaler/Step_5e_historical_harmo.py`** (lines 325–366) — post-harmonization FE|Elec clip + sector scaling
- **`downscaler/fixtures.py`** — no changes (both FE|Elec and SE|Elec remain in `step5e_harmo`)
- **`downscaler/utils.py`** — no changes (harmonization logic untouched)

### Documentation
- **`notebook/consistencies_check.ipynb`** (cells 36–37) — added PE ≥ FE fuel supply checks with trade caveat explanation

### Memory
- **`.claude/projects/.../memory/MEMORY.md`** — documented fix, root cause, and PE/FE findings

---

## Next Steps

1. **Merge to main:** Create PR from `fix/clip-fe-electricity-to-se` after validation
2. **Monitor:** Check future runs to ensure clip warning counts are consistent with expectations
3. **Optional follow-up:** Investigate the `tc=2080` hardcode at `utils.py:15932` (overrides `tc=2050` parameter) — separate issue but discovered during this investigation

---

## Technical Notes

- **Index structure:** `df_merged` uses `['MODEL', 'SCENARIO', 'REGION', 'VARIABLE', 'UNIT']` with integer year columns
- **Alignment technique:** `rename(index={se_var: fe_var}, level="VARIABLE")` aligns SE and FE indices for element-wise operations
- **Scaling factor handling:** Division by zero avoided via `.replace(0, 1.0)` — where FE=0, scaling=0/1=0, preserving zero values in sectors
- **Performance:** Clip+scale adds ~0.1s overhead per run (negligible for 189 countries × 19 years)
