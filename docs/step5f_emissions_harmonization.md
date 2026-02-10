# Step 5f: Emissions Harmonization ("Emissions Clock")

## Overview

Step 5f harmonizes country-level CO2 emissions sub-sectors so that they:

1. Are **consistent with IAM regional totals** (sectorial breakdown is preserved)
2. **Match IEA historical observations** at the base year (country-level fit)

It does this through a **three-stage** process, applied sequentially. Each stage builds on the output of the previous one.

---

## The Three Stages

### Stage 1 — Regional IAM Harmonization (PRIMAP)

**Goal:** Adjust IAM regional emissions trajectories to match PRIMAP historical data.

**How it works:**
- For each IAM region (e.g. `REMIND|EUR`, `REMIND|AFR`), the function `fun_harmonize_df_iam_with_hist_data` shifts the IAM's emissions to match PRIMAP historical observations at the base year.
- The shift is blended out linearly: at `tc=2050`, the data returns to the original IAM trajectory.
- This is applied to `Emissions|CO2|Energy` and its sub-sectors (Transportation, Industry, Heat, Electricity, Residential & Commercial).

**Why:** IAM models don't perfectly match observed historical emissions. This step corrects the regional-level discrepancy before downscaling to countries.

### Stage 2 — Sectorial Consistency (`run_sector_harmo_enhanced_iamc`)

**Goal:** Scale country-level sub-sector emissions so that the sum across countries equals the (now harmonized) IAM regional total.

**How it works:**
- For each (region, scenario) pair, `run_sector_harmo_enhanced_iamc` takes the country-level data (`df`) and the regional IAM target (`df_iam`) and applies a proportional scaling.
- It uses three dictionaries (`step5f_dict1`, `step5f_dict2`, `step5f_dict3`) that define parent-child variable relationships, ensuring that sub-sectors sum to their parent.
- Output format quirks: adds a `METHOD` index level, renames `SCENARIO` to `TARGET`, appends `LONG_TERM` to variable names, and converts year columns to string floats (e.g. `'2010.0'`).

**Why:** After GDP/population-based downscaling, the sum of country emissions within a region may not match the IAM's regional total. This step enforces that constraint.

### Stage 3 — Country-Level IEA Historical Fit

**Goal:** Shift individual country emissions to match IEA historical observations at `harm_year`.

**How it works:**

1. **Load IEA data:** Read the IEA CO2 emissions CSV (`input_data/IEA 2019 CO2 emissions from fuels_ISO.csv`). For each DSCALE variable, the `iea_var_dict` (in `fixtures.py`) maps to specific IEA FLOW + PRODUCT combinations. The data is converted from kt CO2 to Mt CO2.

2. **Shift to match IEA (`fun_match_hist`):**
   - Computes `delta = IEA_value - model_value` at `harm_year`
   - Applies this delta to all years: `shifted = model + delta`
   - Only countries present in both the model output and the IEA dataset are shifted

3. **Blend back (`fun_discount_rate`):**
   - Creates weights `w` that go from `1` at `harm_year` to `0` at `tc=2050`
   - Final result: `result = shifted * w + original * (1 - w)`
   - At `harm_year`: fully IEA-matched
   - At 2050+: fully original (Stage 2 values preserved in the long run)

4. **Rescale fuel sub-sectors proportionally:**
   - After shifting a main sector (e.g. Industry), its fuel sub-sectors (Coal, Gas, Oil) are scaled by the ratio `new_main / old_main`
   - This preserves the fuel mix shares while maintaining consistency with the new main-sector total
   - Both `step5f_dict1` (demand sectors) and `step5f_dict2` (supply sectors) are checked

**Why:** This is the final harmonization step, so matching IEA country-level data is prioritised. After this step, minor inconsistencies with IAM regional totals may appear — this is expected and acceptable.

---

## Data Flow Diagram

```
Input Data
    |
    v
[step5e output]  +  [step5b sectorial CO2]  +  [step5c non-CO2]
    |                       |                        |
    +----------+------------+------------------------+
               |
               v
         Combined df
               |
    +----------+----------+
    |                     |
    v                     v
 STAGE 1               df_iam
 PRIMAP harmo           (raw)
    |                     |
    v                     |
 df_iam                   |
 (harmonized)             |
    |                     |
    +----------+----------+
               |
               v
         STAGE 2
    run_sector_harmo_enhanced_iamc
    (country sums → IAM regional totals)
               |
               v
         STAGE 3
    IEA country-level fit
    (fun_match_hist + blending)
               |
               v
    Rescale fuel sub-sectors
               |
               v
         Final output CSV
```

---

## Variables Harmonized with IEA

| Variable | Description |
|----------|-------------|
| `Emissions\|CO2\|Energy\|Demand\|Transportation` | Road, rail, air, pipeline, navigation |
| `Emissions\|CO2\|Energy\|Demand\|Industry` | Manufacturing industries and construction |
| `Emissions\|CO2\|Energy\|Demand\|Residential and Commercial` | Residential + Commercial and public services |
| `Emissions\|CO2\|Energy\|Supply\|Electricity` | Electricity generation (with correct CHP allocation) |

### CHP (Combined Heat and Power) handling

For the Electricity variable, `data_shepherd` correctly splits CHP emissions between
Electricity and Heat using **IEA electricity/heat output shares**:

```
Electricity = pure_electricity_plants * 1.0
            + CHP_plants * (electricity_output / (electricity_output + heat_output))
```

This is essential for CHP-heavy countries (e.g. Poland ~98% CHP, Denmark ~99% CHP).
Without the split, 100% of CHP emissions would be assigned to Electricity, overshooting
the correct value significantly.

---

## IEA Data Source

- **File:** `input_data/IEA_hist_emissions_for_step5f.csv`
- **Generated by:** `notebook/CLEAN_INPUT_DATA.ipynb` using `data_shepherd.emissions.get_historic_iea(source="IEA_GHG_FUEL_DETAILED_2025")`
- **Format:** IAMC format — index = `[MODEL, SCENARIO, REGION, VARIABLE, UNIT]`, year columns
- **Units:** Mt CO2/yr (no unit conversion needed in Step5f)
- **Same source** as the `explore_emissions` notebook plots, ensuring visual consistency

Note: countries missing from the IEA data are **not harmonized** in Stage 3 — they keep their Stage 2 values.

---

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `fun_harmonize_df_iam_with_hist_data` | `utils_emissions.py` | Stage 1: harmonize IAM regional data with PRIMAP |
| `run_sector_harmo_enhanced_iamc` | `utils_emissions.py` | Stage 2: scale countries to match IAM regional totals |
| `fun_match_hist` | `utils_emissions.py` | Stage 3: compute delta and shift data to match IEA |
| `fun_discount_rate` | `utils_emissions.py` | Stage 3: create blending weights (1 at baseyear, 0 at tc) |
| `fun_get_historical_emissions` | `utils_emissions.py` | Load and filter IEA data by FLOW/PRODUCT |
| `fun_read_hist_emissions` | `utils_emissions.py` | Read the raw IEA CSV |
| `fun_xs` | `utils_pandas.py` | Filter DataFrame by index level values |

---

## Blending Formula

At each year `t` between `harm_year` and `tc`:

```
result(t) = shifted(t) * w(t) + original(t) * (1 - w(t))
```

Where:
- `shifted` = output of `fun_match_hist` (country data shifted to match IEA)
- `original` = output of Stage 2 (sectorial harmonization)
- `w(harm_year) = 1` (fully IEA-matched)
- `w(tc) = 0` (fully original)
- `w` decreases linearly between `harm_year` and `tc`

---

## Fuel Sub-Sector Rescaling

After harmonizing a main sector (e.g. `Emissions|CO2|Energy|Demand|Industry`), its fuel sub-sectors must be adjusted to stay consistent:

```
ratio = new_Industry / old_Industry          (per country, per year)
new_Industry_Coal = old_Industry_Coal * ratio
new_Industry_Gas  = old_Industry_Gas  * ratio
new_Industry_Oil  = old_Industry_Oil  * ratio
```

This preserves the **relative fuel shares** while matching the new harmonized total. The mapping of main sectors to fuel sub-sectors is defined in:
- `step5f_dict1`: demand sectors (Industry, Transportation, Residential & Commercial)
- `step5f_dict2`: supply sectors (Electricity, Energy EXCL BECCS)

---

## Known Limitations

1. **Sectorial consistency:** After Stage 3, sub-sector sums may not exactly equal their parent variable. This is because Stage 3 shifts individual sectors independently. The consistency check at the end prints a warning but does not raise an error.

2. **Missing countries:** Countries not present in the IEA CSV are not harmonized in Stage 3. They retain their Stage 2 values.

3. **Regional sums:** Stage 3 does NOT preserve the property that country sums equal IAM regional totals (which Stage 2 established). The IEA country-level match is prioritised over regional consistency.

4. **CHP allocation:** CHP emissions are split between Electricity and Heat using IEA electricity/heat output shares (via `data_shepherd`). This correctly handles countries like Poland where CHP dominates. The pre-formatted CSV must be regenerated (by re-running `CLEAN_INPUT_DATA.ipynb`) whenever the IEA data source is updated.
