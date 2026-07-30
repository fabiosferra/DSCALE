# Known issue: instability in `Emissions|CO2|Energy|{Oil,Coal,Gas}` (all-sector fuel totals)

**Status:** Not yet root-caused. Documented 2026-07-28 while investigating a separate,
now-fixed bug in step5f's Stage 3 fuel-rescale (see `docs/step5f_emissions_harmonization.md`
and the `tc`/ratio-harmonization discussion in that period's session history).

## Symptom

`Emissions|CO2|Energy|Oil`, `Emissions|CO2|Energy|Coal`, and `Emissions|CO2|Energy|Gas`
(the bare, all-sector fuel totals — **not** `Emissions|CO2|Energy|Supply|Electricity|Coal`
or other sector-scoped variables) show implausible multi-year instability for a number of
countries in the REMIND_Q1_2026 27_07_2026 run: sign flips, swings of hundreds of Mt CO2/yr
between adjacent 5-year steps, and values landing on exact `0.00`/`-0.00`. This is not a
single-year blip — for several countries the instability persists across *multiple
consecutive* 5-year periods, which is a strong signal of a real upstream computation bug
rather than isolated noise.

These variables **do reach the final delivered CSV** (`Emissions_<date>.csv`) — 570 rows
across all countries in the 27_07_2026 run — so this is not dead/intermediate data.

## Evidence (REMIND_Q1_2026, run 27_07_2026)

Sample of the worst cases (`prev → curr → next`, Mt CO2/yr, 5-year steps):

| Region | Variable | ...→prev→curr→next→... |
|---|---|---|
| NPL | `Emissions\|CO2\|Energy\|Oil` | 2055: −35.55 → 2060: −46.10 → 2065: 1097.66 → 2070: 0.00 |
| BLR | `Emissions\|CO2\|Energy\|Oil` | 2035: 9.72 → 2040: −384.11 → 2045: 236.57 |
| RUS | `Emissions\|CO2\|Energy\|Oil` | 2035: 410.38 → 2040: 445.87 → 2045: 15.77 |
| MEX | `Emissions\|CO2\|Energy\|Oil` | 2035: 166.85 → 2040: −23.47 → 2045: 2.52 |
| BRA | `Emissions\|CO2\|Energy\|Oil` | 2035: 107.69 → 2040: 280.65 → 2045: 43.32 |
| NPL | `Emissions\|CO2\|Energy\|Coal` | 2055: −2.64 → 2060: −4.43 → 2065: 202.17 |

Note the NPL Oil case alone cascades across five consecutive periods (2055→2080), never
settling — inconsistent with a single artifact year, and inconsistent with realistic fuel
combustion economics (a country's total oil-CO2 does not plausibly jump from negative, to
~1100 Mt, to exactly zero, within 15 years).

## How this was found

While investigating why ZAF `Emissions|CO2|Energy|Demand|Industry|Solids|Coal` appeared to
fall in 2030 despite rising coal consumption, the step5f "spike fix" (single-year outlier
smoothing added in commit `6eac65e`) was temporarily instrumented to log every row/year its
local-extremum check (check 2, the 0.75× relative-deviation V-shape check) would flag. Of
1,052 total flags, 159 involved an interpolated magnitude above 5 Mt CO2/yr; of those, 88
were in these bare fuel-total variables, split out from the 71 in properly sector-scoped
(`Demand|...`/`Supply|...`) variables. The sector-scoped 71 turned out to be mostly legitimate
large economic swings (like ZAF's), fixed by capping the local-extremum check to a magnitude
ceiling. The 88 bare-fuel-total flags did **not** look like legitimate swings — they're the
basis for this write-up.

## What's not yet known

- **Where these variables are computed.** They are not built by `step5f_temp_vars_dict` /
  `step5f_temp_vars_dict_df_iam` in `Step_5f_emissions.py` (those only construct the
  `... EXCL BECCS` variants). They likely originate upstream (step5b `Emissions_by_sectors_and_revenues`,
  or a cross-sector aggregation elsewhere in `utils_emissions.py`) and pass through step5f's
  Stage 1–3 harmonization like any other variable — but Stage 1–3 only explicitly target the
  six `vars_to_be_harmo` sector variables (`Emissions|CO2|Energy`, `...|Demand|Transportation`,
  `...|Demand|Industry`, `...|Supply|Heat`, `...|Supply|Electricity`,
  `...|Demand|Residential and Commercial`) and the four in `stepf5f_var_to_harmonise`
  (`fixtures.py`) — the bare fuel-only variables aren't in either list, so they likely pass
  through Stage 2's `run_sector_harmo_enhanced_iamc` scaling (via `step5f_dict3`?) without
  the same safeguards, or without harmonization at all.
- **Whether this predates the ratio-harmonization fix.** Unlike the ZAF Industry|Coal case,
  this instability was not confirmed to be a symptom of the additive-offset Stage 3 mechanism
  that was replaced — it may be a distinct bug in how these all-sector fuel totals are summed
  or scaled, independent of Stage 3.
- **Whether it's a downscaling artifact or present in the regional IAM source data already.**
  Not yet checked against `df_iam` / the pre-step5f `REMIND 3.4_2023_harmo_step5e_None.csv`.

## Suggested next steps

1. Grep `utils_emissions.py` / `utils.py` for where `Emissions|CO2|Energy|Oil` (and Coal/Gas)
   get constructed — confirm whether they're a straight sum across sector sub-variables, and
   at what pipeline stage.
2. Pull the pre-Stage-1/pre-Stage-3 values for NPL/BLR/RUS/MEX/BRA `Emissions|CO2|Energy|Oil`
   to see whether the instability is already present before step5f touches them (implicates
   an earlier step) or only appears after Stage 1–3 (implicates step5f's handling of
   variables outside its explicit `vars_to_be_harmo` list).
3. Once root-caused, decide whether these variables need their own harmonization pass, should
   be excluded/rebuilt as a sum of the (correct) sector-scoped sub-variables post-harmonization,
   or the upstream bug should be fixed directly.
