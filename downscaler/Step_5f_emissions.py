from typing import Union
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from downscaler import CONSTANTS

from pandas.testing import assert_frame_equal
from downscaler.utils_pandas import fun_xs, fun_create_var_as_sum, fun_rename_index_name, fun_add_multiply_dfmultindex_by_dfsingleindex
from downscaler.utils_emissions import (
    fun_read_results,
    run_sector_harmo_enhanced_iamc,
    fun_international_variables,
    fun_read_df_iams,
    fun_get_models,
    fun_wildcard,
    fun_get_variable_unit_dictionary,
    fun_add_units,
    fun_invert_dictionary,
    fun_get_regions,
    fun_index_names,
    show_inconsistencies,
    fun_flatten_list,
    fun_inconsistencies,
    fun_rename_index_name,
    fun_harmonize_df_iam_with_hist_data,
)
from downscaler.fixtures import (
    step5f_dict1,   
    step5f_dict2, 
    step5f_dict3, 
    step5f_temp_vars_dict, 
    step5f_temp_vars_dict_df_iam, 
    iea_countries, 
    stepf5f_var_to_harmonise
)


def main(
    project,
    csv_in: Optional[str] = None,
    step: str = "step5",
    models: Optional[list] = None,
    countrylist: Optional[list] = None,
    scenarios: Optional[list] = ["h_cpol", "h_ndc", "o_2c", "o_1p5c"],
    harm_year: int = 2022
):
    """
    Step 5f: Emissions harmonization pipeline (the "emissions clock").

    This step performs a THREE-STAGE harmonization of CO2 emissions data:

      Stage 1 — Regional IAM harmonization (lines ~88-109):
          Harmonizes regional IAM emissions with historical PRIMAP data using
          `fun_harmonize_df_iam_with_hist_data`. This adjusts IAM regional totals
          (e.g. Emissions|CO2|Energy for "REMIND|EUR") to match observed historical
          values, then blends back to the original IAM trajectory by tc=2050.

      Stage 2 — Sectorial consistency (line ~204):
          `run_sector_harmo_enhanced_iamc` scales country-level sub-sector emissions
          so that the sum across countries matches the (now harmonized) IAM regional
          totals. This preserves the sectorial breakdown (e.g. Industry, Transport,
          Electricity) while ensuring country totals are consistent with the IAM.

      Stage 3 — Country-level IEA historical fit (lines ~206-276):
          Shifts individual country emissions to match IEA historical data at the
          base year (harm_year), then blends back to the Stage 2 values by tc=2050.
          This is the FINAL step, so country-level match to IEA is prioritised over
          perfect consistency with IAM regional totals.

    After Stage 3, fuel sub-sectors (e.g. Coal, Gas, Oil under Industry) are
    proportionally rescaled to maintain their relative shares while matching the
    new harmonized main-sector totals.

    Parameters
    ----------
    project : str
        Project identifier (e.g. "REMIND_fuel_mix_testing").
    csv_out : str
        Output CSV filename.
    csv_in : str, optional
        Input CSV identifier.
    step : str
        Pipeline step name (default "step5").
    models : list, optional
        Model filter patterns (e.g. ['*REMIND*']).
    countrylist : list, optional
        Restrict processing to these ISO country codes.
    scenarios : list
        Scenario names to process (e.g. ["NPE-core"]).
    harm_year : int
        Base year for IEA historical harmonization (default 2020).
    """
    files=None
    csv_out=f"Emissions_{csv_in}.csv"
    # Use nested directory structure
    NESTED_DIR = CONSTANTS.NESTED_RES_DIR(step, project, csv_in)

    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================

    print("\n ---- RUN STEP5F Emissions ---- \n")

    # Get models
    avmodels=fun_get_models(project)
    models = fun_wildcard(models, avmodels)

    # Read NGFS full data (output of step5e: energy variables after historical harmonization)
    # i = project, step, files, None, models, f"{csv_in}_2022_harmo_step5e_None" # Careful about the harmonisation date!
    # df = fun_read_results(*i)[0]
    df = pd.read_csv(NESTED_DIR / f"{models[0]}_{harm_year}_harmo_step5e_None.csv")
    df = fun_index_names(df, True, int)

    # Read sectorial CO2 emissions (from step5b: emissions downscaled by sector)
    # if csv_in:
    #     i = project, step, files, None, models, f"{csv_in}_Emissions_by"
    # else:
    #     i = project, step, files, None, models, "Emissions_by"
    # df_co2 = fun_read_results(*tuple(list(i)))[0]
    # d = {"VARIABLE": "Emissions|CO2|Industrial Processes"}
    # df_co2 = fun_xs(df_co2, d, exclude_vars=True)
    df_co2 = pd.read_csv(NESTED_DIR / f"{models[0]}_Emissions_by_sectors_and_revenues.csv")
    df_co2 = fun_index_names(df_co2, True, int)
    d = {"VARIABLE": "Emissions|CO2|Industrial Processes"} # Excluded here because it's recreated later
    df_co2 = fun_xs(df_co2, d, exclude_vars=True)

    # Append sectorial CO2 emissions to the main dataframe
    df = pd.concat([df, df_co2], axis=0)
    # Drop duplicate index rows (same variable in step5e and step5b), keeping step5e values
    df = df[~df.index.duplicated(keep='first')]

    # Append non-CO2 greenhouse gases (from step5c: CH4, N2O, F-gases, etc.)
    i = project, step, files, None, models, "non"
    # non_co2 = fun_read_results(*tuple(list(i)))[0]#.dropna(how="all")
    # df = pd.concat([df, non_co2.droplevel("REGION")])
    non_co2 = pd.read_csv(NESTED_DIR / f"{models[0]}_non_co2.csv")
    non_co2 = fun_index_names(non_co2, True, int)
    df = pd.concat([df, non_co2.droplevel("REGION")])

    # Read Regional IAM results (the "target" trajectories from the IAM model)
    df_iam = fun_read_df_iams(project, models)
    df_iam = fun_index_names(df_iam, True, int)

    # =========================================================================
    # 2. STAGE 1 — REGIONAL IAM HARMONIZATION WITH PRIMAP HISTORICAL DATA
    # =========================================================================
    # For each IAM region (e.g. EUR, AFR, LAM), adjust the IAM's emissions
    # trajectory so that it matches PRIMAP historical observations at the base
    # year, then blend back to the original IAM trajectory by tc=2050.
    # This is done for the main energy CO2 variable and its sub-sectors.
    model=models[0]
    regmap=fun_get_regions(project, model) # regional mapping: {region_name: [ISO codes]}
    countries_in_data = set(df.index.get_level_values("REGION").unique())
    if countrylist is not None:
        countries_in_data = countries_in_data & set(countrylist)
    regmap={k:v for k,v in regmap.items() if len(set(v) & countries_in_data) > 0}
    res={}
    vars_to_be_harmo = [
                        'Emissions|CO2|Energy',
                        'Emissions|CO2|Energy|Demand|Transportation',
                        'Emissions|CO2|Energy|Demand|Industry',
                        'Emissions|CO2|Energy|Supply|Heat',
                        'Emissions|CO2|Energy|Supply|Electricity',
                        'Emissions|CO2|Energy|Demand|Residential and Commercial'
                    ]
    for r,countries in regmap.items():
        res[r]=fun_harmonize_df_iam_with_hist_data(f"{model}|{r[:-1]}", countries, df_iam,df,
                    main_var = 'Emissions|CO2|Energy',
                    vars_to_be_harmo =vars_to_be_harmo,
                tc=2050 # From 2050 onwards, we keep original IAM data (no longer harmonized)
                )
    add=pd.concat(list(res.values()))
    # Replace the original IAM data for harmonized variables with the new harmonized version
    df_iam= pd.concat([df_iam.drop(vars_to_be_harmo, level="VARIABLE"), add])


    # =========================================================================
    # 3. FILTER & CLEAN DATA
    # =========================================================================

    # Keep only emissions variables (filter by unit)
    emi_unit = ["Mt CO2/yr", "Mt CO2-equiv/yr"]
    emi_unit=emi_unit+list(non_co2.reset_index().UNIT.unique()) # add non-CO2 units
    df=fun_xs(df, {"UNIT": emi_unit})

    # Exclude statistical differences variables (artefacts, not real emissions)
    excl = [x for x in df.reset_index().VARIABLE.unique() if "Statistica" in x]
    excl = ["Emissions|Total Non-CO2"] + excl
    df = fun_xs(df, {"VARIABLE": excl}, exclude_vars=True)

    # Create temporary helper variables needed for sectorial harmonization
    # (e.g. aggregates that step5f_dict rules reference but don't exist yet)
    for k, v in step5f_temp_vars_dict.items():
        df = fun_create_var_as_sum(df, k, v)
    for k,v in step5f_temp_vars_dict_df_iam.items():
        df_iam = fun_create_var_as_sum(df_iam, k, v)

    if countrylist is not None:
        df = fun_xs(df, {"REGION": countrylist})

    # Expand wildcard scenario patterns against what's actually in the data
    avscenarios = list(df.index.get_level_values("SCENARIO").unique())
    scenarios = fun_wildcard(scenarios, avscenarios)

    # Slice for selected scenarios only
    df = fun_xs(df, {"SCENARIO": scenarios})

    # =========================================================================
    # 4. FIX NEGATIVE EMISSIONS (apply absolute value)
    # =========================================================================
    # Some country-level demand/supply sub-sectors can have negative values
    # (artefacts from the downscaling). When a variable has both positive and
    # negative values across countries for the same year, we apply abs().
    res={}
    updated_data_list=[]
    for r,clist in regmap.items(): # do not use countrylist here!
        for scen in scenarios:
            dfrs=fun_xs(df, {'SCENARIO':scen, 'REGION':clist}) # dfrs means df region and scenario
            variables=dfrs.index.get_level_values('VARIABLE').unique()
            # Select variables that we want to `shrink`
            variables= [x for x in variables if 'Demand' in x or 'Supply' in x]
            for variable in variables:
                df1= fun_xs(dfrs, {'VARIABLE':variable})# slice variable
                for t in df1.columns:
                    if np.round(df1[t],5).min()*np.round(df1[t],5).max() < 0:
                        # print('Correction applied to', r, 'for variable', variable, 'at time', t)
                        if t<=2070: # we keep track only of data until 2070 (we do not share data beyond 2070)
                            info=f'Absolute value Correction applied to {r} for variable {variable} at time {t}'
                            print(info)
                            updated_data_list+=[info]

                        # Apply shrinkage to the data (minimize `trade` volume)
                        # df1[t] = shrink(df1[t])

                        df1[t] = np.abs(df1[t]) # apply absolute value
                        res[f"{r,scen,variable}"] = df1

    if res:
        df_update=pd.concat(list(res.values())) # Get Updated results after negative values correction
        df_update=fun_index_names(df_update, True, int)
        df=pd.concat([df.drop(df_update.index), df_update]) # Update df with corrected values

    # =========================================================================
    # 5. LOAD IEA HISTORICAL EMISSIONS (for Stage 3)
    # =========================================================================
    # Read the pre-formatted IEA historical emissions CSV, produced by the
    # CLEAN_INPUT_DATA notebook using data_shepherd.emissions.get_historic_iea().
    # This file already has:
    #   - Correct Electricity/Heat CHP split (CHP emissions allocated using
    #     IEA electricity/heat output shares, not 100% to one or the other)
    #   - IAMC format: index = [MODEL, SCENARIO, REGION, VARIABLE, UNIT]
    #   - Units: Mt CO2/yr
    #   - Same data source as the explore_emissions notebook plots
    iea_hist_path = Path("input_data/IEA_hist_emissions_for_step5f.csv")
    vars_to_harmonize_hist = stepf5f_var_to_harmonise 

    if iea_hist_path.exists():
        iea_hist = pd.read_csv(iea_hist_path, index_col=['MODEL', 'SCENARIO', 'REGION', 'VARIABLE', 'UNIT'])
        # Ensure year columns are numeric int (CSV may store them as strings)
        iea_hist.columns = pd.to_numeric(iea_hist.columns).astype(int)
        print(f"Loaded IEA historical emissions: {len(iea_hist)} rows, "
              f"{iea_hist.index.get_level_values('REGION').nunique()} countries")
    else:
        print(f"WARNING: {iea_hist_path} not found — skipping IEA historical harmonization. "
              f"Run the CLEAN_INPUT_DATA notebook to generate this file.")
        iea_hist = None

    # =========================================================================
    # 6. STAGE 2 — SECTORIAL HARMONIZATION + STAGE 3 — IEA COUNTRY-LEVEL FIT
    # =========================================================================
    # Loop over each (region, scenario) pair.
    # For each pair:
    #   a) run_sector_harmo_enhanced_iamc (Stage 2): ensures country sub-sectors
    #      sum to the harmonized IAM regional totals from Stage 1
    #   b) IEA historical fit (Stage 3): shifts individual countries to match
    #      IEA data at the base year, then blends back to Stage 2 values
    model=models[0]
    regmap=fun_get_regions(project, model) # regional mapping
    countries_in_data = set(df.index.get_level_values("REGION").unique())
    if countrylist is not None:
        countries_in_data = countries_in_data & set(countrylist)
    regmap={k:v for k,v in regmap.items() if len(set(v) & countries_in_data) > 0}
    res={}
    res_pre_harmo={}  # Stage 2 results BEFORE IEA historical fit (for comparison)
    ds=[step5f_dict1, step5f_dict2, step5f_dict3]
    for r,region_countries in regmap.items():
        for scen in scenarios:
            print('Harmonzing', model, r, scen)
            df1=fun_xs(df, {'SCENARIO':scen, 'REGION':region_countries}) # df
            df2=fun_rename_index_name(fun_xs(df_iam, {'REGION':f"{model}|{r[:-1]}", 'SCENARIO':scen}).drop(2005, axis=1), {'SCENARIO':'TARGET'}) # df_iam

            # ---- STAGE 2: Sectorial harmonization ----
            # Scales country-level data so that the sum across countries for each
            # sub-sector matches the IAM regional total (df2). Uses step5f_dict1,
            # dict2, dict3 which define the parent→child variable relationships.
            # Output has TARGET (not SCENARIO), METHOD level, LONG_TERM suffix on
            # variable names, and string-float columns (e.g. '2010.0').
            df_harmo = run_sector_harmo_enhanced_iamc(project, df1, ds, df2, verbose=False)

            # Save Stage 2 result before IEA harmonization (for comparison)
            res_pre_harmo[f"{model, r,scen}"] = df_harmo.copy()

            # ---- STAGE 3: Country-level IEA historical fit ----
            # For each variable in vars_to_harmonize_hist, shift country-level
            # emissions to match IEA observations at harm_year, then blend
            # back to the Stage 2 result by tc=2050.
            if iea_hist is not None:
                # -- Prepare df_harmo for historical harmonization --
                # run_sector_harmo_enhanced_iamc outputs a DataFrame with:
                #   - TARGET instead of SCENARIO in the index
                #   - A METHOD level (not needed here)
                #   - LONG_TERM suffix appended to variable names
                #   - Columns as string floats (e.g. '2010.0')
                # We need to clean all of this before matching with IEA data.
                df_for_hist = df_harmo.copy()
                if 'METHOD' in df_for_hist.index.names:
                    df_for_hist = df_for_hist.droplevel('METHOD')
                df_for_hist = fun_rename_index_name(df_for_hist, {'TARGET': 'SCENARIO'})
                df_for_hist.columns = pd.to_numeric(df_for_hist.columns).astype(int)
                # Deduplicate columns: Stage 2 can produce duplicate year columns
                # (e.g. multiple '2010.0' columns that all become 2010 after int cast).
                # Keep only the first of each group.
                if df_for_hist.columns.duplicated().any():
                    df_for_hist = df_for_hist.loc[:, ~df_for_hist.columns.duplicated(keep='first')]

                # Strip the LONG_TERM suffix so variable names match iea_hist
                # e.g. "Emissions|CO2|Energy|Demand|TransportationLONG_TERM"
                #    → "Emissions|CO2|Energy|Demand|Transportation"
                lt_rename = {v: v.replace('LONG_TERM', '') for v in df_for_hist.index.get_level_values('VARIABLE') if 'LONG_TERM' in v}
                if lt_rename:
                    df_for_hist = df_for_hist.rename(index=lt_rename, level='VARIABLE')

                # Ensure iea_hist columns are numeric int (may become generic
                # Index type after pd.concat, breaking integer slice operations)
                iea_hist_fixed = iea_hist.copy()
                iea_hist_fixed.columns = pd.to_numeric(iea_hist_fixed.columns).astype(int)

                # -- Harmonize each variable with IEA historical data --
                harmo_results = {}
                from downscaler.utils_emissions import fun_discount_rate
                for var in vars_to_harmonize_hist:
                    try:
                        # `trade`: country-level data from Stage 2 for this variable
                        # Index: [MODEL, SCENARIO, REGION, VARIABLE, UNIT], columns: years
                        trade = fun_xs(df_for_hist, {'VARIABLE': var})

                        # If harm_year is not in trade columns (e.g. 2022 with 5-year
                        # steps), OR the column exists but is all NaN (variable not in
                        # step5e_harmo → only 5-year IAM values, no interpolated harm_year),
                        # insert/overwrite it as NaN and interpolate linearly from neighbours.
                        if harm_year not in trade.columns or trade[harm_year].isna().all():
                            trade[harm_year] = np.nan
                            trade = trade.sort_index(axis=1).interpolate(axis=1)

                        # `iea_adj`: IEA reference data, trimmed to years <= harm_year
                        # Index: REGION (ISO codes), columns: years
                        iea_raw = iea_hist_fixed.xs(var, level="VARIABLE")
                        iea_adj = iea_raw[[c for c in iea_raw.columns if c <= harm_year]]
                        iea_adj = iea_adj.droplevel(["SCENARIO", "MODEL", "UNIT"])

                        # If harm_year is not in iea_adj columns, interpolate.
                        if harm_year not in iea_adj.columns:
                            iea_adj[harm_year] = np.nan
                            iea_adj = iea_adj.sort_index(axis=1).interpolate(axis=1, limit_direction='forward')

                        # Compute delta at harm_year ONLY, then apply as a constant
                        # shift to all years. This ensures harmonization anchors at
                        # harm_year (e.g. 2023) — NOT at all prior 5-year steps.
                        # fun_match_hist would compute per-year deltas for ALL
                        # historical years, making 2020 always match IEA(2020).
                        delta_df = fun_add_multiply_dfmultindex_by_dfsingleindex(
                            -trade[[harm_year]], iea_adj[[harm_year]], operator='+'
                        )
                        # delta_df[harm_year] = iea(harm_year) - trade(harm_year) per country
                        # Countries not in IEA get NaN → preserved via fillna in blending
                        trade_match = trade.add(delta_df[harm_year], axis=0)
                        if len(trade_match.dropna(how="all")) == 0:
                            continue

                        # BLENDING: create weights w that go from 1 (at harm_year)
                        # to 0 (at tc=2050). This ensures:
                        #   - At harm_year: result = trade_match (fully IEA-matched)
                        #   - At 2050+:     result = trade (fully original Stage 2 value)
                        #   - In between:   linear interpolation
                        # fun_discount_rate returns a Series indexed by year with
                        # values between 0 and 1.
                        # Build year lists that include harm_year so w(harm_year)=1 exactly
                        all_years = sorted(set(list(range(2010, 2505, 5)) + [harm_year]))
                        hist_years = sorted(set(list(range(2005, harm_year + 1, 5)) + [harm_year]))
                        w = fun_discount_rate(all_years, hist_years, 2080, 1, 0)
                        var_harmo = (trade_match.fillna(0) * w + trade.fillna(0) * (1 - w)).dropna(how='all', axis=1)

                        # Conditional clip (per cell): if the pre-harmonization value
                        # was non-negative but harmonization pushed it below zero,
                        # clip to 0. If the IAM already projected negatives for that
                        # country+year (e.g. BECCS), preserve them.
                        trade_aligned = trade.reindex_like(var_harmo).fillna(0)
                        was_non_negative = trade_aligned >= 0  # boolean mask per cell
                        var_harmo = var_harmo.where(~was_non_negative | (var_harmo >= 0), 0)

                        harmo_results[var] = var_harmo
                    except Exception as e:
                        import traceback
                        print(f"WARNING: Could not harmonize {var} with IEA historical data: {e}")
                        traceback.print_exc()

                # -- Proportionally rescale fuel sub-sectors --
                # After shifting main sectors (e.g. Industry), the by-fuel breakdown
                # (e.g. Industry|Coal, Industry|Gas, Industry|Oil) must be rescaled
                # to maintain consistency. We apply the same ratio (new/old) to all
                # fuel sub-sectors of each harmonized main sector.
                # step5f_dict1 maps demand sectors → fuel sub-sectors:
                #   e.g. "...|Industry" → ["...|Industry|Coal", "...|Industry|Gas", ...]
                # step5f_dict2 maps supply sectors → fuel sub-sectors:
                #   e.g. "...|Electricity" → ["...|Electricity|Coal", "...|Electricity|Gas", ...]
                if harmo_results:
                    from itertools import chain
                    all_dicts = chain(step5f_dict1.items(), step5f_dict2.items())
                    for main_var, sub_vars in all_dicts:
                        if main_var in harmo_results:
                            old_main = fun_xs(df_for_hist, {'VARIABLE': main_var})
                            new_main = harmo_results[main_var]
                            # ratio = new_harmonized / old_pre_harmonization (per country, per year)
                            # Where old = 0, ratio = NaN → sub-sector stays at 0
                            common_cols = sorted(set(old_main.columns) & set(new_main.columns))
                            ratio = new_main[common_cols] / old_main[common_cols].replace(0, np.nan)
                            for sub_var in sub_vars:
                                sub_data = fun_xs(df_for_hist, {'VARIABLE': sub_var})
                                if len(sub_data) > 0:
                                    # Align ratio's VARIABLE index level to match sub_var
                                    # (ratio has main_var in its index, sub_data has sub_var)
                                    ratio_aligned = ratio.rename(index={main_var: sub_var}, level='VARIABLE')
                                    harmo_results[sub_var] = sub_data[common_cols] * ratio_aligned[common_cols]

                    # -- Reassemble the full DataFrame --
                    # Remove old versions of all harmonized variables (main + sub-sectors),
                    # replace with harmonized versions, and restore the pipeline's expected
                    # index format (TARGET instead of SCENARIO).
                    vars_done = list(harmo_results.keys())
                    df_remaining = fun_xs(df_for_hist, {'VARIABLE': vars_done}, exclude_vars=True)
                    df_updated = pd.concat([df_remaining] + list(harmo_results.values()), sort=False)
                    # Keep only original year columns
                    year_cols = sorted([c for c in df_updated.columns if isinstance(c, (int, float))])
                    df_harmo = df_updated[year_cols]
                    # Rename SCENARIO back to TARGET for consistency with rest of pipeline
                    df_harmo = fun_rename_index_name(df_harmo, {'SCENARIO': 'TARGET'})

            res[f"{model, r,scen}"] = df_harmo

    # =========================================================================
    # 7. POST-PROCESSING
    # =========================================================================

    # Save pre-harmonization (Stage 2 only) results for comparison
    df_pre_harmo = pd.concat(list(res_pre_harmo.values()))
    df_pre_harmo = fun_index_names(df_pre_harmo, True, int)
    pre_harmo_path = NESTED_DIR / csv_out.replace(".csv", "_pre_iea_harmo.csv")
    df_pre_harmo.to_csv(pre_harmo_path)
    print(f"Saved pre-IEA-harmonization results to {pre_harmo_path}")

    # Concatenate all (region, scenario) results back into a single DataFrame
    df=pd.concat(list(res.values()))
    df=fun_index_names(df, True, int)

    # Remove temporary variables (this block does not seem to work properly - maybe could be commented out)
    d = {"VARIABLE": list(step5f_temp_vars_dict.keys())}
    df = fun_xs(df, d, exclude_vars=True)

    # Calculates international bunkers (World - sum across regions)
    var = "Emissions|CO2|Energy|Demand|Transportation"
    bunkers = fun_international_variables(project, var, models)
    bunkers = bunkers.rename({var: f"{var}|International Bunkers"})
    bunkers = fun_rename_index_name(bunkers, {"SCENARIO": "TARGET"})
    bunkers = fun_index_names(bunkers, True, int)
    if 'METHOD' in df.columns:
        df = df.drop('METHOD', axis=1)
    elif 'METHOD' in df.index.names:
        df = df.droplevel('METHOD')
    # Remove duplicate columns from df (combine duplicates by taking first non-null value)
    if df.columns.duplicated().any():
        # Group duplicate columns and take first non-null value for each
        df = df.groupby(level=0, axis=1).first()

    # Align columns before concatenation (keep harm_year even if bunkers lacks it)
    bunkers_aligned = bunkers.reset_index().set_index(df.index.names)
    idx_names = list(df.index.names)
    df_clean = df.reset_index()
    bunkers_clean = bunkers_aligned.reindex(columns=df.columns).reset_index()
    df = pd.concat([df_clean, bunkers_clean], ignore_index=True, sort=False)
    df = df.set_index(idx_names)

    # Add regional LULUCF data (land use, land-use change and forestry)
    # This comes directly from the IAM (not downscaled to country level)
    v = "Emissions|CO2|AFOLU"
    f = fun_rename_index_name
    df_iam = f(df_iam, {"SCENARIO": "TARGET"}).reset_index().set_index(df.index.names)
    lulucf = df_iam.xs(v, level="VARIABLE", drop_level=False)
    lulucf = lulucf.reindex(columns=df.columns)
    df = pd.concat([df, lulucf])

    # Rename variables: strip LONG_TERM suffix from any remaining variable names
    mydict={x:x.replace('LONG_TERM','') for x in df.index.get_level_values('VARIABLE')}
    df=df.rename(mydict, axis=0)

    # Fix units: ensure all variables have correct units from IAM metadata
    units = fun_get_variable_unit_dictionary(df_iam)
    d = {k: [v] for k, v in units.items()}
    df = fun_add_units(df, df_iam, fun_invert_dictionary(d))

    # Select final time period and fix any remaining missing units
    df = df.rename({"missing": "Mt CO2/yr"}, level="UNIT")
    year_range = sorted(set(list(range(2010, 2105, 5)) + [harm_year]))
    df = df.loc[:, [str(x) for x in year_range]]

    # Interpolate harm_year for rows that don't have it (e.g. bunkers, LULUCF)
    hy_col = str(harm_year)
    if hy_col in df.columns and df[hy_col].isna().any():
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.sort_index(axis=1).interpolate(axis=1, limit_direction='both')

    # Fix 2065 spike: the IAM has no 2065 column (jumps 2060→2070), so Stage 2
    # interpolates it with near-zero denominators → huge artefacts. Re-interpolate
    # 2065 as the midpoint of 2060 and 2070.
    if '2065' in df.columns and '2060' in df.columns and '2070' in df.columns:
        df['2065'] = (df['2060'].astype(float) + df['2070'].astype(float)) / 2

    # =========================================================================
    # 8. SAVE & VALIDATE
    # =========================================================================

    # Check sectorial consistency: do sub-sectors sum to their parent?
    # After Stage 3 (IEA historical fit), some inconsistencies are EXPECTED
    # because we prioritise country-level IEA match over perfect sectorial sums.
    for scen in scenarios:
        check=sum(fun_inconsistencies(fun_xs(df, {'TARGET':[scen]}), [step5f_dict1,step5f_dict2], by_sector=False, by_country=True).values())
        if check>0:
            show_inconsistencies(fun_xs(fun_rename_index_name(df , {'TARGET':'SCENARIO'}), {'SCENARIO':scen, "REGION":iea_countries}), [step5f_dict1,step5f_dict2], absolute=True)
            print(f"WARNING: {scen}: Consistency issues found: {check} (expected after historical harmonization)")
        else:
            print(f"\n {scen}: Results are consistent!")

    # Check if sum across countries matches IAM regional totals
    # (This validates Stage 2 — sectorial harmonization was not undone)
    emiclockvar=set(fun_flatten_list(list(ds[0].values()))+fun_flatten_list(list(ds[1].values()))+list(ds[0].keys())+(list(ds[1].keys())))
    iamvar=set(df_iam.reset_index().VARIABLE.unique())
    checkvar=list(iamvar&emiclockvar)
    print("\n")
    print("## Check if sum across countries matches df_iam ##")
    for r,countrylist in regmap.items():
        for var in checkvar:
            dfsum=fun_xs(df, {"VARIABLE":var, "REGION":countrylist}).groupby(["TARGET"]).sum()
            iam=df_iam.xs((var, f"{model}|{r[:-1]}"), level=("VARIABLE","REGION")).groupby(["TARGET"]).sum()
            iam=fun_rename_index_name(fun_xs(iam, {"TARGET":scenarios}), {"SCENARIO":"TARGET"})[range(2010,2055,5)]
            dfsum=fun_index_names(dfsum, True, int)
            try:
                assert_frame_equal(np.round(dfsum.T,5), np.round(iam.T,5))
            except Exception as e:
                print(r, var, e, "\n")


    # Drop temporary variables from the final output
    df=df.drop(list(step5f_temp_vars_dict.keys()), level="VARIABLE")

    # Sort index for cleaner output
    df = df.sort_index()

    # Save final CSV
    df.to_csv(NESTED_DIR / csv_out)

    return df

# NOTE: you may need to re-run non-co2 emissions (step5c) using the   `run_multiple_file.py` before running this script (this will take approximately 10 mins)
if __name__ == "__main__":
    main(
        project="REMIND_fuel_mix_testing",
        csv_in = '17_02_2026', 
        step="step5",
        models=['REMIND *'],
        scenarios=["NPE-*"],
        harm_year = 2023,
        countrylist= None
              )
