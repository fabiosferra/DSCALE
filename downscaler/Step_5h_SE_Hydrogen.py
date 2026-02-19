# -----------------------------------
# PACKAGE IMPORTS -------------------
# -----------------------------------
import logging
import os
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt

from downscaler import CONSTANTS

from downscaler.utils import (
    fun_check_negative_energy_variables,
    fun_divide_variable_by_another,
    fun_flatten_list,
    fun_get_models,
    fun_get_regions,
    fun_invert_dictionary,
    fun_read_df_iams,
    fun_regional_country_mapping_as_dict,
    fun_xs_fuzzy,
    fun_wildcard,
    iea_countries,
    run_sector_harmo_enhanced_iamc,
    show_inconsistencies,
)
from downscaler.utils_dictionary import fun_append_list_of_dicts

from downscaler.utils_pandas import (
    fun_add_units,
    fun_create_var_as_sum,
    fun_get_variable_unit_dictionary,
    fun_index_names,
    fun_read_csv,
    fun_rename_index_name,
    fun_xs,
)

# -----------------------------------
# LOGGING CONFIG -------------------
# -----------------------------------
logging.basicConfig(level=logging.INFO)


def main(
    project_folder: str,
    file_suffix: str,
    list_of_targets: List[str] = ["*"],
    list_of_models: List[str] = ["*"],
    sel_reg: Optional[str] = None,
    harmonisation_date: int = 2022,
    input_file: Optional[str] = None,
    sectors: List[str] = None,
    carriers: List[str] = None,
) -> pd.DataFrame:
    """Calculate synthetic fuels per-country by computing the difference between
    total energy carrier and sum of sub-sectors.

    Parameters
    ----------
    project_folder : str
        Project folder name (e.g. 'REMIND_fuel_mix_testing')
    file_suffix : str
        File suffix used in the pipeline (e.g. 'fuel_mix_testing_03022026')
    list_of_targets : List[str], optional
        List of scenarios/targets to process, by default ["*"] (all)
    list_of_models : List[str], optional
        List of models to process, by default ["*"] (all)
    sel_reg : Optional[str], optional
        Region to be harmonized (e.g. 'LAMr'). None for all regions, by default None
    iterations : int, optional
        Number of harmonization iterations, by default 15
    synfuel : str, optional
        Name of synfuel variable component, by default 'Electricity'
    sanity_plot : bool, optional
        Whether to show sanity check plots, by default False
    harmonisation_date : int, optional
        Base year for harmonization, by default 2022
    input_file : Optional[str], optional
        Override input filename. If None, auto-generated from parameters, by default None
    sectors : List[str], optional
        List of sectors to process, by default ['Transportation', 'Industry', 'Residential and Commercial']
    carriers : List[str], optional
        List of carriers to process, by default ['Liquids', 'Gases']

    Returns
    -------
    pd.DataFrame
        Dataframe with synfuel calculations completed
    """
    # Set defaults for mutable arguments
    if sectors is None:
        sectors = ['Transportation', 'Industry', 'Residential and Commercial']
    if carriers is None:
        carriers = ['Liquids', 'Gases']

    project = project_folder

    # Read IAM data
    df_iam = fun_read_df_iams(project)
    df_iam = fun_index_names(df_iam, True, int)
    # Use nested directory structure
    mydir = CONSTANTS.NESTED_RES_DIR('step5', project_folder, file_suffix)

    # Get models
    models_all = fun_get_models(project)
    models = fun_wildcard(list_of_models, models_all)

    # Determine scenarios
    if list_of_targets == ["*"]:
        scenarios = df_iam.index.get_level_values('SCENARIO').unique().tolist()
    else:
        scenarios = list_of_targets

    # Determine input file
    if input_file is None:
        # Find the step5e output file matching pattern
        pattern = f"_{harmonisation_date}_harmo_step5g_synfuel.csv"
        step5_files = [x for x in os.listdir(mydir) if pattern in x]
        if not step5_files:
            raise FileNotFoundError(
                f"No step5g output file found matching pattern '*{pattern}' in {mydir}"
            )
        input_file = step5_files[0]

    logging.info(f"Reading input file: {input_file}")
    df = fun_read_csv({0: mydir / input_file}, True, int)[0]

    # Get regional mapping
    myres = {}
    for m in models:
        temp = fun_regional_country_mapping_as_dict(m, project)
        myres[m] = {f"{m}|{k}": v for k, v in temp.items()}
    _ = fun_invert_dictionary(fun_append_list_of_dicts(list(myres.values())))  # noqa: F841

    # Select countrylist based on sel_reg
    if sel_reg is not None:
        countrylist = myres[models[0]][f'{models[0]}|{sel_reg}']
    else:
        countrylist = None  # This will apply the harmonization to all regions/countries

    model = models[0]

    # Get available variables and regions from input data
    available_vars = df.index.get_level_values('VARIABLE').unique().tolist()
    available_regions = df.index.get_level_values('REGION').unique().tolist()

    total_var = f'Secondary Energy|Electricity'

    mydict = {
        total_var: [
            'Secondary Energy|Electricity|Biomass',
            'Secondary Energy|Electricity|Coal',
            'Secondary Energy|Electricity|Gas',
            'Secondary Energy|Electricity|Oil',
            'Secondary Energy|Electricity|Nuclear',
            'Secondary Energy|Electricity|Solar',
            'Secondary Energy|Electricity|Wind',
            'Secondary Energy|Electricity|Hydro',
            'Secondary Energy|Electricity|Geothermal',
            'Secondary Energy|Electricity|Hydrogen',
        ]
    }

    # Sectorial harmonization (for each model, region, scenario)
    regmap = fun_get_regions(project, model)

    # Filter regmap to only include regions that have countries in our data
    regmap = {
        k: [c for c in v if c in available_regions]
        for k, v in regmap.items()
    }
    # Remove empty regions
    regmap = {k: v for k, v in regmap.items() if v}

    if countrylist is not None:
        regmap = {
            k: v for k, v in regmap.items()
            if len(set(v) & set(countrylist)) > 0
        }

    for r, region_countrylist in regmap.items():
        for scen in scenarios:
            # Check if this scenario exists in the data for these regions
            try:
                df1_check = fun_xs(df, {'SCENARIO': scen, 'REGION': region_countrylist})
                if df1_check.empty:
                    logging.info(f"No data for {scen} in region {r}, skipping")
                    continue
            except Exception:
                logging.info(f"Could not slice data for {scen} in region {r}, skipping")
                continue

            df1 = fun_xs(df, {'SCENARIO': scen, 'REGION': region_countrylist})
            df2 = fun_rename_index_name(
                fun_xs(df_iam, {'REGION': f"{model}|{r[:-1]}", 'SCENARIO': scen}).drop(2005, axis=1),
                {'SCENARIO': 'TARGET'}
            )

            # Slice for specific variables
            vars_in_df1 = df1.index.get_level_values('VARIABLE').unique().tolist()
            vars_to_agg = mydict[total_var]

            if total_var not in vars_in_df1:
                logging.info(f"'{total_var}' not in data for {scen}/{r}, skipping")
                break  # Skip all iterations for this scenario/region

            # If Hydrogen already available in df1, move to next region
            if "Secondary Energy|Electricity|Hydrogen" in vars_in_df1:
                continue

            # Compute ratio in IAM region between Hydrogen and total
            IAM_SE_Total = fun_xs(df2, {'VARIABLE': total_var})
            IAM_SE_Hydrogen = fun_xs(df2, {'VARIABLE': f"{total_var}|Hydrogen"})
            IAM_hydrogen_ratio = IAM_SE_Hydrogen.droplevel("VARIABLE")/IAM_SE_Total.droplevel("VARIABLE")

            # Apply the ratio to the countries within region.
            # Align IAM ratio columns to df1 (df1 may include non-5-year columns
            # like 2022 that are absent from df_iam; interpolate to fill those gaps).
            df1_slice = fun_xs(df1, {'VARIABLE': total_var}).droplevel("VARIABLE")
            IAM_hydrogen_ratio_aligned = (
                IAM_hydrogen_ratio
                .reindex(columns=df1_slice.columns)
                .interpolate(axis=1)
            )
            df1_hydrogen = df1_slice * IAM_hydrogen_ratio_aligned.values
            df1_hydrogen["VARIABLE"] = "Secondary Energy|Electricity|Hydrogen"
            df1_hydrogen = df1_hydrogen.reset_index().set_index(['MODEL', 'SCENARIO', 'REGION', 'VARIABLE', 'UNIT'])

            # Create a dataframe with all fuels within Power
            df1_agg = pd.concat([
                fun_xs(df1, {'VARIABLE': vars_to_agg}), 
                df1_hydrogen
            ])

            # Recompute Total Power sector
            df1_total_power = df1_agg.groupby(["MODEL", "SCENARIO", "REGION", "UNIT"]).sum()
            df1_total_power["VARIABLE"] = total_var
            df1_total_power = df1_total_power.reset_index().set_index(['MODEL', 'SCENARIO', 'REGION', 'VARIABLE', 'UNIT'])

            updated_power_sector = pd.concat([df1_total_power, df1_agg])

            # Update df 
            updated_results = fun_index_names(updated_power_sector, True, int)
            common_variables = list(set(updated_power_sector.index) & set(df.index))
            df = pd.concat([df.drop(common_variables), updated_results])

    # Save results
    output_file = f"{model}_{harmonisation_date}_harmo_step5h_hydrogen.csv"
    output_path = mydir / output_file
    df.to_csv(output_path)
    logging.info(f"Results saved to {output_path}")

    return df

if __name__ == "__main__":
    # Example standalone execution with hardcoded values, otherwise is executed via run multiple files
    main(
        project_folder="REMIND_fuel_mix_testing",
        file_suffix="10_02_2026",
        list_of_targets=['NPE-core'],
        list_of_models=["*"],
        sel_reg='LAMr',
        harmonisation_date=2022,
    )