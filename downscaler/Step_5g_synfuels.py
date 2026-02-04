"""
This script calculates the synthetic fuels per-country, by looking at the difference between the
sum of current fuels (e.g. biomass + oil) and the total energy carrier (e.g. liquids),
and ascertaining the 'missing' synfuels as a result

MORE DETAILS:
We create this `synfuel` variable as the difference between total liquids and the sum of sub-sectors
(excluding electricity). Then we harmonize the results by using the `run_sector_harmo_enhanced_iamc`
(that ensures consistency with regional IAMs results if provided)

NOTE: if you want to add historical harmonization you need to be sure that `df_iea` and `df_iam`
are also fully aligned (otherwise you may create inconsistencies)
NOTE: Given that synfuel='Electricity' this code does not work for energy_carrier= Electricity
"""

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


# -----------------------------------
# FUNCTION DEFINITIONS---------------
# -----------------------------------

def reshape_df_after_harmo_in_IAMC_format(df_iam, df):
    """Reshape dataframe after harmonization to IAMC format.

    Handles METHOD column/index removal, variable renaming, unit fixing,
    and renames TARGET to SCENARIO.
    """
    if 'METHOD' in df.columns:
        df = df.drop('METHOD', axis=1)
    elif 'METHOD' in df.index.names:
        df = df.droplevel('METHOD')

    # Rename variables
    mydict = {x: x.replace('LONG_TERM', '') for x in df.index.get_level_values('VARIABLE')}
    df = df.rename(mydict, axis=0)

    # Fix units
    units = fun_get_variable_unit_dictionary(df_iam)
    d = {k: [v] for k, v in units.items()}
    df = fun_add_units(df, df_iam, fun_invert_dictionary(d))

    # Rename target as scenario
    return fun_rename_index_name(df, {"TARGET": "SCENARIO"})


def main(
    project_folder: str,
    file_suffix: str,
    list_of_targets: List[str] = ["*"],
    list_of_models: List[str] = ["*"],
    sel_reg: Optional[str] = None,
    iterations: int = 15,
    synfuel: str = 'Electricity',
    sanity_plot: bool = False,
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
    mydir = CONSTANTS.CURR_RES_DIR('step5')

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
        pattern = f"_{project_folder}_{file_suffix}_{harmonisation_date}_harmo_step5e_None.csv"
        step5_files = [x for x in os.listdir(mydir) if pattern in x]
        if not step5_files:
            raise FileNotFoundError(
                f"No step5e output file found matching pattern '*{pattern}' in {mydir}"
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

    for sector in sectors:
        for carrier in carriers:
            total_var = f'Final Energy|{sector}|{carrier}'

            # Check if the total carrier variable exists in the data
            if total_var not in available_vars:
                logging.info(f"Skipping {sector}|{carrier}: '{total_var}' not found in input data")
                continue

            mydict = {
                total_var: [
                    f'Final Energy|{sector}|{carrier}|Biomass',
                    f'Final Energy|{sector}|{carrier}|Coal',
                    f'Final Energy|{sector}|{carrier}|Electricity',
                    f'Final Energy|{sector}|{carrier}|Natural Gas',
                    f'Final Energy|{sector}|{carrier}|Oil'
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

            if not regmap:
                logging.info(f"No regions with data found for {sector}|{carrier}")
                continue

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

                    for iter_num in range(iterations):
                        ds = [mydict]
                        logging.info(f'Harmonizing {model} {r} {scen} ITERATION {iter_num + 1}')

                        df1 = fun_xs(df, {'SCENARIO': scen, 'REGION': region_countrylist})
                        df2 = fun_rename_index_name(
                            fun_xs(df_iam, {'REGION': f"{model}|{r[:-1]}", 'SCENARIO': scen}).drop(2005, axis=1),
                            {'SCENARIO': 'TARGET'}
                        )

                        # Slice for specific variables
                        all_vars = list(set(fun_flatten_list(list(mydict.values()) + list([mydict.keys()]))))
                        # Filter to only variables that exist in df1
                        vars_in_df1 = df1.index.get_level_values('VARIABLE').unique().tolist()
                        vars_to_slice = [x for x in all_vars if synfuel not in x and x in vars_in_df1]

                        if total_var not in vars_in_df1:
                            logging.info(f"'{total_var}' not in data for {scen}/{r}, skipping")
                            break  # Skip all iterations for this scenario/region

                        df1 = fun_xs(df1, {'VARIABLE': vars_to_slice})
                        df2 = fun_xs(df2, {'VARIABLE': all_vars})

                        # Create Synfuel variable as the difference between total and sum of sub-sectors
                        temp = fun_create_var_as_sum(
                            df1,
                            'sum of subs',
                            [x for x in fun_flatten_list(mydict.values()) if synfuel not in x],
                            unit='EJ/yr'
                        )
                        temp = fun_divide_variable_by_another(
                            temp,
                            f'Final Energy|{sector}|{carrier}|{synfuel}',
                            f'Final Energy|{sector}|{carrier}',
                            'sum of subs',
                            operator='-',
                            unit='EJ/yr',
                            concatenate=True
                        ).clip(1e-9)

                        # Harmonize results (top-down harmonization)
                        temp = run_sector_harmo_enhanced_iamc(
                            project, temp, ds, df2, verbose=False, no_iter=1
                        )
                        temp = reshape_df_after_harmo_in_IAMC_format(df_iam, temp)

                        # Drop the temporary variable 'sum of subs'
                        temp = temp.drop('sum of subs', level='VARIABLE')

                        # Sanity check
                        fun_check_negative_energy_variables(temp)
                        updated_results = temp.reset_index().set_index(df.index.names)

                        # Update df after harmonization
                        updated_results = fun_index_names(updated_results, True, int)
                        common_variables = list(set(updated_results.index) & set(df.index))
                        df = pd.concat([df.drop(common_variables), updated_results])

            logging.info(f'Harmonization completed for {sector}|{carrier}')

            # Sanity check plots
            if sanity_plot:
                print(show_inconsistencies(fun_xs(temp, {'REGION': iea_countries}), ds))

                plt.style.use('seaborn-white')
                for x in [f'Final Energy|{sector}|{carrier}|Biomass', f'Final Energy|{sector}|{carrier}|Electricity']:
                    down_liquids = fun_xs_fuzzy(fun_xs(df, {'REGION': region_countrylist}), [x, scen])
                    iam_liquids = fun_xs_fuzzy(fun_xs(df_iam, {'REGION': [f'{models[0]}|LAM']}), [x, scen])

                    check = pd.concat([iam_liquids.sum().T, down_liquids.sum().T], axis=1)
                    check = check.rename(columns={0: 'IAM', 1: 'downs'})

                    plt.figure()
                    plt.plot(check['IAM'], linestyle='-', label='IAM', alpha=0.8, lw=3)
                    plt.plot(check['downs'], linestyle='--', label='Downscaled (sum across countries)', alpha=1)
                    plt.legend()
                    plt.title(f'{x} - {scen} - LAM')
                    plt.xlabel('Index')
                    plt.ylabel('Values')
                    plt.show()

    # Save results
    output_file = f"{model}_{project_folder}_{file_suffix}_{harmonisation_date}_harmo_step5g_synfuel.csv"
    output_path = mydir / output_file
    df.to_csv(output_path)
    logging.info(f"Results saved to {output_path}")

    return df


if __name__ == "__main__":
    # Example standalone execution with hardcoded values, otherwise is executed via run multiple files
    main(
        project_folder="REMIND_fuel_mix_testing",
        file_suffix="fuel_mix_testing_03022026",
        list_of_targets=['NPE-core'],
        list_of_models=["*"],
        sel_reg='LAMr',
        iterations=15,
        synfuel='Electricity',
        sanity_plot=False,
        harmonisation_date=2022,
    )
