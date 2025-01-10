from typing import Union, Optional

import os
import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from downscaler import CONSTANTS
from downscaler.fixtures import (
    check_consistency_dict,
    iea_var_dict,
    iea_flow_dict,
    dashboard_bar_plots,
    list_emission_by_sectors_step5b,
    ngfs_2023_nomenclature,
    step5e_harmo,
    sel_plot_vars_step5e,
    vars_to_be_harmo_step2b,  # for SSP_2023,
    main_emi_sectors,
)
from downscaler.utils_pandas import (
    fun_select_model_scenarios_combinations,
    fun_index_names,
    fun_get_variable_unit_dictionary,
    fun_check_scenarios_with_missing_energy_data,
    fun_replace_zero_columns_with_na,
    fun_get_native_vs_downs_results,
    fun_add_units,
    fun_drop_missing_units,
)
from downscaler.utils import (
    fun_add_share_of_hydrogen,
    fun_sns_lineplot,
    fun_create_var_as_sum,
    fun_sectoral_checks,
    fun_eu28,
    fun_plot_downscaled_results,
    fun_stackplot_downscaled_results,
    fun_plot_fill_between,
    fun_select_eu_scenarios,
    fun_select_passed_c1_scenarios,
    fun_add_variables_and_harmonize,
    fun_add_trade_variables,
    fun_invert_dictionary,
    fun_read_df_iam_from_multiple_df,
    fun_flatten_list,
    fun_xs,
    fun_add_cdr_variables,
    fun_available_scen,
    fun_plot_eu_ab_side_by_side,
    fun_get_sub_sectors,
    fun_bar_plot_stacked,
    fun_scatter_plot,
    fun_interpolate,
    fun_sns_lineplot_new,
    prepare_data_for_sns_lineplot,
    fun_add_non_biomass_ren_nomenclature,
    fun_drop_duplicates,
    fun_create_var_as_sum_only_for_models_where_missing,
    fun_save_to_excel,
    fun_read_df_countries,
    load_model_mapping,
    fun_aggregate_countries,
    fun_eu27,
    fun_country_map,
    get_git_revision_short_hash,
    fun_check_negative_energy_variables,
    fun_check_negative_energy_variables_by_model,
    fun_aggregate_non_iea_countries_all_models_scen_var,
    fun_aggregate_countries_general,
    fun_check_missing_data,
    SliceableDict,
    fun_find_nearest_values,
    fun_most_recent_iea_data,
    fun_xs_fuzzy,
    convert_time,
    get_native_countries,
)
from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib as plt
from downscaler.utils import fun_blending

def main(
    files: list,
    models_dict: list,
    sel_vars: Union[tuple, list] = ("Emissions|CO2|Energy"),
    add_hist_data: bool = False,
    show_plots: bool = True,
    create_dashboard: bool = True,
    target_var: str = "GHG incl. International transport (intra-eu only)",  # "GHG incl. International transport",
    eu_2030_kyoto: float = 2085,
    eu_2050_kyoto: float = 300,
    use_eea_data: bool = True,
    use_iea_data_from_ed: bool = True,
    harmonize_eea_data_until: int = 2019,
    harmo_vars: list = ["Emissions|CO2"],
    csv_file_name: Union[None, str] = None,
    interpolate: bool = True,
    harmonize_hist_data: bool = True,
    selected_scenarios: Optional[Union[str, list]] = None,  # "Ed_pass_scenarios_category_c1b.csv",
    selected_regions: Union[None, list] = ["EU27"],
    plot_sel_model_dict: Union[dict, None] = None,
    exclude_emissions_by_sector_from_step5b: bool = True,
    feasibility_file_name: Union[None, str] = None,  # "feasibility_2050_type1.csv"
    search_in_input_data_native_iam_results: bool = False,
    read_from_step4: Optional[str] = None,
    coerce_errors: bool = False,
    replace_zero_columns_with_na: bool = False,
    known_issues:Optional[dict]=None,
    aggregate_non_iea_countries:bool=False,
    aggregate_eu_27:bool=False,
    keep_step5_emissions=None,
    # ['Emissions|CO2|Energy|Demand|Industry',
    #                       'Emissions|CO2|Energy|Demand|Residential and Commercial',
    #                       'Emissions|CO2|Energy|Demand|Transportation'
    #                       ]
) -> pd.DataFrame:  # -> pd.DataFrame:

    """This function:
    - Reads in both downscaled vs native EU27 results from different projects
    - Harmonizes the data to match historical data/inventories until a given base year `harmonize_eea_data_until`
    - Creates some dashboard and visualizations (work in progress).

    Returns
    -------
    pd.DataFrame
        Dataframe with harmonized results

    Raises
    ------
    ValueError
        If cannot find data for any of the model you are trying to read in (df is empty).
    ValueError
        If cannot find data for a specific model (either native or downscaled).
    """

    # NOTE: This file
    # 1) tries to read downscaled results from step5 directory.
    # 2) if 1) is not available, tries to read native IAMs results from step5
    # 3) if 2) is not available and `search_in_input_data_native_iam_results=True` tries to read native IAMs fom input data, interpolates 2015-2020 and saves data in step5 (no need to interplate next time, leads to faster code)

    print("Running step5e...")

    CURR_RES_DIR = CONSTANTS.CURR_RES_DIR("step5")
    idx_col = [
        "MODEL",
        "SCENARIO",
        "REGION",
        "VARIABLE",
        "UNIT",
    ]
    models = set(fun_flatten_list(models_dict.values()))
    df = pd.DataFrame()
    for model in models:
        if read_from_step4:
            project = fun_invert_dictionary(models_dict)[model][0]
            file = files[0]
            # read from step4
            df = pd.read_csv(
                CONSTANTS.CURR_RES_DIR("step4") / f"{model}_{read_from_step4}_{file}"
            )
            df = df.rename(columns={"ISO": "REGION"})
            df_iam = fun_index_names(
                pd.read_csv(
                    CONSTANTS.INPUT_DATA_DIR / project / "snapshot_all_regions.csv"
                )
            )

            # Bring back original names e.g. "Carbon Sequestration|CCS|Biomass"
            df_iam = df_iam.rename(ngfs_2023_nomenclature, level="VARIABLE")
            df_iam = fun_add_non_biomass_ren_nomenclature(df_iam, inverse=False)[0]

            # Block below should do same as commented block below
            extra_units = {
                "EJ/yr": ["Secondary Energy|Electricity|Trade"],
                "Mt CO2/yr": [
                    "Emissions|CO2|LULUCF Direct+Indirect",
                    "Emissions|CO2|LULUCF Indirect",
                ],
                "Mt CO2-equiv/yr": ["Emissions|Total Non-CO2"],
            }
            # if "UNIT" not in list(df.columns) + list(df.index.names):
            df = fun_add_units(df, df_iam, extra_units, missing_unit="missing")
            # The below drops all the statistical variable and specific-downscaling variables (to be calculate again in step5e)
            df = fun_drop_missing_units(df)

            df["FILE"] = f"{model}_{project}_{file}"
        else:
            # Read downscaled results (could consider using `fun_read_csv`) from Step5

            for file in files:
                for project in fun_invert_dictionary(models_dict)[model]:
                    suf = f"{model}_{project}_{file}"
                    # read from step5 folder
                    file_path = f"{CURR_RES_DIR}/{suf}"
                    # NOTE If file_path does not exists then try to read from input_data (native IAMs results)

                    if (
                        search_in_input_data_native_iam_results
                        and not os.path.exists(file_path)
                        and file == "snapshot_all_regions.csv"
                    ):
                        # NOTE We do this only if a model is not present yet in the df
                        dfr = df.reset_index()
                        if (
                            "MODEL" not in dfr.columns
                            or model not in dfr.MODEL.unique()
                        ):
                            file_path = (
                                CONSTANTS.INPUT_DATA_DIR
                                / project
                                / model
                                / f"{file.split('.')[0]}_RAW_{model}.csv"
                            )
                    if os.path.exists(file_path):
                        current_file = pd.read_csv(
                            file_path,
                            index_col=idx_col,
                        )
                        # print("using native results")
                        if (
                            search_in_input_data_native_iam_results
                            and (
                                harmonize_eea_data_until
                                and str(harmonize_eea_data_until)
                                not in current_file.columns
                                or harmonize_eea_data_until in current_file.columns
                            )
                            and len(fun_xs(current_file, {"REGION": selected_regions}))
                        ):
                            # we need IAMs results with 2019 data -> if this is not present will try to interpolate
                            print(f"interpolating {model} from 2015-2020...")
                            current_file = fun_interpolate(
                                fun_index_names(current_file, True, int),
                                False,
                                range(2015, 2021),
                                interpolate_columns_present=True,
                            )
                            txt = f"would you like to save the interpolated {model} to step5 directory?"
                            action = input(txt)
                            if action.lower() in ["yes", "y"]:
                                current_file.to_csv(
                                    f"{CURR_RES_DIR}/{model}_{project}_{file}"
                                )
                        # current_file["FILE"]=f"{project}_{file}"
                        current_file["FILE"] = file_path.split("/")[-1]
                        df = pd.concat([df, current_file], axis=0, sort=True)
                    else:
                        print(f"{model}_{project}_{file} does not exists")
            if (
                "MODEL" in df.reset_index().columns
                and model not in df.reset_index().MODEL.unique()
            ):
                raise ValueError(f"Cannot find data for {model}")

    if isinstance(selected_scenarios, list):
        df = fun_xs(df, {"SCENARIO": selected_scenarios})
        selected_scenarios=None # Sot that the rest of the code will behave as before
    if selected_regions is not None:
        df = fun_xs(df, {"REGION": selected_regions})

    if not (len(df)):
        raise ValueError("`df` is empty - we could not find/read any model results")

    # Print Native vs downscaled results
    model_res_type = fun_get_native_vs_downs_results(df)
    print("\n", model_res_type, "\n")

    df = fun_add_non_biomass_ren_nomenclature(df, inverse=True)[0]

    fun_check_negative_energy_variables_by_model(
        df, coerce_errors=True, sel_cols=range(2010, 2060, 5)
    )

    fun_check_scenarios_with_missing_energy_data(df)

    # TEMPORARY!!!
    # fun_sectoral_checks(models_dict,df.drop('EU27', level='REGION'))

    # fun_create_regional_variables_as_sum_of_countrylevel_data(
    #     model,
    #     project,
    #     df.drop("FILE", axis=1),
    #     ["Emissions|CO2|LULUCF Direct+Indirect"],
    # ).reset_index().REGION.unique()
    
    if keep_step5_emissions is None:
        keep_step5_emissions=[]
    if exclude_emissions_by_sector_from_step5b: 
        if len(keep_step5_emissions)>0:
            txt="If you want to exclude step5b emissions,  `keep_step5_emissions` must be either an empty list or None"
            raise ValueError(f'{txt}. You provided: {keep_step5_emissions}')
    df=fun_keep_selected_step5b_emi_variables(keep_step5_emissions, 
                                                    df, 
                                                    model, 
                                                    project, 
                                                    exclude_emissions_by_sector_from_step5b,
                                                    list_emission_by_sectors_step5b
                                                    )
    
    (
        # df,
        df_merged,
        # hist_data_all_countries,
        harmo_str,
        selcols,
        df_merged_not_harmo,
    ) = fun_add_variables_and_harmonize(
        sel_vars,
        add_hist_data,
        use_eea_data,
        use_iea_data_from_ed,
        harmonize_eea_data_until,
        harmo_vars,
        interpolate,
        harmonize_hist_data,
        idx_col,
        df,
        iea_flow_dict,
        iea_var_dict,
        keep_step5_emissions,
    )

    # If `x` is not present in the downscaled results, we calculate it as the sum of sub-sectors using the dictionary `main_emi_sectors`
    
    # df_merged=fun_keep_selected_step5b_emi_variables(keep_step5_emissions, 
    #                                                 df_merged, 
    #                                                 model, 
    #                                                 project, 
    #                                                 exclude_emissions_by_sector_from_step5b,
    #                                                 list_emission_by_sectors_step5b
    #                                                 )



    # Add "Revenue|Government|Tax|Carbon" as the sum of sectorial revenues (if available)

    df_merged = fun_calculate_carbon_revenues_as_sum_subsectors(df_merged)

    # Check negative Energy variables
    coerce_neg = {"downscaled results": coerce_errors, "native results": True}
    for k, v in coerce_neg.items():
        if len(model_res_type[k]):
            for model in model_res_type[k]:
                print(
                    model,
                    fun_check_negative_energy_variables(
                        fun_xs(df_merged, {"MODEL": model}), coerce_errors=v, known_issues=known_issues
                    ),
                )

    # Dropping variables as discussed in https://github.com/iiasa/ngfs-phase-4-internal-workflow/pull/39
    # and https://github.com/iiasa/ngfs-phase-4-internal-workflow/blob/master/definitions/variable/downscaling.yaml#L1-L2
    drop_variables = [
        "Final Energy|Transportation|Liquids|Bioenergy",
        "Final Energy|Transportation|Liquids|Natural Gas",
        "Final Energy|Gases|Biomass",
        "Final Energy|Gases|Coal",
        "Final Energy|Gases|Natural Gas",
        "Final Energy|Gases|Other",
        "Final Energy|Industry|Gases|Coal",
        "Final Energy|Industry|Gases|Other",
        "Final Energy|Industry|Liquids|Gas",
        "Final Energy|Liquids|Biomass",
        "Final Energy|Liquids|Coal",
        "Final Energy|Liquids|Gas",
        "Final Energy|Liquids|Oil",
        "Final Energy|Residential and Commercial|Gases|Coal",
        "Final Energy|Residential and Commercial|Gases|Other",
        "Final Energy|Residential and Commercial|Liquids|Gas",
        "Final Energy|Transportation|Gases|Biomass",
        "Final Energy|Transportation|Gases|Coal",
        "Final Energy|Transportation|Gases|Natural Gas",
        "Final Energy|Transportation|Gases|Other",
        "Final Energy|Transportation|Liquids|Biomass",
        "Final Energy|Transportation|Liquids|Gas",
        "Secondary Energy|Industry",
        "Secondary Energy|Industry|Solids|Biomass",
        "Secondary Energy|Industry|Solids|Coal",
        "Secondary Energy|Residential and Commercial",
        "Secondary Energy|Residential and Commercial|Solids|Biomass",
        "Secondary Energy|Residential and Commercial|Solids|Coal",
    ]
    df_merged = fun_xs(df_merged, {"VARIABLE":drop_variables}, exclude_vars=True)

    # The below is just for NGFS MESSAGE (no data reported for Residential and Commercial|Hydrogen)
    if (
        "Final Energy|Residential and Commercial|Hydrogen"
        not in df_merged.reset_index().VARIABLE.unique()
    ):
        # We report Residential and Commercial|Hydrogen as zero
        df_merged = fun_create_var_as_sum(
            df_merged,
            "Final Energy|Residential and Commercial|Hydrogen",
            {"Final Energy": 0},
            unit="EJ/yr",
        )
    if csv_file_name:
        # Replace zeroes with np.nan if previous next columns contains data that is not equal to zero. For each model and variable
        if replace_zero_columns_with_na:
            df_merged = fun_replace_zero_columns_with_na(df_merged)

        file_tocsv = (
            CURR_RES_DIR
            / f"{csv_file_name}_{harmonize_eea_data_until}_{harmo_str}_step5e"
        )
        # NOTE The below is for all projects except for the 'EU_climate_advisory_board'
        if not len([x for x in models_dict.keys() if "EU_climate_advisory_board" in x]):
            # if "NGFS_2023" in models_dict.keys():
            # Drop international transport variables
            int_transp = [
                "GHG incl. International transport",
                "GHG incl. International transport (intra-eu only)",
            ]
            # df_merged = df_merged.drop(int_transp, level="VARIABLE")
            df_merged = fun_xs(df_merged, {"VARIABLE":int_transp}, exclude_vars=True)

            # NOTE Calculate statistical difference
            # not all variables report a statistical difference, because some variables have been created after the historical data harminization
            df_statistical_difference = (
                (df_merged_not_harmo - df_merged).replace(0, np.nan).dropna(how="all")
            )
            stat_diff_dict = {
                x: f"Statistical Difference|{x}"
                for x in df_statistical_difference.reset_index().VARIABLE.unique()
            }
            df_merged = pd.concat(
                [df_merged, df_statistical_difference.rename(index=stat_diff_dict)],
                axis=0,
            )

            # df_merged = df_merged.drop(
            #     "Statistical Difference|Carbon Sequestration|CCS", level="VARIABLE"
            # )
            # df_merged = df_merged.drop(
            #     "Statistical Difference|Secondary Energy|Electricity|Trade",
            #     level="VARIABLE",
            # )

            # df_merged = df_merged.drop(
            #     "Statistical Difference|Emissions|CO2|Energy and Industrial Processes",
            #     level="VARIABLE",
            # )
            tobedropped=["Statistical Difference|Emissions|CO2|Energy and Industrial Processes",
                          "Statistical Difference|Carbon Sequestration|CCS",
                          "Statistical Difference|Secondary Energy|Electricity|Trade"
                         ]

            df_merged=fun_xs(df_merged, {"VARIABLE":tobedropped}, exclude_vars=True)

            models = df_merged.reset_index().MODEL.unique()
            if len(models == 1):
                # We put the model name as prefix -> f"{MODEL}_csv_file_name"
                file_tocsv = (
                    CURR_RES_DIR
                    / f"{models[0]}_{csv_file_name}_{harmonize_eea_data_until}_{harmo_str}_step5e"
                )
                if read_from_step4:
                    file_tocsv = f"{file_tocsv}_WITH_POLICY"

            selcols = [x for x in selcols if x in range(2010, 2105, 5)]

            # Drop eu 27 as we did some harmonization for NGFS after step 5d
            # (and the primap dataset does not contain eu27 as a region. only eea_data has it)
            if "EU27" in df_merged.reset_index().REGION.unique():
                df_merged = df_merged.drop("EU27", level="REGION")

            # Just for ENGAGE below:
            if project == "ENGAGE_2023":
                selcols = selcols + [harmonize_eea_data_until]
                selcols.sort()
            if "FILE" in df_merged.index.names:
                df_merged = df_merged.droplevel("FILE")
            reg_available=df_merged.reset_index().REGION.unique()
            if 'EU27' in reg_available:
                # Drop EU27 as we did some harmonization in step5e
                df_merged=df_merged.drop("EU27", level="REGION")
            if aggregate_eu_27: 
                if any([True if x in reg_available else False for x in fun_eu27() ]):
                    if not all([True if x in reg_available else False for x in fun_eu27()]):  
                        missing_eu_countries=[x for x in fun_eu27() if x not in reg_available]
                        available_eu_countries=[x for x in fun_eu27() if x in reg_available]
                        txt="Unable to calculate the EU27 aggregate: some EU countries are present in the df"
                        txt2=", while these ones are missing"
                        raise ValueError(f'{txt} ({available_eu_countries}) {txt2}: {missing_eu_countries}')
                    df_merged = fun_aggregate_countries_general(df_merged,"EU27", fun_eu27())

            df_merged = df_merged.rename(
                index={v: k for k, v in ngfs_2023_nomenclature.items()}
            )
            if read_from_step4:
                # Only for the final excel file
                # USE Emissions|Kyoto Gases (incl. indirect LULUCF) instead of `(incl. indirect AFOLU)`
                afoludict = {
                    "Emissions|Kyoto Gases (incl. indirect AFOLU)": "Emissions|Kyoto Gases (incl. indirect LULUCF)"
                }
                df_merged = df_merged.rename(index=afoludict)
            df_countries = fun_read_df_countries(
                CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
            )
            df_countries, regions = load_model_mapping(
                model,
                df_countries,
                CONSTANTS.INPUT_DATA_DIR / project / "default_mapping.csv",
            )
            # Below exclude r in regions (will be added later)
            regions = [x[:-1] for x in regions]
            df_merged = fun_add_non_biomass_ren_nomenclature(df_merged, inverse=True)[0]

            df_csv = df_merged[selcols]
            
            if aggregate_non_iea_countries:
                # NOTE this block is not present twice. It is different from the one below
                df_csv = fun_aggregate_non_iea_countries_all_models_scen_var(project, df_csv)
            df_csv = np.round(df_csv, 4)

            # rename models as "Downscaling[REMIND ...]"
            rename_dict = {
                x: f"Downscaling[{x}]" for x in df_csv.reset_index().MODEL.unique()
            }
            df_csv = fun_index_names(df_csv, True, int).rename(index=rename_dict)

            # Dump yaml file with downscaling units
            myunits = fun_get_variable_unit_dictionary(df_csv)
            f = open(CONSTANTS.CURR_RES_DIR("step5") / "downscaling_units.yaml", "w+")
            yaml.dump(myunits, f, allow_unicode=True)

            # models = df.reset_index().MODEL.unique()
            # if not use_eea_data:
            # drop eu 27 as we did some harmonization for NGFS after step 5d
            # (and the primap dataset does not contain eu27 as a region. only eea_data has it)

            fun_save_to_excel(
                project,  # "NGFS_2023",
                f"{file_tocsv}_Scenario_Explorer_upload_non_iea_by_region.csv",
                model,
                df_csv,
                None,
            )

            if aggregate_non_iea_countries:
                # NOTE this block is not present twice. It is different from the one above.
                # Sum up all non-iea regions 
                # all_regions=df_csv.reset_index().REGION.unique()
                non_iea_regions=[x for x in df_csv.reset_index().REGION.unique() if 'Non-IEA' in x]
                df_csv=fun_aggregate_countries_general(df_csv,"downscaling|countries without IEA statistics", non_iea_regions, remove_single_countries=True)
            # iea_regions=list(set(all_regions)^set(non_iea_regions))
            # non_iea=fun_aggregate_countries_general(df_csv,"downscaling|countries without IEA statistics", non_iea_regions)
            # non_iea=non_iea.xs("downscaling|countries without IEA statistics", level='REGION', drop_level=False)
            # df_csv=pd.concat([fun_xs(df_csv, {"REGION":iea_regions}), non_iea])

            # Create EU27 region
            if aggregate_eu_27: 
                # NOTE this block is present twice and probably is not needed
                df_csv=fun_aggregate_countries_general(df_csv,"EU27", fun_eu27())
            
            # Check if there are still missing data
            if project=='' and 'test_data' in str(CONSTANTS.INPUT_DATA_DIR):
                print('Skipping missing data check, if this is a test case')
            else:
                for var in ['Final Energy', 'Primary Energy','GDP|PPP','Population']:
                    missing=fun_check_missing_data(df_csv.xs(var, level='VARIABLE'), 'REGION', 'SCENARIO')
                    if len(missing)>0:
                        raise ValueError(f'Missing {var} data for {missing}')
            
            fun_save_to_excel(
                project,  # "NGFS_2023",
                f"{file_tocsv}_Scenario_Explorer_upload.csv",
                model,
                df_csv,
                None,
            )
        df_csv = df_merged[selcols]
        if "FILE" in df_merged.index.names:
            df_csv = df_csv.droplevel("FILE")

        # Rename variables
        df_csv = df_csv.rename(
            {
                "GHG incl. International transport": "Emissions|Kyoto Gases (AR4) (EEA)",
                "GHG incl. International transport (intra-eu only)": "Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)",
                # "Emissions|Kyoto Gases (incl. indirect AFOLU)":"Emissions|Kyoto Gases (incl. indirect LULUCF)"
            },
            level="VARIABLE",
        )

        df_csv = np.round(df_csv, 4)
        df_csv.to_csv(f"{file_tocsv}_{selected_regions}.csv")  # all countries

    # Exit the function earlier if we run NGFS_2023 (return below)
    if not create_dashboard and not show_plots:
        print("step5e done!")
        if "FILE" in df_merged.index.names:
            df_merged = df_merged.droplevel("FILE")
        return df_csv if csv_file_name else df_merged[selcols]

    # Add CDR, Share of hydrogen in fen and Imports variables (moving this up will change test case)
    for f in fun_add_cdr_variables, fun_add_trade_variables, fun_add_share_of_hydrogen:
        df_merged = f(df_merged)

    # Remove scenarios in `backlist_scen`
    backlist_scen = ["PR_EU_NewCurPol21_EI"]
    for x in backlist_scen:
        if x in df_merged.reset_index().SCENARIO.unique():
            df_merged = df_merged.drop(x, level="SCENARIO")

    # if 'PR_EU_NewCurPol21_EI' in df_merged.reset_index().SCENARIO.unique():
    #     df_merged=df_merged.drop('PR_EU_NewCurPol21_EI', level='SCENARIO')

    res = fun_selected_scenarios(
        eu_2030_kyoto,
        eu_2050_kyoto,
        selected_scenarios,
        df_merged,
        var=target_var,
    )
    selcols = list(range(2010, 2055, 5))
    if harmonize_eea_data_until:
        selcols = [harmonize_eea_data_until] + selcols
        selcols.sort()

    if create_dashboard:
        dashboard_dir = CONSTANTS.CURR_RES_DIR("step5") / "Step_5e_dashboard"
        for x in ["selected C1 scenarios", "selected C1 scenarios + eu targets"]:
            if "Gross Emissions|CO2" in res[x].reset_index().VARIABLE.unique():
                res[x] = res[x].drop("Gross Emissions|CO2", level="VARIABLE")
            fun_create_dashboard_pdf(
                res[x].loc[:, selcols],
                dashboard_bar_plots,
                selected_regions,
                # dashboard_dir/csv_file_name/x,
                dashboard_dir / x,
                f"EU_ab_dashboard_{x}_{harmonize_eea_data_until}_{harmo_str}.pdf",
            )

    # NOTE: # "viridis"  # "rocket"# "mako" my preferred
    sel_palette = [
        "husl",
        "RdBu",
        "viridis",
        "rocket",
        "mako",
        "Paired",
        "dark",
        "hls",
        "tab10",
    ]
    subtitle = None  # f"The graph visualizes downscaled results for the following countries (ISO3 codes): \n {fun_eu28()}"

    # Florian's graph
    # fig, ax = plt.subplots()
    # fun_sns_lineplot_new(
    #     prepare_data_for_sns_lineplot(
    #         res["selected C1 scenarios + eu targets"][selcols]
    #         .xs("EU27", level="REGION")
    #         .xs("Emissions|Kyoto Gases (incl. indirect AFOLU)", level="VARIABLE")[
    #             selcols
    #         ],
    #         "standard",
    #     ),
    #     df_merged.reset_index().MODEL.unique(),
    #     ax=ax,
    # )

    # df = df.rename({f"{model}_downscaled": model for model in models})

    if plot_sel_model_dict is not None:
        res[
            "selected C1 scenarios + eu targets"
        ] = fun_select_model_scenarios_combinations(
            res["selected C1 scenarios"],  # this is a bigger set than C1+ eu targets
            plot_sel_model_dict,
        )

    if not show_plots:
        print("step5e done!")
        return df_csv

    for var in sel_vars:
        top = 5000 if var == "Emissions|Kyoto Gases (incl. indirect AFOLU)" else None

        if feasibility_file_name is not None:
            elina_feasibility_results = pd.read_csv(
                CONSTANTS.INPUT_DATA_DIR
                / "EU_climate_advisory_board_2023"
                / feasibility_file_name
            )
            grey = elina_feasibility_results[elina_feasibility_results.flagged == 1]
            grey.columns = [x.upper() for x in grey.columns]
            df_graph = res["selected C1 scenarios + eu targets"][selcols + [2055]]
            df_graph.loc[:, 2055] = np.nan
            fun_plot_eu_ab_side_by_side(
                df_graph.replace(0, np.nan),
                var,
                "EU27",
                2050,
                ylim_top=top,
                color="blue",
                sel_model=fun_available_scen(grey.set_index(["MODEL", "SCENARIO"])),
                sel_model_color="grey",
                marker_2030={2030: 2085},
                marker_2050={2050: 300},
            )
        elif "selected C1 scenarios + eu targets" in res:
            fun_plot_eu_ab_side_by_side(
                res["selected C1 scenarios + eu targets"][selcols].replace(0, np.nan),
                var,
                selected_regions[0],
                2050,
                ylim_top=top,
                color="blue",
                sel_model="REMIND 3.1",
            )

    for var in sel_vars:
        if "selected C1 scenarios + eu targets" in res:
            fun_sns_lineplot(
                res["selected C1 scenarios + eu targets"]
                .replace(0, np.nan)
                .interpolate(axis=1, limit_direction="backward", limit_area="inside"),
                list(models) + ["Primap"],
            )

            fun_sns_lineplot(
                res["selected C1 scenarios + eu targets"]
                .xs(var, level="VARIABLE")
                .xs(selected_regions[0], level="REGION"),
                models,
                criteria="standard",
            )

            fun_sns_lineplot(
                res["selected C1 scenarios + eu targets"]
                .xs(var, level="VARIABLE")
                .xs(selected_regions[0], level="REGION"),
                models,
            )

            fun_plot_fill_between(
                "viridis",
                res["selected C1 scenarios + eu targets"]
                # .xs("EN_NPi2020_450", level="SCENARIO")
                .xs("h_ndc", level="SCENARIO").drop([2017, 2018], axis=1),
                var,
                title=f"{var}",
                countrylist=fun_eu27(),
                ls="-",
                xlist=range(1990, 2051),
            )

            fun_plot_downscaled_results(
                "viridis",
                res["selected C1 scenarios + eu targets"],
                var,
                title=f"{var}",
                countrylist=["AUT", "ITA"],  # fun_eu28(),
                ls="-",
                xlist=range(1990, 2051),
                add_model_styles=True,
            )

        for pal in sel_palette:
            fun_stackplot_downscaled_results(
                model,
                pal,
                df_merged.dropna(how="all")
                .xs("lower", level="CRITERIA")
                .xs(files[0], level="FILE"),
                var,
                countrylist=fun_eu27(),
                scenario="GP_CurPol_T45",
                xlist=range(1990, 2051),
                title=f"{model} - {var} - GP_CurPol_T45",
                subtitle=subtitle,
                add_legend=True,
                interpolate=True,
            )
            plt.show

def fun_keep_selected_step5b_emi_variables(keep_step5_emissions, df, model, project, exclude_emissions_by_sector_from_step5b, list_emission_by_sectors_step5b):
    df=df.copy(deep=True)
    if keep_step5_emissions is None:
        keep_step5_emissions=[]
    if len(keep_step5_emissions)>0:
        for x in keep_step5_emissions:
            native_c=get_native_countries(model, project, True)
            mydf=fun_xs(df, {'REGION':native_c}, exclude_vars=True)
            # Check if variable is not present in the downscaled results.
            if len(fun_xs(mydf, {'VARIABLE':x}))==0:
                if x in main_emi_sectors:
                    # Calculate it as the sum of sub-sectors
                    add=fun_create_var_as_sum(mydf, x, main_emi_sectors[x], unit='Mt CO2/yr')
                    add=fun_xs(add, {'VARIABLE':x})
                    # Exclude sub-sectors variables (unless they are listed in `keep_step5_emissions`)
                    excluded_var=[i for i in main_emi_sectors[x] if i not in keep_step5_emissions]
                    seldf=fun_xs(df, {'VARIABLE':excluded_var},exclude_vars=True)
                    df=pd.concat([seldf,add])

                    
                else:
                    txt1=f'Cannot find {x} in the downscaled results `df`, nor in the dictionary'
                    txt2=' `main_emi_sectors` (cannot calculate it as the sum of sub-sectors)'
                    raise ValueError(f'{txt1}, {txt2}')
        df=fun_drop_duplicates(df)

    # List of step5b variables that we want to keep. If None, at least keep "Emissions|CO2|Industrial Processes"
    if keep_step5_emissions is None or len(keep_step5_emissions)==0:
        keep_step5_emissions=["Emissions|CO2|Industrial Processes"]
    else:
        keep_step5_emissions+=["Emissions|CO2|Industrial Processes"]
    keep_step5_emissions=list(set(keep_step5_emissions))
    
    df_merged = fun_drop_duplicates(df)
    if exclude_emissions_by_sector_from_step5b:
        exclude = [
            x
            for x in list_emission_by_sectors_step5b
            if x not in  keep_step5_emissions
        ]
        df_merged = fun_xs(df_merged, {"VARIABLE": exclude}, exclude_vars=True)
    return df_merged
    # consistency checks for all countries, on the basis of models
    # fun_sectoral_checks(models_dict,df)


def fun_calculate_carbon_revenues_as_sum_subsectors(
    df_merged: pd.DataFrame,
) -> pd.DataFrame:
    """Calculates carbon revenues as sum of subsectors (only for models where missings and
    only if all sub-sectors are available). Returns the updated dataframe.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Your datavrane

    Returns
    -------
    pd.DataFrame
        Updated Dataframe
    """
    rev_sub_sectors = [
        "Revenue|Government|Tax|Carbon|Demand|Buildings",
        "Revenue|Government|Tax|Carbon|Demand|Industry",
        "Revenue|Government|Tax|Carbon|Supply",
        "Revenue|Government|Tax|Carbon|Demand|Transport",
    ]
    # NOTE sourcery suggestion for the below does not work
    condition = np.prod(
        [
            True if v in df_merged.reset_index().VARIABLE.unique() else False
            for v in rev_sub_sectors
        ]
    )

    if condition:
        unit = (
            fun_xs(
                df_merged, {"VARIABLE": "Revenue|Government|Tax|Carbon|Demand|Industry"}
            )
            .reset_index()
            .UNIT.unique()[0]
        )
        print(
            "Creating `Revenue|Government|Tax|Carbon` as the sum of sectorial revenues..."
        )
        df_merged = fun_create_var_as_sum_only_for_models_where_missing(
            df_merged,
            "Revenue|Government|Tax|Carbon",
            rev_sub_sectors,
            unit=unit,
            verbose=False,
        )

    return df_merged


def fun_selected_scenarios(
    eu_2030_kyoto: float,
    eu_2050_kyoto: float,
    selected_scenarios: Optional[str],
    df_merged: pd.DataFrame,
    var: str,
) -> dict:

    """Returns a dictionary with:
     "selected C1 scenarios": scenarios that pass the vetting according to the file path `selected_scenarios`
     "selected C1 scenarios + eu targets": scenarios that pass the vetting and that are consistent with `eu_2030_kyoto` emissions threshold and `eu_2050_kyoto` emissions threshold

    Returns
    -------
    dict
        Dictionary with selected scenario (in a dataframe)
    """

    # 1) Select scenarios based on Ed's criteria (category c1a + data quality control ) if selected_scenarios is not None
    df_sel_all_variables = (
        df_merged
        if selected_scenarios is None
        else fun_select_passed_c1_scenarios(selected_scenarios, df_merged)
    )

    res = {"selected C1 scenarios": df_sel_all_variables.copy(deep=True)}
    if eu_2030_kyoto is not None and eu_2050_kyoto is not None:
        # 2) Select scenarios within EU 2030 and 2050 targets
        # Emissions|Kyoto Gases (AR4) (UNFCCC) = 4633.482972 in 1990 for the EU. Source: `input_reference_EEA.csv``
        # 4633.482972 * 0.45 = 2085.067337 which is the EU 2030 target
        res = fun_select_eu_scenarios(
            eu_2030_kyoto, eu_2050_kyoto, res, df_sel_all_variables, var=var
        )

    return res


def fun_create_dashboard_pdf(
    df: pd.DataFrame,
    graph_pos: dict,
    c_list: list,
    file_path: Union[Path, str],
    pdf_file_name: str,
    figsize: tuple = (27, 15),
    colors: Optional[dict] = None,
) -> pd.DataFrame:

    """Creates a PDF dashboard named `pdf_file_name`, based on data in your `df` and a dictionary `graph_pos` with
    the type of graphs to be included in the dashboard, for a list of countries `c_list`.
    It will also save each individual graphas in `file_path`.

    Returns
    -------
    pd.DataFrame
        Your initial datafram
    """

    h=get_git_revision_short_hash()
    pdf_file_name=pdf_file_name.replace(".pdf",f"hash_{h}.pdf")
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    pdf = PdfPages(file_path / pdf_file_name)
    for m, scen_list in fun_available_scen(df).items():
        for scen in scen_list:
            for c in c_list:
                fig = fun_single_figure_with_subplots(
                    df, graph_pos, figsize, m, scen, c, colors=colors
                )
                figname = f"{m}_{scen}".replace("|", "_")
                fig.savefig(file_path / f"{figname}_hash_{h}.png")
                pdf.savefig(fig, dpi=80)
    pdf.close()
    return df


def fun_create_dashboard_pdf_by_country(
    df: pd.DataFrame,
    graph_pos: dict,
    c_list: list,
    file_path: Union[Path, str],
    pdf_file_name: str,
    figsize: tuple = (27, 15),
    colors: Optional[dict] = None,
) -> pd.DataFrame:

    """Creates a PDF dashboard named `pdf_file_name`, based on data in your `df` and a dictionary `graph_pos` with
    the type of graphs to be included in the dashboard, for a list of countries `c_list`.
    It will also save each individual graphas in `file_path`.

    Returns
    -------
    pd.DataFrame
        Your initial datafram
    """

    h=get_git_revision_short_hash()
    pdf_file_name=pdf_file_name.replace(".pdf",f"hash_{h}.pdf")
    variables=[x for x in graph_pos if x not in ['models','scenarios']]
    if len(variables)!=1:
        txt="`graph_pos` should contain only one variables (apart from 'models' and 'scenarios')"
        raise ValueError(f"{txt}. It contains {len(variables)} variables: {variables} ")
    var=variables[0]
    for x in ['models', 'scenarios']:
        if graph_pos[x] == ["*"]:
            graph_pos[x]=list(df.reset_index()[x[:-1].upper()].unique())
    df2=fun_xs(df, {'MODEL':graph_pos['models'], 'SCENARIO':graph_pos['scenarios']})
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    pdf = PdfPages(file_path / pdf_file_name)
    rows_dict={v:k for k,v in dict(enumerate(graph_pos['models'])).items()}
    cols_dict={v:k for k,v in dict(enumerate(graph_pos['scenarios'])).items()}
    for c in c_list:
        max_val=1.1*fun_xs(df2, {'VARIABLE':var, 'REGION':c}).max().max()
        # max_val_10_dt=max(fun_find_nearest_values(range(0, 
        #                                                 max(int(max_val*1.1), int(max_val+1))
        #                                                 ,10), 
        #                                         int(max_val), n_max=2))
        # max_val=max(max_val_10_dt, int(max_val))
        graph_pos[var]["ylim"]= (0, max_val)
        fig = fun_single_figure_with_subplots_by_country(
            df2, SliceableDict(graph_pos).slice(var), figsize, rows_dict, cols_dict, c, colors=colors, add_hist_data=True
            # nrows=1+max(rows_dict.values()),
            # ncols=1+max(cols_dict.values()),
        )
        figname = f"{c}_{var}".replace("|", "_")
        fig.savefig(file_path / f"{figname}_hash_{h}.png")
        pdf.savefig(fig, dpi=80)
    pdf.close()
    return df

def fun_single_figure_with_subplots(
    df, graph_pos, figsize, m, scen, c, colors: Optional[dict] = None, nrows=None, ncols=None
):
    # sourcery skip: low-code-quality
    if nrows is None:
        # NOTE  `graph_pos` heres should not contain `models`, `scenarios`
        nrows = max(x["row"] for x in graph_pos.values()) + 1 
    if ncols is None:
        ncols = max(x["col"] for x in graph_pos.values()) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=80)
    plt.suptitle(f"{m} - {scen} - {c}", fontsize=20)
    # Below we create each single page of the pdf
    for var, d in graph_pos.items():
        # Basic stacked area chart.
        if d["kind"] == "scatter":
            var_dict = {"VARIABLE": [var] + [d["xvar"]] + [d["svar"]]}
            data = fun_xs(df, var_dict).xs(c, level="REGION").loc[m].loc[scen]
            fun_scatter_plot(
                data,
                var,  # yvar
                d["xvar"],  # x var
                d["svar"],  # s var (scatter point size)
                ax=axes,
                row=d["row"],  # Row position
                col=d["col"],  # Col position
                ylim=d["ylim"],
            )

        else:
            sel_sub_sectors = fun_get_sub_sectors(
                df, var, d["by"], same_level=d["same_level"]
            )
            if "xvar" in d and len(d["xvar"]):
                sel_sub_sectors = sel_sub_sectors + [d["xvar"]]
            if len(sel_sub_sectors):
                data = fun_xs(df, {"VARIABLE": sel_sub_sectors})
                if d["by"] == "ec":
                    data = data.sort_index(level="VARIABLE", ascending=False)
                else:
                    data = data.sort_index(level="VARIABLE")
                if c in data.reset_index().REGION.unique():
                    data = data.xs(c, level="REGION")

                    if m in data.reset_index().MODEL.unique():
                        data = data.loc[m]
                        if scen in data.reset_index().SCENARIO.unique():
                            data = data.xs(scen, level="SCENARIO")
                            unit = data.reset_index().UNIT.unique()[0]
                            fun_bar_plot_stacked(
                                data,
                                None,  # ["red", "black"], # Colors
                                var,
                                unit,
                                ax=axes,
                                kind=d["kind"],
                                row=d["row"],  # Row position
                                col=d["col"],  # Col position
                                ylim=d["ylim"],
                                colors=colors,
                            )
                            plt.close()
            else:
                print(
                    f"{sel_sub_sectors} is empty. Please check your figure definition"
                )

                # fig.tight_layout()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def fun_single_figure_with_subplots_by_country(
    df_orig, graph_pos, figsize, rows_dict, cols_dict, c, colors: Optional[dict] = None, nrows=None, ncols=None, add_hist_data=False
):
    variables=[x for x in graph_pos if x not in ['models','scenarios']]
    if len(variables)!=1:
        txt="`graph_pos` should contain only one variables (apart from 'models' and 'scenarios')"
        raise ValueError(f"{txt}. It contains {len(variables)} variables: {variables} ")
    var=variables[0]
    
    available_scen=fun_available_scen(df_orig)
    # sourcery skip: low-code-quality
    if nrows is None:
        # NOTE  `graph_pos` heres should not contain `models`, `scenarios`
        nrows = len(set(fun_flatten_list(list((available_scen.values())))))
    if ncols is None:
        ncols = len(set(list((available_scen.keys()))))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=80)
    plt.suptitle(f"{var} - {c}", fontsize=20)
    # Below we create each single page of the pdf
    for var, d in graph_pos.items():
        for m, scen_list in available_scen.items():
            for scen in scen_list:    
                df=fun_xs(df_orig, {'MODEL':m, 'SCENARIO':scen})
                graph_pos[var]['row']=int(rows_dict[m])
                graph_pos[var]['col']=int(cols_dict[scen])
                # Basic stacked area chart.
                if d["kind"] == "scatter":
                    var_dict = {"VARIABLE": [var] + [d["xvar"]] + [d["svar"]]}
                    data = fun_xs(df, var_dict).xs(c, level="REGION").loc[m].loc[scen]
                    fun_scatter_plot(
                        data,
                        var,  # yvar
                        d["xvar"],  # x var
                        d["svar"],  # s var (scatter point size)
                        ax=axes,
                        row=d["row"],  # Row position
                        col=d["col"],  # Col position
                        ylim=d["ylim"],
                        title=f"{m} - {scen}",
                    )

                else:
                    sel_sub_sectors = fun_get_sub_sectors(
                        df, var, d["by"], same_level=d["same_level"]
                    )
                    if "xvar" in d and len(d["xvar"]):
                        sel_sub_sectors = sel_sub_sectors + [d["xvar"]]
                    if len(sel_sub_sectors):
                        data = fun_xs(df, {"VARIABLE": sel_sub_sectors})
                        if d["by"] == "ec":
                            data = data.sort_index(level="VARIABLE", ascending=False)
                        else:
                            data = data.sort_index(level="VARIABLE")
                        if c in data.reset_index().REGION.unique():
                            data = data.xs(c, level="REGION")

                            if m in data.reset_index().MODEL.unique():
                                data = data.loc[m]
                                if scen in data.reset_index().SCENARIO.unique():
                                    data = data.xs(scen, level="SCENARIO")
                                    unit = data.reset_index().UNIT.unique()[0]
                                    if add_hist_data:
                                        try:
                                            var_list=list(data.index.get_level_values('VARIABLE').unique())
                                            df_iea=fun_most_recent_iea_data()
                                            to_drop=[x for x in df_iea.index.names if x not in data.index.names]
                                            hist=fun_xs(df_iea, {'VARIABLE':var_list, 'REGION':c}).droplevel(to_drop).dropna(how='all', axis=1)
                                            # hist=hist.assign(FILE='not available').set_index('FILE', append=True)
                                            cols_iea=list(range(2000,2015,1))
                                            cols_data=[x for x in data.columns if x not in cols_iea]
                                            if 'FILE' in data.index.names:
                                                data=data.droplevel('FILE')
                                            if len(hist):
                                                data2=pd.concat([hist[cols_iea],data[cols_data]], axis=1)
                                            else:
                                                data2=data
                                        except:
                                            data2=data
                                    else:
                                        data2=data
                                    fun_bar_plot_stacked(
                                        data2,
                                        None,  # ["red", "black"], # Colors
                                        f"{m} - {scen}",
                                        unit,
                                        ax=axes,
                                        kind=d["kind"],
                                        row=d["row"],  # Row position
                                        col=d["col"],  # Col position
                                        ylim=d["ylim"],
                                        colors=colors,
                                    )
                                    plt.close()
                    else:
                        print(
                            f"{sel_sub_sectors} is empty. Please check your figure definition"
                        )

                # fig.tight_layout()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


if __name__ == "__main__":
    main(
        # project_folder="EU_climate_advisory_board_2023",
        files=["Step5d.csv", "snapshot_all_regions.csv"],
        ### BELOW EU AB
        models_dict={
            # "EU_climate_advisory_board": [
            #     "WITCH 5.0",  #  will be included in the 24 scenarios
            #     "IMAGE 3.0",
            #     "MESSAGEix-GLOBIOM 1.1",  #  will be included in the 24 scenarios
            #     "REMIND-MAgPIE 2.1-4.2",  # will be included in the 24 scenarios (engage scenarios)
            #     "POLES-JRC ENGAGE",  # Not passing vetting
            # ],
            # "EU_climate_advisory_board_2023": [
            #     "REMIND-MAgPIE 2.1-4.2",
            #     "REMIND-MAgPIE 2.1-4.3",
            #     "GCAM-PR 5.3",
            #     "IMAGE 3.2",
            #     "AIM_CGE 2.2",  # -> file_suffix: "2023_02_22" for step5d)
            #     ### Native EU27 below
            #     "POLES-JRC",  # eu27 as native region - geco scenarios
            #     "REMIND 3.0",  # eu27 as native region
            #     ### NGFS 2022 below
            #     "GCAM 5.3+ NGFS",  # NGFS scenarios -> file_suffix="NGFS_2022_Round_2nd_version2_June" (for step5d)
            #     "REMIND-MAgPIE 3.0-4.4",  # NGFS scenarios -> file_suffix="NGFS_2022_Round_3rd_version_June" (for step5d)
            #     #### "MESSAGEix-GLOBIOM 1.0", #Eastern EU did not go through
            # ],
            # # "EU_climate_advisory_board_2023_remind": ["REMIND 3.1"], # Will be replaced by "REMIND 3.2"
            "EU_climate_advisory_board_2023_ECEMF": [
                # "WITCH 5.1",  # eu 27 native
                # "REMIND 2.1",  # eu 27 native
                # "PRIMES 2022",  # eu 27 native
                # "Euro-Calliope 2.0",  # eu 27 native
                ### "MESSAGEix-GLOBIOM 1.2",  # downscaled # energy related emissions missing
                ### "IMAGE 3.2",  #  downscaled # energy related emissions missing
                "REMIND 3.2",  # EU 27 native
            ],
        },
        # baseyear="2010",
        # worst_countries=False,
        sel_vars=sel_plot_vars_step5e,
        harmo_vars=step5e_harmo,
        exclude_emissions_by_sector_from_step5b=True,
        add_hist_data=False,
        create_dashboard=False,  # temporary
        # target_var="GHG incl. International transport",
        target_var="GHG incl. International transport (intra-eu only)",
        eu_2030_kyoto=2085,  # 2100,  #  # 4633.482972 * 0.45 = 2085.067337 which is the EU 2030 target, based on EEA data (UNFCCC)
        eu_2050_kyoto=300,
        csv_file_name=None,  # "EUab_2023_06_08_v8",
        harmonize_eea_data_until=2019,
        interpolate=True,
        use_iea_data_from_ed=True,  # IEA energy data
        use_eea_data=True,  # EMISSIONS DATA
        harmonize_hist_data=True,
        search_in_input_data_native_iam_results=False,
        # selected_scenarios= "EU_climate_advisory_board_2023/Ed_pass_scenarios_category_c1b.csv",
        # selected_scenarios="EU_climate_advisory_board_2023/vetting_flags_global_regional_combined_20230508.csv",  # https://iiasahub.sharepoint.com/:x:/r/sites/eceprog/Shared%20Documents/Projects/EUAB/vetting/vetting_flags_global_regional_combined_20230508.xlsx?d=w66181f6d1fce43a0a84ffa15c67f18a2&csf=1&web=1&e=vp20uV
        selected_scenarios="EU_climate_advisory_board_2023/vetting_flags_global_regional_combined_20230512.csv",  # https://iiasahub.sharepoint.com/:x:/r/sites/eceprog/Shared%20Documents/Projects/EUAB/vetting/vetting_flags_global_regional_combined_20230512.xlsx?d=w70d75f1721804459a3713772be5393b0&csf=1&web=1&e=Oc9rTR
        # selected_scenarios="EU_climate_advisory_board_2023/selected_eu_ab_scenario.csv",  # 24 PA scenarios (excluding GCAM PR_EU_55ZE_EI with negative non-co2 emissions)
        selected_regions=["EU27"],
        # feasibility_file_name="feasibility_2050_type1.csv",
        # feasibility_file_name="feasibility_2050_type3.csv",
        feasibility_file_name="filtering_v11_30_05_2023.csv",
        # plot_sel_model_dict={
        #     #'GCAM-PR 5.3': ['PR_EU_55NZE_ETS'],
        #     "REMIND 3.1": [
        #         "NZero",
        #         "def_600",
        #         "flex_600",
        #         "flex_800",
        #         "flex_500",
        #         "flex_300",
        #         "def_500",
        #         "def_300",
        #         "rigid_300",
        #         "NZero_bioLim7p5",
        #         "NZero_bioLim15",
        #         "rigid_600",
        #     ],
        #     "IMAGE 3.2": [
        #         "SSP1_SPA1_19I_RE_LB",
        #         "SSP2_SPA1_19I_LIRE_LB",
        #         "SSP1_SPA1_19I_LIRE_LB",
        #         "SSP1_SPA1_19I_D_LB",
        #     ],
        # "MESSAGEix-GLOBIOM 1.1": [
        #     "EN_NPi2020_600_DR3p",
        #     "EN_NPi2020_600_DR1p",
        #     "EN_NPi2020_450",
        #     "EN_NPi2020_600_DR4p",
        #     "EN_NPi2020_500",
        #     "EN_NPi2020_600_DR2p",
        # ],
        #     # "POLES-JRC": ["GECO-2023 1.5C"],
        #     "REMIND-MAgPIE 2.1-4.2": [
        #         "SusDev_SDP-PkBudg1000",
        #         "EN_NPi2020_600",
        #         "EN_NPi2020_500",
        #         "EN_NPi2020_200f",
        #         "EN_NPi2020_400",
        #         "EN_NPi2020_400f",
        #         "SusDev_SSP1-PkBudg900",
        #         "EN_NPi2020_300f",
        #     ],
        #     "REMIND-MAgPIE 2.1-4.3": [
        #         "DeepElec_SSP2_HighRE_Budg900",
        #         "DeepElec_SSP2_def_Budg900",
        #     ],
        #     "REMIND-MAgPIE 3.0-4.4": ["d_rap", "o_1p5c"],
        #     "WITCH 5.0": ["EN_NPi2020_450", "EN_NPi2020_500"],
        # },
    )

    # =================================================================
    # BELOW NGFS 2022
    # =================================================================
    # main(
    #     project_folder="NGFS_2022",
    #     files=[
    #         "NGFS_2022_Round_2nd_2022_12_07_sensitivity_SLOW_step4_FINAL.csv",
    #         "NGFS_2022_Round_2nd_2022_12_07_sensitivity_FAST_step4_FINAL.csv",
    #     ],
    #     models=[
    #         "MESSAGEix-GLOBIOM 1.1-M-R12",
    #     ],
    #     baseyear="2010",
    #     worst_countries=False,
    #     add_hist_data=True,
    # )
