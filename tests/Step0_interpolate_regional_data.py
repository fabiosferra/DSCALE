import os
from typing import Union

import pandas as pd

# from downscaler.utils_pandas import fun_add_non_biomass_ren_nomenclature
from downscaler.utils import (
    CONSTANTS,
    check_regions_step0,
    check_step0_input,
    fun_add_non_biomass_ren_nomenclature,
    fun_add_regions_scenarios_names,
    fun_check_variables_by_step,
    fun_create_default_mapping,
    fun_eu28,
    fun_eu27,
    fun_get_variables_for_selected_step,
    fun_interpolate_and_abs_ccs_sequestration,
    fun_save_csv_step0,
    fun_split_snapshot_by_model,
    fun_xs,
    fun_create_missing_sectors_as_difference,
    fun_check_negative_energy_variables_by_model,
)
from downscaler.utils_pandas import (
    fun_xs_enhanced,
    fun_index_names,
    fun_create_var_as_sum
)
from downscaler.fixtures import ngfs_2023_nomenclature


def main(
    project_folder: str,
    model_folders: Union[None, list],
    snapshot_with_all_models: Union[None, str],
    country_marker_list: list,
    scenario_marker_dict: Union[None, dict] = None,
    previous_projects_folders: Union[None, list] = None,
    model_reg_folder: str = "ENGAGE_model_mapping",
    rename_df_mapping_dict: Union[None, dict] = None,
    region_name_short: bool = False,
    step1: bool = True,
    step2: bool = True,
    step3: bool = True,
    step5: bool = True,
    save_to_csv: bool = True,
    mandatory_vars=None,
    interpolate=True,
    coerce_errors=False,
    skip_scenarios: list = [],
    **kwargs,
) -> Union[pd.DataFrame, dict]:
    """Prepares IAMs data and default mapping for the downscaling.
    It also checks for inconsistencies in the regional mapping, missing list of variables.
    It returns the updated IAMs results, default mapping and dictionaries with list of missing variables and missing regions by model.

    Parameters
    ----------
    project_folder : str
        Project folder e.g. "NGFS_2022"
    model_folders : Union[None, list,str]
        A list of model folders with individual IAMs results. If None, will use  a `snapshot_with_all_models` csv file.
        If a string, will assume that this folder contains multiple files from different models (to be splitted by model).
    snapshot_with_all_models : Union[None, str]
        Dataframe with all IAMs results (to be split for each single model)
    country_marker_list : list
        List of countries we are interested in e.g. ['AUT']. Regions that contain that list of countries will be marked as 'marker' region
    scenario_marker_dict : Union[None, dict], optional
        A dictionary with the scenarios we are interested in for each modele.g {'MESSAGE':'SSP1'} to be marked as 'marker' scenario, by default None
    previous_projects_folders : Union[None, list], optional
        A list of project folders that we used before, to search for a regional-mapping file (NOTE the order matters, please start from the ones that are  more relevant). , by default None
    model_reg_folder : str, optional
        A folder with the IAMs model registration files, by default "ENGAGE_model_mapping"
    rename_df_mapping_dict : Union[None, dict], optional
        Dictionary that can be used to rename regions e.g. {"REMIND 3.0": {"FSU": "Reforming Economies"}},  by default None
    region_name_short: bool
        Wheter IAMs snapshot contains short vs long regions name e.g ('LAM' vs 'Latin America'), by default False
    step1 : bool, optional
        Wheter we run step1, by default True
    step2 : bool, optional
        Wheter we run step2, by default True
    step3 : bool, optional
        Wheter we run step3, by default True
    step5 : bool, optional
        Wheter we run step5, by default True
    save_to_csv : bool, optional
        Wheter we want to save files to csv (it should be false only for running the test case), by default True

    Returns
    -------
    Union[pd.DataFrame, dict]
        Returns updated dataframes and  dictionaries with a list of missing variables and a list of missing regions by model
    """
    check_var_all_models = {}
    if mandatory_vars is None:
        mandatory_vars = []
    if rename_df_mapping_dict is None:
        rename_df_mapping_dict = {}

    check_step0_input(snapshot_with_all_models)

    # NOTE: remove file called `snapshot_all_regions.csv`.
    # This file can be present in the directory if a simulation was not run succesfully (`snapshot_all_regions.csv` is normally deleted at the end of the simulation, see `run_multiple_files.py`)
    if os.path.exists(f"input_data/{project_folder}/snapshot_all_regions.csv"):
        os.unlink(f"input_data/{project_folder}/snapshot_all_regions.csv")

    var_check_dict = fun_get_variables_for_selected_step(step1, step2, step3, step5)
    reg_check_dict = {}

    # NOTE: If model folders is None, we split the dataframe with all models individual df for each single model (to be saved in different model foder)
    # model_folders = fun_split_snapshot_by_model(
    #     project_folder, model_folders, snapshot_with_all_models
    # )
    if snapshot_with_all_models is None:
        # We search for several files available in one folder.
        # As one file might contain multiple models, we split it by model and -> we create one folder for each model (with IAM snapshot)
        if isinstance(model_folders, str):
            model_folders = fun_split_snapshot_by_model(
                project_folder, None, model_folders
            )
    else:
        model_folders = fun_split_snapshot_by_model(
            project_folder, model_folders, snapshot_with_all_models
        )

    df_mapping = fun_create_default_mapping(
        CONSTANTS.INPUT_DATA_DIR / project_folder,
        model_reg_folder=CONSTANTS.INPUT_DATA_DIR / project_folder / model_reg_folder
        if model_reg_folder
        else None,
        previous_projects_folders=previous_projects_folders,  # NOTE: The order of the folders matters! Start with the folders that contain the most updated data (default_mapping).
        df=None,  # NOTE provide a df if you want to rename marker regions
        col_sheet="Native Region Name",  # e.g. col_sheet_dict.get(model_name, "Native Region Name")
        model=None,
        replace_slash_with_undescore=True,
        clean_file=False,  # if True provide columns just with model, instead of model.REGION
        automatically_detect_region_names=True,
        region_name_short=region_name_short,  # e.g. If True, 'LAM' for 'Latin America'
        country_marker_list=country_marker_list,
        save_to_csv=save_to_csv,
    )

    for model in rename_df_mapping_dict:
        df_mapping = df_mapping.replace(rename_df_mapping_dict[model])

    skip_models_dict = {}
    df_all = pd.DataFrame()
    for model in model_folders:
        df_mapping, df = fun_add_regions_scenarios_names(
            country_marker_list,
            scenario_marker_dict,
            rename_df_mapping_dict,
            project_folder,
            df_mapping,
            model,
            coerce_errors=coerce_errors,
        )

        # Exclude scenarios already available
        if model in skip_scenarios:
            df = fun_xs(
                df.xs(model, drop_level=False),
                {"SCENARIO": skip_scenarios[model]},
                exclude_vars=True,
            )

        # Skip model df is empty or if mandatory variables are missing
        if len(df):
            skip_model = False
        else:
            skip_model = True
        for x in mandatory_vars:
            if x not in df.reset_index().VARIABLE.unique():
                skip_model = True
                print(f"Mandatory variables not present in {model} - skip")

        if skip_model == True:
            skip_models_dict[model] = model

        else:
            # Check regions present in default_mapping.csv but missing in snapshot
            reg_check_dict[model] = check_regions_step0(df_mapping, df)

            df = fun_slice_for_regions_and_scenarios_markers(
                country_marker_list, scenario_marker_dict, df
            )

            # add non-biomass renewables nomenclature
            df = fun_add_non_biomass_ren_nomenclature(df)[0]

            # Rename carbon revenues
            df = df.rename(index=ngfs_2023_nomenclature)

            # Exclude variables that do not require interpolation
            df = fun_index_names(df, True, int)
            if 2065 in df.columns:
                var_wo_interpolation = df[~df[2065].isnull()]
                df = df[~df.index.isin(var_wo_interpolation.index)]
            else:
                var_wo_interpolation=pd.DataFrame()
            # Interpolate
            if interpolate and len(df):
                df = fun_interpolate_and_abs_ccs_sequestration(df, model)
            df = fun_index_names(df, True, int)

            # Bring back variables that do have not been interpolated (e.g. Diagnostics, MAGICC variables)
            if len(var_wo_interpolation):
                df = pd.concat([df, fun_index_names(var_wo_interpolation, True, int)])
            df = df.iloc[:, df.columns.isin(range(2005, 2105, 5))]
            df = fun_index_names(df, True, str)

            # Check variables
            check_var_all_models[model] = fun_check_variables_by_step(
                var_check_dict, model, df
            )

            # Creates missing sector as the difference between main sector and the list of other sectors,  if only one sector is missing.
            # E.g. we can create `Final Energy|Commercial` as the difference between `Final Energy|Residential and Commercial` and `Final Energy|Residential`, (or the other way around)
            df = fun_create_missing_sectors_as_difference(df, {'Final Energy|Residential and Commercial': 
                                                 ['Final Energy|Commercial', 'Final Energy|Residential']}
                                            )

            fun_check_negative_energy_variables_by_model(df, project=project_folder, coerce_errors=coerce_errors)
            df_all = fun_save_csv_step0(
                save_to_csv,
                model,
                df,
                df_all,
                f"input_data/{project_folder}/multiple_df",
            )
            sel_cols = [x for x in df_mapping.columns if "Country." not in x]
            df_mapping = df_mapping[sel_cols]
            if save_to_csv:
                df_mapping.to_csv(f"input_data/{project_folder}/default_mapping.csv")

    check_var_all_models_simple = {}
    for model, steps in check_var_all_models.items():
        steps_dict = {step: list(steps[step].keys()) for step in steps.keys()}
        check_var_all_models_simple[model] = steps_dict

    if save_to_csv:
        reg_missing = pd.DataFrame(reg_check_dict).T
        var_missing = pd.DataFrame(check_var_all_models_simple).T
        reg_and_var_missing = pd.concat([reg_missing, var_missing], axis=1)
        reg_and_var_missing.to_csv(
            CONSTANTS.INPUT_DATA_DIR
            / project_folder
            / "Missing_regions_and_variables.csv"
        )
        pd.DataFrame(check_var_all_models).T.to_csv(
            CONSTANTS.INPUT_DATA_DIR / project_folder / "Missing_variables_detailed.csv"
        )
    print(skip_models_dict)
    return df_mapping, df_all, check_var_all_models, reg_check_dict




def fun_slice_for_regions_and_scenarios_markers(
    country_marker_list: list, scenario_marker_dict: dict, df: pd.DataFrame
) -> pd.DataFrame:
    """Returns sliced dataframe for selected scenarios and regions markers

    Parameters
    ----------
    country_marker_list : list
        List of countries you are interested in (associated to region markers)
    scenario_marker_dict : dict
        Scenarios marker for each model
    df : pd.DataFrame
        Your dataframe

    Returns
    -------
    pd.DataFrame
        Sliced dataframe
    """
    if country_marker_list is not None and len(country_marker_list):
        sel_reg = [x for x in df.reset_index().REGION.unique() if "_marker" in x]
        if len(sel_reg):
            df = fun_xs(df, {"REGION": sel_reg})
    if scenario_marker_dict is not None and len(scenario_marker_dict):
        sel_scen = [x for x in df.reset_index().SCENARIO.unique() if "_marker" in x]
        if len(sel_scen):
            df = fun_xs(df, {"SCENARIO": sel_scen})
    return df


if __name__ == "__main__":
    main(
        # "TEST_2023",  # "EU_climate_advisory_board_2023",
        # "EU_climate_advisory_board_2023_remind",
        # "NGFS_2023",
        "ENGAGE_2023",
        # model_folders="snapshot_v1",  # if a string, will split all dataframes contained in that folder by model
        model_folders=[
            "MESSAGEix-GLOBIOM_1.1"
        ],  # if a list, will assume each folder contains only one file with one model
        snapshot_with_all_models=None,  # "snapshot_to_be_split",
        country_marker_list=None,  # fun_eu27(),
        previous_projects_folders=None,  # ['NGFS_2022'],#None,
        model_reg_folder="ENGAGE_model_mapping",
        # automatically_detect_region_names=True,
        # rename_df_mapping_dict={
        #     "REMIND 3.0": {
        #         "Countries from the Reforming Ecomonies of the Former Soviet Union": "Countries from the Reforming Economies of the Former Soviet Union"
        #     },
        #     "IMAGE 3.0": {
        #         "C. Europe_marker": "Central Europe_marker",
        #         "W. Europe_marker": "Western Europe_marker",
        #     },
        #     "IMAGE 3.0.1": {
        #         "C. Europe_marker": "Central Europe_marker",
        #         "W. Europe_marker": "Western Europe_marker",
        #     },
        # },
        save_to_csv=True,
        # mandatory_vars=["Emissions|Kyoto Gases"],
        mandatory_vars=[],  # Even if Kyoto is missing, can stil calculate carbon
        interpolate=True,
        region_name_short=True,
    )
