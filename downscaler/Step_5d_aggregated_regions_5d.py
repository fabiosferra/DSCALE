import pandas as pd
import os
from pathlib import Path
from downscaler import CONSTANTS
from downscaler.utils_pandas import fun_replace_model_downscaled_with_model
from downscaler.utils import (
    fun_eu27,
    fun_aggregate_countries,
    fun_xs,
    fun_industrial_process_emi_for_aggregated_region,
    fun_get_iam_regions_associated_with_countrylist,
    fun_read_df_iam_from_multiple_df,
    unique,
    fun_index_names,
    fun_drop_duplicates,
)


def main(
    project: str,
    models: list,
    aggregated_region_name: str,
    countrylist: list,
    csv_suffix: str,
    remove_single_countries=True,
) -> pd.DataFrame:
    """Aggregates country-level (e.g. ['ITA', 'FRA', 'DEU' ... ]) data to an aggregated region (e.g. EU27).

    Parameters
    ----------
    file : dict
        List of models
    aggregated_region_name : str
        Name of your aggregated region (e.g. EU27)
    countrylist : list
        List of countries to be aggregated (e.g. ['ITA', 'FRA', 'DEU' ..])
    csv_suffix: str
        CSV file name suffix of downscaled results
    Returns
    -------
    pd.DataFrame
        Dataframe with aggregated data
    """
    CURR_RES_DIR = CONSTANTS.CURR_RES_DIR("step5")
    if callable(CONSTANTS.RES_DIR):
        RESULTS_DATA_DIR = CONSTANTS.RES_DIR(str(Path(os.path.abspath("")) / "step5"))
    else:
        RESULTS_DATA_DIR = CURR_RES_DIR

    if callable(CONSTANTS.INPUT_DATA_DIR):
        datadir = Path(CONSTANTS.INPUT_DATA_DIR()) / project / "multiple_df"
    else:
        datadir = Path(CONSTANTS.INPUT_DATA_DIR) / project / "multiple_df"
    print("running step5d...")
    df_all_models = pd.DataFrame()

    # Creating list of files
    files = {
        m: [
            f"{m}_{csv_suffix}.csv",
            f"{m}_{csv_suffix}_Emissions_by_sectors_and_revenues.csv",
            f"{project}/{m}_AFOLU_emissions.csv",
            # NOTE: Exclude non-co2 by gases, as total non-co2 is now included in '_imports.csv' file
            f"{m}_{project}_non_co2.csv",
            f"{m}_{project}_imports.csv",
        ]
        for m in models
    }

    missing_files = []
    idxname = ["MODEL", "REGION", "VARIABLE", "UNIT", "SCENARIO"]
    # Model loop
    for model in files:
        print(model)
        # File loop -> Read all csv files for each model (including co2, non_co2 and afolu emissions)
        df_all = pd.DataFrame()
        for file in files[model]:
            # Reads NGFS 2022 Results (only for `EU_climate_advisory_board_2023` project)
            if (
                csv_suffix == "2023_02_22"
                and project == "EU_climate_advisory_board_2023"
            ):
                if model == "GCAM 5.3+ NGFS":
                    file = file.replace(csv_suffix, "NGFS_2022_Round_2nd_version2_June")
                elif model == "REMIND-MAgPIE 3.0-4.4":
                    file = file.replace(csv_suffix,"NGFS_2022_Round_3rd_version_June")  
            if (Path(f"{CURR_RES_DIR}/{file}")).is_file(): # NOTE: we really need to use an f string here, otherwise we may get False instead of True
                # e.g. This one returns False: (CURR_RES_DIR/file).is_file(). Although the line below works fine
                # This one works: pd.read_csv('C:\\Users\\sferra\\AppData\\Local\\Temp\\pytest-of-sferra\\pytest-2767\\test_step5c_tris_5d_5e0//MESSAGEix-GLOBIOM 1.1-M-R12_AFOLU_emissions.csv')

                df = pd.read_csv(f"{CURR_RES_DIR}/{file}")
                df.columns = [x.upper() for x in df.columns]

                if "ISO" in df.columns and "REGION" in df.columns:
                    df = df.drop("REGION", axis=1).rename(columns={"ISO": "REGION"})

                # Replace f"{model}_downscaled" with f"{model}"
                df = fun_replace_model_downscaled_with_model(df)
                df = df[df.MODEL != "MODEL"]
                if len(df.MODEL.unique()) != 1:
                    raise ValueError(
                        f"We expect only one model in this file: {file}. We found: {df.MODEL.unique()} "
                    )
                df.loc[:, "MODEL"] = model
                df_all = pd.concat([df_all, df], axis=0, sort=True)
            else:
                missing_files = missing_files + [f"{CURR_RES_DIR}/{file}"]
                print(f"{file}", "does not exists")

        if not len(df_all):
            raise ValueError(
                f"df_all is an empty dataframe. This is the list of missing files {missing_files}"
            )

        # Exclude price variables
        var_selected = [x for x in df.VARIABLE.unique() if "Price" not in x]
        df_price = fun_xs(
            fun_index_names(df, True, int, idxname),
            {"VARIABLE": var_selected},
            exclude_vars=True,
        )
        df = df[df.VARIABLE.isin(var_selected)]

        # We calculate "Emissions|CO2|Industrial Processes" if missing
        if aggregated_region_name is not None and (
            "Emissions|CO2|Industrial Processes"
            not in df_all[df_all.REGION == aggregated_region_name]
            .reset_index()
            .VARIABLE.unique()
        ):
            # Read IAM results for a given model
            df_iam = fun_read_df_iam_from_multiple_df(model, datadir)

            # Check if "Final Energy|Industry" is present in the downscaled results for our `countrylist`
            df_all_check = fun_xs(fun_index_names(df_all), {"REGION": countrylist})
            df_all_check = df_all_check.reset_index().VARIABLE.unique()
            if (
                "Final Energy|Industry" in df_iam.reset_index().VARIABLE.unique()
                and "Final Energy|Industry" in df_all_check
            ):
                # Try to get 'marker' regions
                iam_regions = [
                    x
                    for x in df_iam.reset_index().REGION.unique()
                    if type(x) is str and "marker" in x
                ]
                # if `iam_regions` is empty: get all regions associated with countrylist
                if not len(iam_regions):
                    model = df_iam.reset_index().MODEL.unique()[0]
                    iam_regions = unique(
                        fun_get_iam_regions_associated_with_countrylist(
                            project, countrylist, model
                        ).values()
                    )
                    iam_regions = [f"{model}|{r[:-1]}" for r in iam_regions]
                df_all = fun_industrial_process_emi_for_aggregated_region(
                    aggregated_region_name, iam_regions, countrylist, df_all, df_iam
                )

        df_all = fun_index_names(df_all, True, int, idxname)

        ## drop duplicates to avoid possible double counting (summing up twice)
        df_all = fun_drop_duplicates(df_all)

        # Calculate results for aggregated region (e.g. eu27)
        if aggregated_region_name is not None:
            if aggregated_region_name in df_all.reset_index().REGION.unique():
                df_all=df_all.drop(aggregated_region_name, level='REGION')
            
            df_all = fun_aggregate_countries(
                df_all.reset_index().set_index(idxname),
                aggregated_region_name,
                ["MODEL", "VARIABLE", "UNIT", "SCENARIO"],
                countrylist,
                remove_single_countries=remove_single_countries,
            )

        # Bring back price information
        df_all = pd.concat([df_all, df_price], axis=0)

        # Save to csv
        df_all = df_all.reset_index().set_index(
            ["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"]
        )
        df_all.to_csv(
            f"{RESULTS_DATA_DIR}/{model}_{project}_Step5d.csv", encoding="utf-8-sig"
        )
        df_all_models = pd.concat([df_all_models, df_all], axis=0, sort=True)
    print("step 5d done")
    return df_all_models


if __name__ == "__main__":
    main(
        project="EU_climate_advisory_board_2023",
        models=[
            #     "GCAM-PR 5.3",
            "AIM_CGE 2.2",
            #     "REMIND-MAgPIE 2.1-4.2",
            #     "REMIND-MAgPIE 2.1-4.3",
            #     "MESSAGEix-GLOBIOM 1.0",
            #     "IMAGE 3.2",
        ],
        csv_suffix="2023_02_22_iea_harmo",
        aggregated_region_name="EU27",
        countrylist=fun_eu27(),
    )
