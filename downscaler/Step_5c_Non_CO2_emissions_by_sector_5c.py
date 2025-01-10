import pandas as pd  # Load the Pandas enlong_calc(df_main) ## calculating enlonglibraries with alias 'pd'
import os
from downscaler import CONSTANTS
from downscaler.input_caching import InputFile, get_selection_dict
from downscaler.utils_pandas import fun_create_var_as_sum

## IMPORTING FUNCTIONS
from downscaler.utils import (
    fun_get_gwp_and_ghg_unit,
    fun_get_target_regions_step5,
    fun_harmonize,
    fun_load_non_co2,
    fun_non_co2_downscaling,
    fun_read_df_iam_iamc,
    fun_read_gwp,
    fun_read_df_iam_from_multiple_df,
    split_gains_regions_to_countries
)

RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR("downscaler/Step_5_Scenario_explorer_5.py")


def main(
    project_file: str,
    model_patterns: list,
    region_patterns: list,
    target_patterns: list,
    model_in_region_name: bool = False,
):

    # Get non co2 data from Gains in (MtCo2e/yr)
    df_non_co2_gains = fun_load_non_co2(
        project_file,
        CONSTANTS.INPUT_DATA_DIR,
        gains_non_co2_file="GAINS_nonCO2_Global_21Feb2022_byGas.csv",
        gains_iso_dict="GAINS_ISO3.csv",
        aggregate_by_sectors=False,
        CO2eq=False,
    )

    # df_non_co2_gains =split_gains_regions_to_countries(df_non_co2_gains)
    input_dir = CONSTANTS.INPUT_DATA_DIR / project_file
    input_file = input_dir / "snapshot_all_regions.csv"
    pyam_mapping_file = input_dir / "default_mapping.csv"

    try:
        selection_dict = get_selection_dict(
            InputFile(input_dir / "snapshot_all_regions.csv"),
            model=model_patterns,
            region=region_patterns,
            target=target_patterns,
            variable=["*"],
        )
    except ValueError as e:

        raise ValueError(
            f"{e}\nConsider using the asterisk '*' for the pattern."
        ) from e

    models = selection_dict.keys()

    gwp_data = fun_read_gwp()

    for model in models:
        df_iam = fun_read_df_iam_iamc(input_file)
        sel_var = [
            x
            for x in df_iam.reset_index().VARIABLE.unique()
            if type(x) is str and "mission" in x
        ]
        df_iam = df_iam[df_iam.index.get_level_values("VARIABLE").isin(sel_var)]

        targets, df_countries, regions = fun_get_target_regions_step5(
            InputFile(pyam_mapping_file), selection_dict, model
        )
        df_countries.loc[:, "REGION"] = [str(x)[:-1] for x in df_countries.REGION]
        if model_in_region_name:
            df_countries.loc[:, "REGION"] = [
                f"{model}|{x}" for x in df_countries.REGION
            ]

        res_dict = {}
        for region in regions:
            df_res = pd.DataFrame()
            print(
                f"Calculating non-CO2 emissions for : {region} -  {1+regions.index(region)}/{len(regions)}"
            )
            countrylist = list(df_countries[df_countries.REGION == region].ISO.unique())
            if len(countrylist) > 1:
                for target in targets:
                    print(f"{target} -  {1+targets.index(target)}/{len(targets)}")
                    # NOTE: Gases loop (e.g. 'Emissions|F-Gases')
                    res_dict_all_gases = fun_non_co2_downscaling(
                        df_non_co2_gains, model, df_iam, region, countrylist, target
                    )

                    non_co2_gases = ["CH4", "N2O", "HFC", "SF6"]
                    if len(res_dict_all_gases):
                        df_res_non_co2 = pd.concat(
                            res_dict_all_gases[x] for x in non_co2_gases
                        )

                        gwp_dict, ghg_emi_unit = fun_get_gwp_and_ghg_unit(
                            gwp_data, df_iam, res_dict_all_gases, non_co2_gases
                        )

                        df_res_non_co2 = fun_create_var_as_sum(
                            df_res_non_co2,
                            "Emissions|Total Non-CO2",
                            gwp_dict,
                            unit=ghg_emi_unit,
                        )
                        df_res = pd.concat([df_res, df_res_non_co2], axis=0)
                # NOTE: dataframe should contain only one region when harmonizing data
                if len(df_res):
                    df_res = fun_harmonize(
                        df_res,
                        region,
                        gwp_dict,
                        ghg_emi_unit,
                        df_iam,
                        var="Emissions|Total Non-CO2",
                    )  # .droplevel('REGION')
                    res_dict[region] = df_res
                else:
                    print(
                        f" ** No results are available from GAINS for any of the ISO in this region: {region} ** "
                    )
            else:
                print(f"{region} contains just one country - we skip this region")
        if len(res_dict):
            df_all = pd.concat(
                list(res_dict.values()), axis=0
            )  # pd.concat([res_dict[x] for x in regions], axis=0)
            df_all.to_csv(RESULTS_DATA_DIR / f"{model}_{project_file}_non_co2.csv")
    print("Step 5c done")
    return df_res


if __name__ == "__main__":
    # NOTE: use `run_multiple_files.py`. Do not run here
    main(
        project_file="EU_climate_advisory_board_2023",
        model_patterns=[
            "REMIND-MAgPIE 2.1-4.2",
            "REMIND-MAgPIE 2.1-4.3",
            "MESSAGEix-GLOBIOM 1.0",
            "AIM_CGE 2.2",
            "IMAGE 3.2",
        ],
        # project_file="EU_climate_advisory_board",
        # model_patterns=["MESSAGEix-GLOBIOM 1.1"],
        region_patterns=["*Marker*"],
        target_patterns=["*"],
        model_in_region_name=True,
    )
