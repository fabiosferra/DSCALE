import os
from pathlib import Path

import pandas as pd

from downscaler import CONSTANTS, IFT
from downscaler.fixtures import iea_biomass, iea_coal, iea_flow_dict, iea_gases, iea_oil
from downscaler.utils import (
    InputFile,
    fun_hist_all_data,
    fun_hist_share,
    fun_read_df_countries,
    fun_read_df_iam_all,
    fun_read_df_iea_all,
    fun_read_reg_value,
    load_model_mapping,
)


def main(
    project_file: str,
    models: list,
    model_in_region_name: bool = False,
):

    RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )
    RESULTS_DATA_DIR.mkdir(exist_ok=True)

    # Reading regional IAMs data
    file: IFT = CONSTANTS.INPUT_DATA_DIR / project_file / "snapshot_all_regions.csv"
    df_iam_all_models = fun_read_df_iam_all(
        file=InputFile(CONSTANTS.INPUT_DATA_DIR / file), add_model_name_to_region=False
    )
    df_iam_all_models.loc[:, "REGION"] = df_iam_all_models.loc[:, "REGION"] + "r"

    # trade_var = df_iam_all[(df_iam_all.UNIT=='EJ/yr')&(df_iam_all.VARIABLE.str.contains('Trade'))].VARIABLE.unique()
    trade_var = [
        "Trade|Primary Energy|Biomass|Volume",
        "Trade|Primary Energy|Gas|Volume",
        "Trade|Primary Energy|Coal|Volume",
        "Trade|Primary Energy|Oil|Volume",
    ]

    # Reading IEA data
    df_iea_all = fun_read_df_iea_all(
        CONSTANTS.INPUT_DATA_DIR / "Extended_IEA_en_bal_2019_ISO.csv"
    )

    df_iea_all = df_iea_all[
        (df_iea_all.FLOW.isin(["Imports", "Exports"]))
        & (df_iea_all.PRODUCT.isin(iea_oil + iea_gases + iea_biomass + iea_coal))
    ]

    # Add ISO code for TWN (otherwise missing in the df_iea_all)
    df_iea_all.loc[df_iea_all.COUNTRY.str.contains("Chinese Taipei"), "ISO"] = "TWN"

    max_year = int(pd.to_numeric(df_iea_all.columns, errors="coerce").max())
    range_list = range(1960, max_year, 1)
    range_list = [str(i).zfill(2) for i in range_list]
    df_iea_melt = pd.DataFrame()  # 2020_10_02 h 17.12
    df_iea_melt = df_iea_all.melt(
        id_vars=["COUNTRY", "FLOW", "PRODUCT", "ISO"], value_vars=range_list
    )
    df_iea_melt.rename(columns={"variable": "TIME", "value": "VALUE"}, inplace=True)
    df_iea_melt["VALUE"] = pd.to_numeric(df_iea_melt["VALUE"], errors="coerce")
    df_iea_melt["TIME"] = pd.to_numeric(df_iea_melt["TIME"], errors="coerce")

    # Reading model mapping
    pyam_mapping_file: IFT = (
        CONSTANTS.INPUT_DATA_DIR / project_file / "default_mapping.csv"
    )
    df_countries = fun_read_df_countries(
        CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
    )

    # Dataframe that will conatain the combined results for all regions, targets, variables
    res_all = pd.DataFrame()
    for model in models:

        targets = list(
            df_iam_all_models[df_iam_all_models.MODEL == model].SCENARIO.unique()
        )

        df_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file
        )
        for region in regions:  # ()
            countrylist = (
                df_countries[df_countries.REGION == region].ISO.unique().tolist()
            )

            for target in targets:
                for var in trade_var:
                    if model_in_region_name:
                        # regional level data from IAMs
                        reg_var = fun_read_reg_value(
                            model, model + "|" + region, target, var, df_iam_all_models
                        )["VALUE"]
                    else:
                        reg_var = fun_read_reg_value(
                            model, region, target, var, df_iam_all_models
                        )[
                            "VALUE"
                        ]  # regional level data from IAMs

                    if len(reg_var) == 0:
                        # country-level historical data shares (at the base year 2010)
                        c_var = fun_hist_share(
                            df_iea_melt, var, countrylist, 2010, iea_flow_dict, 0
                        )
                    else:
                        # country-level historical data shares (at the base year 2010)
                        c_var = fun_hist_share(
                            df_iea_melt,
                            var,
                            countrylist,
                            2010,
                            iea_flow_dict,
                            reg_var.loc[2010],
                        )

                    if round(c_var.sum(), 5) != 1:
                        raise Exception(
                            f"sum of country level share is not equal to 1, it is equal to: {c_var.sum()} instead."
                        )

                    # proportional downscaling based on historical country-level results
                    res_var = pd.DataFrame(
                        [c_var * reg_var[t] for t in reg_var.index], index=reg_var.index
                    ).T

                    res_var["VARIABLE"] = var
                    res_var["TARGET"] = target
                    res_var["REGION"] = region
                    res_var["MODEL"] = model
                    # appending the dataframe
                    res_all = pd.concat([res_all,res_var])

        res_all.set_index(
            ["MODEL", "REGION", "VARIABLE", "TARGET"], append=True, inplace=True
        )

        # Write historica data in a csv file
        # pd.DataFrame(fun_fossil_hist(df_iea_melt, trade_var,countrylist, [2010,1990], iea_flow_dict)).reset_index('TIME').pivot(columns='TIME')
        countrylist = df_countries.ISO.tolist()  # all countries
        time_list = [x for x in list(range(1990, 2010)) if x != 2005]
        hist_df = (
            pd.DataFrame(
                -fun_hist_all_data(
                    df_iea_melt, trade_var, countrylist, time_list, iea_flow_dict
                )
            )
            .reset_index("TIME")
            .pivot(columns="TIME")["VALUE"]
        )

        if (
            res_all.xs(target, level="TARGET")[list(range(2010, 2105, 5))]
            .index.get_level_values(0)
            .name
            == None
        ):
            res_all.index.set_names("ISO", level=0, inplace=True)

        # Merge the two dataframes and save merged df
        df_merged_all_scen = pd.DataFrame()
        for target in targets:
            df_merged = hist_df.merge(
                res_all.xs(target, level="TARGET")[list(range(2010, 2105, 5))],
                left_on=["ISO", "VARIABLE"],
                right_on=["ISO", "VARIABLE"],
            )
            df_merged.loc[:, "TARGET"] = target
            df_merged_all_scen=pd.concat([df_merged_all_scen,df_merged])

        # Save to CSV
        df_merged_all_scen.to_csv(
            RESULTS_DATA_DIR / f"{model}_{project_file}_trade_downscaling_MERGED.csv",
            mode="w",
            header=True,
        )  ## historical data
        res_all.to_csv(
            RESULTS_DATA_DIR / f"{model}_{project_file}_trade_downscaling.csv",
            mode="w",
            header=True,
        )  ## Downscaled data
        hist_df.to_csv(
            RESULTS_DATA_DIR
            / f"{model}_{project_file}_hist_data_trade_downscaling.csv",
            mode="w",
            header=True,
        )  ## historical data

    # df_merged_all_scen.to_csv(
    #     RESULTS_DATA_DIR/f'{project_file}_trade_downscaling_MERGED.csv', mode="w", header=True)  # historical data
    # # Downscaled data
    # res_all.to_csv(
    #     RESULTS_DATA_DIR/f'{project_file}_trade_downscaling.csv', mode="w", header=True)
    # # historical data
    # hist_df.to_csv(
    #     RESULTS_DATA_DIR/f'{project_file}_hist_data_trade_downscaling.csv', mode="w", header=True)


if __name__ == "__main__":
    project_file = "NGFS"
    # model='GCAM5.3_NGFS'
    # models = ['REMIND-MAgPIE 2.1-4.2']
    models = ["MESSAGEix-GLOBIOM 1.0", "REMIND-MAgPIE 2.1-4.2", "GCAM5.3_NGFS"]
    model_in_region_name = True

    main(project_file, models, model_in_region_name)
