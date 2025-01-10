import pandas as pd
from pandas.testing import assert_frame_equal
import os
import numpy as np
from downscaler import CONSTANTS
from downscaler.input_caching import get_selection_dict, InputFile

from downscaler.utils_pandas import fun_replace_model_downscaled_with_model
from downscaler.utils_pandas import fun_create_var_as_sum
from downscaler.fixtures import iea_countries

## IMPORTING FUNCTIONS
from downscaler.utils import (
    fun_add_multiply_dfmultindex_by_dfsingleindex,
    fun_get_target_regions_step5,
    fun_get_sub_sectors,
    fun_read_df_iam_iamc,
    fun_xs,
    fun_read_csv,
    fun_harmonize_hist_data_by_preserving_sum_across_countries,
    fun_clip_df_by_dict,
    fun_create_sample_df,
    fun_wildcard,
    fun_get_scenarios

)

RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR("downscaler/Step_5_Scenario_explorer_5.py")

# Temporarily add a test case as tests in VSC are not working 
exp=pd.DataFrame({2010: {('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_delfrag', 'CAN', 'Final Energy', 'EJ/yr'): 0.6, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_delfrag', 'USA', 'Final Energy', 'EJ/yr'): -0.6, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_strain', 'CAN', 'Final Energy', 'EJ/yr'): 0.6, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_strain', 'USA', 'Final Energy', 'EJ/yr'): -0.6}, 2015: {('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_delfrag', 'CAN', 'Final Energy', 'EJ/yr'): -1.0, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_delfrag', 'USA', 'Final Energy', 'EJ/yr'): 1.0, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_strain', 'CAN', 'Final Energy', 'EJ/yr'): -1.0, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_strain', 'USA', 'Final Energy', 'EJ/yr'): 1.0}, 2020: {('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_delfrag', 'CAN', 'Final Energy', 'EJ/yr'): 1.0, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_delfrag', 'USA', 'Final Energy', 'EJ/yr'): -1.0, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_strain', 'CAN', 'Final Energy', 'EJ/yr'): 1.0, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_strain', 
'USA', 'Final Energy', 'EJ/yr'): -1.0}, 2025: {('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_delfrag', 'CAN', 'Final Energy', 'EJ/yr'): 0.95, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_delfrag', 'USA', 'Final Energy', 'EJ/yr'): -0.95, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_strain', 'CAN', 'Final Energy', 'EJ/yr'): 0.95, ('MESSAGEix-GLOBIOM 1.1-M-R12', 'd_strain', 'USA', 'Final Energy', 'EJ/yr'): -0.95}})

# NOTE: Block below is just a test case - temporarily added here because VSC tests,do not work anylonger with current pandas/python versions
# mydf=fun_clip_df_by_dict(fun_create_sample_df(), {'USA':1, 'CAN':-1}, 'REGION', True, True)[range(2010,2030,5)]
# myiea=-mydf
# myiea[2010]=fun_clip_df_by_dict(fun_create_sample_df(), {'USA':-0.6, 'CAN':0.6}, 'REGION', True, True)[range(2010,2030,5)]
# myiea[2015]=-1
# res=fun_harmonize_hist_data_by_preserving_sum_across_countries(mydf, myiea, 'Final Energy', tc=2240)
# assert_frame_equal(exp, np.round(res,2), check_like=True)

# myiea.loc[fun_xs(myiea, {'REGION':'USA'}).index, 2015]=np.nan
# mydf[2015]=5
# fun_harmonize_hist_data_by_preserving_sum_across_countries(mydf[[2015,2020,2025]], myiea.fillna(5)[[2015]], 'Final Energy', tc=2240)
# fun_harmonize_hist_data_by_preserving_sum_across_countries(mydf[[2015,2020,2025]], myiea.fillna(6)[[2015]], 'Final Energy', tc=2240)

# myiea[2015]=np.nan
# mydf[2015]=5




def main(
    project_file: str,
    model_patterns: list,
    region_patterns: list,
    target_patterns: list,
    file_suffix: str,
    model_in_region_name: bool = False,
):

    input_file = CONSTANTS.INPUT_DATA_DIR / project_file / "snapshot_all_regions.csv"
    pyam_mapping_file = CONSTANTS.INPUT_DATA_DIR / project_file / "default_mapping.csv"

    try:
        selection_dict = get_selection_dict(
            InputFile(
                CONSTANTS.INPUT_DATA_DIR / project_file / "snapshot_all_regions.csv"
            ),
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
    idxname = ["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"]

    # NOTE: File was created with this code: https://github.com/iiasa/downscaler_repo/issues/196
    ieafile=CONSTANTS.INPUT_DATA_DIR/"IEA_hist_trade_variables_v2022.csv"
    iea=None
    if os.path.exists(ieafile) :
        iea=fun_read_csv({'aa':ieafile}, True, int)['aa'] # add code here
    else:
        print("*** WARNING *** ")
        print(f"Cannot find {ieafile}. ")
        print("Therefore trade results will be less accurate, beacuse will not consider historical data!!!")
        print("To create the historical data file, please follow the instructions: https://github.com/iiasa/downscaler_repo/issues/196")
        print("")



    scenarios = fun_wildcard(target_patterns, fun_get_scenarios(project_file)) 


   
    # aa=fun_create_sample_df(countrylist=iea_countries)
    # fun_harmonize_hist_data_by_preserving_sum_across_countries(aa, myiea, 'Final Energy', tc=2240)
    for model in models:
        df_iam = fun_read_df_iam_iamc(input_file)
        filename = CONSTANTS.CURR_RES_DIR("step5") / f"{model}_{file_suffix}.csv"
        if not os.path.exists(filename):
            raise ValueError(f"Cannot find {filename}")
        # Load Energy and non-co2 data below
        mf = CONSTANTS.CURR_RES_DIR("step5") / f"{model}"
        files = [f"{mf}_{x}" for x in [f"{file_suffix}", f"{project_file}_non_co2"]]
        df = pd.DataFrame()
        for f in files:
            temp = pd.read_csv(f"{f}.csv")
            if "2005" in temp.columns:
                temp = temp.drop("2005", axis=1)
            if "ISO" in temp.columns and "REGION" in temp.columns:
                temp = temp.drop("REGION", axis=1).rename({"ISO": "REGION"}, axis=1)
            df = pd.concat([df, temp], axis=0)
        df = df.set_index(idxname)
        df.columns = [int(x) for x in df.columns]

        # Replace f"{model}_downscaled" with f"{model}"
        df = fun_replace_model_downscaled_with_model(df)
        # Remove primary energy calculations:
        if "Primary Energy" in df.index.get_level_values("VARIABLE"):
            blacklist = df.xs("Primary Energy", level="VARIABLE", drop_level=False).index
            df = df.loc[~df.index.isin(blacklist)]
        # Recalculate primary energy as the sum of sub-sectors
        primary_en_vars = fun_get_sub_sectors(df, "Primary Energy", "fuel")
        df = fun_create_var_as_sum(df, "Primary Energy", primary_en_vars, unit="EJ/yr")

        targets, df_countries, regions = fun_get_target_regions_step5(
            InputFile(pyam_mapping_file), selection_dict, model
        )
        df_countries.loc[:, "REGION"] = [str(x)[:-1] for x in df_countries.REGION]
        df_iam = df_iam.reset_index().rename(columns={"TARGET": "SCENARIO"})
        df_iam = df_iam.set_index(idxname)
        if model_in_region_name:
            df_countries.loc[:, "REGION"] = [
                f"{model}|{x}" for x in df_countries.REGION
            ]

        df_res = pd.DataFrame()
        trade_variables = [
            x
            for x in df_iam.reset_index().VARIABLE.unique()
            if type(x) is str and "Trade" in x and "Mass" not in x
        ]

        for region in regions:
            text = f"{region} -  {1+regions.index(region)}/{len(regions)}"

            countrylist = list(df_countries[df_countries.REGION == region].ISO.unique())
            if len(countrylist) > 1:
                what = "Secondary energy Hydrogen and Trade"
                print(f"Calculating {what} for : {text}")
                # Secondary energy calculations
                # NOTE: as per Keywan's instructions we apply the share of `Secondary Energy|Hydrogen` / `Primary Energy`
                # We don't use `Secondary Energy|Hydrogen` / `Final Energy|Hydrogen` as this just gives us a conversion measure.
                # As per Ed's request (7 July 2023) for the ENGAGE project we use the same method for "Secondary Energy|Heat", however using 'Final Energy|Heat' as denominator (some countries do not have heat at the base year)
                for k, v in {
                    "Secondary Energy|Hydrogen": "Primary Energy",
                    "Secondary Energy|Heat": "Final Energy|Heat",
                }.items():
                    if k in df_iam.reset_index().VARIABLE.unique():
                        sec_hydrogen = fun_apply_regional_structure(
                            df,
                            df_iam,
                            model,
                            region,
                            countrylist,
                            idxname,
                            k,
                            var_den=v,
                        )

                        df_res = pd.concat([df_res, sec_hydrogen], axis=0)
                        # we need to update `df` to calculate trade imports below:
                        df = pd.concat([df, sec_hydrogen], axis=0)

                ## Imports calculations -> saved as: f"{model}_{project_file}_imports.csv"
                for var in trade_variables:
                    # for var in ['Trade|Secondary Energy|Liquids|Biomass|Volume']:
                    var_den = var.replace("Trade|", "").replace("|Volume", "")
                    iam_vars = df_iam.reset_index().VARIABLE.unique()
                    df_vars = df.reset_index().VARIABLE.unique()
                    if var in iam_vars and var_den in iam_vars and var_den in df_vars:
                        # 1. Applies regional pattern to the country level.
                        trade = fun_apply_regional_structure(
                            df,
                            df_iam,
                            model,
                            region,
                            countrylist,
                            idxname,
                            var,
                            var_den=var_den,
                        )
                        av_scen=trade.reset_index().SCENARIO.unique()
                        trade=fun_xs(trade, {"SCENARIO":scenarios})
                        if not len(trade):
                            raise ValueError(f"Selected scenarios {scenarios} not present in the dataframe for region {region}. Available scenarios: {av_scen}")
                        # 2. Harmonize trade with historical 2020 data (preserves importing/exporting countries) and keeps sum across countries unchanged
                        if iea is not None:
                            trade_all_final = fun_harmonize_hist_data_by_preserving_sum_across_countries(trade, iea, var)
                        else:
                            trade_all_final=trade   
                        df_res = pd.concat([df_res, trade_all_final], axis=0)
                    
        df_res.to_csv(RESULTS_DATA_DIR / f"{model}_{project_file}_imports.csv")
    print("Step 5c bis (Trade variables) done")
    return df_res


def fun_apply_regional_structure(
    df,
    df_iam,
    model,
    region,
    countrylist,
    idxname,
    var_num,
    var_den="Primary Energy",
):
    df = fun_xs(df, {"REGION": countrylist})
    # Calculates secondary energy `var` (e.g. secondary energy|hydrogen) as a fraction of  `var_den` (e.g. primary energy), by keeping the same structure of regional IAMs results"
    iam_sec_hydrogen_share = df_iam.xs(var_num, level="VARIABLE") / df_iam.xs(
        var_den, level="VARIABLE"
    )
    iam_sec_hydrogen_share = (
        iam_sec_hydrogen_share.xs(f"{region}r", level="REGION")
        .loc[model]
        .droplevel("UNIT")
    )
    if var_den not in df.reset_index().VARIABLE.unique():
        return pd.DataFrame()
    sec_hydrogen = fun_add_multiply_dfmultindex_by_dfsingleindex(
        df.xs(var_den, level="VARIABLE"),
        iam_sec_hydrogen_share,
        operator="*",
    )

    sec_hydrogen["VARIABLE"] = var_num
    sec_hydrogen = sec_hydrogen.reset_index().set_index(idxname)
    return sec_hydrogen


if __name__ == "__main__":
    # NOTE: use `run_multiple_files.py`. Do not run here
    main(
        project_file="EU_climate_advisory_board_2023",
        model_patterns=[
            # "REMIND-MAgPIE 2.1-4.2",
            # "REMIND-MAgPIE 2.1-4.3",
            # "MESSAGEix-GLOBIOM 1.0",
            # "AIM_CGE 2.2",
            "IMAGE 3.2",
        ],  # ["*"],
        region_patterns=["*Marker*"],
        target_patterns=["*"],
        file_suffix="2023_02_22",
        model_in_region_name=True,
        use_eea_data=True,  # If False uses PRIMAP
    )
