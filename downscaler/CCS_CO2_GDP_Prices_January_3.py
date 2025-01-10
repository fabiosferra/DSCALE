import os
import time
from pathlib import Path
from typing import Union

import pandas as pd

import downscaler
import downscaler.Electricity_2b
from downscaler import CONSTANTS
from downscaler.fixtures import fun_conv_settings
from downscaler.input_caching import get_selection_dict
from downscaler.utils import (
    InputFile,
    convert_to_list,
    fun_get_ssp,
    fun_pd_sel,
    fun_read_df_countries,
    fun_read_df_iam_all,
    fun_read_reg_value,
    fun_sel_iam,
    fun_validation,
    load_model_mapping,
    setindex,
    unique,
    fun_rename_index_name,
    fun_drop_duplicates,
    fun_xs,
    fun_drop_columns,
    fun_read_csv,
)



def fun_iamc_unmelt(_df_all: pd.DataFrame) -> pd.DataFrame:
    """2021_01_28
    This function Unmelts dataframe: from iamc/long format  => to wide shape format
    NOTE: it does nor work if _df_all contains more than one scenario. need to select only 1 scenario in _df_all
    TO REVERT TO THE STANDARD DATAFARME PLEASE USE => fun_iamc_melt (available in utils.py)
    """

    m = ["MODEL", "SCENARIO", "ISO", "VARIABLE"]
    setindex(_df_all, m)

    setindex(_df_all, ["ISO", "MODEL", "SCENARIO", "VARIABLE"])  # .pivot(['VARIABLE'])
    _df_all_new = pd.DataFrame(
        _df_all.stack()
    )  ## Creating a new dataframe (in wide format)
    setindex(_df_all_new, False)

    ## column name corresponing to 'TIME'
    col_time_name = _df_all_new.iloc[
        :, _df_all_new.columns.str.contains("level").fillna(False)
    ].columns[0]

    ## renaming 'TIME' column
    _df_all_new.rename(columns={col_time_name: "TIME", 0: "VALUE"}, inplace=True)
    _df_all_new["TIME"] = _df_all_new["TIME"].astype(int)

    ## Pivot 'VARIABLE'
    setindex(_df_all_new, ["ISO", "MODEL", "SCENARIO", "TIME"])
    _df_all_new = _df_all_new.pivot(columns="VARIABLE")
    _df_all_new = _df_all_new["VALUE"]
    return _df_all_new


def fun_primary_fossil(_df_all: pd.DataFrame, _var_iam: str) -> pd.DataFrame:
    """
    This function created 'Primary Energy|Fossil' as the sum of:
    - Primary Energy|Coal
    - Primary Energy|Gas
    - Primary Energy|Oil

    """
    col_fossil = list(
        _df_all.iloc[
            :,
            (_df_all.columns.str.contains("Primary Energy"))
            & (
                (
                    _df_all.columns.str.contains("Coal")
                    | _df_all.columns.str.contains("Oil")
                    | _df_all.columns.str.contains("Gas")
                )
            )
            & (_df_all.columns.str.contains(_var_iam)),
        ].columns
    )

    ## 2021_12_15
    _df_all["Primary Energy|Fossil" + _var_iam] = (
        _df_all[col_fossil].fillna(0).sum(axis=1)
    )
    return _df_all


def fun_region_name(model: str, region: str) -> str:  ## 2020_12_03
    if (
        region.find(model) == -1
    ):  ## we do this for MESSAGE but only if model name is not  present in region_name
        region_name = model + "|" + region

    else:  ## model name is already present in region_name (the region_name is already correct)
        region_name = region

    return region_name


def fun_ccs_downs_enlong(
    model: str,
    region: str,
    target: str,
    var_iam: str,
    var: str,
    df_all: pd.DataFrame,
    df_iam_all_models: pd.DataFrame,
) -> pd.DataFrame:
    """
    # 2021_01_18
    This function downscales CCS technologies by using the same regional allocation (same %)
    It also calculates  w/o CCS as the difference

    """
    var_iam0 = var_iam  ## Requested level

    ## Step 0 Var_iam should contain string 'w/ CCS'. If not, we update the variable name
    if var_iam.find("w/o") != -1:
        print("replacing")
        var_iam = var_iam.replace("w/o", "w/")

    # Step 1a) Get regional Share at requested Energy level (e.g. Final)
    ccs_region = fun_read_reg_value(model, region, target, var_iam, df_iam_all_models)[
        "VALUE"
    ]
    woccs_region = fun_read_reg_value(
        model, region, target, var_iam.replace("w/", "w/o"), df_iam_all_models
    )["VALUE"]

    # Step 1b) Get regional information at Secondary energy level (if requested level is not available)
    if len(ccs_region.dropna(how="all")) == 0:  # ## If CCS info not available
        print("Get info from Secondary energy level")
        var_iam = (
            "Secondary Energy" + "|" + "|".join(var_iam.rsplit("|")[2:])
        )  ## e.g. Biomass w/ CCS
        #         print('var_iam',var_iam)
        ccs_region = fun_read_reg_value(
            model, region, target, var_iam, df_iam_all_models
        )["VALUE"]
        woccs_region = fun_read_reg_value(
            model, region, target, var_iam.replace("w/", "w/o"), df_iam_all_models
        )["VALUE"]

    # Step 2) Calculating regional share
    ratio_region = ccs_region.fillna(0) / (ccs_region.fillna(0) + woccs_region)

    # Step 3) Calculating variable, based on regional share
    var_iam_tot_fuel = "|".join(var_iam0.rsplit("|")[:-1])

    ## Fix primary energy error  from step2 (2022_03_08). `Primary energy|Coal` sometimes drop to zero even though there are data in at the regional level
    df_all[var_iam_tot_fuel + var].ffill(axis=0, inplace=True)
    ratio_primary = (
        fun_read_reg_value(
            model,
            region,
            target,
            var_iam_tot_fuel.replace("w/", "w/o"),
            df_iam_all_models,
        )["VALUE"]
        / df_all[var_iam_tot_fuel + var].groupby("TIME").sum()
    )
    df_all[var_iam_tot_fuel + var] = df_all[var_iam_tot_fuel + var] * ratio_primary
    df_all[var_iam0 + var] = df_all[var_iam_tot_fuel + var] * ratio_region

    # Step 4) Calculating variable w/o CCS, based on regional share
    df_all[var_iam0.replace("w/", "w/o") + var] = df_all[var_iam_tot_fuel + var] * (
        1 - ratio_region
    )

    ## Step 5) If w/o CCS info Not available, it will coincide with Total value information
    if len(df_all[var_iam0.replace("w/", "w/o") + var].dropna(how="all")) == 0:
        df_all[var_iam0.replace("w/", "w/o") + var] = df_all[var_iam_tot_fuel + var]

    return df_all


def fun_co2_emissions_primary(
    model: str,
    target: str,
    region: str,
    var_iam: str,
    var: str,
    df_all: pd.DataFrame,
    df_emi_factors: pd.DataFrame,
    df_iam_all_models: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function calculates emissions for Fossil fuels (W/ CCS and W/O CCS), based on emissions factors from df_emi_factors.
    For technologies W/CCS it calculated Emissions Captured/

    For Biomass CCS capture: it harmonises emissions to match 'Carbon Sequestration|CCS|Biomass'
    For COAL/OIL/GAS: it harmonises emissions to match 'Carbon Sequestration|CCS|Fossil'


    """

    from_c_to_co2 = 44 / 12  ## conversion from carbon to CO2

    ## Step 1 Creating two of variables: both CCS and non-CCS variables
    if var_iam.find("w/o") != -1:
        var_iam2 = var_iam.replace("w/o", "w/")
    else:
        var_iam2 = var_iam.replace("w/", "w/o")
    var_iam_list = [var_iam, var_iam2]  ## This is:  ['w/ CCS',   'w/o CCS']

    fuel_list = ["Biomass", "Coal", "Gas", "Oil"]

    ## Step 2 Calculating emissions
    for f in fuel_list:
        for var_iam in var_iam_list:
            ## Step 2.1: Get emission factors associated to variable/model
            n = 1
            groups = var_iam.split("|")
            groups
            idx = ("|".join(groups[:n])[1], "|".join(groups[n:]))[1]
            df_emi_factors.loc[idx, model]

            ## Step 2.2 Applying emission factors:
            #### Step 2.2.1a  BIOMASS W/ CCS EMISSIONS (based on emi factor)
            if (
                var_iam.find("Biomass") != -1 and var_iam.find("w/ CCS") != -1
            ):  ## BECCS emission captured
                print("BECCS", var_iam)

                ## 2021_12_15
                df_all[
                    var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
                ] = (
                    df_all[var_iam.replace("|w/ CCS", "") + var].fillna(0)
                    * df_emi_factors.loc[idx, model]
                    * from_c_to_co2
                )  # *fun_primary_conv(model,target, region, var_iam)

                #### Step 2.2.1b  Harmonisation to match IAM results ('Carbon Sequestration|CCS|Biomass')
                if (
                    len(
                        fun_read_reg_value(
                            model,
                            region,
                            target,
                            "Carbon Sequestration|CCS|Biomass",
                            df_iam_all_models,
                        )["VALUE"]
                    )
                    >= 1
                ):
                    ratio = df_all[
                        var_iam.replace("Primary Energy", "Primary Energy Emissions")
                        + var
                    ] / df_all[
                        var_iam.replace("Primary Energy", "Primary Energy Emissions")
                        + var
                    ].groupby(
                        "TIME"
                    ).sum()
                    df_all[
                        var_iam.replace("Primary Energy", "Primary Energy Emissions")
                        + var
                    ] = (
                        ratio
                        * fun_read_reg_value(
                            model,
                            region,
                            target,
                            "Carbon Sequestration|CCS|Biomass",
                            df_iam_all_models,
                        )["VALUE"]
                    )
                    df_all["Carbon Sequestration|CCS|Biomass" + var] = df_all[
                        var_iam.replace("Primary Energy", "Primary Energy Emissions")
                        + var
                    ]

                #### Step 2.2.2 BIOMASS W/O CCS EMISSIONS (based on emi factor)
            elif var_iam.find("Biomass") != -1:
                df_all[
                    var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
                ] = (
                    df_all[var_iam.replace("|w/o CCS", "") + var].fillna(0)
                    * df_emi_factors.loc[idx, model]
                    * from_c_to_co2
                )

                # (no harmonisation as we do not have emissions for biomass w/o CCS)

                ### Step 2.2.3 COAL/GAS/OIL EMISSIONS (With and Without CCS) (based on emi factor)
            else:  ## fossil emissions:
                #                 print('questo else non va bene' )
                df_all[
                    var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
                ] = (
                    df_all[var_iam + var]
                    * df_emi_factors.loc[idx, model]
                    * from_c_to_co2
                )  # *fun_primary_conv(model,target, region, var_iam)

                # (no harmonisation as we do not have emissions for individual COAL/OIL/GAS)

    ## Step 3 FOSSIL CCS CAPTURED Harmonisation. to match IAM results ('Carbon Sequestration|CCS|Fossil')
    #     if len(fun_read_reg_value(model, region, target, 'Carbon Sequestration|CCS|Fossil')['VALUE'])>=1:
    try:
        fossil_list = [
            "Primary Energy Emissions|Coal|w/ CCS",
            "Primary Energy Emissions|Gas|w/ CCS",
            "Primary Energy Emissions|Oil|w/ CCS",
        ]
        fossil_list = [f + var for f in fossil_list]

        ## 2021_12_15 above replaced by below
        ratio = (
            fun_read_reg_value(
                model,
                region,
                target,
                "Carbon Sequestration|CCS|Fossil",
                df_iam_all_models,
            )["VALUE"].fillna(0)
            / df_all[fossil_list].sum(axis=1).groupby("TIME").sum()
        )

        for fos in fossil_list:
            df_all[fos] = df_all[fos] * ratio
        print("harmonised FOSSI CCS Captured!")
        df_all["Carbon Sequestration|CCS|Fossil" + var] = df_all[fossil_list].sum(
            axis=1
        )
    except:
        pass
    return df_all


def fun_pd_long_format(_df: pd.DataFrame) -> pd.DataFrame:
    """
    It Returns a df in long format .
    """
    m = ["TIME", "ISO"]
    setindex(_df, m)
    a = _df.stack()  # .unstack()#_main['Final Energy|Industry|HeatENSHORT_REF']
    b = pd.DataFrame(a)
    setindex(b, False)
    setindex(b, "ISO")
    b.rename(columns={"level_2": "VARIABLE"}, inplace=True)
    b.rename(columns={0: "VALUE"}, inplace=True)
    return b

    ## VAR_IAM IS ACTUALLY (PROBABLY NOT NEEDED IN THIS FUNCTION)


def fun_co2_emissions_energy(
    model: str,
    target: str,
    region: str,
    var_iam: str,
    var: str,
    df_all: pd.DataFrame,
    df_iam_all_models: pd.DataFrame,
    fossil_ccs_leakage: float = 0.1,
) -> pd.DataFrame:
    """
    This function calculates 'Emissions|CO2|Energy' based on previosuly downscaled 'Primary Energy Emissions' by fuels.
    It Assumes some leakage (fossil_ccs_leakage=0.1) from 'Carbon Sequestration|CCS|Fossil'.

    Then it harmonises the results tom match: 'Emissions|CO2|Energy' at the regional level

    """
    wo_ccs_cols = [
        "Primary Energy Emissions|Coal|w/o CCS",
        "Primary Energy Emissions|Gas|w/o CCS",
        "Primary Energy Emissions|Oil|w/o CCS",
    ]
    wccs_cols = [
        "Primary Energy Emissions|Coal|w/ CCS",
        "Primary Energy Emissions|Gas|w/ CCS",
        "Primary Energy Emissions|Oil|w/ CCS",
    ]

    wo_ccs_cols = [f + var for f in wo_ccs_cols]
    wccs_cols = [f + var for f in wccs_cols]

    df_all[wo_ccs_cols].fillna(0).sum(axis=1)
    df_all[wccs_cols].fillna(0).sum(axis=1)

    ## Calculating emissions from Fossils without harmonisation:
    df_all["Emissions|CO2|Energy" + var] = (
        df_all[wo_ccs_cols].fillna(0).sum(axis=1)
        + df_all[wccs_cols].fillna(0).sum(axis=1) * fossil_ccs_leakage
        #                                         -df_all['Primary Energy Emissions|Biomass|w/ CCS'+var]
    )

    ## Harmoninising Emissions from FOSSILS:
    beccs=fun_read_reg_value(
            model, region, target, "Carbon Sequestration|CCS|Biomass", df_iam_all_models
        )["VALUE"].fillna(0)
    emi_co2=fun_read_reg_value(
            model, region, target, "Emissions|CO2|Energy", df_iam_all_models
        )["VALUE"].fillna(0)
    gross_emi= emi_co2+beccs if len(beccs) else emi_co2
    ratio = gross_emi / df_all["Emissions|CO2|Energy" + var].groupby("TIME").sum()
    df_all["Emissions|CO2|Energy" + var] = df_all["Emissions|CO2|Energy" + var] * ratio

    ## Calculating 'Emissions|CO2|Energy' as difference between: Fossil Emissions - BECCS captured:
    # (Both fossil emissions and BECCS have been harmonised to match IAM results)

    # df_all['Emissions|CO2|Energy'+var]=df_all['Emissions|CO2|Energy'+var]-df_all['Primary Energy Emissions|Biomass|w/ CCS'+var]

    ## 2021_12_15 below:
    df_all["Emissions|CO2|Energy" + var] = df_all[
        "Emissions|CO2|Energy" + var
    ] - df_all["Primary Energy Emissions|Biomass|w/ CCS" + var].fillna(0)
    return df_all


## Creating a function for GDP downscaling 2021_01_29


def fun_ssp_downscaling(
    model: str,
    region: str,
    target: str,
    var_iam: str,
    countrylist: list,
    ssp_model: str,
    df_iam_all_models: pd.DataFrame,
    ssp_scenario: str = "SSP2",
    ssp_file: str = "GDP_NGFS_merged.csv",  # "SspDb_country_data_2013-06-12.csv",
    ref_scen: str = "h_cpol",  ## GDP losses compared to this reference scenario
) -> pd.DataFrame:
    """
    This function downscales GDP (or any other SSP variable) at the country level by applying the regional % change (against a reference scenario) at the country level.
    GDP losses are computed against a reference scenario (ref_scen).
    If not possibles to downscale data (e.g. regional data not provided by IAMs) it return the raw SSP data.

    NOTEs:
    i) this function usually is applied to GDP, but could be also applied to population
    ii) This function does not guarantee that the sum of GDP coincides with region IAM results
    (e.g. mismatch could happen in case of  wrong regional mapping),
    Hence it differs from the function fun_downs_gdp (which guarantees consistency with IAM results)
    """

    col_list_time = range(2010, 2105, 5)
    col_list_time = [str(c) for c in col_list_time]

    ## STEP 0 reading Pop data from df_ssp ## 2020_12_17
    df_ssp = pd.read_csv(
        CONSTANTS.INPUT_DATA_DIR / ssp_file, sep=",", encoding="utf-8"
    )  # encoding='latin-1')
    df_ssp.rename(columns={"REGION": "ISO"}, inplace=True)

    # NOTE: block below fixes NGFS2023 (27 Oct 2023).
    if target in df_ssp.SCENARIO.unique():
        ssp_scenario = target
    else:
        ssp_scenario = fun_get_ssp(target)

    ssp_model = get_ssp_model(model, ssp_model, df_ssp)
    ## STEP 1 Selecting gdp data from df_ssp ## 2020_12_17
    df_gdp = df_ssp[(df_ssp.MODEL == ssp_model) & (df_ssp.VARIABLE == var_iam)].copy(
        deep=True
    )
    if target in df_ssp.SCENARIO.unique():
        df_gdp = df_gdp[df_gdp.SCENARIO == ssp_scenario]
    else:
        df_gdp = df_gdp[df_gdp.SCENARIO.str.contains(ssp_scenario)]

    df_gdp = df_gdp[df_gdp.ISO.isin(countrylist)]

    ## STP2 Calculating % Losses at the regional level (compared to ref_scen)
    ratio_gdp = (
        fun_read_reg_value(
            model,
            region,
            target,
            var_iam,
            df_iam_all_models,
        )["VALUE"]
        / fun_read_reg_value(model, region, ref_scen, var_iam, df_iam_all_models)[
            "VALUE"
        ]
    )
    # if ref_scen not in df_iam_all_models.VARIABLE:
    #     raise ValueError(f"The {ref_scen} scenario is missing in the regional IAMs dataframe (this is our reference scenario for downscaling GDP.")
    ## Convert pandas index from float to string (astype)  https://stackoverflow.com/questions/35368645/pandas-change-df-index-from-float64-to-unicode-or-string/35368792
    ratio_gdp.index = ratio_gdp.index.map(str)

    ## STP3 Applying % Losses at the country level
    list2 = range(2010, 2105, 5)  ## Time horizon
    list2str = [str(i) for i in list2]
    df_gdp[list2str] = df_gdp[list2str] * ratio_gdp[list2str]

    df_gdp_down = df_gdp

    ## STEP 4 Renaming MODEL/SCENARIO Both REMIND and MESSAGE, use the OECD and IIASA datasets respectively for GDP and population (See emails from Jerome and Volker both on the 16/12/2020.
    ### step 4.1 header of the csv file
    col_list = [
        "MODEL",
        "SCENARIO",
        "ISO",
        "VARIABLE",
        "UNIT",
    ] + col_list_time  ## ['MODEL', 'SCENARIO', 'ISO', 'VARIABLE', 'UNIT', '2010', '2015', '2020', '2025', '2030', '2035', '2040', '2045', '2050', '2055', '2060', '2065', '2070', '2075', '2080', '2085', '2090', '2095', '2100']
    setindex(df_gdp_down, False)
    df_gdp_down["MODEL"] = model  ## Renaming model
    df_gdp_down["SCENARIO"] = target  ## Renaming scenario
    setindex(df_gdp_down, ["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"])

    try:
        return df_gdp_down.drop(
            "2005", axis=1
        )  ## returns downscaled data and dropping 2005 data)
    except:
        if len(df_gdp_down) >= 1:
            return df_gdp_down  ## return downscaled data ( without dropping 2005 data)
        else:
            setindex(df_gdp, ["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"])
            return df_gdp  ## return data as in SSP baseline (no downscaling)


def fun_unmelt_df_iam_all_models(
    model: str,
    target: str,
    region: str,
    variabile_iam: str,
    df_iam_all_models: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function returns selected variables of the df_iam_all_models in a wide format.
    If len(region)>=1 then it selects results only for one region.
    Otherwise it returns results for all regions (please use => region='').

    This funcion works:
    - with no region selected (e.g. region='')
    - with 1 region selected using string (e.g. region='Western Europe')
    - with 1 or more regions selected using lists (e.g. region=['MESSAGEix-GLOBIOM 1.0|Centrally Planned Asia and Chinar'])

    """

    ## Here we define how many regions we should return
    if len(region) >= 1:
        select_region_flag = True
    else:
        select_region_flag = False

    ## Change type from string to list:
    if type(model) == str:
        model = [model]
    if type(target) == str:
        target = [target]
    if type(variabile_iam) == str:
        variabile_iam = [variabile_iam]

    ## NOTE: df_iam_all_models not defined in this function => will use  global variable
    setindex(
        df_iam_all_models, ["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"]
    )  # .unstack()

    if select_region_flag == True:  ## Here we select one region (if len region >=2)
        if type(region) == str:  ## Region is a string
            df_wide = df_iam_all_models[
                (df_iam_all_models.index.get_level_values(0).isin(model))
                & (df_iam_all_models.index.get_level_values(1).isin(target))
                & (df_iam_all_models.index.get_level_values(2).str.contains(region))
                & (df_iam_all_models.index.get_level_values(3).isin(variabile_iam))
            ].pivot(columns="TIME")["VALUE"]
        else:  ## Region is a list
            df_wide = df_iam_all_models[
                (df_iam_all_models.index.get_level_values(0).isin(model))
                & (df_iam_all_models.index.get_level_values(1).isin(target))
                & (df_iam_all_models.index.get_level_values(2).isin(region))
                & (df_iam_all_models.index.get_level_values(3).isin(variabile_iam))
            ].pivot(columns="TIME")["VALUE"]

    else:  ## Here we select all regions (otherwise)
        df_wide = df_iam_all_models[
            (df_iam_all_models.index.get_level_values(0).isin(model))
            & (df_iam_all_models.index.get_level_values(1).isin(target))
            & (df_iam_all_models.index.get_level_values(3).isin(variabile_iam))
        ].pivot(columns="TIME")["VALUE"]

    setindex(df_iam_all_models, False)

    try:
        return df_wide.drop(2005, axis=1)  ## dropping 2005 values
    except:
        return df_wide


RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR(
    str(Path(os.path.abspath("")) / Path(__file__).name)
)
RESULTS_DATA_DIR.mkdir(exist_ok=True)


def main(
    project_name="NGFS",
    model_patterns: Union[str, list] = "*",
    region_patterns: Union[str, list] = "*",
    target_patterns: Union[str, list] = "*",
    gdp_pop_down=True,
    add_model_in_region_name=False,
    file_suffix="__NGFS_November",  #'__round_1p4_enlong_reg_lenght.csv'## CSV in
    file="snapshot_all_regions.csv",
    single_pathway=True,  ## We show only one pathway for each scenario
    add_gdp_pop_data=False,
    harmonize_gdp_pop_with_reg_iam_results=False,
    _scen_dict=None,
    conv_dict_inv=None,
    run_sensitivity=False,
    sensitivity_dict=None,
    default_ssp_scenario: str = "SSP2",
    gdp_model="NGFS",
    pop_model="IIASA-WiC POP",
    ref_scen="h_cpol",
    method='wo_smooth_enlong',
    long_term='ENLONG_RATIO',
    func='log-log',
    criteria='standard',

    # downscaler.USE_CACHING=True,
):
    """This file downscales Emissions and Price variables and save data in a file called f'{model}_{file_suffix}'.
    If `gdp_pop_down==True` it  downscales `GDP|PPP` and `Population` and save data in f'GDP_{file_suffix}_updated_gdp_harmo'.
    In this file both GDP and Population are harmonized to match regional IAM results

    Parameters
    ----------
    project_name : str, optional
        Project name, by default "NGFS"
    model_patterns : Union[str, list], optional
        List of models, by default "*"
    region_patterns : Union[str, list], optional
        List of regions, by default "*"
    target_patterns : Union[str, list], optional
        List of targets, by default "*"
    gdp_pop_down : bool, optional
        Whether you want to downscale GDP and Population, by default True
    add_model_in_region_name : bool, optional
        Whether you want to add model in the region name (in IAMs regions), by default False
    file_suffix : str, optional
        File suffix for the csv file name, by default "__NGFS_November"
    single_pathway : bool, optional
        Wheter yoy want to run a single vs a range of pathways, by default True
    harmonize_gdp_pop_with_reg_iam_results : bool, optional
        Wheter you want to harmonize gdp and pop with regional IAM results, by default False
    _scen_dict : _type_, optional
        Scenario dictionary with assumptions, by default None
    conv_dict_inv : _type_, optional
        Convergence dictionary, by default None
    run_sensitivity : bool, optional
        Whether you want to run a sensitivity run, by default False
    sensitivity_dict : _type_, optional
        Sensitivity assumptions, by default None
    default_ssp_scenario : str, optional
        default storyline for GDP/POP downscaling, by default "SSP2"
    gdp_model : str, optional
        GDP model for the socioeconomic data, by default "NGFS"
    pop_model : str, optional
        Population model for the socioeconomic data, by default "IIASA-WiC POP"
    ref_scen : str, optional
        Reference scenario for GDP/POP projections, by default "h_cpol"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    ValueError
        _description_
    """

    RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )

    # FIXTURE_DIR = Path(__file__).parents[1] / "input_data"
    input_file = CONSTANTS.INPUT_DATA_DIR / project_name / "snapshot_all_regions.csv"
    country_mapping_file = (
        CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
    )  # NOTE: can be replaced by pyam_mapping_file

    pyam_mapping_file = CONSTANTS.INPUT_DATA_DIR / project_name / "default_mapping.csv"

    region_patterns = convert_to_list(region_patterns)
    model_patterns = convert_to_list(model_patterns)
    target_patterns = convert_to_list(target_patterns)

    PREV_STEP_RES_DIR = CONSTANTS.PREV_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )

    print(region_patterns)

    # ## STEP 1 READ Regional iam results and AJUSTING REGION NAME
    df_iam_all_models = fun_read_df_iam_all(
        file=InputFile(input_file), add_model_name_to_region=add_model_in_region_name,
    )  ## READING IN IAMS RESULTS

    # df_iam_all = df_iam_all_models.copy(deep=True)
    # get_selection_dict will produce an error
    iam_results_file = InputFile(
        CONSTANTS.INPUT_DATA_DIR / project_name / "snapshot_all_regions.csv"
    )
    try:
        selection_dict = get_selection_dict(
            iam_results_file,
            model=model_patterns,
            region=region_patterns,
            target=target_patterns,
            variable=["*"],
        )
    except ValueError as e:
        raise ValueError(f"{e}\nConsider using the asterisk '*' for the pattern.")

    ## slice df based on selection_dict
    df_iam_all_models = fun_sel_iam(selection_dict, df_iam_all_models)
    df_iam_all_models.loc[:, "REGION"] = df_iam_all_models.loc[:, "REGION"] + "r"
    df_iam_all_models_input = df_iam_all_models.copy(deep=True)

    df_iam_all_models = df_iam_all_models_input[
        df_iam_all_models_input.VARIABLE.isin(["GDP|PPP", "Population"])
    ]

    # For oil, gas, and coal, we use the following emission factors: 18.4, 15.3 and 26.1 Mt C/EJ.
    # The capture rate of most CCS technologies is assumed to be 90 % (there is just one nearly never used technology with 99%),
    # so I guess setting the leakage rate to 10% is most consisten for REMIND.

    ## Emission Factors
    dict_prova = {
        "Oil|w/o CCS": 18.4,
        "Gas|w/o CCS": 15.3,
        "Coal|w/o CCS": 26.1,
        "Biomass|w/o CCS": 0,
        "Oil|w/ CCS": 18.4,
        "Gas|w/ CCS": 15.3,
        "Coal|w/ CCS": 26.1,
        "Biomass|w/ CCS": -20,
    }

    emi_prova = []
    emi_factors_data = pd.DataFrame()
    for model in selection_dict.keys():
        emi_prova = []
        for i, j in dict_prova.items():
            # print(model, i,j, type(emi_prova))
            emi_prova = emi_prova + [{model: j, "fuel": i}]  #'Oil|w/o CCS'
        df1 = pd.DataFrame(emi_prova)
        setindex(df1, "fuel")
        emi_factors_data = pd.concat([emi_factors_data, df1], axis=1)
    df_emi_factors = emi_factors_data

    var_iam = "Primary Energy|Biomass|w/o CCS"

    # Both REMIND and MESSAGE, use the OECD and IIASA datasets  for GDP and population respectively (See emails from Jerome and Volker both on the 16/12/2020)

    # var_list = ["2200_BLEND", "2250_BLEND", "2300_BLEND"]
    var_iam_list = [
        "Primary Energy|Coal|w/ CCS",
        "Primary Energy|Fossil|w/ CCS",
        "Primary Energy|Gas|w/ CCS",
        "Primary Energy|Oil|w/ CCS",
    ]

    file_name = "_updated_gdp.csv"  ##  CSV (out) (to be saved in IAMC format)
    # file_suffix = "__downscaled_separated_FEB_18_final.csv"
    # input_file= 'snapshot_all_regions_round_1p4_2021_05_11.csv'

    ###################################################################################################
    ## Creating a CSV with all GDP and Population projections (all countries, including native regions)
    ###################################################################################################
    print("This is the csv file:", input_file)
    s0 = time.time()
    timehorizon = [str(i) for i in range(2010, 2105, 5)]

    ## STEP 1 READ Regional iam results and AJUSTING REGION NAME
    count = 0
    header_flag = True  ## Header of CSV file
    mode = "w"
    for model in selection_dict.keys():
        targets = selection_dict[model]["targets"]
        df_countries = fun_read_df_countries(country_mapping_file)
        df_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file
        )
        regions = selection_dict[model]["regions"]
        if gdp_pop_down:
            fun_downscale_gdp(
                project_name,
                file_suffix,
                _scen_dict,
                RESULTS_DATA_DIR,
                input_file,
                df_iam_all_models,
                model,
                gdp_model,
                pop_model,
                file_name,
                s0,
                timehorizon,
                header_flag,
                mode,
                targets,
                df_countries,
                regions,
                default_ssp_scenario,
                ref_scen=ref_scen,
            )

            if harmonize_gdp_pop_with_reg_iam_results:
                ## NOTE: THE BELOW CHANGES the UNIT of GDP:
                # - from billion US$2005/y (as reported by SSP DATA)
                # - to billion US$2010/y (as reported by IAMs  DATA)
                fun_harmonize_gdp_with_reg_iam_results_and_save_to_csv(
                    file_suffix,
                    RESULTS_DATA_DIR,
                    input_file,
                    country_mapping_file,
                    pyam_mapping_file,
                    selection_dict,
                )

    ##################################################################
    ## Creating CSV file with Emissions, prices and all energy info ##
    ##################################################################

    time_horizon = list(range(2010, 2105, 5))

    ## added 2021_10_18 (removing 'Primary Energy|Fossil from list)
    var_iam_list = [
        "Primary Energy|Coal|w/ CCS",
        "Primary Energy|Fossil|w/ CCS",
        "Primary Energy|Gas|w/ CCS",
        "Primary Energy|Oil|w/ CCS",
    ]

    s0 = time.time()
    timehorizon = [str(i) for i in range(2010, 2105, 5)]

    count = 0

    for model in selection_dict.keys():
        header_flag = True  ## Header of CSV file
        mode = "w"
        targets = selection_dict[model]["targets"]
        ## STEP 2 IMPORTING REGIONS for selected model
        df_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file
        )
        regions = selection_dict[model]["regions"]
        ## STEP 1 READING DATA and Reshape df
        df_all_scen = fun_read_csv(
                        {'aa':
                        PREV_STEP_RES_DIR / (model + file_suffix + ".csv")},
                        True, str
                        )['aa']
        
        # Filter the dataframe for selected METHOD, LONG_TERM, FUNC, CRITERIA
        if 'METHOD' in df_all_scen.index.names:
            df_all_scen=df_all_scen.xs(method ,level='METHOD')
        if 'LONG_TERM'  in df_all_scen.index.names:
            df_all_scen=df_all_scen.xs(long_term,level='LONG_TERM')
        if 'FUNC' in df_all_scen.index.names:
            df_all_scen=df_all_scen.xs(func,level='FUNC')
        if 'CRITERIA' in df_all_scen.index.names:
            df_all_scen=df_all_scen.xs(criteria,level='CRITERIA')
        
        if 'UNIT' in df_all_scen.index.names:
            df_all_scen=df_all_scen.droplevel('UNIT')
        # try:
        #     df_all_scen.drop(
        #         [("MODEL", "SCENARIO", "ISO", "VARIABLE")], axis=0, inplace=True
        #     )
        # except:
        #     pass

        ## 2021_10_18 remove possible lines with numeric data in the 'MODEL' column (index)
        try:
            blacklist = df_all_scen[
                df_all_scen.index.get_level_values("MODEL").str.isnumeric().fillna(True)
            ].index.tolist()
            df_all_scen.drop(blacklist, axis=0, inplace=True)
        except:
            pass

        ## 2021_10_18 added due to duplicated values
        df_all_scen = df_all_scen.iloc[~df_all_scen.index.duplicated(keep="first")]

        ## getting dataframe in the right (wide) format
        df_all_scen = fun_iamc_unmelt(df_all_scen)

        regions = [r + "r" for r in regions]
        ## STEP 3 REGION LOOP
        for region in regions:  # [:1]:
            r = region
            print(region)
            countrylist = unique(
                df_countries[
                    df_countries.REGION == region.replace(model + "|", "")
                ].ISO.to_list()
            )

            ## This means we have more that one country (we downscale CCS and emissions)df_all_scen
            if len(countrylist) >= 2:
                for scenario in targets:
                    # with sensitivity
                    if run_sensitivity:
                        if sensitivity_dict is None:
                            raise ValueError(
                                f"If you want to run a sensitivity for {scenario} scenario, please provide a `sensitivity_dict` (please check the `scenario_config.csv` file) or this PR https://github.com/iiasa/downscaler_repo/pull/135"
                            )
                        if scenario not in sensitivity_dict:
                            raise ValueError(
                                f"{scenario} scenario is not present in the `sensitivity_dict`. Please check the `scenario_config.csv` file"
                            )
                        # for scenario in sensitivity_dict:
                        for sens in sensitivity_dict[scenario]:
                            header_flag = True
                            (
                                header_flag,
                                mode,
                                df_return,
                            ) = fun_downscale_emi_add_price_and_save_to_csv(
                                project_name,
                                file_suffix,
                                file,
                                single_pathway,
                                conv_dict_inv,
                                RESULTS_DATA_DIR,
                                df_iam_all_models_input,
                                model,
                                df_emi_factors,
                                var_iam_list,
                                header_flag,
                                mode,
                                time_horizon,
                                df_all_scen,
                                region,
                                countrylist,
                                scenario,
                                run_sensitivity,
                                sens,
                            )
                            header_flag = True
                            mode = "a"
                    else:
                        # without sensitivity
                        (
                            header_flag,
                            mode,
                            df_return,
                        ) = fun_downscale_emi_add_price_and_save_to_csv(
                            project_name,
                            file_suffix,
                            file,
                            single_pathway,
                            conv_dict_inv,
                            RESULTS_DATA_DIR,
                            df_iam_all_models_input,
                            model,
                            df_emi_factors,
                            var_iam_list,
                            header_flag,
                            mode,
                            time_horizon,
                            df_all_scen,
                            region,
                            countrylist,
                            scenario,
                            False,
                            None,
                        )

    if add_gdp_pop_data:
        for model in selection_dict:
            downs_path = RESULTS_DATA_DIR / f"{model}{file_suffix}.csv"
            df_downs = pd.read_csv(
                downs_path,
                index_col=["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"],
            )
            cols = [str(x) for x in range(2010, 2105, 5)]

            if ("MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT") in df_downs.index:
                df_downs = df_downs.drop(
                    ("MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT")
                )
            df_downs = fun_drop_duplicates(df_downs)
            df_downs = fun_xs(df_downs, {"SCENARIO": targets})
            df_downs.to_csv(downs_path)
            df_return = df_downs

    print("Elapsed:", round(-s0 + time.time(), 2), "Seconds")

    #######################################################################
    # BLUEPRINT FOR anchoring Final Energy and emissions to historical data
    # NOTE: in 2010 we already match the data
    # DO NOT DELETE THE BELOW!!!!
    ######################################################################
    # from downscaler.utils import fun_create_iea_dict_from_iea_flow_dict
    # from downscaler.fixtures import iea_flow_dict

    # df_iea_all = pd.read_csv(
    #     CONSTANTS.INPUT_DATA_DIR / "Extended_IEA_en_bal_2019_ISO.csv",
    #     sep=",",
    #     encoding="latin-1",
    # )

    # iea_dict = {
    #     "Emissions|CO2|Energy": {"flow": ["CO2 fuel combustion"], "product": ["Total"],}
    # }

    # var = "Final Energy"
    # # #df_return=
    # fun_anchor_emi_to_hist_data_step3(
    #     input_file,
    #     countrylist,
    #     df_return,
    #     fun_create_iea_dict_from_iea_flow_dict([var], iea_flow_dict),
    #     var,
    #     [2015, 2020],
    # )

    # var = "Emissions|CO2|Energy"
    # # #df_return=
    # fun_anchor_emi_to_hist_data_step3(
    #     input_file, countrylist, df_return, iea_dict, var, [2015, 2020]
    # )
    # NOTE: this  would also require updating primary energy variables
    # NOTE: to anchor emissions data you need to update the `df_return` (currently we are not anchoring emissions, nor energy data)
    return df_return


def fun_downscale_emi_add_price_and_save_to_csv(
    project_name,
    file_suffix,
    file,
    single_pathway,
    conv_dict_inv,
    RESULTS_DATA_DIR,
    df_iam_all_models_input,
    model,
    df_emi_factors,
    var_iam_list,
    header_flag,
    mode,
    time_horizon,
    df_all_scen,
    region,
    countrylist,
    scenario,
    run_sensitivity,
    sensitivity_conv,
):
    conv_settings = fun_conv_settings(run_sensitivity)

    ## Sclicing df_iam_all_models
    if scenario in df_all_scen.index.get_level_values("SCENARIO").unique():
        df_iam_all_models = df_iam_all_models_input[
            (df_iam_all_models_input.REGION == region)
            & ((df_iam_all_models_input.SCENARIO == scenario))
        ]

        ## STEP 4.1 Creating Dataframe 'df_all' with region-specific results:
        target = scenario
        df_all = fun_pd_sel(
            df_all_scen[
                (df_all_scen.index.get_level_values(2) == scenario)
                & (df_all_scen.index.get_level_values(1) == model)
            ],
            "",
            countrylist,
        )

        if len(df_all) == 0:
            print(
                f"*** No downscaled data found for {model} {region}. We skip this region ***"
            )
            df_return = pd.DataFrame()
        else:
            ## Single pathway for each scenario
            if single_pathway == True:
                if run_sensitivity:
                    var_list_fen = [f"{conv_settings['Final'][sensitivity_conv]}_BLEND"]
                    var_list = [f"{conv_settings['Secondary'][sensitivity_conv]}_BLEND"]
                    downscaled_co2_file_name = RESULTS_DATA_DIR / (
                        model + file_suffix + f"_sensitivity_{sensitivity_conv}.csv"
                    )
                else:
                    downscaled_co2_file_name = RESULTS_DATA_DIR / (
                        model + file_suffix + ".csv"
                    )
                    if conv_dict_inv is not None and target in conv_dict_inv:
                        conv = conv_dict_inv[target]
                    else:
                        conv = "MED"

                    var_list_fen = [f"{conv_settings['Final'][conv]}_BLEND"]
                    var_list = [f"{conv_settings['Secondary'][conv]}_BLEND"]
            else:  ## Range of pathways for all scenarios
                var_list_fen = [
                    "2100_BLEND",
                    "2150_BLEND",
                    "2200_BLEND",
                ]
                var_list = ["2200_BLEND", "2250_BLEND", "2300_BLEND"]

                ## STEP 4.2 Creating a new variable: 'Primary Energy|Fossil' for the full range of projections
                ## loop across range of projections
            for variabile in var_list:
                var = variabile
                print("var", variabile)
                fun_primary_fossil(df_all, variabile)

                ## STEP 4.3 Distinguish CCS and w/o CCS technologies
                for var_iam in var_iam_list:
                    print("var_iam", var_iam)
                    fun_ccs_downs_enlong(
                        model,
                        region,
                        scenario,
                        var_iam,
                        variabile,
                        df_all,
                        df_iam_all_models,
                    )

                    ## STEP 4.4 Calculating emissions by fuel (same share of regional_iam results) incl. 'Carbon Sequestration|CCS|Biomass', 'Carbon Sequestration|CCS|Fossil'
            fuel_list = ["Biomass", "Coal", "Gas", "Oil"]
            #             var_list=['2200_BLEND','2250_BLEND''2300_BLEND']
            main_var = "Primary Energy|fuel|w/ CCS"
            for f in fuel_list:
                var_iam = main_var.replace("fuel", f)
                print(var_iam)
                for variabile in var_list:
                    print(variabile)
                    fun_co2_emissions_primary(
                        model,
                        scenario,
                        region,
                        var_iam,
                        variabile,
                        df_all,
                        df_emi_factors,
                        df_iam_all_models,
                    )

                    ## STEP 4.5 Calculating 'Emissions|CO2|Energy'
            for variabile in var_list:
                fun_co2_emissions_energy(
                    model,
                    scenario,
                    region,
                    var_iam,
                    variabile,
                    df_all,
                    df_iam_all_models,
                    fossil_ccs_leakage=0.1,
                )

                ## STEP 5 Reshape df_all in iamc_format (new dataframe = c)
            a = fun_pd_long_format(df_all)  # .pivot(columns='TIME')
            b = a[(a.VALUE != model) & (a.VALUE != scenario)]  # .pivot(columns='TIME')
            c = setindex(b, ["ISO", "VARIABLE"]).pivot(columns="TIME")
            c["MODEL"] = model
            c["SCENARIO"] = scenario
            c["UNIT"] = "EJ/yr"
            setindex(c, ["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"])

            ## STEP 5.1 Changing UNITs for Emissions

            mtco2_unit_list = list(
                df_all.iloc[
                    :,
                    (df_all.columns.str.contains("Emissions"))
                    | df_all.columns.str.contains("Sequestration"),
                ].columns
            )  ## Creating a list of variables with unit = 'Mt CO2/yr'

            setindex(c, ["MODEL", "SCENARIO", "ISO", "VARIABLE"])  # ['VALUE']#['TIME']
            idxc=fun_xs(c, {"VARIABLE":mtco2_unit_list}).index
            c.loc[
                idxc,
                "UNIT",
            ] = "Mt CO2/yr"  #:,:,'Emissions|CO2|Energy2250_BLEND']

            setindex(c, ["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"])
            # c.loc[model, scenario, countrylist, mtco2_unit_list]

            if single_pathway == True:  ## Selecting only one pathway for each scenario:
                c = c[
                    ## Condition 1 (Selected Convergence FEN variables)
                    (
                        (
                            c.index.get_level_values("VARIABLE").str.startswith(
                                "Final Energy"
                            )
                        )
                        & (
                            c.index.get_level_values("VARIABLE").str.endswith(
                                var_list_fen[0]
                            )
                        )
                    )  ## FEN variables selection
                    ## Condition 2 (Selected Convergence for Other variables)
                    | (
                        ~c.index.get_level_values("VARIABLE").str.startswith(
                            "Final Energy"
                        )
                    )
                    & (
                        c.index.get_level_values("VARIABLE").str.endswith(var_list[0])
                    )  ## Primary (and other) variables selection
                ]  # .index.get_level_values('VARIABLE').unique()

                ## Change the variable name (without convergence info, as there is only one pathway now)
                index0 = c.index.names
                setindex(c, False)
                c["VARIABLE"] = c["VARIABLE"].str[
                    :-10
                ]  ## e.g. Removing 2300_BLEND (last 10 characters)
                setindex(c, index0)

                ## STEP 6 Save dataframe with CCS and CO2 info as a CSV file
            c["VALUE"].to_csv(
                downscaled_co2_file_name,
                mode=mode,
                header=header_flag,
            )
            df_return = c["VALUE"]
            header_flag = False  ## We set this equal to false right after we save CSV for the first time
            mode = "a"
            ## Step 8.1 reading NGFS data
            df_iam_all = pd.read_csv(
                CONSTANTS.INPUT_DATA_DIR / project_name / file,
                sep=",",
                encoding="latin-1",
            )
            df_iam_all.columns = map(
                str.upper, df_iam_all.columns
            )  ## Upper case in all columns!
            df_iam_all.loc[:, "REGION"] = df_iam_all.loc[:, "REGION"] + "r"

            ## STEP 8.2 reading price information
            price_var = [
                x
                for x in df_iam_all.VARIABLE.unique()
                if type(x) is str
                if "Price" in x
            ]
            if len(df_iam_all[df_iam_all.VARIABLE.isin(price_var)]) > 0:
                df_price = df_iam_all[df_iam_all.VARIABLE.isin(price_var)].copy(
                    deep=True
                )

                ## Step 8.3 Columns list
                col_time_list = time_horizon  # range(2010,2105,5)
                col_time_list = [str(c) for c in col_time_list]
                col_list = [
                    "MODEL",
                    "SCENARIO",
                    "REGION",
                    "VARIABLE",
                    "UNIT",
                ] + col_time_list

                ## Step 8.4 Appending the dataframe for each country in this region/target (containing the regional information)
                for c in countrylist:
                    df_price_country = df_price[
                        (df_price.SCENARIO == target)
                        & (df_price.REGION == fun_region_name(model, region))
                    ].copy(
                        deep=True
                    )  # .REGION.unique()

                    df_price_country.loc[:, "REGION"] = c
                    setindex(
                        df_price_country,
                        ["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"],
                    )
                    df_price_country[col_time_list].to_csv(
                        RESULTS_DATA_DIR / (model + file_suffix + ".csv"),
                        mode=mode,
                        header=header_flag,
                    )
            else:
                print("No price info in df_iam_all - we do not downscale prices")
                df_price = pd.DataFrame()
    else:
        print(f"{scenario} not found in step2 data. We skip this target")
        df_return = pd.DataFrame
    return header_flag, mode, df_return


def fun_harmonize_gdp_with_reg_iam_results_and_save_to_csv(
    file_suffix,
    RESULTS_DATA_DIR,
    input_file,
    country_mapping_file,
    pyam_mapping_file,
    selection_dict,
):
    _mode = "w"
    _header = True
    cols = [str(x) for x in range(2010, 2105, 5)]
    for model in selection_dict:
        path_df_socioeconomic = RESULTS_DATA_DIR / f"GDP{file_suffix}_updated_gdp.csv"

        df_countries = fun_read_df_countries(country_mapping_file)
        df_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file
        )
        df_iam = pd.read_csv(input_file, sep=",", encoding="utf-8")
        df_iam.columns = [x.upper() for x in df_iam.columns]
        df_iam.loc[:, "REGION"] = [
            x.replace(f"{model}|", "") + "r" for x in df_iam.REGION
        ]

        reg_dict = df_countries.set_index("ISO")[
            "REGION"
        ].to_dict()  ## Region dictionary

        gdp_unit_iam = df_iam[df_iam.VARIABLE == "GDP|PPP"].UNIT.unique()[0]

        df_socioeconomic_all = pd.read_csv(
            path_df_socioeconomic,
            index_col=["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"],
        )
        df_socioeconomic_all["REGION"] = [
            reg_dict.get(x, "na")
            for x in df_socioeconomic_all.index.get_level_values("ISO")
        ]

        ## change unit as reported by IAMs
        curr_units=df_socioeconomic_all.xs('GDP|PPP', level='VARIABLE').reset_index()['UNIT'].unique()
        if len(curr_units)!=1:
            txt="Unable to automatically detect unit `df_socioeconomic_all`, we found"
            raise ValueError(f'{txt}:{curr_units}')
        curr_unit=curr_units[0]
        df_socioeconomic_all = df_socioeconomic_all.rename(index={curr_unit: gdp_unit_iam})  

        regions = selection_dict[model]["regions"]

        regions = [r + "r" for r in regions]

        for region in regions:
            ## IAM results below
            countrylist = unique(
                df_countries[
                    df_countries.REGION == region.replace(model + "|", "")
                ].ISO.to_list()
            )

            df_socioeconomic = df_socioeconomic_all[
                df_socioeconomic_all.index.get_level_values("ISO").isin(countrylist)
            ]

            ## IAM data below
            num = (
                df_iam[
                    (df_iam.VARIABLE.isin(["Population", "GDP|PPP"]))
                    & (df_iam.MODEL == model)
                    # & (df_iam.REGION == region)
                    & (df_iam.REGION == region.replace(model + "|", ""))
                ]
                .groupby(["MODEL", "SCENARIO", "VARIABLE", "REGION", "UNIT"])
                .sum()
            )

            ## SSP data below
            den = df_socioeconomic.groupby(
                ["MODEL", "SCENARIO", "VARIABLE", "REGION", "UNIT"]
            ).sum()

            ## Regional Adjustment factor below:

            ratio = num / den

            ## 2022_04_04 to replicate what happens:
            # df_socioeconomic.mul(ratio).dropna(how='all')
            # idx= df_socioeconomic.mul(ratio).dropna(how='all').index
            # sel_idx=[x  for x in idx if x[4] in df_countries[df_countries.REGION==x[5]].ISO.unique()]
            # df_socioeconomic=df_socioeconomic.loc[sel_idx]

            ## Check if differences in GDP|PPP can across the two datasets (SSP vs IAM results) can be explained by a fixed conversion factor
            min_ratio = (
                (ratio)
                .xs("GDP|PPP", level="VARIABLE")
                .dropna(axis=0, how="all")
                .dropna(axis=1)
                .min()
                .min()
            )
            max_ratio = (
                (ratio)
                .xs("GDP|PPP", level="VARIABLE")
                .dropna(axis=0, how="all")
                .dropna(axis=1)
                .max()
                .max()
            )
            if (max_ratio - min_ratio) > 0.02:  ## maximum variation allowed over time
                print("#######")
                print(
                    f"Warning: the ratio of GDP|PPP from IAM data divided by SSP data is not constant over time: max variation = {max_ratio - min_ratio}."
                    f"This suggests that GDP|PPP data are not consistent (the difference across the two datasets cannot be explained by a fixed conversion factor over time)"
                )
                print("#######")

                ## Below we replace socioeconic data with harmonised data (to match regional IAM results)
                # df_socioeconomic = df_socioeconomic * ratio.dropna(how="all")
                # # df_socioeconomic.drop("REGION", axis=1).to_csv(path_df_socioeconomic)

                ## 2022_04_04 to replicate what happens:
            ratio_all = pd.DataFrame()
            ratio["ISO"] = "ISO"

            for c in countrylist:
                ratio.loc[:, "ISO"] = c
                #ratio_all = ratio_all.append(ratio)
                ratio_all = pd.concat([ratio_all, ratio])

            if len(ratio_all) > 0:
                df_socioeconomic_mul = df_socioeconomic.reset_index().set_index(
                    ["MODEL", "SCENARIO", "REGION", "ISO", "VARIABLE", "UNIT"]
                ) * ratio_all.reset_index().set_index(
                    ["MODEL", "SCENARIO", "REGION", "ISO", "VARIABLE", "UNIT"]
                )

                # df_socioeconomic_mul = df_socioeconomic.reset_index().set_index(ratio.index.names)[cols]*ratio[cols]

                # df_socioeconomic_mul = df_socioeconomic.mul(ratio).dropna(how="all")
                df_socioeconomic = df_socioeconomic_mul

                # Here we save harmonized GDP and Population
                df_socioeconomic.reset_index().drop(
                    ["REGION", "2005"], axis=1
                ).set_index(["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"]).dropna(
                    how="all"
                ).to_csv(
                    str(path_df_socioeconomic).replace(".csv", "_harmo.csv"),
                    mode=_mode,
                    header=_header,  ## needs to be always in 'a' mode here otherwise we will lose other ('previously downscaled') GDP results
                )
                _mode = "a"
                _header = False
            else:
                print("For", model, region, "we have empty data", len(ratio_all))
    print("#####################")
    print("GDP downscaling done")
    print("#####################")


def fun_downscale_gdp(
    project_name,
    file_suffix,
    _scen_dict,
    RESULTS_DATA_DIR,
    input_file,
    df_iam_all_models,
    model,
    gdp_model,
    pop_model,
    file_name,
    s0,
    timehorizon,
    header_flag,
    mode,
    targets,
    df_countries,
    regions,
    default_ssp_scenario,
    ref_scen="h_cpol",
    ssp_model="OECD Env-Growth",  # NOTE better as fixtures
):
    if (CONSTANTS.INPUT_DATA_DIR / project_name / "SSP_projections.csv").is_file():
        ssp_file = Path(project_name) / "SSP_projections.csv"
        df_ssp = pd.read_csv(CONSTANTS.INPUT_DATA_DIR / ssp_file, sep=",", encoding="utf-8")
        ssp_model = get_ssp_model(model, ssp_model, df_ssp)
    elif os.path.exists(
        CONSTANTS.INPUT_DATA_DIR / project_name / "GDP_NGFS_merged.csv"
    ):
        ssp_file = Path(project_name) / "GDP_NGFS_merged.csv"
    else:
        ssp_file = Path(CONSTANTS.INPUT_DATA_DIR) / "SspDb_country_data_2013-06-12.csv"

    df_ssp = pd.read_csv(CONSTANTS.INPUT_DATA_DIR / ssp_file, sep=",", encoding="utf-8")
    df_iam = pd.read_csv(input_file, sep=",", encoding="utf-8")
    df_iam.columns = [x.upper() for x in df_iam.columns]
    ## we just write the region name e.g. => MESSAGE|North America, => North America
    df_iam.loc[:, "REGION"] = [x.rsplit(f"{model}|")[0] + "r" for x in df_iam.REGION]

    ## Add REGION column to df_ssp
    setindex(df_countries, "ISO")
    reg_dict = df_countries["REGION"].to_dict()
    setindex(df_countries, "COUNTRY_NAME")
    df_ssp["REGION"] = [reg_dict.get(x, "") for x in df_ssp.ISO]

    df_ssp_all = (
        pd.DataFrame()
    )  ## Create a new df_ssp_all with multiple targets for each row
    # WE need if statement below, otherwise NGFS does not work
    if ssp_model in df_ssp.MODEL.unique():
        df_ssp = df_ssp[df_ssp.MODEL == ssp_model]
    df_ssp_init = df_ssp.copy(deep=True)

    # NOTE: This target loop (around 18 lines) is's a refactoring that was never implemented (consider this for the future)
    # for target in targets:
    #     if target in df_ssp_init.SCENARIO.unique():
    #         df_ssp = df_ssp_init[df_ssp_init.SCENARIO == target]
    #     else:
    #         ssp_scenario = fun_get_ssp(
    #             target, default_ssp=default_ssp_scenario, _scen_dict=_scen_dict
    #         )
    #         ## Filter dataframe based on selected ssp_scenario
    #         df_ssp = df_ssp_init[df_ssp_init["SCENARIO"].str.contains(ssp_scenario)]
    #         if len(df_ssp) == 0:
    #             raise ValueError(
    #                 f"cannot find {ssp_scenario} nor {target} in df_ssp.SCENARIO: {df_ssp_init.SCENARIO.unique()}"
    #             )
    #     df_ssp = df_ssp[df_ssp.MODEL == ssp_model]
    #     df_ssp["MODEL"] = model  ## rename MODEL column
    #     df_ssp["SCENARIO"] = target
    #     df_ssp_all = df_ssp_all.append(df_ssp)
    # time_col = [str(x) for x in range(2005, 2105, 5)]

    ## TODO - Blueprint

    ## Create VARIABLE loop for POPULATION, GDP VARIABLES
    ## Use groupby to perform operations (harmonise variable) across all targets, regions (for the current model) for the selected variable

    ## Check that the results do not change

    # NOTE: old implementation below
    ## STEP 3 REGION LOOP
    regions = [r + "r" for r in regions]
    for region in regions:
        print(region)
        countrylist = unique(
            df_countries[
                df_countries.REGION == region.replace(model + "|", "")
            ].ISO.to_list()
        )

        for scenario in targets:
            # This if/else block could be moved inside the `fun_ssp_downscaling` function (which reads again df_ssp)
            if scenario in df_ssp_init.SCENARIO.unique():
                ssp_scenario = scenario
            else:
                ssp_scenario = fun_get_ssp(
                    scenario, default_ssp=default_ssp_scenario, _scen_dict=_scen_dict
                )

                if (
                    len(df_ssp_init[df_ssp_init["SCENARIO"].str.contains(ssp_scenario)])
                    == 0
                ):
                    raise ValueError(
                        f"cannot find {ssp_scenario} nor {scenario} in df_ssp.SCENARIO: {df_ssp_init.SCENARIO.unique()}"
                    )

            if len(countrylist) >= 1:
                df_gdp_down = fun_ssp_downscaling(
                    model,
                    region,
                    scenario,  # we need scenario instead of ssp_scenario to run NGFS
                    "GDP|PPP",
                    countrylist,
                    gdp_model,
                    df_iam_all_models,
                    ssp_scenario=ssp_scenario,
                    # ssp_file="SspDb_country_data_2013-06-12.csv"
                    ssp_file=ssp_file,
                    ref_scen=ref_scen,
                )

                df_gdp_down[timehorizon].to_csv(
                    RESULTS_DATA_DIR / ("GDP" + file_suffix + file_name),
                    mode=mode,
                    header=header_flag,
                )

                mode = "a"
                header_flag = False  ## We set this equal to false right after we save CSV for the first time

                ## Downscale Population (by keeping same proportions) and save as CSV file
                ## Updated 2021_02_18
                df_pop_down = fun_ssp_downscaling(
                    model,
                    region,
                    scenario,  # we need scenario instead of ssp_scenario to run NGFS,
                    "Population",
                    countrylist,
                    pop_model,
                    df_iam_all_models,
                    ssp_scenario=ssp_scenario,
                    # ssp_file="SspDb_country_data_2013-06-12.csv"
                    ssp_file=ssp_file,
                    ref_scen=ref_scen,
                )

                df_pop_down[timehorizon].to_csv(
                    RESULTS_DATA_DIR / ("GDP" + file_suffix + file_name),
                    mode=mode,
                    header=header_flag,
                )

        print("Elapsed:", round(-s0 + time.time(), 2), "Seconds")

def get_ssp_model(model, ssp_model, df_ssp):
    if ssp_model not in df_ssp.MODEL.unique():
        if model in df_ssp.MODEL.unique():
            ssp_model = model
        else:
            raise ValueError(
                    f"cannot find {ssp_model} nor {model} in df_ssp.MODEL: {df_ssp.MODEL.unique()}"
                )
            
    return ssp_model


if __name__ == "__main__":
    downscaler.USE_CACHING = False
    project_name = "NGFS_2022"
    # model_patterns = ["*REMIND*", "*MESSAGE*"]
    model_patterns = ["*"]
    region_patterns = ["*"]
    target_patterns = "*"

    # main(
    #     project_name=project_name,
    #     model_patterns=model_patterns,
    #     region_patterns=region_patterns,
    #     target_patterns=target_patterns,
    #     gdp_pop_down=True,
    #     add_model_in_region_name=False,
    #     file_suffix="_NGFS_November",  #'__round_1p4_enlong_reg_lenght.csv'## CSV in
    #     file="snapshot_all_regions.csv",
    #     single_pathway=True,  ## We show only one pathway for each scenario
    #     add_gdp_pop_data=True,  ## from 2010
    #     harmonize_gdp_pop_with_reg_iam_results=False,  ## This will change the GDP unit from USD 2005 (as reported by SSP data) to USD 2010 (as reported by IAMs)
    # )

    try:
        selection_dict = get_selection_dict(
            InputFile(
                CONSTANTS.INPUT_DATA_DIR / project_name / "snapshot_all_regions.csv"
            ),
            model=model_patterns,
            region=region_patterns,
            target=target_patterns,
            variable=["*"],
        )
    except ValueError as e:
        raise ValueError(f"{e}\nConsider using the asterisk '*' for the pattern.")

    fun_validation(
        CONSTANTS,
        RESULTS_DATA_DIR,
        selection_dict,
        project_name=project_name,
        # csv_str="MODEL_NGFS22_March_prova5_all",
        # csv_str="GCAM 5.3+ NGFS_PROVA_REG",
        csv_str="GDP_NGFS_2022_March_first_round_v2_unit_usd2010_updated_gdp_harmo",
        model_patterns=model_patterns,
        region_patterns=region_patterns,
        target_patterns=target_patterns,
        vars=[
            # "Primary Energy|Coal|w/ CCS",
            # "Primary Energy|Coal|w/o CCS",
            # "Primary Energy|Gas|w/ CCS",
            # "Primary Energy|Gas|w/o CCS",
            # "Primary Energy|Coal",
            # "Primary Energy|Oil",
            # "Primary Energy|Gas",
            # # "Primary Energy|Biomass",
            # "Emissions|CO2|Energy",
            # "Carbon Sequestration|CCS|Biomass",
            "Population",
            "GDP|PPP",  ## unit is differet billion US$2010/yr vs billion US$2005/yr.
        ],
        cols=[str(x) for x in range(2020, 2055, 5)],
    )
