import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd  # Load the Pandas enlong_calc(df_main) ## calculating enlonglibraries with alias 'pd'

from downscaler import CONSTANTS
from downscaler.input_caching import InputFile, get_selection_dict
from downscaler.fixtures import ngfs_2023_nomenclature, step2_var, step3_var
from downscaler.utils import (  # harmo_ratio, add_region_simple, 
    InputFile,
    convert_to_list,
    fun_convert_time,
    fun_iamc_melt,
    fun_iamc_unmelt,
    fun_read_df_countries,
    fun_read_reg_value,
    fun_validation,
    load_model_mapping,
    setindex,
    unique,
    fun_read_hist_kyoto_data,
    fun_rename_index_name,
    fun_add_non_biomass_ren_nomenclature,
    fun_index_names,
)


def fun_co2_emissions_energy(
    model: str,
    target: str,
    region: str,
    df_iam_all_models: pd.DataFrame,
    var: str,
    df_all: pd.DataFrame,
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

    em = (
        df_all.iloc[:, df_all.columns.isin(wo_ccs_cols)].sum(axis=1)
        + df_all[wccs_cols].fillna(0).sum(axis=1) * fossil_ccs_leakage
    )
    df_all[f"Emissions|CO2|Energy{var}"] = em

    ## Harmoninising Emissions from FOSSILS:
    ratio = (
        fun_read_reg_value(
            model, region, target, "Emissions|CO2|Energy", df_iam_all_models
        )["VALUE"]
        + fun_read_reg_value(
            model,
            region,
            target,
            "Carbon Sequestration|CCS|Biomass",
            df_iam_all_models,
        )["VALUE"].fillna(0)
    ) / df_all[f"Emissions|CO2|Energy{var}"].groupby("TIME").sum()
    df_all[f"Emissions|CO2|Energy{var}"] = df_all[f"Emissions|CO2|Energy{var}"] * ratio

    ## Calculating 'Emissions|CO2|Energy' as difference between: Fossil Emissions - BECCS captured:
    # (Both fossil emissions and BECCS have been harmonised to match IAM results)
    df_all[f"Emissions|CO2|Energy{var}"] = df_all[
        f"Emissions|CO2|Energy{var}"
    ] - df_all[f"Primary Energy Emissions|Biomass|w/ CCS{var}"].fillna(0)
    return df_all

    ## function 1 - Calculating gap to meet policy


def fun_emi_gap(
    countrylist_w_policies: list,
    var: str,
    _start_policy: int,
    _df_all: pd.DataFrame,
    emi_target: pd.DataFrame,
) -> pd.DataFrame:  ## beccs weight in the next function
    """
    This function calculates the Emissions gap based on emi_target (read csv file with emission targets across counrties).
    It return the updated dataframe, with a new columns named 'GAP', which is the (emissions) gap to be filled.
    Other new columns (that could be deleted): TARGET, TARGET_VALUE, LINEAR

    Impact of policy depends on date of when the policy start to go into effect (_start_policy) until the year of  _end_policy (contained in the emi_target dataframe).

    For example, the impact of this policy increases from 1 (target-) in 2030 (or _start_policy) to 0 in 2050 (or _end_policy).

    Note:
    'TARGET_VALUE'= Target Value in a given (static) time (e.g in 2050 )
    'TARGET'= Dynamic target  based on linear interpolation (from _start_policy until _end_policy)


    THIS COULD BE ALSO USED FOR CALCULATING ENERGY POLICIES SHORT TERM GAP, BY:
    1) CHANGING: target_variable
    2) by using _start_policy =2020
    """

    target_variable = f"Emissions|CO2|Energy{var}"
    #     beccs_weight=0.5

    ## Step 1: Dynamic Target equal to target variable (emissions)
    _df_all.loc[:, "TARGET"] = _df_all.loc[:, target_variable].copy()

    ## Step 2:a) Reading target emissions value  (e.g. target= 0) emissions and
    #         b) TIME (e.g. by 2050) - both of them are country specific
    #         c) alculating LINEAR interpolation for each country
    for c in countrylist_w_policies:
        ## 2a) Reading in targeted value of emissions (in a given time)
        _df_all.loc[c, "TARGET_VALUE"] = emi_target.loc[c, ["VALUE"]][0]

        ## 2b) Reading in year of (e.g. net zero) policy
        _end_policy = str(emi_target.loc[c]["TIME"].copy())

        ## 2c) Calculating Linear interpolation based on TIME (step 2b)
        _beta = 1 / (
            float(_start_policy) - float(_end_policy)
        )  ## beta (slope) of linear interpolation
        _alpha = -_beta * float(_end_policy)  ## Alpha (constant) of linear regression

        _df_all.loc[:, "LINEAR"] = (
            _beta * _df_all.index.get_level_values("TIME").astype(float) + _alpha
        )
        #         except:
        #             _df_all.loc[:,'LINEAR']= _beta*_df_all.index.get_level_values('TIME').tolist()+_alpha
        _df_all.loc[:, "LINEAR"] = _df_all.loc[:, "LINEAR"].clip(
            0, 1
        )  ## Clipped linear interpolation from 0 to 1

        ## Step 3 We compute TARGET emissions by using this function: MAX (target_value,  Emissions- TARGET *(1-linear)  )

        _df_all.loc[:, "TARGET"] = _df_all.loc[:, target_variable] * (
            _df_all.loc[:, "LINEAR"]
        ) + _df_all.loc[:, "TARGET_VALUE"] * (1 - _df_all.loc[:, "LINEAR"])

    ## COMPUTING GAP (note, if emissions are already below TARGET, THEN  GAP=0 => no need to modify ambition level )
    _df_all.loc[:, "GAP"] = _df_all.loc[countrylist_w_policies, target_variable].fillna(
        0
    ) - _df_all.loc[countrylist_w_policies, "TARGET"].fillna(0)

    _df_all.loc[:, "GAP"] = _df_all.loc[:, "GAP"].clip(
        0, np.inf
    )  ## Only positive (otherwise we might decrease ambition)
    return _df_all


def fun_fill_gap(
    model: str,
    region: str,
    target: str,
    countrylist: list,
    _df_all: pd.DataFrame,
    var_iam_mod: str,
    gap: str,
    var: str,
    weight: float,
    df_iam_all_models: pd.DataFrame,
    _lower_bound: float = -np.inf,
    _upper_bound: float = np.inf,
) -> pd.DataFrame:
    """
    Search for a specified 'gap' column in the dataframe (_df_all), and try to fill part of it (percentage => weight) by increasing a specified variable (var_iam_mod)

    This could be also used for short-term energy policies (e.g. 2030) - or can be easily adapted

    """
    _df_all, _orig_type = fun_convert_time(
        _df_all, _type=int
    )  ## Covert TIME from STRING to FLOAT

    ## Step1 Increasing var_iam_mod to fill a percentage (weight) of the gap - e.g. increasing BECCS to fill 50% of emi gap, in all time periods
    _df_all.loc[:, var_iam_mod + var] = _df_all.loc[
        :, var_iam_mod + var
    ] + weight * _df_all.loc[:, gap].fillna(0)

    ## Step 2 Introducing upper/lower bounds on BECCS
    ## 2a)This is valid only for BECCS: maximum 100% of BECCS, by assumimg an emission factor => 20 Mt C/EJ - added 2021_03_04 h 11.16
    if var_iam_mod == "Carbon Sequestration|CCS|Biomass":
        _df_all.loc[:, var_iam_mod + var] = np.minimum(
            _df_all.loc[:, var_iam_mod + var],
            _df_all.loc[:, "Primary Energy|Biomass" + var] * 20 * 44 / 12,
        )
    ## 2b) General upper/lower bounds using .clip()
    _df_all.loc[:, var_iam_mod + var] = _df_all.loc[:, var_iam_mod + var].clip(
        _lower_bound, _upper_bound
    )

    ## Step3 Harmonising BECCS to match IAM results
    try:
        sel=_df_all.loc[_df_all.index.get_level_values(0).isin(countrylist),var_iam_mod + var]
        ratio = sel/sel.groupby("TIME").sum()
        # ratio = _df_all.loc[countrylist, var_iam_mod + var] / _df_all.loc[
        #     countrylist, var_iam_mod + var
        # ].groupby("TIME").sum(axis=0)
    except:
        sel=_df_all.loc[_df_all.index.get_level_values(0).isin(countrylist),var_iam_mod + var]
        ratio = ratio = sel * 0  
        ## 2021_04_06 Zero if sum equal to zero

    ## Line below updated 2021_03_30
    _df_all.loc[
        _df_all.index.get_level_values("ISO").isin(countrylist), var_iam_mod + var
    ] = (
        ratio
        * fun_read_reg_value(model, region, target, var_iam_mod, df_iam_all_models)[
            "VALUE"
        ]
    )

    _df_all, _orig_type = fun_convert_time(
        _df_all, _type=_orig_type
    )  ## Convert TIME to original type
    return _df_all


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

    var_iam_list = create_var_iam_list(var_iam)
    fuel_list = ["Oil", "Biomass", "Coal", "Gas"]

    for f in fuel_list:
        for var_iam in var_iam_list:
            idx = get_emi_factor_index(var_iam)
            apply_emission_factors(
                model, var_iam, var, df_all, df_emi_factors, idx, from_c_to_co2
            )
            harmonise_emissions(
                model, region, target, var_iam, var, df_all, df_iam_all_models
            )

    harmonise_fossil_ccs(model, region, target, var, df_all, df_iam_all_models)
    return df_all


def create_var_iam_list(var_iam: str) -> list:
    if var_iam.find("w/o") != -1:
        var_iam2 = var_iam.replace("w/o", "w/")
    else:
        var_iam2 = var_iam.replace("w/", "w/o")
    return [var_iam, var_iam2]


def get_emi_factor_index(var_iam: str) -> str:
    n = 1
    groups = var_iam.split("|")
    return ("|".join(groups[:n])[1], "|".join(groups[n:]))[1]


def apply_emission_factors(
    model: str,
    var_iam: str,
    var: str,
    df_all: pd.DataFrame,
    df_emi_factors: pd.DataFrame,
    idx: str,
    from_c_to_co2: float,
):
    if var_iam.find("Biomass") != -1 and var_iam.find("w/ CCS") != -1:
        df_all[
            var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
        ] = (
            df_all[var_iam.replace("|w/ CCS", "") + var]
            * df_emi_factors.loc[idx, model]
            * from_c_to_co2
        )
    elif var_iam.find("Biomass") != -1:
        df_all[
            var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
        ] = (
            df_all[var_iam.replace("|w/o CCS", "") + var]
            * df_emi_factors.loc[idx, model]
            * from_c_to_co2
        )
    else:
        try:
            df_all[
                var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
            ] = (
                df_all[var_iam + var]
                * df_emi_factors.loc[idx, model]
                * from_c_to_co2
            )
        except:
            df_all[
                var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
            ] = 0
            print(var_iam, "not working")


def harmonise_emissions(
    model: str,
    region: str,
    target: str,
    var_iam: str,
    var: str,
    df_all: pd.DataFrame,
    df_iam_all_models: pd.DataFrame,
):
    if var_iam.find("Biomass") != -1 and var_iam.find("w/ CCS") != -1:
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
            try:
                ratio = df_all[
                    var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
                ] / df_all[
                    var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
                ].groupby("TIME").sum()
            except:
                ratio = df_all[
                    var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
                ] / (
                    1e-5
                    + df_all[
                        var_iam.replace("Primary Energy", "Primary Energy Emissions")
                        + var
                    ]
                    .groupby("TIME")
                    .sum()
                )

            df_all[
                var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
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
                var_iam.replace("Primary Energy", "Primary Energy Emissions") + var
            ]


def harmonise_fossil_ccs(
    model: str,
    region: str,
    target: str,
    var: str,
    df_all: pd.DataFrame,
    df_iam_all_models: pd.DataFrame,
):
    try:
        fossil_list = [
            "Primary Energy Emissions|Coal|w/ CCS",
            "Primary Energy Emissions|Gas|w/ CCS",
            "Primary Energy Emissions|Oil|w/ CCS",
        ]
        fossil_list = [f + var for f in fossil_list]
        try:
            ratio = (
                fun_read_reg_value(
                    model,
                    region,
                    target,
                    "Carbon Sequestration|CCS|Fossil",
                    df_iam_all_models,
                )["VALUE"]
                / df_all[fossil_list].sum(axis=1).groupby("TIME").sum()
            )
        except:
            ratio = 0 * df_all[fossil_list].sum(axis=1)
        for fos in fossil_list:
            df_all[fos] = df_all[fos] * ratio
        print("harmonised FOSSI CCS Captured!")
        df_all["Carbon Sequestration|CCS|Fossil" + var] = df_all[fossil_list].sum(
            axis=1
        )
    except:
        pass


def fun_fill_gap_primary_fossils(
    df_all: pd.DataFrame,
    scen: str,
    var: str,
    model: str,
    region: str,
    df_iam_all_models: pd.DataFrame,
    weight=0.5,
) -> pd.DataFrame:
    """
    This function updates Primary energy and Secondary energy by fuels
    to meet the EMISSION GAP (in line with long term policies).
    """

    _df_all = df_all.copy(deep=True)

    var_iam_mod = [
        "Primary Energy|Coal|w/o CCS",
        "Primary Energy|Gas|w/o CCS",
        "Primary Energy|Oil|w/o CCS",
        "Primary Energy|Coal|w/ CCS",
        "Primary Energy|Gas|w/ CCS",
        "Primary Energy|Oil|w/ CCS",
    ]

    var_iam_compensating = [
        "Primary Energy|Biomass",
        "Primary Energy|Non-Biomass Renewables|Geothermal",
        "Primary Energy|Non-Biomass Renewables|Hydro",
        "Primary Energy|Non-Biomass Renewables|Solar",
        "Primary Energy|Non-Biomass Renewables|Wind",
        "Primary Energy|Geothermal",
        "Primary Energy|Hydro",
        "Primary Energy|Solar",
        "Primary Energy|Wind",
    ]

    target = scen

    gap = "GAP"

    fossil = [i + var for i in var_iam_mod]  ## List of fossil fuels variables + var
    ## List of fossil fuels variables + var
    ren = [i + var for i in var_iam_compensating]

    print("THIS IS FOSSIL:", fossil)
    _df_all, _orig_type = fun_convert_time(
        _df_all, _type=int
    )  ## Covert TIME from STRING to FLOAT

    ## Step1 Converting GAP from MTCO2 into EJ
    ## EMI FACTOR =>  Emissions from fossils (excluding emissions captured by BECCS) / divided BY EJ from fossils
    try:
        _df_all.loc[:, "FOSSIL_EMI_FACTOR" + var] = (
            _df_all.loc[:, "Emissions|CO2|Energy" + var]
            + _df_all.loc[:, "Carbon Sequestration|CCS|Biomass" + var].fillna(0)
        ) / _df_all.loc[:, fossil].sum(axis=1)
    except:
        ## Updated 2021_10_19 (1e-10 instead of 0)
        try:
            _df_all.loc[:, "FOSSIL_EMI_FACTOR" + var] = np.maximum(
                1e-10,
                _df_all.loc[:, "Emissions|CO2|Energy" + var]
                + _df_all.loc[:, "Carbon Sequestration|CCS|Biomass" + var].fillna(0),
            ) / (1e-5 + _df_all.loc[:, fossil].sum(axis=1))
        except:
            _fossil_tot = [
                "Primary Energy|Coal",
                "Primary Energy|Gas",
                "Primary Energy|Oil",
            ]
            _df_all.loc[:, "FOSSIL_EMI_FACTOR" + var] = np.maximum(
                1e-10,
                _df_all.loc[:, "Emissions|CO2|Energy" + var]
                + _df_all.loc[:, "Carbon Sequestration|CCS|Biomass" + var].fillna(0),
            ) / (1e-5 + _df_all.loc[:, _fossil_tot].sum(axis=1))

    ## gap converted from MtCO2 to EJ
    # TO BE MULTIPLIED BY WEIGHT?
    _df_all.loc[:, "GAP_EJ"] = weight * (
        _df_all.loc[:, gap] / _df_all.loc[:, "FOSSIL_EMI_FACTOR" + var]
    )

    try:
        _df_all.loc[:, "RATIO"] = (
            1
            - _df_all.loc[:, "GAP_EJ"]
            / np.maximum(
                1e-5, +_df_all.iloc[:, _df_all.columns.isin(fossil)].sum(axis=1)
            )
        ).fillna(1)
    except:
        _df_all.loc[:, "RATIO"] = 1

    ## using lambda or loop across all columns
    # loop below is usually skipped
    for col in fossil:
        ## 2022_01_31
        if col == "Primary Energy|Oil|w/o CCS":
            print("Now working on oil:", col)
        try:
            _df_all.loc[:, col] = _df_all.loc[:, col] * _df_all.loc[:, "RATIO"]
            if _df_all.loc[:, col].min() < 0:  ## added 2021_03_31 h 16.23
                print(
                    col,
                    "is negative!!!! we Clip values and do a regional harmonisation",
                )
                _df_all.loc[:, col] = _df_all.loc[:, col].clip(
                    1e-10, np.inf
                )  ## We Clip value >0
                _df_all.loc[:, col] / (
                    _df_all.loc[:, col].groupby("TIME").sum()
                ).fillna(0)
                try:
                    _df_all.loc[:, "RATIO_HARMO"] = _df_all.loc[:, col] / (
                        _df_all.loc[:, col].groupby("TIME").sum()
                    ).fillna(0)
                except:
                    _df_all.loc[:, "RATIO_HARMO"] = 0 * _df_all.loc[:, col]

                reg_value = fun_read_reg_value(
                    model,
                    region,
                    target,
                    col,  ## if range of projections need to include var
                    df_iam_all_models,
                )["VALUE"]
                _df_all[col] = _df_all.loc[:, "RATIO_HARMO"] * reg_value

        ## then harmonization
        except:
            print(col, "Not working")

    print("ren:", ren)
    ## Step 3 Increasing ALL renewables by the same percentage (ratio) to meet the same energy demand

    try:  ## This one does not always work
        _df_all.loc[:, "RATIO"] = (
            (
                1
                + _df_all.loc[:, "GAP_EJ"]
                / _df_all.iloc[:, _df_all.columns.isin(ren)].fillna(0).sum(axis=1)
            ).fillna(1)
        ).replace(np.inf, "1")
    except:  ## In case of division by zero error: we distinguish countries with /without renewables
        mask1 = (
            _df_all.iloc[:, _df_all.columns.isin(ren)].sum(axis=1) == 0
        )  ## Countries/time with zero renewables
        mask2 = (
            _df_all.iloc[:, _df_all.columns.isin(ren)].sum(axis=1) != 0
        )  ## Countries/time with positive renewables
        _df_all.loc[mask1, "RATIO"] = 1
        _df_all.loc[mask2, "RATIO"] = (
            1
            + _df_all.loc[mask2, "GAP_EJ"]
            / _df_all.loc[mask2]
            .iloc[:, _df_all.loc[mask2].columns.isin(ren)]
            .fillna(0)
            .sum(axis=1)
        ).fillna(1)

    for col in ren:  #  to be updated by using lambda instead of loop across all columns
        try:
            _df_all.loc[:, col] = _df_all.loc[:, col] * _df_all.loc[:, "RATIO"]
        except:
            print(col, "Not working Step 3")

    ## Step 4.1 updating total Coal, Oil and Gas (as the sum of with and without CCS)
    for f in ["Coal", "Oil", "Gas"]:
        try:  ## Total coal = coal/wo ccs
            _df_all.loc[:, "Primary Energy|" + f + var] = _df_all.loc[
                :, "Primary Energy|" + f + "|w/o CCS" + var
            ]

        except:
            pass

        try:  ## Adding w/ ccs (we do this in 2 steps as sometimes w/ccs info is missing)
            _df_all.loc[:, "Primary Energy|" + f + var] = _df_all.loc[
                :, "Primary Energy|" + f + "|w/o CCS" + var
            ] + _df_all.loc[:, "Primary Energy|" + f + "|w/ CCS" + var].fillna(0)
        except:
            print("Primary Energy|" + f + "|w/o CCS" + var, " not working")
            pass

    return _df_all


def fun_secondary_update(
    _df_all: pd.DataFrame,
    _df_orig: pd.DataFrame,
    scen: str,
    model: str,
    region: str,
    df_iam_all_models: pd.DataFrame,
) -> pd.DataFrame:
    var = ""
    # Step5 Propagate adjustments made (at Primary energy levels) to Secondary and final energy levels.
    ## NOTE:  No need to re-harmonise afterwards - this approach guarantees consistency with regional results

    ## Created Dictionaries below to access secondary and primary energy (total by fuel, e.g coal), by using the same name convention (e.g. coal)
    _df_all, _orig_type = fun_convert_time(
        _df_all, _type=float
    )  ## Convert TIME to original type, to use the same index as df_all
    _df_orig, _orig_type = fun_convert_time(_df_orig, _type=float)

    col_dict_secondary = {
        "Coal": [
            "Secondary Energy|Electricity|Coal",
            "Secondary Energy|Gases|Coal",
            "Secondary Energy|Liquids|Coal",
            "Secondary Energy|Solids|Coal",
        ],
        "Gas": [
            "Secondary Energy|Electricity|Gas",
            "Secondary Energy|Gases|Gas",
            "Secondary Energy|Liquids|Gas",
            "Secondary Energy|Solids|Gas",
            "Secondary Energy|Electricity|Natural Gas",
            "Secondary Energy|Gases|Natural Gas",
            "Secondary Energy|Liquids|Natural Gas",
            "Secondary Energy|Solids|Natural Gas",
        ],
        "Oil": [
            "Secondary Energy|Electricity|Oil",
            "Secondary Energy|Gases|Oil",
            "Secondary Energy|Liquids|Oil",
            "Secondary Energy|Solids|Oil",
        ],
        "Biomass": [
            "Secondary Energy|Electricity|Biomass",
            "Secondary Energy|Gases|Biomass",
            "Secondary Energy|Liquids|Biomass",
            "Secondary Energy|Solids|Biomass",
        ],
        "Geothermal": [
            "Secondary Energy|Electricity|Geothermal",
            "Secondary Energy|Gases|Geothermal",
            "Secondary Energy|Liquids|Geothermal",
            "Secondary Energy|Solids|Geothermal",
        ],
        "Wind": [
            "Secondary Energy|Electricity|Wind",
            "Secondary Energy|Gases|Wind",
            "Secondary Energy|Liquids|Wind",
            "Secondary Energy|Solids|Wind",
        ],
        "Solar": [
            "Secondary Energy|Electricity|Solar",
            "Secondary Energy|Gases|Solar",
            "Secondary Energy|Liquids|Solar",
            "Secondary Energy|Solids|Solar",
        ],
        "Hydro": [
            "Secondary Energy|Electricity|Hydro",
            "Secondary Energy|Gases|Hydro",
            "Secondary Energy|Liquids|Hydro",
            "Secondary Energy|Solids|Hydro",
        ],
        "Nuclear": ["Secondary Energy|Electricity|Nuclear"],
    }

    col_dict_primary = {
        "Coal": [
            "Primary Energy|Coal|w/ CCS",
            "Primary Energy|Coal|w/o CCS",
        ],
        "Gas": [
            "Primary Energy|Gas|w/ CCS",
            "Primary Energy|Gas|w/o CCS",
        ],
        "Oil": [
            "Primary Energy|Oil|w/ CCS",
            "Primary Energy|Oil|w/o CCS",
        ],
        "Biomass": [
            "Primary Energy|Biomass|w/ CCS",
            "Primary Energy|Biomass|w/o CCS",
        ],
        "Geothermal": ["Primary Energy|Geothermal"],
        "Wind": ["Primary Energy|Wind"],
        "Solar": ["Primary Energy|Solar"],
        "Hydro": ["Primary Energy|Hydro"],
        "Nuclear": ["Primary Energy|Nuclear"],
    }

    var_iam_mod = [
        "Primary Energy|Coal|w/o CCS",
        "Primary Energy|Gas|w/o CCS",
        "Primary Energy|Oil|w/o CCS",
        "Primary Energy|Coal|w/ CCS",
        "Primary Energy|Gas|w/ CCS",
        "Primary Energy|Oil|w/ CCS",
    ]

    var_iam_compensating = [
        "Primary Energy|Biomass",
        "Primary Energy|Non-Biomass Renewables|Geothermal",
        "Primary Energy|Non-Biomass Renewables|Hydro",
        "Primary Energy|Non-Biomass Renewables|Solar",
        "Primary Energy|Non-Biomass Renewables|Wind",
        "Primary Energy|Geothermal",
        "Primary Energy|Hydro",
        "Primary Energy|Solar",
        "Primary Energy|Wind",
    ]

    target = scen
    fossil = [i + var for i in var_iam_mod]  ## List of fossil fuels variables + var
    ren = [
        i + var for i in var_iam_compensating
    ]  ## List of fossil fuels variables + var

    ## Calculating adjustemts made in all columns (both fossils and renewables)
    f = ren + fossil

    ## Step 5a)  Calculating Delta Primary (NOTE: no change in total biomass, we only change proportions between CCS and wo/CCS)
    ## We compute the Difference compared to the original dataframe
    delta_primary = (
        _df_all.iloc[:, (_df_all.columns.isin(f))]
        - _df_orig.iloc[:, (_df_orig.columns.isin(f))]
    )

    #     ## Continue step 5- propagating delta primary to Secondary enrgy
    for f in [
        "Coal",
        "Gas",
        "Oil",
        "Wind",
        "Solar",
        "Hydro",
        "Geothermal",
        "Nuclear",
        "Biomass",
    ]:

        columns_secondary = [i + var for i in col_dict_secondary[f]]  ## aggiungiamo var
        columns_primary = [i + var for i in col_dict_primary[f]]

        #     print(columns_secondary, columns_primary)

        ## Step 5b.1) allocating difference to secondary energy level

        ## Step 5b.1.1) Calculatint REGIONAL Ratio (primary_to_secondary) between total Primary and Secondary (AT THE REGIONAL LEVEL) for total fuel (e.g. coal)
        _df_orig["Total_secondary"] = (
            _df_orig.iloc[:, (_df_orig.columns.isin(columns_secondary))]
            .fillna(0)
            .sum(axis=1)
        )  ## e.g. total secondary coal
        _df_orig["Total_primary"] = (
            _df_orig.iloc[:, _df_orig.columns.isin(columns_primary)]
            .fillna(0)
            .sum(axis=1)
        )  ## e.g. total primary coal

        #### Ratio below (at the regional level)=> why regional? we use regional ratio to keep consistency with regional IAM results (no need to re-harmonise => we propagate delta_primary to secondary level (for all countries) based on regional conversion rates)
        primary_to_secondary = np.maximum(
            1e-5, _df_orig["Total_primary"].groupby("TIME").sum()
        ) / np.maximum(1e-5, _df_orig["Total_secondary"].groupby("TIME").sum())
        print("================")
        print("Primary:", _df_orig["Total_primary"].groupby("TIME").sum())
        print("Secondary:", _df_orig["Total_secondary"].groupby("TIME").sum())
        print("primary_to_secondary", primary_to_secondary)

        ## Step 5b.1.2)  Calculating REGIONAL Ratio (ec_to_secondary) between Energy carrier (e.g. electricity) within total secondary for a given fuel (e.g. coal)

        for ec in ["Liquids", "Solids", "Gases", "Electricity"]:
            try:
                cols_list_sel = _df_orig.iloc[
                    :, (_df_orig.columns.isin(columns_secondary))
                ].columns.tolist()
                sel_col = [i for i in cols_list_sel if i.find(ec) >= 0][0]
                try:  # we select the variable associated to ec, and we calculate ratio: ec_to_secondary (within a given fuel)
                    ec_to_secondary = (
                        np.maximum(1e-5, _df_orig[sel_col].groupby("TIME").sum())
                    ) / np.maximum(
                        1e-5, _df_orig["Total_secondary"].groupby("TIME").sum()
                    )
                except:
                    ec_to_secondary = 0

                ## Step 4 Allocate the Primary energy delta (across countries), to ec and fuels, baseed on the two ratios above
                try:
                    updated_value = _df_orig[sel_col] + (
                        delta_primary.iloc[
                            :, delta_primary.columns.isin(columns_primary)
                        ]
                        .sum(axis=1)
                        .fillna(0)
                        / primary_to_secondary.fillna(0)
                        / ec_to_secondary.fillna(0)
                    ).fillna(0)
                    # _df_all[sel_col] = np.maximum(0, updated_value)
                    # FIX Secondary Energy|Electricity|Gas in GBR in 2030: workaround 1): maximum 50% reduction compared to original results without policies
                    _df_all[sel_col] = np.maximum(_df_orig[sel_col] / 2, updated_value)
                except:
                    updated_value = _df_orig[sel_col] + (
                        delta_primary.iloc[
                            :, delta_primary.columns.isin(columns_primary)
                        ]
                        .sum(axis=1)
                        .fillna(0)
                        / primary_to_secondary
                        / ec_to_secondary
                    ).fillna(0)
                    # _df_all[sel_col] = np.maximum(0, updated_value)
                    # FIX Secondary Energy|Electricity|Gas in GBR in 2030: workaround 1): maximum 50% reduction compared to original results without policies
                    _df_all[sel_col] = np.maximum(_df_orig[sel_col] / 2, updated_value)

                if updated_value.min() < 0:
                    print(ec, f, "negative values found")
                else:
                    print(ec, f, "works fine")

                if updated_value.min() < 0:
                    _df_all.loc[:, "RATIO"] = _df_all.loc[:, sel_col] / np.maximum(
                        1e-5, _df_all.loc[:, sel_col].groupby("TIME").sum()
                    ).fillna(0)
                    #                 print (ec,f, 'negative values found', _df_all.loc['ARG','RATIO'])

                    reg_value = fun_read_reg_value(
                        model,
                        region,
                        target,
                        sel_col,  ## if range of projections need to include var
                        df_iam_all_models,
                    )["VALUE"]
                    #                 print('reg_value',ec,f,reg_value)
                    _df_all[sel_col] = _df_all.loc[:, "RATIO"] * reg_value

            #                     print('AFTER HARMO',ec,f,_df_all[sel_col].loc[('ARG',model,target,2050)])
            except:
                pass

    return _df_all


def fun_emi_target_short(
    model: str,
    _scen: str,
    df_ghg_all_time: pd.DataFrame,
    emi_target_short: pd.DataFrame,
    co2_energy_only: bool,  ## If True Applies target only to co2_energy
    # input_data_file="input_data/indc_plus_emi_targets.csv",
    type_target: str = "Unconditional",
    ## temporarily set to TRUE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    incl_lulucf: bool = False,
) -> pd.DataFrame:

    """This function reads the INDC targets at the country level (% reduction) and calculate the absolute target based on the user's choice:
    - apply target to co2 energy only (True/False)
    - Type of target (e.g. "Unconditional")
    - incl_lulucf (True/False)

    It returns the updated DataFrame with the targets in absolute value.
    """

    ## Below we calculate reference target VALUE for short-term policies in 4 steps:
    gwp_dict = {
        "Mt CO2eq/yr": 1,
        "Mt CO2/yr": 1,
        "Mt CH4/yr": 25,
        "Mt N2O/yr": 298 / 1e3,
    }

    ## Based on Jonas's email 29_03_2021. However we include Emissions CO2 energy to get total GHG at the base year
    ## Here (2030 TARGETS) We assume targets include AGRICULTURE emissions BUT NOT AFOLU OR LULUCF
    var_sel = [
        "Emissions|CH4|Agriculture",
        "Emissions|CH4|Energy",
        "Emissions|CH4|Other",
        "Emissions|CO2|Industrial Processes",
        "Emissions|F-Gases|Other",
        "Emissions|N2O|Agriculture",
        "Emissions|N2O|Energy",
        "Emissions|N2O|Other",
    ]

    if incl_lulucf == True:
        var_sel = var_sel + [
            "Emissions|CH4|LULUCF",
            "Emissions|CO2|LULUCF",
            "Emissions|N2O|LULUCF",
        ]

    var_ghg_sel = var_sel + [
        "Emissions|CO2|Energy"
    ]  ##  GHG emission (we need this to calculate historical GHG at reference year)

    emi_target_short["REF_YEAR_GHG"] = "missing"
    emi_target_short["VALUE"] = "missing"

    ## REF_YEAR loop
    for i in emi_target_short.REF_YEAR.unique().tolist():

        ## HISTORICAL GHG EMISSIONS IN REF_YEAR
        # ghg_ref_year = (
        #     df_ghg_all_time.loc[
        #         ("Reference", "PRIMAP__Historic|country_reported__" + model)
        #     ]
        #     .unstack()[str(i)]
        #     .fillna(0)
        # )
        # NOTE: historical data do not change across models. Just using MESSAGE here
        ghg_ref_year = (
            df_ghg_all_time.loc[
                (
                    "Reference",
                    "PRIMAP__Historic|country_reported__MESSAGEix-GLOBIOM 1.0",
                )
            ]
            .unstack()[str(i)]
            .fillna(0)
        )

        if co2_energy_only:
            ghg_ref_year = ghg_ref_year.xs(
                "Emissions|CO2|Energy", level="VARIABLE", drop_level=False
            )

        ## Step 1 Calculating GHG emissions in reference year
        ## Conversion to MTCO2 based on GWP, and  sum across all selected variables => we get total GHG emissions in reference year
        ref_year_ghg = (
            (
                ghg_ref_year[
                    ghg_ref_year.index.get_level_values("VARIABLE").isin(var_ghg_sel)
                ]
                * [gwp_dict.get(i, 0) for i in ghg_ref_year.columns]
            )
            .sum(axis=1)  ## Conversion to MTCO2 based on GWP
            .groupby("ISO")
            .sum()
        )  ## Sum across all selected GHG variables (var_ghg_sel)

        ## Step 2 Copy GHG values in ref_year in the emi_target_short dataframe
        setindex(emi_target_short, "ISO")
        c_index = emi_target_short[
            emi_target_short.REF_YEAR == i
        ].index  ## Countries with this ref_year target
        emi_target_short.loc[
            emi_target_short.REF_YEAR == i, "REF_YEAR_GHG"
        ] = ref_year_ghg.loc[ref_year_ghg.index.isin(c_index)]

        ## STEP 3 Calculating VALUE (GHG emission target) in TIME (2030) based on reference year (e.g. 2010)
        emi_target_short.loc[
            emi_target_short.REF_YEAR == i, "VALUE"
        ] = emi_target_short.loc[emi_target_short.REF_YEAR == i, "REF_YEAR_GHG"] * (
            1 + emi_target_short.loc[emi_target_short.REF_YEAR == i, type_target]
        )

    ## STEP 4 Calculating target for Energy related emission in 2030 (TIME): We subtract non-CO2 and non-energy CO2 (var_sel) from GHG emissions
    # This is only necessary if the target is for GHG emission. If it refers to CO2 only (co2_energy_only==True), then we don't need to subtract
    if co2_energy_only == False:
        for (
            i
        ) in emi_target_short.TIME.unique().tolist():  ## this is usually 2030 (i=2030)

            ## 4a)Projected GHG emissions in TIME (year of short-term policies)
            ghg_projected = df_ghg_all_time.loc[(model)].unstack()[str(i)].fillna(0)

            ## 4b) Calculating  non-co2 and non-energy CO2 in 2030 (i)
            projected_ghg_2030 = (
                (
                    ghg_projected[
                        (
                            ghg_projected.index.get_level_values("VARIABLE").isin(
                                var_sel
                            )
                        )  ## Variable selection
                        & (
                            ghg_projected.index.get_level_values("SCENARIO") == _scen
                        )  ## Scenario selection
                    ]
                    * [
                        gwp_dict.get(i, 0) for i in ghg_projected.columns
                    ]  ## GWP conversion
                )
                .sum(axis=1)  ## Conversion to MTCO2 based on GWP
                .groupby("ISO")
                .sum()
            )  ## Sum across non-CO2 and non-energy CO2 emission variables (var_sel)

            ## 4c) We subtract non-co2 and non-energy co2 from  GHG emission in 2030 (i)
            emi_target_short.loc[emi_target_short.TIME == i, "VALUE"] = (
                emi_target_short.loc[emi_target_short.TIME == i, "VALUE"]
                - projected_ghg_2030
            )

    if emi_target_short["VALUE"].min() < 0:
        print("Careful, negative emission targets in 2030!!!!!!!!!!!!!!")

    emi_target_short["MODEL"] = model
    emi_target_short["SCENARIO"] = _scen

    return emi_target_short


def fun_short(
    model: str,
    selection_dict: dict,
    policy_scen_short: list,
    df_ghg_all_time: pd.DataFrame,
    df_emi_factors: pd.DataFrame,
    co2_energy_only: bool,
    df_iam_all_models: pd.DataFrame,
    pyam_mapping_file: InputFile,  # new
    model_patterns: list,  # new
    region_patterns: list,  # new
    target_patterns: list,  # new
    csv_out: str,
    RESULTS_DATA_DIR,
    input_emi_target_short: pd.DataFrame,  # 2nd march
    df_all_scen: pd.DataFrame,
    df_countries: pd.DataFrame(),
    save_csv_emission_target_files: bool = True,
    start_policy: int = 2025,
    # csv_out=csv_out, # removed
    countries_w_2030_emi_policies: list = [
        "AUT",
        "BEL",
        "BGR",
        "CYP",
        "CZE",
        "DEU",
        "DNK",
        "ESP",
        "EST",
        "FIN",
        "FRA",
        "GRC",
        "HRV",
        "HUN",
        "IRL",
        "ITA",
        "LTU",
        "LUX",
        "LVA",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "SWE",
        "JPN",
        "KOR",
        "CAN",
        "ZAF",
        "GBR",
        "USA",
    ],
    var: str = "",
    weight=0.5,
) -> pd.DataFrame:

    df_all_iamc_appended = pd.DataFrame()

    sel_scen = [i for i in selection_dict[model]["targets"] if i in policy_scen_short]

    for scen in sel_scen:
        # for scen in policy_scen_short:
        print("==============")
        print(scen)
        print("==============")

        # {c:hist.loc[c,t]*(1+input_emi_target_short.set_index('ISO').loc[c,"Unconditional"]) for c,t in ref_year_dict.items()}
        # hist values from PRIMAP:
        # hist=fun_rename_index_name(fun_read_hist_kyoto_data(False, 2018)[0][[1990, 2010]], {'REGION':'ISO'})
        if df_ghg_all_time is None or len(df_ghg_all_time) == 0:
            emi_target = fun_short_term_co2_energy_targets(
                co2_energy_only, input_emi_target_short, model, scen
            )
            if not co2_energy_only:
                emi_target = fun_subtract_nonco2_from_energy_targets(
                    df_all_scen, scen, emi_target, "2030"
                )
        else:
            emi_target = fun_emi_target_short(
                model, scen, df_ghg_all_time, input_emi_target_short, co2_energy_only
            )
        # NOTE about `emi_target`:
        # •	REF_YEAR_GHG: is the 1990 or 2010 emissions (e.g. 2010 for the USA)
        # •	VALUE: is the target value for CO2 emissions (in 2030)

        ## CAREFUL!!!! THE BELOW SHOULD CONTAIN ALL SHORT TERM  POLICIES NOT LONG TERM
        regions_w_policies = (
            df_countries[df_countries.ISO.isin(countries_w_2030_emi_policies)]
            .REGION.unique()
            .tolist()
        )
        regions_wo_policies = (
            df_countries[~df_countries.ISO.isin(countries_w_2030_emi_policies)]
            .REGION.unique()
            .tolist()
        )

        try:  ## remove nan region
            regions_wo_policies.remove(np.nan)
        except:
            pass

        sel_region = [
            i.replace(model + "|", "") + "r"
            for i in selection_dict[model]["regions"]
            if i.replace(model + "|", "") + "r" in regions_w_policies
        ]
        sel_region = unique(sel_region)
        if len(sel_region) == 0:
            print(
                f"We found no regions that require short-term adjustments. Please check your selected `region_patterns`: {region_patterns}"
            )
            raise ValueError(
                f"Policies not implemented for model_patterns {model_patterns}, region_patterns {region_patterns}, and target_patterns {target_patterns}"
            )

        for region in sel_region:
            print("==============")
            print(scen, region)
            print("==============")
            countrylist = df_countries[df_countries.REGION == region].ISO.unique()

            ## 2021_12_20 only for non-native regions
            if len(countrylist) > 1:  ## Exclude native regions
                ## 2021_12_20 below:
                df_all = fun_iamc_unmelt(
                    df_all_scen[
                        (
                            df_all_scen.index.get_level_values(1) == scen
                        )  ## Selecting 1 scenario (otherwise fun_iamc_unmelt not working)
                        & (df_all_scen.index.get_level_values("ISO").isin(countrylist))
                    ]  # .drop_duplicates()
                )  ## Selecting region (based on countrylist)

                df_all_orig = df_all.copy(deep=True)

                emi_target = emi_target[
                    (
                        emi_target.index.isin(countries_w_2030_emi_policies)
                    )  ## COUNTRY with policies SELECTION
                    & (emi_target.index.isin(countrylist))  ## Countries within region
                    & (emi_target.MODEL == model)  ## model selection
                    & (emi_target.SCENARIO == scen)  ## Scenario selection
                ]

                ## Creating List of countries with policies (probably no need for this anylonger..)
                countrylist_w_policies = list(
                    set(countrylist).intersection(emi_target.index.tolist())
                )

                ## Calculating emissions gap (based on target) and make linear interpolation (based on time of policy)
                fun_emi_gap(
                    countrylist_w_policies, var, start_policy, df_all, emi_target
                )
                # print(df_all.loc['ZAF'].droplevel('MODEL').iloc[:,-4:])
                # NOTE: so far coincides with Jupyter notebook 2022_01_28

                ## Try to fill 50% of emissions gap by increasing 'Carbon Sequestration|CCS|Biomass' (and re-harmonise to match IAM results)
                if "Carbon Sequestration|CCS|Biomass" in df_all.columns:
                    fun_fill_gap(
                        model,
                        region,
                        scen,
                        countrylist,
                        df_all,
                        "Carbon Sequestration|CCS|Biomass",
                        "GAP",
                        var,
                        weight,
                        df_iam_all_models,
                    )
                else:
                    # If BECCS is missing, we double the weight for reducing fossil
                    weight = weight * 2
                mydf = pd.DataFrame(
                    df_all.xs(
                        ("2030", model, scen), level=("TIME", "MODEL", "SCENARIO")
                    )[["TARGET", "GAP", "Emissions|CO2|Energy"]]
                )

                print(mydf[mydf.GAP > 0])

                if save_csv_emission_target_files:
                    (RESULTS_DATA_DIR / "countries_adj_2030").mkdir(exist_ok=True)
                    if len(mydf[mydf.GAP > 0]):
                        mydf[mydf.GAP > 0].to_csv(
                            RESULTS_DATA_DIR
                            / "countries_adj_2030"
                            / (f"{model}{csv_out}_{region}_{scen}.csv")
                        )

                # NOTE: so far coincides with Jupyter notebook 2022_01_28
                # Set 'Primary Energy Emissions|Biomass|w/ CCS' = 'Carbon Sequestration|CCS|Biomass'
                if "Carbon Sequestration|CCS|Biomass" in df_all.columns:
                    df_all[f"Primary Energy Emissions|Biomass|w/ CCS{var}"] = df_all[
                        f"Carbon Sequestration|CCS|Biomass{var}"
                    ]
                    ## Re-Calculating emissions from energy (with updated beccs)
                    _beccs_before_adj = fun_iamc_unmelt(
                        df_all_scen[
                            (
                                df_all_scen.index.get_level_values(1) == scen
                            )  ## BECCS (before adjustments)
                            & (
                                df_all_scen.index.get_level_values("ISO").isin(
                                    countrylist
                                )
                            )
                        ]
                    )[f"Carbon Sequestration|CCS|Biomass{var}"]
                    _delta_beccs = (
                        _beccs_before_adj
                        - df_all[f"Carbon Sequestration|CCS|Biomass{var}"]
                    )
                    df_all[f"Emissions|CO2|Energy{var}"] = df_all[
                        f"Emissions|CO2|Energy{var}"
                    ] + _delta_beccs.fillna(0)
                else:
                    df_all[f"Primary Energy Emissions|Biomass|w/ CCS{var}"] = 0

                # NOTE: so far coincides with Jupyter notebook 2022_01_28
                ## Remaning gap (50% or more) to be filled by replacing fossils with renewables (wind, solar, hydro, geothermal, biomass), and re-harmonise to match IAM results
                df_all = fun_fill_gap_primary_fossils(
                    df_all, scen, "", model, region, df_iam_all_models, weight
                )
                # NOTE: so far coincides with Jupyter notebook 2022_01_28 (Emissions|CO2|Energy, CCS Sequestration for AUSTRIA, 1p5C)
                ## Calculating emissions by fuel (same share of regional_iam results) incl. 'Carbon Sequestration|CCS|Biomass', 'Carbon Sequestration|CCS|Fossil'
                fuel_list = ["Biomass", "Coal", "Gas", "Oil"]
                main_var = "Primary Energy|fuel|w/ CCS"
                for f in fuel_list:
                    var_iam = main_var.replace("fuel", f)
                    print(var_iam)
                    fun_co2_emissions_primary(
                        model,
                        scen,
                        region,
                        var_iam,
                        "",
                        df_all,
                        df_emi_factors,
                        df_iam_all_models,
                    )
                fun_co2_emissions_energy(
                    model,
                    scen,
                    region,
                    df_iam_all_models,
                    "",
                    df_all,
                    fossil_ccs_leakage=0.1,
                )
                ## The function above (fun_co2_emissions_energy) already harmonises emissions to match regional results
                ## NOTE so far Emissions|CO2|Energy and Primary Energy|Oil coincide with Jupyter notebook 2022_01_28
                ## Updating secondary energy variables:
                fun_secondary_update(
                    df_all, df_all_orig, scen, model, region, df_iam_all_models
                )
                ## NOTE 2022_01_18 df_all has the same lenght of df_all in jupyter notebook Secondary Energy|Electricity|Gas coincide with jupyeter notebook for GCAM AUT, o_1p5C
                ## cleaning up data:
                df_all = df_all.drop(
                    [
                        "TARGET",
                        "GAP",
                        "FOSSIL_EMI_FACTOR",
                        "GAP_EJ",
                        "RATIO",
                    ],
                    axis=1,
                )

                try:
                    df_all = df_all.drop(["TARGET_VALUE", "LINEAR"], axis=1)
                except:
                    pass

                try:
                    df_all = df_all.drop("RATIO_HARMO", axis=1)
                except:
                    pass

                df_all, _orig_type = fun_convert_time(df_all, _type=str)

                ## IAMC_FORMAT
                df_all_iamc = fun_iamc_melt(df_all)
                df_all_iamc["UNIT"] = "UNIT"
                ## Excluding Primary Energy|Fossil
                df_all_iamc = df_all_iamc[
                    ~df_all_iamc.index.get_level_values("VARIABLE").isin(
                        [
                            "Primary Energy|Fossil",
                            "Primary Energy|Fossil|w/ CCS",
                            "Primary Energy|Fossil|w/o CCS",
                        ]
                    )
                ]

                ## NOTE: df_all has the same columns as jupyter notebook 2022_01_28
                ## also len(df_all_iamc) == 984 , same as jupyter notebook (EU-15r GCAM)
                ## Save to CSV

                # df_all_iamc_appended = df_all_iamc_appended.append(df_all_iamc)
                df_all_iamc_appended = pd.concat([df_all_iamc_appended,df_all_iamc])

    return df_all_iamc_appended


def fun_subtract_nonco2_from_energy_targets(df_all_scen, scen, emi_target_short, t):
    nonco2 = df_all_scen.xs(
        "Emissions|Kyoto Gases (incl. indirect AFOLU)", level="VARIABLE"
    ) - df_all_scen.xs("Emissions|CO2", level="VARIABLE")
    nonco2 = nonco2[t].xs(scen, level="SCENARIO").droplevel("MODEL")

    # CO2 needs to reduce more to compensate for the non-co2 emissions
    emi_target_short.loc[:, "VALUE"] = emi_target_short.loc[:, "VALUE"] - nonco2
    return emi_target_short


def fun_short_term_co2_energy_targets(
    co2_energy_only: bool, input_emi_target_short, model, scen
):
    if co2_energy_only:
        refvar = "Emissions|CO2|Energy"
    else:
        refvar = "Emissions|Kyoto Gases (incl. indirect AFOLU)"
    hist = fun_rename_index_name(
        fun_read_hist_kyoto_data(False, 2018)[0][[1990, 2010]],
        {"REGION": "ISO"},
    )
    hist = hist.xs(refvar, level="VARIABLE").droplevel(["MODEL", "SCENARIO", "UNIT"])
    ref_year_dict = input_emi_target_short[
        input_emi_target_short.ISO.isin(hist.index)
    ].set_index("ISO")["REF_YEAR"]

    hist_dict = {c: hist.loc[c, t] for c, t in ref_year_dict.items()}
    red_dict = {
        c: (1 + input_emi_target_short.set_index("ISO").loc[c, "Unconditional"])
        for c in hist_dict
    }
    target_short = {c: hist_dict[c] * (red_dict[c]) for c in hist_dict}

    emi_target_short = pd.concat(
        [
            input_emi_target_short.set_index("ISO"),
            pd.DataFrame([target_short], index=["VALUE"]).T,
        ],
        axis=1,
    ).dropna(axis=0, how='all')

    # emi_target_short=pd.concat([emi_target_short, pd.DataFrame([ref_year_dict], index=['REF_YEAR']).T], axis=1).dropna(axis=0)
    emi_target_short["MODEL"] = model
    emi_target_short["SCENARIO"] = scen
    return emi_target_short
    #     ## NOTE: 2022_01_28_'Primary Energy|Oil|w/o CCS' is missing, AND THIS IS OK.
    #     ## 'Carbon Sequestration|CCS|Biomass' IS ACTUALLY OK. IT seems that SHORT_ONLY csv file is modified at a later point in time.
    #     ## ALL RESULTS SEEM TO BE OK. (coincide with Jupyter notebooj). But do not coincide with GCAM5.3_NGFS__NGFS_November_step4_SHORT_only from Laptop (not sure why)
    #     ## To be investigated
    #     df_all_iamc.to_csv('results/4_Policy_Adjustments/'+model+csv_out+'_step4_SHORT_only'+'.csv', mode='a', header=False)


def fun_long(
    model: str,
    df_nonco2: pd.DataFrame,
    policy_scen_long: list,
    selection_dict: dict,  ## new
    df_all_scen: pd.DataFrame,  # new
    df_countries: pd.DataFrame,  # new
    df_emi_factors: pd.DataFrame,  # new
    df_iam_all_models: pd.DataFrame,
    RESULTS_DATA_DIR,  # new!
    csv_out: str,
    save_csv_emission_target_files: bool = True,
    start_policy: int = 2030,
    countries_w_2050_emi_policies: list = [
        "AUT",
        "BEL",
        "BGR",
        "CYP",
        "CZE",
        "DEU",
        "DNK",
        "ESP",
        "EST",
        "FIN",
        "FRA",
        "GRC",
        "HRV",
        "HUN",
        "IRL",
        "ITA",
        "LTU",
        "LUX",
        "LVA",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "SWE",
        "JPN",
        "KOR",
        "CAN",
        "ZAF",
        "GBR",
        "USA",
    ],
    var: str = "",
    weight_long=0.5,
) -> pd.DataFrame:

    ## POLICIES IN LINE WITH NGFS PROTOCOL 2021_04_08 (USED EX-POST ONLY FOR LONG TERM POLICIES)
    df_all_iamc_appended = pd.DataFrame()

    ## Adding time (2050 dimension) to df_nonco2 (which contains nonco2+LULUCF data from Matt )
    setindex(df_nonco2, "ISO")  # ['VALUE']

    print("LONG term policies:", model, policy_scen_long)
    setindex(df_all_scen, ["MODEL", "SCENARIO", "ISO", "VARIABLE"])
    if "UNIT" in df_all_scen.columns:
        df_all_scen.drop("UNIT", axis=1, inplace=True)

    sel_scen = [i for i in selection_dict[model]["targets"] if i in policy_scen_long]

    for scen in sel_scen:
        regions_w_policies = (
            df_countries[df_countries.ISO.isin(countries_w_2050_emi_policies)]
            .REGION.unique()
            .tolist()
        )
        regions_wo_policies = (
            df_countries[~df_countries.ISO.isin(countries_w_2050_emi_policies)]
            .REGION.unique()
            .tolist()
        )

        try:  ## remove nan region
            regions_wo_policies.remove(np.nan)
        except:
            pass

        ## 2022_02_02 way forward below:
        sel_region = [
            i.replace(model + "|", "") + "r"
            for i in selection_dict[model]["regions"]
            if i.replace(model + "|", "") + "r" in regions_w_policies
        ]
        sel_region = unique(sel_region)
        for region in sel_region:

            emi_target = df_nonco2[df_nonco2.index.isin(countries_w_2050_emi_policies)]

            print("==============")
            print(model, scen, region)
            print("==============")
            countrylist = df_countries[df_countries.REGION == region].ISO.unique()
            countrylist_w_policies = list(
                set(countrylist).intersection(emi_target.index.tolist())
            )

            if (
                len(countrylist_w_policies) == 0
                or len(df_countries[df_countries.REGION == region]) <= 1
            ):
                if len(df_countries[df_countries.REGION == region]) <= 1:
                    print("Region contains one country we skip this region:", region)
                else:
                    print(
                        f"In {region} we found no countries with policies - we skip long term policies adjustments"
                    )
                df_all = df_all_scen[
                    (
                        df_all_scen.index.get_level_values(1) == scen
                    )  ## Selecting 1 scenario (otherwise fun_iamc_unmelt not working)
                    & (df_all_scen.index.get_level_values("ISO").isin(countrylist))
                ]
                setindex(df_all, False)
                df_all_iamc = df_all[
                    [
                        "VARIABLE",
                        "MODEL",
                        "SCENARIO",
                        "ISO",
                        "2010",
                        "2015",
                        "2020",
                        "2025",
                        "2030",
                        "2035",
                        "2040",
                        "2045",
                        "2050",
                        "2055",
                        "2060",
                        "2065",
                        "2070",
                        "2075",
                        "2080",
                        "2085",
                        "2090",
                        "2095",
                        "2100",
                    ]
                ]

            else:  ## Region contains more than one country, we adjust scenario with policies
                ## Select region, scenarios and unmelt (df in wide format)
                df_all = fun_iamc_unmelt(
                    df_all_scen[
                        (
                            df_all_scen.index.get_level_values(1) == scen
                        )  ## Selecting 1 scenario (otherwise fun_iamc_unmelt not working)
                        & (df_all_scen.index.get_level_values("ISO").isin(countrylist))
                    ]
                )  ## Selecting region (based on countrylist)
                df_all_orig = df_all.copy(deep=True)

                emi_target = df_nonco2[
                    (
                        df_nonco2.index.isin(countries_w_2050_emi_policies)
                    )  ## COUNTRY with policies SELECTION
                    & (df_nonco2.index.isin(countrylist))  ## Countries within region
                    & (df_nonco2.MODEL == model)  ## model selection
                    & (df_nonco2.SCENARIO == scen)  ## Scenario selection
                ]

                ## Creating List of countries with policies (probably no need for this anylonger..)
                countrylist_w_policies = list(
                    set(countrylist).intersection(emi_target.index.tolist())
                )

                ## Calculating emissions gap (based on target) and make linear interpolation (based on time of policy)
                fun_emi_gap(
                    countrylist_w_policies, var, start_policy, df_all, emi_target
                )

                ## Try to fill 50% of emissions gap by increasing 'Carbon Sequestration|CCS|Biomass' (and re-harmonise to match IAM results)
                fun_fill_gap(
                    model,
                    region,
                    scen,
                    countrylist,
                    df_all,
                    "Carbon Sequestration|CCS|Biomass",
                    "GAP",
                    var,
                    weight_long,
                    df_iam_all_models,
                )

                # Set 'Primary Energy Emissions|Biomass|w/ CCS' = 'Carbon Sequestration|CCS|Biomass'
                df_all["Primary Energy Emissions|Biomass|w/ CCS" + var] = df_all[
                    "Carbon Sequestration|CCS|Biomass" + var
                ]

                ## NOTE: Countries have different timing for mid-century strategy. For most countries it's 2050.
                # Here we just save data for 2050 for all countries (emissions target and gap).

                # To check what is the exact timing of long term emissions target please use:
                # emi_target["TIME"].to_dict().

                # To save data for the exact emission target you would need to
                # 1) use a dictionary with the timing of long term emissions target for all countries (emi_target["TIME"].to_dict())
                # 2) filter mydf using the dictionary with the exact emission target for all countries, example from the code below (not working):
                # time_dict=emi_target["TIME"].to_dict() ## dictionary with timing of emissions
                # t_list = [
                #     .get(str(x), str(2050))
                #     for x in df_all.index.get_level_values("ISO")
                # ]  ## list of long term emissions targets (timing)
                # df_all[df_all.index.get_level_values("TIME").isin(t_list)]
                # mydf = pd.DataFrame(
                #     df_all[df_all.index.get_level_values("TIME").isin(t_list)].xs([model, scen], level=["MODEL", "SCENARIO"])[
                #         ["TARGET", "GAP", "Emissions|CO2|Energy"]
                #     ]
                # )

                mydf = pd.DataFrame(
                    df_all.xs(
                        ("2050", model, scen), level=("TIME", "MODEL", "SCENARIO")
                    )[["TARGET", "GAP", "Emissions|CO2|Energy"]]
                )

                print(mydf[mydf.GAP > 0])
                if save_csv_emission_target_files:
                    (RESULTS_DATA_DIR / "countries_adj_2050").mkdir(exist_ok=True)
                    if len(mydf[mydf.GAP > 0]):
                        mydf[mydf.GAP > 0].to_csv(
                            RESULTS_DATA_DIR
                            / "countries_adj_2050"
                            / (f"{model}{csv_out}_{region}_{scen}.csv")
                        )

                ## Re-Calculating emissions from energy (with updated beccs)
                _beccs_before_adj = fun_iamc_unmelt(
                    df_all_scen[
                        (
                            df_all_scen.index.get_level_values(1) == scen
                        )  ## BECCS (before adjustments)
                        & (df_all_scen.index.get_level_values("ISO").isin(countrylist))
                    ]
                )["Carbon Sequestration|CCS|Biomass" + var]
                _delta_beccs = (
                    _beccs_before_adj - df_all["Carbon Sequestration|CCS|Biomass" + var]
                )
                df_all["Emissions|CO2|Energy" + var] = df_all[
                    "Emissions|CO2|Energy" + var
                ] + _delta_beccs.fillna(
                    0
                )  # Updated emissions

                ## Recalculating the emissions gap (to be filled by replacing fossils with renewables)
                fun_emi_gap(
                    countrylist_w_policies, var, start_policy, df_all, emi_target
                )

                ## Remaning gap (50% or more) to be filled by replacing fossils with renewables (wind, solar, hydro, geothermal, biomass), and re-harmonise to match IAM results
                df_all = fun_fill_gap_primary_fossils(
                    df_all,
                    scen,
                    "",
                    model,
                    region,
                    df_iam_all_models,
                    weight=weight_long,
                )

                ## Calculating emissions by fuel (same share of regional_iam results) incl. 'Carbon Sequestration|CCS|Biomass', 'Carbon Sequestration|CCS|Fossil'
                fuel_list = ["Biomass", "Coal", "Gas", "Oil"]
                #             var_list=['2200_BLEND','2250_BLEND''2300_BLEND']
                main_var = "Primary Energy|fuel|w/ CCS"
                for f in fuel_list:
                    var_iam = main_var.replace("fuel", f)
                    print(var_iam)
                    fun_co2_emissions_primary(
                        model,
                        scen,
                        region,
                        var_iam,
                        "",
                        df_all,
                        df_emi_factors,
                        df_iam_all_models,
                    )
                fun_co2_emissions_energy(
                    model,
                    scen,
                    region,
                    df_iam_all_models,
                    "",
                    df_all,
                    fossil_ccs_leakage=0.1,
                )
                ## The function above (fun_co2_emissions_energy) already harmonises emissions to match regional results

                ## Updating secondary energy variables:
                fun_secondary_update(
                    df_all, df_all_orig, scen, model, region, df_iam_all_models
                )

                ## cleaning up data:
                df_all = df_all.drop(
                    [
                        "TARGET",
                        "GAP",
                        "FOSSIL_EMI_FACTOR",
                        "GAP_EJ",
                        "RATIO",
                    ],
                    axis=1,
                )

                try:
                    df_all = df_all.drop(["TARGET_VALUE", "LINEAR"], axis=1)
                except:
                    pass

                try:
                    df_all = df_all.drop("RATIO_HARMO", axis=1)
                except:
                    pass

                ## IAMC_FORMAT
                df_all_iamc = fun_iamc_melt(df_all)

                ## Excluding Primary Energy|Fossil
                df_all_iamc = df_all_iamc[
                    ~df_all_iamc.index.get_level_values("VARIABLE").isin(
                        [
                            "Primary Energy|Fossil",
                            "Primary Energy|Fossil|w/ CCS",
                            "Primary Energy|Fossil|w/o CCS",
                        ]
                    )
                ]
                df_all_iamc_appended = pd.concat([df_all_iamc_appended,df_all_iamc])
                # df_all_iamc_appended = df_all_iamc_appended.append(df_all_iamc)
    return df_all_iamc_appended


def fun_read_input_data(
    input_file,
    selection_dict,
    co2_energy_only,
    if_missing_model_scen_use_this: list = [],
):
    df_iam_all_models = pd.read_csv(input_file)
    df_iam_all_models.columns = [x.upper() for x in df_iam_all_models.columns]
    df_iam_all_models = pd.melt(
        df_iam_all_models,
        id_vars=["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"],
        value_vars=[str(x) for x in range(2005, 2105, 5)],
    )

    df_iam_all_models = df_iam_all_models[
        (~(df_iam_all_models.MODEL == "Reference"))
    ]  ### Get only Energy and emissions variables (2021_03_02)
    df_iam_all_models.loc[:, "REGION"] = (
        df_iam_all_models.loc[:, "REGION"] + "r"
    )  ## adjusting region name to avoid overlap with ISO codes
    df_iam_all_models.columns = [x.upper() for x in df_iam_all_models.columns]
    df_iam_all_models.columns = [
        "MODEL",
        "SCENARIO",
        "REGION",
        "VARIABLE",
        "UNIT",
        "TIME",
        "VALUE",
    ]
    df_iam_all_models.loc[:, "TIME"] = [int(x) for x in df_iam_all_models.TIME]

    df_countries = fun_read_df_countries(
        CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv",
    )
    if co2_energy_only:
        clist = df_countries.ISO.unique()
        models = list(selection_dict.keys())
        targetslist = []
        elements = [selection_dict[x]["targets"] for x in models]
        for tar in elements:
            targetslist = targetslist + tar
        targetslist = unique(targetslist)

        dflist = []
        for c in clist:
            for m in models:
                for tar in targetslist:
                    dflist = dflist + [(c, m, tar, 0)]

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
            print(model, i, j, type(emi_prova))
            emi_prova = emi_prova + [{model: j, "fuel": i}]  #'Oil|w/o CCS'
        df1 = pd.DataFrame(emi_prova)
        setindex(df1, "fuel")
        emi_factors_data = pd.concat([emi_factors_data, df1], axis=1)

    df_emi_factors = emi_factors_data

    ## Timing of net zero is 2050 unless specified in dictionary (timing_net_zero)
    # df_nonco2["TIME"] = [timing_net_zero.get(i, 2050) for i in df_nonco2.index]

    ## Timing of net zero is 2050 unless specified in dictionary (timing_net_zero)
    # df_nonco2["TIME"] = [timing_net_zero.get(i, 2050) for i in df_nonco2.index]
    ## NOTE: df_nonco2 it's only used in long term projections. This is correct as it refers to 2050 data.

    input_emi_target_short = pd.read_csv(
        CONSTANTS.INPUT_DATA_DIR / "indc_plus_emi_targets.csv"
    )

    return (
        input_emi_target_short,
        df_emi_factors,
        df_iam_all_models,
    )


RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR(
    str(Path(os.path.abspath("")) / Path(__file__).name)
)
RESULTS_DATA_DIR.mkdir(exist_ok=True)


def main(
    project_name="NGFS",
    region_patterns: Union[str, list] = "*",
    model_patterns: Union[str, list] = "*",
    target_patterns: Union[str, list] = "*",
    policy_scen_long=[
        f"{s}{i}"
        for s in ["o_1p5c", "o_lowdem", "d_delfrag", "d_rap", "d_strain"]
        for i in ["", "_d50", "_d95high", "_d95"]
    ]
    + [
        # ECEMF runs below
        'WP1 NetZero',
        # EU AB (mostly ENGAGE) runs below
        "Delayed transition",
        "Divergent Net Zero",
        "EN_NPi2020_450",
        "EN_INDCi2100_NDCp",
        "EN_NPi2020_800",
        "EN_INDCi2030_300f",
        "EN_INDCi2030_400f",
        "EN_INDCi2030_900",
        "EN_NPi2020_500",
        "EN_INDCi2030_600_COV",
        "EN_INDCi2030_600_COV_NDCp",
        "EN_NPi2020_400",
        "NP_2025_-902050",
    ],  ## List of Scenarios to be adjusted with Long policies
    policy_scen_short=[
        f"{s}{i}"
        for s in ["o_1p5c", "o_lowdem", "d_delfrag", "d_rap", "d_strain", 'h_cpol', 'h_ndc', 'o_2c']
        for i in ["", "_d50", "_d95high", "_d95"]
    ]+ [
        # ECEMF runs below
        'WP1 NPI', 
        'WP1 NetZero',
        # EU AB (mostly ENGAGE) runs below
        "Delayed transition",
        "Divergent Net Zero",
        "EN_NPi2020_450",
        "EN_INDCi2100_NDCp",
        "EN_NPi2020_800",
        "EN_INDCi2030_300f",
        "EN_INDCi2030_400f",
        "EN_INDCi2030_900",
        "EN_NPi2020_500",
        "EN_INDCi2030_600_COV",
        "EN_INDCi2030_600_COV_NDCp",
        "EN_NPi2020_400",
        "NP_2025_-902050",
    ],  ## List of Scenarios to be adjusted with policies
    csv_out: str = "Step_4",
    co2_energy_only: bool = False,
    countries_w_2030_emi_policies: list = [
        "AUT",
        "BEL",
        "BGR",
        "CYP",
        "CZE",
        "DEU",
        "DNK",
        "ESP",
        "EST",
        "FIN",
        "FRA",
        "GRC",
        "HRV",
        "HUN",
        "IRL",
        "ITA",
        "LTU",
        "LUX",
        "LVA",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "SWE",
        "JPN",
        "KOR",
        "CAN",
        "ZAF",
        "GBR",
        # "USA",
    ],
    countries_w_2050_emi_policies: list = [
        "AUT",
        "BEL",
        "BGR",
        "CYP",
        "CZE",
        "DEU",
        "DNK",
        "ESP",
        "EST",
        "FIN",
        "FRA",
        "GRC",
        "HRV",
        "HUN",
        "IRL",
        "ITA",
        "LTU",
        "LUX",
        "LVA",
        "MLT",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVK",
        "SVN",
        "SWE",
        "JPN",
        "KOR",
        "CAN",
        "ZAF",
        "GBR",
        # "USA",
    ],
    # ghg_data_from_ngfs_2021=["MESSAGEix-GLOBIOM 1.0", "o_1p5c"]
    ghg_data_from_ngfs_2021=[],
    timing_net_zero={
        "CHN": 2060,
        "KAZ": 2060,
        "SWE": 2045,
        "AUT": 2040,
        "ISL": 2040,
    },
    # Weight_short=> 50% is the split between Beccs vs Energy. And the other 50% is the share of the GAP to be filled (we don't want to fill 100% to avoid artifacts)
    weight_short=0.5 * 0.5,
    # Weight_long => 50% is the split between Beccs vs Energy. Here we want to fill 100% of the gap.
    weight_long=0.5 * 1,
    read_from_step5e: bool = False,
    harmonize_until:int=2018,
    # downscaler.USE_CACHING=True,
):

    input_file = CONSTANTS.INPUT_DATA_DIR / project_name / "snapshot_all_regions.csv"
    pyam_mapping_file = CONSTANTS.INPUT_DATA_DIR / project_name / "default_mapping.csv"

    region_patterns = convert_to_list(region_patterns)
    model_patterns = convert_to_list(model_patterns)
    target_patterns = convert_to_list(target_patterns)

    RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )
    RESULTS_DATA_DIR.mkdir(exist_ok=True)
    PREV_STEP_RES_DIR = CONSTANTS.PREV_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )

    iam_results_file = InputFile(
        CONSTANTS.INPUT_DATA_DIR / project_name / "snapshot_all_regions.csv"
    )
    pyam_mapping_file = InputFile(
        CONSTANTS.INPUT_DATA_DIR / project_name / "default_mapping.csv"
    )

    (CONSTANTS.CURR_RES_DIR(__file__) / project_name).mkdir(exist_ok=True)

    # get_selection_dict will produce an error
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

    ## Read input data
    (
        input_emi_target_short,  #
        # df_nonco2,  # NON-CO2 emissions in 2050, (Data Climate Analytics), also changes on a runs basis
        df_emi_factors,  # emission factors (CONTANTS)
        # df_ghg_all_time,  # non-co2 + LULUCF data (hist + projections) - combiedn ndc (dynamic)
        df_iam_all_models_input,  # regional data (dynamic)
    ) = fun_read_input_data(
        input_file,
        selection_dict,
        co2_energy_only,
        if_missing_model_scen_use_this=ghg_data_from_ngfs_2021,
    )
    # 'Mt CO2/yr','EJ/yr'
    df_iam_all_models_input = df_iam_all_models_input[
        df_iam_all_models_input.UNIT.isin(["Mt CO2/yr", "EJ/yr"])
    ]
    # Main loop
    for model in selection_dict.keys():
        targets_tobe_adj = [
            x
            for x in selection_dict[model]["targets"]
            if x in policy_scen_long + policy_scen_short
        ]
        if len(targets_tobe_adj) == 0:
            raise ValueError(
                f"We could not find any target that belongs to the list of `policy_scen_long` nor `policy_scen_short` for model {model}.\n"
                f"Your selected targets for the {model} model: {selection_dict[model]['targets']} "
            )
        ## NOTE:
        # emi_target: Dataframe with short term emissions targets
        # List of countries with 2030 (short term) targets: emi_target_short.ISO.unique()
        # df_countries[df_countries.ISO.isin(list(emi_target_short.ISO.unique()))]
        df_iam_all_models = df_iam_all_models_input[
            df_iam_all_models_input.MODEL == model
        ]
        df_countries = fun_read_df_countries(
            CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv",
        )
        df_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file.file
        )

        ## reading previously downscaled results
        if read_from_step5e:
            # From step5e
            excel_file = f"{model}_{project_name}{csv_out}_{str(harmonize_until)}_harmo_step5e_Scenario_Explorer_upload_FINAL.xlsx"
            df_all_scen = pd.read_excel(
                CONSTANTS.CURR_RES_DIR(str("step5")) / excel_file, engine="openpyxl"
            )
        else:
            # From step4
            df_all_scen = pd.read_csv(PREV_STEP_RES_DIR / f"{model}{csv_out}.csv")

        df_all_scen = df_all_scen.rename({"REGION": "ISO"}, axis=1)
        setindex(df_all_scen, ["MODEL", "SCENARIO", "ISO", "VARIABLE"])
        if "UNIT" in df_all_scen.columns:
            df_all_scen.drop("UNIT", axis=1, inplace=True)  ## NOTE: not sure if needed

        # Bring back original names e.g. "Carbon Sequestration|CCS|Biomass"
        df_all_scen = df_all_scen.rename(
            {v: k for v, k in ngfs_2023_nomenclature.items()}, level="VARIABLE"
        )
        df_all_scen = fun_add_non_biomass_ren_nomenclature(df_all_scen, inverse=False)[
            0
        ]

        ## Drop duplicated values, otherwise not working for 'GCAM 5.3+ NGFS' region='USAr'
        if len(df_all_scen[df_all_scen.index.duplicated()]) > 0:
            print(
                "*** Warning: we found duplicated values in step 3 that will be removed ***"
            )
            df_all_scen = df_all_scen[~df_all_scen.index.duplicated()]

        df_all_scen_short = df_all_scen.copy(deep=True)
        df_all_scen_short = fun_index_names(df_all_scen_short, True, str)
        df_all_scen_short = df_all_scen_short.rename({f"Downscaling[{model}]": model})
        # for _ in range(0,2):
        ## a) Short term policies adjustments:
        df_all_scen_short = fun_short(
            model,
            selection_dict,
            policy_scen_short,
            pd.DataFrame(),
            df_emi_factors,
            co2_energy_only,
            df_iam_all_models,
            pyam_mapping_file,
            model_patterns,
            region_patterns,
            target_patterns,
            csv_out,
            RESULTS_DATA_DIR,
            input_emi_target_short,
            # df_all_scen,
            df_all_scen_short[[str(x) for x in range(2010, 2105, 5)]],
            df_countries,
            start_policy=2025,
            countries_w_2030_emi_policies=countries_w_2030_emi_policies,
            weight=weight_short,
        )

        # Save short policies adj. to csv
        df_all_scen_short.to_csv(
            # "results/4_Policy_Adjustments/"
            # + model
            # + csv_out
            # + "_step4_SHORT_only"
            # + ".csv"
            Path(RESULTS_DATA_DIR / f"{model}{csv_out}_step4_SHORT_only.csv"),
        )

        if len(df_all_scen_short[df_all_scen_short.index.duplicated()]) > 0:
            print(
                "*** Warning: we found duplicated values in step 3 that will be removed ***"
            )
            df_all_scen_short = df_all_scen_short[~df_all_scen_short.index.duplicated()]

        setindex(df_all_scen_short, False)

        ## b) long term policies
        long_term_target = pd.DataFrame(
            [{c: timing_net_zero.get(c, 2050) for c in countries_w_2050_emi_policies}],
            index=["TIME"],
        ).T
        long_term_target["VALUE"] = 0
        long_term_target["MODEL"] = model
        long_term_target = pd.concat(
            [
                long_term_target,
                pd.DataFrame(
                    [{"SCENARIO": "scenario"}], index=countries_w_2050_emi_policies
                ),
            ],
            axis=1,
        ).rename_axis("ISO")
        policy_scen_long = [
            x
            for x in policy_scen_long
            if x in df_all_scen.reset_index().SCENARIO.unique()
        ]
        long_term_target = pd.concat(
            [long_term_target.replace({"scenario": scen}) for scen in policy_scen_long],
            axis=0,
        )

        # Add scenario (currently missing)
        df_all_scen_long = fun_long(
            model,
            long_term_target,
            policy_scen_long,
            selection_dict,
            df_all_scen_short,  # Short term policies (previously downscaled data) will be adjusted with long term policies
            df_countries,
            df_emi_factors,
            df_iam_all_models,
            RESULTS_DATA_DIR,
            csv_out,
            start_policy=2030,
            countries_w_2050_emi_policies=countries_w_2050_emi_policies,
            weight_long=weight_long,
        )

        df_all_scen_long.to_csv(
            # "results/4_Policy_Adjustments/"
            # + model
            # + csv_out
            # + "_step4_LONG.csv"
            RESULTS_DATA_DIR
            / f"{model}{csv_out}_step4_LONG.csv"  # should use this one instead 21 June 2023
        )

        print(
            "Scenarios with short term adjustments:",
            df_all_scen_short.index.get_level_values("SCENARIO").unique(),
        )
        print(
            "ISO with short term adjustments:",
            df_all_scen_short.index.get_level_values("ISO").unique(),
        )
        print("--------------")
        print(
            "Scenarios with long term adjustments:",
            df_all_scen_long.index.get_level_values("SCENARIO").unique(),
        )
        print(
            "ISO with long term adjustments:",
            df_all_scen_long.index.get_level_values("ISO").unique(),
        )

        # df_append will include short term scenarios that are not included in the df_all_scen_long (e.g. h_ndc).  df_append need to be appended to df_all_scen_long
        df_append = df_all_scen_short.loc[
            ~df_all_scen_short.index.isin(df_all_scen_long.index)
        ]

        df_all_scen_long.columns = [str(int(x)) for x in df_all_scen_long.columns]
        df_all_scen_short_long = pd.concat([df_all_scen_long,df_append])
        ## This contains all adjusted scenarios (short term + long term policies)
        # df_all_scen_short_long = df_all_scen_long.append(
        #     df_append
        # )  ## This contains all adjusted scenarios (short term + long term policies)

        ## Dropping duplicates
        df_all_scen_short_long = df_all_scen_short_long[
            ~df_all_scen_short_long.index.duplicated()
        ]

        ## Now we load step3 data
        file_not_adjusted_scen = csv_out + ".csv"
        if read_from_step5e:
            # From step5e
            excel_file = f"{model}_{project_name}{csv_out}_{str(harmonize_until)}_harmo_step5e_Scenario_Explorer_upload_FINAL.xlsx"
            df_not_adjusted = pd.read_excel(
                CONSTANTS.CURR_RES_DIR(str("step5")) / excel_file, engine="openpyxl"
            ).rename({"REGION": "ISO"}, axis=1)
            df_not_adjusted = fun_index_names(
                df_not_adjusted, True, str, sel_idx=df_all_scen_short_long.index.names
            )
            df_not_adjusted = df_not_adjusted.rename({f"Downscaling[{model}]": model})
        else:
            df_not_adjusted = pd.read_csv(
                PREV_STEP_RES_DIR / f"{model}{file_not_adjusted_scen}"
            ).rename({"REGION": "ISO"}, axis=1)
            df_not_adjusted.set_index(df_all_scen_short_long.index.names, inplace=True)

        # Bring back original names e.g. "Carbon Sequestration|CCS|Biomass"
        df_not_adjusted = df_not_adjusted.rename(
            {v: k for v, k in ngfs_2023_nomenclature.items()}, level="VARIABLE"
        )
        df_not_adjusted = fun_add_non_biomass_ren_nomenclature(
            df_not_adjusted, inverse=False
        )[0]
        if len(df_not_adjusted[df_not_adjusted.index.duplicated()]) > 0:
            print(
                "*** Warning: we found duplicated values in step 3 that will be removed ***"
            )
            df_not_adjusted = df_not_adjusted[~df_not_adjusted.index.duplicated()]

        if "REGION" in df_all_scen_short_long.reset_index().columns:
            colreg = "REGION"
        else:
            colreg = "ISO"
        if "EU27" in df_all_scen_short_long.reset_index()[colreg]:
            df_all_scen_short_long = df_all_scen_short_long.drop("EU27", level=colreg)
        df_all_scen_short_long = fun_index_names(df_all_scen_short_long, True, int)

        df_all_scen_short_long = fun_validation(
            CONSTANTS,
            RESULTS_DATA_DIR,
            selection_dict,
            project_name,
            df_all_scen_short_long,
            # fun_index_names(df_adjusted_all, True, int),#.droplevel('UNIT'),
            # fun_remove_np_in_df_index(df_adjusted_all)
            model_patterns,
            region_patterns,
            target_patterns,
            unique(
                step2_var
                + step3_var
                + [
                    "Primary Energy|Coal|w/ CCS",
                    "Primary Energy|Coal|w/o CCS",
                    "Primary Energy|Gas|w/ CCS",
                    "Primary Energy|Gas|w/o CCS",
                    "Emissions|CO2|Energy",
                    "Carbon Sequestration|CCS|Biomass",
                    # "Carbon Sequestration|CCS|Fossil",
                ]
            ),
            harmonize=True,
        ).droplevel("REGION")

        df_all_scen_short_long.to_csv(
            # "results/4_Policy_Adjustments/"
            # + model
            # + csv_out
            # + "_step4_SHORT_and_LONG.csv"
            RESULTS_DATA_DIR
            / f"{model}{csv_out}_step4_SHORT_and_LONG.csv"
        )

        ## We replace step3 data with `df_all_scen_short_long` data
        cols = df_all_scen_short_long.columns
        # cols = [str(int(x)) for x in cols]
        idx = df_all_scen_short_long.index
        idx = [x for x in idx if x in df_not_adjusted.index]
        # df_adjusted_all = df_not_adjusted
        df_adjusted_all = fun_index_names(df_not_adjusted, True, int)
        if not read_from_step5e and "UNIT" in df_adjusted_all.index.names:
            df_adjusted_all = df_adjusted_all.droplevel("UNIT")
        df_adjusted_all.loc[idx, cols] = df_all_scen_short_long.loc[idx, cols]

        if "UNIT" in df_adjusted_all.columns:
            df_adjusted_all = df_adjusted_all.droplevel("UNIT")
        # df_adjusted_all.to_csv(
        #     # "results/4_Policy_Adjustments/" + model + csv_out + "_step4_FINAL.csv"
        #     RESULTS_DATA_DIR
        #     / f"{model}{csv_out}_step4_BEFORE_FUN_VALIDATION.csv"
        # )

        df_adjusted_all.to_csv(
            # "results/4_Policy_Adjustments/" + model + csv_out + "_step4_FINAL.csv"
            RESULTS_DATA_DIR
            / f"{model}{csv_out}_step4_FINAL.csv"
        )

    # # REGIONAL VALIDATION (SUM ACROSS COUNTRIES Macthing regional IAM results)

    print("downscaling done (step 4)")
    return df_adjusted_all


if __name__ == "__main__":
    project_name = "NGFS"
    model_patterns = "*MESSAGE*"
    region_patterns = "*Saharan*"
    target_patterns = "*"

    main(
        project_name="NGFS",
        region_patterns=[
            "*"
        ],  # [i.replace(model+'|', '')+'r' for i in selection_dict[model]['regions'] if i.replace(model+'|', '')+'r' in regions_w_policies]
        # region_patterns="*Europe*",  #
        model_patterns="*MESSAGE*",
        target_patterns="*",
        policy_scen_long=[
            "o_1p5c",
            "o_1p5c_d50",
            "o_1p5c_d95",
            "Net Zero 2050",
            # EU AB (mostly ENGAGE) runs below
            "Delayed transition",
            "Divergent Net Zero",
            "EN_NPi2020_450",
            "EN_INDCi2100_NDCp",
            "EN_NPi2020_800",
            "EN_INDCi2030_300f",
            "EN_INDCi2030_400f",
            "EN_INDCi2030_900",
            "EN_NPi2020_500",
            "EN_INDCi2030_600_COV",
            "EN_INDCi2030_600_COV_NDCp",
            "EN_NPi2020_400",
            "NP_2025_-902050",
        ],  ## List of Scenarios to be adjusted with Long policies
        policy_scen_short=[
            "h_ndc_d50",
            "h_ndc_d95",
            "h_ndc",
            "h_cpol_d50",
            "h_cpol_d95",
            "h_cpol",
            "d_delfrag_d50",
            "d_delfrag_d95",
            "d_delfrag",
            "o_1p5c",
            "o_1p5c_d50",
            "o_1p5c_d95",
            # EU AB (mostly ENGAGE) runs below
            "Delayed transition",
            "Divergent Net Zero",
            "EN_NPi2020_450",
            "EN_INDCi2100_NDCp",
            "EN_NPi2020_800",
            "EN_INDCi2030_300f",
            "EN_INDCi2030_400f",
            "EN_INDCi2030_900",
            "EN_NPi2020_500",
            "EN_INDCi2030_600_COV",
            "EN_INDCi2030_600_COV_NDCp",
            "EN_NPi2020_400",
            "NP_2025_-902050",
        ],  ## List of Scenarios to be adjusted with policies
        csv_out="_NGFS_November",
        co2_energy_only=True,
    )
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
        csv_str="MODEL_NGFS22_March_prova5_all_step4_FINAL",
        model_patterns=model_patterns,
        region_patterns=region_patterns,
        target_patterns=target_patterns,
        vars=[
            "Emissions|CO2|Energy",
            "Carbon Sequestration|CCS|Biomass",
            "Primary Energy|Gas",
            "Primary Energy|Oil",
            "Primary Energy|Coal",
            "Primary Energy|Biomass",
            "Primary Energy|Coal|w/ CCS",
            "Primary Energy|Coal|w/o CCS",
            # "Secondary Energy|Electricity|Biomass",
            "Secondary Energy|Electricity|Wind",
            "Secondary Energy|Electricity|Solar",
            "Secondary Energy|Electricity|Gas",
            "Secondary Energy|Electricity|Coal",
            "Secondary Energy|Gases|Coal",
            "Secondary Energy|Liquids|Oil",
        ],
    )
