import csv
import os
import traceback
from typing import Callable, List, Union, Optional, Dict
from pathlib import Path

import numpy as np
import pandas as pd


from downscaler import CONSTANTS, IFT, RFT, plt
from downscaler.utils import (
    fun_clip_geo_hydro_step2b,
    fun_dynamic_downs_recursive,
    fun_make_step2b_input_data,
    fun_static_downscaling,
    fun_df_desired,
    fun_standardise_df_desired_within_region,
    fun_check_reg_results,
    fun_dynamic_downs,
    fun_rand_elc_criteria,
    fun_all_criteria,
    fun_weighted_criteria_and_finalise,
    fun_calculate_electricity_emissions,
    fun_select_criteria_with_max_range,
    fun_get_ssp,
    fun_flatten_list,
    unique,
    fun_read_cached_sensitivity,
    fun_read_cache_simple,
)
from downscaler.fixtures import fuel_list_steb2b as fuel_list
from downscaler.fixtures import sectors_step2b as sectors
from downscaler.fixtures import file_name_dict, lifetime_dict, IAM_fuel_dict, df_weights


def main(
    var,
    file,
    region,
    model,
    target,
    file_suffix,
    df_iam_all_models,  # from steb 2b =>this one does not contain Final enegy electricity
    df_iea_all,
    df_platts,
    df_weights,
    pyam_mapping_file,
    base_year: int = 2010,
    tol: float = 0.05,
    step1b=False,
    _func="log-log",
    _scen_dict=None,
    default_ssp_scenario: str = "SSP2",
    _run_sensitivity: bool = True,
    range_random_weights: list = list(range(1, 10)),
    # range_random_weights: list = list(range(1, 1001)), # This is our default range
    # range_random_weights: list = list(range(1, 5)),
    read_cached_sensitivity: Union[None, pd.DataFrame] = None,
) -> dict:

    if read_cached_sensitivity is not None:
        range_random_weights = fun_read_cache_simple(
            read_cached_sensitivity, range_random_weights
        )

    PREV_STEP_RES_DIR = CONSTANTS.PREV_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )

    (
        df_iea_all,
        df_platts,
        m,
        cost_curve_list,
        df_gov_all,
        countrylist,
        df_iam_sel,
        df_demand,
        df_iea_melt,
        df_iam,
        time_periods,
    ) = fun_make_step2b_input_data(
        var,
        file,
        region,
        model,
        target,
        file_suffix,
        df_iam_all_models,
        df_weights,
        pyam_mapping_file,
        PREV_STEP_RES_DIR,
        file_name_dict,
        sectors,
        fuel_list,
        IAM_fuel_dict,
        df_iea_all,
        df_platts,
        base_year,
        _func=_func,
    )

    ssp = fun_get_ssp(target, default_ssp="SSP2", _scen_dict=_scen_dict)
    df_iam=df_iam.set_index('TIME')
    # NOTE df_last_year_share.groupby('IAM_FUEL').sum() is equal to 1

    ## Initialising a "df_desired" dataframe that will contain electricty mix data, based on df_gov_all data
    df_desired, df_base_year_share, df_last_year_share = fun_df_desired(
        base_year,
        2016,
        df_demand,
        df_gov_all,
        countrylist,
        ssp,
        fuel_list,
        df_iam,
        df_iea_melt,
        time_periods,
    )

    df_iam_sel=df_iam_sel.set_index('TIME')

    # Creating different criteria for electricity generation downscaling")
    df_gov, df_criteria_not_weighted = fun_all_criteria(
        ssp,
        m,
        cost_curve_list,
        df_gov_all.loc[time_periods],
        countrylist,
        df_iam_sel,
        df_demand,
        df_iam,
        df_desired,
        df_base_year_share,
        df_last_year_share,
        fuel_list,
        file_name_dict,
        lifetime_dict,
    )

    # Range based on random electricity weights (about 1 min for 100 criteria. 0.5 sec for criteria)
    full_range = ["standard"]
    if _run_sensitivity:
        full_range = ["standard"] + range_random_weights
    df_weights_dict = {}
    df_desired_dict = {}
    df_all_criteria_dict = {}

    for ra in full_range:  ## uncertainty criteria loop
        if ra == "standard":
            df_weights_dict[ra] = df_weights
        else:
            df_weights_dict[ra] = fun_rand_elc_criteria(__seed=ra).set_index("IAM_FUEL")

    print('starting criteria/sensitivity loop')
    for ra, w in df_weights_dict.items():
        # Below print % progress when calculating criteria
        if isinstance(ra, int) or isinstance(ra, float):
            siter=full_range[-1]
            # count=((ra+1)/siter)
            # if np.round(100*count,2) in range(0,100,1):
            #     print(f"criteria: {np.round(100*count,0)} % complete \r",)
            if ra in list(range(0,10000,10)):
                print(f"criteria: {ra} / {siter+1} done \r",)
        
        
        # Combine criteria based on df_weights
        df_all_criteria_dict[ra] = fun_weighted_criteria_and_finalise(
            var,
            df_platts,
            w,
            countrylist,
            df_demand,
            fuel_list,
            df_gov,
            df_criteria_not_weighted,
        )

        temp_dict = {
            "static": fun_static_downscaling(
                df_all_criteria_dict[ra],
                base_year,
                countrylist,
                df_iam_sel,
                df_iam,
                time_periods,
                df_desired,
                df_base_year_share,
                df_last_year_share,
                fuel_list,
            ).copy(deep=True)
        }

        # Dynamic downscaling (this function modified df_desired inplace for some reasons)
        df_desired = fun_dynamic_downs(
            df_iam_sel,
            temp_dict["static"],
            df_all_criteria_dict[ra],
            2015,
            fuel_list,
        )

        df_desired = fun_clip_geo_hydro_step2b(df_desired)
        df_desired = fun_standardise_df_desired_within_region(
            fuel_list, df_desired, df_iam
        )

        # Check if df_desired matches regional iam results
        fun_check_reg_results(df_iam_sel, df_desired, fuel_list)

        df_desired = df_desired.drop_duplicates(subset=None, keep="last")
        temp_dict["dynamic"] = df_desired.copy(deep=True)
        temp_dict["dynamic_recursive"] = fun_dynamic_downs_recursive(
            df_iam_sel,
            temp_dict["static"],
            df_all_criteria_dict[ra],
            2015,
            fuel_list,
        )

        # Add emissions from electricity and elc demand
        for y in temp_dict:
            temp_dict[y]["emi"] = fun_calculate_electricity_emissions(
                temp_dict[y], df_iam_all_models, model, region, target
            ).copy(deep=True)
            temp_dict[y]["DEMAND"] = df_demand["ENSHORT_REF"].copy()

        df_desired_dict[ra] = temp_dict

    # 2022_11_09 identify max/min scenarios.
    sel_criteria, all_seeds_by_country = fun_select_criteria_with_max_range(
        countrylist, full_range, df_desired_dict
    )
    
    # DO NOT DELETE BELOW COMMENTED BLOCK
    # SUGGEST TO USE COMMENTED BLOCK BELOW which takes min/max for each single fuels, not just for electricity related emissions
    # res={}
    # # Get results for time (t), country (c) and fuel (f)
    # for c in ['AUS','NZL','JPN']:
    #     for f in fuels:
    #         res_single_fuel_country=df_desired.xs((t, c), level=('TIME','ISO'))[[f]].describe().T[['min','max']]
    #         for x in ['min','max']:
    #             res[f"{c}-{f}-{x}"]=df_desired[df_desired.eq(res_single_fuel_country[x].iloc[0]).any(1)].CRITERIA.iloc[0]

    # # List of all criteria associated to min/max across all countries
    # all_criteria=set(list(res.values())+['standard'])

    


    # BLOCK BELOW IS USED TO PLOT THE RESULTS
    # f='COAL'
    # c='JPN'
    # for x in full_range:
    #     df_desired_dict[x]['static'].xs(c, level='ISO')[f].plot(c='#DE4E4F', alpha=0.1, linewidth=2)
    #     df_desired_dict[x]['dynamic'].xs(c, level='ISO')[f].plot(c='#0B84A5', alpha=0.1, linewidth=2)
    #     df_desired_dict[x]['dynamic_recursive'].xs(c, level='ISO')[f].plot(c='#F6C85F', alpha=0.1, linewidth=2)
    # #plt.ylim(top=0.3)
    # plt.xlim(right=2035)
    # plt.show()
    # plt.title(f'{c}|{f}')

    # # new colors
    # f='COAL'
    # c='JPN'
    # a=0.25
    # lw=1.5
    # for x in full_range:
    #     df_desired_dict[x]['static'].xs(c, level='ISO')[f].plot(c='#FFC182', alpha=a, linewidth=lw)
    #     df_desired_dict[x]['dynamic'].xs(c, level='ISO')[f].plot(c='#84C3C3', alpha=a, linewidth=lw)
    #     df_desired_dict[x]['dynamic_recursive'].xs(c, level='ISO')[f].plot(c='#8282D1', alpha=a, linewidth=lw)
    # plt.ylim(bottom=0, top=1.3)
    # plt.xlim(right=2050)
    # plt.show()
    # plt.title(f'{c}|{f}')

    # dictionary with: default criteria, upper range, lower range:
    calc_type = "dynamic_recursive"
    all_criteria = ["standard"] + sel_criteria
    all_data = [df_desired_dict[x][calc_type].drop("emi", axis=1) for x in all_criteria]
    res_summary = dict(zip(["standard", "upper", "lower"], all_data))
    # NOTE 'upper' and 'lower' are arbitrary names:
    # a selected criteria can be in the upper end of the range for one country,
    # and lower end of the range for another  (sum across all countries
    # needs to match IAMs results)

    # 1) Dataframe with cached sensitivity results (contains min/max criteria) to be saved in  `sensitivity_cache.csv`
    my_col = min(range_random_weights), max(range_random_weights)
    sel_seeds = pd.DataFrame(
        {
            "MODEL": [model],
            "REGION": [region],
            "SCENARIO": [target],
            "full_range": [my_col],
            "sel_min_max": [str(sel_criteria)],
            "detailed": [str(all_seeds_by_country)],
        }
    )

    # save electricity sensitivity results
    if _run_sensitivity:
        df_sens=pd.concat([df_desired_dict[x][i].assign(CRITERIA=x, METHOD=i).set_index(['CRITERIA', 'METHOD'], append=True) 
                        for x in full_range for i in df_desired_dict[x].keys()])
        # _criteria from_{full_range[1]}_to_{full_range[-1]}
        folder=CONSTANTS.CURR_RES_DIR('step2')/"step2b_sensitivity"
        os.makedirs(folder, exist_ok=True)
        df_sens.to_csv(folder/f"{model}_{region}_{target}_{file_suffix}.csv")

    # returns results dictionary for all criteria, and selected min/max criteria seeds
    # sel_seeds dataframe to be saved to csv in the primary_energy_2a.py (here we do not have PROJECT folder information)
    return res_summary, sel_seeds.set_index("MODEL")
