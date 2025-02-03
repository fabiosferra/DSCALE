import glob
import os
import shutil
import traceback
from multiprocessing import Lock
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from joblib import Parallel, delayed
from pyexpat import model

from downscaler import CONSTANTS, IFT
from downscaler.fixtures import (
    IAM_fuel_dict,
    clipped_fuels_2010,
    fun_conv_settings,
    gdp,
    iea_flow_dict,
    list_of_ec,
    list_of_fuels,
    list_of_sectors,
    pop,
    sectors_adj,
    step2_primary_energy,
    vars_to_be_harmo_step2a,
    sectors_step2b,
)
from downscaler.utils_pandas import fun_create_var_as_sum, fun_rename_index_name, fun_xs, fun_read_csv
from downscaler.input_caching import get_selection_dict
from downscaler.utils import (
    InputFile,
    convert_to_list,
    fun_add_gdp_pop_from_ssp,
    fun_anchor_emi_to_hist_data_step3,
    fun_apply_secondary_mix_to_final_sectors,
    fun_blending,
    fun_check_iea_gases_list_fuels,
    fun_create_iea_dict_from_iea_flow_dict,
    fun_from_final_to_secondary_enshort_combi,
    fun_get_ssp,
    fun_iterative_adj,
    fun_list_of_models,
    fun_list_time_of_conv,
    fun_load_data_from_step1_and_modify_column_names,
    fun_load_df_iam_all,
    fun_load_iea,
    fun_load_platts,
    fun_minimize_elc_trade_and_clip_hydro_geoth,
    fun_pd_long_format,
    fun_pd_wide_format,
    fun_primary_energy,
    fun_rand_elc_criteria,
    fun_read_df_countries,
    fun_read_df_iam_all_and_slice,
    fun_read_df_iam_iamc,
    fun_read_df_ssp,
    fun_run_step2b,
    fun_secondary_and_final,
    fun_slice_df_iam_all_models,
    fun_targets_regions_df_countries,
    fun_validation,
    make_input_data,
    setindex,
    fun_wildcard,
    fun_create_residential_and_commercial_as_sum_in_step2,    
    fun_step2_format_from_iamc,
    fun_select_step2_variables,
    fun_iamc_format_and_select_variables,
    fun_read_down_NOT_harmo,
    fun_find_nearest_values,
)

CSV_LOCK = Lock()

## step 2 blueprint 2022_06_08
# 1 update functions
# 2 create tests


def main(
    file: IFT = CONSTANTS.INPUT_DATA_DIR / "NGFS" / "snapshot_all_regions.csv",
    pyam_mapping_file: IFT = (
        CONSTANTS.INPUT_DATA_DIR / "NGFS" / "default_mapping.csv"
    ),
    region_patterns: Union[str, list] = "*",
    model_patterns: Union[str, list] = "*",
    target_patterns: Union[str, list] = "*",
    random_electricity_weights: bool = False,
    ## Maximum number of simulations with random electricity weights
    range_rand_elc_weights: range = range(1, 1000),
    elc_trade_adj: bool = True,
    csv_out: str = "round_1p4_enlong_reg_lenght",
    df_iam_all_input=None,
    step1b: bool = True,
    step_2b_input: bool = None,
    # !!!!!!!!!!!!!!!!!!!!!!!!!  THIS IS A TEMPORARY HACK TO MAKE IT WORK FOR THE PAPER !!!!!!!!!!!!!!!!!!!!!!!!
    # step_2b_input='MESSAGEix-GLOBIOM 1.1-M-R12_Pacific OECDr_h_cpol_2023_12_21.csv', # TEMPORARY HACK TO MAKE IT WORK FOR THE PAPER (step2 sensitivity analysis with full range)
    save_to_csv: bool = True,
    run_step_2b_test: bool = False,
    scen_dict: Optional[dict] = None,
    run_sensitivity: bool = None,
    use_cached_sensitivity=True,
    default_ssp_scenario: str = "SSP2",
    long_term=None, #,'ENLONG_RATIO',
    # method='wo_smooth_enlong',
    method =None, # Iterare across all methods
    # _func:str='log-log',
):
    if method=='wo_smooth_enlong' and long_term is not None and long_term!= 'ENLONG_RATIO':
        txt="the METHOD `wo_smooth_enlong` is only supported for long_term='ENLONG_RATIO' "
        txt2= 'Or please select another method'
        raise ValueError(f"{txt}. You choose: long_term={long_term}.{txt2} ")

    
    
    ele_list = ["Electricity output (GWh)", "Imports", "Exports"]
    liquids_gases_list = ["|Liquids", "|Gases"]
    allowed_sub_sectors = list_of_fuels + list_of_ec + list_of_sectors
    region_patterns = convert_to_list(region_patterns)
    model_patterns = convert_to_list(model_patterns)
    target_patterns = convert_to_list(target_patterns)
    range_random_weights = (
        list(range_rand_elc_weights) if random_electricity_weights else [1]
    )
    df_iam = fun_read_df_iam_iamc(file)
    RESULTS_DATA_DIR = (
        CONSTANTS.CURR_RES_DIR(str(Path(os.path.abspath("")) / Path(__file__).name))
        / "tmp"
    )

    if step_2b_input is None:
        if not run_step_2b_test:
            RESULTS_DATA_DIR.mkdir(exist_ok=True)

    PREV_STEP_RES_DIR = CONSTANTS.PREV_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )

    ele_list = ["Electricity output (GWh)", "Imports", "Exports"]
    liquids_gases_list = ["|Liquids", "|Gases"]

    region_patterns = convert_to_list(region_patterns)
    model_patterns = convert_to_list(model_patterns)
    target_patterns = convert_to_list(target_patterns)
    range_random_weights = (
        list(range_rand_elc_weights) if random_electricity_weights else [1]
    )

    # Load data
    df_platts = fun_load_platts()
    df_iea_all, df_iea_melt = fun_load_iea()[1:]
    df_iam_all, df_iam_all_models = fun_load_df_iam_all(
        df_iam_all_input, model_patterns, target_patterns, file
    )
    df_countries = fun_read_df_countries(
        CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv",
    )

    # Check iea_gases list of fuels
    fun_check_iea_gases_list_fuels()

    # List of models based on model_patterns
    models = fun_list_of_models(df_iam_all, model_patterns)
    file_suffix = csv_out  ## Output file of step 1

    # MAIN LOOP

    conv_settings = fun_conv_settings(run_sensitivity)  # sensitivity
    df_all = pd.DataFrame()
    for model in models:
        count = 0
        # list of targets, regions and df_countries for a given model
        targets, regions, df_countries = fun_targets_regions_df_countries(
            model,
            df_iam_all_models,
            region_patterns,
            target_patterns,
            pyam_mapping_file,
            df_countries,
        )
        for target in targets:
            ssp_scenario = fun_get_ssp(
                target, default_ssp=default_ssp_scenario, _scen_dict=scen_dict
            )
            df_ssp_melt = fun_read_df_ssp(pop, gdp, ssp_scenario, "OECD Env-Growth")[1]
            for region in regions:
                countrylist = df_countries[df_countries.REGION == region].ISO.to_list()
                if len(countrylist) > 1:
                    if model == "MESSAGEix-GLOBIOM 1.1-M-R12" and 'Pacific OECD' in region and target=='h_cpol' and file_suffix in ['2024_09_09_TEST', '2024_09_19']:
                        print('************************************************************')
                        print(f'NOTE: We will read previously downscaled electricity results from {step_2b_input}')
                        print('************************************************************')
                    elif step_2b_input is not None:
                        path_parts = str(CONSTANTS.CURR_RES_DIR()).split('\\')
                        # If this is not a test, ask the user if they want to read the electricity results from a previous run
                        if not all('test' in part for part in path_parts[-3:]):
                            # NOTE: using step_2b_input is a hack for the technical paper
                            txt = f"Are you sure you want to read the electricity results from {step_2b_input} - without running the algorithm? (y/n) NOTE: This is a hack for the technical paper. If you don't know what you are doing, please type 'n' and pass `step_2b_input=None`."
                            action = input(txt)
                            if action.lower() not in ["yes", "y"]:
                                raise ValueError("Simulation aborted by the user")
                    else:
                        pass
                    df_iam_all, df_iam_all_models = fun_slice_df_iam_all_models(
                        model,
                        region,
                        target,
                        df_iam_all,
                    )
                    df_iam_electr = df_iam_all_models[
                        (df_iam_all_models.VARIABLE.str.startswith("Secondary Energy"))
                    ]
                    df_iea_electr = df_iea_all[df_iea_all.FLOW.isin(ele_list)]


                    count = 0
                    df_step1harmo=fun_read_down_NOT_harmo(
                                        PREV_STEP_RES_DIR,
                                        model,
                                        region.replace("|", "_"),
                                        target,
                                        step1b,
                                        file_suffix,
                                    )
                    func_list=df_step1harmo.FUNC.unique()
                    func_list=[x for x in func_list if isinstance(x,str)]
                    
                    if len(func_list)==0:
                        raise ValueError(f"ERROR: No functions found in `df_step1harmo.FUNC`")

                    # TODO: Add a loop for long_term in ["ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO"]
                    # ,"ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO", 
                    # for long_term in  ["ENLONG_RATIO"]:
                    if run_sensitivity:
                        long_term_list = ["ENLONG_RATIO","ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO", "ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO"]
                    else:
                        long_term_list = ["ENLONG_RATIO"]
                    for long_term in  long_term_list:
                        print(f"*** {long_term} ***")
                        var_list = [long_term, "ENSHORT_REF"]
                        # This method loop can be moved after step2b as it has no influence on the elecricity calculations (trade minimization happens later)
                        # if method is None:
                        #     if long_term=='ENLONG_RATIO':
                        #         method_list = ['wo_smooth_enlong']
                        #     elif "METHOD" in df_step1harmo.columns:
                        #         method_list = [x for x in df_step1harmo.METHOD.unique() if x not in ['wo_smooth_enlong']]
                        # else:
                        #     method_list = [method]

                        if long_term=='ENLONG_RATIO':
                            method_list = ['wo_smooth_enlong']
                        else:
                            method_list = [x for x in df_step1harmo.METHOD.unique() if x not in ['wo_smooth_enlong']]  
                        for _func in func_list:
                            for method in method_list:
                                print(f"*** {long_term}  -   {method} ***")
                                # for var in var_list:
                                df_all = fun_load_data_from_step1_and_modify_column_names(
                                    PREV_STEP_RES_DIR,
                                    region,
                                    model,
                                    file_suffix,
                                    target,
                                    step1b,
                                    var_list,
                                    _func=_func
                                )

                                # NOTE DEPRECATED - TO BE DELETED In step2  uses `ENLONG_RATIO` as a column. We then assume it always search for `wo_smooth_enlong` as a METHOD
                                # Therefore we only need to replace this with the values in our default method
                                # sel_method={'col':'ENLONG_RATIO', 'METHOD':'wo_smooth_enlong'}
                                # df.loc['wo_smooth_enlong', 'ENLONG_RATIO']
                                if 'METHOD' in df_all.reset_index().columns:
                                    df_all=df_all.set_index('METHOD', append=True).xs(method, level='METHOD')
                                for ra in range_random_weights:  ## uncertainty criteria loop
                                    count = 0  # we need an header for each csv file, therefore we set count=0
                                                
                                    if random_electricity_weights:
                                        df_weights = fun_rand_elc_criteria(__seed=ra)
                                    else:
                                        from downscaler.fixtures import df_weights
                                    # try:
                                    # name of CSV file
                                    file_name_iamc = f"{model}_{region}_{target}_{csv_out}"
                                    countrylist = df_countries[
                                        df_countries.REGION == region
                                    ].ISO.to_list()
                                    print("==================")
                                    print(region, countrylist)
                                    print("==================")

                                    sens_cache = None
                                    sens_file_path = RESULTS_DATA_DIR / "../sensitivity_cache.csv"
                                    idxcol = ["MODEL", "REGION", "SCENARIO", "file"]
                                    if run_sensitivity and use_cached_sensitivity:
                                        if os.path.isfile(sens_file_path):
                                            sens_cache = pd.read_csv(sens_file_path)
                                            sens_cache = sens_cache.set_index(idxcol)
                                            try:
                                                sens_cache = (
                                                    sens_cache.loc[model]
                                                    .loc[region]
                                                    .loc[target]
                                                    .loc[str(file)]
                                                )  # .reset_index()
                                                # NOTE: the below can happen if we have only 1 simualtion available in the df
                                                if type(sens_cache) == pd.Series:
                                                    sens_cache = pd.DataFrame(sens_cache).T
                                            except:
                                                sens_cache = None

                                        else:
                                            pass
                                            ## ASK USER (INPUT) IF WISH TO CONTINUE

                                    ## uncertainty criteria loop
                                    if 'METHOD' in df_all.reset_index().columns:
                                        df_all=df_all.set_index('METHOD', append=True).xs(method, level='METHOD')
                                    

                                    if step_2b_input is None:
                                        res_and_comm=fun_wildcard(['Final Energy|Residential and Commercial*'], sectors_step2b)
                                        comm=fun_wildcard(['Final Energy|Commercial*'], sectors_step2b)
                                        res=fun_wildcard(['Final Energy|Residential*'], sectors_step2b)
                                        if len(res_and_comm)==0 and (len(comm)*len(res))!=0:
                                            # Create sectors "Residential and commercial" as sum of Residential and Commercial
                                            df_all=fun_create_residential_and_commercial_as_sum_in_step2(var_list, df_all, model, target, region)
                                        df_all_dict, df_desired_dict, df_sel_seeds = fun_run_step2b(
                                            df_all,
                                            model,
                                            region,
                                            target,
                                            file,
                                            file_suffix,
                                            df_iam_electr,
                                            df_iea_electr,
                                            df_platts,
                                            df_weights,
                                            pyam_mapping_file,
                                            step1b,
                                            random_electricity_weights,
                                            _scen_dict=scen_dict,
                                            _run_sensitivity=run_sensitivity,
                                            read_cached_sensitivity=sens_cache,
                                            _func=_func,
                                        )


                                        # NOTE: we do not have a project_file folder.
                                        # Therefore we add the `file` path folder (to distinguish the project e.g. NGFS 2022)

                                        df_sel_seeds["file"] = file
                                        if run_sensitivity:
                                            # Save cached sensitivity results (always in append mode)
                                            df_sel_seeds.to_csv(
                                                RESULTS_DATA_DIR / ".." / "sensitivity_cache.csv",
                                                mode="a",
                                            )

                                        if run_step_2b_test:
                                            return df_all["standard"], df_desired["standard"]
                                    elif run_step_2b_test:
                                        raise ValueError(
                                            "If you want to run step_2b,  step_2b_input must be None. Otherwise the function will return your (exogenous) step_2b_input data"
                                        )
                                    elif isinstance(step_2b_input, str):
                                        # Step1 - Find all relevant criteria weights (max/min across all countries)
                                        myfile=CONSTANTS.CURR_RES_DIR('step2')/'step2b_sensitivity'/step_2b_input
                                        dfin=fun_read_csv({'aa':myfile}, True, int)['aa'].reset_index().set_index(['TIME','ISO'])
                                        dfin=dfin[dfin.METHOD=='dynamic_recursive']
                                        fuels = ["COAL", "NUC", "GAS","BIO", "SOL","WIND", "HYDRO","OIL", "GEO"]
                                        t=2050

                                        res={}
                                        # Get results for time (t), country (c) and fuel (f)
                                        criteria_range = ['min','max', 
                                                        '50%'
                                                        ] # 50% is the median. So we take MIN/MAX/MEDIAN
                                        for c in ['AUS','NZL','JPN']:
                                            for f in fuels:
                                                res_single_fuel_country=dfin.xs((t, c), level=('TIME','ISO'))[[f]].describe().T[criteria_range]
                                                for x in criteria_range:
                                                    # Below we take the closest value to the min/max/median. (because for the median we do not have a criteria (it's the median across all criteria), we take the closest value)
                                                    closest=fun_find_nearest_values(list(dfin.xs((t, c), level=('TIME','ISO'))[f].unique()), res_single_fuel_country[x].iloc[0], 1)[0]
                                                    # Below we take the criteria corresponding to the closest (min/max/median) value
                                                    res[f"{c}-{f}-{x}"]=dfin[dfin.eq(closest).any(1)].CRITERIA.iloc[0]
                                                    
                                        # List of all criteria associated to min/max across all countries
                                        all_criteria=set(list(res.values())+['standard'])

                                        # Step 2 - Create `df_desired_dict` dictionary that contains results for all criteria
                                        df_desired_dict={x:dfin[dfin.CRITERIA==x][fuels+['DEMAND']] for x in all_criteria}

                                        # Step3 - Create `myfinal_dict `dictionary to rename columns e.g. {COAL:'Secondary Energy|Electricity|CoalENSHORT_REF') to be used in step4
                                        mytemp_dict={v:k for k,v in IAM_fuel_dict.items()}
                                        myfinal_dict={x:f"Secondary Energy|Electricity{mytemp_dict[x]}ENSHORT_REF" for x in dfin[dfin.CRITERIA=='standard'][fuels].columns}

                                        # Step 4 Create `df_desired_dict` dictionary that contains results for all criteria using step2 format -> e.g. 'Secondary Energy|Electricity|CoalENSHORT_REF'
                                        df_all_dict= {k:pd.concat([df_all, v.rename(myfinal_dict, axis=1)], axis=1) for k,v in df_desired_dict.items()}
                                    else:
                                        df_all_dict = {}
                                        df_desired_dict = {}
                                        for x in ["standard"]:
                                            df_all_dict[x], df_desired_dict[x] = step_2b_input



                                for ra in df_all_dict.keys():
                                    df_all = df_all_dict[ra].copy()
                                    df_desired = df_desired_dict[ra].copy()
                                    # SECONDARY ENERGY MIX ENLONG (we already did the ENSHORT downscaling for electricity)
                                    # The below creates the following: for each fuel (we do it both for final and secondary -> final  needed for solids):
                                    # 'Final Energy|Electricity|BiomassENLONG_RATIO'
                                    # 'Secondary Energy|Electricity|BiomassENLONG_RATIO'
                                    # ... etc.
                                    fun_secondary_and_final(
                                        "Secondary Energy|Electricity",
                                        model,
                                        region,
                                        target,
                                        df_all,
                                        df_iam_all_models,
                                        allowed_sub_sectors,
                                        df_iea_melt,
                                        _var=long_term,
                                    )

                                    try:
                                        # This allows models like Promethes to work (as they do not provide detailed sectoral secondary energy data)
                                        df_all = fun_fuel_mix_secondary_and_final_by_sectors(
                                                liquids_gases_list,
                                                allowed_sub_sectors,
                                                df_iea_melt,
                                                df_iam_all_models,
                                                var_list,
                                                model,
                                                target,
                                                region,
                                                df_all,
                                            )
                                    except Exception as e:
                                        print(e)
                                        print(
                                            f"*** fun_fuel_mix_secondary_and_final_by_sectors did not work for {region} ***"
                                        )
                                    
                                    # We add GDP AND POPULATION TO DF_ALL (from df_ssp_melt, not downscaled)
                                    df_all = fun_add_gdp_pop_from_ssp(
                                            df_all, df_ssp_melt, gdp, pop
                                        )

                                    # if ra == "standard":
                                    df_all = fun_minimize_elc_trade_and_clip_hydro_geoth(
                                        model,
                                        region,
                                        target,
                                        df_all,
                                        df_iam_all_models,
                                        elc_trade_adj,
                                        df_desired,
                                        clipped_fuels_2010,
                                        var_list,
                                        min_iter=0,  # 3,
                                        conv_threshold=1e3,  # 0.001,
                                    )

                                    if not elc_trade_adj:
                                            file_name_iamc = (
                                                file_name_iamc + "_" + "wo_elc_trade_adj"
                                            )

                                    for var in var_list:
                                        df_all = fun_primary_energy(
                                            model,
                                            region,
                                            target,
                                            df_all,
                                            var,
                                            df_iam_all_models,
                                        )

                                    df_all.sort_index(inplace=True)

                                    # Blueprint to match Historial Primary energy data
                                    # Get ENLONG and ENSHORT data in IAMc format
                                    res = {
                                        x: fun_iamc_format_and_select_variables(
                                            model,
                                            target,
                                            region,
                                            df_all,
                                            ra,
                                            file_name_iamc=file_name_iamc,
                                            str_to_find=x,
                                            random_electricity_weights=random_electricity_weights
                                        )
                                        for x in [long_term, "ENSHORT_REF"]
                                    }

                                    var = "Primary Energy"

                                    # varname_long = {
                                    #     x: x.replace("ENLONG_RATIO", "")
                                    #     for x in res["ENLONG"].reset_index().VARIABLE
                                    # }
                                    # varname.update(varname_long)

                                    iea_dict = fun_create_iea_dict_from_iea_flow_dict(
                                        [var], iea_flow_dict
                                    )

                                    real_name_dict = {
                                        # "ENSHORT": "ENSHORT_REF",
                                        # "ENLONG": "ENLONG_RATIO",
                                        "ENSHORT_REF": "ENSHORT_REF",
                                        "ENLONG": long_term,
                                        }
                                    

                                    for x in ["ENSHORT_REF"]:
                                        varname = {
                                            v: v.replace(real_name_dict[x], "")
                                            for v in res[x].reset_index().VARIABLE
                                        }
                                        # Crate variable "Primary Energy" as the sum of
                                        for i, j in step2_primary_energy.items():
                                            res[x] = fun_create_var_as_sum(
                                                res[x].rename(varname), i, j, unit="EJ/yr"
                                            )
                                        # Achor primary energy to historical data.
                                        # NOTE:The below works with energy data as well (e.g. Final Energy)
                                        res[x] = fun_anchor_emi_to_hist_data_step3(
                                            file,
                                            countrylist,
                                            res[x].rename(varname),
                                            iea_dict,
                                            var,
                                            [2010, 2015, 2020],
                                        )

                                        ## TODO add a top down harmonization here (from Total primary to the fuels)
                                        ## TODO add here
                                        inv_dict = {
                                            v: f"{v}{real_name_dict[x]}"
                                            for v in varname.values()
                                        }
                                        res[x] = res[x].drop("Primary Energy", level="VARIABLE")
                                        res[x] = res[x].rename(inv_dict)

                                    # Now that we have harmonized th data we go back to step2 format
                                    df_all = pd.concat(
                                        [
                                            fun_step2_format_from_iamc(res[x])
                                            for x in [long_term, "ENSHORT_REF"]
                                        ],
                                        axis=1,
                                    )

                                    # NOTE: Block below currently commented out - should be brought back
                                    # in the PR  `Feature/step1 refactor #200`

                                    # for x in ["ENSHORT_REF", long_term]:
                                    #         df_all = run_sector_harmo_enhanced(
                                    #             df_all,
                                    #             fun_sub_sectors_dict(df_all, sect_adj),
                                    #             x,
                                    #             df_iam.xs(
                                    #                 [reg, target],
                                    #                 level=["REGION", "TARGET"],
                                    #                 drop_level=False,
                                    #             ),
                                    #         )
                                    
                                    
                                    # # Exlude Final Energy variables not present in IAMs results
                                    # vars_not_present_in_iams = [
                                    #     x
                                    #     for x in df_all.reset_index().SECTOR.unique()
                                    #     if x not in df_iam_all.VARIABLE.unique()
                                    # ]
                                    # vars_not_present_in_iams = [
                                    #     x
                                    #     for x in vars_not_present_in_iams
                                    #     if "Final Energy" in x
                                    # ]
                                    # df_all = fun_xs(
                                    #     df_all,
                                    #     {"SECTOR": vars_not_present_in_iams},
                                    #     exclude_vars=True,
                                    # )

                                    # # Go back to ste1b format to create bledning projections
                                    # df_all = fun_from_step2_to_step1b_format(
                                    #     df_all.xs("none", level="METHOD"),
                                    #     target,
                                    #     cols=["ENSHORT_REF", long_term],
                                    #     reverse=True,
                                    # )

                                    df_all=fun_blending_general(run_sensitivity, long_term, df_all, use_asterics=False)

                                    file_name_iamc = (
                                        f"{file_name_iamc.replace('.csv','')}_{ra}"
                                        if random_electricity_weights
                                        else f"{file_name_iamc.replace('.csv','')}"
                                    )


                                    # Select step2 variables that contain strings below
                                    if model == "MESSAGEix-GLOBIOM 1.1-M-R12" and 'Pacific OECD' in region and target=='h_cpol' and file_suffix in ['2024_09_09_TEST', '2024_09_19']:
                                        # We just keep secondary energy|electricity variables to avoid a large file
                                        mystep2_strings = ['BLEND', 'Secondary', 'Electricity',]
                                    else:
                                        mystep2_strings = ['BLEND']
                                    
                                    df_all, file_name_iamc = fun_select_step2_variables(
                                        random_electricity_weights,
                                        model,
                                        target,
                                        region,
                                        df_all,
                                        ra,
                                        file_name_iamc,
                                        list_of_strings=mystep2_strings
                                    )
                                   
                                    if save_to_csv:
                                        df_all["CRITERIA"] = ra
                                        df_all = df_all.set_index("CRITERIA", append=True)
                                        
                                        df_all["FUNC"] = _func
                                        df_all = df_all.set_index("FUNC", append=True)
                                        
                                        df_all["LONG_TERM"] = long_term
                                        df_all = df_all.set_index("LONG_TERM", append=True)

                                        df_all["METHOD"] = method
                                        df_all = df_all.set_index("METHOD", append=True)

                                        if count == 0:
                                            os.makedirs(RESULTS_DATA_DIR, exist_ok=True)
                                            # Changed to mode "w" for writing the start of the file
                                            df_all.to_csv(
                                                RESULTS_DATA_DIR / f"{file_name_iamc.replace('.csv','')}.csv",
                                                mode="w",
                                                header=True,
                                            )

                                        else:
                                            with CSV_LOCK:
                                                df_all.to_csv(
                                                    RESULTS_DATA_DIR / f"{file_name_iamc.replace('.csv','')}.csv",
                                                    mode="a",
                                                    header=False,
                                                )
                                    count += 1

                                # except Exception as e:
                                #     print(e)
                                #     print(traceback.format_exc())
                                #     print(
                                #         "This model/region/scenario does not work:",
                                #         model,
                                #         region,
                                #         target,
                                #     )

    return df_all


def fun_blending_general(run_sensitivity: bool, long_term: int,  df_all: pd.DataFrame, use_asterics: bool) -> pd.DataFrame:
    """
    General blending function that optionally adds an asterics '*' in the variable name of sensitivity analysis results
    (example: `Final Energy2050_BLEND*`) and concatenate with the standard results (without asterics). 
    
    - If `use_asterics` is True and `run_sensitivity` is enabled, sensitivity results are combined 
      with the standard results (sensitivity results include an asterisk in the variable name). 
    - If `use_asterics` is False, the function only returns sensitivity or standard results, depending 
      on the `run_sensitivity` flag, without adding asterisks to differentiate them.

    Parameters
    ----------
    run_sensitivity : bool
        Flag to indicate whether to run the sensitivity analysis.
    long_term : int
        Time frame for long-term convergence.
    df_all : pd.DataFrame
        The main dataframe containing the data to be blended.
    use_asterics : bool
        Flag indicating whether to use the asterics blending method or the regular one.

    Returns
    -------
    pd.DataFrame
        Blended dataframe with adjusted data based on sensitivity and convergence settings.
    """
    df_all = df_all.copy(deep=True)
    if use_asterics:
        # Add asterics in the variable name if they are coming from the sensitivity analysis and concatenate with starndard results (without asterics)
        return fun_blending_asterics(run_sensitivity, long_term, df_all)
    else:
        # Do not add asterics in the variable name. This is the regular blending function.
        conv_settings = fun_conv_settings(run_sensitivity)  # Get convergence settings
        return fun_blending(
            df_all,
            run_sensitivity,
            standard_range=fun_list_time_of_conv(conv_settings, var="Final"),
            secondary_primary_range=fun_list_time_of_conv(conv_settings, var="Secondary"),
            _to=long_term
        )


def fun_blending_asterics(run_sensitivity: bool, long_term: int,  df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Add asterics '*' in the variable name if they are coming from the sensitivity analysis and concatenate 
    with standard results (without asterics). Example. `Final Energy2050_BLEND*`

    Parameters
    ----------
    run_sensitivity : bool
        Flag to indicate whether to run the sensitivity analysis.
    long_term : int
        Time frame for long-term convergence.
    df_all : pd.DataFrame
        The main dataframe containing the data to be blended.

    Returns
    -------
    pd.DataFrame
        Blended dataframe, with an asterics symbol (`*`) in the variable name for the sensitivity-analysis results.
    """
    res2 = {}
    for x in [True, False]:
        conv_settings=fun_conv_settings(x)
        res2[x] = fun_blending(
            df_all,
            x,
            standard_range=fun_list_time_of_conv(conv_settings, var="Final"),
            secondary_primary_range=fun_list_time_of_conv(conv_settings, var="Secondary"),
            _to=long_term
        )

    # If sensitivity is enabled, mark sensitivity analysis results with asterisks (*) and concatenate with the standard results
    # Reason: we use different convergence settings for sensitivity analysis
    if run_sensitivity:
        res2[True].columns = [f"{col}*" for col in res2[True].columns]
        df_all = pd.concat(list(res2.values()), axis=1)
    else:
        df_all = res2[False]
    
    return df_all


def fun_from_step2_to_step1b_format_single_var(
    df_all: pd.DataFrame, var: str, reverse: bool = False
) -> pd.DataFrame:
    """Changes df format from step2 to step1b for a given variable `var` (e.g. ENSHORT_REF). If `reverse=True` it does
    the opposite (changes from step1b to step2 format)
 
    Parameters
    ----------
    df_all : pd.DataFrame
        Your dataframe
    var : str
        Your column variable (e.g. ENSHORt_REF)
    reverse : bool, optional
        Reverse operation, by default False
 
    Returns
    -------
    pd.DataFrame
        _description_
    """
    if reverse:
        return df_all.rename(
            {x: f"{x}{var}" for x in df_all.reset_index().VARIABLE.unique()}, axis=0
        )[var].unstack()
    short = pd.DataFrame(
        df_all[[x for x in df_all.columns if var in x]].stack()
    ).rename({0: var}, axis=1)
    short.index.names = ["TIME", "ISO", "VARIABLE"]
    return short.rename(
        {x: x.replace(var, "") for x in short.reset_index().VARIABLE.unique() if x.endswith(var)}, axis=0
    )
 
 
def fun_from_step2_to_step1b_format(
    df: pd.DataFrame,
    target: Optional[str],
    cols: list = ["ENSHORT_REF", "ENLONG_RATIO"],
    reverse: bool = False,
) -> pd.DataFrame:
    """Changes `df` format from step2 to step1b for all variables in `cols`. If `reverse=True` it does
    the opposite (changes from step1b to step2 format)
 
    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe
    target:str
        Scenario (e.g. 'h_cpol')
    cols : list, optional
        List of columns/variables, by default ['ENSHORT_REF', 'ENLONG_RATIO']
    reverse : bool, optional
        Reverse operation, by default False
 
    Returns
    -------
    pd.DataFrame
        Updated dataframe format
    """
    res = df.copy(deep=True)
   
    # Detect the `long_term` variable
    long_term=[x for x in cols if x!='ENSHORT_REF'][0]
    # Three lines below: rename `long_term` (variable name) as 'LONG_TERM'
    cols=[ 'LONG_TERM' if long_term in x else x  for x in cols]
    renamedict={x:x.replace(long_term,'LONG_TERM') for x in res.columns if x.endswith(long_term)}
    res= res.rename(renamedict, axis=1)
 
    rename_dict = {"VARIABLE": "SECTOR"}
   
    if reverse:
        res = res.droplevel("TARGET") ## this should be `res`!! Not `df`
        res = fun_rename_index_name(res, {v: k for k, v in rename_dict.items()})
        # Line below to be deleted, just for testing
        # fun_from_step2_to_step1b_format_single_var(res, cols[1], reverse=reverse)
    res = pd.concat(
        [
            fun_from_step2_to_step1b_format_single_var(res, x, reverse=reverse)
            for x in cols
        ],
        axis=1,
    )
    if not reverse:
        if target is None:
            raise ValueError("You need to provide a target if `reverse==False`")
        res = fun_rename_index_name(res, rename_dict)
        res["TARGET"] = target
        res = res.reset_index().set_index(["TIME", "ISO", "TARGET", "SECTOR"])
        # Two lines below: rename 'LONG_TERM'  as `long_term` (we go back to original `long_term` variable name)
        renamedict={x:long_term if 'LONG_TERM' in x else x  for x in res.columns}
    if reverse:
        renamedict={x:x.replace('LONG_TERM',long_term) if 'LONG_TERM' in x else x  for x in res.columns}    
    res=res.rename(renamedict, axis=1)
    return res

def fun_iamc_format_from_step2(df_all: pd.DataFrame) -> pd.DataFrame:
    """Returns data in IAMC format, starting from step2 data format.
    To revert data back to step2 format please use function `fun_step2_format_from_iamc`
    Parameters
    ----------
    df_all : pd.DataFrame
        Dataframe in step2 data format

    Returns
    -------
    pd.DataFrame
        Dataframe in IAMC format
    """
    if df_all.index.names != ["TIME", "ISO"]:
        raise ValueError(
            f"`df.index.names` should be ['TIME', 'ISO'], you provided {df_all.index.names} "
        )
    df_all_iamc = df_all.stack().unstack("TIME")
    df_all_iamc.index.names = ["ISO", "VARIABLE"]
    return df_all_iamc


def fun_step2_format_from_iamc(df_iamc: pd.DataFrame) -> pd.DataFrame:
    """Re-shape dataframe ising a step2 format, by taking a dataframe in IAMC format as input

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe in IAMC format

    Returns
    -------
    pd.DataFrame
        Dataframe in step2 data format
    """
    expected_idx_names = ["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"]
    if df_iamc.index.names != expected_idx_names:
        raise ValueError(
            f"`df.index.names` should be {expected_idx_names}, you provided {df_iamc.index.names} "
        )
    df_iamc = df_iamc.droplevel(["MODEL", "SCENARIO", "UNIT"])
    df_iamc = df_iamc.stack().unstack(level=1).reset_index()
    if "TIME" not in df_iamc.columns:
        df_iamc = df_iamc.rename({"level_1": "TIME"}, axis=1)
    return df_iamc.set_index(["TIME", "ISO"]).sort_index()


def fun_iamc_format_and_select_variables(
    model: str,
    target: str,
    region: str,
    df_all: pd.DataFrame,
    ra: int,
    file_name_iamc:str,
    str_to_find: str = "BLEND",
    random_electricity_weights:bool=False
) -> Union[pd.DataFrame, str]:
    """This function selects variables, and add an unit columns to the dataframe.
       It returns the updated dataframe with the selected variables and the the csv file name (which depends based on criteria weights)

    Args:
        random_electricity_weights (bool): False if we use default electricity criteria weights, otherwise True
        model (str): model
        target (str): scenario
        region (str): region
        df_all (pd.DataFrame): pd.DataFrame
        ra (int): random criteria weight seed
        file_name_iamc (str): Name of csv file

    Returns:
        Union[pd.DataFrame, str]: Dataframe with selected downscaled results and csv file name
    """

    # Selecting variables to be included
    df_long = df_all.iloc[:, df_all.columns.str.contains(str_to_find)]
    df_long = fun_pd_long_format(df_long)
    # Csv file name
    if random_electricity_weights:
        file_name_iamc = f"{file_name_iamc}_{ra}.csv"
    elif ".csv" not in file_name_iamc:
        file_name_iamc += ".csv"

    df_all = fun_pd_wide_format(df_long, region, model, target, "EJ/yr")
    df_all["UNIT"] = "EJ/yr"
    iamc_index = ["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"]
    setindex(df_all, iamc_index)
    return df_all


def fun_fuel_mix_secondary_and_final_by_sectors(
    liquids_gases_list: list,
    allowed_sub_sectors: list,
    df_iea_melt: pd.DataFrame,
    df_iam_all_models: pd.DataFrame,
    var_list: list,
    model: str,
    target: str,
    region: str,
    df_all: pd.DataFrame,
) -> pd.DataFrame:
    ## FUEL MIX at sectorial level-
    """This function calculates the fuel mix at the sectorial level.
        It returns the updated dataframe

        Calculations are based on the following steps:

        1) We calculate the secondary and final energy mix for solids, liquids, gases by using `fun_secondary_and_final`
        2) We apply the secondary energy mix to final sectors (industry, transportation, residential and commercial) by using `fun_apply_secondary_mix_to_final_sectors`
        3) Now that we have the fuel mix by sector, we do a sectoral harmonization using `fun_iterative_adj`.
        Example. 'Final Energy|Industry|Solids' as the sum of: 'Final Energy|Industry|Solids|BiomassENSHORT_REF' and 'Final Energy|Industry|Solids|CoalENSHORT_REF' in each country
        4) Now that we have the hamonized fuel mix at the Final Energy level, we convert the fuel mix to the Secondary energy level, by using fun_from_final_to_secondary_enshort_combi


    Args:
        liquids_gases_list (_type_): List of liquids, gases
        allowed_sub_sectors (_type_): list_of_fuels + list_of_ec + list_of_sectors
        df_iea_melt (_type_): IEA historical data
        df_iam_all_models (_type_): IAMs data
        var_list (_type_): list of variables
        model (_type_): model e.g. MESSAGE
        target (_type_): scenario e.g. h_cpol
        region (_type_): region e.g. MESSAGE|Western Eurepe
        df_all (_type_): Dataframe with downscaled IAMs results


    Returns:
        _type_: Dataframe with downscaled IAMs results
    """

    # NOTE: Secondary and final downscaling ENLONG (before ENSHORT calculations)
    #### With sector list
    sectors_list = [f"|{x}" for x in list_of_sectors]
    for var in var_list:
        for s in sectors_list:
            try:
                var_iam = "Final Energy" + s + "|Solids"
                # Below we create the fuels mix (biomass, coal) of solids.
                # example:
                # - 'Final Energy|Residential and Commercial|Solids|BiomassENLONG_RATIO'
                # - 'Secondary Energy|Residential and Commercial|Solids|BiomassENLONG_RATIO'

                fun_secondary_and_final(
                    var_iam,
                    model,
                    region,
                    target,
                    df_all,
                    df_iam_all_models,
                    allowed_sub_sectors,
                    df_iea_melt,
                    _var=var,
                )
            except Exception as e:
                print(e)
                print(traceback.format_exc())

        for ec in liquids_gases_list:
            var_iam = "Secondary Energy" + ec
            fun_secondary_and_final(
                var_iam,
                model,
                region,
                target,
                df_all,
                df_iam_all_models,
                allowed_sub_sectors,
                df_iea_melt,
                _var=var,
            )
            try:
                df_all = fun_apply_secondary_mix_to_final_sectors(
                    df_all, ec.replace("|", ""), var=var
                )
            except Exception as e:
                print(e)
                print(
                    f"*** fun_apply_secondary_mix_to_final_sectors did not work for {ec.replace('|','')} ***"
                )

    # Fuel Adjustments in final energy sectors:
    for var_iam in sectors_adj:
        try:
            # overwrite_total=False NEEDS TO BE FALSE OTHERWISE NOT WORKING
            # harmonizes fuels within sectors. Example. 'Final Energy|Industry|Solids' as the sum of:
            # 'Final Energy|Industry|Solids|BiomassENSHORT_REF'
            # 'Final Energy|Industry|Solids|CoalENSHORT_REF'
            df_all = fun_iterative_adj(
                model,
                region,
                target,
                var_iam,
                df_all,
                df_iam_all_models,
                allowed_sub_sectors,
                overwrite_total=False,
            )

            # print(df_all[var_iam + "ENSHORT_REF"])
        except Exception as e:
            print(e)
            print(traceback.format_exc())

    #  Converting Final Energy by fuel into Secondary Energy (Gases, Liquids, Solids)
    for var in var_list:
        for x in ["Gases", "Liquids", "Solids"]:
            df_all = fun_from_final_to_secondary_enshort_combi(
                f"Secondary Energy|{x}",
                model,
                region,
                target,
                df_all,
                df_iam_all_models,
                allowed_sub_sectors,
                _var=var,
            )

    return df_all


def combine_results(input_list, random_electricity_weights: bool) -> None:
    """Gets a results folder, combines the results on a per model basis & delets the folder."""
    RESULTS_DATA_DIR = (
        CONSTANTS.CURR_RES_DIR(str(Path(os.path.abspath("")) / Path(__file__).name))
        / "tmp"
    )

    if random_electricity_weights:
        ## We just make a copy of tmp in the uncertainty folder
        src_dir = RESULTS_DATA_DIR
        dest_dir = (
            CONSTANTS.CURR_RES_DIR(str(Path(os.path.abspath("")) / Path(__file__).name))
            / "uncertainty"
        )
        files = os.listdir(src_dir)
        shutil.copytree(
            src_dir, dest_dir
        )  ## we copy all files in an "uncertainty" directory

    else:
        models = {input_set["model_patterns"] for input_set in input_list}
        for m in models:
            i = 0
            for input_set in input_list:
                if input_set["model_patterns"] != m:
                    continue
                r = input_set["region_patterns"].replace("|", "").replace(m, "")
                t = input_set["target_patterns"]
                res_files = glob.glob(f"{str(RESULTS_DATA_DIR)}/{m}_{r}_{t}*.csv")
                if len(res_files) == 1:
                    with open(res_files[0], "r") as infile:
                        print(f"{m} {i} {Path(res_files[0]).name}")
                        lines = infile.readlines()
                        if i > 0:
                            lines = lines[1:]
                    res_file_name = Path(res_files[0]).name
                    res_file_name = (
                        f"{m}_{res_file_name[res_file_name.rfind(t)+len(t)+1:]}"
                    )
                    mode = (
                        "w" if i == 0 else "a"
                    )  # NOTE: Will overwrite file when you run a new simulation
                    with open(RESULTS_DATA_DIR.parent / res_file_name, mode) as outfile:
                        outfile.writelines(lines)
                    i += 1
                elif len(res_files) > 1:
                    res_files = glob.glob(
                        f"{str(RESULTS_DATA_DIR)}/{m}_{r}_{t}_{input_set['csv_out']}*.csv"
                    )  # we add the csv_out file
                    ## We write the file
                    with open(res_files[0], "r") as infile:
                        print(f"{m} {i} {Path(res_files[0]).name}")
                        lines = infile.readlines()
                        if i > 0:
                            lines = lines[1:]
                    res_file_name = Path(res_files[0]).name
                    res_file_name = (
                        f"{m}_{res_file_name[res_file_name.rfind(t)+len(t)+1:]}"
                    )
                    mode = (
                        "w" if i == 0 else "a"
                    )  # NOTE: Will overwrite file when you run a new simulation
                    with open(RESULTS_DATA_DIR.parent / res_file_name, mode) as outfile:
                        outfile.writelines(lines)
                    i += 1

                    if len(res_files) > 1:
                        raise ValueError(
                            f"Found more than 1 file (found {len(res_files)}) for {m}, {r}, {t}, skipping this one. Files in question: {res_files}."
                        )
                else:
                    print(f"Did not find any result files for {m}, {r}, {t}")


def run_step2(input_list, random_electricity_weights, n_jobs):

    """Run step 2"""

    print("CPUs available:", os.cpu_count())
    acual_cpus = min(n_jobs, os.cpu_count() - 1, len(input_list))
    print(f"CPUs Currently running: {acual_cpus}")

    ## Individual runs below:
    # for run in input_list:
    #     print(run)
    #     main(**run)

    RESULTS_DATA_DIR = (
        CONSTANTS.CURR_RES_DIR(str(Path(os.path.abspath("")) / Path(__file__).name))
        / "tmp"
    )

    if random_electricity_weights == True:  ## remove uncertainty directory
        UNCERTAINTY_DIR = RESULTS_DATA_DIR = (
            CONSTANTS.CURR_RES_DIR(str(Path(os.path.abspath("")) / Path(__file__).name))
            / "uncertainty"
        )

        if os.path.isdir(
            UNCERTAINTY_DIR
        ):  ## CHECK IF an uncertainty directiory exists:
            print("**********************************")
            confirm_delete_folder = input(
                "This simulation will delete the `uncertainty` folder. NOTE: the `tmp` folder will be NOT deleted. Are you sure you want to delete the `results/2_Primary_and_Secondary_Energy/uncertainty` folder? Please type y/n"
            )

            if confirm_delete_folder == "y":

                # UNCERTAINTY_DIR.mkdir(exist_ok=True) ## Creates a directory if it does not exist
                shutil.rmtree(
                    UNCERTAINTY_DIR, ignore_errors=True
                )  ## Remove directory (the line above is needed to avoid errors here if the directory does not exist)
            else:
                raise ValueError(
                    "Simulation aborted. You might consider making a copy of the `uncertainty` folder byefore runnig the simulation"
                )

    else:
        print("Deleting the `results/2_Primary_and_Secondary_Energy/tmp` folder...")
        shutil.rmtree(RESULTS_DATA_DIR, ignore_errors=True)  ## remove tmp directory

    ## Running simulatios in parallel
    Parallel(n_jobs=acual_cpus)(delayed(main)(**run) for run in input_list)
    # combine_results(input_list,random_electricity_weights=random_electricity_weights ) ## we take this out from the run step2 function




if __name__ == "__main__":
    project_file = "NGFS_2022"  # "NGFS"  ## directory in input data
    list_of_models = ["*MESSAGE*"]
    list_of_regions = ["*Pacific OECD*"]  # ["*"]  # ["*Europe*"]
    list_of_targets = ["h_cpol"]  # ["*"]  # ["h_cpol"]
    elc_trade_adj_input = True  ## Minimise trade True/False
    random_electricity_weights = False
    csv_out = (
        # "test_step_2_refactor_all_steps_remove_fun_final_enshort"
        "test_step_2_refactor_all_steps_remove_fun_final_enshort_all_steps"
        # "test_step_2_refactor_all_steps"  ## name of csv file (output of the simulation)
    )
    seed_range = range(701, 704)
    model_in_region_name: bool = True  ##NOTE If true we only scan for regions that contains the model name: example: 'REMIND-MAgPIE 2.1-4.2|Canada, NZ, Australia*'
    n_jobs = 1  # -1 means run the maximum number of parallel processes. for details see: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    use_step1b_harmo_data = True
    input_file = CONSTANTS.INPUT_DATA_DIR / project_file / "snapshot_all_regions.csv"

    # df_iam_all = fun_read_df_iam_all(
    #     file=InputFile(input_file), add_model_name_to_region=False
    # )

    df_iam_all = fun_read_df_iam_all_and_slice(
        list_of_models, list_of_targets, input_file
    )

    ## NOTE: please consider to add 'r' in the utils
    df_iam_all.loc[:, "REGION"] = [f"{str(i)}r" for i in df_iam_all.loc[:, "REGION"]]

    # Make input_list based on selection above
    input_list = make_input_data(
        project_file,
        list_of_models,
        list_of_regions,
        list_of_targets,
        random_electricity_weights,
        csv_out,
        elc_trade_adj_input,
        seed_range,
        model_in_region_name,
        df_iam_all,
        get_selection_dict,
        tc_max_range=[2200],
    )

    # Delete tmp folder and run step2
    run_step2(input_list, random_electricity_weights, n_jobs)

    # Combine results found in tmp folder
    combine_results(input_list, random_electricity_weights)

    try:
        selection_dict = get_selection_dict(
            InputFile(
                CONSTANTS.INPUT_DATA_DIR / project_file / "snapshot_all_regions.csv"
            ),
            model=list_of_models,
            region=list_of_regions,
            target=list_of_targets,
            variable=["*"],
        )
    except ValueError as e:
        raise ValueError(
            f"{e}\nConsider using the asterisk '*' for the pattern."
        ) from e

    for model in selection_dict:
        df = pd.read_csv(
            f"results/2_Primary_and_Secondary_Energy/{model}_{csv_out}.csv"
        )
        # we pick one blend (e.g. 2200)
        df = df[df.VARIABLE.str.contains("2200")]
        df.loc[:, "VARIABLE"] = [x.split("2")[0] for x in df.loc[:, "VARIABLE"]]

        df = fun_validation(
            CONSTANTS,
            Path("results/2_Primary_and_Secondary_Energy"),
            selection_dict,
            project_name=project_file,
            model_patterns=list_of_models,
            region_patterns=list_of_regions,
            target_patterns=list_of_targets,
            # vars=["Final Energy|Residential and Commercial|Heat"],
            # vars=["Final Energy|Industry|Solids|Biomass"],
            # vars=vars_to_be_harmo_step5,
            vars=vars_to_be_harmo_step2a,
            # vars = vars_to_be_harmo_step2a, # excl electricity,
            # vars=["Secondary Energy|Electricity|Coal"],
            pd_dataframe_or_csv_str=df,
            harmonize=False,
            save_to_csv=False,
            no_decimals=2,
            cols=[str(x) for x in range(2010, 2055, 5)],
        )

        # Harmonization test: 1) harmonize and replace results and 2) check if harmonized data match regional IAM results
        # for _ in range(2):
        #     df = fun_validation(
        #         CONSTANTS,
        #         Path("results/2_Primary_and_Secondary_Energy"),
        #         selection_dict,
        #         project_name=project_file,
        #         model_patterns=list_of_models,
        #         region_patterns=list_of_regions,
        #         target_patterns=list_of_targets,
        #         # vars=["Final Energy|Residential and Commercial|Heat"],
        #         vars=["Final Energy|Industry|Solids|Biomass"],
        #         # vars=vars_to_be_harmo_step5,
        #         pd_dataframe_or_csv_str=df,
        #         harmonize=True,
        #         save_to_csv=False,
        #         no_decimals=5,
        #     )

