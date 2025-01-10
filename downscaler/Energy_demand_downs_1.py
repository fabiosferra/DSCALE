import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import psutil
import scipy
from joblib import Parallel, delayed

from downscaler import CONSTANTS, IFT, plt
from downscaler.fixtures import (
    dict_y_den,
    gdp,
    iea_biomass,
    iea_coal,
    iea_flow_dict,
    iea_gas,
    iea_gases,
    iea_oil,
    pop,
    pop_iea,
    ref_countries,
    step1cols,
)
from downscaler.input_caching import get_selection_dict

## IMPORTING FUNCTIONS
from downscaler.utils import (  # harmo_ratio, add_region_simple, 
    InputFile,
    fun_convert_time,
    fun_pd_sel,
    fun_read_reg_value,
    make_left_list,
    make_optimal_list,
    set_bounds,
    setindex,
    unmelt,
    fun_historic_data,
    step1_process_region_countries,
    step1_process_region_and_dataframe,
    step1_process_model_data,
    step1_update_targets_with_ref,
    step1_process_ssp_scenario,
    step1_initialize_main_dataframe,
    step1_print_progress,
    step1_add_2100_gdp_pop_to_hist_data,
    step1_calculate_enlong,
    step1_fun_hist_ind,
    step1_harmo_summer,
    step1_bounds_harmo,
    step1_init_right_list_2100,
    step1_load_ssp_data,
    step1_load_iea_data,
    step1_get_models,
    step1_check_fuel_distribution,
    step1_get_sectors,
    find_countries_declining_gdpcap,
    step1_fun_enshort_ei,
    step1_short_term_calculations,
    optimal_ratio,
    step1_select_sectors,
    fun_xs,
    fun_flatten_list,
    fun_regional_country_mapping_as_dict,
    fun_check_missing_variables,
)

from downscaler.utils_pandas import fun_index_names
from downscaler.fixtures import (
    iea_gases,
    iea_gas,
    iea_oil,
    iea_coal,
    iea_biomass,
    iea_flow_dict,
    dict_y_den,
    ref_countries,
    gdp,
    pop,
    pop_iea,
    check_IEA_countries,
)
from downscaler.input_caching import get_selection_dict


def main(
    input_file: IFT = CONSTANTS.INPUT_DATA_DIR
    # / "snapshot_all_regions_round_1p4_2021_05_11.csv",  ## GCAM D_DELFRAG SCENARIO 2021_05_11
    / "snapshot_all_regions_round_1p4_2021_04_27.csv",
    max_print=2,  ## maximum number of graphs showed
    # func_type=["log-log",  'lin', 's-curve'],# , for the future check the s shape function, also check the lin again
    # func_type=["log-log", 's-curve'],# , for the future check the s shape function, also check the lin again
    # func_type="lin",
    func_type="log-log",  
    # func_type='s-curve',
    ssp_scenario="SSP2",  # NOTE: maybe better to use as a fixture
    ssp_model="OECD Env-Growth",  # NOTE: maybe better to use as a fixture
    iam_base_year=2010,  # NOTE: maybe fixture
    end_hist=2015,  ## end point of historical data  added 2021_03_20
    ref_target="h_cpol",  ## REFERENCE TARGET. This will be the first that we run. Why we need this: in some countries (countries_incl_2100) we need to include 2100 enlong for short term extraplolations as they have declining gdpcap (historically) => issue with regression
    target_patterns=["*h_cpol*"],
    scen_fx_ref_target=[
        "d_delfrag"
    ],  ## Special scenarios fixed to ref_target until 2030
    model_patterns=[
        "GCAM5.3_NGFS"
    ],  # NOTE: looks like this is currently not used, i.e. we just take all the models from the iam input data
    region_patterns=["*Latin*"],
    n_sectors=None,  # number of sectors to compute, if None use all
    convergence: int = 2200,
    show_graph: bool = False,
    file_suffix: str = "New_file_all_sectors",
    # pyam_mapping_file:str =  "default_mapping.csv",  # NOTE: mapping fixture
    pyam_mapping_file: IFT = (
        CONSTANTS.INPUT_DATA_DIR / "NGFS" / "default_mapping.csv"
    ),
    df_iam_all_input:pd.DataFrame=pd.DataFrame(),  # IAM data
    df_iea_h=None,
    regions_input=[],
    scen_dict=None,  # ssp/scenario mapping (for gdp and pop)
    split_res_and_comm=False,
):
    project=str(input_file.parent).split('\\')[-1]

    # Checking all fuels are distributed among energy carriers
    step1_check_fuel_distribution(iea_gases, iea_gas, iea_oil, iea_coal, iea_biomass)

    m = ["TIME", "ISO"]  # multi-index
    _folder=Path(os.path.abspath(""))
    RESULTS_DATA_DIR =  CONSTANTS.CURR_RES_DIR(_folder/ Path(__file__).name)

    pyam_mapping_file = CONSTANTS.INPUT_DATA_DIR / pyam_mapping_file
    historic_data_file = CONSTANTS.INPUT_DATA_DIR / "Historical_data.csv"
    iea_data_file = CONSTANTS.INPUT_DATA_DIR / "Extended_IEA_en_bal_2019_ISO.csv"
    iea_fuel_file = CONSTANTS.INPUT_DATA_DIR / "IEA_Fuel_dict.csv"  # NOTE: mapping fixture
    country_mapping_file = (CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv") # can be replaced by pyam_mapping_file
    
    # Load/process IEA data
    i=(df_iea_h, historic_data_file, country_mapping_file, iea_data_file, iea_fuel_file, gdp)
    df_iea_h, df_iea_all, df_countries = step1_load_iea_data(*i)

    # Load/process IAM data
    df_iam_all= df_iam_all_input

    # Getting % usage of virtual_memory ( 3rd field)
    print("RAM memory % used in energy_demand_downs:", psutil.virtual_memory()[2])
    
    # Load SSP data
    i=(input_file, CONSTANTS.INPUT_DATA_DIR, model_patterns, ssp_model, ssp_scenario, df_iam_all, pop, gdp)
    df_ssp,df_ssp_melt, change_ssp_model,change_ssp_scenario= step1_load_ssp_data(*i)

    # Load models
    models = step1_get_models(df_iam_all, model_patterns)

    # Print Local time
    seconds = time.time()
    local_time = time.ctime(seconds)
    print("Local time:", local_time)

    # Get sectors
    sectors =step1_get_sectors(df_iam_all,split_res_and_comm)

    # List of countrties with negative income per capita growth from 1980-2010. They require 2100 datapoint in short term projections
    countries_incl_2100 =find_countries_declining_gdpcap(df_iea_h, df_countries)

    # NOTE: For wildcard pattern matching re-define the models list with all that matches. for example ['REMIND*'] -> [REMIND-MAgPIE ..., ]
    if len(models)==1:
        model=models[0]  # NOTE: We only have one model in the list
    else:
        raise ValueError(f"We expect to have only one model in `models`, it contains {models}")

    # NOTE: we don't use this slice df_iam_all
    df_iam_all, regions, targets=step1_process_model_data(df_iam_all, model, df_countries, pyam_mapping_file, region_patterns, target_patterns)

    # Check regions lenght - if only one region, we skip it
    if len(regions)==1:
        r=regions[0]  # NOTE: We only have one model in the list
    else:
        raise ValueError(f"We expect to have only one model in `regions`, it contains {regions}")
    region, countrylist= step1_process_region_countries(r, df_countries)
    
    # Check countrylist lenght - if only one country, we skip this region
    if len(countrylist) == 1:
        print("*** Region contains one country: WE SKIP THIS REGION ****")
        return None
    
    # =======================
    # MAIN LOOP
    # =======================
    # Downscaling IAMs results (energy demand). We do this for all: Regions, Targets (scenarios), Sectors

    right_list_stored = []
    _print = True
    right_list_stored = []
    seconds0 = time.time()
    count_loop = 0

    # Create a list of function types to loop over
    func_type_list = [func_type] if isinstance(func_type, str) else func_type
    for func_type in func_type_list:  # func_type loop   
        region, df_iea_melt, df_iea_h, = step1_process_region_and_dataframe(model, region, df_iam_all, df_iea_all, df_iea_h, sectors, iea_flow_dict, gdp, pop_iea)
        ## We Put ref_target at beginning of the targets list:
        targets=step1_update_targets_with_ref(ref_target, targets)

        dfa_all=pd.DataFrame()
        for target in targets:
            df_ssp_melt= step1_process_ssp_scenario(scen_dict,
                        target,
                        ssp_scenario,
                        df_ssp,
                        change_ssp_model,
                        change_ssp_scenario,
                        model,
                        ssp_model)

            setindex(df_iam_all, False)  
            
            # CHECK if both GDP and Population data are present `in df_ssp_melt`
            for x in ['GDP|PPP', 'Population']:
                if x not in df_ssp_melt.reset_index().VARIABLE.unique():
                    raise ValueError(f'{x} not present in `df_ssp_melt`')
            
            # NOTE: 2010 is hardcoded here, to be changed to iam_base_year
            df_iam = df_iam_all.query('SCENARIO == @target and TIME >= 2010')

            # Create `df_main` (this will contain all the downscaled results). We initialize it with GDP/POP from the SSP database
            df_main = step1_initialize_main_dataframe(df_ssp_melt,countrylist, gdp,pop,m)
            
            # Update countrylist based on the new index
            countrylist = setindex(df_main, "ISO").dropna().index.unique().to_list()
            
            # Filter for available sectors:
            sectors = [x for x in sectors if x in df_iam.reset_index().VARIABLE.unique()]

            # Select sectors 
            sectors, last_s = step1_select_sectors(n_sectors, sectors)

            # NOTE: possible change the loop to for _s, s in enumerate(sectors)
            for _s in range(0, last_s, 1):  ## All sectors loop
                if count_loop >= max_print:
                    show_graph = False
                    _print = False

                # Print info
                step1_print_progress(target, models, targets, sectors, _s, last_s, region, regions, count_loop, seconds0)

                ## NOt harmonised results _harmo=False
                df_main = fun_pd_sel(df_main, "", "")
                
                # dropping duplicate columns  https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
                df_main = df_main.loc[:, ~df_main.columns.duplicated()]  

                # NOTE: add 2100 GDP and POP to `df_iea_h`
                df_iea_h=df_iea_h.reset_index().set_index(m)
                df_iea_h = step1_add_2100_gdp_pop_to_hist_data(df_ssp_melt, df_iea_h, 2100, countrylist, gdp, pop, pop_iea)
                setindex(df_iam, "TIME")

                print('Calculating ENLONG')
                # NOTE: Debugging information
                print("checking sectors:", sectors[_s]," and: ", dict_y_den[sectors[_s]] )

                ## Estimating alpha and beta below:
                df_main=step1_calculate_enlong(df_main, func_type, _s, region, gdp, pop, countrylist, df_iam, sectors, target)

                # NOTE: We plan to cache this
                # Here we create again Y_DEN, X_DEN
                df_iea_h=step1_fun_hist_ind(                         
                                    _s,
                                    df_iea_h,
                                    df_iea_melt,
                                    countrylist,
                                    ref_countries,
                                    sectors,
                                    iea_flow_dict,
                                    m,
                                    gdp,
                                    pop_iea,
                                    dict_y_den,
                                    df_main)


                if convergence != 2200:
                    filename = f"{region}_{file_suffix}_max-tc_{convergence}.csv"
                else:
                    filename = f"{region}_{file_suffix}.csv"
                filename = filename.replace(
                    "|", "_"
                )  ## Using CSV name separator that will work with Windows 2021_05_08

                if (
                    filename.find(model) == -1
                ):  ## This means the model name is not contained in the filename (csv file name to be saved):
                    filename = (
                        model + "_" + filename
                    )  ## We add the model name to the csv filename

                path_to_file = RESULTS_DATA_DIR / filename

                # NOTE: Is this just for the d_delfrag scenario? No, this is for all scenarios excluding h_cpol

                try:
                    setindex(dfa_all, "TIME")
                    try: 
                        dfa_all = dfa_all.drop("TIME").copy(deep=True)
                    except:
                        pass

                    dfa_all, orig_index = fun_convert_time(
                        dfa_all, int
                    )  ## conver time to integer

                    dfa = dfa_all[
                        (dfa_all.SECTOR == sectors[_s])
                        & (dfa_all.TARGET == ref_target)].copy(
                        deep=True)

                except:
                    dfa = pd.DataFrame()
                    dfa_all = pd.DataFrame()
                    print("NOTE: we do not read dfa_all")

                ## STEP 2 COPY SHORT-TERM DATA ALREADY AVAILABLE FROM ref_target scenario
                if (len(dfa) >= 1 and target != ref_target):  
                    # # If we found existing data for ref_target and  this is not a ref_target scenario
                    # print("We load ENSHORT data from", ref_target, "scenario")
                    # # Apply functions for data preparation
                    # fun_pd_sel(df_main, "", "")
                    # fun_pd_sel(dfa, "", "")
                    df_main, den_enshort, dfa, dfa_all=load_enshort_data(
                                    df_main,
                                    dfa,
                                    dfa_all,
                                    df_iea_melt,
                                    countrylist,
                                    sectors,
                                    ref_target,
                                    target,
                                    gdp,
                                    pop,
                                    _s,
                                    dict_y_den,
                    )
                    # Additional calculations for ENLONG
                    df_main = step1_calculate_enlong(
                        df_main, func_type, _s, region, gdp, pop, countrylist, df_iam, sectors, target
                    )

                    # Set the MAX_TC column based on target and sector data
                    setindex(dfa_all, "ISO")
                    df_main["MAX_TC"] = dfa_all.loc[
                        (dfa_all.SECTOR == sectors[_s]) & (dfa_all.TARGET == ref_target) & 
                        (dfa_all.index.isin(df_main.index.unique())), "MAX_TC"
                    ]
                    setindex(dfa, "ISO")

                else:  
                    # We run the historical trend regression (only for h_cpol (ref_target) scenario)
                    df_main=step1_short_term_calculations(countrylist, 
                                                        df_main, 
                                                        df_iea_h, 
                                                        df_iea_melt, 
                                                        sectors, 
                                                        dict_y_den, 
                                                        gdp, 
                                                        pop_iea, 
                                                        iam_base_year, 
                                                        _print, 
                                                        func_type, 
                                                        end_hist, 
                                                        target, 
                                                        ref_target, 
                                                        convergence, 
                                                        countries_incl_2100, 
                                                        # countries_incl_2100 if func_type == "log-log" else countrylist, 
                                                        dfa_all, 
                                                        dfa,
                                                        _s=_s,
                                                        t=2100)

                df_main["REGION"] = region

                ## COUNTRY-LEVEL HARMONISATION (to match IAM results at the regional level)
                setindex(df_main, m)
                if _s != 0:
                    df_main["ENLONG_RATIO"] = np.minimum(df_main["ENLONG_RATIO"], den_enshort)  
            

                # ENSHORT: filling missing data with ENLONG
                for i in ["ENSHORT"]:
                    df_main.loc[
                        df_main[i].fillna(0) == 0, i
                    ] = df_main.loc[
                        df_main[i].fillna(0) == 0, "ENLONG"
                    ]
                
                # Harmonize results
                mydict={"ENSHORT_RATIO":"ENSHORT", "ENLONG_RATIO":"ENLONG"}
                for k,v in mydict.items(): 
                    df_main[k] = step1_harmo_summer(df_main, v, _s, sectors, df_iam, region)

                # line below solves intergral calculations problems
                df_main.fillna(0, inplace=True)  

                # Apply max_tc convergence to ENSHORT (applies to all sectors). 
                # ENSHORT_INIT calculated by using convergence rules (convergence between short term and long term)
                df_main = fun_max_tc_convergence(df_main)

                df_main["ENSHORT_RATIO"] = step1_harmo_summer(
                    df_main, "ENSHORT", _s, sectors, df_iam, region,
                )

                # Create right list if it is empty
                if right_list_stored == []:
                    right_list = step1_init_right_list_2100(
                        df_main, gdp, pop, min(len(countrylist), 4)
                    )

                    left_list = make_left_list(
                        df_main, 100, right_list, gdp, region
                    )  ## make left list
                    df_main = set_bounds(
                        df_main, 1.1, 0.9
                    )  ## add bounds
                    print("========================")
                    print("Finding Optimal list...")
                    print(f'Len of right list: {len(right_list)}')
                    countrylist = make_optimal_list(
                        df_main,
                        left_list,
                        right_list,
                        100,
                        region,
                        _threshold=0.75,
                        all_permutations=True,
                    )  # finding optimal country_list
                    right_list_stored = countrylist[-len(right_list) :]

                    print(
                        "This is the optimal right list: ",
                        right_list_stored,
                    )
                    print("This is our countrylist: ", countrylist)

                # Create left list (this is sector dependent)
                df_main["GDP"] = df_main[gdp]
                setindex(df_main, "ISO")
                df_main["GDPCUM"] = [
                    df_main.loc[c].GDP.sum() for c in df_main.index
                ]  # calculating cumulative GDP

                if (target == ref_target): 
                    df_main["ENSHORT_INIT"] = df_main["ENSHORT"].copy(deep=True)  
                left_list = make_left_list(df_main, 100, right_list, gdp, region)

                # Make countrylist as [left_list + right_list_stored]
                countrylist = (left_list + right_list_stored)

                # adding bounds (creating UPPER/LOWER bounds columns based on enshort_ratio, enshort_init)
                df_main = set_bounds(df_main, 1.1, 0.9)

                if target != ref_target and _s != 0:
                    setindex(df_main, m)
                    den_enshort = fun_pd_sel(
                        dfa_all[
                            (dfa_all.TARGET == target)
                            & (
                                dfa_all.SECTOR
                                == dict_y_den[sectors[_s]]
                            )
                        ],
                        "",
                        "",
                    )["ENSHORT_REF"].astype(float)
                    df_main["Y_DEN"] = den_enshort
                    setindex(
                        df_main, "ISO"
                    )  ## put it back original index

                
                # Find the optiomal ratio based on the countrylist
                df_main, opt_ratio, minvalue = optimal_ratio(
                    df_main,
                    countrylist,
                    region,
                    sectors[_s],
                    target,
                    _opt_ratio_exogenous=False,
                    _save_graph=False,
                    _show_graph=show_graph,
                )  ## Finding optimal R ratio


                print(
                    "Optimal stored list, and optimal ratio stored: ",
                    right_list_stored,
                    opt_ratio,
                )

                df_main["NUM_SHORT"] = df_main.ENSHORT_REF
                df_main["NUM_LONG"] = df_main.ENLONG_RATIO

                # BOUNDS ON ENSHORT_REF + HARMONISATION    
                for i in range(0, 3, 1):  
                    # we do this multiple times  to make sure all bounds are satisfied
                    try:
                        step1_bounds_harmo(
                            df_main, "ENSHORT_REF", "", "Y_DEN", _s, df_iam, region, sectors
                        )
                    except Exception as e:
                        print(f"Function step1_bounds_harmo did not work!!!!!! Error: {e}")

                # ======================================
                # VISUALISATIONS
                # ======================================
                setindex(df_main, False)

                if count_loop <= max_print:
                    # df_wide = unmelt(
                    #     df_main, "ENSHORT_REF"
                    # )  # from long to wide format to make staked graph

                    time_iam = df_iam.index.get_level_values(0).tolist()



                # ======================================
                # SPECIAL SCENARIOS: scenarios fixed to
                # ref_target (baseline) until 2030.
                # 2021_03_11
                # ======================================

                if target in scen_fx_ref_target:
                    fun_pd_sel(dfa_all, "", "")
                    fun_pd_sel(df_main, "", "")
                    for v in ["ENLONG_RATIO", "ENSHORT_REF"]:
                        ## Step 1) Fixing scenario to ref_target
                        df_main.loc[2010:2030, [v]] = (
                            dfa_all[
                                (dfa_all.SECTOR == sectors[_s])
                                & (dfa_all.TARGET == ref_target)
                            ]
                            .loc[2010:2030, v]
                            .astype(float)
                        )

                        ## Step 2) harmonisation below to match IAM results
                        ratio = (
                            df_main.loc[:, v]
                            / df_main.loc[:, v].groupby("TIME").sum()
                        )
                        df_main.loc[:, v] = (
                            ratio
                            * fun_read_reg_value(
                                model,
                                region,
                                target,
                                sectors[_s],
                                df_iam_all,
                            )["VALUE"]
                        ).loc[2010:]
                else:  ## else We Consistency of base year data ascross scenarios (added 2021_03_15).It would be nice to remove the two steps below (and calculate ENSHORT_REF in such a way that keeps consistency with 2010 data across scenarios)
                    ## STEP 1) WE ADJUST ENSHORT_REF TO MATCH ENSHORT_RATIO IN 2010 (THIS MEANS THAT  if historical data do not match IAM IN 2010, we adjust all countries proportionally by the same %)
                    ## In this manner 2010 data will not change across scenarios
                    fun_pd_sel(df_main, "", "")  ## set correct index
                    enshort_ref_2010_adj = (
                        df_main.loc[2010, "ENSHORT_RATIO"]
                        / df_main.loc[2010, "ENSHORT_REF"]
                    )
                    df_main["ENSHORT_REF"] = (
                        df_main["ENSHORT_REF"] * enshort_ref_2010_adj
                    )  # .plot()
                    ## STEP 2) Introduce again upper (maximum) value to enshort_ref
                    ## Step 1 Selecting the denominator (that is sector and  scenario dependent)
                    if _s == 0:
                        den_enshort = fun_pd_sel(df_main, "", "")[
                            gdp
                        ].astype(float)

                    else:
                        ## Contrary to dfa (that contains 1 target and 1 sector), dfa_all contains all targets and sectors
                        den_enshort = fun_pd_sel(
                            dfa_all[
                                (dfa_all.TARGET == target)
                                & (
                                    dfa_all.SECTOR
                                    == dict_y_den[sectors[_s]]
                                )
                            ],
                            "",
                            "",
                        )["ENSHORT_REF"].astype(float)

                    if _s != 0:
                        df_main["ENSHORT_REF"] = np.minimum(
                            den_enshort, df_main["ENSHORT_REF"]
                        )

                    enshort_ref_match_IAM = (
                        df_main["ENSHORT_RATIO"].groupby("TIME").sum()
                        / df_main["ENSHORT_REF"].groupby("TIME").sum()
                    )
                    df_main["ENSHORT_REF"] = (
                        df_main["ENSHORT_REF"] * enshort_ref_match_IAM
                    )

                # ======================================
                # SAVE DATA (SELECTED DATA) in CSV format
                # ======================================
                for i in ["ENLONG_RATIO", "ENSHORT_REF", "ENSHORT_RATIO", "ENSHORT_INIT"]:
                    df_main[f"{sectors[_s]}_{i}"] = df_main[i]


                df_main["TARGET"] = target
                df_main["SECTOR"] = sectors[_s]
                df_main["FUNC"] = func_type
                df_main["OPT_RATIO"] = opt_ratio
                df_main["COUNTRYLIST"] = str(countrylist)
                setindex(df_main, m)
                fixcols=[gdp]+step1cols+['fit_func']
                df_to_save = df_main.loc[:,fixcols]  

                df_to_save = fun_pd_sel(df_to_save, time_iam, "")  

                # SAVING CSV FILE
                if (
                    _s == 0 and target == ref_target and func_type == func_type_list[0]
                ):  # (this is equivalent to => targets.index(target)==0): ## We print a new csv file (is sector=0 and first target in the loop (target == targets[0]))
                    #                     df_to_save.to_csv(region+"_"+file_suffix+'.csv')
                    df_to_save.to_csv(path_to_file)

                else:  ## we append the cvs file for each sector
                    #                     df_to_save.to_csv(region+"_"+file_suffix+'.csv', mode='a', header=True)
                    df_to_save.to_csv(
                        path_to_file, mode="a", header=False
                    )
                # Saving `dfa_all` results. We want to save all results from `ref_target` and  `target`
                if len(dfa_all)>0:
                    dfa_all=  pd.concat([dfa_all, df_to_save.reset_index().set_index(dfa_all.index.names)])
                else:
                    dfa_all=  pd.concat([dfa_all, df_to_save.reset_index().set_index('TIME')])
                dfa_all=dfa_all[dfa_all.TARGET.isin([ref_target, target])]
                # Just store relevant sectors (the one present in dict_y_den)
                # dfa_all=dfa_all[dfa_all.SECTOR.isin(list(set(dict_y_den.values())))]

                count_loop = count_loop + 1

def fun_max_tc_convergence(df_main):
    _orig_index = df_main.index.names
    setindex(df_main, "ISO")
    df_main["MAX_TC"] = df_main["MAX_TC"].astype(float)
    df_main["ENSHORT"] = df_main["ENSHORT"].astype(float)
    df_main["CONV_WEIGHT"] = (
                    (df_main.TIME - df_main["MAX_TC"].astype(float))
                    / (2010 - df_main["MAX_TC"].astype(float))
                ).clip(0, 1)
    
    ## 2021_03_23 h 15.30 non linear weight depending on beta (the highest is the beta, the faster will be the convergence)
    df_main["CONV_WEIGHT"] = df_main["CONV_WEIGHT"] ** (df_main["BETA"].clip(1, np.inf))

    df_main["ENSHORT_HIST"] = df_main["ENSHORT"].copy(deep=True)
    df_main["ENSHORT"] = (
        df_main["ENSHORT"]
        * df_main["CONV_WEIGHT"].astype(float)
        + df_main["ENLONG_RATIO"]
        * (1 - df_main["CONV_WEIGHT"].astype(float))
    ).copy(deep=True)

    setindex(
        df_main, _orig_index
    )  ## Put back original index

    return df_main




def run_step1(input_list, n_jobs):
    """Run step 1"""
    ## NOTE We do not parallelise for targets as we need a precise order (h_cpol should come first)
    # input_list = [{"input_file": input_file, "model_patterns": m, "region_patterns": f"{r}*", "convergence": conv, "file_suffix":file_suffix, "n_sectors": n_sectors} for m, sel in selection_dict.items() for r in sel["regions"] for conv in conv_range
    # if r.find('R5')==-1] ## This line excluds R5 regions

    # n_jobs=8

    print("max number of CPUs:", os.cpu_count())
    print(input_list)
    print(len(input_list))
    acual_cpus = min(n_jobs, os.cpu_count() - 1, len(input_list))

    print("We run it with ", acual_cpus, " CPUs")

    Parallel(n_jobs=acual_cpus)(delayed(main)(**run) for run in input_list)


def load_enshort_data(
    df_main: pd.DataFrame,
    dfa: pd.DataFrame,
    dfa_all: pd.DataFrame,
    df_iea_melt: pd.DataFrame,
    countrylist: List[str],
    sectors: List[str],
    ref_target: str,
    target: str,
    gdp: str,
    pop: str,
    _s: int,
    dict_y_den: dict,
) -> pd.DataFrame:
    """
    Load ENSHORT data based on a given scenario, updating parameters for calculations 
    of energy intensity and setting specific values for different sectors and countries.

    Parameters
    ----------
    df_main : pd.DataFrame
        Main DataFrame where ENSHORT calculations are updated.
    dfa : pd.DataFrame
        DataFrame with initial ENSHORT data and reference values for selected columns.
    dfa_all : pd.DataFrame
        DataFrame containing data for all targets and sectors.
    df_iea_melt : pd.DataFrame
        DataFrame with historic energy data for multiple sectors and countries.
    countrylist : list[str]
        List of countries in the region of interest.
    sectors : list[str]
        List of sector names relevant to the calculations.
    ref_target : str
        Reference scenario target name.
    target : str
        Target scenario name.
    gdp : str
        Column name representing GDP in the data.
    pop : str
        Column name representing population in the data.
    _s : int
        Index of the sector currently being processed.
    dict_y_den : dict
        Dictionary mapping sectors to their respective denominator for calculations.

    Returns
    -------
    pd.DataFrame
        Updated df_main DataFrame with processed ENSHORT data for the given scenario.
    """

    m=["TIME", "ISO"]
    print("We load ENSHORT data from ",ref_target," scenario",)
    fun_pd_sel(df_main, "", "")
    fun_pd_sel(dfa, "", "")

    # Initialize ENSHORT values in df_main
    df_main["ENSHORT_INIT"] = dfa["ENSHORT_INIT"].astype(float)
    df_main["ENSHORT"] = dfa["ENSHORT_INIT"].astype(float)

    # Copy additional parameters from `dfa`
    for col in ["R_SQUARED", "ALPHA", "BETA", "HIST_START_YEAR", "HIST_END_YEAR"]:
        df_main[col] = dfa[col].astype(float)

    # Check for invalid ALPHA values
    if len(df_main[df_main["ALPHA"] == -np.inf]) >= 2:
        print("*** Problem with ENSHORT_REF: Check ALPHA values")

    
    ## Historical 2010 data for this sector (all countries)
    # NOTE: Replace hard-coded 2010 with variable
    hist_all = fun_historic_data(
        sectors[_s],
        countrylist,
        df_iea_melt,
        as_percentage=False,
        sum_countries=False,
    ).loc[2010]

    # Calculating enshort and ENSHORT INIT   in 3 steps:
    # Step 1 Selecting the denominator (that is sector and  scenario dependent)
    if _s == 0:
        den_enshort = fun_pd_sel(df_main, "", "")[
            gdp
        ].astype(float)

    else:
        ## Contrary to dfa (that contains 1 target and 1 sector), dfa_all contains all targets and sectors
        den_enshort = fun_pd_sel(
            dfa_all[
                (dfa_all.TARGET == target)
                & (
                    dfa_all.SECTOR
                    == dict_y_den[sectors[_s]]
                )
            ],
            "",
            "",
        )["ENSHORT_REF"].astype(float)


    ## Step 2 Calculating indicator (e.g. energy intensity) based on parameters (same of ref_target)
    setindex(df_main, "ISO")
    df_main["HIST_VALUE"] = hist_all.clip(1e-7, np.inf) 

    setindex(df_main,m)
    alpha_all = (
        np.log(
            (df_main["HIST_VALUE"] / den_enshort).clip(
                0, 1
            )
        )
        - df_main["BETA"]
        * np.log(df_main[gdp] / df_main[pop]).loc[2010]
    ).copy(deep=True)

    # We need Only Calibrated alpha in 2010 (otherwise alpha will change dyanmically over time) 2021_03_17
    setindex(df_main, "ISO")
    df_main["ALPHA"] = alpha_all.loc[2010]

    if len(df_main[df_main["ALPHA"] == -np.inf]) >= 2:
        print("*** PROBLEM with ENSHORT_REF line 896")

    setindex(df_main, m)
    
    df_main["ENSHORT"] = df_main["ENSHORT_INIT"].copy(deep=True)

    return df_main, den_enshort, dfa, dfa_all



if __name__ == "__main__":
    # project_file ="NGFS" ## directory in input data
    # list_of_models = ["*MESSAGE*"]
    # list_of_regions = ["*Europe*"]
    # list_of_targets = ["h_cpol"]

    # file_suffix = "New_file_all_sectors"
    project_file = "SSP"  ## 'NGFS' / 'SSP' directory in input data
    list_of_models = ["*AIM_CGE*"]  ## WITH IMAGE R5 IT WORKS!!!!!!!!
    list_of_regions = ["*XLM*"]
    # list_of_targets = ["h_cpol"]
    list_of_targets = [
        "*SSP3-70 (Baseline)*"
    ]  ## need to use this also for final energy
    ref_target = "SSP3-70 (Baseline)"

    # file_suffix = "New_file_all_sectors"
    file_suffix = "SSP_Downscaling_PROVA"
    n_sectors = None  ## If None All sectors included
    # n_sectors=2
    default_max_tc_conv = True  ## Use Default parameter
    conv_range = [2200]  ## default parameter = 2200
    # conv_range = range(2150, 2200, 25) ## default parameter = 2200

    # pyam_mapping_file = "default_mapping.csv"

    input_file = CONSTANTS.INPUT_DATA_DIR / project_file / "snapshot_all_regions.csv"
    pyam_mapping_file = CONSTANTS.INPUT_DATA_DIR / project_file / "default_mapping.csv"

    selection_dict = get_selection_dict(
        InputFile(input_file), list_of_models, list_of_regions
    )

    if default_max_tc_conv == True:
        conv_range = [2200]

    # ## NOTE We do not parallelise for targets as we need a precise order (h_cpol should come first)
    input_list = [
        {
            "input_file": input_file,
            "model_patterns": m,
            "region_patterns": f"{r}*",
            "convergence": conv,
            "file_suffix": file_suffix,
            "n_sectors": n_sectors,
            "pyam_mapping_file": pyam_mapping_file,
            "ref_target": ref_target,
            "target_patterns": list_of_targets,
        }
        for m, sel in selection_dict.items()
        for r in sel["regions"]
        for conv in conv_range
        # if r.find('R5')==-1
    ]  ## This line excluds R5 regions

    n_jobs = 8

    # print('max number of CPUs:',os.cpu_count())
    # print(input_list)
    # print(len(input_list))
    # acual_cpus=min(n_jobs, os.cpu_count()-1, len(input_list))
    # print('We run it with ',acual_cpus, ' CPUs')
    # Parallel(n_jobs=acual_cpus)(delayed(main)(**run) for run in input_list)

    run_step1(input_list, n_jobs)
