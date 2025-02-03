# Wrapper
import os
import logging
from pathlib import Path
from typing import Union, Optional, List
import pandas as pd
from joblib import Parallel, delayed

import downscaler
from downscaler import (
    CONSTANTS,
    CCS_CO2_GDP_Prices_January_3,
    Energy_demand_downs_1,
    Energy_demand_sectors_harmonization_1b,
    Policies_emissions_4,
    Primary_Energy_2a,
    Step_5_Scenario_explorer_5,
    Step_5b_emissions_by_sector_5b,
    Step_5c_Non_CO2_emissions_by_sector_5c,
    Step_5cbis_Secondary_energy_and_imports_5c_bis,
    AFOLU_emissions,  # step 5c tris
    Step_5d_aggregated_regions_5d,
    Step_5e_historical_harmo,
)
from downscaler.input_caching import get_selection_dict
from downscaler.utils_dictionary import fun_append_list_of_dicts
from downscaler.utils_pandas import fun_rename_index_name
from downscaler.utils import (
    InputFile,
    fun_finalize_step2_all_targets,
    fun_flatten_list,
    fun_make_step1_input,
    fun_read_config_file,
    fun_read_df_iam_all_and_slice,
    match_any_with_wildcard,
    fun_countrylist,
    fun_eu27,
    fun_read_df_iams,
    fun_regional_country_mapping_as_dict,
    fun_get_models,
    fun_wildcard,
    fun_create_var_as_sum,
    fun_current_file_suffix,
    fun_get_iam_regions_associated_with_countrylist,
    fun_invert_dictionary
)
from downscaler.fixtures import (
    fast_list,
    med_list,
    slow_list,
    sel_plot_vars_step5e,
    step5e_harmo,
    countries_w_2030_emi_policies,
    countries_w_2050_emi_policies,
    timing_net_zero,
    all_countries,
)

conv_dict = {
    "slow": slow_list,
    "med": med_list,
    "fast": fast_list,
}


def main(
    project_folder: str = "NGFS_2022",
    func_type: List[str] = ["log-log"],
    file_suffix: str = "EU_ab_2022_12_22",
    list_of_models=["*"],
    list_of_regions=["*"],
    list_of_targets=["*"],
    ref_target="",  # "SSP2-Baseline"  ## reference target (for short trend projections). This will be put automatically as the first target to run in step 1
    ngfs_2021_ghg_data=["MESSAGEix-GLOBIOM 1.0", "o_1p5c"],
    n_sectors=None,  ## all sectors (otherwise cannot run primary energy)
    add_gdp_pop_data=False,
    model_in_region_name: bool = True,  ##NOTE If true we only scan for regions that contains the model name: example: 'REMIND-MAgPIE 2.1-4.2|Canada, NZ, Australia*'
    add_twn=False,
    _gdp_pop_down=False,
    run_step5_without_policies=True,
    run_sensitivity=False,
    run_sensitivity_from_step2_to_5=False,
    conv_dict=conv_dict,
    country_marker_list=None,  # this one is needed for AFOLU
    step1=False,
    step1b=False,
    # Step1c = True  # add native countries
    step2=False,
    step2_pick_one_pathway=False,
    step3=False,
    read_from_step5e=True,  # run policy based on step5e results
    step5=False,
    run_step5_with_policies=False,
    step5b=False,  # (optional) Emissions by sectors and revenues downscaling
    step5c=False,
    step5c_bis=False,
    step5c_tris=False,
    step5d=False,
    step5e=False,
    step4=False,  # (optional)
    step5e_after_policy=False,
    step6=False,
    default_ssp_scenario="SSP2",
    gdp_model="NGFS",
    pop_model="IIASA-WiC POP",
    harmonize_eea_data_until=2018,
    countrylist=fun_eu27(),
    aggregated_region_name="EU27",
    n_jobs=6,  #  -1 means run the maximum number of parallel processes. for details see: https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    # Using averahe Last 10 years for indirect emissions means that _> all countries will have negative Indirect emissions at the base year except for Egypy and Lesotho
    indirect_emi_average: Union[list, range] = range(2010, 2021),
    grassi_dynamic: bool = True,
    grassi_scen_mapping: Union[None, dict] = None,
    co2_energy_only=False,
    countries_w_2030_emi_policies: list = countries_w_2030_emi_policies,
    countries_w_2050_emi_policies: list = countries_w_2050_emi_policies,
    timing_net_zero: dict = timing_net_zero,
    harmonize_hist_data: bool = True,
    split_res_and_comm: bool = False,
    input_arguments: Union[dict, None] = None,
    long_term='ENLONG_RATIO',
    method='wo_smooth_enlong',
    ol:List[str] = ["GDPCAP", "TIME"], # for step1b (additional long-term variants)
    tcl:List[int] = [2050, 2100], # for step1b (additional long-term variants)
    known_issues: Optional[dict]=None, # Negative energy variables (at regional level). e.g. {'SHAPE_2023':{'IMAGE 3.3': {'IMAGE 3.3|SAF': 'Primary Energy|Oil|w/ CCS'}}}
    resolve_inconsistencies_in_step1b:bool =False,
    add_only_revenues:bool=True,
    keep_step5_emissions:Optional[bool]=None,
    random_electricity_weights = False,
    # aggregate_non_iea_countries=True, # after step4
    **kwargs,
):
    if file_suffix is None:
        if not step1:
            raise ValueError('Please provide a file_suffix, if you do not want to run the downscaling from step1')
        file_suffix=fun_current_file_suffix()
    # defaults value
    ngfs_2021_ghg_data = []  # default value is an empty list

    downscaler.USE_CACHING = True  # kept here for bookkeeping, defaults to True
    ## Select steps

    ## STEP1 FINAL ENERGY PARAMETERS
    default_max_tc_conv = True  ## Use Default parameter
    conv_range = [2300, 2100]  ## default parameter = 2200

    ## STEP2 PRIMARY ENERGY PARAMETERS

    use_step1b_harmo_data = True
    elc_trade_adj_input = True  ## Minimise trade True/False
    seed_range = range(701, 703, 1)

    ## Other parameters below - do not modify
    input_file = CONSTANTS.INPUT_DATA_DIR / project_folder / "snapshot_all_regions.csv"
    pyam_mapping_file = (
        CONSTANTS.INPUT_DATA_DIR / project_folder / "default_mapping.csv"
    )
    
    try:
        # Get models based on `list_of_models`
        f = fun_wildcard
        project=project_folder
        models=list_of_models
        models_all = fun_get_models(project) if project else None
        models = f(list_of_models, models_all)
        

        # We check wheter to interpret `list_of_regions` as IAMs regions or ISO code:
        is_region=len([x for x in list_of_regions if x not in all_countries])>0
        if not is_region:
            pass
            res={}
            for model in models:
                res[model]=fun_get_iam_regions_associated_with_countrylist(project, list_of_regions, model)
            #     res[model]=fun_regional_country_mapping_as_dict(model, project_folder)
            list_of_regions=[list(fun_invert_dictionary({k:[v] for k,v in x.items()}).keys()) for x in list(res.values())]
            list_of_regions=fun_flatten_list(list_of_regions)
            list_of_regions=list(set([f'*{x[:-1]}*' for x in list_of_regions]))
    except Exception as e:
        print(e)    
    selection_dict = get_selection_dict(
            InputFile(input_file), list_of_models, list_of_regions
        )
    csv_out = file_suffix  ## name of csv file (output of the simulation)
    if default_max_tc_conv:
        conv_range = [2200]

    df_iam_all = fun_read_df_iam_all_and_slice(
        list_of_models, list_of_targets, input_file
    )

    if split_res_and_comm:
        s_required = ["Final Energy|Residential", "Final Energy|Commercial"]
        df = df_iam_all.set_index(["SCENARIO", "MODEL"])
        miss_dict = []
        for m in df_iam_all.MODEL.unique():
            for s in df.xs(m, level="MODEL").reset_index().SCENARIO.unique():
                for x in s_required:
                    if x not in df.loc[s, m].reset_index().VARIABLE.unique():
                        miss_dict = miss_dict + [f"{m} - {s}"]
        if len(miss_dict):
            txt = "in all regional IAMs scenarios - we will not split residential and commercial sectors"
            print(f"\nCannot find {s_required} {txt}. Missing in {set(miss_dict)}\n")
            split_res_and_comm = False

    list_of_allowed_regions = [x for x in df_iam_all.REGION.unique() if type(x) is str]
    df_iam_all = df_iam_all[df_iam_all.REGION.isin(list_of_allowed_regions)]
    reg_dict = {i: f"{i}r" for i in list_of_allowed_regions}
    df_iam_all = df_iam_all.replace(reg_dict)
    ## NOTE: please consider to add 'r' in the utils
    # df_iam_all.loc[:, "REGION"] = [f"{str(i)}r" for i in df_iam_all.REGION]

    ## Reading df_iea_h
    FIXTURE_DIR = CONSTANTS.INPUT_DATA_DIR  # Path(__file__).parents[1] / "input_data"
    historic_data_file = FIXTURE_DIR / "Historical_data.csv"
    if step1:
        df_iea_h = pd.read_csv(
            historic_data_file,
            index_col=["ISO"],
            sep=",",
            encoding="utf-8",
        )
    else:
        df_iea_h=pd.DataFrame()
    if project_folder == "CDLINKS_JARMO":
        from fixtures import fast_list, slow_list, med_list

        conv_dict = {
            "slow": slow_list,
            "med": med_list,
            "fast": fast_list,
        }
    else:
        config_file_path = FIXTURE_DIR / project_folder / "scenario_config.csv"
        if os.path.isfile(config_file_path):
            conv_dict, scen_dict, sensitivity_dict = fun_read_config_file(
                config_file_path
            )
        else:
            # conv_dict=None
            sensitivity_dict = None
            scen_dict = None

    from downscaler.utils import fun_read_df_iam_all, fun_regions

    def fun_regions_model(m, pyam_mapping_file, list_of_regions, model_in_region_name):
        regions = [
            r
            for r in fun_regions(m, project_folder)
            if match_any_with_wildcard(r, list_of_regions)
        ]

        if model_in_region_name:
            regions = [f"{str(m)}|{str(i)}" if i.find(m) == -1 else i for i in regions]
        return regions


    input_list = fun_make_step1_input(
        conv_range,
        list_of_regions,
        list_of_targets,
        ref_target,
        file_suffix,
        model_in_region_name,
        input_file,
        pyam_mapping_file,
        selection_dict,
        n_sectors,
        df_iam_all,
        df_iea_h,
        fun_regions_model,
        default_ssp_scenario,
        split_res_and_comm,
        scen_dict=scen_dict,
    )
    # Add `func_type` and  to input_list
    input_list=[fun_append_list_of_dicts([run, 
                                          {'func_type':func_type}]
                                          ) for run in input_list]
    # Add `sectors` to input_list
    input_list=[fun_append_list_of_dicts([run, 
                                          {'n_sectors':n_sectors}]
                                          ) for run in input_list]
    # Block below checks if ref_target is in the scenarios list
    targets = fun_flatten_list(
        [input_set["target_patterns"] for input_set in input_list]
    )
    scenarios = df_iam_all.SCENARIO.unique()
    targets = fun_wildcard(targets, scenarios)
    if step3 and _gdp_pop_down and ref_target not in targets:
        raise ValueError(
            f"The {ref_target} ref_target is missing in your selected `list_of_targets`: {list_of_targets}. "
            "Please add it to your `list_of_targets` if you want to downscale GDP (this is our reference scenario for downscaling GDP."
        )

    if step1:  # RUN STEP 1
        if ref_target is not None and ref_target not in list(scenarios) + [""]:
            step1txt_error = "was not selected among the list of scenarios, or it not present in the regional IAMs dataframe."
            raise ValueError(
                f"Your `ref_target` {ref_target} {step1txt_error} List of scenarios selected/available: {scenarios}"
            )
        print("max number of CPUs:", os.cpu_count())
        print(input_list)
        print(len(input_list))

        acual_cpus = min(n_jobs, os.cpu_count() - 1, len(input_list))
        print("We run it with ", acual_cpus, " CPUs")

        Parallel(n_jobs=acual_cpus)(
            delayed(Energy_demand_downs_1.main)(**run) for run in input_list
        )

    if step1b:
        Energy_demand_sectors_harmonization_1b.main(
            project_folder,
            file_suffix,
            region_patterns=list_of_regions,
            model_patterns=list_of_models,
            target_patterns=list_of_targets,
            max_iter=5,
            conv_dict=conv_dict,
            model_in_region_name=model_in_region_name,
            run_sensitivity=run_sensitivity,
            split_res_and_comm=split_res_and_comm,
            ol= ol, # for step1b (additional long-term variants)
            tcl = tcl, # for step1b (additional long-term variants)
            resolve_inconsistencies=resolve_inconsistencies_in_step1b,
            ref_target=ref_target,
        )

        #  BELOW SAVING NATIVE IAM Countries (e.g. JPN, USA) in the `step1/project_name/IAMC_DATA/` folder
        models = fun_get_models(project_folder)
        df_iam = fun_read_df_iams(project_folder)
        native_dict = {
            m: fun_regional_country_mapping_as_dict(m, project_folder) for m in models
        }
        for m, v in native_dict.items():
            for r, c in v.items():
                if len(c) == 1:
                    df_save = df_iam.xs(
                        [m, f"{m}|{r[:-1]}"],
                        level=["MODEL", "REGION"],
                        drop_level=False,
                    )
                    df_save = fun_rename_index_name(
                        df_save, {"REGION": "ISO", "SCENARIO": "TARGET"}
                    ).droplevel("UNIT")
                    df_save = df_save.assign(CONVERGENCE="none").drop("2005", axis=1)
                    df_save = df_save.assign(METHOD="none").set_index(
                        ["CONVERGENCE", "METHOD"], append=True
                    )
                    df_save = df_save.rename(
                        {df_save.reset_index().ISO.unique()[0]: c[0]}
                    )
                    # # Calculates 'Final Energy|Residential and Commercial' (if missing) as the sum of sub-sectors 
                    # mydict={'Final Energy|Residential and Commercial': 
                    #         ['Final Energy|Commercial', 'Final Energy|Residential']}
                    # for main,subs in mydict.items():
                    #     if main not in df_save.reset_index().VARIABLE.unique():
                    #         unit=df_save.xs(main, level='VARIABLE').reset_index().UNIT.unique()[0]
                    #         df_save=fun_create_var_as_sum(df_save, main,subs, unit=unit)
                    step1folder = CONSTANTS.CURR_RES_DIR("step1") / project_folder             
                    df_save.to_csv(
                        step1folder / "IAMC_DATA" / f"{m}_{r[:-1]}_native.csv"
                    )

    ## RUN STEP 2
    if step2:
        input_list2 = Primary_Energy_2a.make_input_data(
            project_folder,
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
            tc_max_range=conv_range,
            scen_dict=scen_dict,
            run_sensitivity=run_sensitivity_from_step2_to_5,  # If False use standard convergence settings for primary/secondary (if True we use 2050,2100,2150)
            default_ssp_scenario=default_ssp_scenario,
            long_term=long_term,
            method=method,
        )

        ## Use step1b=True. This does not correspond to step1b defined above (we can have previously available step1b results, and  dont't want to re-run step1b)
        if use_step1b_harmo_data:
            [x.update({"step1b": True}) for x in input_list2]

        # Delete tmp folder and run step2
        Primary_Energy_2a.run_step2(input_list2, random_electricity_weights, n_jobs)

        # Combine results found in tmp folder
        Primary_Energy_2a.combine_results(input_list2, random_electricity_weights)

    if conv_dict is not None:
        conv_dict_inv = {v: key.upper() for key, val in conv_dict.items() for v in val}
    else:
        conv_dict_inv = None
    if step2_pick_one_pathway:
        models = {input_set["model_patterns"] for input_set in input_list}
        for model in models:
            file = CONSTANTS.CURR_RES_DIR("step2") / f"{model}_{file_suffix}.csv"
            df = pd.read_csv(file)
            cols_index = ["MODEL", "ISO", "VARIABLE", "SCENARIO", "UNIT"]
            if "CRITERIA" in df.columns:
                cols_index = cols_index + ["CRITERIA"]
            df = df.set_index(cols_index)
            if "CRITERIA" in df.columns:
                df = df.xs("standard", level="CRITERIA")
            if conv_dict is None:
                df_finalized = fun_finalize_step2_all_targets(df, "MED", None)
            else:
                conv_dict_inv = {
                    v: key.upper() for key, val in conv_dict.items() for v in val
                }
                df_finalized = fun_finalize_step2_all_targets(
                    df, None, conv_dict_inv, check_taget_dict_keys=False
                )
            df_finalized.to_csv(str(file).replace("csv", "one_pathway.csv"))

    if step3:
        CCS_CO2_GDP_Prices_January_3.main(
            project_name=project_folder,
            model_patterns=list_of_models,
            region_patterns=list_of_regions,
            target_patterns=list_of_targets,
            gdp_pop_down=_gdp_pop_down,
            add_model_in_region_name=(1 - model_in_region_name),
            file_suffix=f"_{file_suffix}",  #'__round_1p4_enlong_reg_lenght.csv'## CSV in
            file="snapshot_all_regions.csv",
            single_pathway=True,  ## We show only one pathway for each scenario
            add_gdp_pop_data=add_gdp_pop_data,  ## from 2010
            harmonize_gdp_pop_with_reg_iam_results=True,  ## This will change the GDP unit from USD 2005 (as reported by SSP data) to USD 2010 (as reported by IAMs)
            _scen_dict=scen_dict,
            conv_dict_inv=conv_dict_inv,
            run_sensitivity= run_sensitivity_from_step2_to_5,
            sensitivity_dict=sensitivity_dict,
            default_ssp_scenario=default_ssp_scenario,
            gdp_model=gdp_model,
            pop_model=pop_model,
            ref_scen=ref_target,
            method='wo_smooth_enlong',
            long_term='ENLONG_RATIO',
            func='log-log',
            criteria='standard',
        )

    # Create dictionary with downscaling parameters to be stored in excel file
    # NOTE: we exclude functions and pd.DataFrames
    inpt_arg = {}
    sens_list = [""]
    sens_suf = ""
    for k, v in dict(locals()).items():
        if not isinstance(v, pd.DataFrame) and not callable(v):
            inpt_arg[k] = v
    if step5:
        txt = "We are running step5 excluding policies (we read results from step3). Do you wish to proceed? (y/n)"
        #        if not step4:
        if not step4 and not run_step5_without_policies:
            action = input(txt)
            if action.lower() in ["yes", "y"]:
                run_step5_without_policies = True
                print("We run step5 excluding policies")
            else:
                raise ValueError(
                    f"Simulation aborted by the user (user input={action})"
                )
        print("Running step5...")

        [
            Step_5_Scenario_explorer_5.main(
                project_name=project_folder,
                add_model_in_region_name=(1 - model_in_region_name),
                region_patterns=list_of_regions,
                model_patterns=list_of_models,
                target_patterns=list_of_targets,
                csv_str=file_suffix,
                # step4=step4,
                step4=False,  # we run step4 after step5e
                add_twn=add_twn,
                _scen_dict=scen_dict,
                run_sensitivity=run_sensitivity_from_step2_to_5,
                sensitivity_suffix=f"_sensitivity_{x}",
                add_gdp=_gdp_pop_down,
                input_arguments=inpt_arg,
                save_to_excel=1 - step5b,
            )
            for x in sens_list
        ]

    if step5b:
        step5b_suffix = ""
        if run_step5_with_policies:
            step5b_suffix = "_step4_FINAL"
        # file_prefix_list = ["Explorer_", ""]
        # for file_prefix in file_prefix_list:
        print("Running step5b...")

        Step_5b_emissions_by_sector_5b.main(
            project_folder,
            list_of_models,
            list_of_regions,
            list_of_targets,
            f"MODEL_{file_suffix}{sens_suf}{step5b_suffix}.csv",
            "_Emissions_by_sectors_and_revenues",  # if empty string it appends data to existing csv file. Otherwise it creates a new file. CAREFUL: we always have two datasets for each model 'Explorer_Model...csv' and 'Model...csv'!
            model_in_region_name,
            add_only_revenues=True,  # exclude sectorial emissions
            add_revenues=True, #add_only_revenues,
            input_arguments=inpt_arg,
            # run_sensitivity=run_sensitivity_from_step2_to_5,
        )
        file_visual = f"_{file_suffix}_FINAL.csv"
    else:
        file_visual = f"_{file_suffix}.csv"

    if step5c:
        print("Running step5c (non-co2 variables)...")
        Step_5c_Non_CO2_emissions_by_sector_5c.main(
            project_folder,
            list_of_models,
            list_of_regions,
            list_of_targets,
            model_in_region_name,
        )

    if step5c_bis:
        print("Running step5c_bis (Trade variables)...")
        Step_5cbis_Secondary_energy_and_imports_5c_bis.main(
            project_folder,
            list_of_models,
            list_of_regions,
            list_of_targets,
            file_suffix=file_suffix,
            model_in_region_name=model_in_region_name,
        )

    country_mapping_file = (
        CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
    )
    pyam_mapping_file = (
        CONSTANTS.INPUT_DATA_DIR / project_folder / "default_mapping.csv"
    )

    models = {input_set["model_patterns"] for input_set in input_list}
    regions = {input_set["region_patterns"] for input_set in input_list}
    regions = [x.replace("*", "") for x in regions]
    if step5c_tris:  # AFOLU
        print("Running step5c_tris (AFOLU downscaling)...")
        for model in models:
            afolu_regions = ["marker"] if country_marker_list else regions
            # for region in afolu_regions:
            res_afolu = {
                region: AFOLU_emissions.main(
                    folder=project_folder,
                    model=model,
                    baseyear=2020,
                    marker_region=region,
                    countrylist=country_marker_list
                    if country_marker_list
                    else fun_countrylist(
                        list(models)[0],
                        project_folder,
                        f"{region.split('|')[1]}r",
                    ),
                    indirect_emi_average=indirect_emi_average,
                    grassi_dynamic=grassi_dynamic,
                    grassi_scen_mapping=grassi_scen_mapping,
                    agg_region=None,
                    show_plots=False,
                    ylim=(-800, 800),
                    save_to_csv=False,
                )
                for region in afolu_regions
            }
            # Save to csv
            afolu_folder = CONSTANTS.CURR_RES_DIR("step5") / project_folder
            (afolu_folder).mkdir(exist_ok=True)
            pd.concat(list(res_afolu.values()), axis=0).drop(
                "hist inventories", level="MODEL", axis=0
            ).to_csv(f"{afolu_folder}/{model}_AFOLU_emissions.csv")
        print("step5c_tris done!")
    if step5d:
        Step_5d_aggregated_regions_5d.main(
            project=project_folder,
            models=models,
            csv_suffix=file_suffix,
            aggregated_region_name=aggregated_region_name,
            countrylist=countrylist,
            remove_single_countries=False,
        )
    # file_visual = "_NGFS_2023_04_19_2019_harmo_step5e_None.csv"
    if step5e:
        Step_5e_historical_harmo.main(
            files=["Step5d.csv", "snapshot_all_regions.csv"],
            models_dict={project_folder: list(models)},
            sel_vars=sel_plot_vars_step5e,  # just for plots - not used here as we exit the function earlier
            harmo_vars=step5e_harmo,  # variables to be harmonized
            exclude_emissions_by_sector_from_step5b=add_only_revenues,
            add_hist_data=False,  # to be removed at some point
            create_dashboard=False,
            csv_file_name=f"{project_folder}_{file_suffix}",  # "EUab_2023_04_15_all_models_incl_remind_3p1",
            # We use 2018 as there are some negative values in 2019 for Primary Energy|Oil in `EST` and `ARE` in the iea_data
            harmonize_eea_data_until=harmonize_eea_data_until,
            interpolate=True,  # (to get 2018 data)
            use_iea_data_from_ed=True,  # IEA energy data
            use_eea_data=False,  # Emissions from EEA. If false we use PRIMAP
            harmonize_hist_data=harmonize_hist_data,
            selected_scenarios=list(scenarios),
            # selected_scenarios="EU_climate_advisory_board_2023/Ed_pass_scenarios_category_c1b.csv",
            selected_regions=None,
            feasibility_file_name=None,
            show_plots=False,
            known_issues=known_issues,
            keep_step5_emissions=keep_step5_emissions,
            # aggregate_non_iea_countries=True,
            # aggregate_eu_27=True,
        )

    sens_list = [""]
    sens_suf = ""
    if run_sensitivity_from_step2_to_5:
        sens_list = fun_flatten_list(sensitivity_dict.values(), _unique=True)
        sens_suf = "_sensitivity_"
    if step4:
        [
            Policies_emissions_4.main(
                project_name=project_folder,
                region_patterns=list_of_regions,
                model_patterns=list_of_models,
                target_patterns=list_of_targets,
                csv_out=f"_{file_suffix}{sens_suf}{x}",
                co2_energy_only=co2_energy_only,
                ghg_data_from_ngfs_2021=ngfs_2021_ghg_data,  # default value is an empty list
                read_from_step5e=read_from_step5e,
                countries_w_2030_emi_policies=countries_w_2030_emi_policies,
                countries_w_2050_emi_policies=countries_w_2050_emi_policies,
                timing_net_zero=timing_net_zero,
                harmonize_until=harmonize_eea_data_until,
            )
            for x in sens_list
        ]

    # here needs to read from step4
    if step5e_after_policy:
        Step_5e_historical_harmo.main(
            files=["step4_FINAL.csv"],
            models_dict={project_folder: list(models)},
            sel_vars=sel_plot_vars_step5e,  # just for plots - not used here as we exit the function earlier
            harmo_vars=step5e_harmo,  # variables to be harmonized
            exclude_emissions_by_sector_from_step5b=add_only_revenues,
            add_hist_data=False,  # to be removed at some point
            create_dashboard=False,
            csv_file_name=f"{project_folder}_{file_suffix}",  # "EUab_2023_04_15_all_models_incl_remind_3p1",
            # We use 2018 as there are some negative values in 2019 for Primary Energy|Oil in `EST` and `ARE` in the iea_data
            harmonize_eea_data_until=harmonize_eea_data_until,
            interpolate=True,  # (to get 2018 data)
            use_iea_data_from_ed=True,  # IEA energy data
            use_eea_data=False,  # Emissions from EEA. If false we use PRIMAP
            harmonize_hist_data=harmonize_hist_data,
            selected_scenarios=list(scenarios),
            # selected_scenarios="EU_climate_advisory_board_2023/Ed_pass_scenarios_category_c1b.csv",
            selected_regions=None,
            feasibility_file_name=None,
            show_plots=False,
            read_from_step4=file_suffix,
            coerce_errors=False,
            known_issues=known_issues,
            aggregate_non_iea_countries=True,
            # aggregate_non_iea_countries=aggregate_non_iea_countries,
            # aggregate_eu_27=True,
            keep_step5_emissions=keep_step5_emissions,
        )

        # file_visual = f"_NGFS_{file_suffix}_2019_harmo_step5e_None.csv"
    
    if step6:
        print('We do not run step6 - just the core downscaling method')


if __name__ == "__main__":
    main()
