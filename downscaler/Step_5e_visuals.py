import argparse
from typing import Union
from pprint import pprint 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from itertools import cycle
import plotly.express as px
from cmcrameri import cm
from downscaler import CONSTANTS
from downscaler.Policies_emissions_4 import fun_short_term_co2_energy_targets
from downscaler.fixtures import colors, NGFS_dashboard_bar_plots, legend_dict, g20_countries, iea_countries, iea_flow_dict, step2_primary_energy, colors, iea_flow_long_short_dict,iea_product_long_short_dict, check_IEA_countries
from downscaler.utils_pandas import (
    fun_select_model_scenarios_combinations,
    fun_index_names,
    fun_dict_level1_vs_list_level2,
    fun_xs,
    fun_read_csv,
    fun_check_missing_elements_in_dataframe
)
from downscaler.utils_string import fun_check_if_all_characters_are_numbers
from downscaler.utils import (
    fun_available_scen,
    fun_plot_eu_ab_side_by_side,
    fun_sns_lineplot_new,
    group_keys_based_on_similar_values,
    prepare_data_for_sns_lineplot,
    fun_add_criteria,
    fun_aggregate_countries,
    fun_eu27,
    fun_save_figure,
    fun_line_plots_colors_and_markers,
    fun_analyse_main_drivers_across_scenarios,
    fun_log_log_graphs,
    fun_get_iam_regions_associated_with_countrylist,
    fun_invert_dictionary,
    fun_regional_country_mapping_as_dict,
    fun_rename_index_name,
    fun_wildcard,
    fun_get_models,
    fun_blending_with_sensitivity,
    fun_flatten_list,
    fun_read_results,
    fun_create_regional_variables_as_sum_of_countrylevel_data,
    fun_phase_out_in_all_countries,
    fun_macc_graph,
    fun_carbon_budget_step5e,
    fun_read_df_iams,
    fun_fuzzy_match,
    fun_shorten_region_name,
    fun_find_all_xlsx_files_and_convert_to_csv,
    fun_growth_index,
    fun_combine_df_with_most_recent_scenarios,
    fun_non_iea_countries,
    fun_read_energy_iea_data_in_iamc_format,
    fun_create_var_as_sum,
    fun_remove_non_biomass_ren_nomenclature,
    fun_most_recent_iea_data,
    fun_sort_dict,
    fun_slope_over_time_iamc_format,
    fun_find_nearest_values,
    find_historic_pattern,
    SliceableDict,
    find_and_show_patterns,
    fun_show_energy_transition,
    fun_aggregate_countries_general,
    fun_divide_variable_by_another,
    fun_xs_fuzzy,
    check_variable_coverage_issues,
    evaluate_local_variables,
    find_intersection_among_list_of_lists,
)
from downscaler.fixtures import all_countries
from Step_5e_historical_harmo import fun_create_dashboard_pdf, fun_create_dashboard_pdf_by_country
from matplotlib.backends.backend_pdf import PdfPages

from utils_dictionary import fun_sort_dict_by_value_length
from downscaler.utils_visual import fun_phase_out_date_colormap

input_emi_target_short = pd.read_csv(
    CONSTANTS.INPUT_DATA_DIR / "indc_plus_emi_targets.csv"
)


targets_dict = fun_short_term_co2_energy_targets(
    False, input_emi_target_short, "MESSAGEix-GLOBIOM 1.1-M-R12", "h_cpol"
)
targets_dict = targets_dict[targets_dict.TIME == 2030]["VALUE"].to_dict()

markers_dict = {
    k: {"marker_2030": {2030: v}, "marker_2050": {2050: 0}}
    for k, v in targets_dict.items()
}

dashboard_scatter_plots = {
    "Primary Energy": {
        "kind": "scatter",
        "xvar": "Final Energy",
        "svar": "Emissions|CO2|Energy",  # scatter point size
        "row": 0,
        "col": 0,
        "ylim": None,
    },
}


from downscaler.utils import *
from downscaler.fixtures import *



def main(
    files: Optional[Union[list, str]] = "iconics_NZ_data_and_table_20230512_v17.xlsx",
    step: str = "step5",
    sel_vars: Union[list, str] = ["Emissions|Kyoto Gases (AR4) (EEA - intra-EU only)"],
    sel_scen: Optional[Union[str, list]] = None,#"model_scenarios_combinations_63.csv",
    countrylist: Optional[list] = ["EU27"],
    iea_countries_only:bool = False,
    markers_dict: dict = {},
    models: Optional[list] = None,
    create_dashbord: str = "dashboard.pdf",
    project=None,
    top_20_macc_plot:Optional[str]=None,
    eu_ab_plot: bool = True,
    ngfs_plot: bool = True,
    florian_plot: bool = True,
    ternary_plot:bool=False,
    stacked_countries_plot:bool = False,
    legend_color_box: Optional[str] = None,  # "lower left",  # for florians plot
    legend_ls_box: Optional[str] = None,  # "upper right",  # for florians plot
    log_log_graphs: bool = True,
    phase_out_plot:bool=True, # phase out dates by countries
    carbon_budget_plot:bool=True,
    analyse_drivers: bool = False,
    split_df_by_model_and_save_to_csv: bool = False,
    step1_sensitivity: bool = False,
    step1_extended_sensitivity:bool = False,
    sel_step1_methods: Optional[list] = None,
    palette:Optional[str]=None,    
    combine_df_with_most_recent_scenarios=False,
    check_missing_data:bool=False,
):
    
    # NOTE: there are some CODE blueprints available 
    # - coal phase out and intensity graph for NGFS paper:https://github.com/iiasa/downscaler_repo/commit/f333b9fe54067d650d8caf80e1748941a97fc4a9
    # - florian_plot for step1 sensitivity analyis: https://github.com/iiasa/downscaler_repo/commit/9da4edfd2e7dddcf19e1576bc7cc5b3061c8e9ad
    print("NOTE:")
    print(
        "If the models results are in one (or more) files, you can select: \n"
        "-> `models=None` \n"
        "-> `files=[your list of files e.g. MESSAGE_2023_07_14_WITH_POLICY.xlsx]` \n"
    )
    print(
        "If all of your models share the same file suffix, you can select: \n"
        "-> `models=[your list of models e.g. MESSAGE, GCAM]` \n"
        "-> `files=[your suffix file name (excluding the MODEL prefix) e.g. `2023_07_14_WITH_POLICY.xlsx`]` \n "
    )

    items_plot=('eu_ab_plot', 'ngfs_plot', 'florian_plot', 'ternary_plot', 'top_20_macc_plot', 'log_log_graphs','carbon_budget_plot','phase_out_plot', 'stacked_countries_plot')
    items_feature=('combine_df_with_most_recent_scenarios', 'step1_sensitivity', 'step1_extended_sensitivity', 'analyse_drivers',"check_missing_data", 
            'split_df_by_model_and_save_to_csv', 'sel_step1_methods')
    print('\n')
    print('**** SUMMARY ****')
    print(f'Your project: {project}')
    print(f'Your step: {step}')
    print(f'Your models: {models}')
    print(f'Your scenarios: {sel_scen}')
    print(f'Your variables: {sel_vars}')
    print(f'Your countries: {countrylist}')
    print(f"Your plots:", evaluate_local_variables(items_plot, locals()))
    print(f"Your features:", evaluate_local_variables(items_feature, locals()))
    print('\n')

    # # Combine results across models
    # res=fun_combine_df_with_most_recent_scenarios(project, step, save_to_csv=False, 
    #                                                     #   sub_folder=2023
    #                                                       )
    if combine_df_with_most_recent_scenarios:
        fun_combine_scenario_step5e_visual(step, project)

    # fun_patterns_summary_by_country(models)
    # pd.concat([fun_patterns_summary_by_country(df, df_iea, c, 'o_1p5c', ref_future_scen='h_cpol', models=models, K=1) for c in ['CHN','IND','IDN','ZAF']], hist_time_periods=[2001, 2017])
    # df_iea=fun_most_recent_iea_data()

    # TODO:
    # - [ ] Rename `find_historic_pattern` as it also work for the future (h_cpol)
    # - [ ] Enhance `find_historic_pattern` so that time_mapping is explicietly defined for each scenario?? 

    # - [ ] Make `find_all_patterns` it run for `df` instead of `df_iea`:
    # - [ ] Rename `find_all_patterns` as `find_all_patterns_for_single_scenario`

    # res = find_all_patterns(df_iea, 'IEA', 'Historic data', max_number_of_keys=6)
    # https://jsfiddle.net/gh/get/library/pure/highcharts/highcharts/tree/master/samples/highcharts/demo/network-graph/

    # # file='Extended_IEA_en_bal_2021.csv'
    # file='IEA_Balances_Rev2022.csv'
    # df = fun_create_iea_file_IAMC_format(CONSTANTS.INPUT_DATA_DIR/file)
    # res = fun_validate_IEA(df)

    ## This one does not work: df.xs('CHN', level='REGION').xs('Secondary Energy|Electricity|Coal', level='VARIABLE')[2015]
    # also final energy solids for china does not work
    if CONSTANTS.CURR_RES_DIR('step5')==CONSTANTS.CURR_RES_DIR(step):
        file_dict={'NGFS_2022':'1624889226995-NGFS_Scenario_Data_Downscaled_National_Data.csv',
                'NGFS_2023':"NGFS_2023_Downscaled_National_Data.csv",#'V4.1 NGFS Phase 4'
                        # This was created based on these files: 
                        # - 'MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2023_2024_02_29_2018_harmo_step5e_WITH_POLICY_None.csv', 
                        # - 'REMIND-MAgPIE 3.2-4.6_NGFS_2023_2024_02_21_2018_harmo_step5e_WITH_POLICY_None.csv', 
                        # - 'GCAM 6.0 NGFS_NGFS_2023_2024_02_29_2018_harmo_step5e_WITH_POLICY_None.csv'
                    # 'NGFS_2023':'NGFS_2024_test_EU27.csv' ## THIS ONE IS A test
                #'NGFS_2023':'NGSF_and_emiclock_2023.csv', too complex due to duplicated dtaa. Plus emission clock project only contains results fromMESSAGE.
                    # 'NGSF_and_emiclock_2023.csv' generated from: 
                    #   "Emissions_clock_2023_09_15_v3_sent.csv"
                    #   "NGFS_2023_Downscaled_National_Data.csv" (see commented code above)
                }   

    # Sanity check - input data 
    if not project and CONSTANTS.CURR_RES_DIR(step)!=CONSTANTS.CURR_RES_DIR('step1'):
        txt = "Please pass a `project` to the main function (currently is None)"
        raise ValueError(txt)
    
    if top_20_macc_plot:
        file_top20=f"{project}/{top_20_macc_plot}"
        df_test=pd.read_csv(CONSTANTS.CURR_RES_DIR('step5')/file_top20, index_col=['MODEL','VARIABLE','REGION'])
        # df_test=pd.read_csv(CURR_RES_DIR/'test_macc.csv', index_col=['MODEL','VARIABLE','REGION'])
        if countrylist is not None:
            df_test=fun_xs(df_test, {'REGION':countrylist})
        for complement_to_one in [False, True]:
            for i in ['gov','eff','cor','rea']:
                for x in df_test.columns[:2]:
                    fig = fun_macc_graph(df_test.reset_index().iloc[:-1], [x], "pre", legend_dict, sel_y=i
                                , complement_to_one=complement_to_one
                                )
                    top20f=f"Top_20_1p5c_{x}_indicator={i}"
                    top20f=f"Top_20_1p5c_{x}_indicator=1-{i}" if complement_to_one else top20f
                    fun_save_figure(
                        fig,
                        CONSTANTS.CURR_RES_DIR("step5") / "Step5e_visuals" / project,
                        top20f
                    )
                    plt.clf()
    # Get models
    f = fun_wildcard
    models_all = fun_get_models(project) if project else None
    models = models_all if models is None or project is None else f(models, models_all)

    if isinstance(countrylist, list):
        c_dict={'EU27':fun_eu27(), 'EU28':fun_eu28()}
        if find_max_nesting(countrylist)==1:
           countrylist_temp=[c_dict.get(c,c) for c in countrylist]
        else:
           countrylist_temp=fun_flatten_list([c_dict.get(c,c) for c in countrylist])
        if len(countrylist_temp)<10:
            iam_regions={c:fun_country_to_region(project, c, models) for c in countrylist_temp } 
            print('**** REGIONAL MAPPING ****')
            # print(f"These are the IAMs regions associated to your countrylist: \n {iam_regions}")
            print(iam_regions)


    
    if CONSTANTS.CURR_RES_DIR(step)==CONSTANTS.CURR_RES_DIR('step5'):
        files=file_dict.get(project, files)
        if files and files[-3:] not in ['csv', 'lsx', 'xls']:
            all_files=os.listdir(CONSTANTS.CURR_RES_DIR('step5')/files)
            if not len([x for x in all_files if '.csv' in x]):
                # create CSV files
                fun_find_all_xlsx_files_and_convert_to_csv(CONSTANTS.CURR_RES_DIR('step5')/files)
            all_files=[x for x in all_files if '.csv' in x]
            if not len(all_files):
                raise ValueError(f"Unable to find csv files in this folder {CONSTANTS.CURR_RES_DIR('step5')/files}")
            files={m:CONSTANTS.CURR_RES_DIR(step)/files/i  for m in models for i in all_files if m in i }
    scenarios=None
    scen_dict={'Delayed transition':'d_delfrag', 
            'Fragmented World':'d_strain', 
            'Current Policies':'h_cpol',
            'Nationally Determined Contributions (NDCs)':'h_ndc', 
            'Net Zero 2050':'o_1p5c',
            'Below 2°C':'o_2c', 
            'Low demand':'o_lowdem'}
    if models:
        CURR_RES_DIR = CONSTANTS.CURR_RES_DIR(step)
        
        if step =='step2b':
            # Box plot for step2b
            fun_box_plot_step2(files, models, project)

        print("**** READ DOWNSCALED RESULTS ****")
        df, files, files_dict, scenarios = fun_read_results(
                project,
                step,
                files,
                countrylist,
                models,
                rename_dict=scen_dict,
            )
        df=np.round(df,4)
    # models= fun_flatten_list([fun_fuzzy_match(df.reset_index().MODEL.unique(),x) for x in models])
    countrylist=countrylist or all_countries

    # if CONSTANTS.CURR_RES_DIR('step1')==CONSTANTS.CURR_RES_DIR(step):
    #     df=df[df.TIME!='TIME']
    #     df['TIME']=[int(x) for x in df['TIME']]

    # fun_xs_fuzzy(df, ['a',' ', 'IND','AAA'], _and=True)
    # fun_xs_fuzzy(df, ['h_CPOL', 'A','GDP'], _and=True)[range(2015,2025,5)]/1e3
    # fun_xs_fuzzy(df, ['h_CPOL', 'J','GDP'], _and=True)[range(2015,2025,5)]/1e3
    # fun_xs_fuzzy(df, ['h_CPOL', 'VEN','GDP'], _and=True)[range(2015,2025,5)]/1e3
    # fun_xs_fuzzy(df, [''], _and=True)[range(2015,2025,5)]/1e3
    
    # Blueprint to fix scenarios 
    # TODO: replace `fun_xs_fuzzy` with `fun_xs`
    # df_iam=fun_read_df_iams(project, models)
    # df=pd.concat([fix_scenarios(df, df_iam, scen, refscen='h_cpol', dt=5 ) for scen in scenarios])

    # # fix scenarios to h_cpol
    # myvars=df.reset_index().VARIABLE.unique()
    # refscen='h_cpol'
    # myscen='o_1p5c'
    # dt=5
    # df_iam=df_iam.copy(deep=True)
    # df_iam=fun_index_names(df_iam, True, int)
    # df_iam=fun_xs(df_iam, {'VARIABLE':list(myvars)})
    # seltime=[]
    # for t in range(df.columns.min(),df.columns.max()+dt,dt):
    #     try:
    #         assert_frame_equal(df_iam.xs(refscen, level='SCENARIO')[[t]], df_iam.xs(myscen, level='SCENARIO')[[t]])
    #         seltime+=[t]
    #     except:
    #         break
    # set1=fun_xs(df, {'SCENARIO':myscen}).index
    # set2=fun_xs(df, {'SCENARIO':refscen}).rename({refscen:myscen}).index
    # myindex=list(set1&set2)
    # df_fixed=fun_xs(df, {'SCENARIO':refscen}).rename({refscen:myscen}).loc[myindex,seltime]
    # df.loc[df_fixed.index, df_fixed.columns]=df_fixed
    
    # Check issues (missing variables) in the downscaling:
    if check_missing_data:
        if CONSTANTS.CURR_RES_DIR(step)==CONSTANTS.CURR_RES_DIR('step5'):
            print("\n**** CHECK MISSING VARIABLES ****")
            # General check
            fun_check_missing_data(df)
            var_to_check=["Population","GDP|PPP","Final Energy",'Secondary Energy|Electricity',
                        'Revenue|Government|Tax|Carbon',
                        "Emissions|CO2|Energy|Demand|Industry",
                        'Emissions|Total Non-CO2',
                        'Trade|Primary Energy|Coal|Volume'
                        ]
            fun_check_missing_variables(df,  var_to_check, check_IEA_countries)

            # print("\n **** COUNT ALL VARIABLES REPORTED ****") 
            for model in df.reset_index().MODEL.unique():
                # print(f"\n{model}")
                # Check how many variables are reported for each country/scenario :
                d=check_variable_coverage_issues(df, model)
                if len(d)==0:
                    print(f"{model}: all good with variable coverage")
                else:
                    print(f"{model}: Only a few variables reported for :\n {d}")

        # df= fun_aggregate_countries_general(df.droplevel('FILE'),"EU27", fun_eu27())
        # df=df.assign(FILE='A').set_index('FILE', append=True)
        # res_all={model:  find_all_patterns(df, model, 'o_1p5c', max_number_of_keys=6, time_mapping={2035:2020, 2035:2020}) for model in models}
        # res = find_all_patterns(df, models[0], 'o_1p5c',# ref_scen='h_cpol',
        #                          max_number_of_keys=6, time_mapping={2035:2020, 2035:2020})
        # # `key` is the variable of interest (num), `value` is the main sector (den) -> to calculate share of main sector

    if project is not None:
        scenarios = f(sel_scen, scenarios) if isinstance(sel_scen, list) else scenarios
        scenarios=scenarios+['Historic data'] if sel_scen is not None and 'Historic data' in sel_scen else scenarios
        # Code below preserves scenarios order as defined in sel_scen (for stacked_plot)
        scenarios_ordered = fun_flatten_list([fun_fuzzy_match(scenarios,x, n=1) for x in sel_scen]) if sel_scen else scenarios
        if set(scenarios_ordered)==set(scenarios):
            scenarios=scenarios_ordered
        if sel_scen is not None and set(scenarios) == 0:
            raise ValueError(f"sel_scen {sel_scen} are not present in df: {scenarios}")
    
    
    if phase_out_plot: # phase out date by countries
        df=fun_divide_variable_by_another(df, 'GDPCAP', 'GDP|PPP', 'Population', operator='/',concatenate=True)
        if CONSTANTS.CURR_RES_DIR(step) != CONSTANTS.CURR_RES_DIR('step5'):
            raise ValueError(f'phase_out_plot are rupported only for `step5`. You selected {step}')
        
        my_dict={
                # 'Primary Energy|Coal|w/o CCS':'Primary Energy',
                 'Primary Energy|Coal':'Primary Energy',
                 'Primary Energy|Gas':'Primary Energy',
                # 'Primary Energy|Gas|w/o CCS':'Primary Energy',
                'Primary Energy|Oil':'Primary Energy',
                'Secondary Energy|Electricity|Coal':'Secondary Energy|Electricity',
                'Secondary Energy|Electricity|Gas':'Secondary Energy|Electricity',
                # 'Secondary Energy|Electricity|Oil':'Secondary Energy|Electricity',
                # 'Final Energy|Transportation|Hydrogen':'Final Energy|Transportation',
                # 'Secondary Energy|Electricity|Coal': 'Secondary Energy|Electricity',
                # 'Primary Energy|Coal|w/o CCS':'Primary Energy',
                # 'Emissions|CO2|Energy':None,
        }
        
        # NOTE: Below we plot phase out graphs for all countries in countrylist
        res={}
        if len(scenarios)>2:
            text=f'Are you sure you want to plot phase out graphs for all these scenarios:{scenarios}? y/n'
            action=input(text)
            if action.lower() not in ["yes", "y"]:
                raise ValueError(f"Simulation aborted by the user (user input={action})")
        for var in sel_vars:    
            if var not in my_dict:
                raise ValueError(f'Cannot find {var} in `my_dict`. Please add it to the dictionary')
            my_dict_sel=SliceableDict(my_dict).slice(var)
            for scen in scenarios:
                largest_countries=fun_xs(df, {'VARIABLE':var, 'SCENARIO':scen})[2020]
                largest_countries=list(largest_countries.groupby('REGION').median().sort_values(ascending=False).index[:30])
                for k,v in my_dict_sel.items():
                    threshold=0.05 if v is not None else 0 
                    data=fun_phase_out_in_all_countries(fun_xs(df, {'REGION':largest_countries}), scen,k, v, countrylist, True, True, threshold)
                    data=data[data['median']>2020].sort_values('median')
                    data= pd.concat([df.xs(['GDPCAP', scen], level=['VARIABLE','SCENARIO'])[2020].droplevel(['UNIT','FILE']).unstack('MODEL').median(axis=1), data], axis=1).dropna(axis=0).rename({0:'GDPCAP'},axis=1)
                    res[k]=data.sort_values('median')
                    res[k].loc[:,'median']=res[k].median(axis=1)
                    
                    fun_phase_out_date_colormap(res[k], colored_error_bars=False)
                    plt.xlabel('Median and range across models')
                    plt.ylabel('Country')
                    plt.title(f"{k} - phase out dates ({scen})", size=15)
                    plt.xlim(left=2020, right=2100)
                    plt.show()
                    
                    # # Set the figure size
                    # plt.figure(figsize=(14, 8))
                    # # plot a bar chart
                    # pal = sns.color_palette("Greens_d", len(data))
                    # pal2=np.array(pal[::-1])[res[k]['GDPCAP'].argsort().argsort()]
                    # fig= sns.barplot(x="median", y=res[k].index, data=res[k], estimator=np.mean, ci=85, capsize=.2, palette=pal2)
                    # plt.colorbar(fig)
                    # # ax = sns.barplot(x="median", y="index", data=res[k].reset_index(), estimator=np.mean, ci=85, capsize=.2, color='lightblue')
                    # plt.title(f"{k} - phase out dates ({scen})")
                    # plt.xlim(left=2020, right=2100)
                    # plt.show()

        # NOTE: Below we plot phase out graphs for all countries in ONE PLOT
        a=pd.concat([pd.DataFrame(v['median']).rename({'median':k}, axis=1) for k,v in res.items()], axis=1)
        sns.barplot(x="value", y="index",  data=pd.melt(a.reset_index().sort_values(by=list(my_dict.keys())[0]),  id_vars='index'), hue='variable')
        plt.title(f"Phase out dates ({scen})")
        plt.xlim(left=2020, right=2100)

        
        
        # my_dict={
        #         'Final Energy':'Population',
        # }
        # NOTE: Final Energy|Transportation|Hydrogen' is good for a test case
        
        # NOTE: Below we identify top 10 countries where fast decarbonization is required and plot graph for ALL countries:

        df_iea=fun_index_names(fun_most_recent_iea_data(), True, int)
        for var in sel_vars:    
            if var not in my_dict:
                raise ValueError(f'Cannot find {var} in `my_dict`. Please add it to the dictionary')
            my_dict_sel=SliceableDict(my_dict).slice(var)

            for num,den in my_dict_sel.items():
                # Get historical data for selected data
                df_iea_sel=df_iea.xs(num, level='VARIABLE')/df_iea.xs(den, level='VARIABLE')
                # Get historical decarbonization rates (slope over time):
                
                slope_hist=fun_sort_dict(fun_slope_over_time_iamc_format(df_iea_sel.dropna(how='all', axis=1))['IEA']['Historic data'], by='values')

                # NOTE: The link below provide a code snippet which demostrates that we get same slopes if we interpolate or if we use an yearly dataframe:
                # https://github.com/iiasa/downscaler_repo/issues/187

                # Get future decarbonization rates (slope over time)
                mydf=(df.xs(num, level='VARIABLE')/df.xs(den, level='VARIABLE'))

                find_historic_pattern(df_iea, df, 'POL', models[0], scenarios[0])

                scenarios = f(sel_scen, scenarios) if isinstance(sel_scen, list) else scenarios

                for scen in scenarios:
                    
                    # We get a list of the top 30 countries for that `var`
                    largest_countries=fun_xs(df, {'VARIABLE':var, 'SCENARIO':scen})[2020]
                    largest_countries=list(largest_countries.groupby('REGION').median().sort_values(ascending=False).index[:30])
                    
                    # We check slope between 2020-2030
                    mydf_scen=mydf.dropna(how='all', axis=1).xs(scen, level='SCENARIO',drop_level=False)[list(range(2020,2035,5))]
                    # Line below calculates slope for a median across all models. It can be commented out to get results by model
                    mydf_scen=mydf_scen.groupby(['SCENARIO','REGION','UNIT']).median().assign(MODEL='median').set_index('MODEL', append=True)
                    slope_median_across_models= fun_slope_over_time_iamc_format(fun_xs(mydf_scen, {'REGION':largest_countries}))
                    # slope_median_across_models= fun_slope_over_time_iamc_format(mydf_scen)
                    # slope_model=slope_all_models[model][scen]
                    slope_median_across_models=slope_median_across_models['median'][scen]
                    # We exclude slopes equal to zero, np.nan or None
                    slope_model={k:v for k,v in slope_median_across_models.items() if v is not None and v!=0 and not pd.DataFrame([v]).isnull()[0][0]}
                    slope_model= fun_sort_dict(slope_model, by='values')

                    # TODO: 
                    # Optionally could also muliply slope_model by coal use in 2020:  identify both countries that are large and that require fast transition!


                    # # First we Identify top 10 countries (with most negative slope for that fuel)
                    # top_ten_countries=list(slope_model.keys())[:10]
                    # # Then among those countries we take the four largest:
                    # largest_countries=fun_xs(df, {'REGION':top_ten_countries,'VARIABLE':var, 'SCENARIO':scen})[2020]
                    # largest_countries=largest_countries.groupby('REGION').median().sort_values(ascending=False).index[:4]
                    # We identify similar transitions for these countries
                    for c in list({k:v for k,v in slope_model.items()}.keys())[0: 4]:
                    # for c in largest_countries:
                            # find_and_show_patterns(df, scen, model, c, sector_mapping, ref_scen='Historic data')
                            # fun_show_patterns_energy_transition(models[0], df, c, scen, scen_ref, sector_mapping, c_list, group_by_time=False)
                        
                        try:
                            # Plot graph for top 10 countries (with most negative slope for that `var`)
                            # NOTE: this graph does not tell you about what replaces the `var`. (e.g. coal replaced by gas instead of renewables) 
                            # This is the reason why we also have the function `find_and_show_patterns` that finds similar transitions across countries for all fuels 

                            # Dataframe with historical data below:
                            df1=fun_xs(df_iea_sel, {'REGION':c}).droplevel(['SCENARIO','UNIT']).dropna(how='all', axis=1)
                            
                            # Dataframe with future data
                            df2=fun_index_names(mydf, True, int).xs(scen, level='SCENARIO').xs(c, level='REGION', drop_level=False).droplevel(['FILE','UNIT']).dropna(how='all', axis=1)
                            # Find countries with similar shares and fast historical decarbonization :
                            # countries_with_similar_shares=(df_iea.xs(num, level='VARIABLE')/df_iea.xs(den, level='VARIABLE')).median(axis=1)
                            # countries_with_similar_shares=countries_with_similar_shares[(countries_with_similar_shares>df1.min(axis=1)[0])&(countries_with_similar_shares<df1.max(axis=1)[0])]
                            # countries_with_similar_shares=(df_iea.xs(num, level='VARIABLE')/df_iea.xs(den, level='VARIABLE')).median(axis=1)
                            # countries_with_similar_shares=countries_with_similar_shares[countries_with_similar_shares>df1.max(axis=1)[0]]
                            countries_with_higher_max=df_iea_sel[df_iea_sel>df1.max(axis=1)[0]].dropna(how='all', axis=1).dropna(how='all', axis=0).reset_index().REGION.unique()
                            countries_with_lower_min=df_iea_sel[df_iea_sel[2018]<df1[2015][0]].reset_index().REGION.unique()
                            countries_with_similar_shares=list(set(countries_with_higher_max)&set(countries_with_lower_min))
                            countries_with_similar_shares=[x for x in countries_with_similar_shares if x!=c]
                            slack=1
                            while len(countries_with_similar_shares)<=2:
                                countries_with_higher_max=df_iea_sel[df_iea_sel>df1.max(axis=1)[0]/slack].dropna(how='all', axis=1).dropna(how='all', axis=0).reset_index().REGION.unique()
                                countries_with_lower_min=df_iea_sel[df_iea_sel[2018]<df1[2015][0]*(slack-0.2)].reset_index().REGION.unique()
                                slack=slack+0.05
                                countries_with_similar_shares=list(set(countries_with_higher_max)&set(countries_with_lower_min))
                                countries_with_similar_shares=[x for x in countries_with_similar_shares if x!=c]
                                if slack==2:
                                    break
                            # Find country with similar decarbonization rates:
                            slope_hist_sel=SliceableDict(slope_hist).slice(*tuple(countries_with_similar_shares))
                            # find_val=fun_find_nearest_values(list(slope_hist_sel.values()), slope_model[c], n_max=1)
                            find_val=fun_find_nearest_values(list(slope_hist_sel.values()), slope_model[c], n_max=min(3,len(list(slope_hist_sel.values()))))
                            similar_country=[k  for k,v in slope_hist_sel.items() if v in find_val]
                            # print(f'Decarbonization rate required in {c}, is similar to the one observed historically in: {similar_country}')
                            
                            df3=fun_xs(df_iea_sel, {'REGION':similar_country}).droplevel(['SCENARIO','UNIT']).dropna(how='all', axis=1)
                            # combi=pd.concat([df1,df2])
                            # ax=combi.T.plot()#(colors=['black','red','blue','grey'])
                            # df3.T.plot(colors='grey', ax=ax, ls='dotted')
                            # plt.title(f"Share of {num.split('|')[-1]} in {c}")
                            ax1=df1.dropna(how='all', axis=1).T.plot(colors='black')#(colors=['black','red','blue','grey'])
                            ax2=df2.dropna(how='all', axis=1)[range(2010,2055,5)].T.plot(ax=ax1)
                            df3.dropna(how='all', axis=1).T.plot(colors='grey', ax=ax2, ls='dotted')
                            plt.title(f"Share of {num.split('|')[-1]} in {c}")
                            plt.show()


                            for model in models:
                                main_sector='Secondary Energy|Electricity'
                                sector_mapping={main_sector: [f"{main_sector}|{x.split('|')[-1]}" for x in step2_primary_energy['Primary Energy']]}  
                                
                                # TODO: Can also think about grouping similar countries in an automatised way e.g. 
                                # - e.g. first grouping based on very high coal/oil/gas/renewables in 2020. 
                                # - then   among that fuel (eg. coal) -> serch country with similar pattern in 2030.
                                # in this manner we group countries to identify similar transitions across countries. 
                                for ref_scen in ['h_cpol','Historic data']:
                                    fig=find_and_show_patterns(df, scen, model, c, sector_mapping, ref_scen=ref_scen, max_countries=3)
                                    fun_save_figure(
                                            fig,
                                            CONSTANTS.CURR_RES_DIR("step5") / "Step5e_visuals" / project/model,
                                            f"Patterns_{main_sector.replace('|', '_')}_{c}_{scen}_vs_{ref_scen}",
                                            )
                        except:
                            print(f'{c} not working')
                        

        
    if carbon_budget_plot:
        scen='o_1p5c'
        if CONSTANTS.CURR_RES_DIR(step) != CONSTANTS.CURR_RES_DIR('step5'):
            raise ValueError(f'carbon_budget_plot are rupported only for `step5`. You selected {step}')
        c_budg=fun_carbon_budget_step5e(df, scen, countrylist, var='Emissions|CO2')
        c_budg.plot(kind='barh', log=True)
        ax.set_xscale("log")
        plt.title(f'Remaning Carbon Budget 2020-2030 | {scen} pathway [MtCO2]')

    # pd.concat([fun_phase_out_in_dates(df, 'Final Energy|Hydrogen',c, 'o_1p5c', False) for c in ['DEU','FRA','ITA','AUT','TUR']])
    # pd.concat([fun_phase_out_in_dates(df, 'Secondary Energy|Electricity|Coal',c, 'o_1p5c', True) for c in ['DEU','FRA','ITA','AUT']])
    # fun_phase_out_in_dates(df, 'Final Energy','DEU', 'o_1p5c', False)
    # fun_phase_out_in_dates(df, 'Final Ene','DEU', 'o_1p5c', False)

    # a=pd.concat([fun_phase_out_in_dates(df,'Secondary Energy|Electricity|Coal',c, 'o_1p5c', True) for c in df.reset_index().REGION.unique()]).sort_values(models[0])

    # SCATTER PLOTS BLEPRINT
    # variables=[x for x in df.reset_index().VARIABLE.unique() if "venue" in x][:1]+[x for x in df.reset_index().VARIABLE.unique() if "Electricity" in x and "Final" in x][5:10]
    # d={n:v for n,v in enumerate(variables)}
    # A=pd.concat([df.xs(x, level="VARIABLE", drop_level=False)[2040].unstack("VARIABLE").droplevel("UNIT") for x in variables], axis=1)
    # pd.plotting.scatter_matrix(A.rename({v:k for k,v in d.items()}, axis=1), alpha=0.5)
    # #plt.yscale("log")
    # #plt.xscale("log")
    # plt.tight_layout()


    # Create one file for each model if:
    # - `files` contains only one file
    # - `models` contains more than one model
    if models:
        split_flag = len(models) > 1 and len(files) == 1 and files[0][-3:] == "csv"
        if split_df_by_model_and_save_to_csv and split_flag:
            [
                df.xs(m, drop_level=False).to_csv(CURR_RES_DIR / f"{m}_{files[0]}")
                for m in models
            ]



    if iea_countries_only and stacked_countries_plot:
        raise ValueError('When plotting `stacked_countries_plot` we aggregate non-iea countries a single country block.' 
                         'Therefore please pass `iea_countries_only=False` ')

    selcols = list(range(2010, 2055, 5))
    if ternary_plot:
        # Reference: https://plotly.com/python/ternary-plots/
        folder = CONSTANTS.CURR_RES_DIR("step5")/ "Step5e_visuals"
        folder=folder / project
        allowed_ternary_vars=['Final Energy', 'Final Energy_v2', 'Primary Energy', 'Emissions']
        ternary_vars=[x for x in allowed_ternary_vars if x in sel_vars]
        if len(ternary_vars)==0:
            txt="If you want to plot ternary graphs, `sel_vars` should contain at least one of these variables:"
            raise ValueError(f'{txt} {allowed_ternary_vars}')

        # Create aggregated variables
        ren=['Geothermal', 'Hydro','Solar', 'Wind']
        ternary_dict={'Final Energy':{'Liquids & Gases':['Final Energy|Liquids', 'Final Energy|Gases'],
                                      'Solids': ['Final Energy|Solids'],
                                      'Electrification/Heat/Hydrogen':['Final Energy|Electricity', 'Final Energy|Heat', 'Final Energy|Hydrogen']},
                        
                        'Final Energy_v2':{'Industry':['Final Energy|Industry'],
                                      'Transportation': ['Final Energy|Transportation'],
                                      'Residential and Commercial':['Final Energy|Residential and Commercial']},
                        
                        'Primary Energy':{
                                      'Fossils with CCS, BECCS, Biomass': ['Primary Energy|Coal|w/ CCS','Primary Energy|Gas|w/ CCS', 'Primary Energy|Oil|w/ CCS', 'Primary Energy|Biomass'],
                                      'Fossils w/o CCS':['Primary Energy|Coal|w/o CCS','Primary Energy|Gas|w/o CCS', 'Primary Energy|Oil|w/o CCS'],
                                      'Renewables and Nuclear':['Primary Energy|Geothermal', 'Primary Energy|Hydro','Primary Energy|Nuclear', 'Primary Energy|Solar','Primary Energy|Wind']},
                      }
        if 'Historic data' in scenarios:
            # Final Energy from IEA 2019 an
            iea = fun_most_recent_iea_data()
            df2=pd.concat([df.droplevel('FILE')[selcols], fun_index_names(iea, True, int)])
        else:
            df2=df.droplevel(['FILE','UNIT'])
        df_iam=fun_index_names(fun_read_df_iams(project, models).droplevel('UNIT'), True, int)[selcols]
        df_iam=fun_remove_non_biomass_ren_nomenclature(df_iam)
        df2=pd.concat([df2.reset_index().set_index(df_iam.index.names), fun_xs(df_iam, {'MODEL':models})])
        for var in ternary_vars:
            selected_vars=[var.replace('_v2','')]+list(ternary_dict[var].keys())+fun_flatten_list(list(ternary_dict[var].values()))
            for k,v in ternary_dict[var].items():
                df2=fun_create_var_as_sum(df2,k,v)
                df2=df2.drop('UNIT', axis=1) if 'UNIT' in df2.columns else df2
            df3=df2.groupby(df2.index.names).sum()
            df3=fun_xs(df3, {'VARIABLE':selected_vars, 'SCENARIO':scenarios})
            
            # All selected countries in one graph
            sel_regions=[f'{m}|{x[:-1]}' for m in models for x in list(set(fun_get_iam_regions_associated_with_countrylist(project, countrylist, m).values())) ]
            # sel_regions=[] # We do not show IAM regions

            # 1) All countries/scenarios in one graph
            df4=fun_xs(df3, {'REGION':countrylist+sel_regions}).stack().unstack('VARIABLE').reset_index()
            fig = px.scatter_ternary(df4, 
                                    a=list(ternary_dict[var].keys())[-1], 
                                    b=list(ternary_dict[var].keys())[-2], 
                                    c=list(ternary_dict[var].keys())[-3],  
                                    #hover_name="REGION",
                                    color="REGION", 
                                    size=var.replace('_v2',''), 
                                    size_max=30,
                                    title=f"Ternary graph for: {countrylist}",
                                    #color_discrete_map = {"Historic data": "grey"} ,
                                    symbol = 'SCENARIO'
                                    )
            fun_save_figure(fig, folder, f"Ternary_{var}_all_countries")
            # fig.show() # Opens graph in your browser
            # 2) We show graph by scenarios
            for scen in scenarios:
                df4=fun_xs(df3, {'REGION':countrylist,#+sel_regions,
                                 'SCENARIO':['Historic data']+[scen]
                                 }).stack().unstack('VARIABLE').reset_index()
                if scen=='Historic data':
                    df4=pd.concat([df4, fun_xs(fun_xs(df2, {"SCENARIO":scenarios}), {'VARIABLE':selected_vars,'REGION':[f'{m}|{x[:-1]}' 
                                            for m in models for x in list(set(fun_get_iam_regions_associated_with_countrylist(project, countrylist, m).values())) 
                                            ]}).fillna(0).clip(1e-10).stack().unstack('VARIABLE').reset_index()])

                fig = px.scatter_ternary(df4, 
                                        a=list(ternary_dict[var].keys())[-1], 
                                        b=list(ternary_dict[var].keys())[-2], 
                                        c=list(ternary_dict[var].keys())[-3],  
                                        hover_name="level_3", # this is time
                                        color="SCENARIO", 
                                        size=var.replace('_v2',''), 
                                        size_max=30,
                                        title=f"Ternary graph for: {scen}",
                                        color_discrete_map = {"Historic data": "grey"} ,
                                        symbol = 'REGION'
                                        )
                fun_save_figure(fig, folder, f"Ternary_{var}_{scen}")
                # fig.show() # Opens graph in your browser
            # Individual graphs for each country
            for c in countrylist:
                # TODO: slice for country belonging to region in df_iam ????
                # Plot individual countries
                # sel_regions=[f'{m}|{x[:-1]}' for m in models for x in list(set(fun_get_iam_regions_associated_with_countrylist(project, countrylist, m).values())) ]
                sel_regions=[]
                df4=fun_xs(df3, {'REGION':[c]+sel_regions}).droplevel('REGION').stack().unstack('VARIABLE').reset_index()
                df4.columns=['TIME' if 'level' in x else x for x in df4.columns]
                fig = px.scatter_ternary(df4, a=list(ternary_dict[var].keys())[-1], 
                                                b=list(ternary_dict[var].keys())[-2], 
                                                c=list(ternary_dict[var].keys())[-3],  
                                                hover_name="TIME", 
                                                color="SCENARIO",  
                                                size=var.replace('_v2',''), 
                                                title=f"Ternary graph for: {c}",  
                                                size_max=10, 
                                                color_discrete_map = {"Historic data": "grey"} ,
                                                symbol = 'MODEL'
                                                ) 
                fun_save_figure(fig, folder, f"Ternary_{var}_{c}")
                # fig.show() # Opens graph in your browser
    
    if CONSTANTS.CURR_RES_DIR(step) == CONSTANTS.CURR_RES_DIR(
        "Energy_demand_downs_1.py"
    ) or stacked_countries_plot:
        if florian_plot:
            idx = ["TIME", "ISO", "TARGET", "SECTOR", "METHOD", "MODEL"]
            d = {"ISO": countrylist, "SECTOR": sel_vars}
            df2 = fun_xs(df.set_index(idx), d)
            func_list=df2.FUNC.unique()
            df_graph = pd.concat(
                [fun_blending_with_sensitivity(df2[df2.FUNC==func].xs(m, level="MODEL")).assign(MODEL=m).assign(FUNC=func) for m in models for func in func_list]
            )
            idx = list(df_graph.index.names)
            df_graph = df_graph.reset_index().set_index(["MODEL","FUNC"] + idx)
            df_graph = df_graph.assign(UNIT="EJ/yr").set_index("UNIT", append=True)
            d = {"TARGET": "SCENARIO", "ISO": "REGION"}
            df_graph = fun_rename_index_name(df_graph[selcols], d)
            for var in sel_vars:
                for c in countrylist:
                    i = (scenarios, project, df_graph, var, c)
                    ii = i + (True, legend_color_box, legend_ls_box, 1, 0.5)
                    # To explore how it works:  CHECK HERE: jupyter_notebooks/exploring_sns_plots.ipynb!!!!
                    fun_florian_plot(*ii)

        methodlist = [None] if sel_step1_methods is None else sel_step1_methods

        fun = fun_get_iam_regions_associated_with_countrylist
        if step not in ["step1", "Energy_demand_downs_1.py"] and log_log_graphs:

            txt = "Please use a `step1` file for log-log graphs, you selected step"
            raise ValueError(f"{txt} = `{step}`")
        
        # if len(files) > 1:
        #     txt = f"Please provide a single file if you want to plot the log_log graphs. You provided `files`:{files}"
        #     raise ValueError(f"{txt}{files}")

        if not project and log_log_graphs:
            for sel in sel_vars:
                sel_pal="magma" if palette is None else palette
                fig = fun_log_log_graphs(
                    None,
                    None,
                    # will plot all countries in that region.
                    # if you want to show only selected countries, use regions[r]
                    # regions[r],
                    countrylist,  # [:6],
                    black_countries=None,
                    grey_countries=None,
                    sector=sel,
                    project=project,
                    scenario=None,
                    file_suffix=None,
                    palette=sel_pal,  # "crest",  # "viridis",#"mako",  #'GnBu', 'mako','brg', ""hsv","cool"
                    step1_sensitivity=step1_sensitivity,
                    step1_extended_sens=step1_extended_sensitivity,
                    sel_method=None,  # ['wo_smooth_enlong_ENLONG_RATIO'],
                    remove_countries_wo_hist_data=iea_countries_only,
                    # ls_dict={"ENSHORT_HIST": "--"},
                )
                folder = (
                    CONSTANTS.CURR_RES_DIR("step5")
                    / "Step5e_visuals"
                    / 'hist_data'
                )
                figname = f"Log_log_hist_data"
                fun_save_figure(fig, folder, figname)
                plt.close()
            print('Exit function, as project is None. If you want to do more, please select a project')
            return None
        else:
            if stacked_countries_plot:
                # Open in full screen and save the image
                stacked_data_step5(project, files_dict, countrylist, iea_countries_only, sel_vars, scenarios,df, 
                suptitle_font=20, font_title=16, myfont=16, font_legend=14,
                font_label=16, font_tick=13, save_figure=True, 
                my_legend_loc=[-8, -0.3], figsize=(20,10), wspace=0.6)
                # stacked_data_step5(project, files_dict, countrylist, iea_countries_only, sel_vars, scenarios,df)
            if log_log_graphs:
                # Read IAM data and index it properly
                df_iam = fun_index_names(fun_read_df_iams(project), True, int)

                for model, file in files_dict.items():
                    regions_all = fun_regional_country_mapping_as_dict(model, project, iea_countries_only)
                    regions_to_be_downs = {k: v for k, v in regions_all.items() if len(v) > 1}
                    regions = regions_to_be_downs

                    if countrylist is not None:
                        regions = fun_get_iam_regions_associated_with_countrylist(project, countrylist, model)
                        regions = fun_invert_dictionary({k: [v] for k, v in regions.items()})
                        # Exclude Native countries (no csv file for native countries)
                        regions = {k: v for k, v in regions.items() if k in regions_to_be_downs}

                    for r in regions:
                        if not isinstance(file, list):
                            file = [file]
                        selfile = [x for x in file if r in str(x)]
                        res = {}
                        for sel in sel_vars:
                            for s in scenarios:
                                # mytext=file.split(model)[-1].split('_')
                                available_scen=df[(df.MODEL == model)].TARGET.unique() if step in ["step1", "Energy_demand_downs_1.py"] else df.reset_index().SCENARIO.unique()
                                if s in available_scen:
                                    for n, mt in enumerate(methodlist):
                                        print(f"{model}_{r} - {s}")
                                        mt = mt if mt is None else [mt]    
                                        sel_pal="winter" if palette is None else palette
                                        # fun_log_log_graphs(
                                        #     model,
                                        #     r,
                                        #     # will plot all countries in that region.
                                        #     # if you want to show only selected countries, use regions[r]
                                        #     # regions[r],
                                        #     ['AUS'],  # [:6],
                                        #     black_countries=None,
                                        #     sector=sel,
                                        #     project=project,
                                        #     scenario=s,
                                        #     file_suffix=selfile,
                                        #     palette=sel_pal,  # "crest",  # "viridis",#"mako",  #'GnBu', 'mako','brg', ""hsv","cool"
                                        #     step1_sensitivity=False,
                                        #     step1_extended_sens=False,
                                        #     sel_method=['ENSHORT_HIST','ENLONG'],  # ['wo_smooth_enlong_ENLONG_RATIO'], ['ENSHORT_HIST', 'ENLONG'],
                                        #     remove_countries_wo_hist_data=iea_countries_only,
                                        #     # ls_dict={"ENSHORT_HIST": "--"},
                                        # )

                                        ## Temportary just for test below
                                        # fun_log_log_graphs(
                                        #     model,
                                        #     r,
                                        #     # will plot all countries in that region.
                                        #     # if you want to show only selected countries, use regions[r]
                                        #     # regions[r],
                                        #     regions_all[r],  # [:6],
                                        #     black_countries=None,
                                        #     grey_countries=None,
                                        #     sector=sel,
                                        #     project=project,
                                        #     scenario=s,
                                        #     file_suffix=selfile,
                                        #     palette=sel_pal,  # "crest",  # "viridis",#"mako",  #'GnBu', 'mako','brg', ""hsv","cool"
                                        #     step1_sensitivity=False,
                                        #     step1_extended_sens=step1_extended_sensitivity,
                                        #     sel_method=['ENSHORT_HIST',],# 'ENSHORT_REF'],
                                        #     remove_countries_wo_hist_data=iea_countries_only,
                                        #     func_type='s-curve',
                                        #     # func_type='log-log'
                                        #     # ls_dict={"ENSHORT_HIST": "--"},
                                        # )

                                        fig = fun_log_log_graphs(
                                            model,
                                            r,
                                            # will plot all countries in that region.
                                            # if you want to show only selected countries, use regions[r]
                                            # regions[r],
                                            regions_all[r],  # [:6],
                                            black_countries=None,
                                            sector=sel,
                                            project=project,
                                            scenario=s,
                                            file_suffix=selfile,
                                            palette=sel_pal,  # "crest",  # "viridis",#"mako",  #'GnBu', 'mako','brg', ""hsv","cool"
                                            step1_sensitivity=step1_sensitivity,
                                            step1_extended_sens=step1_extended_sensitivity,
                                            sel_method=mt,  # ['wo_smooth_enlong_ENLONG_RATIO'], #['ENSHORT_HIST', 'ENLONG'],
                                            remove_countries_wo_hist_data=iea_countries_only,
                                            # ls_dict={"ENSHORT_HIST": "--"},
                                        )
                                        folder = (
                                            CONSTANTS.CURR_RES_DIR("step5")
                                            / "Step5e_visuals"
                                            / project
                                            / model
                                            /'log_log'
                                        )
                                        folder = (
                                            folder / f"method_{n}_{mt[0]}"
                                            if mt is not None
                                            else folder
                                        )
                                        figname = f"{r}_{s}"
                                        fun_save_figure(fig, folder, figname)
                                        plt.close()


    # Regional model mapping comparison e.g. if `countrylist=[MESSAGEix-GLOBIOM 1.1-M-R12|Western Europe]`
    model_reg_comparison = {x.rsplit("|")[0] for x in countrylist if "|" in x}
    if len(model_reg_comparison):
        df = df.droplevel("FILE")
        for m in model_reg_comparison:
            df = fun_create_regional_variables_as_sum_of_countrylevel_data(
                m,
                project,
                df,
                sel_vars,
            )  # .reset_index().REGION.unique()

    if isinstance(sel_scen, str):
        sel_scen = pd.read_csv(CURR_RES_DIR / sel_scen)
        sel_scen = fun_available_scen(fun_index_names(sel_scen, True, int))
        legend_el = [
            Line2D([0], [0], color="black", lw=4, label="Historical"),
            Line2D([0], [0], color="b", lw=4, label="Compliant"),
            Line2D([0], [0], color="grey", lw=4, label="Non-compliant"),
        ]
    elif sel_scen is None:
        sel_scen = fun_available_scen(df)
        legend_el = [
            Line2D([0], [0], color="black", lw=4, label="Historical"),
            Line2D([0], [0], color="b", lw=4, label="Scenarios"),
        ]

    markers_dict["EU27"] = {
        "marker_2030": {2030: 2110.82},
        "marker_2050": {2050: 300},
    }

    if not countrylist:
        countrylist = df.reset_index().REGION.unique()

    if CONSTANTS.CURR_RES_DIR(step)!=CONSTANTS.CURR_RES_DIR('step1'):
        df_graph = df[selcols]
        na_cols = df_graph.columns[df_graph.columns > 2050]
        df_graph.loc[:, na_cols] = np.nan

    if "EU27" in countrylist and 'EU27' not in df_graph.reset_index().REGION.unique():
        df_graph = fun_aggregate_countries_general(
            df_graph,  # .droplevel(['FILE']),
            "EU27",
            ["MODEL", "VARIABLE", "UNIT", "SCENARIO", "FILE"],
            fun_eu27(),
            remove_single_countries=False,
        )
        df_graph["FILE"] = "not available"
        df_graph = df_graph.set_index("FILE", append=True)

    # Create dashboard
    if create_dashbord is not None:
        fun_create_dashboard_pdf_by_country(
            df_graph.loc[:, selcols],
            {
            'models':['*'],    
            'scenarios':scenarios,
             "Secondary Energy|Electricity": {
                    "by": "fuel",
                    # "row": 0,
                    # "col": 2,
                    "same_level": True,
                    "kind": "area",
                    # "ylim": (0, 60),
                }
            },
            countrylist,
            CONSTANTS.CURR_RES_DIR("step5") / "Step_5e_dashboard_by_country",
            create_dashbord, 
            colors=colors,
        )
        fun_create_dashboard_pdf(
            df_graph.loc[:, selcols],
            NGFS_dashboard_bar_plots,
            countrylist,
            CONSTANTS.CURR_RES_DIR("step5") / "Step_5e_dashboard",
            create_dashbord,
            colors=colors,
        )

    for var in sel_vars:
        top = None

        # we plot the `compliant` 63 scenarios in blue. The rest in grey
        if not isinstance(sel_scen, list):
            blue = fun_select_model_scenarios_combinations(df_graph, sel_scen)
        else:
            blue = fun_select_model_scenarios_combinations(
                df_graph,
                fun_dict_level1_vs_list_level2(
                    fun_xs(df_graph, {"SCENARIO": sel_scen}), "MODEL", "SCENARIO"
                ),
            )

        if ngfs_plot:
            if isinstance(sel_scen, dict):
                sel_scen=list(find_intersection_among_list_of_lists(list(sel_scen.values())))
            fig = fun_line_plots_colors_and_markers(
                        fun_xs(df_graph, {"SCENARIO": sel_scen, 'VARIABLE':var}),
                        var,
                        countrylist,
                        color_dim="REGION",
                        marker_dim="MODEL",
                    )
            fun_save_figure(
                        fig,
                        CONSTANTS.CURR_RES_DIR("step5") / "Step5e_visuals" / project,
                        f"NGFS_{var.replace('|', '_')}_indexed_{str(countrylist)}",
                    )
        
        for c in countrylist:
            print(c)
            try:
                if eu_ab_plot:
                    fun_plot_eu_ab_side_by_side(
                        df_graph.xs(c, level="REGION", drop_level=False).replace(
                            0, np.nan
                        ),
                        var,
                        c,
                        2030,
                        ylim_top=top,
                        color="grey",
                        sel_model=fun_available_scen(blue),
                        sel_model_color="blue",
                        marker_2030=markers_dict.get(c, {"marker_2030": None})[
                            "marker_2030"
                        ],
                        marker_2050=markers_dict.get(c, {"marker_2050": None})[
                            "marker_2050"
                        ],
                        # marker_2050=markers_dict[c]["marker_2050"],
                        legend_elements=legend_el,
                    )

                # NGFS graphs
                if ngfs_plot:
                    for indexed_results in [False, True]:
                        fig = fun_line_plots_colors_and_markers(
                            fun_xs(df_graph, {"SCENARIO": sel_scen}),
                            var,
                            c,
                            color_dim="SCENARIO",
                            marker_dim="MODEL",
                            growth_rate=indexed_results
                        )
                        indexed='_indexed' if indexed_results else ''
                        fun_save_figure(
                            fig,
                            CONSTANTS.CURR_RES_DIR("step5") / "Step5e_visuals" / project,
                            f"NGFS_{var.replace('|', '_')}_{c}{indexed}",
                        )

                if florian_plot:
                    i = (scenarios, project, df_graph, var, c)
                    ii = i + (True, legend_color_box, legend_ls_box, 1, 0.5)
                    # To explore how it works:  CHECK HERE: jupyter_notebooks/exploring_sns_plots.ipynb!!!!
                    fun_florian_plot(*ii)

            except:
                raise Exception(f"graph did not work for {var},{c}")
        if analyse_drivers:
            fun_analyse_main_drivers_across_scenarios(
                df_graph, var, c, n=150, exclude_similar_variables=False
            )

def fun_combine_scenario_step5e_visual(step:str, project:str, sub_folder:Optional[str]=None)->pd.DataFrame:
    if CONSTANTS.CURR_RES_DIR('step5')==CONSTANTS.CURR_RES_DIR(step):
        combi=fun_combine_df_with_most_recent_scenarios(project, step, save_to_csv=False, sub_folder=sub_folder)
        if len(combi)>0:
            fixed_suffix='2020_harmo_step5e_WITH_POLICY_None.csv'
            txt1='We will combine all the most recent scenarios from each model into a new file, and save it with this file suffix'
            action=input(f'{txt1}: {fun_current_file_suffix()}_{fixed_suffix}. Do you wish to continue? y/n')
            if action.lower() in ["yes", "y"]:
                for model in combi.reset_index().MODEL.unique():
                    filename=CONSTANTS.CURR_RES_DIR('step5')/f'{model}_{project}_{fun_current_file_suffix()}_{fixed_suffix}'
                    fun_xs(combi, {'MODEL':model}).to_csv(filename)
            else:
                raise ValueError('Aborted by the user')
        else:
            print('Each model already has all scenarios in one file - there is nothing to combine')
        return combi       
    else:    
        action=input(f'Are you sure you want to create a new df with the most recent scenarios in {step}, and save it to csv?')
        if action.lower() in ["yes", "y"]:
            res=fun_combine_df_with_most_recent_scenarios(project, step, save_to_csv=True, 
                                                            sub_folder=sub_folder
                                                            )
            return res

def fun_box_plot_step2(files:dict, models:list, project:str,t=2050, fuels=['OIL', 'COAL', 'GAS', 'NUC', 'HYDRO', 'BIO', 'GEO', 'SOL',  'WIND'])->plt.plot:
    """Creates a box plot for 'Secondary Energy|Electricity' by fuel, based on the sensitivity analysis results from step2b, 
    (range of results based on diffent criteria weights).

    Parameters
    ----------
    files : dict
        Your dictionary where to find files for each model: {model:path_to_csv_file}
    models : list
        Your list of models
    project : str
        your project e.g (NGFS_2023)
    t : int, optional
        Selected time period for your box plot, by default 2050
    fuels : list, optional
        List of fuels, by default ['OIL', 'COAL', 'GAS', 'NUC', 'HYDRO', 'BIO', 'GEO', 'SOL',  'WIND']

    Returns
    -------
    plt.plot
        Your box plot for secondary energy electricity generation

    Raises
    ------
    ValueError
        If `files` is not a dictionary.

    Notes
    -----
    - The function reads data from CSV files specified in `files` and indexed by `models`.
    - For each model, it generates a box plot of fuel types for each country at the specified time period.
    - The plots are saved in a directory structure based on the project and model names.
    """    
    for m in models:
        if not isinstance(files, dict):
            txt='Box plot for step2b requires `files` as a dictionary.'
            raise ValueError(f'{txt}. You passed: {type(files)} {files}')
        df= pd.read_csv(CONSTANTS.CURR_RES_DIR('step2')/'step2b_sensitivity'/f"{files[m]}")
        countrylist=list(df.ISO.unique())
        df=df.set_index(['TIME','ISO',  'CRITERIA', 'METHOD'])
        for c in countrylist:
            dfsel=df.xs(c, level='ISO').xs('dynamic_recursive', level='METHOD').loc[t,fuels]
            fig=fun_box_plot(dfsel)
            folder=CONSTANTS.CURR_RES_DIR("step5") / "Step5e_visuals" / project/m
            folder=folder/'step2b_sensitivity'
            # n= number of randomly generated criteria in each distribution (red and blue)
            plt.title(f"{c} in {t} [n={int(np.round(len(dfsel)/2,0))}]")
            fun_save_figure(fig,folder, f"Boxplot_elc_before_trade_min_{c}_{t}")             
            plt.close()
    return plt

def fun_patterns_summary_by_country(df:pd.DataFrame, df_iea:pd.DataFrame, c:str, scen:str,  ref_future_scen:str='h_cpol',models:Optional[list]=None, K:int=1,     
    future_time_periods:list=[2020,2035],
    hist_time_periods:list=[2000,2015], show_val:bool=False)->pd.DataFrame:
    """Summarize patterns for a given country `c` for historical and future scenarios based on the provided dataframes (`df_iea` and `df`, respectively).
    It returns a dataframe.

    It find similar patterns for a given country `c` and a given scenario `scen`, based on historical data from `df_iea` and reference 
    future scenario (`ref_future_scen` contained in the `df`).
    Each country `c` will be in the index of the returned dataframe. Each country will have K countries for each model (with most similar patterns), for
    historical and a future scenario (columns).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing scenario data.
    df_iea : pd.DataFrame
        DataFrame containing IEA data.
    c : str
        Country code for which patterns are to be analyzed.
    scen : str
        Scenario name for which patterns are to be analyzed.
    ref_future_scen : str, optional
        Reference future scenario name (default is 'h_cpol').
    models : list of str, optional
        List of models to consider (default is None, which means all models in `df`).
    K : int, optional
        Number of countries witn best pattern match to consider for each model(default is 1).
    future_time_periods:list
        List of future time periods to be considered: [from, to], by default [2020,2035]
    hist_time_periods:list
        List of historical time periods to be considered: [from, to], by default [2000,2015]
    show_val: bool
        Whether you want to show values associated with pattern matching (instead of country with best macth).
        This gives you an indication of how difficult (high value) /easy (low value) was to find a match, by default False
    
    Returns:
    --------
    pd.DataFrame
        Summary DataFrame containing historical and future patterns by country.
    """

    # Check input parameters
    sanity_check_inputs_of_fun_patterns(df, scen, ref_future_scen, future_time_periods, hist_time_periods)
    
    # Initialize dictionaries to store results
    myres_future_all={}
    myres_hist_all={}
    
    # If models are not provided, consider all unique models in the dataframe
    models=df.reset_index().MODEL.unique() if models is None else models

    # Iterate over each model
    for model in models:
        # Find future patterns
        myres=find_historic_pattern(fun_xs(df, {'SCENARIO':ref_future_scen}), fun_xs(df, {'SCENARIO':scen}), c, model, scen, 
                                    time_mapping={min(future_time_periods):min(future_time_periods), 
                                                max(future_time_periods):max(future_time_periods)})
        myres=list({k:np.round(v,3) for k,v in myres.items() if k!=c}.items())[0: K]
        myres_future_all[model]=sum(list(dict(myres).values())) if show_val else list(dict(myres).keys())

        # Find historical patterns
        myres=find_historic_pattern(df_iea, fun_xs(df, {'SCENARIO':scen}), c, model, scen, 
                                    time_mapping={min(future_time_periods):min(hist_time_periods), 
                                                max(future_time_periods):max(hist_time_periods)})
        myres=list({k:np.round(v,3) for k,v in myres.items() if k!=c}.items())[0: K]
        myres_hist_all[model]=sum(list(dict(myres).values())) if show_val else list(dict(myres).keys()) 

    # Create a DataFrame with the results and return
    return pd.DataFrame([{f"Historical {hist_time_periods}":myres_hist_all.values(),f'{ref_future_scen} {future_time_periods} ':myres_future_all.values()}], index=[c])

def sanity_check_inputs_of_fun_patterns(df:pd.DataFrame, scen:str, ref_future_scen:str, future_time_periods:list, hist_time_periods:list):
    
    """Sanity check of input parameters used in function `fun_patterns_summary_by_country`.

    df : pd.DataFrame
        DataFrame containing scenario data.
    scen : str
        Scenario name for which patterns are to be analyzed.
    ref_future_scen : str, optional
        Reference future scenario name (default is 'h_cpol').
    future_time_periods:list
        List of future time periods to be considered: [from, to], by default [2020,2035]
    hist_time_periods:list
        List of historical time periods to be considered: [from, to], by default [2000,2015]
    Raises
    ------
    ValueError
        If `ref_future_scen` is not present in the `df`
    ValueError
        If `scen` is not present in the `df`
    ValueError
        If historical and future time period dot not have the same time lenght
    """    
    
    # Raise Value Error if selected scenarios are not present in `df`
    scen_available=df.reset_index().SCENARIO.unique()
    if ref_future_scen not in scen_available:
        raise ValueError(f'Your selected `ref_future_scen` {ref_future_scen} is not present in the `df`. Scenarios available: {scen_available}')
    if scen not in scen_available:
        raise ValueError(f'Your selected `scen` {scen} is not present in the `df`. Scenarios available: {scen_available}')
    
    # Sanity check for time periods:
    future_time_period_lenght=max(future_time_periods)-min(future_time_periods)
    hist_time_period_lenght=max(hist_time_periods)-min(hist_time_periods)
    if future_time_period_lenght!= hist_time_period_lenght:
        txt='Historical and future time period must have the same time lenght. They contain'
        raise ValueError(f'{txt} {hist_time_period_lenght} ({hist_time_periods}) and {future_time_period_lenght} ({future_time_periods}) years respectively')

def fun_validate_IEA(df:pd.DataFrame)->dict:
    """Validates new IEA data by:
    1) comparing with previous IEA data (based on `fun_most_recent_iea_data`) -> and show plots 
       for variables with large differences and
    2) running consistency checks (print results). Normally there are issues with UZB in 1994 
       (industry/buildings/transport unable to explain total final energy demand)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with new IEA data in IAMC format

    Returns
    -------
    dict
        Dictiorary with largest discrepancies compared to previous IEA data
    """    
    # Step1 compared with previous IEA data
    df_iea=fun_most_recent_iea_data()
    df_iea=df_iea.reset_index().set_index(df.index.names)

    check_var=list(set(df_iea.reset_index().VARIABLE.unique())&set(df.reset_index().VARIABLE.unique()))
    res={}
    for var in check_var:
        # common=set(df_iea.xs(var, level='VARIABLE').reset_index().REGION.unique())&set(df.xs(var, level='VARIABLE').reset_index().REGION.unique())
        # check=pd.concat([fun_xs(df_iea, {'REGION':list(common)}).xs(var, level='VARIABLE').sum(), fun_xs(df, {'REGION':list(common)}).xs(var, level='VARIABLE').sum()], axis=1)
        # check=pd.concat([df_iea.shift(1, axis=1).drop(['World','EU27'], level='REGION').xs(var, level='VARIABLE').sum(), df.xs(var, level='VARIABLE').sum()], axis=1)
        check=pd.concat([df_iea.drop(['World','EU27'], level='REGION').xs(var, level='VARIABLE').sum(), df.xs(var, level='VARIABLE').sum()], axis=1)
        checkval=np.abs((check[0]/check[1]).loc[2015]-1)
        if checkval>0.1:
            check.plot()
            plt.legend(labels=['df_iea','df'])
            plt.title(var)
            plt.show()
        res[var]=checkval
     
    res=fun_sort_dict(res, by='values', reverse=True)

    # Step2 runs consistency checks 
    ec = [ "Liquids",  "Solids", "Electricity","Gases", "Hydrogen", "Heat",]
    sectors= ["Industry", "Transportation", "Residential and Commercial"]
    var_dict_demand = fun_make_var_dict(sectors, ec, demand_dict=True)[0]
    var_dict_supply = fun_make_var_dict(ec, sectors, demand_dict=False)
    
    # 1) Show inconsistencies, prioritize 'Final Energy' if any inconsistencies are found
    print("\n 1 Prioritize 'Final Energy' if inconsistencies are found:")
    print(show_inconsistencies(df[list(df.columns)[:-1]], [var_dict_demand, var_dict_supply], priority='Final Energy'))

    # 2) Show inconsistencies, largest inconsistencies found
    print("\n 2 Show inconsistencies, largest inconsistencies found:")
    print(show_inconsistencies(df[list(df.columns)[:-1]], [var_dict_demand, var_dict_supply], priority=None))
    return res

def fun_create_iea_file_IAMC_format(file_path:Path)->pd.DataFrame:
    """Creates a iea dataframe in IAMC format.

    Parameters
    ----------
    file : Path
        Path to your IEA file (normally it should be located in the `CONSTANTS.INPUT_DATA_DIR` folder).

    Returns
    -------
    pd.DataFrame
        IEA dataframe in IAMC format
    """    
    folder=file_path.parent
    file=str(file_path).split('\\')[-1]
    tmp_dir=folder/'tmp'
    
    if os.path.exists(tmp_dir):
        txt='Before creating the IEA file, you must (manually) delete this temporary folder'
        raise ValueError(f"{txt}: {tmp_dir}") 

    iea_drop_col_dict={'Extended_IEA_en_bal_2021.csv':['MEASURE', 'Country','Flags', 'Flag Codes', 'Time','Unit','FLOW','PRODUCT'],
                      'IEA_Balances_Rev2022.csv':['Flag Codes','MEASURE']
                      }
    main_unit_iea_dict={'Extended_IEA_en_bal_2021.csv':'ktoe', 'IEA_Balances_Rev2022.csv':'TJ'}
    main_unit=main_unit_iea_dict.get(file, 'TJ')
    print(f'We assume the unit of this file is: {main_unit}')
    fun_read_large_iea_file(folder/file, drop_cols=iea_drop_col_dict[file], main_unit=main_unit)
    df=fun_concatenate_chunks(tmp_dir)
    df=fun_finalize_iea_files_in_IAMC_format(tmp_dir/'IAMC', main_unit=main_unit)
    year=file.split('.')[0][-4:]
    df.to_csv(folder/f'input_reference_iea_{year}.csv')
    return df

# def fun_save_large_iea_file_in_chunks()
def fun_read_large_iea_file(large_file_path:Path, chunk:int=64**4, drop_cols=None, main_unit='ktoe'):
    """Read a large IEA file in chunks, filter data, and save each chunk as a CSV.

    Parameters:
    -----------
    large_file_path : Path
        Path to the large IEA file to be read.
    chunk : int, optional
        Size of each chunk to read from the file, by default 64**4
    main_unit: str
        Unit of the dataframe, by default 'ktoe' 

    Returns:
    --------
    None

    Notes:
    ------
    This function reads a large IEA file in chunks, filters the data by selecting
    rows where the 'Unit' column equals 'ktoe', performs some data manipulation,
    and then saves each chunk as a separate CSV file in a 'tmp' directory located
    next to the original file.
    """
    # Part 1) Read  large file by iterating over chunks and save all files in a `new_dir` folder
    df_all=pd.read_csv(large_file_path, chunksize=chunk)
    count=0
    # error_list=[]
    if drop_cols is None:
        drop_cols=['MEASURE', 'Country','Flags', 'Flag Codes', 'Time','Unit','FLOW','PRODUCT']
    # We iterate over these chunks:
    for df in df_all: 
        # Slice for correct unit
        if 'Unit' in df.columns:
            unitcol='Unit' 
        elif 'MEASURE' in df.columns:       
            unitcol='MEASURE'
        else:
            raise ValueError(f'`df.columns` does not contain `Unit` nor `MEASURE`: it contains {df.columns}')
        if main_unit in df[unitcol].unique() and len(df[df[unitcol]==main_unit])>0: 
            df=df[df[unitcol]==main_unit]
        else:
            print(f'Unable to slice for {main_unit}. These are the units present in the df {df[unitcol].unique()}')
        if len(df)>0:
            for x in drop_cols:
                if x in df.columns:
                    df=df.drop(x, axis=1)
            df.columns=[x.upper() for x in df.columns]
            df=df.set_index(['COUNTRY','PRODUCT','FLOW','TIME']).dropna()
            df=fun_rename_index_name(df, {'COUNTRY':'ISO'})
            count+=1
            
            # Save csv for each chunk
            new_dir=large_file_path.parent/'tmp'
            new_dir.mkdir(exist_ok=True)
            df.to_csv(new_dir/f'Extended_IEA_en_bal_2021_ISO_{count}.csv')
            print(f"Saving csv chunk {count}... in this folder {new_dir}")
            

def fun_concatenate_chunks(tmp_folder:Path, conversion_factor = 0.041868 / 1e3)->pd.DataFrame:
    """Concatenate CSV files located in a temporary folder and save them in IEA and IAMC formats.

    Parameters:
    -----------
    tmp_folder : Path
        Path to the folder containing CSV files.
    conversion_factor: float
        Conversion factors (to convert standard unit of the dataframe to EJ)
    Returns:
    --------
    df : DataFrame
        Concatenated DataFrame from all CSV files.

    Notes:
    ------
    This function concatenates CSV files located in a temporary folder and saves them in two different formats:
    - IEA format: CSV files contain data grouped by 'Product', 'Flow', and 'ISO'. 
    - IAMC format: CSV files contain data with IAMC variable names instead of 'Product' and 'Flow'.
    """
    # Read all CSV files in the temporary folder and concatenate them
    df=pd.concat([pd.read_csv(tmp_folder/x) for x in os.listdir(tmp_folder) ])
    
    # Get unique ISO codes and include only those present in the list of all countries
    myregions=df.reset_index().ISO.unique()
    myregions=[x for x in myregions if x in all_countries]


    df=df.set_index(['ISO','PRODUCT','FLOW','TIME'])

    
    # List to store ISO codes causing errors during processing
    error_list=[]
    
    # Part 1: save one file for each in region. Dataframe will be saved in:
    # a) `IEA_dir` folder  -> using a IEA wide format (including all `Product`/`Flow`). Example: `Total final consumption`, `Total`
    # b) `IAMC_dir` folder -> using an IAMC format (using IAMC `VARIABLE` names instead of `Product`/`Flow`). Example 'Final Energy'
    
    # Iterate over each group of ISO codes
    print('Save individual countries:')
    for count, myr in enumerate(myregions):
        # Slice DataFrame for the current ISO code
        dfr=fun_xs(df, {'ISO':myr})
        if len(dfr):
            # Group and aggregate data by 'Product', 'Flow', and 'TIME'
            dfr=dfr.groupby(['ISO', 'PRODUCT', 'FLOW','TIME']).sum().unstack('TIME')
            
            # Reset index and format DataFrame
            dfr=dfr['VALUE']
            dfr=fun_index_names(dfr.reset_index(), True, int)
            
            # Create directories for saving CSV files in IEA and IAMC formats
            IEA_dir=tmp_folder/'IEA_FORMAT'
            IEA_dir.mkdir(exist_ok=True)

            IAMC_dir=tmp_folder/'IAMC'
            IAMC_dir.mkdir(exist_ok=True)

            # Save dataframe in IEA format (by product/flow)
            dfr.to_csv(IEA_dir/f"chunk_{count}_{myr}.csv")
            try:
                # Convert DataFrame to IAMC format and save
                dfr=dfr.rename(iea_product_long_short_dict, level='PRODUCT')
                dfr=dfr.rename(iea_flow_long_short_dict, level='FLOW')
                dfr= fun_read_energy_iea_data_in_iamc_format(
                    iea_flow_dict, save_to_csv=False, verbose=False, df=dfr, conversion_from_ktoe_to_ej=conversion_factor
                )
                print(f'saving chunk {count} {myr}')
                dfr.to_csv(IAMC_dir/f"chunk_{count}_{myr}.csv")
            except:
                error_list+=[myr]
    # Print ISO codes causing errors during processing
    if len(error_list)>0:
        print('Errors for these regions:')
        print(set(error_list))

    return df

def fun_finalize_iea_files_in_IAMC_format(IAMC_dir:Path, direct_equivalent_variables:list=['Primary Energy|Nuclear','Primary Energy|Geothermal'], main_unit='ktoe')->pd.DataFrame:
    """Concatenate CSV files in IAMC format located in a directory and process them.

    Parameters:
    -----------
    IAMC_dir : Path
        Path to the directory containing CSV files in IAMC format.
    direct_equivalent_variables: list
        List of variables to be reported using a direct equivalent conversions (will be divided by 3)
    Returns:
    --------
    df : DataFrame
        Concatenated and processed DataFrame.

    Notes:
    ------
    This function reads all CSV files in IAMC format located in the specified directory, concatenates them, 
    and performs the following processing steps:
    1. Sum up possible duplicated variables (reason: incomplete variables may have been splitted in different chunks).
    2. Convert secondary energy variables from GWH to Ktoe.
    3. Convert list of `direct_equivalent_variables` using a direct equivalent conversion factor (divide by 3).
    4. Update total primary energy as the sum of fuels.
    5. Create aggregated `fossil` variables in `Primary Energy` and `Secondary Energy|Electricity`
    """
    df=pd.concat([pd.read_csv(IAMC_dir/x) for x in os.listdir(IAMC_dir)])
    df=fun_index_names(df, True, int)
    res={}
    # 1) Sum up possible duplicated values (some 'products/'flow' may have been splitted in different chunks, leading to duplicated variables)
    for var in df.reset_index().VARIABLE.unique():
        idx=df.index.names
        sumby=[x for x in idx if x not in 'VARIABLE']
        res[var]=df.xs(var, level='VARIABLE').groupby(sumby).sum().assign(VARIABLE=var).reset_index().set_index(idx)
    df=fun_drop_duplicates(pd.concat(list(res.values())))
    
    # 2) Add conversion for secondary energy variables (originally reported in GWH): from GWH to Ktoe -> 
    # NOTE: final results are  in EJ/y (as we have previously converted all variables from ktoe to EJ/yr)
    secondary=[x for x in df.reset_index().VARIABLE.unique() if 'Secondary' in x]
    df_secondary=fun_xs(df, {'VARIABLE':secondary})/11.63 # Conversion from GWH to ktoe
    df.loc[df_secondary.index]=df_secondary # Update secondary energy variables

    # 3) Use using direct equivalent conversion factor for nuclear and Geothermal

    for primary in direct_equivalent_variables:
        df_primary=fun_xs(df, {'VARIABLE':primary})/3 # Direct equivalent for nuclear
        df.loc[df_primary.index]=df_primary # Update dataframe
    
    # Adjust unit (excpet for electricity, which unit is Gwh)
    if main_unit in ['TJ', 'TJ/yr']:
        df_main=fun_xs(df, {'VARIABLE':[x for x in df.reset_index().VARIABLE.unique() if 'Secondary Energy|Electricity' not in x]})/(41.868)
        df.loc[df_main.index]=df_main
    elif main_unit!='ktoe':
        raise ValueError(f"Main units supported are: ['ktoe', 'TJ', 'TJ/yr'], you choose: {main_unit}")


    # 4) Update total primary energy as the sum of fuels
    df=df.drop('Primary Energy', level='VARIABLE')
    df=fun_create_var_as_sum(df, 'Primary Energy', step2_primary_energy['Primary Energy'], unit='EJ/yr')

    # 5) Create total for fossils
    for i in ['Primary Energy','Secondary Energy|Electricity']:
        myfossil=[f'{i}|{fuel}' for fuel in ['Coal', 'Oil','Gas']]
        df=fun_create_var_as_sum(df,f'{i}|Fossil', myfossil, unit='EJ/yr')
    return df

def collage_different_files(project, models, 
            file_name_out, # ,
            mydict, ):
    
    
    for model in models:
        mydict={k:f"{CONSTANTS.CURR_RES_DIR('step5')}/{model}_{v}" if 'ssp' not in k else f'{CONSTANTS.INPUT_DATA_DIR/project/v}' for k,v in mydict.items()}

        # Read GDP and Pop
        res=fun_read_csv({'df_ssp':CONSTANTS.INPUT_DATA_DIR/project/'SSP_projections.csv'}, True, int)
        
        # Read `df` and `df_add` (to be added to main `df`)
        # res['df']=fun_read_csv_or_excel('NGFS_2024_2024_04_10_2018_harmo_step5e_WITH_POLICY_Scenario_Explorer_upload_FINAL.xlsx', [model]).droplevel('FILE')
        # res['df_add']=fun_read_csv_or_excel('NGFS_2024_2024_04_11_2018_harmo_step5e_Scenario_Explorer_upload_FINAL.xlsx', [model]).droplevel('FILE')

        res=fun_read_csv(
            mydict, 
            True, int
            )

        # Make list of countries to be added to `df`
        df=res['df']
        for x in ['df_add', 'df_add2']:
            if x in res:
                add_countries=list(res[x].reset_index().REGION.unique())
                # Select countries
                df=pd.concat([fun_xs(df, {'REGION':add_countries}, exclude_vars=True), res[x]])

        # Add missing GDP/Pop to main `df` (from `df_ssp`) -> we call this `df_final`
        df_ssp=fun_rename_index_name(res['df_ssp'], {'ISO':'REGION'}).xs(model, drop_level=False)
        idx_missing_socioeconomic=[x for x in df_ssp.index if x not in df.index]
        df_final=pd.concat([df, df_ssp.loc[idx_missing_socioeconomic]])


        # df_final =df_final.rename({'Emissions|Kyoto Gases (incl. indirect LULUCF)': 'Emissions|Kyoto Gases (incl. indirect AFOLU)'}, level='VARIABLE')
        df_final =df_final.rename({'Emissions|Kyoto Gases (incl. indirect AFOLU)': 'Emissions|Kyoto Gases (incl. indirect LULUCF)'}, level='VARIABLE')
        # Aggregate non-iea countries -> we call it `df_final_agg`
        iso_to_be_aggregated=[x for x in df.reset_index().REGION.unique() if x not in iea_countries+['EU27']]
        df_final_agg=fun_aggregate_countries_general(df_final,'downscaling|countries without IEA statistics', iso_to_be_aggregated, remove_single_countries=True)

        # Check list of non-iea countries (see print)
        print([x for x in df_final_agg.reset_index().REGION.unique() if x not in iea_countries])
        df_final_agg=fun_drop_duplicates(df_final_agg)

        # Select scenarios
        scenarios=list(fun_get_scenarios(project))
        df_final_agg=fun_xs(df_final_agg, {'SCENARIO':scenarios})
        
        # Check if there are still missing data
        for var in ['Final Energy', 'Primary Energy','GDP|PPP','Population']:
            missing=fun_check_missing_data(df_final_agg.xs(var, level='VARIABLE'), 'REGION', 'SCENARIO')
            if len(missing)>0:
                raise ValueError(f'Missing {var} data for {missing}')
            
        
        df_final_agg=df_final_agg.rename({model:f"Downscaling[{model}]"}, level='MODEL')
        # Save to excel
        fname=str(CONSTANTS.CURR_RES_DIR('step5')/f"{model}_{file_name_out}")
        fun_save_to_excel(
                        project,  # "NGFS_2023",
                        fname,
                        model,
                        df_final_agg,
                        None,
                    )
        return df_final_agg

def find_all_patterns(df:pd.DataFrame, model:str, scen:str, max_number_of_keys=6, time_mapping={2000:2000, 2015:2015})->dict:
    """Find  similar transition patterns across countries in a given scenario (e.g. 'Historical data'), for a given model and a given time period e.g. 2000-2015. 

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe with county level results
    model : str
        Your selected model
    scen : str
        Your selected scenario
    max_number_of_keys : int, optional
        Max number of patterns, by default 6
    time_mapping : dict, optional
        Time period mapping {from:from, to:to}, by default {2000:2000, 2015:2015}

    Returns
    -------
    dict
        Dictionary with country groups
    """    
    # This function aims at finding similar transition patterns in a given scenario (e.g. HIstorical data), for a given time period e.g. 2000-2015. 
    # It groups countries based on similar patterns

    print('Find all patterns:')
    # Find country pairs with similar transition. Results will be stored in -> `res`
    print('finding country pairs...')
    res=find_similar_energy_transition_pairs(df, model, scen,ref_scen=scen, time_mapping=time_mapping)
    
    # We group more countries together, based on similar transition in ->`res2`
    res2=fun_group_similar_countries(fun_invert_dictionary({k:[v] for k,v in res.items()}))
    # Check that we have not missed any country: (NOTE here we expect countries to be present in multiple groups)
    check_countries_duplicates(res2, list(res.values()), only_one_country_in_each_group=False)
    
    # Consolidate groups and check that we have not missed any country in -> `res3`:
    res3=fun_consolidate_groups(res2)
    res3={f"group {1+x[0]}":res3[x[1]] for x in enumerate(res3)}
    check_countries_duplicates(res3, list(res.values()))
    
    ## Reduces the number of patterns (re-aggregates groups based on similar patterns) in -> `res4`
    print(f'reducing number of patterns to maximum {max_number_of_keys}...')
    res4 = reduce_number_of_patterns(df, res3, model, scen, max_number_of_keys=6)
    
    ## Plotting the results (group of countries) based on `res4`
    if len(time_mapping)==1:
        # This means we do not have historical data. )
        timeperiod=[min(min(list(time_mapping.items()))), max(max(list(time_mapping.items())))]
        default_time_hist=timeperiod
        default_time_future=timeperiod
    else:
        # This means we have a cobination of historical data and future scenario.
        default_time_hist=list(time_mapping.values())
        default_time_future=list(time_mapping.keys())
    for g,countrylist in res4.items():
        fun_show_energy_transition(f"{model} - {g}", fun_xs(df, {"MODEL":model}) , {scen:list(countrylist)}, step2_primary_energy, group_by_time=True, 
            default_time_hist=default_time_hist,
            default_time_future=default_time_future
            )
            
    return res4

def reduce_number_of_patterns(df, res3, model, scen, max_number_of_keys=6):
    count=0
    res5=res3
    while(len(res5))>max_number_of_keys:
        res5=aggregate_pattern_groups(res5, df, model, scen)
        count+=1
        if count >10:
            break
    # Check that we have not missed any country:
    check_countries_duplicates(res5, fun_flatten_list(list(res3.values())))
    res5={f"group {1+x[0]}":res5[x[1]] for x in enumerate(res5)} # h 14.36
    return res5

def check_countries_duplicates(res5, expected_countrylist, only_one_country_in_each_group=True):
    countrylist=fun_flatten_list(list(res5.values()))
    missing=[x for x in expected_countrylist if x not in countrylist]
    if len(missing)>0:
        raise ValueError(f'we miss these countries: {missing}')
    
    # Check that we don't have countries in multiple groups
    if only_one_country_in_each_group:
        mydict=fun_sort_dict({x:countrylist.count(x) for x in countrylist}, by='values', reverse=True).items()
        duplicates={k:v for k,v in mydict if v!=1}
        if len(duplicates)>0:
            txt='To silent the error pass `only_one_country_in_each_group=False`'
            raise ValueError(f'these countries are present in multiple groups: {duplicates}. {txt}')

def aggregate_pattern_groups(res3, df, model, scen):
    # Reduce the number of keys (group of countries), by increasing number of countries in the values (re-allocate countries to bigger groups)
    
    # Step 1: Find groups with minimum lenght in `res3` and make a list `min_list`. The groups in `min_list` (that contain only a few countries) will be re-allocated to other groups, based on similar pattern
    min_len=min([len(x) for x in res3.values()])
    min_list=[k for k,v in res3.items() if len(v)==min_len]
    
    # Step2: To do that, we first calculate aggregated region for each group e.g. create 'group 1' region as the sum of all countries in this group
    agg_df=pd.DataFrame()
    for g,countrylist in res3.items():
        temp=fun_aggregate_countries_general(df,g, list(countrylist)).xs(g, level='REGION', drop_level=False)
        agg_df=pd.concat([agg_df, temp])
    
    # Step3: Find pairs for aggregated groups (based on similar patterns)
    res4 = find_group_pairs(res3, agg_df, model, scen)
  
    # Step4: Avoid intermediate destinations. e.g. res4={'group 16':'group 19','group 19':'group 1' }. This means that 'group 16' will be in the end moved to 'group 1'. 
    # Therefore we re-define our dictionary. In our example as ->{'group 16':'group 1','group 19':'group 1' } (see line below)
    # NOTE: We have intermediate destitations if both k and v in `res4` belong to `min_list`
    res4= {k:res4[v] if v in min_list and k in min_list else v for k,v in res4.items() }

    # Step5: Aggregate country lists in the dictionary based on similar patterns found above and return the updated dictionary
    res5={}
    for k,v in res4.items():
        # NOTE: we want to move countries in `k` group to the bigger `v` group.
        # `k` is the group to be moved
        # `v` is the destination (bigger) group.    
        # if k=='group 19' and set(res3[k])==set(['LVA','PAK','EST']):
        #     iii=1
        #     print('check this one')
        if k in min_list:
            # Move `k` group (with a few countries) to the bigger `v` group
            if v in res5:
                # NOTE: for v we try to use res5 as we might have already updated this group earlier. 
                # for k we use res3[k],  because k is for sure not present in res5 (the purpose is to move k in v group)
                res5[v]=list(set(res5[v]+res3[k]))
            else:
                # If we is not present in res5 then we use res3
                res5[v]=list(set(res3[v]+res3[k]))
            
        else:
            res5[k]=list(set(res3[k]))
    # SliceableDict(res4).slice(*tuple(min_list))
    check_countries_duplicates(res5, fun_flatten_list(list(res3.values())))
    return fun_sort_dict_by_value_length(res5, reverse=True)
    # return {f"group {1+x[0]}":res5[x[1]] for x in enumerate(res5)}


def find_group_pairs(res3, agg_df, model, scen):
    res4={}
    for c in res3:
        if c in agg_df.reset_index().REGION.unique() and 'Primary Energy' in agg_df.xs(c, level='REGION').reset_index().VARIABLE.unique():
            sel=list(find_historic_pattern(
            agg_df, 
            # fun_xs(df_iea, {'REGION':countries_already_selected}, exclude_vars=True), 
            fun_xs(agg_df, {'REGION':c}), 
            c, model, scen, time_mapping={2000:2000, 2015:2015}).keys())
            sel=[x for x in sel if x!=c]
            if len(sel)>0:
                res4[c]=sel[0]
    return res4

def fun_consolidate_groups(d):
    res2=d.copy()
    countries_being_mapped=fun_flatten_list([list(x) for x in res2.values()])
    for c in countries_being_mapped:
        all_groups=fun_invert_dictionary(res2)[c] # # Finds all groups where `c` belongs to
        main_group=all_groups[:1][0] # Find the main (largest) `group`. NOTE: groups are sorted based on len of values. Therefore the `main` group is always the first in this # # list
        groups_from_which_c_should_be_removed=[x for x in all_groups if x!= main_group]
        for g in groups_from_which_c_should_be_removed:
            res2[g]=[x for x in res2[g] if x!=c]
    
    res3={k:v  for k,v in res2.items() if len(v)>0}
    return {f"group {1+x[0]}":res3[x[1]] for x in enumerate(res3)}

def fun_group_similar_countries(d:dict)->dict:
    """Returns dictionary with the best 

    Parameters
    ----------
    d : dict
        _description_

    Returns
    -------
    dict
        _description_
    """    
    count=0
    while len(d)>10:
        d=group_keys_based_on_similar_values(fun_invert_dictionary({k:v if isinstance(v, list) else [v] for k,v in d.items()}))
        count+=1
        if count > 10:
            break

    return fun_sort_dict_by_value_length(d, reverse=True)

def find_similar_energy_transition_pairs(df:pd.DataFrame,model:str, scenario:str, ref_scen:str='Historic data', time_mapping:dict={2000:2000, 2015:2015}, sector_mapping:Optional[dict]=None)->dict:
    """Finds similar energy transition patterens across countries. 
    Returns a dictionary with country pairs based on most similar transition. e.g. {'AGO': 'GHA', 'ALB': 'CRI'...}

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe in IAMC format
    model : str
        Your selected model
    scenario : str
        Selected scenario
    time_mapping : _type_, optional
        Dictionary with time mapping (e.g this allows comparing future scenario with historical data), by default {2000:2000, 2015:2015}
    sector_mapping : Optional[dict], optional
        Dictionary with sector definition e.g. {"Primary Energy": ["Primary Energy|Biomass",...]}.
        This is needed to calculate shares (to find countries with similar energy mix), by default None

    Returns
    -------
    dict
        Dictionary with country pairs based on similar energy transition e.g. {'AGO': 'GHA', 'ALB': 'CRI'...}
    """    
    mydf=df.copy(deep=True)
    if sector_mapping is None:
        sector_mapping={"Primary Energy": [
            "Primary Energy|Biomass",
            "Primary Energy|Coal",
            "Primary Energy|Gas",
            "Primary Energy|Geothermal",
            "Primary Energy|Hydro",
            "Primary Energy|Nuclear",
            "Primary Energy|Oil",
            "Primary Energy|Solar",
            "Primary Energy|Wind",]}
    
    # Slice dataframe for selected variables
    main_sector=list(sector_mapping.keys())[0]
    sel_variables=[main_sector]+fun_flatten_list(list(sector_mapping.values()))
    mydf=fun_xs(mydf, {"VARIABLE":sel_variables})
    
    # Get list of countries
    countrylist= list(mydf.reset_index().REGION.unique())

    # Get country pairs based on most similar transition
    res={}
    for c in countrylist:
        if c in mydf.reset_index().REGION.unique() and main_sector in mydf.xs(c, level='REGION').reset_index().VARIABLE.unique():
            sel=list(find_historic_pattern(
                    # df_iea=fun_xs(mydf, {'REGION':countrylist, "SCENARIO":ref_scen}),
                    df_iea=fun_xs(mydf, { "SCENARIO":ref_scen}),    
                    df=fun_xs(mydf, {'REGION':countrylist,"SCENARIO":scenario}), 
                    c=c, 
                    model=model, 
                    scenario=scenario, 
                    time_mapping=time_mapping,
                    sector_mapping=sector_mapping
                    ).keys(),
                    
                )
            sel=[x for x in sel if x!=c]
            if len(sel)>0:
                res[c]=sel[0]
    return res



# def find_historic_pattern(df_iea:pd.DataFrame, df:pd.DataFrame, c:str, main_sector:str='Primary Energy', time_mapping:dict={2015:2000})->dict:
#     fun_check_iamc_index(df)
#     for x in ['MODEL','SCENARIO']:
#         check=df.reset_index()[x].unique()
#         if len(check)!=1:
#             raise ValueError(f"`df` should contain a single {x}. It contains:{check}")
#     res_df=pd.DataFrame()
#     df2=df.copy(deep=True)
#     if 'FILE' in df2.index:
#         df2=df2.droplevel('FILE')

#     for k,v in time_mapping.items():
#         dfa=(df_iea/df_iea.xs(main_sector, level='VARIABLE'))[v].droplevel(['MODEL','SCENARIO','UNIT'])
#         dfb=(df2/df2.xs(main_sector, level='VARIABLE'))[k].droplevel(['MODEL','SCENARIO','UNIT'])
#         (dfa-dfb).dropna(how='all')
#         temp=pd.DataFrame({v:(dfa-dfb.xs(c, level='REGION'))**2}).dropna(how='all').groupby('REGION').sum()
#         res_df=pd.concat([res_df,temp], axis=1)
#     # return (res_df.sum(axis=1)).sort_values().to_dict()
#     return res_df.dropna().sum(axis=1).sort_values().to_dict()

# import pandas as pd




def fun_box_plot(df):
    size=len(df)
    n=int(np.round(size/2,0))
    # Take 2 randomly generated distributions: red and blue
    ax=df.iloc[1:n].plot.box(color='red')
    df.iloc[n:].plot.box(ax=ax, color='blue')
    # Add default in green
    df.iloc[0:1].plot.box(ax=ax, color='green')
    plt.ylabel('EJ/yr')
    return plt

def fun_prepare_stacked_plot_data(sel_vars, scenarios, df, selcols, sel, df_iam, model, regions_all, r):
    model=fun_fuzzy_match(df.reset_index().MODEL.unique(),model)[0]
    dfs=fun_xs(df.loc[model], {"REGION": regions_all[r], "SCENARIO":list(scenarios), })
                            # Get a unique countrylist for all variables in sel_vars:
    countrylist=list(set.intersection(*(set(dfs.xs(e, level='VARIABLE').reset_index().REGION.unique()) for e in sel_vars)))
    dfs=fun_xs(dfs,{"REGION": countrylist,"VARIABLE":sel})
                            # Calculate statistical difference (sum across countries)
                            # statdiff=fun_xs(df.loc[model], {"REGION": regions_all[r], "SCENARIO":scenarios,  "VARIABLE":f"Statistical Difference|{sel}"})
                            # statdiff=statdiff.groupby('SCENARIO').sum().assign(REGION='Statistical difference').set_index('REGION', append=True)

    unit=dfs.reset_index().UNIT.unique()[0]
                            # Calculate indirect LULUCF emissions (sum across countries)                            
                            # indirect=pd.DataFrame()
                            # if unit == 'Mt CO2-equiv/yr' or sel=='Emissions|CO2':
                            #     indirect=fun_xs(df.loc[model], {"REGION": regions_all[r], "SCENARIO":scenarios,  "VARIABLE":'Emissions|CO2|LULUCF Indirect'})
                            #     indirect=indirect.groupby('SCENARIO').sum().assign(
                            #         # REGION='indirect emissions'
                            #         REGION=F"{model}|{r[:-1]}"
                            #         ).set_index('REGION', append=True)
                            
    for drop in ['FILE', 'UNIT']:
        if drop in dfs.index.names:
            dfs=dfs[selcols].droplevel(drop)
                            # dfs=dfs[range(2010,2105,5)].droplevel(['FILE','UNIT'])
    dfs=dfs.droplevel('VARIABLE') if isinstance(sel, str) else dfs
    single_country=[x for x in countrylist if x in regions_all[r]]
                            # dfs=pd.concat([dfs, statdiff, #indirect
                            #                ]) # Add statistical difference  and indirect emissions (if applicable)
    if not isinstance(sel, str):
        if len(single_country) !=1:
            txt=f"If you want to plot a list of stacked variables ({sel}) for one country, you need to select only one country for the region {r}. You have selected: {single_country}')."
            txt2=f'Alternatively, if you want a stacked plot with all countries in the region {r}, you need to select only one variable. e.g. `Final Energy`. You selected: {sel}'
            raise ValueError(f'{txt} \n {txt2}')
        single_country=single_country[0]
        dfs=dfs.xs(single_country, level='REGION', drop_level=False)
                            # indirect_min=indirect[selcols].min().min() if len(indirect) else 0

    # Calculates Non-IEA countries (and drop them from dfs, just before pd.concat):
    dfs = fun_non_iea_countries(scenarios, dfs, single_country)
                            
    # NOTE: 
    # WEU region contains only IEA countries. But there is is still a stat diff for CO2 emissions (due to indirect emissions)
    # if you want to put everything (non-iea countries, actual statistical difference, indirect emissions) in the stat difference,
    # please remove block above calculating Non-iea countries

    if not isinstance(scenarios, list):
        scenarios=list(scenarios)
    
    # Calculates statistical difference below           
    if isinstance(sel, str):
        # stacked countries, 
        iam_dict={'Emissions|Kyoto Gases (incl. indirect LULUCF)':'Emissions|Kyoto Gases'}
        if model in df_iam.reset_index().MODEL.unique():
            dfiam_sel=fun_xs(df_iam.loc[model].xs(f'{model}|{r[:-1]}', drop_level=False).xs(iam_dict.get(sel,sel), level='VARIABLE'), {'SCENARIO':scenarios})
                                    # indirect=0 if not len(indirect) else indirect
            dfiam_sel=dfiam_sel.droplevel('UNIT').reset_index().set_index(dfs.index.names)
            statdiff=dfiam_sel-dfs.groupby('SCENARIO').sum()
            statdiff=statdiff.rename({statdiff.reset_index().REGION.unique()[0]:'Stat. Diff.'})
            dfs=pd.concat([dfs,dfiam_sel,statdiff])

    y_lim=(1.2*fun_xs(dfs,{"REGION":single_country+["Non-IEA", "Stat. Diff."]}).clip(-np.inf, 0).groupby("SCENARIO").sum().min().min()
                                   ,1.05*
                                   max(#fun_xs(dfs[selcols].clip(0), {'REGION':regions_all[r]}).groupby('SCENARIO').sum().max().max(),
                                       fun_xs(dfs,{"REGION":single_country+["Non-IEA", "Stat. Diff."]}).clip(0).groupby("SCENARIO").sum().max().max(),
                                       # dfs.groupby("SCENARIO").sum().max().max(),
                                       fun_xs(dfs[selcols], {'REGION':f"{model}|{r[:-1]}"}).groupby('SCENARIO').sum().max().max()))
               
    return model, dfs,unit,y_lim


def fun_stacked_plot(df: pd.DataFrame, 
                     line_columns: Union[str, List[str]], 
                     line_color: str = 'black', 
                     line_marker: str = 'o',
                     bar_edge_color: str = 'lightgray', 
                     xlabel: str = 'Time', 
                     ylabel: str = 'Values', 
                     title: str = 'Combined Plot', 
                     background: Optional[str] = None, 
                     ax: Optional[plt.Axes] = None, 
                     _legend: bool = True, 
                     y_lim: Optional[tuple] = None,
                     palette: str = "Spectral",
                     col_fix: Optional[Dict[str, Tuple]] = None, 
                     font_legend: int = 8,
                     font_label: int = 8,
                     font_title: int = 10,
                     my_legend_loc: List[float] = [1.1, 0.005],
                     rotation_xticks: Optional[int] = None
                    ) -> plt:
    """
    Create a combined stacked bar and line plot with specified parameters.

    Parameters:
    - df (pd.DataFrame): Dataframe with downscaled results and regional data (where index=countries/region, columns=Time).
    - line_color (str): Color of the line plot (default: 'black').
    - line_marker (str): Marker style for the line plot (default: 'o').
    - bar_edge_color (str): Color of the border of the entire plot (default: 'lightgray').
    - xlabel (str): Label for the x-axis (default: 'Time').
    - ylabel (str): Label for the y-axis (default: 'Values').
    - title (str): Title of the plot (default: 'Combined Plot').
    - my_legend_loc (list): Location of the legend (default: [1.1, 0.005]).
    
    Returns:
    - plt: The plot object.
    """
    
    line_columns = [line_columns] if isinstance(line_columns, str) else line_columns

    df = df.copy(deep=True)
    countrylist = [x for x in df.index if '|' not in x and x not in line_columns]
    reg_dict = {x: x.split('|')[-1] for x in df.reset_index().REGION.unique() if x in line_columns}
    df = df.rename(reg_dict)
    
    region = None
    if len(list(reg_dict.values())):
        region = list(reg_dict.values())[0]

    df = fun_index_names(df, True, str)
    
    if len(countrylist) == 1:
        df = df.droplevel('REGION')

    df = df.T
    # Plotting
    if not ax:
        fig, ax = plt.subplots()

    if background:
        ax.set_facecolor(background)
        ax.yaxis.grid(color='gray', linestyle='dotted', alpha=0.2)
    if isinstance(palette, matplotlib.colors.ListedColormap):
        # mycolors=[palette(i) for i in np.arange(1,palette._i_under,len(countrylist))] # palette._i_under-1 is the number of colors in the palette (usually 256)
        mycolors=list(range(1,palette._i_under,int(np.round(palette._i_under/len(countrylist),0))))
        mycolors=[palette(i) for i in mycolors]
        col_dict=dict(zip(countrylist, mycolors))
    else:     
        cmap = plt.get_cmap(palette, len(countrylist))
        col_dict = {c: cmap(n) for n, c in enumerate(countrylist)}

    if col_fix:
        for k, v in col_fix.items():
            if k in countrylist:
                col_dict[k] = v

    df[list(col_dict.keys())].plot(kind='bar', stacked=True, ax=ax, edgecolor=bar_edge_color, figsize=(15, 8), 
                                   color=list(col_dict.values()))

    line_columns = list(reg_dict.values())
    lines = ["--", ":", "-."]
    linecycler = cycle(lines)
    lines_dict = {x: next(linecycler) for x in line_columns}

    for x in lines_dict:
        if x in df.columns and x in line_columns:
            df[x].plot(ax=ax, color=line_color, marker=line_marker, label=x, ls=lines_dict[x])

    ax.set_xlabel(xlabel, fontsize=font_label)
    ax.set_ylabel(ylabel, fontsize=font_label)
    ax.set_title(title, fontsize=font_title)
    
    # Rotate x-ticks labels vertically
    if rotation_xticks is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Adding legend
    if isinstance(countrylist[-1], tuple):
        ax.legend([x[-1] if isinstance(x, tuple) else x for x in countrylist], prop={'size': font_legend})
    else:
        ax.legend(ncol=10, loc=my_legend_loc, prop={'size': font_legend})

    # Change the border color of the entire plot
    ax.spines['top'].set_color(bar_edge_color)
    ax.spines['bottom'].set_color(bar_edge_color)
    ax.spines['left'].set_color(bar_edge_color)
    ax.spines['right'].set_color(bar_edge_color)

    plt.tight_layout()
    ax.set_xlim(-1, len(df.index.unique()))

    if y_lim:
        ax.set_ylim(y_lim)
        
    if not _legend:
        ax.get_legend().remove()

    return plt

# Example Usage:
# plot_combined_graph(df, line_column='Value1', line_color='blue', title='Custom Title')


def fun_florian_plot(
    sel_scen,
    project,
    df_graph,
    var,
    c,
    all_in_one_legend=False,
    legend_color_box="upper right",
    legend_ls_box="lower left",
    lw: float = 1,
    alpha: float = 0.5,
    palette="husl",
    hue_list=["MODEL", "SCENARIO"],
    marker=None,
):
    sns.set(
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#EDEDED",
            "axes.edgecolor": "grey",
            # "figure.figsize": (5,10),
        }
    )

    unit = df_graph.xs(var, level="VARIABLE").reset_index().UNIT.unique()[0]
    if not isinstance(sel_scen, list):
        sel_scen = list(sel_scen)
    # Florian's graph
    for hue in hue_list:
        fig, ax = plt.subplots()
        if isinstance(sel_scen, dict):
            sel_scen_florian = list(set(fun_flatten_list(list(sel_scen.values()))))
        else:
            sel_scen_florian = sel_scen
        df_criteria = fun_add_criteria(
            "standard",
            fun_xs(df_graph, {"SCENARIO": sel_scen_florian}),
        )

        # Creating a dictionary for colors

        col_dict = {hue: palette}
        # if values are strings we assume it's a palette, and create dict for each key accordingly
        if len(col_dict.values()) and isinstance(list(col_dict.values())[0], str):
            for k, v in col_dict.items():
                col_dict[k] = {
                    c: sns.color_palette(v, len(set(df_graph.reset_index()[k])))[n]
                    for n, c in enumerate(set(df_graph.reset_index()[k]))
                }
            col_dict = list(col_dict.values())[0]

        # Different data preparation if there is CONVERGENCE and METHOD in the dataframe
        idxn = df_criteria.index.names
        on = ["CONVERGENCE", "METHOD"]
        on = [o for o in on if o in df_criteria.index.names]
        if len(on):
            df_criteria = df_criteria.reset_index()

            df_criteria = df_criteria.assign(
                ALLPATH=df_criteria[["CONVERGENCE", "METHOD"]]
                .astype(str)
                .apply("-".join, axis=1)
            )  #
            df_criteria = df_criteria.set_index(idxn + ["ALLPATH"]).droplevel(on)
            df_criteria = pd.concat([pd.melt(
                pd.concat(
                    [
                        prepare_data_for_sns_lineplot(
                            df_criteria.xs((c, a, func), level=("REGION", "ALLPATH", 'FUNC')).xs(
                                var, level="VARIABLE"
                            ),
                            "standard",
                        )
                        .set_index(["MODEL", "SCENARIO", "YEAR"])
                        .rename({"value": f"value_{a}"}, axis=1)
                        for a in df_criteria.xs(var, level="VARIABLE")
                        .reset_index()
                        .ALLPATH.unique()
                    ],
                    axis=1,
                ).reset_index(),
                ["MODEL", "SCENARIO", "YEAR"],
            ).assign(FUNC=func) for func in df_criteria.reset_index().FUNC.unique()])
        else:
            df_criteria = prepare_data_for_sns_lineplot(
                fun_add_criteria(
                    "standard",
                    fun_xs(df_graph, {"SCENARIO": sel_scen_florian}),
                )
                .xs(c, level="REGION")
                .xs(var, level="VARIABLE"),
                "standard",
            )

        # Plot graph
        [fun_sns_lineplot_new(
            # prepare_data_for_sns_lineplot(
            #     df_criteria.xs(c, level="REGION").xs(var, level="VARIABLE"), "standard" ),
            df_criteria[df_criteria.FUNC==func],
            title=f"{var} - {c}",
            y_axis_label=unit,
            ax=ax,
            hue=hue,
            all_in_one_legend=all_in_one_legend,
            legend_color_box=legend_color_box,
            legend_ls_box=legend_ls_box,
            col_dict=col_dict,
            lw=lw,
            alpha=alpha,
            marker=marker,
        ) for func in df_criteria.FUNC.unique()]
        # fun_sns_lineplot_new(
        #     # prepare_data_for_sns_lineplot(
        #     #     df_criteria.xs(c, level="REGION").xs(var, level="VARIABLE"), "standard" ),
        #     df_criteria,
        #     title=f"{var} - {c}",
        #     y_axis_label=unit,
        #     ax=ax,
        #     hue=hue,
        #     all_in_one_legend=all_in_one_legend,
        #     legend_color_box=legend_color_box,
        #     legend_ls_box=legend_ls_box,
        #     col_dict=col_dict,
        #     lw=lw,
        #     alpha=alpha,
        # ) 
        fun_save_figure(
            fig,
            CONSTANTS.CURR_RES_DIR("step5") / "Step5e_visuals" / project,
            f"Florian_graph_{var.replace('|', '_')}_{c}_by_{hue}",
        )


def stacked_data_step5(
    project: str, 
    files_dict: Dict[str, Union[str, List[str]]], 
    countrylist: Optional[List[str]], 
    iea_countries_only: bool, 
    sel_vars: List[str], 
    scenarios: List[str], 
    df: pd.DataFrame,
    selcols: Optional[List[int]] = None,  # Optional selcols with default None
    font_tick: int = 13,  # Optional font_tick with default value of 13
    myfont: int = 18,  # Optional myfont with default value of 18
    suptitle_font: int = 22,  # Optional suptitle font size for plt.suptitle
    font_title: int = 20,  # Optional font size for individual plot titles
    font_legend: int = 12,  # Optional font size for legend
    font_label: int = 15,  # Optional font size for labels (y-axis)
    my_legend_loc=[0.5,-0.05],  # Optional legend location with default value of [0.5,-0.05]
    figsize=(10, 20),  # Optional figsize with default value of (10, 20)
    wspace=0.6,  # Optional wspace with default value of 0.5
    save_figure: bool = True,  # Optional save_figure with default value of True
    rotation_xticks: Optional[int] = None,  # Optional rotation_xticks with default value of None
    palette: str = "Spectral",  # Optional palette with default value of "Spectral"
    folder:Optional[str]=None,  # Optional folder with default value of None
    dpi:Optional[int]=None,  # Optional dpi with default value of None
    size_inches:Optional[tuple]=None # Optional size_inches when saving figure, with default value of None (e.g. (10, 20))
) -> None:
    """
    Process IAM data for stacked plot generation based on regions and variables.

    Args:
        project (str): The project name to read IAM data.
        files_dict (Dict[str, Union[str, List[str]]]): Dictionary containing model names as keys and file paths as values.
        countrylist (Optional[List[str]]): List of countries for which data is processed.
        iea_countries_only (bool): Flag indicating if only IEA countries should be used.
        stacked_countries_plot (bool): Whether to generate a stacked plot by countries or variables.
        sel_vars (List[str]): List of selected variables for plotting.
        scenarios (List[str]): Scenarios for data analysis.
        df (pd.DataFrame): Main dataframe containing the input data for analysis.
        selcols (Optional[List[int]]): List of selected columns for the years. Defaults to [2020, 2030, 2050].
        font_tick (int): Font size for tick labels. Defaults to 13.
        myfont (int): Font size for variable titles in the plot. Defaults to 18.
        suptitle_font (int): Font size for the main plot title (plt.suptitle). Defaults to 22.
        font_title (int): Font size for individual subplot titles. Defaults to 20.
        font_legend (int): Font size for the legend in the plots. Defaults to 12.
        font_label (int): Font size for labels (like y-axis label). Defaults to 15.
        save_figure (bool): Save figure as png, defaults to True
        size_inches (Optional[tuple]): Size of the figure in inches when saving figure, defaults to None.

    Returns:
        None
    """
    # Default value for selcols if not provided
    if selcols is None:
        selcols = [2020, 2030, 2050]  # Default years for selection

    # Read IAM data and index it properly
    df_iam = fun_index_names(fun_read_df_iams(project), True, int)

    for model, file in files_dict.items():
        regions_all = fun_regional_country_mapping_as_dict(model, project, iea_countries_only)
        regions_to_be_downs = {k: v for k, v in regions_all.items() if len(v) > 1}
        regions = regions_to_be_downs

        if countrylist is not None:
            regions = fun_get_iam_regions_associated_with_countrylist(project, countrylist, model)
            regions = fun_invert_dictionary({k: [v] for k, v in regions.items()})
            # Exclude Native countries (no csv file for native countries)
            regions = {k: v for k, v in regions.items() if k in regions_to_be_downs}

        for r in regions:
            if not isinstance(file, list):
                file = [file]
            selfile = [x for x in file if r in str(x)]
            res = {}

            for sel in sel_vars:
                # Prepare data for plotting
                model, dfs, unit, y_lim = fun_prepare_stacked_plot_data(
                    sel_vars, scenarios, df, selcols, sel, df_iam, model, regions_all, r
                )
                res[sel] = {'dfs': dfs[selcols], 'unit': unit, 'y_lim': y_lim}

            # Create scenario data dictionary for each variable and scenario
            scen_data = {f"{sel}||{s}": res[sel]['dfs'].xs(s, drop_level=False) for sel in sel_vars for s in scenarios}
            
            # Set font sizes for ticks
            plt.rc('xtick', labelsize=font_tick)
            plt.rc('ytick', labelsize=font_tick)

            # Create subplots
            fig, axes = plt.subplots(1, len(scen_data.items()), figsize=figsize)
            plt.suptitle(f"{model}|{r[:-1]}", fontsize=suptitle_font)

            if len(scenarios) < 2:
                raise ValueError(f'Please specify at least two scenarios to create a subplot. You selected: {scenarios}')

            # Plot each scenario and variable
            for (k, d), ax in zip(scen_data.items(), axes.flat):
                _legend = True if k == list(scen_data.keys())[-1] else False
                var = k.rsplit("||")[0]
                scen = k.rsplit("||")[-1]
                unit = res[var]['unit'] if scen == scenarios[0] else None
                title = scen if len(sel_vars) == 3 else f"{var[:15]}\n{scen}"
                fig = fun_stacked_plot(d.loc[scen], [f"{model}|{r[:-1]}"], title=title,
                                        ylabel=unit, ax=ax, _legend=_legend, y_lim=res[var]['y_lim'],
                                        background='white', line_marker=None, font_legend=font_legend,  # Using font_legend
                                        font_label=font_label,  # Using font_label
                                        font_title=font_title,  # Using the optional font_title argument
                                        rotation_xticks=rotation_xticks,  # Using the optional rotation_xticks argument
                                        palette=palette,  # Using the optional palette argument
                                        col_fix={"Non-IEA": (0, 0, 0, 0.5),
                                                "Stat. Diff.": (0, 0, 0, 0.8)}, my_legend_loc=my_legend_loc)  # Using the optional my_legend_loc argument

            # Adjust the layout of the subplots
            fig.subplots_adjust(top=0.85, wspace=wspace, bottom=0.4)

            # Handle plot title for multiple variables
            if len(sel_vars) == 3:
                fig.text(-6.67, 1.12, sel_vars[0], horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=myfont)
                fig.text(-3.4, 1.12, sel_vars[1], horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=myfont)
                fig.text(-0.35, 1.12, sel_vars[2], horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=myfont)

            # Adjust subplot spacing
            plt.subplots_adjust()

            # Save the figure to the appropriate folder
            if folder is None:
                folder = CONSTANTS.CURR_RES_DIR("step5") / "Step5e_visuals" / project / model
                folder = folder / 'stacked_countries_plot' if isinstance(sel, str) else folder / 'stacked_variables_plot'
                folder = folder / fun_shorten_region_name(r)
            if save_figure:
                fun_save_figure(fig, folder, r[:-1], dpi=dpi, size_inches=size_inches)
            return fig

if __name__ == "__main__":
    main(
    files=None,
    models=["*MESSAGE*"],
    project='NGFS_2023',
    step='step1',
    sel_vars=[
        "Final Energy", # stacked countries plot
        ],
    markers_dict=markers_dict,
    iea_countries_only=True, 
    countrylist=["AUS" ],
    sel_scen=['h_cpol', "*1p5C*"],
    palette=None,#'flare',
    create_dashbord=None,  # "NGFS_2022_dashboard.pdf"
    eu_ab_plot=False,
    ngfs_plot=False, # Simple Line plots across model/scenario
    florian_plot=False,
    top_20_macc_plot= None,# 'test_macc.csv', #'Main_table.csv',
    log_log_graphs=True,
    stacked_countries_plot=False,
    phase_out_plot=False,
    carbon_budget_plot=False,
    step1_sensitivity=True, 
    step1_extended_sensitivity = False,
    analyse_drivers=False,
    split_df_by_model_and_save_to_csv=False,
    # If you want to plot individual method in each graph, please provide a list of `sel_step1_methods`
    # sel_step1_methods=None,  # If None will plot all methods in a single graph. Otherwise provide a list of methods
    sel_step1_methods=[
        "wo_smooth_enlong_ENLONG_RATIO",
        # "GDPCAP_False_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO",
        # "GDPCAP_False_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO",
        # "GDPCAP_True_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO",
        # "GDPCAP_True_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO",
        # "TIME_False_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO",
        # "TIME_False_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO",
        # "TIME_True_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO",
        # "TIME_True_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO",
    ],
)

