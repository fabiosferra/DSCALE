import argparse
from typing import Union, Dict, List
from pprint import pprint 
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go 
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
from downscaler.utils_string import fun_check_if_all_characters_are_numbers, extract_text_in_parentheses
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
import Step_5e_visuals
from Step_5e_visuals import stacked_data_step5
from downscaler.fixtures import all_countries
from Step_5e_historical_harmo import fun_create_dashboard_pdf, fun_create_dashboard_pdf_by_country
from matplotlib.backends.backend_pdf import PdfPages

from utils_dictionary import fun_sort_dict_by_value_length
from downscaler.utils_visual import fun_phase_out_date_colormap, rgba_to_rgb, colors_to_rgba, create_parallel_coordinates, convert_colormap_to_rgb, rgb_to_hex, create_color_palette

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




from downscaler.utils import *
from downscaler.fixtures import *
from hindcasting_results_comparison import main as run_hindcasting_plots

def main(
    files: Optional[Union[list, str]] = "iconics_NZ_data_and_table_20230512_v17.xlsx",
    step: str = "step5",
    sel_vars: Union[list, str] = [ "Final Energy",
            "Emissions|CO2|Energy",
            "Emissions|Kyoto Gases (incl. indirect LULUCF)"],
    sel_scen: Optional[Union[str, list]] = ['h_cpol', 'o_1p5c'],#"model_scenarios_combinations_63.csv",
    countrylist: Optional[list] = ["EU27"],
    iea_countries_only:bool = False,
    markers_dict: dict = {},
    models: Optional[list] = None,
    project=None,
    ngfs_plot: bool = False,
    florian_plot: bool = True,
    stacked_countries_plot:bool = False,
    legend_color_box: Optional[str] = None,  # "lower left",  # for florians plot
    legend_ls_box: Optional[str] = None,  # "upper right",  # for florians plot
    log_log_graphs: bool = True,
    step1_sensitivity: bool = False,
    step1_extended_sensitivity:bool = False,
    sel_step1_methods: Optional[list] = None,
    palette:Optional[str]=None,    
    combine_df_with_most_recent_scenarios=False,
    figure_2=False,
    figure_4=False,
    figure_5=False,
    figure_5_parallel=False, # parallel coordinates plot
    # figure_step2b_boxplot=False,
    figure_step2b_violin_plot=False,
    figure_72_boxplot=False,
    figure_72_violin=False,
    figure_73_weights_scatter=False, # Scatter plot: shows which criteria (used in the random weights) influence the results (eg. SOL in AUS is mainly driven by cost curves criteria)
    figure_8=False,
):
    if figure_2:
        import plotly.express as px

        gdp = pd.read_csv(CONSTANTS.INPUT_DATA_DIR / "Historical_data.csv")
        ktoe_to_ej = 0.041868 * 1e-3
        gdp["EI"] = gdp["TFC"] / gdp["GDP|PPP"] * ktoe_to_ej

        # Map iso to continent
        grassi_reg_mapping = (
                pd.read_csv(CONSTANTS.INPUT_DATA_DIR / "Grassi_regional_mapping.csv")
                .dropna()
                .set_index(["R5_region", "ISO"])
            )
        grassi_dict=grassi_reg_mapping.reset_index().set_index('ISO')['R5_region'].to_dict()

        # Create dataframe to plot
        dfplot=gdp[['ISO','REGION','EI','GDPCAP','POPULATION',"GDP|PPP"]].replace('missing',np.nan)
        dfplot['continent']=[grassi_dict.get(x,np.nan) for x in dfplot.ISO]

        dfplot['LNPOP']=[max(1e-10,np.log(x)) for x in dfplot['POPULATION']]
        dfplot=dfplot.dropna()

        # Actual plot
        fig = px.scatter(dfplot, 
                x="GDPCAP", y="EI", 
                # animation_frame="year", animation_group="country",
                size="POPULATION", color="ISO", hover_name="ISO", facet_col="continent",
                template='plotly_white',
                log_x=True, log_y=True, size_max=35 ,#range_x=[0.0001,1e6]#, range_y=[25,90]
                )

        # Customize facet column titles
        fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
        fig.show()

    
    
    myfolder=CONSTANTS.CURR_RES_DIR("step5") / "Step5e_visuals" / 'Paper1_technical_paper'/project
    mypalette=cm.managua # cm.batlow #cm.lapaz
    
    # Get models
    f = fun_wildcard
    models_all = fun_get_models(project) if project else None
    models = models_all if models is None or project is None else f(models, models_all)

    t=2030

    # if figure_step2b_boxplot:
    #     step='step2b'
    #     n=500
    #     files={'MESSAGEix-GLOBIOM 1.1-M-R12':'MESSAGEix-GLOBIOM 1.1-M-R12_Pacific OECDr_h_cpol_2023_12_21.csv'}
    #     fuels=['OIL', 'COAL', 'GAS', 'NUC', 'HYDRO', 'BIO', 'GEO', 'SOL',  'WIND']
    #     # Box plot for step2b
    #     Step_5e_visuals.fun_box_plot_step2(files, models, project)
        
    if figure_step2b_violin_plot:
        mypalette=cm.bamako # cm.batlow #cm.lapaz
        mypal=convert_colormap_to_rgb(mypalette)
        # mypal=colors_to_rgba(mypal) # convert to rgba
        # mypal=rgba_to_rgb(mypal) #  convert to RGB colors
        mypal={'A': rgb_to_hex(mypal)[30], 'B': rgb_to_hex(mypal)[-30], 'standard':'black'}
        step='step2b'
        n=500
        files={'MESSAGEix-GLOBIOM 1.1-M-R12':'MESSAGEix-GLOBIOM 1.1-M-R12_Pacific OECDr_h_cpol_2023_12_21.csv'}
        fuels=['OIL', 'COAL', 'GAS', 'NUC', 'HYDRO', 'BIO', 'GEO', 'SOL',  'WIND']
        for m in models:
            dfin= pd.read_csv(CONSTANTS.CURR_RES_DIR('step2')/'step2b_sensitivity'/f"{files[m]}")
            countrylist=list(dfin.ISO.unique())
            dfin=dfin.set_index(['TIME','ISO',  'CRITERIA', 'METHOD'])
            for c in ['AUS','JPN']:
                df=dfin.xs(c, level='ISO').xs('dynamic_recursive', level='METHOD').loc[t,fuels]

                # Creat two sets of distributions from criteria (n=500 for each distribution)
                df.index=[str(x) for x in df.index] 
                df['SET']=['1' if x!='standard' and int(x)<n else '2' for x in df.index]

                # Create res
                res=df.set_index(['SET'], append=True).stack().reset_index()
                res=res.set_index(['level_0', 'SET','level_2'])
                res.index.names=['CRITERIA','SET','FUEL'] 
                res=res.reset_index()

                # Finalize RES
                res=pd.concat([res[res.CRITERIA=='standard'].replace('2','standard'), 
                        res[res.CRITERIA!='standard']])

                # Make violin plot
                fig=create_fuel_violin_box_plot(res, fuels=['COAL', 'GAS', 'WIND', 'SOL', 'HYDRO'], c=c, t=t,
                                                labels_dict={'1':'A','2':'B'}, 
                                                labels_colors={'A':'#0f443e', 'B':'#e0c76c', 'standard':'black'}
                                                )
                # white backrgound
                white_background=True
                if white_background:
                    fig=add_white_background_to_plotly(fig)
                
                fig.show()
                fun_save_figure(fig, myfolder/'Figure_step2b', f'Violin_plot_{c}_{t}')                   
                
    if figure_4:
        step='step5'
        countrylist, models, scenarios, df, files_dict = fun_read_data_paper1(files, step, sel_scen, countrylist, models, project, check_variables=False)
                
        # dict with country and legend location {Country:legend location}
        mycountrydict={
                        'ETH':[-8, -0.4], # legend location a bit lower
                        'AUS':[-8, -0.3], 
                       'CHN':[-8, -0.3],
                        'AUT':[-8, -0.4], # legend location a bit lower
                         }
        for m in models:
            for c,v in mycountrydict.items():
                myreg=fun_regional_country_mapping_as_dict(m,project)[fun_country_to_region(project, c)[m]]
                stacked_data_step5(project, files_dict, myreg, iea_countries_only, sel_vars, scenarios,df, 
                        suptitle_font=20, font_title=16, myfont=16, font_legend=14,
                        font_label=16, font_tick=13, save_figure=True, 
                        my_legend_loc=v, figsize=(20,10), wspace=0.6, rotation_xticks=90, palette=mypalette,
                        folder=myfolder/'Figure_4', 
                        size_inches=(20,10), dpi=100 # save figure features to mimic full screen
                        # dpi=300
                        )
                plt.close()
        
        # stacked_data_step5(project, files_dict, ['DEU'], iea_countries_only, sel_vars, scenarios,df, 
        # suptitle_font=20, font_title=15, myfont=15, font_legend=11,
        # font_label=15, font_tick=11, save_figure=True, my_legend_loc=[-8, -0.4])
    if figure_5 or figure_5_parallel:
        # mypalette=create_color_palette(["#595A86", "#8C4C4B"], steps=256, show_palette=False)
        mypalette=cm.managua  #cm.vik
        files=None
        models=["*MESSAGE*"]
        project='NGFS_2023'
        step='step1' 
        sel_vars=["Final Energy"]
        markers_dict=markers_dict
        countrylist=['JPN','AUS','NZL'] # EU
        sel_scen=['h_cpol']  # ["SSP2-NPi"],  #
        palette=None#'flare',
        # create_dashbord= "NGFS_2024_dashboard.pdf",
        ngfs_plot=False # Simple Line plots across model/scenario (one plot for each country)
        florian_plot=True
        combine_df_with_most_recent_scenarios=False
        log_log_graphs=False
        stacked_countries_plot=False
        step1_sensitivity=True
        step1_extended_sensitivity = True
        # If you want to plot individual method in each graph, please provide a list of `sel_step1_methods`
        sel_step1_methods=None

        countrylist, models, scenarios, df, files_dict = fun_read_data_paper1(files, step, sel_scen, countrylist, models, project)


        if iea_countries_only and stacked_countries_plot:
            raise ValueError('When plotting `stacked_countries_plot` we aggregate non-iea countries a single country block.' 
                            'Therefore please pass `iea_countries_only=False` ')

        selcols = list(range(2010, 2055, 5))
        
        sns.set(
            rc={
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "grid.color": "#EDEDED",
                "axes.edgecolor": "grey",
                # "figure.figsize": (5,10),
            }
        )
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
                    # mydict={'wo_smooth_enlong_ENLONG_RATIO':' `IAMatt` disconnected to 2010 (Default method)',
                    #         'TIME_True_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO': '`IAMatt` connected from 2010-2100, over time (linear)',
                    #         'TIME_True_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO': '`IAMatt` connected from 2010-2050, over time (linear)', #'Linear convergence over time in 2050',
                    #         'TIME_False_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO': '`IAMatt` connected from 2010-2100, over time (log-scale)',
                    #         'TIME_False_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO': '`IAMatt` connected from 2010-2050, over time (log-scale)',
                    #         'GDPCAP_True_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO': '`IAMatt` connected from 2010-2100, over GDP per capita (log-scale)',
                    #         'GDPCAP_True_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO': '`IAMatt`  connected from 2010-2050, over GDP per capita (log-scale)',
                    #         'GDPCAP_False_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO': '`IAMatt` connected from 2010-2100, over GDP per capita (linear)',
                    #         'GDPCAP_False_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO': '`IAMatt` connected from 2010-2050, over GDP per capita (linear)',
                    #         }
                    mydict={'wo_smooth_enlong_ENLONG_RATIO':'Default method',
                            'TIME_True_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO': '2100, over time (linear)',
                            'TIME_True_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO': '2050, over time (linear)', #'Linear convergence over time in 2050',
                            'TIME_False_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO': '2100, over time (log-scale)',
                            'TIME_False_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO': '2050, over time (log-scale)',
                            'GDPCAP_True_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO': '2100, over GDP per capita (log-scale)',
                            'GDPCAP_True_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO': '2050, over GDP per capita (log-scale)',
                            'GDPCAP_False_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO': '2100, over GDP per capita (linear)',
                            'GDPCAP_False_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO': '2050, over GDP per capita (linear)',
                            }
                    
                    # Indexed calculations (2010=1)
                    df_graph2=pd.concat([df_graph[[t]]/df_graph[[2010]].rename({2010:t}, axis=1) for t in df_graph.columns], axis=1)
                    
                    # Parallel coordinates plot in time t
                    if figure_5_parallel==True:
                        for x in ['FUNC','CONVERGENCE','METHOD']:
                            myfig=create_parallel_coordinates(df_graph2, x, 2050, palette=mypalette, countrylist=countrylist)
                            fun_save_figure(
                                    myfig,
                                    myfolder/'Figure_5',
                                    f'Parallel coordinates_over{x}'
                                )
                    
                    # Convert the palette to a list of RGB colors in the format 'rgb(r, g, b)' if the palette is a ListedColormap object
                    mypal=convert_colormap_to_rgb(mypalette)
                    # mypal=colors_to_rgba(mypal) # convert to rgba
                    # mypal=rgba_to_rgb(mypal) #  convert to RGB colors
                    mypal={'s-curve': rgb_to_hex(mypal)[-50], 
                           'log-log': rgb_to_hex(mypal)[50]}

                    if figure_5==True:
                        # Note: the order matters! Start with `filter_condition_5p3`
                        for fun in [ filter_condition_5p3, filter_condition_5p1, filter_condition_5p2]:
                            add_line = True if fun == filter_condition_5p3 else False
                            
                            fig =  plot_5p_generalv3(
                                countrylist=countrylist, 
                                project=project, 
                                legend_color_box=legend_color_box, 
                                legend_ls_box=legend_ls_box, 
                                scenarios=scenarios, 
                                # df_graph=df_graph, 
                                var=var, 
                                mydict=mydict, 
                                df_graph2=df_graph2, # Indexed dataframe
                                mypalette=mypal,
                                filter_condition=fun,
                                add_default_line=add_line, 
                                font_scale=2,
                                # x_label='TIME',
                                legend_fontsize=18, 
                                xtick_fontsize=20, 
                                ytick_fontsize=20,
                                # ylabel_fontsize=20,
                                # title_fontsize=22,
                                suptitle_fontsize=24,
                                figsize=(18,10),
                            )
                            
                            fun_save_figure(
                                    fig,
                                    myfolder/'Figure_5',
                                    f'Figure_{(fun.__name__)[-3:]}'
                                )
                            
                    
    if figure_72_boxplot:
        t=2050
        # Read dataframe from step2
        df=fun_read_csv(
        {'aa':CONSTANTS.CURR_RES_DIR('step2')/'MESSAGEix-GLOBIOM 1.1-M-R12_2024_09_09_TEST.csv'}, 
        True, int
        )['aa']
        
        # Write criteria as string
        df=df.reset_index()
        df['CRITERIA']=[str(x) for x in df.CRITERIA]
        df=fun_index_names(df)
        
        # Create a graph for each country 
        for c in ['AUS','JPN']:

            temp={}
            res={}
            d={
                'standard':                       {'conv':'2250', 'criteria': {'ISO': c,'FUNC':'log-log', 'LONG_TERM':'ENLONG_RATIO', 'CRITERIA':'standard',}},
                'weights':                        {'conv':'2250', 'criteria': {'ISO':c, 'FUNC':'log-log', 'LONG_TERM':'ENLONG_RATIO'}}, # we do not filter for criteria
                'weights + demand':               {'conv':'2250', 'criteria': {'ISO':c}}, # we do not filter for functional form and IAMatt
                'weights + demand + convergence': {'conv':'', 'criteria': {'ISO':c}}, # we do not filter for convergence (NOTE: will exclude variables with asteriks - coming from extended  sensitivity analysis)
                'weights + demand + convergence*':{'conv':'', 'criteria': {'ISO':c}}, # we do not filter for convergence (and will take all results, including variables with asteriks)
            }
            df_secondary = fun_xs_fuzzy(df[[t]], ['Secondary'])
            df_electricity = fun_xs_fuzzy(df_secondary, 'Electricity')
            for k,v in d.items():
                for x in ['oil','coal','gas','nuclear','hydro','biomass','geothermal', 'solar','wind']:
                    # Breaking down the chained calls step by step
                    df_energy_conv = fun_xs_fuzzy(df_electricity, f"{x}{v['conv']}")

                    # Final result with fun_xs applied
                    df_sel = fun_xs(df_energy_conv,v['criteria'])

                    if '*' not in k:
                        asterics_index=fun_xs_fuzzy(df_sel, ['*']).index
                        df_sel=df_sel[~df_sel.index.isin(asterics_index)]
                    df_sel['fuel']=x
                    # temp[x]=df_sel.reset_index()[['fuel', 2050]].set_index('fuel').T.rename({2050:k}).T
                    temp[x]= df_sel.reset_index()[['fuel', t]].set_index('fuel').T.rename({t:k}).stack().reset_index()
                res[k]=pd.concat(list(temp.values()))
            res=pd.concat(list(res.values()))

            res=res.replace({'oil':'OIL', 'coal':'COAL', 'gas':'GAS', 'nuclear':'NUC', 'hydro':'HYDRO','geothermal':'GEO', 'biomass':'BIO', 'solar':'SOL', 'wind':'WIND'})

            # Create the boxplot with a custom palette
            sns.boxplot(x=res['fuel'], 
                        y=res[0], 
                        hue=res['level_0'], 
                        # palette=custom_palette,
                        palette='Set2',
                        # linewidth=1
                    )

            # Remove 'standard' from the legend
            handles, labels = plt.gca().get_legend_handles_labels()
            new_handles = [h for h, l in zip(handles, labels) if l != 'standard']
            new_labels = [l for l in labels if l != 'standard']

            # Update the legend
            plt.legend(new_handles, new_labels)

            plt.title(f'{c} in {t}')
            plt.ylabel('EJ/yr')
            # Show the plot
            plt.show()
            plt.clf()

    if figure_72_violin:
        # Older plot can be found here: https://github.com/iiasa/downscaler_repo/issues/181
        fueldict={'oil':'OIL', 'coal':'COAL', 'gas':'GAS', 'nuclear':'NUC', 'hydro':'HYDRO','geothermal':'GEO', 'biomass':'BIO', 'solar':'SOL', 'wind':'WIND'}
        # Read dataframe from step2
        df=fun_read_csv(
            {'aa':CONSTANTS.CURR_RES_DIR('step2')/'MESSAGEix-GLOBIOM 1.1-M-R12_2024_09_19.csv'}, 
            True, int
            )['aa']
        t=2030
        # Write criteria as string
        df=df.reset_index()
        df['CRITERIA']=[str(x) for x in df.CRITERIA]
        av_criteria=df.CRITERIA.unique()
        
        
        df['IAM_FUEL']=[(x[len('Secondary Energy|Electricity|'):-len('XXxxxBLEND')]).lower() for x in df.reset_index().VARIABLE]
        df=df.replace(fueldict)
        df=fun_index_names(df)

        # Read criteria weights
        fname=CONSTANTS.CURR_RES_DIR('step2')/'step2b_sensitivity'/'criteria_weights.csv'
        weights=fun_read_csv({'aa':fname},True, int)['aa'].reset_index()
        weights['CRITERIA']=[str(x) for x in weights.CRITERIA]

        # Create a graph for each country 
        for c in ['JPN', 'AUS']:

            temp={}
            res={}
            d={
                #'standard':                       {'conv':'2250', 'criteria': {'ISO': c,'FUNC':'log-log', 'LONG_TERM':'ENLONG_RATIO', 'CRITERIA':'standard',}},
                'Criteria weights':                {'conv':'2250', 'criteria': {'ISO':c, 'FUNC':'log-log', 'LONG_TERM':'ENLONG_RATIO'}}, # we do not filter for criteria
                'Demand projections':              {'conv':'2250', 'criteria': {'ISO':c, 'CRITERIA':'standard'}}, # we do not filter for demand
                'Convergence years (2050-2300)':   {'conv':'', 'criteria':     {'ISO':c, 'FUNC':'log-log', 'LONG_TERM':'ENLONG_RATIO', 'CRITERIA':'standard'}}, # we do not filter for convergence 
                #'weights + demand + convergence*':{'conv':'', 'criteria': {'ISO':c}}, # we do not filter for convergence (and will take all results, including variables with asteriks)
            }
            df_secondary = fun_xs_fuzzy(df[[t]], ['Secondary'])
            df_electricity = fun_xs_fuzzy(df_secondary, 'Electricity')
            for k,v in d.items():
                for x in ['oil',
                           'coal','gas','nuclear','hydro','biomass','geothermal', 'solar','wind'
                          ]:
                    # Breaking down the chained calls step by step
                    df_energy_conv = fun_xs_fuzzy(df_electricity, f"{x}{v['conv']}")

                    # Final result with fun_xs applied
                    df_sel = fun_xs(df_energy_conv,v['criteria'])

                    # if '*' not in k:
                    #     asterics_index=fun_xs_fuzzy(df_sel, ['*']).index
                    #     df_sel=df_sel[~df_sel.index.isin(asterics_index)]
                    df_sel['fuel']=x
                    # temp[x]=df_sel.reset_index()[['fuel', 2050]].set_index('fuel').T.rename({2050:k}).T
                    
                    
                    add_n=False # Add number of observations in the legend
                    if add_n:
                        temp[x]= df_sel.reset_index()[['fuel', t]].set_index('fuel').T.rename({t:f"{k} (n={len(df_sel)})"}).stack().reset_index()
                    else:
                        temp[x]= df_sel.reset_index()[['fuel', t]].set_index('fuel').T.rename({t:k}).stack().reset_index()
                res[k]=pd.concat(list(temp.values()))
            res=pd.concat(list(res.values()))

            res=res.replace(fueldict)

            # level_0, fuel, 0
            # We exlude fuels that equal to zero in at the regional level (GEO, NUC, BIO). OIL is very close to zero in `t`
            i='Uncertainty due to:'
            res=res.rename({'level_0':i}, axis=1)

            # Add a small random noise to the data to avoid overlapping points (otherwise NUC not working for JPN)
            res[0]=np.random.uniform(low=1e-8, high=1e-9, size=(len(res),))+res[0]


            fuels_list=[  ['COAL',"GAS", "WIND","SOL" ],
                          ['HYDRO','NUC', 'GEO', 'BIO','OIL']
                        ]
            
            for num, fuels in enumerate(fuels_list):
                fig = px.violin(res[res.fuel.isin(fuels)], y=0, color=i,#"level_0", 
                                color_discrete_sequence=["#DCC261", # SUPPLY/WEIGHTS (gold)
                                                        "#81117C",  # DEMAND (purple)
                                                        "#398383", # Convergence (acqua/greenish/bluish)
                                                        ], # define your color sequence
                                x="fuel", # different x category => smoker=[yes, no]
                                points='all', # show all points, (coloured by sex) for each x category
                                violinmode='overlay', # draw violins on top of each other
                                # default violinmode is 'group' as in example above
                                # box=True, # box plot
                                hover_data=res.columns,
                                title=f"{c} in {t}",
                                labels={'0':'EJ/yr'},
                                )

                fig=my_violin_layout(fig, title=f'{c} - {t}')
                
                # white backrgound
                white_background=True

                if white_background:
                    fig=add_white_background_to_plotly(fig)
                fig.show()
                fun_save_figure(
                                    fig,
                                    myfolder/'Figure_72_violin',
                                    f'{c}_fuel_{num}'
                                )


    if figure_73_weights_scatter:
        path='COMPOSITE' # Type of path `NAT`/`COMPOSITE`
        countries = ['AUS', 'JPN', 'NZL'] # List of countries
        t=2030
        # Older plot can be found here: https://github.com/iiasa/downscaler_repo/issues/181
        fueldict={'oil':'OIL', 'coal':'COAL', 'gas':'GAS', 'nuclear':'NUC', 'hydro':'HYDRO','geothermal':'GEO', 'biomass':'BIO', 'solar':'SOL', 'wind':'WIND'}
        
        # Read criteria weights
        fname=CONSTANTS.CURR_RES_DIR('step2')/'step2b_sensitivity'/'criteria_weights.csv'
        weights=fun_read_csv({'aa':fname},True, int)['aa'].reset_index()
        weights['CRITERIA']=[str(x) for x in weights.CRITERIA]
        
        res={}
        for path in ['NAT','COMPOSITE']:
            # Read dataframe from step2 either `NAT` or `COMPOSITE` path
            if path=='COMPOSITE':
                df=fun_read_csv(
                    {'aa':CONSTANTS.CURR_RES_DIR('step2')/'MESSAGEix-GLOBIOM 1.1-M-R12_2024_09_19.csv'}, 
                    True, int
                    )['aa']
                df=df.reset_index()
            elif path=='NAT':
                df=pd.read_csv(CONSTANTS.CURR_RES_DIR('step2')/'step2b_sensitivity'/'MESSAGEix-GLOBIOM 1.1-M-R12_Pacific OECDr_h_cpol_2023_12_21.csv')
            else:
                raise ValueError(f'Please pass a valid `path`: either `NAT` or `COMPOSITE`. You passed: {path}')    

            # Write criteria as string
            df['CRITERIA']=[str(x) for x in df.CRITERIA]
            av_criteria=df.CRITERIA.unique()
            
            if path=='COMPOSITE':
                df['IAM_FUEL']=[(x[len('Secondary Energy|Electricity|'):-len('XXxxxBLEND')]).lower() for x in df.reset_index().VARIABLE]
                df=df.replace(fueldict)
                df=fun_index_names(df)
            else:
                df=df.set_index(['TIME','ISO','CRITERIA','METHOD']).xs('dynamic_recursive', level='METHOD')
                df=pd.concat([df.unstack('TIME')[f].assign(IAM_FUEL=f).set_index('IAM_FUEL', append=True) for f in fueldict.values()]).reset_index('ISO')

            # Sclice for available criteria
            weights_sel=weights[weights.CRITERIA.isin(av_criteria)]

            # Dictionary `d` with fuels and associated criteria
            ww=weights.groupby('IAM_FUEL').median().reset_index()
            d={f:[i for i in ww[ww.IAM_FUEL==f] if isinstance(ww[ww.IAM_FUEL==f][i].iloc[0], float) and ww[ww.IAM_FUEL==f][i].iloc[0]>0] for f in fueldict.values()}

            # Merge the two dataframes
            if path=='COMPOSITE':
                df1=fun_xs(df, {'CRITERIA':list(av_criteria)}).drop('standard', level='CRITERIA').reset_index().set_index(['CRITERIA','IAM_FUEL'])
            else:
                df1=df
            df_merged=pd.concat([df1, 
                weights_sel.set_index(['CRITERIA','IAM_FUEL']).loc[df1.index]
                ]
                ,1)
            res[path]=df_merged

        # Loop through each fuel in the dictionary
        for fuel, criteria in d.items():
            # Create subplots: len(criteria) rows for criteria, len(countries) columns for countries
            fig = fun_scatter_plots_criteria_weights(t, res, countries, fuel, criteria, white_background= True)

            # Show the plot
            # plt.show()
            fun_save_figure(
                                fig,
                                myfolder/'Figure_73_scatter_weights',
                                f'{fuel}'
                            )


    if figure_8:
        print('************')
        print('Run hindcasting plots - Figure 8')
        print('This will take some time as we read xslx files...')
        sns.set_style("whitegrid")
        run_hindcasting_plots(
                                project="SIMPLE_hindcasting",
                                file="SIMPLE_hindcasting_2023_07_20_2010_harmo_step5e_Scenario_Explorer_upload_FINAL.xlsx",
                                var_list=["Emissions|CO2|Energy", 
                                        # 'Primary Energy|Coal', 'Primary Energy|Oil', 'Primary Energy|Gas', 
                                        #   'Final Energy', 'Final Energy|Electricity', 'Final Energy|Liquids', 'Final Energy|Gases', 'Final Energy|Solids',
                                          ],
                                models=[
                                        "IEA_PRIMAP_MESSAGEix-GLOBIOM 1.1-R12",
                                        "IEA_PRIMAP_GCAM 6.0 NGFS",
                                        "IEA_PRIMAP_REMIND-MAgPIE 3.2-4.6",
                                    ],
                                read_from_step="step5",
                                countrylist= iea_countries,
                                # countrylist=['AUS'],
                                mypath=CONSTANTS.CURR_RES_DIR('step5')/'Step5e_visuals'/'Paper1_technical_paper',
                            )

from plotly.graph_objects import Figure

def add_white_background_to_plotly(fig: Figure) -> Figure:
    """
    Applies a white background, light grey gridlines, and a black border around the plot area in a Plotly figure.
    
    Args:
        fig (Figure): A Plotly figure object to which the styling will be applied.
    
    Returns:
        Figure: The modified Plotly figure with the specified layout adjustments.
    
    This function customizes the appearance of the plot by:
    - Setting the plot and paper backgrounds to white.
    - Configuring gridlines and zero lines with specific colors.
    - Adding a black border around the figure.
    - Adjusting marker size within traces for clarity.
    - Setting plot margins and frame dimensions for optimal layout.
    """
    
    fig.update_layout(
        plot_bgcolor='white',        # Set the plot background color to white
        paper_bgcolor='white',       # Set the paper background color to white
        xaxis=dict(
            gridcolor='white',       # Set x-axis gridlines to white
            zerolinecolor='white'    # Set x-axis zero line to white
        ),
        yaxis=dict(
            gridcolor='lightgrey',   # Set y-axis gridlines to light grey
            zerolinecolor='lightgrey' # Set y-axis zero line to light grey
        ),
        margin=dict(l=50, r=50, t=40, b=10)  # Set small margins for framing effect
    )
    
    # Adjust point size using update_traces
    fig.update_traces(
        marker=dict(size=3)  # Set the size of points; reduce this number for smaller points
    )
    
    # Add a black border around the figure
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background outside plot area
        margin=dict(l=50, r=50, t=40, b=10),  # Margins to reveal framing
        autosize=True,
        width=900, height=400,
        shapes=[
            dict(
                type="rect",
                x0=0, x1=1, y0=0, y1=1,  # Covers the whole plot
                xref="paper", yref="paper",
                # line=dict(color="black", width=2),  # Black border with thickness of 2
                line=dict(color="lightgrey", width=2),  # Grey border with thickness of 2
                layer="below"
            )
        ]
    )
    
    return fig


def data_preparation_fig5(project, legend_color_box, legend_ls_box, scenarios, c, df_graph, var, mydict, df_graph2):
    i = (scenarios, project, df_graph, var, c)
    ii = i + (True, legend_color_box, legend_ls_box, 1, 0.5)
                        # To explore how it works:  CHECK HERE: jupyter_notebooks/exploring_sns_plots.ipynb!!!!
                        # fun_florian_plot(*ii)

    data=fun_from_iamc_to_step1b(df_graph2).reset_index()

    data= data[(data.SECTOR=='Final Energy')&(data.ISO==c)&(data.TARGET=='h_cpol')]
    data=data.replace(mydict)
    style_order=list(data.reset_index().METHOD.unique())
    style_order.reverse()
    y_label='Index (2010=1)'#'EJ/yr'
    data=data.rename({'RENAME_ME':y_label}, axis=1)
    return data,style_order,y_label

def fun_read_data_paper1(files, step, sel_scen, countrylist, models, project, check_variables=True):
    if CONSTANTS.CURR_RES_DIR('step5')==CONSTANTS.CURR_RES_DIR(step):
        file_dict={#'NGFS_2022':'1624889226995-NGFS_Scenario_Data_Downscaled_National_Data.csv',
                    'NGFS_2023':"MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2023_2025_01_31_Test_replicate_paper_2018_harmo_step5e_WITH_POLICY_None.csv",#'V4.1 NGFS Phase 4'
                    }   

    # Sanity check - input data 
    if not project and CONSTANTS.CURR_RES_DIR(step)!=CONSTANTS.CURR_RES_DIR('step1'):
        txt = "Please pass a `project` to the main function (currently is None)"
        raise ValueError(txt)
        
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
        countrylist_temp=fun_flatten_list_recursive(countrylist_temp)
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

        
    # Blueprint to fix scenarios 
    # TODO: replace `fun_xs_fuzzy` with `fun_xs`
    # df_iam=fun_read_df_iams(project, models)
    # df=pd.concat([fix_scenarios(df, df_iam, scen, refscen='h_cpol', dt=5 ) for scen in scenarios])
    
    # Check issues (missing variables) in the downscaling:
    if check_variables:
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
            fun_check_missing_variables(df,  var_to_check, check_IEA_countries, coerce_errors=True)

                # print("\n **** COUNT ALL VARIABLES REPORTED ****") 
            for model in df.reset_index().MODEL.unique():
                    # print(f"\n{model}")
                    # Check how many variables are reported for each country/scenario :
                d=check_variable_coverage_issues(df, model)
                if len(d)==0:
                    print(f"{model}: all good with variable coverage")
                else:
                    print(f"{model}: Only a few variables reported for :\n {d}")

    if project is not None:
        scenarios = f(sel_scen, scenarios) if isinstance(sel_scen, list) else scenarios
        scenarios=scenarios+['Historic data'] if sel_scen is not None and 'Historic data' in sel_scen else scenarios
            # Code below preserves scenarios order as defined in sel_scen (for stacked_plot)
        scenarios_ordered = fun_flatten_list([fun_fuzzy_match(scenarios,x, n=1) for x in sel_scen]) if sel_scen else scenarios
        if set(scenarios_ordered)==set(scenarios):
            scenarios=scenarios_ordered
        if sel_scen is not None and set(scenarios) == 0:
            raise ValueError(f"sel_scen {sel_scen} are not present in df: {scenarios}")
    print(f'Read data from these files {files}')
    return countrylist,models,scenarios,df, files_dict


def fun_scatter_plots_criteria_weights(
    t: int, 
    df_dict: Dict[str, pd.DataFrame], 
    countries: List[str], 
    fuel: str, 
    criteria: List[str],
    white_background: bool = True
) -> plt.Figure:
    """
    Generates scatter plots for multiple paths ('NAT'/'COMPOSITE'), each represented by a different color, 
    based on the provided criteria and countries.

    Parameters:
    ----------
    t : int
        The year or time value to be used for the analysis e.g 2030.
    df_dict : Dict[str, pd.DataFrame]
        A dictionary where the keys are path names and the values are dataframes 
        containing the data for each path.
    countries : List[str]
        A list of ISO-3 country names for which scatter plots will be generated.
    fuel : str
        The fuel type to filter data in the dataframes (e.g., 'SOL', 'WIND').
    criteria : List[str]
        A list of criteria used for the x-axis in the scatter plots. e.g. `DF_GW_ALL_FUELS`

    Returns:
    -------
    plt.Figure
        A matplotlib figure object containing the scatter plots.
    """
    
    # Create a color palette to differentiate between paths
    palette = sns.color_palette("husl", len(df_dict))

    # Create subplots with tight layout
    fig, axes = plt.subplots(len(criteria), len(countries), figsize=(15, 10))  
    
    # Add a suptitle for the entire figure
    plt.suptitle(f'{fuel} in {t}', fontsize=16)

    # Adjust layout to make room for suptitle
    plt.subplots_adjust(top=0.88, right=0.85)  # Adjust 'top' and 'right' for space

    # If there's only one row of criteria, axes won't be a 2D array, so handle it accordingly
    if len(criteria) == 1:
        axes = [axes]
    
    # Loop over each country and each criterion to create scatter plots
    for i, country in enumerate(countries):
        for j, criterion in enumerate(criteria):
            ax = axes[j, i]  # Access the current subplot (row: criterion, col: country)

            # Loop through each path in the dictionary and plot with different colors
            for k, (path, df_merged) in enumerate(df_dict.items()):
                # Extract the data for the current fuel and country
                data = df_merged.set_index('ISO', append=True).xs([fuel, country], level=['IAM_FUEL', 'ISO']).reset_index()

                # Create scatter plot for the current criterion
                ax.scatter(y=data[t], x=data[criterion], label=path, color=palette[k], alpha=0.7)

            # Set the title for the subplot
            if j == 0:  # Set title only on the first row
                ax.set_title(f'{country}')

            # Set labels
            ax.set_xlabel(criterion)
            ax.set_ylabel('EJ/yr')

            # Add legend for the different paths
            if j == len(criteria) - 1 and i == len(countries) - 1:  # Add legend in the last subplot
                ax.legend(title="Paths", bbox_to_anchor=(1.1, 1), loc='upper left')  # Shift legend left

            if white_background:
                # Set white background for the plot and figure
                fig.patch.set_facecolor('white')  # Background for the entire figure
                ax.set_facecolor('white')          # Background for the plot area (axis background)

                # Set grid color and style
                ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)  # Grey grid lines

                # Set border color if needed
                for spine in ax.spines.values():
                    spine.set_edgecolor('black')   # Black border around the plot area
    return fig



import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Callable

def filter_condition_5p1(data, mydict):
    data=data[(data.METHOD == mydict['wo_smooth_enlong_ENLONG_RATIO']) & 
                           (data.CONVERGENCE == '2150')]
    style='METHOD'
    return data, style

# Filter condition for plot_5p2
def filter_condition_5p2(data, mydict):
    data=data[(data.METHOD == mydict['wo_smooth_enlong_ENLONG_RATIO'])]
    style='METHOD'
    return data, style
 

# Filter condition for plot_5p3
def filter_condition_5p3(data, mydict):
    data=data
    style='METHOD'
    return data, style
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Callable

def plot_5p_general(
    countrylist: List[str], 
    project: Any, 
    legend_color_box: Any, 
    legend_ls_box: Any, 
    scenarios: Any, 
    df_graph: Any, 
    var: str, 
    mydict: Dict[str, Any], 
    df_graph2: Any,  # Indexed dataframe
    mypalette: Any,
    filter_condition: Callable[[Any, Dict[str, Any]], Any],  # Callable for filtering the data
    add_default_line: bool = False,
    add_legend: bool = True,
    legend_fontsize: int = 12,  # Optional argument for legend font size
    xtick_fontsize: int = 10,  # Optional argument for x-tick font size
    ytick_fontsize: int = 10,  # Optional argument for y-tick font size
    x_label: str = 'TIME',  # Optional argument for x-axis label
    xlabel_fontsize: int = 12,  # Optional argument for x-label font size
    ylabel_fontsize: int = 12,  # Optional argument for y-label font size
    title_fontsize: int = 15,  # Optional argument for subplot title font size
    suptitle_fontsize: int = 20,  # Optional argument for suptitle font size
    font_scale: Optional[int] =None, # Optional argument for font scale
    rect: Tuple[float, float, float, float] = [0, 0, 1, 0.95],  # Optional argument for tight layout
) -> None:
    """
    Generalized function to create a figure with 3 subplots, each plotting data for different countries.
    The data is processed for each country, and subplots are created side by side.

    Args:
        countrylist (List[str]): List of country codes or names.
        project (Any): Project-specific data or configuration.
        legend_color_box (Any): Color legend settings for the plot.
        legend_ls_box (Any): Line style legend settings for the plot.
        scenarios (Any): Scenarios data for the project.
        df_graph (Any): Main DataFrame for plotting.
        var (str): Variable name to be used in the title and plotting.
        mydict (Dict[str, Any]): Dictionary for mapping or replacing data values.
        df_graph2 (Any): Indexed dataframe (2010=1).
        mypalette (Any): Colormap for plotting.
        filter_condition (Callable): A function that filters the DataFrame based on different criteria.
        add_default_line (bool): If True, adds a default line for a specific condition (like in `plot_5p3`).
        legend_fontsize (int): Font size for the legend text.
        xtick_fontsize (int): Font size for x-ticks.
        ytick_fontsize (int): Font size for y-ticks.
        xlabel_fontsize (int): Font size for x-label.
        ylabel_fontsize (int): Font size for y-label.
        title_fontsize (int): Font size for subplot titles.
        suptitle_fontsize (int): Font size for the figure's main title.

    Returns:
        None: Displays the figure with subplots.
    """
    #increase font size of all elements
    if font_scale is None:
        sns.set(font_scale=2)
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)  # Creates 3 subplots side by side

    for x, c in enumerate(countrylist):
        # Data preparation function (assumed to return the prepared data, style order, and y-axis label)
        data, style_order, y_label = data_preparation_fig5(
            project, legend_color_box, legend_ls_box, scenarios, c, df_graph, var, mydict, df_graph2
        )
        
        # Set legend only for the last subplot
        if add_legend:
            mylegend = None if x != len(countrylist) - 1 else True
        else:
            mylegend = None

        # Apply the filter condition to the data
        filtered_data, style = filter_condition(data, mydict)

        # Main lineplot
        sns.lineplot(
            x=x_label, 
            y=y_label, 
            data=filtered_data,
            hue='FUNC', 
            palette=mypalette, 
            style=style, 
            style_order=style_order, 
            ax=axes[x], 
            legend=mylegend
        )

        # If add_default_line is True, add the default line for specific condition
        if add_default_line:
            sns.lineplot(
                x="TIME", 
                y=y_label, 
                data=data[(data.METHOD == mydict['wo_smooth_enlong_ENLONG_RATIO']) & 
                          (data.CONVERGENCE == '2150') & 
                          (data.FUNC == 'log-log')],
                hue='FUNC', 
                palette={'log-log': 'black'}, 
                legend=None, 
                ax=axes[x]
            )

        # Set subplot title as country name and adjust font size
        axes[x].set_title(c, fontsize=title_fontsize)
        
        # Customize xtick and ytick labels for each subplot
        axes[x].tick_params(axis='x', labelsize=xtick_fontsize, rotation=90)
        axes[x].tick_params(axis='y', labelsize=ytick_fontsize)
        
        # Set x and y labels and adjust font size
        axes[x].set_xlabel("TIME", fontsize=xlabel_fontsize)
        axes[x].set_ylabel(y_label, fontsize=ylabel_fontsize)

        # Increase legend font size if legend exists
        legend = axes[x].get_legend()

        # Set number of x-ticks
        axes[x].xaxis.set_major_locator(MaxNLocator(nbins=5))  # Set the number of x-ticks

        if legend is not None:
            plt.setp(legend.get_texts(), fontsize=legend_fontsize)  # Set legend font size
            # Move legend to the center of the plot
            legend.set_bbox_to_anchor((1.05, 1), transform=axes[x].transAxes)  # Center position
            legend.set_frame_on(False)  # Optionally, remove the frame around the legend for cleaner appearance
            
    # plt.legend(loc='center')  # Adjust legend position
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=rect)
    
    # Add a super title for the entire figure and adjust font size
    plt.suptitle(var, fontsize=suptitle_fontsize, x=0.4)

    # Show the final plot
    return fig

def plot_5p_generalv2(
    countrylist: List[str], 
    project: Any, 
    legend_color_box: Any, 
    legend_ls_box: Any, 
    scenarios: Any, 
    df_graph: Any, 
    var: str, 
    mydict: Dict[str, Any], 
    df_graph2: Any,  # Indexed dataframe
    mypalette: Any,
    filter_condition: Callable[[Any, Dict[str, Any]], Any],  # Callable for filtering the data
    add_default_line: bool = False,
    add_legend: bool = True,
    legend_fontsize: int = 12,  # Optional argument for legend font size
    xtick_fontsize: int = 10,  # Optional argument for x-tick font size
    ytick_fontsize: int = 10,  # Optional argument for y-tick font size
    x_label: str = 'TIME',  # Optional argument for x-axis label
    xlabel_fontsize: int = 12,  # Optional argument for x-label font size
    ylabel_fontsize: int = 12,  # Optional argument for y-label font size
    title_fontsize: int = 15,  # Optional argument for subplot title font size
    suptitle_fontsize: int = 20,  # Optional argument for suptitle font size
    font_scale: Optional[int] =None, # Optional argument for font scale
    rect: Tuple[float, float, float, float] = (0, 0, 1, 0.95),  # Optional argument for tight layout
) -> None:
    """
    Generalized function to create a figure with 3 subplots, each plotting data for different countries.
    The data is processed for each country, and subplots are created side by side.

    Args:
        countrylist (List[str]): List of country codes or names.
        project (Any): Project-specific data or configuration.
        legend_color_box (Any): Color legend settings for the plot.
        legend_ls_box (Any): Line style legend settings for the plot.
        scenarios (Any): Scenarios data for the project.
        df_graph (Any): Main DataFrame for plotting.
        var (str): Variable name to be used in the title and plotting.
        mydict (Dict[str, Any]): Dictionary for mapping or replacing data values.
        df_graph2 (Any): Indexed dataframe (2010=1).
        mypalette (Any): Colormap for plotting.
        filter_condition (Callable): A function that filters the DataFrame based on different criteria.
        add_default_line (bool): If True, adds a default line for a specific condition (like in `plot_5p3`).
        legend_fontsize (int): Font size for the legend text.
        xtick_fontsize (int): Font size for x-ticks.
        ytick_fontsize (int): Font size for y-ticks.
        xlabel_fontsize (int): Font size for x-label.
        ylabel_fontsize (int): Font size for y-label.
        title_fontsize (int): Font size for subplot titles.
        suptitle_fontsize (int): Font size for the figure's main title.

    Returns:
        None: Displays the figure with subplots.
    """
    

    # STEP1 preparing the data
    filtered_data_all=pd.DataFrame()
    for x, c in enumerate(countrylist):
            # Data preparation function (assumed to return the prepared data, style order, and y-axis label)
            data, style_order, y_label = data_preparation_fig5(
                project, legend_color_box, legend_ls_box, scenarios, c, df_graph, var, mydict, df_graph2
            )
            
            # Set legend only for the last subplot
            if add_legend:
                mylegend = None if x != len(countrylist) - 1 else True
            else:
                mylegend = None

            # Apply the filter condition to the data
            filtered_data, style = filter_condition(data, mydict)
            filtered_data_all=pd.concat([filtered_data_all, filtered_data])

    df=filtered_data_all
    ##############################################################
    # Get specific info

    d1={x:int(x[:4]) for x in df.METHOD.unique() if fun_check_if_all_characters_are_numbers(x[:4])} # tc until
    df['Until']=[d1.get(x,'default') for x in df.METHOD]

    d2={x:x[11:15] for x in df.METHOD.unique() if fun_check_if_all_characters_are_numbers(x[:4])} # over
    df['Over']=[d2.get(x,'default') for x in df.METHOD]

    d3={x:extract_text_in_parentheses(x) for x in df.METHOD.unique() if fun_check_if_all_characters_are_numbers(x[:4])} # scale
    df['Scale']=[d3.get(x,'default') for x in df.METHOD]

    ##########################################################################
    df2=df[df.METHOD!='Default method']
    if font_scale is None:
        sns.set(font_scale=2) # Increase font size of all elements
    # Set up the figure and axes
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, len(df['ISO'].unique()), figsize=(18, 5), sharey=True)
    fig.suptitle("Life Expectancy vs GDP Per Capita by Continent")

    # Initialize the scatter plots
    plots = {}
    # Create line plots for each continent
    for i, (continent, ax) in enumerate(zip(df2['ISO'].unique(), axes)):
        continent_data = df2[df2['ISO'] == continent]
        plots[continent] = sns.lineplot(
            data=continent_data,#[continent_data['TIME'] == 2010],  # Start with the first year
            x="TIME", y="Index (2010=1)", 
            hue='FUNC',
            style='METHOD',
            markers={x:'o' if d2[x]=='GDP ' else 's' for x in continent_data.METHOD.unique()},              # Line styles by 'Over'
            #markers=marker_dict,       # Use the custom marker dictionary
            # dashes={'GDP ':'','time':'', 'linear':'dotted'},          # Use the custom line style dictionary
            dashes={x: (1, 0) if d3[x] == 'linear' else (5, 5) for x in continent_data.METHOD.unique()},  # Solid for 'linear', dashed for others
            legend=None if i != len(df2['ISO'].unique()) - 1 else True, 
            ax=ax, 
            size='Until', sizes=(1, 2.5)
        )
        ax.set_title(continent)
        
        # If add_default_line is True, add the default line for specific condition
        if add_default_line:
            sns.lineplot(
                x="TIME", 
                y=y_label, 
                data=data[(data.METHOD == mydict['wo_smooth_enlong_ENLONG_RATIO']) & 
                        (data.CONVERGENCE == '2150') & 
                        (data.FUNC == 'log-log')],
                hue='FUNC', 
                palette={'log-log': 'black'}, 
                legend=None, 
                ax=axes[x]
            )
    
        # Set subplot title as country name and adjust font size
        axes[x].set_title(c, fontsize=title_fontsize)
        
        # Customize xtick and ytick labels for each subplot
        axes[x].tick_params(axis='x', labelsize=xtick_fontsize, rotation=90)
        axes[x].tick_params(axis='y', labelsize=ytick_fontsize)
        
        # Set x and y labels and adjust font size
        axes[x].set_xlabel("TIME", fontsize=xlabel_fontsize)
        axes[x].set_ylabel(y_label, fontsize=ylabel_fontsize)

        # Increase legend font size if legend exists
        legend = axes[x].get_legend()

        # Set number of x-ticks
        axes[x].xaxis.set_major_locator(MaxNLocator(nbins=5))  # Set the number of x-ticks

        if legend is not None:
            plt.setp(legend.get_texts(), fontsize=legend_fontsize)  # Set legend font size
            # Move legend to the center of the plot
            legend.set_bbox_to_anchor((1.05, 1), transform=axes[x].transAxes)  # Center position
            legend.set_frame_on(False)  # Optionally, remove the frame around the legend for cleaner appearance
            
    # plt.legend(loc='center')  # Adjust legend position
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=rect)
    
    # Add a super title for the entire figure and adjust font size
    plt.suptitle(var, fontsize=suptitle_fontsize, x=0.4)

    # Show the final plot
    return fig# ideas 
# sns.clustermap(data.set_index(['ISO','TIME','SECTOR','METHOD','CONVERGENCE','FUNC','TARGET']).unstack('TIME')['EJ/yr'], linewidth=1, cmap='Blues_r')


from typing import List, Any, Dict, Callable, Optional, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_5p_generalv3(
    countrylist: List[str], 
    project: Any, 
    legend_color_box: Any, 
    legend_ls_box: Any, 
    scenarios: Any, 
    # df_graph: Any, 
    var: str, 
    mydict: Dict[str, Any], 
    df_graph2: Any,  
    mypalette: Any,
    filter_condition: Callable[[Any, Dict[str, Any]], Any], 
    add_default_line: bool = False,
    add_legend: bool = True,
    legend_fontsize: int = 12,
    xtick_fontsize: int = 10,
    ytick_fontsize: int = 10,
    # xlabel_fontsize: int = 12,
    # ylabel_fontsize: int = 12,
    # title_fontsize: int = 15,
    suptitle_fontsize: int = 20,
    font_scale: Optional[int] = None,
    # rect: Tuple[float, float, float, float] = [0, 0, 1, 0.95],
    figsize: Tuple[int, int] = (18, 10), # width and height of the figure
) -> None:
    
    # Create the dataset
    filtered_data_all = pd.concat(
        [filter_condition(data_preparation_fig5(project, legend_color_box, legend_ls_box, scenarios, c, df_graph2, var, mydict, df_graph2)[0], mydict)[0]
         for c in countrylist],
        ignore_index=True
    )

    # Extract  information from the 'METHOD' column: 'Until', 'Over', and 'Scale'
    df2 = filtered_data_all.assign(
        Until=lambda x: x['METHOD'].apply(lambda m: int(m[:4]) if m[:4].isdigit() else 'default'),
        Over=lambda x: x['METHOD'].apply(lambda m: m[11:15] if m[:4].isdigit() else 'default'),
        Scale=lambda x: x['METHOD'].apply(lambda m: extract_text_in_parentheses(m) if m[:4].isdigit() else 'default')
    )
    
    # Create the figure
    sns.set(font_scale=font_scale or 2, style="whitegrid")
    fig, axes = plt.subplots(1, len(df2['ISO'].unique()), figsize=figsize, sharey=True)
    fig.suptitle(var, fontsize=suptitle_fontsize, x=0.37)
    sizes_dict={'default': 2.5, 2050: 1.5, 2100: 0.5} # default corresponds to a 2010 convergence (we immediately use long-term projections)
    for i, (country, ax) in enumerate(zip(df2['ISO'].unique(), axes)):
        country_data = df2[df2['ISO'] == country]
        if add_default_line:
            sns.lineplot(
                data=country_data[(country_data['METHOD'] == mydict.get('wo_smooth_enlong_ENLONG_RATIO', '')) & 
                              (country_data['CONVERGENCE'] == '2150') & 
                              (country_data['FUNC'] == 'log-log')&
                              (country_data.ISO==country)],
                x="TIME", y="Index (2010=1)",
                hue='FUNC', palette={'log-log': 'black'},
                legend=False, ax=ax,
                # try the below
                markers={x: 'o' if 'GDP' in x else 's' for x in country_data['METHOD'].unique()},
                dashes={x: (1, 0) if 'linear' in x else (5, 5) for x in country_data['METHOD'].unique()},
                size='Until', sizes=sizes_dict,
            )
            country_data=country_data[country_data['METHOD'] != 'Default method']

        # Create markers dictionary
        markers_dict={x: 'o' if 'GDP' in x else 's' for x in country_data['METHOD'].unique()}
        markers_dict["Default method"]=(1,0)#""

        # Create dashes dictionary
        dashes_dict={x: (1, 1) if 'linear' in x  # dotted line for 'linear' scale
                     else (5, 5) for x in country_data['METHOD'].unique() # dashed line for other (log) scale
                     }
        dashes_dict["Default method"]=(1,0) # solid line for 'Default method'


        # Create line plot for each country
        sns.lineplot(
            data=country_data,
            x="TIME", y="Index (2010=1)",
            hue='FUNC', palette=mypalette,
            style='METHOD',
            markers=markers_dict,
            dashes=dashes_dict,
            size='Until', sizes=sizes_dict,
            legend=None if i != len(df2['ISO'].unique()) - 1 else 'brief',
            ax=ax
        )
        # print(country_data.Until.unique())
        
        # Customize the subplot
        ax.set(title=country, xlabel="TIME", ylabel="Index (2010=1)")
        ax.tick_params(axis='x', labelsize=xtick_fontsize, rotation=90)
        ax.tick_params(axis='y', labelsize=ytick_fontsize)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        
        if i == len(df2['ISO'].unique()) - 1 and add_legend:
            legend = ax.get_legend()
            if legend:
                legend.set_bbox_to_anchor((1.05, 1), transform=ax.transAxes)
                plt.setp(legend.get_texts(), fontsize=legend_fontsize)
                legend.set_frame_on(False)
    
    # if rect is not None:
    plt.tight_layout()

    # Adjust layout to make room for suptitle
    plt.subplots_adjust(top=0.87)#, right=1)  # Adjust 'top' and 'right' for space
    # plt.show()
    return fig

def create_fuel_violin_box_plot(res: pd.DataFrame, fuels: list, c: str, t: str, labels_dict:dict={'1':'A','2':'B'}, labels_colors:dict={'A':'red', 'B':'blue', 'standard':'black'}, points:str=False) -> go.Figure:
    """
    Creates a violin and box plot for the specified fuels and criteria in the dataframe.
    
    Parameters:
    ----------
    res : pd.DataFrame
        A pandas DataFrame containing the data with columns 'FUEL', 'SET', and 'CRITERIA'.
    fuels : list
        A list of fuel types to be included in the plot (e.g., ['COAL', 'GAS', 'WIND', 'SOL', 'HYDRO']).
    c : str
        The label to use in the plot title (e.g., a country or region).
    t : str
        A time or additional label to use in the plot title (e.g., year).
    
    Returns:
    -------
    go.Figure
        A Plotly Figure object containing the violin and box plots.
    """
    
    # Initialize the plot
    plot = go.Figure()

    # Filter the data for the specified fuels
    res2 = res[res.FUEL.isin(fuels)]


    # Rename '1' as 'A' and '2' as 'B' for better visualization
    res2=res2.replace(labels_dict)

    # My custom color dictionary for each set of criteria (e.g., distribution '1', '2', 'standard')
    

    for k,v in labels_colors.items():
        
        # Add violin trace for SET '2'
        plot.add_trace(go.Violin(x=res2[res2.SET == k]['FUEL'],
                                y=res2[res2.SET == k][0],
                                line_color=v,
                                points=points,
                                name=k))



    # # Update layout with title, axis labels, and font settings
    plot=my_violin_layout(plot, title=f'{c} - {t}')
    # plot.update_layout(
    #     violinmode='overlay',
    #     title={
    #         'text': f'{c} - {t}',  # Plot title with dynamic country/region and time
    #         'font': {'size': 26}    # Title font size
    #     },
    #     yaxis_title='EJ/yr',        # Y-axis title
    #     xaxis_title='Fuels',        # X-axis title
    #     font=dict(
    #         family="Arial",         # Font family
    #         size=20,                # Font size for labels and ticks
    #         color="black"           # Font color
    #     )
    # )

    # Set opacity for all traces
    plot.update_traces(opacity=0.8)

    # Return the Plotly figure
    return plot


def my_violin_layout(fig, title):
    # Update layout with title, axis labels, and font settings
    return fig.update_layout(
        violinmode='overlay',
        title={
            'text': title,  # Plot title with dynamic country/region and time
            'font': {'size': 26}    # Title font size
        },
        yaxis_title='EJ/yr',        # Y-axis title
        xaxis_title='Fuels',        # X-axis title
        font=dict(
            family="Arial",         # Font family
            size=20,                # Font size for labels and ticks
            color="black"           # Font color
        )
    )
            
            
if __name__ == "__main__":
    # FOR HINDCASTING PLEASE USE `hindcasting_results_comparison.py`
    # main() # EUAB
    # NGFS Below:
    main(
        # figure_step2b_boxplot=True, # Step 2b sensitivity analysis - Boxplot
        # figure_step2b_violin_plot=True, # Step 2b sensitivity analysis - Violin plot - # CRAMERY MYPALETTE
        models=["*MESSAGE*"],
        # files={'MESSAGEix-GLOBIOM 1.1-M-R12':'MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2023_2025_01_31_Test_replicate_paper_2018_harmo_step5e_WITH_POLICY_None.csv'},
        project='NGFS_2023',
        figure_2=True, # Energy Intensity plot
        figure_4=True, # Figure 4 (stacked plot)
        # figure_5=True, # Figure 5 (without parallel coordinates)
        # figure_5_parallel=True, # Figure 5 (parallel coordinates)
        # figure_72_boxplot=True, # Figure 7.2 (Step 2b sensitivity analysis) - Boxplot
        # figure_step2b_violin_plot=True, # Step 2b sensitivity analysis (JUST WEIGHTS) - Violin plot - # CRAMERY MYPALETTE
        # figure_72_violin=True, # Figure 7.2 (Step 2b sensitivity analysis: WEIGHTS/DEMAND/CONVERGENCE) - Violin
        # figure_73_weights_scatter=True, # Figure 7.3 (Step 2b sensitivity analysis) - Scatter plot for SUPPLEMENTARY INFORMATION
        # figure_8=True, # Figure 8 (hindcasting)
        )

