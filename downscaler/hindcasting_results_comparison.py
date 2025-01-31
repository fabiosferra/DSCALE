import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from downscaler import CONSTANTS
from downscaler.utils_pandas import fun_read_csv_or_excel
from downscaler.utils import (
    fun_interpolate,
    fun_rename_index_name,
    fun_xs,
    fun_flatten_list,
    fun_save_figure,
    fun_ghg_emi_from_primap,
    fun_index_names,
    fun_most_recent_iea_data,
    fun_get_iam_regions_associated_with_countrylist,
    fun_read_df_iams,

)
from downscaler.utils_simple_downscaling import fun_downscale_using_same_proportions_across_regions
import matplotlib.pyplot as plt
from downscaler.utils import fun_eu28, fun_regional_country_mapping_as_dict
from downscaler.fixtures import iea_countries

def main(
    project: str,
    file: str,
    var_list: list,
    models: Optional[list],
    read_from_step: str = "step5",
    countrylist=None,
    mypath=None, # Path to save the graphs
):
    """Compares hindcasting results with historical data and saves plots in an "hindcasting_graphs" folder.

    Parameters
    ----------
    project : str
        Your project
    file : str
        Csv suffix of your file
    models : list
        Models to be compared
    var_list : list
        List of variables
    read_from_step : str, optional
        From which step you want to read your results, by default "step5"
    """
    p_dir = CONSTANTS.INPUT_DATA_DIR / project
    step_dir = CONSTANTS.CURR_RES_DIR(read_from_step)/'2023'
    if models is None:
        # Get models list from default_mapping.csv file
        default_mapping = pd.read_csv(p_dir / "default_mapping.csv")
        models = [x for x in default_mapping.columns if x.endswith(".REGION")]
        models = [x.replace(".REGION", "") for x in models]

    df_downs = fun_read_csv_or_excel(file, models, folder=step_dir).droplevel("FILE")
    df_downs = fun_interpolate(df_downs, False, range(2010, 2021), True)
    df_downs = fun_rename_index_name(df_downs, {"ISO": "REGION"})
    df_hist = fun_read_csv_or_excel("hist_country_level_data.csv", None, folder=p_dir)
    df_iam = fun_index_names(fun_read_df_iams(project, ), True, int) 
    df_iea = pd.concat([fun_most_recent_iea_data(), fun_ghg_emi_from_primap(None).droplevel('FILE')])

    # CHECK max error by MODEL/region and suggest
    # for var in var_list:
    #     i = (project, models, df_downs, df_hist, var, 4, 3)
    #     sugg_reg = fun_suggest_countries_all_models_combined(*i)
    #     print(sugg_reg.ISO.unique())

    # Plot countries
    for var in var_list:
        varn = var.replace("|", "_")
        if mypath is None:
            path = CONSTANTS.CURR_RES_DIR(read_from_step) / f"hindcasting_graphs/{project}/{varn}"
        else:
            path = mypath / f"hindcasting_graphs/{project}/{varn}"
        os.makedirs(path, exist_ok=True)
        clist_av = df_downs.xs(var, level="VARIABLE").reset_index().REGION.unique()
        countrylist = clist_av if countrylist is None else countrylist
        for c in set(countrylist) & set(clist_av):
            
            file_name=c
            # Alternative path in case of single country analysis (for multiple variables)
            if len(countrylist)==1 and len(var_list)>=2:
                path = mypath / f"hindcasting_graphs/{project}/{c}"
                file_name=varn

            # Simple alternative (proportional) downscaling method
            df_prop_method = fun_prop_method(project, models, df_iam, df_iea, var, c)
            df_prop_method = df_prop_method.assign(REGION=c).assign(VARIABLE=var).assign(UNIT=np.nan).assign(SCENARIO='HISTCR')
            df_prop_method = df_prop_method.reset_index().set_index(df_downs.index.names)
            
            fig = fun_hindcasting_performance_graph(df_downs,df_prop_method, #df_downs, 
                                        var, c)
            # figure below without comparison with proportional method
            # fig = fun_hindcasting_graph(pd.concat([df_prop_method.drop('Primap'), df_downs]), #df_downs, 
            #                             df_hist, var, c)
            fun_save_figure(fig, path,file_name)
            # fig.figure.savefig(path / f"{c}.png")
            plt.close()
    print("done")

def fun_prop_method(project, models, df_iam, df_iea, var, c):
    # Reads either PRIMAP or IEA to get historical data
    primap=fun_ghg_emi_from_primap([c]) if 'Emissi' in var else df_iea.xs(c, level='REGION', drop_level=False)

    # Proportional dowsncaling
    res={}
    for m in models:
        r=list(fun_get_iam_regions_associated_with_countrylist(project, [c], m).values())[0][:-1]
        res[m]=fun_downscale_using_same_proportions_across_regions(var, 2010, 
                                                                    primap, 
                                                                    df_iam.xs([f'{m}|{r}',var], level=['REGION','VARIABLE'], drop_level=False)
                                                                    ).assign(MODEL=m).set_index('MODEL', append=True).droplevel(['UNIT','VARIABLE'])
    res=pd.concat(list(res.values()))

    # Interplate results from 2005-2020
    mylist= list(range(2005,2021))
    res= fun_interpolate(res.assign(REGION=c).assign(SCENARIO='HIST').set_index(['REGION','SCENARIO'], append=True), False, mylist, True)

    # Add historical data to `res` and drop index
    drop_list=['SCENARIO','REGION','UNIT']
    drop_list= drop_list+['FILE'] if 'FILE' in primap.index.names else drop_list
    return pd.concat([res.droplevel(['REGION','SCENARIO']), primap.xs(var, level='VARIABLE').droplevel(drop_list)])

def fun_hindcasting_graph(df_downs, df_hist, var, c):
    fig, axes = plt.subplots()
    df_graph = pd.concat([df_downs, df_hist]).xs(var, level="VARIABLE")
    df_graph = df_graph.xs(c, level="REGION").droplevel(["SCENARIO", "UNIT"])
    df_graph = df_graph[range(2005, 2021)].T
    df_graph = df_graph.replace(0, np.nan)
    maxv = np.round(df_graph.max().max() * 1.5, 1)
    df_graph.plot(ylim=(0, maxv), title=f"{var}|{c}")
    return plt

def fun_hindcasting_performance_graph(df_graph: pd.DataFrame, df_prop_method: pd.DataFrame, var: str, c: str) -> plt:
    """
    Creates a hindcasting graph for the specified variable and region, comparing downscaled model results 
    with historical data using both the standard downscaling method and a simple proportional method.

    Parameters
    ----------
    df_graph : pd.DataFrame
        DataFrame containing the downscaled model results with a multi-level index, including 'VARIABLE', 'REGION', 'SCENARIO', and 'UNIT'.
    df_prop_method : pd.DataFrame
        DataFrame containing the proportional method results with a similar structure as `df_graph`. This dataframe should also include the historical data.
    var : str
        The variable of interest (e.g., 'Emissions|CO2|Energy', 'Secondary Energy|Electricity'),  found in the 'VARIABLE' level of the index.
    c : str
        The country of interest (ISO3 code), found in the 'REGION' level of the index. e.g., 'USA', 'CHN'.

    Returns
    -------
    matplotlib.pyplot
        A matplotlib plot showing the downscaled results and proportional method for comparison with historical data.
    
    Notes
    -----
    - This function performs hindcasting, a method of comparing historical data with model projections.
    - The MODEL name of the historical data source depends on the variable: 'Primap' for emissions-related variables, 'IEA' for energy variables.
    - The graph includes both the downscaled model results (solid lines) and proportional method results (dashed lines) for the specified region.
    - Historical data is shown in black for visual comparison.
    
    Steps
    -----
    1. Filters the input data (`df_graph` and `df_prop_method`) to include only the selected `var` and `c` (region).
    2. Creates a time series plot with downscaled results (`df_graph`) for the years 2005-2020, with missing data (NaN) for years 2005-2009.
    3. The proportional method results (`df_prop_method`) are plotted alongside the downscaled results.
    4. The historical (available in the `df_prop_method`) data is plotted in black for comparison.
    
    Example
    -------
    ```python
    plt = fun_hindcasting_graph2(df_graph, df_prop_method, 'CO2 Emissions', 'USA')
    plt.show()
    ```
    """
    
    # Copy dataframes
    df_graph = df_graph.xs(var, level="VARIABLE").copy(deep=True)
    df_prop_method = df_prop_method.xs(var, level="VARIABLE").copy(deep=True)
    unit=df_graph.reset_index().UNIT.unique()[0]
    # Determine the historical MODEL based on the variable name
    histmodel = 'Primap' if 'Emissi' in var else 'IEA'

    # Filter downscaled results and prepare for plotting
    df_graph = df_graph.xs(c, level="REGION").droplevel(["SCENARIO", "UNIT"])
    
    # Add NaN values for the years 2005-2009 (we don't have downscaled data for these years)
    for t in range(2005, 2010):
        df_graph[t] = np.nan
        
    # Filter years between 2005 and 2020 for plotting
    df_graph = df_graph[range(2005, 2021)].T
    df_graph = df_graph.replace(0, np.nan)
    
    colorlist = ['blue', 'green', 'red']  # Color list for plotting

    # Prepare proportional method data for plotting
    df_prop_method = df_prop_method.xs(c, level="REGION").droplevel(["SCENARIO", "UNIT"])
    df_prop_method = df_prop_method[range(2005, 2021)].T
    df_prop_method = df_prop_method.replace(0, np.nan)
    
    # Rename proportional method columns to include 'simple' except for the historical model
    df_prop_method.columns = [f"{x} - simple" if x != histmodel else x for x in df_prop_method.columns]

    # Plot the downscaled model results
    maxv = np.round(df_graph.max().max() * 1.5, 1)
    ax = df_graph.plot(ylim=(0, maxv), title=f"{var} - {c}", ls='-', color=colorlist)

    # Plot the proportional method results (dashed lines) on the same axis
    ax2 = df_prop_method.drop(histmodel, axis=1).plot(ylim=(0, maxv), title=f"{var}|{c}", ls='--', ax=ax, color=colorlist)
    
    # Plot the historical data (solid black line)
    df_prop_method[histmodel].plot(ylim=(0, maxv), title=f"{var} - {c}", ls='-', ax=ax2, color='black')
    plt.ylabel(unit)
    return plt


def fun_suggest_countries_as_single_region(
    project,
    model,
    df_downs,
    df_hist,
    var,
    no_big_countries: Optional[int] = 4,
    no_worst_countries: Optional[int] = 3,
    y=2019,
) -> dict:
    """Suggest countries to be modeled as single-region in IAMs results (e.g. {'Latin America':['BRA','VEN'], 'Western Europe':['TUR']}).
    Suggest countries to be singled-out for all regions in a given model by using two criteria:
    - Countries should be relatively big (please select `no_big_countries` in a given region)
    - Hindcasting performance of downscaled results should be poor (please select `no_worst_countries` to be considered in each region)

    Parameters
    ----------
    project : str
        your project (e.g. NGFS)
    model : str
        Your model (hindcasting is based on different regional mappings based on a given model)
    df_downs :  pd.DataFrame,
        Dataframe with dowscaled hindcasting results over historical periods
    df_hist :  pd.DataFrame,
        Dataframe with observed historical data
    var : str
        Target variable to check hindcasting performance. (we check hindcasting performance only for the big countries in the region)
    no_big_countries : int, optional
        Maximum number of large countries in the region. Increasing this number will increase presence of small countries.
        If `no_big_countries=None`, there will be no distiction between small and big countries (we consider infinite number of big countries) by default 4
    no_worst_countries : int, optional
        Number of countries (with the worst hindcasting performance) to be considered. If None we do not asses the hindcasting performance, by default 3
    y: int, optional
        Year for checking hindasting performance
    Returns
    -------
    dict
        Dictionary with suggested countries, by region
    """

    cmap = fun_regional_country_mapping_as_dict(model, project)
    res = {}
    # Region loop
    for k, v in cmap.items():
        model_results = fun_xs(df_downs, {"REGION": v})
        model_results = model_results.xs(f"Downscaling[{model}]", level="MODEL")[y]
        model_results = model_results.xs(var, level="VARIABLE")
        # Get at least 3 big countries (from a given region) from the downscaled results
        selc = model_results.sort_values(ascending=False)
        selc = selc[:no_big_countries]
        selc = selc.reset_index().REGION.unique()
        get_results = fun_xs(model_results, {"REGION": list(selc)})
        # Check error for the big countries
        if len(model_results) <= 1:
            res[k] = "None"
        else:
            performance = (1 - get_results / df_hist) ** 2
            performance = performance.sort_values(ascending=False).dropna()
            perf_countr = performance.reset_index().REGION.unique()
            mylist = list(perf_countr)[:no_worst_countries]
            res[k] = mylist

    return res


def fun_suggest_countries_all_models_combined(
    project: str,
    models: list,
    df_downs: pd.DataFrame,
    df_hist: pd.DataFrame,
    var: str,
    no_big_countries: Optional[int] = 4,
    no_worst_countries: Optional[int] = 3,
    y: int = 2019,
) -> pd.DataFrame:
    """Provide a list of countries that are not present as single region in any of the IAMs and with a poor hindcasting performance in the downscaled results.
    Those countries could be represented as single region by IAMs. We suggest new single-regions in IAMs by using two criteria:
    - Countries should be relatively big (please select `no_big_countries` in a given region)
    - Hindcasting performance of downscaled results should be poor (please select `no_worst_countries` to be considered in each region)

    Parameters
    ----------
    project : str
        you project
    models : list
        list of models (hindcasting of downscaled results is done using different regional mappings based on models e.g. MESSAGE)
    df_downs : pd.DataFrame
        downscaled (hindcasting) results
    df_hist : pd.DataFrame
        Dataframe with historical data
    var : str
        Variable to be checked for hindcastig performance
    no_big_countries : Optional[int]
        Number of countries considered as `big` in a given region. Increasing this value will increase the presence of small countries.
        If `no_big_countries=None`, there will be no distiction between small and big countries (we consider infinite number of big countries), by default 4
    no_worst_countries : Optional[int]
        Number of countries with the worse hindcasting performance. If None we do not asses the hindcasting performance, by default 3
    y : int, optional
        Year for evaluating hindcasting performance, by default 2019

    Returns
    -------
    pd.DataFrame
        Dataframe with list of countries to be suggested as single region (by combining info from all model).
        Countries are grouped by IPCC regions.
    """



    # Read df below just to get the IPCC region groupings (Developed Countries, Asia...)
    ipcc = pd.read_csv(CONSTANTS.INPUT_DATA_DIR / "Grassi_regional_mapping.csv")
    df_hist = df_hist.xs("Primap", level="MODEL").xs(var, level="VARIABLE")
    df_hist = df_hist.droplevel("FILE")[y]
    res_all_models = {
        m: fun_suggest_countries_as_single_region(
            project,
            m,
            df_downs,
            df_hist,
            var,
            no_big_countries=no_big_countries,
            no_worst_countries=no_worst_countries,
            y=y,
        )
        for m in models
    }
    # Combine country list from all models -> get list of countries (suggested as single region) that are present across all models.
    # In other words we want to get a list of countries that are not well represented in any
    comb_list = set.intersection(
        *[set(fun_flatten_list(list(res_all_models[m].values()))) for m in models]
    )
    ipcc = ipcc[ipcc.ISO.isin((list(comb_list)))]
    return ipcc.sort_values("R5_region").set_index("R5_region")


if __name__ == "__main__":
    main(
        project="SIMPLE_hindcasting",
        # file="IEA_PRIMAP_GCAM 6.0 NGFS_2023_07_19.csv",
        # file="2023_07_19.csv",
        file="SIMPLE_hindcasting_2023_07_20_2010_harmo_step5e_Scenario_Explorer_upload_FINAL.xlsx",
        # file="2023_07_24_PR_131.csv",  # PR_131 from step3
        # file="2023_07_20.csv",  # PR_175 from step3
        var_list=[  # "Final Energy",'Primary Energy'
            #   "Final Energy|Electricity",
            #   "Final Energy|Liquids",
            # "Final Energy|Solids",
            # "Final Energy|Gases",
            #'Secondary Energy|Electricity',
            # 'Secondary Energy|Liquids',
            # 'Secondary Energy|Solids',
            # "Secondary Energy|Gases",
            # "Emissions|CO2|Energy",
            "Primary Energy|Coal",
            # 'Secondary Energy|Electricity|Hydro',
        ],
        models=[
            "IEA_PRIMAP_MESSAGEix-GLOBIOM 1.1-R12",
            "IEA_PRIMAP_GCAM 6.0 NGFS",
            "IEA_PRIMAP_REMIND-MAgPIE 3.2-4.6",
        ],
        # countrylist=['NPL', 'LKA', 'BGD']#
        read_from_step="step5",
        # countrylist=["TUR","ALB","SRB","CHE","ISL","NOR"]
        # countrylist=["TUR"],  # fun_eu28(),  # None,#['FIN']
        # countrylist=['ETH', 'UZB', 'KAZ', 'UKR', 'BLR', 'BIH', 'KGZ', 'TUR', 'TKM',  'PHL', 'MDA', 'SRB', 'NLD', 'HKG', 'ECU',]
        # countrylist=['DEU', 'AUS', 'TUR', 'CHN', 'NGA', 'ZAF','IRN', 'JPN', 'CHE','FRA','CAN']
        countrylist=['ZAF']
    )
