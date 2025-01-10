import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd  # Load the Pandas enlong_calc(df_main) ## calculating enlonglibraries with alias 'pd'

from downscaler import CONSTANTS
from downscaler.fixtures import (
    fast_list,
    fun_conv_settings,
    hydrogen_list,
    iea_flow_dict,
    med_list,
    slow_list,
    iea_countries,
)
from downscaler.utils_pandas import (
    fun_rename_index_name,
    fun_xs,
)
from downscaler.input_caching import InputFile, get_selection_dict
from downscaler.utils import (  # harmo_ratio, add_region_simple, 
    InputFile,
    convert_to_list,
    fun_blending,
    fun_bottom_up_harmonization,
    fun_heat_downs,
    fun_hydrogen_downs,
    fun_list_time_of_conv,
    fun_load_iea,
    fun_make_var_dict,
    fun_read_df_iam_all,
    fun_read_df_iam_iamc,
    fun_top_down_harmonization,
    setindex,
    fun_str_sector_list,
    fun_smooth_enlong,
    get_git_revision_short_hash,
    run_sector_harmo_enhanced,
    fun_flatten_list,
    fun_blending_with_sensitivity,
    fun_create_var_as_sum,
    fun_add_units,
    fun_create_missing_sectors_as_sum,
    fun_check_negative_energy_variables_by_model,
    fun_check_inconsistencies,
    fun_from_step1b_to_iamc,
    fun_resolve_inconsistencies_step1b,
    fun_from_iamc_to_step1b,
    fun_read_iea_data_from_iamc_format,
    fun_harmonize_hist_data_by_preserving_sum_across_countries
)

from downscaler.fixtures import hydrogen_list, iea_flow_dict
from downscaler.input_caching import (
    InputFile,
    get_selection_dict,
)


def run_sector_harmo(
    df: pd.DataFrame,
    model: str,
    region: str,
    df_iam: pd.DataFrame,
    new_var_dict: dict,
    var_dict_demand: dict,
    var_dict_supply: dict,
    CONV_DATA_DIR,
    var_list: List = ["ENSHORT_REF", "ENLONG_RATIO"],
    max_iterations: int = 5,
    model_in_region_name: bool = False,
):
    """
    Run sectorial harmonization for a given `model` / `region`.
     - It creates new variables (defined in `new_var_dict`) using a bottom-up approach (as the sum of a list of variables)
     - It harmonizes existing variables (defined in `var_dict`) using a top-down approach.

    We do this for all variables type listed in `var_list` (ENSHORT_REF / ENLONG_RATIO).
    We keep on iterating for a maximum of `max_iterations`

    Args:
        df (pd.DataFrame): [Dataframe with previously downscaled results from step1]
        model (str): [model]
        region (str): [region]
        df_iam (pd.DataFrame): [Regional IAM results in IAMc format]
        new_var_dict (dict): [Dictionary with new variables to be created using a bottom-up approach]
        var_dict (dict): [Dictionary with list of variables to be harmonined]
        var_list (List): [variables type to be harmonized e.g. ENSHORT_REF / ENLONG_RATIO]
        max_iterations (int): [Maximum iterations]

    Returns:
        df [Dataframe]: [the updated dataframe]
        res [Dictionary]: [Result dictionary for ENSHORT_REF/ ENLONG_RATIO]
    """
    res = {}
    reg = (
        f"{region}r" if model_in_region_name else region.replace(f"{model}|", "") + "r"
    )

    region_replaced = region.replace(f"{model}|", "")

    for x in var_list:
        df_sect_harm = df.unstack("TIME")[x]

        ## Creating new variables using a 'Bottom up' approach: new_var_dict.keys = sum of new_var_dict.values()
        ## Example: 'Final Energy|Industry' defined as the sum of ['Final Energy|Industry|Liquids', 'Final Energy|Industry|Solids', ...]
        if new_var_dict:
            for j, i in new_var_dict.items():
                i_list = [
                    x for x in i if x in df_sect_harm.reset_index().SECTOR.unique()
                ]
                if i_list:
                    if j in df_iam.reset_index().VARIABLE.unique():
                        df_sect_harm = fun_bottom_up_harmonization(
                            model,
                            reg,
                            j,
                            i,
                            df_sect_harm,
                            df_iam,
                        )

        var_dict_list = [
            var_dict_demand,
            var_dict_supply,
        ]  ## list of variables that we want to update
        var_dict_list = {"demand": var_dict_demand, "supply": var_dict_supply}
        res_conv_all_dict = {}
        for v, var_dict in var_dict_list.items():
            conv_all = pd.DataFrame(
                index=max_iterations,  # columns=list(var_dict_supply.keys())
            )
            for m in max_iterations:
                if var_dict_demand:
                    for j, i in var_dict.items():
                        myl = [
                            x for x in i if x in df_iam.reset_index().VARIABLE.unique()
                        ]
                        if j in df_sect_harm.reset_index().SECTOR.unique():
                            if myl:
                                df_sect_harm = fun_top_down_harmonization(
                                    model,
                                    reg,
                                    j,
                                    myl,
                                    df_sect_harm,
                                    df_iam,
                                    # mandatory_harmonization=False,  # do not harmonize if a sub-sector is missing
                                )

                    ## Conv_all block
                    for n in list(var_dict.keys()):  ## conv_all
                        main = n  # main sector e.g. Final Energy
                        subs = var_dict[
                            n
                        ]  # sub sectors e.g. ['Final Energy|Industry', 'Final Energy|Transportation', 'Final Energy|Residen...Commercial']
                        if main in df_sect_harm.reset_index().SECTOR.unique():
                            main_countries = df_sect_harm.xs((main), level=("SECTOR"))
                            subs_countries = df_sect_harm[
                                df_sect_harm.index.get_level_values("SECTOR").isin(subs)
                            ]  # .xs([ target_patterns[0]], level=[ 'TARGET'])
                            conv_all.loc[m, n] = (
                                np.abs(
                                    main_countries
                                    - subs_countries.groupby(["ISO", "TARGET"]).sum()
                                ).max()
                            ).max()  # Maximum deviation between main setcor and sub-sectors (across all countries and all targets)
            res_conv_all_dict[v] = conv_all

        # Convergence values for each sector (max across all countries/scenarios):
        conv_comb = pd.concat(
            [res_conv_all_dict["demand"], res_conv_all_dict["supply"]], axis=1
        )  ## combined supply and demand
        conv_comb.to_csv(
            CONV_DATA_DIR / f"{model}_{region_replaced}_conv_{v}_{x}.csv",
            mode="w",
            header=True,
        )
        ## Dataframe with Convergence value for each sector / iterations (across all countries and scenarios)
        print(conv_comb)
        print(region, x, "done")

        ## Unstack results and copy values to daframe
        df_unstack = df_sect_harm.stack()
        df_unstack.index.names = ["ISO", "TARGET", "SECTOR", "TIME"]
        df_unstack = pd.DataFrame(df_unstack, columns=["VALUE"])
        setindex(df_unstack, list(df.index.names))

        df.loc[:, x] = df_unstack["VALUE"]  ## Copy value to df daframe
        res[x] = df_unstack

    ## Append new variables created using a bottom up approach (listed in new_var_dict), to df
    for k in new_var_dict:
        df_append = df[df.index.get_level_values("SECTOR") == "Final Energy"].copy()
        df_append.rename(index={"Final Energy": k}, inplace=True)
        df_append.loc[:, :] = np.nan
        for x in var_list:
            df_append.loc[:, x] = res[x]["VALUE"]
        if k not in df.index.get_level_values("SECTOR").unique():
            df=pd.concat([df,df_append])
            # df = df.append(df_append)
    return df


def main(
    project_name: str,
    csv_in: str,
    region_patterns: Union[str, list] = "*",
    model_patterns: Union[str, list] = "*",
    target_patterns: Union[str, list] = "*",
    csv_out: str = "harmo",
    max_iter=5,
    conv_dict={},
    default_blend="2150_BLEND",  # Chosen convergence (for all energy sector variables and all scenarios)
    model_in_region_name: bool = False,
    func_form: str = "log-log",
    run_sensitivity: bool = False,
    split_res_and_comm: bool = False,
    ol:List[str] = ["GDPCAP", "TIME"],
    tcl:List[int] = [2050, 2100],
    resolve_inconsistencies:bool=False,
    ref_target=''
    # downscaler.USE_CACHING=True,
):
    max_iterations = list(
        range(max_iter)
    )  ## maximum iterations for adjusting all variables listed in the var_dict_demand (or in the var_dict_supply)

    input_file = CONSTANTS.INPUT_DATA_DIR / project_name / "snapshot_all_regions.csv"

    region_patterns = convert_to_list(region_patterns)
    model_patterns = convert_to_list(model_patterns)
    target_patterns = convert_to_list(target_patterns)

    RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )
    RESULTS_DATA_DIR.mkdir(exist_ok=True)

    CONV_DATA_DIR = (
        CONSTANTS.CURR_RES_DIR(str(Path(os.path.abspath("")) / Path(__file__).name))
        / project_name
        / "conv"
    )

    IAMC_DATA_DIR = (
        CONSTANTS.CURR_RES_DIR(str(Path(os.path.abspath("")) / Path(__file__).name))
        / project_name
        / "IAMC_DATA"
    )
    (RESULTS_DATA_DIR / project_name).mkdir(exist_ok=True)
    CONV_DATA_DIR.mkdir(exist_ok=True)
    IAMC_DATA_DIR.mkdir(exist_ok=True)

    iam_results_file = InputFile(
        CONSTANTS.INPUT_DATA_DIR / project_name / "snapshot_all_regions.csv"
    )

    (CONSTANTS.CURR_RES_DIR(__file__) / project_name).mkdir(
        exist_ok=True
    )  ## Created a folder '{project_name}' inside step 1 folder

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

    df_iam = fun_read_df_iam_iamc(input_file)
    df_iam_all_models = fun_read_df_iam_all(
        file=InputFile(input_file), add_model_name_to_region=1 - model_in_region_name
    )
    df_iea_all, df_iea_melt = fun_load_iea()[1:]

    ec_list = [
        "Liquids",
        "Solids",
        "Electricity",
        "Gases",
        "Hydrogen",
        "Heat",
    ]


    if split_res_and_comm:
        sectors_list = ["Industry", "Transportation", "Residential", "Commercial"]
    else:
        sectors_list = ["Industry", "Transportation", "Residential and Commercial"]


    var_dict_demand, new_var_dict = fun_make_var_dict(
        sectors_list, ec_list, demand_dict=True
    )
    # Update new_var_dict with hydrogen and Heat (e.g. we calculate "Final Energy|Hydrogen" as rge sum of sub-sectors )
    new_var_dict.update(
        {
            f"Final Energy|{ec}": [
                f"Final Energy|sector|{ec}".replace("sector", x) for x in sectors_list
            ]
            for ec in ["Hydrogen", "Heat"]
        }
    )
    var_dict_supply = fun_make_var_dict(ec_list, sectors_list, demand_dict=False)
    conv_settings = fun_conv_settings(run_sensitivity)
    df_iea=fun_read_iea_data_from_iamc_format(2020,[])

    for model in selection_dict.keys():  ## model loop
        for region in selection_dict[model]["regions"]:
            targets = selection_dict[model]["targets"]
            path = Path(RESULTS_DATA_DIR / f"{region.replace('|','_')}r_{csv_in}.csv")
            if path.is_file():
                df_iamr = df_iam.xs(f"{region}r", level="REGION", drop_level=False)
                df_iamr = df_iamr.xs("EJ/yr", level="UNIT", drop_level=False)
                # reading previosly downscaled step 1 results
                df_all = pd.read_csv(path, index_col=["TIME", "ISO", "TARGET", "SECTOR"])
                el_sectors = fun_str_sector_list(df_all, df_iam, sector="Electricity")
                sectors_list = [x.replace("Final Energy|", "") for x in el_sectors]
                sectors_list = [x.replace("|Electricity", "") for x in sectors_list]
                (
                    var_dict_demand,
                    new_var_dict,
                    var_dict_supply,
                ) = fun_make_new_demand_supply_dict(sectors_list, ec_list)

                # Slice Selected targets
                df_all = df_all[df_all.index.get_level_values("TARGET").isin(targets)]
                
                func_type_list =list(df_all.FUNC.unique())

                for func_form in func_type_list:  # func_type loop
                
                    df = df_all[df_all["FUNC"] == func_form]
                    if var_dict_supply and var_dict_demand:
                        max_iter_demand_supply_interactions = max_iterations
                    else:  ## if one of the two dictionary is empty we don't need to iterate across the demand / supply side variables
                        max_iter_demand_supply_interactions = [0]

                    reg = (
                        f"{region}r"
                        if model_in_region_name
                        else region.replace(f"{model}|", "") + "r"
                    )
                    # Downscale heat
                    try:
                        df = fun_heat_downs(
                            df,
                            df_iam,
                            model,
                            reg,
                            targets,
                            df_iea_melt,
                            iea_flow_dict,
                            func_form,
                        )
                    except Exception as e:
                        print(e)
                    # Downscale hydrogen
                    available_iam_vars = set(
                        df_iam_all_models.reset_index().VARIABLE.unique()
                    )

                    for hydrogen in hydrogen_list:
                        required_hydrogen_vars = {
                            f"{hydrogen}|Electricity",
                            f"{hydrogen}|Hydrogen",
                        }
                        if (
                            available_iam_vars.intersection(required_hydrogen_vars)
                        ) == required_hydrogen_vars:
                            df = fun_hydrogen_downs(
                                df, df_iam_all_models, model, reg, targets, var_iam=hydrogen
                            )

                    # # Harmonize variables to match historical data until 2020
                    # for var in ['Final Energy',
                    #             'Final Energy|Electricity','Final Energy|Solids','Final Energy|Liquids','Final Energy|Gases'
                    #             ]:
                    #     f=fun_harmonize_hist_data_by_preserving_sum_across_countries_step1b
                    #     df=f(df, model, region, targets, df_iam,df_iea, ref_target, var,['ENSHORT_REF','ENLONG_RATIO'])

                    # Harmonize variables to match historical data until 2020. We do this only for ENSHORT_REF
                    for var in ['Final Energy',
                                #'Final Energy|Electricity','Final Energy|Solids','Final Energy|Liquids','Final Energy|Gases'
                                ]:
                        f=fun_harmonize_hist_data_by_preserving_sum_across_countries_step1b
                        df=f(df, df_iea, var)

                    # Add commit hash to file
                    df["COMMIT_HASH"] = get_git_revision_short_hash()

                    # Adding functional form for all variables (inc. newly created Final Energy|Heat)
                    df.loc[:, "FUNC"] = func_form

                    # PLACEHOLDER/block FOR enhanced step1 sensitivity
                    # Calculates smooth enlong (2010-tc interpolation)
                    res = {"wo_smooth_enlong": df}
                    # if extended sensitivity
                    u, o = "use_linear_method", "over"

                    f = fun_smooth_enlong
                    il = [{o: j, "tc_list": tcl, u: k} for j in ol for k in [False, True]]
                    res = {}
                    for tg in targets:
                        dft = df.xs(tg, level="TARGET", drop_level=False)
                        res[tg] = {
                            f"{i[o]}_{i[u]}": f(**dict(i, **{"df": dft})) for i in il
                        }
                    print("Running `fun_smooth_enlong`...")
                    df = df.assign(METHOD="wo_smooth_enlong")
                    df2 = pd.concat(
                        [v.assign(METHOD=k) for re in res.values() for k, v in re.items()]
                    )
                    df = pd.concat([df, df2])
                    
                    # NOTE: to visualy check if results works fine see code here: https://github.com/iiasa/downscaler_repo/commit/989ee0b33bbc2dad5eb022e90f6016237b5421c8

                    max_iterations = 2
                    cols = ["ENSHORT_REF", "ENLONG_RATIO"]
                    cols = cols + [x for x in df.columns if "then" in x and "EI_" not in x]

                    # Make sure we get data >0
                    df.loc[:, cols] = df.loc[:, cols].clip(0)
                    df = df.set_index("METHOD", append=True)


                    print("Running `run_sector_harmo_enhanced`...")
                    for _ in range(max_iterations):
                        for x in cols:
                            #  the order of dictionaries matters!!
                            for d in [var_dict_demand, var_dict_supply]:
                                try:
                                    df = run_sector_harmo_enhanced(
                                        df,
                                        d,
                                        x,
                                        df_iamr,
                                    )
                                except:
                                    raise ValueError(
                                        "Harmonization using `run_sector_harmo_enhanced` did not work"
                                    )
                                    
                    # THE BELOW IS JUST A BLUEPRING AND IT DOES NOT WORK YET.
                    # GOAL: TRY RESOLVE INCONSISTENCIES, by adding 'other' sector or energy carrier
                    # # TODO  please make sure you also recalculate the sum across sectors in step5.
                    # TODO Optionally add `Final Energy|Other Sector(downscaling)|{by all energy carriers}` in df_iam
                    # res_df={}
                    if resolve_inconsistencies:
                        res_df={}
                        for col in cols:
                            # We just check inconsistencies for one method
                            k=df.reset_index().METHOD.unique()[0]
                            if col not in ['ENSHORT_REF', 'ENLONG_RATIO']:
                                k=[x for x in df.reset_index().METHOD.unique() if x not in 'wo_smooth_enlong'][0]
                            res=fun_check_inconsistencies(fun_from_step1b_to_iamc(df, col=col, method=k), var_dict_supply)
                            res.update(fun_check_inconsistencies(fun_from_step1b_to_iamc(df, col=col, method=k), var_dict_demand))
                            if len(res):
                                res_df[col] = fun_resolve_inconsistencies_step1b( sectors_list, ec_list, df[cols], col,df_iam,more_sectors=True, more_ec=True)
                            else:
                                res_df[col] = df[[col]]
                        df=pd.concat(list(res_df.values())+[df.drop(cols, axis=1)], axis=1)
                            
                    # NOTE:
                    # The Dataframe will now contain new data (Final Energy|Industry / Transportation / Residential and commercial) that  usually are created
                    # In step 5. Please remove this new data from Step5 if you run this code.
                    variables=df.reset_index().SECTOR.unique()
                    vars_to_be_harmo=['ENSHORT_REF',
                                    'ENLONG_RATIO',
                                    "ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO", 
                                    "ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO"]
                    for var in variables:
                        f=fun_fix_short_term_data_to_ref_scen
                        df=f(df, region, targets, df_iam,ref_target, var, vars_to_be_harmo)
                    
                    # Save df results using standard (step1) data format (contains ENSHORT_REF / ENLONG_RATIO) for all functional forms
                    filename=f"{region.replace('|','_')}r_{csv_in}_{csv_out}.csv"
                    append='a' if func_form != func_type_list[0] else 'w'
                    header=True if func_form == func_type_list[0] else False
                    # i=(RESULTS_DATA_DIR/filename,"a", False) if func_form == func_type_list[0] else True (RESULTS_DATA_DIR/filename)  
                    df.to_csv(RESULTS_DATA_DIR/filename, mode=append, header=header)

                    ## Blended results
                    fun_save_blended_results(
                        csv_in,
                        csv_out,
                        conv_dict,
                        default_blend,
                        run_sensitivity,
                        IAMC_DATA_DIR,
                        conv_settings,
                        model,
                        region,
                        targets,
                        df,
                        append=append,
                        _func=func_form
                    )

            else:
                print(
                    f"We could not find step1 data for {model} {region}. We skip this model / region"
                )


# def fun_harmonize_hist_data_by_preserving_sum_across_countries_step1b(df:pd.DataFrame, model:str, region:str, targets:list, df_iam:pd.DataFrame,df_iea:pd.DataFrame, ref_target:str, var:str, vars_to_be_harmo:list=['ENSHORT_REF','ENLONG_RATIO'], hist_time_range:range=range(2010,2025,5))->pd.DataFrame:
#     """
#     Harmonize historical data while preserving the sum across countries, focusing on a specified variable.

#     This function ensures the sum of the specified variable is preserved across countries by harmonizing historical data. 
#     It performs the necessary adjustments before introducing different methods into the DataFrame.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         The original DataFrame containing energy data in `step1b` format.
#     model : str
#         The model name to be used for renaming in the intermediate IAMC format (e.g. message).
#     region : str
#         The region for which the data is being harmonized.
#     targets : list
#         A list of scenarios for which the harmonization is performed.
#     df_iam : pd.DataFrame
#         DataFrame containing IAMs (Integrated Assessment Model) results.
#     df_iea : pd.DataFrame
#         DataFrame containing IEA (International Energy Agency) historical data.
#     ref_target : str
#         The reference scenario used to maintain consistent short-term data (2010-2020) across scenarios 
#         (when IAM results do not vary across those scenarios).
#     var : str
#         The variable (e.g., 'Final Energy') for which the data is being harmonized.
#     vars_to_be_harmo: list, optional
#         Type List of variables to be harmonized, by default ['ENSHORT_REF','ENLONG_RATIO']
#     hist_time_range: range, optional
#         Historical time range, by default range(2010,2025,5)

#     Returns
#     -------
#     pd.DataFrame
#         A harmonized DataFrame (in step1b format) with results adjusted to preserve 
#         the sum across countries for the specified variable.

#     Raises
#     ------
#     ValueError
#         If the 'METHOD' column or index level is present in the input DataFrame `df`.

#     Notes
#     -----
#     - The function assumes the absence of the 'METHOD' column or index level in the input DataFrame `df`.
#     - Step 1) It converts the data to the IAMC format and performs harmonization on the specified variable.
#     - Step 2) The results are adjusted to ensure consistency across scenarios up to the year 2020.
#     """

#     if 'METHOD' in df.columns or 'METHOD' in df.index.names:
#         txt='`METHOD` should not be present in df.columns nor in the df.index.names'
#         txt2='This function should be used before you introduce different `METHOD`s in your dataframe' 
#         raise ValueError(f'{txt}.{txt2} ')
#     df=df.copy(deep=True)      
#     df_iamc=fun_from_step1b_to_iamc(df,col='ENSHORT_REF').rename({'model':model})
    
#     df_iamc=fun_rename_index_name(df_iamc, {'ISO':'REGION'})
#     df_iamc=fun_harmonize_hist_data_by_preserving_sum_across_countries(df_iamc, df_iea, var)
    
#     # Step1:
#     # Dropping METHOD as we do not have yet in `df`
#     fen_harmo=fun_from_iamc_to_step1b(df_iamc).rename({'RENAME_ME':'ENSHORT_REF'}, axis=1).droplevel('METHOD')
#     fen=df.xs(var, level='SECTOR', drop_level=False)
#     # Update Final Energy results
#     fen = fen_harmo.loc[fen.index,'ENSHORT_REF']
                
#     # Step 2: Keep `ref_target` results until 2020, if iam results are not changing across scenarios
#     for t in hist_time_range:
#         if len(np.round(df_iam.xs([var,f'{region}r'], level=['VARIABLE','REGION']),2)[t].unique())==1:
#             fen_fix=pd.concat([fen[fen.index.get_level_values('TARGET')==ref_target].rename({ref_target:x}) for x in targets])
#             fen_fix=fen_fix.xs(t, drop_level=False)
#             fen.loc[fen_fix.index]=fen_fix # use same results as in ref scenario
#     fen=fen[fen.index.get_level_values('TIME').isin(list(hist_time_range))]
#     for v in vars_to_be_harmo:
#         df.loc[fen.index,v]=fen
#     return df



def fun_harmonize_hist_data_by_preserving_sum_across_countries_step1b(df:pd.DataFrame, df_iea:pd.DataFrame, var:str, vars_to_be_harmo:list=['ENSHORT_REF',"ENLONG_RATIO"], hist_time_range:range=range(2010,2025,5))->pd.DataFrame:
    """
    Harmonize historical data while preserving the sum across countries, focusing on a specified variable.

    This function ensures the sum of the specified variable is preserved across countries by harmonizing historical data. 
    It performs the necessary adjustments before introducing different methods into the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame containing energy data in `step1b` format.
    df_iea : pd.DataFrame
        DataFrame containing IEA (International Energy Agency) historical data.
    var : str
        The variable (e.g., 'Final Energy') for which the data is being harmonized.
    vars_to_be_harmo: list, optional
        Type List of variables to be harmonized, by default ['ENSHORT_REF','ENLONG_RATIO']
    hist_time_range: range, optional
        Historical time range, by default range(2010,2025,5)
    
    Returns
    -------
    pd.DataFrame
        A harmonized DataFrame (in step1b format) with results adjusted to preserve 
        the sum across countries for the specified variable.

    Raises
    ------
    ValueError
        If the 'METHOD' column or index level is present in the input DataFrame `df`.

    Notes
    -----
    - The function assumes the absence of the 'METHOD' column or index level in the input DataFrame `df`.
    - Step 1) It converts the data to the IAMC format and performs harmonization on the specified variable.
    """

    if 'METHOD' in df.columns or 'METHOD' in df.index.names:
        txt='`METHOD` should not be present in df.columns nor in the df.index.names'
        txt2='This function should be used before you introduce different `METHOD`s in your dataframe' 
        raise ValueError(f'{txt}.{txt2} ')
    df=df.copy(deep=True)      
    df_iamc=fun_from_step1b_to_iamc(df,col='ENSHORT_REF')# .rename({'model':model})
    
    df_iamc=fun_rename_index_name(df_iamc, {'ISO':'REGION'})
    df_iamc=fun_harmonize_hist_data_by_preserving_sum_across_countries(df_iamc, df_iea, var)
    
    # Step1:
    # Dropping METHOD as we do not have yet in `df`
    fen_harmo=fun_from_iamc_to_step1b(df_iamc).rename({'RENAME_ME':'ENSHORT_REF'}, axis=1).droplevel('METHOD')
    fen=df.xs(var, level='SECTOR', drop_level=False)
    # Update Final Energy results
    fen = fen_harmo.loc[fen.index,'ENSHORT_REF']
                
    # # Step 2: Keep `ref_target` results until 2020, if iam results are not changing across scenarios
    # for t in hist_time_range:
    #     if len(np.round(df_iam.xs([var,f'{region}r'], level=['VARIABLE','REGION']),2)[t].unique())==1:
    #         fen_fix=pd.concat([fen[fen.index.get_level_values('TARGET')==ref_target].rename({ref_target:x}) for x in targets])
    #         fen_fix=fen_fix.xs(t, drop_level=False)
    #         fen.loc[fen_fix.index]=fen_fix # use same results as in ref scenario
    fen=fen[fen.index.get_level_values('TIME').isin(list(hist_time_range))]
    for v in vars_to_be_harmo:
        df.loc[fen.index,v]=fen
    return df


def fun_fix_short_term_data_to_ref_scen(df:pd.DataFrame, region:str, targets:list, df_iam:pd.DataFrame,ref_target:str, var:str, vars_to_be_harmo:list=['ENSHORT_REF','ENLONG_RATIO'], hist_time_range:range=range(2010,2025,5))->pd.DataFrame:
    """
    Fix short-term data to match a reference scenario if IAM results are consistent across scenarios.

    This function updates the short-term data (for a `hist_time_range`) in the given DataFrame to match the results of a reference 
    target `ref_target` scenario for a specified variable `var` and `region`. It ensures that the short-term data is consistent 
    across all target scenarios if the IAM results do not vary between scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be fixed.
    region : str
        The region for which the data is to be fixed.
    targets : list
        The list of target scenarios to be considered.
    df_iam : pd.DataFrame
        The DataFrame containing IAM results.
    ref_target : str
        The reference target scenario used to maintain consistent short-term data across scenarios when IAM 
        results do not vary across those scenarios.
    var : str
        The variable for which the data is to be fixed (e.g., 'Final Energy').
    vars_to_be_harmo : list, optional
        The list of variables in the DataFrame that need to be harmonized. Default is ['ENSHORT_REF', 'ENLONG_RATIO'].
    hist_time_range : range, optional
        The range of historical time steps for which the data is to be fixed. Default is range(2010, 2025, 5).

    Returns
    -------
    pd.DataFrame
        The DataFrame with short-term data fixed to match the reference scenario.
    """

    df=df.copy(deep=True)      
    # df_iamc=fun_from_step1b_to_iamc(df,col='ENSHORT_REF').rename({'model':model})
    # df_iamc=fun_rename_index_name(df_iamc, {'ISO':'REGION'})
    # df_iamc=fun_harmonize_hist_data_by_preserving_sum_across_countries(df_iamc, df_iea, var)
    
    # Step1:
    # Dropping METHOD as we do not have yet in `df`
    # fen_harmo=fun_from_iamc_to_step1b(df_iamc).rename({'RENAME_ME':'ENSHORT_REF'}, axis=1).droplevel('METHOD')
    fen=df.xs(var, level='SECTOR', drop_level=False)
    # Update Final Energy results
    # fen = fen_harmo.loc[fen.index,'ENSHORT_REF']
                
    # Step 2: Keep `ref_target` results until 2020, if iam results are not changing across scenarios
    if ref_target in df.index.get_level_values('TARGET').unique():
        for t in hist_time_range:
            if len(np.round(df_iam.xs([var,f'{region}r'], level=['VARIABLE','REGION']),4)[t].unique())==1:
                fen_fix=pd.concat([fen[fen.index.get_level_values('TARGET')==ref_target].rename({ref_target:x}) for x in targets])
                fen_fix=fen_fix.xs(t, drop_level=False)
                fen.loc[fen_fix.index]=fen_fix # use same results as in ref scenario
        fen=fen[fen.index.get_level_values('TIME').isin(list(hist_time_range))]
    for v in vars_to_be_harmo:
        df.loc[fen.index,v]=fen
    return df


def fun_save_blended_results(
    csv_in,
    csv_out,
    conv_dict,
    default_blend,
    run_sensitivity,
    IAMC_DATA_DIR,
    conv_settings,
    model,
    region,
    targets,
    df,
    append=False,
    _func=None,
):
    if _func is None:
        _func= list(df.FUNC.dropna().unique())
    _func = [_func] if isinstance(_func, str) else _func
    if len(_func)>1:
        raise ValueError(f'df should contain only one functional form. Found {_func}')
    df=df[df.FUNC.isin(_func)]
    if "METHOD" in df.reset_index().columns:
        df_blended = fun_blending_with_sensitivity(df)
        df_blended["MODEL"] = model
        df_blended = df_blended.set_index(["MODEL", df_blended.index])
        add_var = {}
        # Add GDP|PPP and Population in df_res
        for x in ["GDP|PPP", "Population"]:
            pick_onevar = df.reset_index().SECTOR.unique()[0]
            temp = (
                df[x]
                .unstack("TIME")
                .xs(pick_onevar, level="SECTOR", drop_level=False)
                .rename({pick_onevar: x})
            )
            temp = fun_rename_index_name(temp, {"SECTOR": "VARIABLE"}).xs(
                "wo_smooth_enlong", level="METHOD", drop_level=False
            )
            temp = temp.rename({"wo_smooth_enlong": "none"}).assign(CONVERGENCE="none")
            temp["MODEL"] = model
            # tempconv=pd.concat([temp.assign(CONVERGENCE='none') for tc in df_blended.reset_index().CONVERGENCE.unique()])
            add_var[x] = temp.reset_index().set_index(df_blended.index.names)
        df_blended = pd.concat([df_blended, pd.concat(add_var.values())])

        # CHECK for negative energy variables
        df_check = fun_rename_index_name(df_blended, {"ISO": "REGION"})
        f = fun_check_negative_energy_variables_by_model
        # We temporarily use assign UNIT=`EJ/yr` as workaround to make work the function below
        f(df_check.assign(UNIT="EJ/yr").set_index("UNIT", append=True))

        # Temporarily Adding units
        variables=df_blended.reset_index().VARIABLE.unique()
        units={x:'EJ/yr' for x in variables}
        df_blended=fun_add_units(df_blended, None, units)
        # Calculate Final Energy|Residential and Commercial by energy carrier
        for sect in ['','|Electricity', '|Gases','|Heat','|Hydrogen','|Liquids','|Solids']:
            mydict={f'Final Energy|Residential and Commercial{sect}': 
                            [f'Final Energy|Commercial{sect}', f'Final Energy|Residential{sect}']}
            df_blended = fun_create_missing_sectors_as_sum(df_blended, mydict, unit='EJ/yr')

        # Add functioncal form information
        df_blended['FUNC']=_func[0]
        df_blended = df_blended.set_index("FUNC", append=True)

        # Drop units
        df_blended=df_blended.droplevel('UNIT')
        # NOTE: UNIT columns is missing in `df_blended`.
        # `df_blended` contains Population and GDP|PPP. If you want to add a UNIT column please use function `fun_add_units`
        fname = f"{region.replace('|','_')}r_{csv_in}_{csv_out}_IAMC_sensitivity_analysis.csv"
        df_blended.to_csv(IAMC_DATA_DIR / fname, mode=append)
        fun_xs(df_blended, {'ISO':iea_countries}).to_csv(IAMC_DATA_DIR / fname.replace('.csv','iea_countries.csv'), mode=append)
    else:
        # Older IAMC format (which depended on con_dict). We keep this code so that it works if we have data without extended sensitivity
        df_blended = fun_blending(df, run_sensitivity)
        
        # Add functioncal form information
        df_blended['FUNC']=_func[0]
        df_blended = df_blended.set_index("FUNC", append=True)

        ## Save results using IAMC data format for a chosen_blend
        if conv_dict and len(
            set(targets).intersection(set(fun_flatten_list(conv_dict.values())))
        ):
            final_energy_dict = conv_settings["Final"]
            idx_app = ["SECTOR", "TARGET"]
            if "METHOD" in df_blended.reset_index().columns:
                idx_app = idx_app + ["METHOD"]
            df_blended.set_index(idx_app, append=True, inplace=True)
            for i, j in conv_dict.items():
                df_sel = df_blended[f"{final_energy_dict[i.upper()]}_BLEND"]
                df_sel = df_sel.unstack("TIME")
                df_sel = df_sel[df_sel.index.get_level_values("TARGET").isin(j)]
                fname = (
                    f"{region.replace('|','_')}r_{csv_in}_{csv_out}_IAMC_blend_{i}.csv"
                )
                if len(df_sel) > 0:
                    df_sel.to_csv(IAMC_DATA_DIR / fname, mode=append)

        else:  # If convt_ditc is None we use a 'default_blend' for all scenarios        
            # Add functioncal form information
            df_blended['FUNC']=_func[0]
            df_blended = df_blended.set_index("FUNC", append=True)

            setindex(df_blended, ["TIME", "ISO", "SECTOR", "TARGET", "FUNC"]).unstack("TIME")[
                default_blend
            ].to_csv(
                IAMC_DATA_DIR
                / f"{region.replace('|','_')}r_{csv_in}_{csv_out}_IAMC_{default_blend}.csv", mode=append
            )




def fun_make_new_demand_supply_dict(sectors_list, ec_list):
    sectors_list = [x.replace("||", "|") for x in sectors_list]
    var_dict_demand, new_var_dict = fun_make_var_dict(
        sectors_list, ec_list, demand_dict=True
    )
    # Update new_var_dict with hydrogen and Heat (e.g. we calculate "Final Energy|Hydrogen" as rge sum of sub-sectors )
    new_var_dict.update(
        {
            f"Final Energy|{ec}": [
                f"Final Energy|sector|{ec}".replace("sector", x) for x in sectors_list
            ]
            for ec in ["Hydrogen", "Heat"]
        }
    )
    var_dict_supply = fun_make_var_dict(ec_list, sectors_list, demand_dict=False)
    return var_dict_demand, new_var_dict, var_dict_supply


conv_dict = {
    "slow": slow_list,
    "med": med_list,
    "fast": fast_list,
}


if __name__ == "__main__":
    main(
        "CDLINKS_JARMO",
        "CD_LINKS_JARMO",
        region_patterns="*",
        model_patterns="*MESSAGE*",
        # target_patterns="CD_Links_SSP1_v2NPi2020_1000-con-prim-dir-ncr",  # QAT
        target_patterns=["*"],
        csv_out="harmo",
        max_iter=5,
        conv_dict=conv_dict,
        default_blend="2150_BLEND",  # If con_dict is an empty dictionary, we use this convergence for all scenarios
    )
