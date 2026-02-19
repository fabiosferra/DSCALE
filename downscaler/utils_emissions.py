import copy
import csv
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from downscaler.utils_dictionary import sum_two_dictionaries, fun_sort_dict, sum_multiple_dictionaries
from downscaler.utils_list import (
    fun_sort_list_order_based_on_element_name,
    fun_fuzzy_match,
    fun_wildcard,
)
from downscaler.utils_pandas import (
    fun_xs,
    fun_index_names,
    fun_add_multiply_dfmultindex_by_dfsingleindex,
    fun_check_iamc_index,
    fun_rename_index_name,
    fun_drop_duplicates,
    fun_get_variable_unit_dictionary,
    fun_check_if_all_characters_are_numbers,
    fun_read_csv_or_excel,
    fun_add_units,
    fun_read_csv,
    summarize_dataframe,
)

from downscaler import CONSTANTS, IFT
from downscaler.fixtures import iea_countries, all_countries


# ============================================================
# Functions needed by Step_5f_emi_clock_claude_edits.py
# Cleaned from original utils_emissions.py (581 -> 44 functions)
# ============================================================


def setindex(_df, _index):
    """
    fun_df_setindex

    This function sets an index (_index) in you dataframe (_df) .
    If _index is not in the df it will not keep the df unchanged.
    If _index=False it will reset the index

    _df= your dataframe
    _index= Your index. If =False will  reset_index

    """
    try:
        _df.reset_index(inplace=True)
    except:
        dont = 1

    if _index != False:
        _df.set_index(_index, inplace=True)

    try:
        # where 1 is column,  0 is row
        _df2 = _df.drop(labels="level_0", axis=1, inplace=True)
    except:
        _df2 = _df

    try:
        # where 1 is column,  0 is row
        _df3 = _df2.drop(labels="index", axis=1, inplace=True)
    except:
        _df3 = _df2

    return _df3


def unique(list1):
    """
    fun lst_unique
    This function returns unique elements in a list, by preserving the initial list order
    (something that set(list) will not do)
    """
    # intilize a null list
    unique_list = []
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def fun_read_df_countries(file: IFT) -> pd.DataFrame:
    """Reading Country-region mapping. It returns a Dataframe (_df_countries)"""
    _df_countries = pd.read_csv(file, sep=",", encoding="latin-1")  
    setindex(_df_countries, "COUNTRY_NAME")

    return _df_countries


def load_model_mapping(_model_name, _df_countries, file: IFT):
    """
    Load regional-country mapping (the same used in Pyam)
    It returns an updated df_countries and a list of regions
    """

    # df_pyam_mapping = pd.read_csv(
    #     file, index_col=["ISO"], sep=",", encoding="utf-8"
    # )  # encoding='latin-1')

    df_pyam_mapping = pd.read_csv(
        # "default_mapping.csv", index_col=["ISO"], sep=",", encoding="utf-8"
        # downscaler.CONSTANTS.INPUT_DATA_DIR / "default_mapping.csv",
        # index_col=["ISO"],
        # sep=",",
        # encoding="latin"
        file,
        index_col=["ISO"],
        sep=",",
        encoding="latin",
    )

    # Mapping Model name dictionary (NGFS mapping)
    ditc_model_name = {
        #  "MESSAGEix-GLOBIOM 1.0": "MESSAGE-GLOBIOM",
        "REMIND-MAgPIE 1.7-3.0": "REMIND-MAGPIE",
        "GCAM 4.2": "GCAM4",
        "GCAM 4.4": "GCAM4",
        "GCAM 5.2": "GCAM4",
    }
    if _model_name in ditc_model_name:
        txt = (
            f"We are searching for {ditc_model_name[_model_name]} instead of {_model_name}. \n"
            " Please type `y` if you want to continue with the current behavior (suggested for NGFS project). "
            f"Otherwise please type `n` if you want to use {_model_name} (suggested for other projects)"
        )
        action = input(txt)
        if action.lower() in ["yes", "y"]:
            _model = ditc_model_name[_model_name]
        else:
            _model = _model_name
    else:
        _model = _model_name

    # Creating Region list, Excluding nan REGION
    try:
        region_list = df_pyam_mapping[_model + ".REGION"].unique().tolist()
    except:
        raise ValueError(
            f"Error when reading regional mapping for {_model_name}. Please make sure the project/file name is correct: {file}"
        )
    ## Purpose of the below is to remove any region= na. THIS IS DANGEROUS as not always the last region is na
    # blacklist = [region_list[-1]] ## THIS IS DANGEROUS
    # region_list = [str(e) + "r" for e in region_list if e not in blacklist]

    ## 2021_11_05 Purpose of the below is to remove any region= na  (if region==na it means that type(region)== float instead of string):
    region_list = [str(i) + "r" for i in region_list if type(i) != float]

    # Copying variables to df_countries
    setindex(_df_countries, "ISO")
    _df_countries.loc[:, _model + ".REGION"] = df_pyam_mapping.loc[
        :, _model + ".REGION"
    ]
    _df_countries.loc[:, "R5_region"] = df_pyam_mapping.loc[:, "R5_region"]
    #     _df_countries.loc[:,'REGION']= _df_countries.loc[:, _model+'.REGION'] ## NATIVE REGIONS same regions as pyam
    _df_countries.loc[:, "REGION"] = (
        _df_countries.loc[:, _model + ".REGION"] + "r"
    )  # NATIVE REGIONS same regions as pyam

    _df_countries.loc[:, "IPCC"] = _df_countries.loc[
        :, "R5_region"
    ]  # R5 REGIONS same regions as pyam
    setindex(_df_countries, "COUNTRY_NAME")

    return _df_countries, region_list


def fun_country_map(model, country_mapping_file, pyam_mapping_file):
    df_countries = fun_read_df_countries(country_mapping_file)
    df_countries, regions = load_model_mapping(model, df_countries, pyam_mapping_file)

    return df_countries[["ISO", "REGION"]]


def fun_country2region(model, c, country_mapping_file, pyam_mapping_file):
    country_dict = fun_country_map(model, country_mapping_file, pyam_mapping_file)
    txt1 = f"Please consider adding it to the `country_dict` in the function `fun_get_iam_regions_associated_with_countrylist`,"
    txt2 = f"as we do for the EU27 and EU28"
    if c not in country_dict.ISO.unique():
        raise ValueError(
            f"{c} country is not available in the default mapping file. {txt1}{txt2}"
        )
    region = country_dict[country_dict.ISO == c]["REGION"][0]
    return region


# NOTE: ssp_model and ssp_scenario should be turned into fixtures


def fun_eu28():
    """Returns list of EU28 ISO codes"""
    return [
        "BGR",
        "CYP",
        "DNK",
        "IRL",
        "EST",
        "AUT",
        "CZE",
        "FIN",
        "FRA",
        "DEU",
        "GRC",
        "HRV",
        "HUN",
        "ITA",
        "LVA",
        "LTU",
        "SVK",
        "MLT",
        "BEL",
        "LUX",
        "NLD",
        "POL",
        "PRT",
        "ROU",
        "SVN",
        "ESP",
        "SWE",
        "GBR",
    ]


def fun_eu27():
    """Returns list of EU28 ISO codes"""
    eu27 = fun_eu28()
    eu27.remove("GBR")
    return eu27


def fun_flatten_list(l, _unique=False):
    mylist = [item for sublist in l for item in sublist]
    return unique([item for sublist in l for item in sublist]) if _unique else mylist

# Calculate frequency (occurrency) for each tuple (method 1)
# frequency={all_seeds.count(x):x for x in all_seeds}
# # Selected seeds with max occurrency
# sel_seeds= frequency[max(frequency)]

# Calculate frequency (occurrency) for each tuple (method 2 - better)
# frequency={x:all_seeds.count(x) for x in all_seeds}
# # Selected seeds with max occurrency
# sel_seeds = {i for i in frequency if frequency[i]==max(frequency.values())}

# Sort seeds tuple by max occurrence (method 3 - more flexible)


def fun_invert_dictionary(mydict):
    new_dic = {}
    for k, v in mydict.items():
        for x in v:
            new_dic.setdefault(x, []).append(k)
    return new_dic


def fun_read_df_iam_from_multiple_df(model: str, datadir: Path) -> pd.DataFrame:
    """Find df_iam (dataframe with IAM results) for a given `model`, in a `datadir`
       folder (with multiple dataframes from different models).

    Parameters
    ----------
    model : str
        Model that you are looking for (e.g `WITCH 5.0`)
    datadir : Path
        Folder with IAMs reuluts e.g. `input_data/EU_climate_advisory_board/multiple_df`

    Returns
    -------
    pd.DataFrame
        Dataframe with selected IAM moldel results (at the regional level)

    Raises
    ------
    ValueError
        _description_
    """
    flag = 0
    file_list = fun_sort_list_order_based_on_element_name(
        list(datadir.iterdir()), model
    )
    while flag < 1:
        for file in file_list:
            df_iam = pd.read_csv(file)
            df_iam.columns = [x.upper() for x in df_iam.columns]
            model_read = (
                df_iam.reset_index().MODEL.unique()[0].replace("_downscaled", "")
            )
            if model == model_read:
                flag = 1
                break
            if file == list(datadir.iterdir())[-1]:
                raise ValueError(f"Cannot find df_iam for {model} model")
    df_iam = df_iam.set_index(["MODEL", "REGION", "VARIABLE", "UNIT", "SCENARIO"])
    return df_iam


def fun_discount_rate(
    all_time_cols: Union[list, set, tuple],
    hist_time_cols: Union[list, set, tuple],
    tc: Union[None, int],  # time of convergenc
    start,  # value at the base year
    end,  # value at tc time period
):
    if tc is None:
        discount_rate = start
    else:
        discount_dict = {x: start for x in hist_time_cols}
        discount_dict[tc] = end
        discount_rate = pd.DataFrame([discount_dict])
        [
            discount_rate.insert(len(discount_rate.columns), x, np.nan)
            for x in all_time_cols
            if x not in discount_rate.columns
        ]
        discount_rate = discount_rate.sort_index(axis=1).interpolate(axis=1).loc[0]
    return discount_rate


def fun_get_iam_regions_associated_with_countrylist(
    project: str, countrylist: list, model: str
) -> dict:
    """Provides a dictionary with iam_regions associated with a list of countries `countrylist`

    Parameters
    ----------
    project : str
        Your project folder (where to look for the model mapping) e.g. 'NGFS_2023'
    countrylist : list
        List of countries for which you want to get a list of associated iams regions
    model : str
        Your chosen IAM e.g. `'REMIND-MAgPIE 3.1-4.6'`

    Returns
    -------
    dict
        Dictionary with: {country : iam_region}
        Example -> {'AUT': 'EU 28r'}
    """
    country_dict = {"EU27": fun_eu27(), "EU28": fun_eu28()}
    countrylist = countrylist or all_countries
    countrylist = set(fun_flatten_list([country_dict.get(c, [c]) for c in countrylist]))
    return {
        c: fun_country2region(
            model,
            c,
            CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv",
            CONSTANTS.INPUT_DATA_DIR / project / "default_mapping.csv",
        )
        for c in countrylist
    }


def fun_regional_country_mapping_as_dict(model: str, project: str, iea_countries_only:bool=False) -> dict:
    """Returns a dictionary with regional-country mapping for a given `model` from a given `project`

    Parameters
    ----------
    model : str
        your selected model
    project : str
        Project (where to find the `default_mapping.csv`)
    iea_countries_only : Optional[bool]
        Wheter we want to select only iea countries, defaults to False
    Returns
    -------
    dict
        Regional-country mapping
    """
    mapping=fun_country_map(
            model,
            CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv",
            CONSTANTS.INPUT_DATA_DIR / f"{project}/default_mapping.csv",
        )
    if iea_countries_only:
        mapping =mapping[mapping.ISO.isin(iea_countries)]
    mydict =mapping.dropna().reset_index().set_index(["ISO"])["REGION"].to_dict()
    return fun_invert_dictionary({k: [v] for k, v in mydict.items()})


def fun_find_nearest_values(mylist: list, target_value: float, n_max=2):
    res = {}
    for x in mylist:
        #  x/1e9 avoid repeating the same results for different x values
        res[(x - target_value) ** 2 + x / 1e9] = x
    r = list(res.keys())
    r.sort()

    res_sort = [res[r[i]] for i in range(n_max)]
    res_sort.sort()
    return res_sort


def fun_read_df_from_step1(file, countrylist, models, project):
    df = pd.DataFrame()
    missing_reg = []
    missing_targets = {}
    for model in models:
        df_iam = fun_read_df_iams(project, [model])
        iam_scenarios = df_iam.reset_index().SCENARIO.unique()
        step1_file = file  # fun_step1_file_name(file, project, model)
        regions_all = fun_regional_country_mapping_as_dict(model, project)
        regions_to_be_downs = {k: v for k, v in regions_all.items() if len(v) > 1}
        regions = regions_to_be_downs
        if countrylist is not None:
            regions = fun_get_iam_regions_associated_with_countrylist(
                project, countrylist, model
            )
            regions = fun_invert_dictionary({k: [v] for k, v in regions.items()})
            # Exclude Native countries (we do not have csv file for native countries)
            regions = {k: v for k, v in regions.items() if k in regions_to_be_downs}
        for region in regions:
            read_file = [x for x in step1_file if region in x]
            if len(read_file) == 0 and len(regions[region]) > 1:
                missing_reg = missing_reg + [region]
            elif len(read_file) > 1:
                txt = "We found multiple step1 files for the"
                raise ValueError(f" {txt} {region} region: {read_file}")
            elif len(read_file) == 0:
                txt = f"We cannot find {region} region for {model}. We only found: {step1_file} in the most recent files."
                action = input(
                    f"{txt}: {missing_targets}. \n Do you want to continue y/n?"
                )
                if action.lower() not in ["yes", "y"]:
                    raise ValueError("Aborted by the user")
                return pd.DataFrame()
            else:
                df_read = pd.read_csv(CONSTANTS.CURR_RES_DIR("step1") / read_file[0])
                df_read["MODEL"] = model
                av_scen = df_read.TARGET.unique()
                # Check available targets:
                if len(set(av_scen) ^ (set(iam_scenarios))):
                    missing_targets[region] = set(av_scen) ^ (set(iam_scenarios))
                df = pd.concat([df, df_read])
        if len(missing_reg):
            t0 = f"Unable to find these regions for {model}:"
            t1 = "NOTE: All of these regions comprise more than 1 country (they should be all downscaled)."
            t2 = f"However we only checked regions associated to this countrylist: {countrylist}."
            t3 = " If you want to check all regions please specify `countrylist=None`"
            final_t = f"{t0} {missing_reg}, in these step1 files: {step1_file}. {t1} "
            final_t = f"{final_t} {t2} {t3}" if countrylist else final_t
            raise ValueError(final_t)
        if len(missing_targets):
            txt = f"Some scenarios are missing for some regions in this file {file}"
            action = input(f"{txt}: {missing_targets}. \n Do you want to continue y/n?")
            if action.lower() not in ["yes", "y"]:
                raise ValueError("Aborted by the user")
            # elif action in ["no", "n"]:
            #     raise ValueError(f"{model} is not available in the default mapping")
            # else:
            #     df_mapping[f"{model}.REGION"] = df_mapping[action]

            # raise ValueError(f"{txt}: {missing_targets}")

    return df


def fun_get_models(project: str, sub_folder: str = "multiple_df", nrows=1) -> list:
    """
    Retrieve the list of unique models from CSV files within a specified project directory.

    This function reads the first `nrows` rows of each CSV file located in the `input_data/project/multiple_df` 
    directory. It extracts and compiles a list of unique models from the files.
    
    Parameters
    ----------
    project : str
        Project folder (e.g. NGFS_2023)
    sub_folder : str, optional
        Folder with regional IAMs data (saved as CSV files), by default "multiple_df"
    nrows : int, optional
        The number of rows to read from each CSV file, by default 1. Reading only the first few rows 
        can improve performance if the files are large.

    Returns
    -------
    List[str]
        A list of unique model names found in the specified CSV files.
    
    Notes
    -----
    - The function assumes that each CSV file contains a column with the name 'Model' (case insensitive).
    - It scans the specified subdirectory within the project's input data directory for CSV files 
      and compiles a list of unique model names across all files.
    - The CONSTANTS.INPUT_DATA_DIR is assumed to be a predefined constant pointing to the base input directory.

    Example
    -------
    ```python
    models = fun_get_models("NGFS_2023")
    print(models)
    ```
    """
    folder = CONSTANTS.INPUT_DATA_DIR
    l = [x for x in os.listdir(folder / project / sub_folder) if ".csv" in x]
    models=[]
    for x in l:
        if nrows is not None:
            mydf=pd.read_csv(folder / project / sub_folder / x, nrows=nrows)
        else:
            mydf=pd.read_csv(folder / project / sub_folder / x)
        sel_col=[i for i in mydf.columns if i.lower()=='model'][0]
        models+=list(mydf[sel_col].unique())
    return list(set(models))


def fun_get_scenarios(project: str, sub_folder: str = "multiple_df") -> list:
    """Returns the list of scenarios for a given `project`. It reads the CSV files
    contained in the `input_data/project/multiple_df` folder, and returns a list of scenarios.

    Parameters
    ----------
    project : str
        Project folder (e.g. NGFS_2022)
    sub_folder : str, optional
        Folder with regional IAMs data (saved as CSV files), by default "multiple_df"

    Returns
    -------
    list
        list of scenarios
    """
    folder = CONSTANTS.INPUT_DATA_DIR
    l = [x for x in os.listdir(folder / project / sub_folder) if ".csv" in x]
    return pd.concat(
        [pd.read_csv(folder / project / sub_folder / x) for x in l]
    ).SCENARIO.unique()


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


def run_sector_harmo_enhanced(
    df: pd.DataFrame,
    d: dict,
    x: str,
    df_iam: Optional[pd.DataFrame] = None,
    w:int=1, # 1 is our standard assumption (we do not sum sub-sector -> sum_anyway=False ). 0 means we sum anyway
    # d2:dict= None,
    verbose: bool = True
) -> pd.DataFrame:
    """Run sectorial harmonization in your `df` based on a dictionary (`d`), for a given
    variable (`x`), e.g "ENSHORT_REF". Harmoniziation with regional IAMs results will be
    skipped if `df_iam` is None.

    It harmonizes the sub-sectors with the steps below:
    - step0: Makes sure that the main sector `k` is present in the datarame. If not it creates it as the sum of sub-sectors
    - step1: Makes sure that the main sector `k` is consistent with regional IAM results (if `df_iam` is provided)
    - step2: Rescales the sub-sectors `v` proportionally
    - step3: Makes sure that each sub-sector is consistent with regional IAMs results (if `df_iam` is provided)

    NOTE: This function is similar to `run_sector_harmo`. Compared to `run_sector_harmo ` this one
    pereforms a better regional IAMs harmonization (for all variables that are also present in df_iam).
    For this reason it also take a bit longer compared to `run_sector_harmo`.
    Apart from that, the two functions are very similar in terms of performance (try to ensure that
    the sum of sub-sectors matches the main sector).

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe in Step1b format.
        (you can convert your dataframe in step1b format using `fun_from_step2_to_step1b_format` and
        `fun_step2_format_from_iamc`)
    d : dict
        Dictionary with the main sectors as keys and a list of subsectors as values.
        Example:  {Final Energy: ["Final Energy|Electricity", "Final Energy|Liquids"]}
        Dictionary can be created using the function `fun_sub_sectors_dict`
    x : str
        Variable that you want to harminize(e.g. `ENSHORT_REF` or `ENLONG_RATIO`)
    df_iam : pd.DataFrame, Optional
        Dataframe with regional IAMs results (to ensure consisency with regional IAMs results).
        If df_iam is not provided harminzation with IAMs results will be skipped, by default None
    d2: pd.DataFrame, Optional:
        Another dictionary that can be used to clip values, by default None. Example: 'Final Energy|Transportation|Liquids' 
        should be smaller than 'Final Energy|Liquids' (that can be found in dictionart `d`) and smaller than 
        'Final Energy|Transportation' (that is not present in `d` but can be found in dictionary `d2`)   


    Returns
    -------
    pd.DataFrame
        Updated dataframe
    """
    if df_iam is None:
        print(
            "Will skip harmonization with regional IAMs results, because you did not pass a `df_iam`"
        )
    df = df.copy(deep=True)
    dropcols=['COMMIT_HASH', 'COUNTRYLIST','FUNC']
    df_drop=df.iloc[:,df.columns.isin(dropcols)]
    for col in dropcols:
        if col in df.columns:
            df=df.drop(col, axis=1)
    for k, v in d.items():
        # step0 Create variable `k` as the sum of sub-sectors  (`v`)
        if k=='Final Energy':
            df = fun_sum_of_sub_sectors(df, x, k, v, sum_anyway=False)
        else:
            a=fun_sum_of_sub_sectors(df, x, k, v, sum_anyway=False)
            b=fun_sum_of_sub_sectors(df, x, k, v, sum_anyway=True)
            df = a*w+b*(1-w)
        # if k still not present, we skip the block below and return the dataframe
        if k in df[x].reset_index().SECTOR.unique():
            # step1 make sure that `k` (e.g. final energy) matches regional iam results
            df = (
                fun_harmonize_df_with_IAM(df, df_iam, x, k, verbose=verbose)
                if df_iam is not None
                else df
            )

            # step2 - Rescale the sub-sectors results proportionally, based on the share of  the sum of liquids,gases  divided by the total in each country
            if not len(fun_xs(df, {"SECTOR": v})[x].dropna(how="all")):
                # if the sub-sectors (v) are not present, we create them as the sum of sub-sub-sectors (vv)
                for vv in v:
                    if vv in d:
                        try:
                            df = fun_sum_of_sub_sectors(
                                df, x, vv, d[vv], sum_anyway=True
                            )
                        except:
                            print("fun_sum_of_sub_sectors not working")
                    else:
                        print(f"{vv} not in d.keys(): {d.keys()}")

            v_updated0 = df.xs(k, level="SECTOR")[x].fillna(0)
           
            den = (
                fun_xs(df, {"SECTOR": v})[x]
                .groupby(["TIME", "ISO", "TARGET", "METHOD"])
                .sum()
            )
            # NOTE: den is the sum of sub-sectors by country
            # `ratio` is the % share of each sub-sector (relative to the total of sub-sectors)
            ratio = (
                    fun_xs(df, {"SECTOR": v})[x]
                    # / fun_xs(df, {"SECTOR": v}).groupby(["TIME", "ISO"]).sum()[x]
                    / den.replace(0,np.nan)
                )

            # NOTE: `ratio` (percentage of each sub-sector) will be multiplied by the main sector `v_updated0`:
            # If ratio is Nan we assume is equal to 1
            v_updated = ratio.fillna(1)*v_updated0 # NOTE keep `ratio` on the left hand side!! This maintain the same index
            v_updated = v_updated.dropna()  # this line is needed

            if len(v_updated) == 0:
                if verbose:
                    print(f"WARNING: Skipping sector '{k}' - sub-sectors {v} not found in data")
                continue
            try:
                v_updated = v_updated.reset_index().set_index(df.index.names)
            except:
                print("error in updating the data")
            try:
                df.loc[v_updated.index, x] = v_updated
            except:
                print("error in updating the data")

            # step3 - Same as step1 (but for each of the sub_sectors). NOTE: create a function for step1 so that can be re-used here
            for vv in v:
                df = (
                    fun_harmonize_df_with_IAM(df, df_iam, x, vv, verbose=verbose)
                    if df_iam is not None
                    else df
                )

    return pd.concat([df, df_drop], axis=1)


def fun_sum_of_sub_sectors(df, x, k, v, sum_anyway: bool = False):
    df = df.copy(deep=True)
    idx = ["TIME", "ISO", "TARGET"]
    if "METHOD" in df.reset_index().columns:
        idx = idx + ["METHOD"]
    sum = fun_xs(df, {"SECTOR": v}).groupby(idx)[[x]].sum()
    sum["SECTOR"] = k
    if k in df.reset_index().SECTOR.unique():
        if sum_anyway:
            # df = df.drop(k, level="SECTOR")
            res = pd.concat([df, sum.reset_index().set_index(df.index.names)], axis=0)
            df.loc[res.index, x] = res.loc[:, x]
    else:
        df = pd.concat([df, sum.reset_index().set_index(df.index.names)], axis=0)
    return df


def fun_harmonize_df_with_IAM(df, df_iam, x, k, verbose=True):
    df = df.copy(deep=True)
    regions = df_iam.reset_index().REGION.unique()
    if len(regions) != 1:
        txt = "df_iam should contain only one region. It contains"
        raise ValueError(f"{txt} {len(regions)}: {regions}")
    iam_sectors = df_iam.reset_index().VARIABLE.unique()
    df_sectors = df.reset_index().SECTOR.unique()
    txt = "This variable will be not harmonized to match regional IAMs results"
    if k not in iam_sectors:
        if verbose:
            print(f"Cannot find {k} in `df_iam`. {txt}")
        return df
    if k not in df_sectors:
        if verbose:
            print(f"Cannot find {k} in `df`. {txt}")
        return df

    # Check if there are missing data/years in the `df.index`. (because the df.stack() method drops np.nan)
    num = (
        df.xs(k, level="SECTOR")[x].groupby(["TIME", "TARGET"]).sum().unstack("TARGET")
    )

    # Use proxi variable if data is equal to zero across all countries for a given variable `k`
    if np.prod(num == 0)[0]:
        if len(df.reset_index().TIME.unique())>1:
            if len(df.xs(k, level='SECTOR')[x].replace(0, np.nan).dropna())>1:
                df = fun_fill_zero_data_with_proxi_variable(df, x, k)

    # Append time `t` if  missing in the df.index, using values from closest time periods (e.g. if 2045 is missing, append dataframe using 2040 values)
    df = fun_append_missing_time_index(df, df_iam, x, k)
    set_idx = ["TIME", "TARGET"]
    if "METHOD" in df.reset_index().columns:
        set_idx = ["TIME", "TARGET"] + ["METHOD"]
    u = [x for x in set_idx if "TIME" not in x]

    num = df.xs(k, level="SECTOR")[x].groupby(set_idx).sum().unstack(u)

    # Calculates ratio to harmonize df with regiobal iam results
    try:
        ratio = 1 / (
            num.replace(0,np.nan) / df_iam.xs(k, level="VARIABLE").droplevel(["UNIT", "REGION", "MODEL"]).T
        )
    except:
        a=1
    ratio = ratio.replace(np.inf, np.nan)
    # For years not present in df_iam (e.g. 2022 when IAM has 5-year intervals),
    # ratio is NaN — keep original values by filling with 1 (no adjustment).
    ratio = ratio.fillna(1)
    k_updated = (
        df.xs(k, level="SECTOR")[x].unstack("TIME").reset_index().set_index(u + ["ISO"])
        * ratio.T
    )
    k_updated = k_updated.stack(dropna=False)
    k_updated.index.names = ["TIME" if x is None else x for x in k_updated.index.names]

    if isinstance(k_updated, pd.Series):
        k_updated = pd.DataFrame(k_updated).rename({0: x}, axis=1)
    k_updated["SECTOR"] = k
    k_updated.reset_index().set_index(df.index.names)
    k_updated = k_updated.reset_index().set_index(df.index.names)
    if 2005 in k_updated.index and 2005 not in df.index:
        k_updated = k_updated.drop(2005)
    df.loc[df.index.isin(k_updated.index),x]=k_updated.loc[k_updated.index, x]
    return df


def fun_append_missing_time_index(
    df: pd.DataFrame, df_iam: pd.DataFrame, col: str, var: str
) -> pd.DataFrame:
    """Append missing time index in a dataframe (long format , e.g. step1b format)

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe in step1b format
    df_iam : pd.DataFrame
        Regional IAMs results (needed to check time availability)
    col : str
        Column e.g. `ENSHORT_REF`
    var : str
        Variable e.g. `Final Energy`

    Returns
    -------
    pd.DataFrame
        Updated dataframe
    """
    num = (
        df.xs(var, level="SECTOR")[col]
        .groupby(["TIME", "TARGET"])
        .sum()
        .unstack("TARGET")
    )
    if 2005 in df_iam.columns:
        df_iam=df_iam.drop(2005, axis=1)
    time_missing = list(set(df_iam.columns) - set([t for t in num.index]))
    time_missing = [col for col in time_missing if col not in [2005]]

    # If there are missing data, fill them with nearest time values (e.g. if 2050 is missing, use values from 2045)
    if len(time_missing):
        for t in time_missing:
            # nearest value for time t (e.g. if 2050 is missing, use values from 2045)
            tvalue = fun_find_nearest_values(num.index, t, n_max=1)[0]
            df_append=(df.xs(var, level="SECTOR", drop_level=False)
                .xs(tvalue, level="TIME", drop_level=False)
                .rename({tvalue: t}))
            df = pd.concat([df, df_append])
            # df = df.append(
            #     df.xs(var, level="SECTOR", drop_level=False)
            #     .xs(tvalue, level="TIME", drop_level=False)
            #     .rename({tvalue: t})
            # )
    return df


def fun_fill_zero_data_with_proxi_variable(
    df: pd.DataFrame, col: str, var: str
) -> pd.DataFrame:
    """Use proxi variable if `var` contains data equal to zero data for all countries. Proxi variable
    is found by using `fun_fuzzy_match`.

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe in step1b format
    col : str
        Column e.g. `ENSHORT_REF`
    var : str
        Variable e.g. `Final Energy`

    Returns
    -------
    pd.DataFrame
        Updated dataframe
    """
    num = (
        df.xs(var, level="SECTOR")[col]
        .groupby(["TIME", "TARGET"])
        .sum()
        .unstack("TARGET")
    )
    proxi_vars = fun_fuzzy_match(
        list(df.drop(var, level="SECTOR").reset_index().SECTOR.unique()), var
    )
    proxi_vars = proxi_vars + ["Final Energy"]
    for proxi_var in proxi_vars:
        if np.prod(num == 0)[0]:
            # No data at all in the dataframe (all equal to zero), we use a proxi variable
            proxi = df.xs(var, level="SECTOR", drop_level=False) + df.xs(
                proxi_var, level="SECTOR", drop_level=True
            )
            proxi = proxi.reset_index().set_index(df.index.names)
            df.loc[proxi.index, col] = proxi.loc[:, col]
            num = (
                df.xs(var, level="SECTOR")[col]
                .groupby(["TIME", "TARGET"])
                .sum()
                .unstack("TARGET")
            )
            if not np.prod(num == 0)[0]:
                break
    return df


def fun_get_files_by_model(
    models: list,
    project: str,
    folder: str,
    search=None,
    countrylist=None,
    project_in_file_name=False,
) -> dict:
    """Returns a dictionary with `model` as key and the most
    recent csv file as value (from a given `folder` and a given `project`).

    Parameters
    ----------
    models : list
        List of models
    project : str
        Your selected project e.g. 'NGFS_2023'
    folder : str
        Folder with downscaled results

    Returns
    -------
    dict
        Dictionary with model as key and most recent file as value.
    """   
    files_dict = {}
    # Sort all files in folder by time of creation
    # https://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python
    files_all = [
        str(x).rsplit("\\")[-1] for x in sorted(folder.iterdir(), key=os.path.getmtime)
    ]

    scenarios = fun_get_scenarios(project)
    for model in models:
        # All files containing a given model
        filesm = [x for x in files_all if model in x and 'csv' in x]
        if len(filesm) == 0:
            raise ValueError(f"Cannot find any csv files for {model} in {folder}")
        if folder == CONSTANTS.CURR_RES_DIR("step1"):
            filesm = [
                f
                for f in filesm
                if pd.read_csv(f"{folder}/{f}", nrows=1).TARGET[0] in scenarios
            ]
            regions_all = fun_regional_country_mapping_as_dict(model, project)
            regions_to_be_downs = {k: v for k, v in regions_all.items() if len(v) > 1}
            regions = regions_to_be_downs
            regions = fun_get_iam_regions_associated_with_countrylist(
                project, countrylist, model
            )
            regions = fun_invert_dictionary({k: [v] for k, v in regions.items()})
            # Exclude Native countries (we do not have csv file for native countries)
            regions = {k: v for k, v in regions.items() if k in regions_to_be_downs}

            files = [x for region in regions for x in filesm if region in x]
        else:
            files=filesm
        fun = fun_check_if_all_characters_are_numbers
        files, date = get_files_and_date(project, search, project_in_file_name, files, fun)

        if len(files) == 0:
            print("LEN FILE = 0")
            # Try again with `project_in_file_name=False`  (e.g. non-co2 emissions files do not contain project name)
            # print("HERE")
            # files, date = get_files_and_date(project, search, False, files, fun)
            pass

        # files = [f for f in files if date in f and "csv" in f] # date, csv, project, "WITH_POLICY", "None"
        if len(files) == 1:
            files_dict[model] = files[0]
        else:
            for x in [date, "csv", project, "WITH_POLICY", "None"]:
                res = [f for f in files if x in f]
                if len(res):
                    files = res
                if len(files) == 1:
                    files_dict[model] = files[0]
                    break  # break loop as soon as we found unique file
        if folder == CONSTANTS.CURR_RES_DIR("step1"):
            harmo_files = [x for x in files if "harmo" in x]
            files = harmo_files if len(harmo_files) else files
            if not len(harmo_files):
                print(
                    f"NOTE: we are using unharmonized files for {model} (we could not find `harmo.csv` files)."
                )
            files_dict[model] = files
        elif len(files) > 1:
            txt = "Unable to automatically detect the most recent step5 file for"
            print("FILES [-1]:", files[-1])
            txt2 = f"This file is the most recent: {files[-1]} \n do you want to continue (y/n)? Or please type your selected file"
            # raise ValueError(f"{txt} {model}. We found multiple files: {files}. {txt2}")

            action = input(f"{txt} {model}. We found multiple files: {files}. {txt2}")
            if action.lower() in ["yes", "y"]:
                files_dict[model] = files[-1]
            elif action in files:
                files_dict[model] = action
            else:
                raise ValueError(
                    f"Simulation aborted by the user (user input={action})"
                )
        elif len(files) ==0 : 
            pass

    return files_dict


def get_files_and_date(project, search, project_in_file_name, files, fun):
    if project_in_file_name:
        files=[x for x in files if project in x]
    dates = ["_".join([i for i in x.rsplit("_") if fun(i)]) for x in files]
    match={x:re.search(r'\d{4}_\d{2}_\d{2}', x) for x in dates}
    dates=list({k:v.group() for k,v in match.items() if v is not None }.values())
    date = [x for x in dates if len(x.split("_")) == 3][-1]
    if search is not None:
        files = [f for f in files if search in f]
    return files, date


def fun_read_results(
    project: str,
    step: str,
    files: Optional[list] = None,
    countrylist: Optional[list] = None,
    models: Optional[list] = None,
    search: Optional[str] = None,
    rename_dict:Optional[dict]=None,
):
    if isinstance(files, dict):
        df=fun_index_names(pd.concat(pd.read_csv(x) for x in files.values()), True, int)
        if rename_dict:
            df=df.rename(rename_dict)
        scenarios=df.reset_index().SCENARIO.unique()
        # Remove `Downscaling[]` from  Model names
        models=df.reset_index().MODEL.unique()
        model_dict = {x: x.replace("Downscaling[", "").replace("]", "").replace('_downscaled','') for x in models}
        df = df.rename(model_dict)
        return df, list(files.values()), files, scenarios
    
    folder = CONSTANTS.CURR_RES_DIR(step)

    if not models:
        models = fun_get_models(project)

    if isinstance(files, str):
        search = files.split(".csv")[0]
    if files is not None and len(files) == 1:
        search = files[0].split(".csv")[0]

    df = pd.DataFrame() 
    files_dict = {}  
    if isinstance(files, str):
        for suf in ['.csv', '.xlsx']:
            if os.path.exists(folder/f"{search}{suf}"):
                df=fun_read_csv_or_excel(folder/f"{search}{suf}", None)
                files_dict={m:f"{search}{suf}" for m in models}
                files=[files]
            # if "SCENARIO" in df.index.names:
            #     scenarios = df.reset_index().SCENARIO.unique()
            # elif "TARGET" in df.columns:
            #     scenarios = df.TARGET.unique()
            # return df, [f"{search}.csv"], {m:f"{search}{suf}" for m in models}, scenarios
    
    # Get latest file avilable for each model: 
    project_in_file_name = CONSTANTS.CURR_RES_DIR('step5')==CONSTANTS.CURR_RES_DIR(step) 
    if not files:
        files_dict = fun_get_files_by_model(
            models, project, folder, search, countrylist, project_in_file_name=project_in_file_name
        )
        files = set(fun_flatten_list(list(files_dict.values())))
        # print(f"Will use this `files_dict`: {files_dict}")
    elif not(files_dict):
        if isinstance(files, str):
            # if a string we allow for reading `not harmo` files:
            files_dict = fun_get_files_by_model(
                models, project, folder, search, countrylist
            )
            files = set(fun_flatten_list(list(files_dict.values())))
            # print(f"Will use this `files_dict`: {files_dict}")
        elif search is not None:
            files_dict = fun_get_files_by_model(
                models, project, folder, search, countrylist
            )
        # elif isinstance(files, list):
        else:
            files_dict = {m: files for m in models}
    
    if not len(df):
        if "1_Final_Energy" in str(folder):
            for m, file in files_dict.items():
                df = pd.concat(
                    [df, fun_read_df_from_step1(file, countrylist, [m], project)]
                )
        else:
            for m, file in files_dict.items():
                print(file)
                df = pd.concat([df, fun_read_csv_or_excel(file, [m], folder=folder)])
                print(f"reading {file}")

    print("done with reading dataframes")

    if rename_dict:
        df=df.rename(rename_dict)

    if "SCENARIO" in df.index.names:
        scenarios = df.reset_index().SCENARIO.unique()
    elif "TARGET" in df.columns:
        scenarios = df.TARGET.unique()

    if "MODEL" in df.reset_index().columns:
        models = df.reset_index().MODEL.unique()

    # Remove `Downscaling[]` from  Model names
    model_dict = {x: x.replace("Downscaling[", "").replace("]", "").replace('_downscaled','') for x in models}
    if set(model_dict.values()) != set(model_dict.keys()):
        # if model_dict has the same keys/values pair there is no need to rename models
        if 'MODEL' in df.index.get_level_values('MODEL'):
            df=df.drop('MODEL', level='MODEL')
        df = df.rename(model_dict)
        models = df.reset_index().MODEL.unique()
    if len(df) == 0:
        raise ValueError(
            f"There are no downscaled results available in {step} for: {countrylist} from models: {models}"
        )

    return df, files, files_dict, scenarios


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


def run_sector_harmo_enhanced_iamc(
    project: str,
    df: pd.DataFrame,
    dicts: List[dict],
    df_iam: Optional[pd.DataFrame] = None,
    no_iter: int = 2,
    verbose: bool = True
) -> pd.DataFrame:
    """Returns dataframe (`df`) with harmonized sub-sectors based on a list of dictionaries `dict`
    and a given `project` (from which we get the regional mapping).

    Parameters
    ----------
    project : str
        Your project e.g. `NGFS_2023`
    df : pd.DataFrame
        Your dataframe. Dataframe should be in IAMc format
    dicts : List[dict]
        List of dictionaries with main sectors as `keys` and a list of sub-sectors as `values`.
    df_iam : Optional[pd.DataFrame], optional
        Dataframe with IAM results in IAMc format. If None, variables will be not harmonized
        to match regional IAMs results, by default None
    no_iter: int
        Number of iterations for sub-sectors adjustments, by default 2

    Returns
    -------
    pd.DataFrame
        Updated dataframe
    """

    df = df.copy(deep=True)
    fun_check_iamc_index(df)

    # Drop duplicates and get units for each variable
    df = fun_drop_duplicates(df)
    units = fun_get_variable_unit_dictionary(df)

    rename_iso_as_region = "REGION" in df.index.names
    models = df.reset_index().MODEL.unique()
    scenarios = df.reset_index().SCENARIO.unique()

    res = pd.DataFrame()
    for m in models:
        resm = pd.DataFrame()
        df_selm = df.xs(m, level="MODEL", drop_level=False)
        regions_dict = fun_regional_country_mapping_as_dict(m, project)
        for s in scenarios:
            df_selr = df_selm.xs(s, level="SCENARIO", drop_level=False)
            for r, clist in regions_dict.items():
                df_sel = fun_xs(df_selr, {"REGION": clist})
                if len(df_sel) == 0:
                    continue
                df_sel = fun_step2_format_from_iamc(
                    fun_rename_index_name(df_sel, {"REGION": "ISO"})
                )
                df_sel = fun_from_step2_to_step1b_format(df_sel, s, cols=[""])
                if 'METHOD' not in df_sel.index.names:
                    df_sel['METHOD']=''
                    df_sel=df_sel.set_index('METHOD', append=True)
                for _ in range(no_iter):
                    for d in dicts:
                        df_sel = run_sector_harmo_enhanced(df_sel, d, "", df_iam, verbose=verbose)
                # TODO
                df_sel = fun_from_step2_to_step1b_format(
                    df_sel, s, cols=[""], reverse=True
                )
                df_sel["TARGET"] = s
                df_sel = df_sel.set_index("TARGET", append=True)
                resm = pd.concat([resm, df_sel])
        resm["MODEL"] = m
    res = pd.concat([res, resm.set_index("MODEL", append=True)])

    # Convert back to IAMc format
    res = res.stack().unstack("TIME")

    # Reshape df
    res = res.reset_index().set_index(["MODEL", "ISO", "TARGET", "VARIABLE"])

    # Add units
    d = {k: [v] for k, v in units.items()}
    res = fun_add_units(res, None, fun_invert_dictionary(d))

    if rename_iso_as_region:
        res = fun_rename_index_name(res, {"ISO": "REGION"})
    return res


def fun_read_df_iams(project: str, models: Optional[List[str]] = None) -> pd.DataFrame:
    """Reads IAMs results for a list of `models` from a given `project`

    Parameters
    ----------
    project : str
        Your project e.g. "NGFS_2023
    models : Optional[List[str]]
        List of models

    Returns
    -------
    pd.DataFrame
        Dataframe with IAMs results in IAMc format
    """
    if models is None:
        models = fun_get_models(project)
    f = fun_read_df_iam_from_multiple_df
    return pd.concat(
        [f(m, CONSTANTS.INPUT_DATA_DIR / project / "multiple_df") for m in models]
    )


def fun_international_variables(
    project: str, var: str, 
    models: Optional[List[str]] = None,
    df_iam:pd.DataFrame=None,
    av_region:Optional[List[str]]=None,
    reg_name:str="World", iso_name:str="reg sum",
) -> pd.DataFrame:
    """Calculates international variables as the difference between World and the sum
     across regions for a giveb variable `var`.

    Parameters
    ----------
    project : str
        Your project e.g. 'NGFS_2023'
    models : List[models]
        List of models for which you want to calculate bunkers emissions
    av_region: Optional[List[str]]
        List of available regions. If None we we read it from df_iam, by default None
    reg_name: str
        Region name from which we substract the sum across regions (or countries)
        
    Returns
    -------
    pd.DataFrame
        _description_
    """
    if models is None:
        models = fun_get_models(project)

    group = ["MODEL", "SCENARIO", "VARIABLE", "UNIT"]

    if not df_iam:
        df_iam = fun_read_df_iams(project, models)
    idxcol = df_iam.index.names
    res_all = pd.DataFrame()
    for model in models:
        res = {}
        if not av_region:
            av_reg = df_iam.xs(model, level="MODEL").reset_index().REGION.unique()
        i_dict = {reg_name: reg_name, iso_name: [x for x in av_reg if model in x]}
        d = {"VARIABLE": var, "MODEL": model}
        for k, v in i_dict.items():
            d["REGION"] = v
            res[k] = fun_xs(df_iam, d)
        tot = res[reg_name].groupby(group).sum() - res[iso_name].groupby(group).sum()
        res_all = pd.concat([res_all, tot])

    res_all["REGION"] = reg_name
    return res_all.reset_index().set_index(idxcol)


def fun_check_inconsistencies(df_iam:pd.DataFrame, var_dict_demand:dict, verbose=False, coerce=True, absolute:bool=False)->dict:
    """Check inconsistencies in a dataframe (IAMC format) greater than 1%, for all variables defined in `var_dict_demand`

    Parameters
    ----------
    df_iam : pd.DataFrame
        Your dataframe in IAMC format
    var_dict_demand : dict
        A dictionary with your variable definition e.g {'Final Energy':['Final Energy|Industry', 'Final Energy|Transportation']}
    verbose: bool
        Whether you want the function to print variables not found in the dataframe, by default False.
    coerce: bool
        Whether you want to coerce raise value errors, by default True.
    absolute: bool
        Whether you want to calculate inconsistencies in absolute value raise value errors, by default False.
    Returns
    -------
    dict
        Variables with a discrepancy higher than 1% in absolute value
    """    
    df_iam=df_iam.copy(deep=True)
    if len(df_iam)==0:
        raise ValueError('Your dataframe is empty. Unable to check inconsistencies')
    df_iam.index.names=[x.replace('SCENARIO','TARGET').replace('SECTOR','VARIABLE').replace('ISO','REGION') for x in df_iam.index.names]
    variables=df_iam.reset_index().VARIABLE.unique()
    res={}
    missing_vars=[]
    threshold = 0.001 if absolute else 0.01 # NOTE 0.001 is EJ/yr if absolute, else 1%
    for k,v in var_dict_demand.items():
        if k in variables:
            num=df_iam.xs(k, level='VARIABLE')
            den=fun_xs(df_iam, {'VARIABLE':v}).groupby(['MODEL','TARGET','REGION','UNIT']).sum()
            den=den.reset_index().set_index(num.index.names)
            num=num.clip(1e-5)
            den=den.clip(1e-5)
            if absolute:
                val=abs(num-den).max().max()
            else:
                val=abs(num/den-1).max().max()
            if val>threshold:
                res[k]=val
        else:
            missing_vars=missing_vars+[k]
            if verbose:
                print(f"{k} not present in df. Var present: {variables}")
    if set(missing_vars)==set(var_dict_demand.keys()):
        if not coerce:
            raise ValueError('None of the keys in your dictionary (e.g`var_dict_demand`) are present in the dataframe ')
    return res


def fun_from_step1b_to_iamc(df, col='ENLONG_RATIO', method=None, coerce_errors=False):
    df=df.copy(deep=True)
    if 'METHOD' in df.reset_index().columns:
        if not method:
            methods=df.reset_index().METHOD.unique()
            raise ValueError(f'Please choose a method among those: {methods}, you passed None')
        if method =='wo_smooth_enlong' and col not in ['ENLONG_RATIO', 'ENSHORT_REF']:
            text=f"Method `{method}` is not available for `col`= {col}. Dataframe will just contain `nan` values"
            if coerce_errors:
                print(text)
            else:
                raise ValueError(text)
        df=df.xs(method, level='METHOD')
    df= fun_rename_index_name(df[col], {'TARGET':'SCENARIO', 'SECTOR':'VARIABLE'}).unstack('TIME')
    df=df.assign(MODEL='model', UNIT='EJ/yr').set_index(['MODEL','UNIT'], append=True)
    return df.reset_index().set_index(["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"])


def fun_inconsistencies(df:pd.DataFrame, list_of_dicts:List[dict], by_sector:bool=True, by_country:bool=False, method:Optional[str]=None, col:Optional[str]=None, absolute:bool=False )->dict:
    """
    Check inconsistencies by country or sector in a dataframe `df`, either in step1b or IAMC format.
    If `df` is in step1b format you need to select a `method` and column `col`.
    NOTE: Please check https://github.com/iiasa/downscaler_repo/issues/181 for alternative/simpler versions

    Parameters:
    df (pd.DataFrame): DataFrame with inconsistencies to be checked (in IAMC or step1b format).
    list_of_dicts (Union[List[dict], dict]): List of dictionaries or a single dictionary containing variables and their sub-sectors.
    by_sector (bool, optional): Whether to calculate inconsistencies by sector. Defaults to True.
    by_country (bool, optional): Whether to calculate inconsistencies by country. Defaults to False.
    method (str, Optional): Method to check inconsistencies, if your dataframe is in step1b format.
    col (str, Optional): The column of the dataframe to be checked (e.g., 'ENLONG_RATIO'), if your dataframe is in step1b format.
    absolute (bool): Whether you want to calculate inconsistencies in absolute value raise value errors. Defaults to False.
    Returns:
    Union[dict]: Inconsistencies by country or sector.
    """
    df=df.copy(deep=True)
    if 'TIME' in df.index.names:
        if not method and 'METHOD' in df.index.names:
            raise ValueError('Your dataframe seems to be in step1b format (it contains a METHOD index). Please pass a `method`e .g. `wo_smooth_enlong`')
        if not col:
            raise ValueError('Your dataframe seems to be in step1b format (it contains a METHOD index). Please pass a `col` e.g. `ENLONG_RATIO` ')
        df=fun_from_step1b_to_iamc(df, col=col, method=method)
    else:
        df=fun_rename_index_name(df, {'REGION':'ISO'})
    
    # Check if list_of_dicts is a single dictionary, if so, convert it to a list
    if isinstance(list_of_dicts, dict):
        list_of_dicts = [list_of_dicts]

    # Validate input arguments
    if not (by_sector or by_country):
        raise ValueError("Either `by_sector` or `by_country` needs to be True")

    iso_unique = df.reset_index().ISO.unique()
    inconsistencies={}
    sum_inconsistencies={}    
    
    if by_sector==True and by_country==False:
        for x in enumerate(list_of_dicts):
            if x[0]==0:
                d=fun_check_inconsistencies(df,x[1], absolute=absolute)
            else:
                # Sum up current and previous dictionary
                d=sum_two_dictionaries(d,fun_check_inconsistencies(df,x[1],absolute=absolute))
        # This is better than `d.update()` to avoid sectors present in both dictionaries being overwritten 
        return fun_sort_dict(d, by='values', reverse=True)

    for c in iso_unique:
        df_bycountry=df.xs(c, level='ISO', drop_level=False)
        for x in enumerate(list_of_dicts):
            if x[0]==0:
                d=fun_check_inconsistencies(df_bycountry,x[1],absolute=absolute)
            else:
                # Sum up current and previous dictionary
                d=sum_two_dictionaries(d,fun_check_inconsistencies(df_bycountry,x[1],absolute=absolute))
        inconsistencies[c] =fun_sort_dict(d,by='values', reverse=True)
        # Sum of inconsistencies across sectors, by country
        sum_inconsistencies[c] = sum(inconsistencies[c].values())
    # Order by country with largest inconsistencies
    sum_inconsistencies=fun_sort_dict(sum_inconsistencies, by='values', reverse=True)
    inconsistencies={k:inconsistencies[k] for k in sum_inconsistencies.keys()}
    if by_country and not by_sector:
        return sum_inconsistencies 
    if by_country and by_sector: 
        return inconsistencies


def get_worst_method_inconsistencies(obs: pd.DataFrame, var_dicts: List[Dict[str, List[str]]], col: str, absolute:bool=False) -> Dict[str, float]:
    """
    Identify the worst-performing method and show inconsistencies.

    Parameters:
    obs (pd.DataFrame): DataFrame with inconsistencies.
    var_dicts (List[Dict[str, List[str]]]): List of dictionaries with variable mappings.
    col (str): Column name.

    Returns:
    Dict[str, float]: Dictionary containing inconsistencies for the worst-performing method.
    """
    # Select relevant methods below
    available_methods=obs.reset_index().METHOD.unique()
    if col in ['ENSHORT_REF', 'ENLONG_RATIO']:
        methods=[x for x in available_methods if x in ['wo_smooth_enlong', 'mymethod']] 
    else:
        methods=[x for x in available_methods if x not in ['wo_smooth_enlong']]

    # Calculate inconsistencies for each method below
    res_all_methods = {
        method: fun_inconsistencies(obs, var_dicts, col=col, method=method, absolute=absolute) for method in methods
    }
    res_all_methods_sum = fun_sort_dict(
        {k: sum(list(v.values())) for k, v in res_all_methods.items()}, by='values', reverse=True
    )
    worst_method = list(res_all_methods_sum.keys())[0]
    return {worst_method:res_all_methods[worst_method]}


def get_worst_scenario_inconsistencies(df:pd.DataFrame, my_dicts:List[Dict[str, List[str]]], absolute:bool=False)->Dict[str, float]:
    """Identify scenario with largest inconsistencies (and show them in a dictionary)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be checked
    my_dicts : List[Dict[str, List[str]]]
        Dictionaries with scenario mapping

    Returns
    -------
    Dict[str, float]
        Dictionary containing inconsistencies for the worst-performing scenario.
    """    
    res_all_scen={scen:fun_inconsistencies(df.xs(scen, level='SCENARIO', drop_level=False), my_dicts, absolute=absolute) for scen in df.reset_index().SCENARIO.unique()}
    return fun_sort_dict(
        {k: sum(list(v.values())) for k, v in res_all_scen.items()}, by='values', reverse=True
    )


def get_worst_column(df:pd.DataFrame, my_dicts:List[Dict[str, List[str]]], absolute:bool=False)->Dict[str, float]:
    """Identify column with largest inconsistencies (and show them in a dictionary)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be checked
    my_dicts : List[Dict[str, List[str]]]
        Dictionaries with scenario mapping

    Returns
    -------
    Dict[str, float]
        Dictionary containing inconsistencies for the worst column found.
    """
    res_all_t={t:fun_inconsistencies(df[[t]], my_dicts, absolute=absolute) for t in df.columns}
    return fun_sort_dict({k: sum(list(v.values())) for k, v in res_all_t.items()}, by='values', reverse=True)


def show_inconsistencies(df:pd.DataFrame, # var_dict_demand:dict, var_dict_supply:dict      
                         my_dicts:List[dict],
                         method=None,
                         scen=None,
                         c=None,
                         var:Union[int, str, List[str]]=0,  # 0 Means the worse variable. 1 Means the second worse
                         t=None,
                         priority:str='Final Energy',
                         absolute:bool=False
                         )->pd.DataFrame:
    
    """Show inconsistencies in dataframe (in step1b or IAMC format)

    Parameters:
    df (pd.DataFrame): DataFrame to be checked.
    var_dicts (List[Dict[str, List[str]]]): List of dictionaries with variable mappings.
    method (Optional str): Wheter you want to check a specific method (e.g. 'wo_smooth_enlong'). If None will search for the worst performing method, by default None.
    scen (Optional str): Wheter you want to check a specific scenario. If None will search for the worst performing scenario,  by default None.
    c (Optional str): Wheter you want to check a specific country. If None will search for the worst performing country,  by default None.
    var (Union[int, str, List[str]], defaults to 0): Wheter you want to check a specific variable. 0 means you want to check for the worst performing variable, 1 means the second worst etc., by default 0.
    t (Optional str): Wheter you want to check a specific time. If None will search for the worst performing time period,  by default None.
    priority (Optional str): Whether you want to prioritize a variable. Show this variable if consistencies are found (otherwise shows other variables with inconsistencies), by default `Final Energy`.
    absolute (bool): Whether you want to calculate inconsistencies in absolute value raise value errors. Defaults to False.

    Returns:
    Dict[str, float]: Dataframe showing largest inconsistencies found, for a given `method`, `scen`, `c`, `var`, and `t`.
    """

    # TODO automatically detect worse column

    # Step 0) Detect worse method in the `df` (if step1b format) and convert to IAMC format
    res_method_col=None
    if 'METHOD' in df.index.names:
        res_method_col = get_worst_column_and_method_inconsistencies(df, my_dicts, method, absolute=absolute)
        # worse_column_method, col = get_worst_column_method_inconsistencies(df, my_dicts, method)

        col=list(res_method_col.keys())[0]
        # Convert df to IAMc format for the worst method
        df=fun_from_step1b_to_iamc(df, col=col, method=res_method_col[col]) 

    df=fun_rename_index_name(df, {"REGION":"ISO"})
    
    # Select time, scenario, country, variable if provided
    if t is not None:
        df=df[[t]]
    if scen is not None:
        df=fun_xs(df, {'SCENARIO':scen})
    if c is not None:
        df=fun_xs(df, {'ISO':c})
    if not isinstance(var, int):
        df=fun_xs(df, {'VARIABLE':var})

    # Step1 detect worse scenario
    if scen is None:
        res=get_worst_scenario_inconsistencies(df, my_dicts, absolute=absolute)
        scen=list(res.keys())[0]
    df=fun_xs(df, {'SCENARIO':scen})
    
    # Step 2) Select worse country
    if c is None:
        res=fun_inconsistencies(df, my_dicts, by_country=True, absolute=absolute)
        c=list(res.keys())[0]
    
    
    df=fun_xs(df, {'ISO':c})

    # Step 3) Select worse sector (or energy carriers)
    if len(list(res.values())[0].keys())>0:
        all_vars=list(res.values())[0].keys()
    else:
        print('All sectors are consistent!')
        return pd.DataFrame()
    if priority in all_vars:
        var=priority
    else:
        var=list(list(res.values())[0].keys())[var]
    
    # var=list(res.keys())[0]
    
    # Select sub sectors associated to `var`. NOTE: my_dicts[0] is `var_dict_demand`,   my_dicts[1] is  `var_dict_supply` (or viceversa)
    subs=[]
    for d in my_dicts:
        if var in d:
           subs=subs+d[var]  
    # Select all relevant variables (`var` + subsectors)
    all_vars=[var]+subs


    # Step4 get worse time period:
    if t is None:
        t=list(get_worst_column(fun_xs(df, {'VARIABLE':all_vars}), my_dicts, absolute=absolute).keys())[0]
    df=df[[t]]

    if res_method_col is not None:
        df=df.assign(METHOD=res_method_col[col], COLUMN=col).set_index(['METHOD', 'COLUMN'], append=True)
    
    # NOTE Block below (optional) calculates sum according to dictionaries (with variable mapping) and concat to `df`
    data=pd.concat([fun_xs(df, {'VARIABLE':[var]}), fun_xs(df, {'VARIABLE':subs})])
    sum_dict={}
    for k,v in enumerate(my_dicts):
        if var in v:
            sum_dict[k]=fun_xs(df, {'VARIABLE':v[var]}).groupby([x for x in df.index.names if x!='VARIABLE']).sum().rename({t:f'd{k}'}, axis=1)    # Step4 Return Slice dataframe by variables and iso
    sector_sum=pd.concat(list(sum_dict.values()), axis=1).assign(VARIABLE=var).reset_index().set_index(data.index.names)    
    df=pd.concat([data, sector_sum], axis=1).droplevel(['UNIT'])
    
    return pd.concat([fun_xs(df, {'VARIABLE':[var]}), fun_xs(df, {'VARIABLE':subs})])


def get_worst_column_and_method_inconsistencies(df, my_dicts, method, absolute:bool=False):
    res_method_col={}
    df=df.copy(deep=True)
    selcols=[x for x in df.columns if ('ENSHORT_REF' in x or 'ENLONG_RATIO' in x) and 'EI_' not in x]
    if method is not None:
        df=fun_xs(df, {'METHOD':method})

    for col in selcols:
        res_worse_method=get_worst_method_inconsistencies(df, my_dicts, col, absolute=absolute)
            # Inconsistencies by country
        # tmp_method=list(res_worse_method.keys())[0]
        # res=res_worse_method[tmp_method]
        temp={sum(list(v.values())):k for k,v in res_worse_method.items()}
        temp=fun_sort_dict(temp, by='keys', reverse=True)
        val=list(temp.keys())[0]
        res_method_col[val]={col:temp[val]}
    
    res_method_col=fun_sort_dict(res_method_col, by='keys', reverse=True)
        
    res_method_col=res_method_col[list(res_method_col.keys())[0]]
    return res_method_col


def fun_match_hist(baseyear:int, trade:pd.DataFrame, iea_adj:pd.DataFrame)->pd.DataFrame:
    """
    Adjust `trade` data to replicate historical `iea_adj` data up to a specified `baseyear` 
    and extend the base-year adjustment to future years.

    Parameters
    ----------
    baseyear : int
        The base year up to which historical data is to be replicated (e.g. 2020).
    trade : pd.DataFrame
        DataFrame containing the data to be adjusted (e.g. trade data).
    iea_adj : pd.DataFrame
        DataFrame containing the historical IEA data used for adjustment.

    Returns
    -------
    pd.DataFrame
        Adjusted DataFrame with trade data that replicates historical data up to the base year and extends adjustments to future years.

    Notes
    -----
    The function performs the following steps:
    1. Calculates the delta needed to replicate historical trade data up to the base year.
    2. Extends the calculated delta to future years (up to 2105 in increments of 5 years).
    3. Adjusts the original trade data using the calculated delta.
    """
    # Calculate delta to replicate historical trade data
    delta=fun_add_multiply_dfmultindex_by_dfsingleindex(-trade.loc[:,:baseyear], iea_adj, operator='+')
    delta.columns=pd.to_numeric(delta.columns)

    # Calculate historical trade data until base year (e.g 2020)
    add=pd.concat([delta[[baseyear]].rename({baseyear:x}, axis=1) for x in range(2025,2105,5)], axis=1)
    delta=pd.concat([delta, add], axis=1)
    
    # Adjust trade data
    trade_2020=trade+delta
    return trade_2020


def search(df:pd.DataFrame, val:Union[str, float, int], search_all:bool=False)->list:
    """
    Search for a value in a DataFrame and return matching rows or a summary.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to search in.
    val : str or number
        The value to search for in the DataFrame.
    search_all : bool, optional
        If True, search all columns; otherwise, search index columns, by default False.

    Returns
    -------
    list
        A DataFrame with matching values or a summary dictionary (which can be used as input in `fun_xs` function).
    """
    d=summarize_dataframe(df,search_all)
    mylist=fun_flatten_list(list(d.values()))
    val=str(val)
    # Check if the the wildcard `*` is present in `val`. This means we want to find any string that contain `val`
    flag=False
    if '*' in val:
        val=val.replace('*','')
        flag=True 

    # Try to search for exact `val`, unless we don't find it, or wildcard `*` present in `val`-> Flag =True
    res=[i for i in mylist if str(val).upper() == str(i).upper()]
    if len(res)==0 or flag:
        res= [i for i in mylist if str(val).upper() in str(i).upper()]
    # if len(res)>10:
    #     return res
    if set(res)==set(mylist):
        return df
    
    reslist=[df.eq(x).replace(False, np.nan).dropna(how='all').dropna(axis=1).replace(1,x) for x in res]
    # reslist=[df.where(df.eq(1)).dropna(axis=0) for x in res]
    resdf=pd.DataFrame()
    if len(reslist):
        #raise ValueError(f'Cannot find {val} in the dataframe')
        resdf=pd.concat(reslist, sort=True)
    if len(resdf)>0:
        return resdf
    mynewlist=[SliceableDict(fun_invert_dictionary(d)).slice(x) for x in res]
    mynewlist=[{k:str(v[0]) for k,v in d.items() } for d in mynewlist]
    # try:
    #     res = sum_multiple_dictionaries(mynewlist, only_common_keys=False)
    # except:
    res={k: [v] for d in mynewlist for k, v in d.items()}
    return fun_invert_dictionary(res)


def fun_get_regions(project: str, model: str, sub_folder: str = 'multiple_df') -> Dict[str, List[str]]:
    """
    Retrieve regions and their associated country mappings for a given project and model.

    Args:
        project (str): The name of the project, used to locate input data.
        model (str): The name of the model (e.g., "AIM/CGE"), which is processed to match mapping keys.
        sub_folder (str, optional): Subdirectory name containing data files. Defaults to 'multiple_df'.

    Returns:
        Dict[str, List[str]]: A dictionary mapping regions (keys) to lists of associated countries (values).
    """
    # Read IAM data from the specified model and project directory
    df = fun_read_df_iam_from_multiple_df(model, CONSTANTS.INPUT_DATA_DIR / project / sub_folder)
    
    # Extract unique regions and append 'r' suffix
    regions = [x for x in df.reset_index().REGION.unique()]
    regions = [f"{x.split('|')[1]}r" if '|' in x else f"{x}r" for x in regions]
    
    # Rename the model string to replace '/' with '_' (e.g., "AIM/CGE" -> "AIM_CGE")
    model_renamed = model.replace('/', '_')  
    
    # Get the regional-to-country mapping as a dictionary
    regmap = fun_regional_country_mapping_as_dict(model_renamed, project)
    
    # Filter the mapping to include only relevant regions
    return {k: v for k, v in regmap.items() if k in regions}


def fun_harmonize_df_iam_with_hist_data(
    r: str,
    countrylist: List[str],
    df_iam: pd.DataFrame,
    hist: pd.DataFrame,
    main_var: str = 'Emissions|CO2|Energy',
    vars_to_be_harmo: List[str] = [
        'Emissions|CO2|Energy',
        'Emissions|CO2|Energy|Demand|Transportation',
        'Emissions|CO2|Energy|Demand|Industry',
        'Emissions|CO2|Energy|Supply|Heat',
        'Emissions|CO2|Energy|Supply|Electricity',
        'Emissions|CO2|Energy|Demand|Residential and Commercial'
    ],
    tc: Optional[int] = 2050
) -> pd.DataFrame:
    """
    Harmonize IAM data with historical data for a given region and set of variables.
    It works with data in IAMC format. Index names in the two dataframes must be the same.
    
    It rescales IAMs data to match historical 2020 data for a main variable `main_var` using a ratio harmonization. 
    Then rescales all other variables `vars_to_be_harmo` by the same percentange - to keep sectorial consistency.
    If `tc` is None a ratio harmonization will be applied over the whole time periods.
    Otherwise, it will apply a ratio harmonization up to the time of convergence, as we do in step_5e
    (beyond the time of convergence `tc` IAMs data will remain unchanged).
    

    Args:
        r (str): The name of the region.
        countrylist (List[str]): List of countries within the specified region.
        df_iam (pd.DataFrame): DataFrame containing IAM data.
        hist (pd.DataFrame): DataFrame containing historical data.
        main_var (str, optional): The primary variable for ratio calculation. Defaults to 'Emissions|CO2|Energy'.
        vars_to_be_harmo (List[str], optional): List of variables to harmonize. Defaults to a predefined list of variables.
        tc (Optional[int], optional): The time of convergence for harmonization. If None, no convergence is applied. Defaults to 2050.

    Returns:
        pd.DataFrame: The harmonized IAM data.
    """
    # Calculate ratio of historical data to IAM data for the main variable
    ratio = (
        fun_xs(hist, {'REGION': countrylist, 'VARIABLE': main_var})
        .groupby('SCENARIO').sum() /
        df_iam.xs((r, main_var), level=("REGION", "VARIABLE")).droplevel(['MODEL', 'UNIT'])
    )[[2010, 2015, 2020]]
    
    # Drop rows with all NaN values and extend ratio over future years
    ratio = ratio.dropna(how='all')
    add = pd.concat([ratio[[2020]].rename({2020: x}, axis=1) for x in range(2020, 2055, 5)], axis=1)
    ratio = pd.concat([ratio.dropna(how='all', axis=1), add], axis=1)

    # Apply the ratio to harmonize the IAM data
    df_iam_harmo = fun_xs(
        fun_add_multiply_dfmultindex_by_dfsingleindex(df_iam, ratio, operator='*'),
        {"VARIABLE": vars_to_be_harmo}
    )

    # If no time of convergence is specified, return the harmonized data
    if tc is None:
        return fun_xs(df_iam_harmo, {"REGION":r})

    # Apply composite weighting for harmonization up to the time of convergence
    w = fun_discount_rate(range(2010, 2105, 5), range(2010, 2025, 5), tc, 1, 0)
    df_iam_composite = df_iam_harmo * w + (1 - w) * fun_xs(df_iam, {"VARIABLE": vars_to_be_harmo})
    
    return fun_xs(df_iam_composite, {"REGION":r})

