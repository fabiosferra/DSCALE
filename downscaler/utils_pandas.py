import pandas as pd
import numpy as np
import os
from downscaler import CONSTANTS
from typing import Callable, Dict, List, Optional, Union, List, Dict
from downscaler.utils_string import (
    fun_check_if_all_characters_are_numbers,
    fun_sorts_list_of_strings,
)


def fun_skip_first_row(df: pd.DataFrame, row: int = 0) -> pd.DataFrame:
    """Skips first row of a Dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe
    row : int, optional
        Row that you want to skip (first=0), by default 0

    Returns
    -------
    pd.DataFrame
        Dataframe without the first row
    """
    df.columns = df.iloc[row]
    return df.drop(df.index[row])


import pandas as pd


def fun_xs(df: pd.DataFrame, d: dict, exclude_vars: bool = False, error_if_var_is_missing:bool=False) -> pd.DataFrame:
    """Slices `df` based on dictionary `d`.
    It uses the df.index.get_level_values(k).isin(v) method, where k=d.keys() and v= d.values()

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be sliced
    d : dict
        Dictionary, where d.keys() are the levels of multindex in `df`, and df.values() are the labels contained in the index.
    exclude_vars: bool, optional
        Wheter you want to exclude/remove variables in your dictionary, default to False
    error_if_var_is_missing: bool, optional
        Whether you want to raise ValueError if selected variables are not present in the df, default to False

    Returns
    -------
    pd.DataFrame
        Sliced dataframe
    """
    for v in d.values():
        if not (isinstance(v, str) or isinstance(v, list)):
            raise ValueError(f'Values in dictionary `d` need to be either list or string, you passed {type(v)}')
    if len(df):
        for k in d:
            if k not in df.index.names and k not in "columns":
                txt = "is not present in df.index.names:"
                raise ValueError(f"{k} {txt} {df.index.names}")
        if error_if_var_is_missing:
            for k, v in d.items():
                if v not in df.reset_index()[k].unique():
                    raise ValueError(f"Cannot find `{v}` {k.lower()} in the dataframe")
    for (
        k,
        v,
    ) in d.items():
        if type(v) is not list:
            v = [v]
        if k in df.index.names:
            if exclude_vars:
                df = df[~df.index.get_level_values(k).isin(v)]
            else:
                df = df[df.index.get_level_values(k).isin(v)]
        else:
            df = df.filter(v)

    return df

def summarize_dataframe(df_orig:pd.DataFrame, search_all=False)->dict:
    """
    Summarize the unique values in the columns or index of a DataFrame.

    Parameters
    ----------
    df_orig : pd.DataFrame
        The original DataFrame to be summarized.
    search_all : bool, optional
        If True, summarize all columns; otherwise, summarize index columns, by default False.

    Returns
    -------
    dict
        A dictionary with column/index names as `keys` and lists of unique values as `values`.
    """
    df=df_orig.copy(deep=True)
    mycols=df.columns if search_all else df_orig.index.names
    df=df.reset_index()
    summary = {x: df[x].unique() for x in mycols}
    return summary

def fun_index_names(
    df: pd.DataFrame,
    upper_case_idx_names: bool = False,
    col_type: Optional[Union[str, int]] = None,
    sel_idx: Optional[list] = None,
) -> pd.DataFrame:
    """Add all columns that are not float/numbers as index names (in addition to existing index.names).
    Will return a dataframe with multindex, where all columns will be numbers (e.g. columns= years).

    Parameters
    ----------
    df : pd.DataFrame
        Initial dataframe
    upper_case_idx_names: bool
        Wheter you want to have idx names strings as upper case e.g. Model-> MODEL
    col_type: Union[None, str, int]
        Wheter you have a preference regarding the type of your columns.
        None will keep columns strings as they are (no transformation will happen).
        `str` will transform columns as strings e.g. ->'2010'.
        `int` will transform columns as integers e.g. ->2010. By default None
    sel_idx: Union[None, list]
        Wheter you want to set a predefined Multindex (list of coluns to be set as indexes)

    Returns
    -------
    pd.DataFrame
        Updated dataframe with multindex (where all columns are time series/numbers)
    """
    df = df.copy(deep=True)
    if col_type==int:
        # first convert to a float. later to an integer
        df=fun_index_names(df,upper_case_idx_names, float)

    init_idx = list(df.index.names)

    if upper_case_idx_names:
        df.columns = [
            x if fun_check_if_all_characters_are_numbers(x) else x.upper()
            for x in df.columns
        ]

    cols = [
        x
        for x in df.columns
        if fun_check_if_all_characters_are_numbers(x)
        if "index" not in str(x)
    ]

    # Define your index names
    idx = init_idx + list(df.loc[:, ~df.columns.isin(cols)].columns)
    idx = [x for x in idx if type(x) is str]  # and "index" not in str(x)]

    black_list_idx = ["index", "unnamed", "level_"]
    for b in black_list_idx:
        idx = [x for x in idx if b not in x.lower()]

    if sel_idx:
        idx = sel_idx
    
    if col_type is not None:
        df.columns = [
            col_type(x) if fun_check_if_all_characters_are_numbers(x) else x
            for x in df.columns
        ]
        cols = [
            col_type(x) if fun_check_if_all_characters_are_numbers(x) else x
            for x in cols
        ]

    # # Sort idx based on iamc indexes
    # iamc_idx=['MODEL','SCENARIO','TARGET','REGION','ISO','VARIABLE','UNIT',]
    # idx=fun_sorts_list_of_strings(idx,iamc_idx )
    return df.reset_index().set_index(idx)[cols]


def fun_read_csv(
    files_dict: dict, upper_idx_columns: bool, col_type: Union[int, str]
) -> dict:
    """Reads csv files based on a dictionary `files_dict` using the function `fun_index_names`
    for data in wide format, like IAMc data.
    You can choose whether to have idx columns using an upper case `upper_idx_columns` and whether
    you want columns as string or integer.

    Parameters
    ----------
    files_dict : dict
        Dictionary with a given name and csv a file location e.g. {'Historical data':'myfolder/iea_data.csv'}
    upper_idx_columns : bool
        Whether you want your idx names in upper case
    col_type : Union[int, str]
        Whether you want your columns as str or integers

    Returns
    -------
    dict
        A dictionary with the same keys as in `files_dict` and pd.DataFrames as values
    """

    return {
        k: fun_index_names(pd.read_csv(f), upper_idx_columns, col_type)
        for k, f in files_dict.items()
    }


def fun_add_multiply_dfmultindex_by_dfsingleindex(
    df_multindex: pd.DataFrame,
    df_singleindex: Union[pd.DataFrame, pd.Series],
    operator: str = "+",
) -> pd.DataFrame:
    """Makes Arithmetic operations (add/multiply etc.) across 1) a dataframe with multindex `df_multindex` and 2) another dataframe with single index `df_singleindex`
    if single index is in multindex (and if the two dataframes have columns in common).

    Parameters
    ----------
    df_multindex : pd.DataFrame
        Dataframe with multindex
    df_singleindex : Union[pd.DataFrame, pd.Series]
        Dataframe with single index
    operator : str, optional
        Type of arithmetic operation, by default "+"

    Returns
    -------
    pd.DataFrame
        Updated dataframe with multindex (arithmetic operations)

    Raises
    ------
    ValueError
        If chosen operator is invalid (not Arithmetic operator)
    ValueError
        If `df_singleindex` has a multindex
    ValueError
        If `df_singleindex.index.names` is not included in `df_multindex.index.names`
    ValueError
        There are no columns in common in the two dataframe
    """
    ALLOWED_OPERATORS = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "**": "pow",
        "/": "div",
        "//": "floordiv",
        "%": "mod",
    }
    if operator not in ALLOWED_OPERATORS:
        raise ValueError(
            f"Only Arithmetic Operators are allowed: {ALLOWED_OPERATORS}. You provided: {operator} "
        )

    if isinstance(df_singleindex, pd.Series):
        df_singleindex = pd.DataFrame({x: df_singleindex for x in df_multindex.columns})

    df_multindex = df_multindex.copy(deep=True)
    df_singleindex = df_singleindex.copy(deep=True)
    idxnames = df_singleindex.index.names
    if len(idxnames) != 1:
        raise ValueError(
            f" `df_singleindex` must contain an single index. You provided a dataframe with multi-index: {idxnames}"
        )
    if idxnames[0] not in df_multindex.index.names:
        raise ValueError(
            f"df_singleindex.index.names must be included in df_multindex.index.names: {df_multindex.index.names}\n"
            f"df_singleindex.index.names: {df_singleindex.index.names} "
        )
    if not [x for x in df_singleindex.columns if x in df_multindex.columns]:
        raise ValueError(
            f"Columns of the two dataframes are different\n"
            f"df_multindex = {df_multindex.columns} \n"
            f"df_singleindex = {df_singleindex.columns}"
        )
    ratio_dict = df_singleindex.to_dict()
    cols = sorted(set(df_multindex.columns).intersection(set(df_singleindex.columns)))
    for t in cols:
        mydict = ratio_dict[t]
        df_multindex["MUL"] = [
            mydict.get(r, np.nan)
            for r in df_multindex.index.get_level_values(idxnames[0])
        ]

        df_multindex.loc[:, int(t)] = getattr(
            df_multindex.loc[:, int(t)], ALLOWED_OPERATORS[operator]
        )(df_multindex.loc[:, "MUL"])
    return df_multindex.drop("MUL", axis=1)


## Growth rate
def fun_operations_compared_to_base_year(
    _df: pd.DataFrame, _base_year: Union[str, int], operator="add", axis=0
) -> pd.DataFrame:
    """This function makes operations compared to the base year.

    Returns
    -------
    pd.DataFrame
        Dataframe with difference compared to base year
    """
    ALLOWED_OPERATORS = ("add", "sub", "mul", "div")
    if operator not in ALLOWED_OPERATORS:
        raise ValueError(
            f"Only pandas operators are allowed: {ALLOWED_OPERATORS}. You provided: {operator} "
        )
    _df = _df.copy(deep=True)
    # NOTE code below is equivalent to:
    #  return eval(f"_df.iloc[:, :].{operator}(_df[_base_year], axis={axis})")
    return getattr(_df.iloc[:, :], operator)(_df[_base_year], axis=axis)


def fun_remove_np_in_df_index(df: pd.DataFrame, sel_type=str) -> pd.DataFrame:
    """Slices dataframe to get only strings (or `sel_type`) in the `df`.index.
    This function can be used to remove np.nan values in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe
    sel_type : _type_, optional
        Selected type, by default str

    Returns
    -------
    pd.DataFrame
        Sliced dataframe
    """
    for k in df.index.names:
        df = fun_xs(
            df, {k: [x for x in df.reset_index()[k].unique() if type(x) is sel_type]}
        )
    return df


def fun_fill_na_with_previous_next_values(
    df: pd.DataFrame,
    use_previous_value: bool = True,
    use_next_value: bool = True,
    _axis=1,
) -> pd.DataFrame:
    """This function fills na values with previous `use_previous_value` or next `use_next_value` values (or both) in your dataframe.
    If you choose both, it will fill na with previous values first (e.g. fill 2015 with 2010 data).
    If there are still na values it will use the next values (e.g. fill 2015 with 2020 data).
    It retuns the update dataframe
    """
    df = df.replace(np.inf, np.nan)
    df = df.replace(-np.inf, np.nan)
    if use_previous_value:
        df = df.ffill(axis=_axis)
    if use_next_value:
        df = df.bfill(axis=_axis)

    return df


def fun_replace_model_downscaled_with_model(df: pd.DataFrame) -> pd.DataFrame:
    """Updated model names in your df: Replaces f"{model}_downscaled" with f"{model}".

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with updated model names
    """
    myidx = None
    if "MODEL" in df.index.names:
        myidx = df.index.names
        df = df.reset_index()
    df.loc[:, "MODEL"] = [x.replace("_downscaled", "") for x in df.MODEL]
    return df.set_index(myidx) if myidx is not None else df


def fun_replace_d_dot_iso(df: pd.DataFrame, mycol: str) -> pd.DataFrame:
    """Replaces D.ISO in the column `mycol` of the dataframe `df`.
    Return the updated dataframe with ISO names (without D.)

    Args:
        df (pd.DataFrame): _description_
        mycol (str): selected column e.g. 'REGION'

    Returns:
        pd.Dataframe: updated dataframe
    """
    myidx = df.index.names
    cols = df.columns
    df = df.reset_index()
    if "ISO" in df.columns:
        df.loc[:, mycol] = [x.replace("D.", "") for x in df.ISO]
    elif "REGION" in df.columns:
        df.loc[:, mycol] = [x.replace("D.", "") for x in df.REGION]
    else:
        raise ValueError("Could not find `ISO` nor `REGION` column in the dataframe")
    return df.set_index(myidx)[cols]


def fun_select_model_scenarios_combinations(
    df_all: pd.DataFrame, d: dict
) -> pd.DataFrame:
    """Selects model/scenario combinations in a given dataframe `df_all` based on a
        dictionary `d` where key is the model, and value is a scenario list: `d={model:[scenarios list]}`

    Parameters
    ----------
    df_all : pd.DataFrame
        Your initial dataframe
    d : dict
        Dictionary with selected model/scenarios. Where key=model and value=list of scenarios
        Example: {'REMIND 3.1':['rigid_600']}

    Returns
    -------
    pd.DataFrame
        Dataframe with seletced model/scenarios
    """
    d = {k: v for k, v in d.items() if k in df_all.reset_index().MODEL.unique()}
    df_sel = pd.DataFrame()
    for model, scen in d.items():
        if model in df_all.reset_index().MODEL.unique():
            df_current = fun_xs(
                df_all.xs(model, level="MODEL", drop_level=False), {"SCENARIO": scen}
            )
            df_sel = pd.concat([df_sel, df_current], axis=0)
    return df_sel


def fun_check_df_contains_single_region_country(
    df: pd.DataFrame, df_name: str = ""
) -> str:
    """Checks if a dataframe contains a single iso/region.

    Parameters
    ----------
    df : pd.DataFrame
        Your datafarme
    df_name : str, optional
        How you want to name your dataframe, by default ''

    Returns
    -------
    str
        Single REGION or ISO detected in the dataframe

    Raises
    ------
    ValueError
        If dataframe contains both an ISO and REGION column
    ValueError
        If dataframe does not contain any REGION/ISO colum
    ValueError
        If the len() of the REGION/ISO column is not equal to one. (we expect a single REGION/ISO in the `df`)
    """
    df = df.reset_index()
    allcols = df.columns
    if "REGION" in allcols and "ISO" in allcols:
        raise ValueError(
            f"Dataframe `{df_name}` should contain either a REGION or a ISO column, not both"
        )
    elif "REGION" in allcols:
        regions_list = df.REGION.unique()
    elif "ISO" in df:
        regions_list = df.ISO.unique()
    else:
        raise ValueError(f"Dataframe `{df_name}` does not contain any REGION/ISO")
    if len(regions_list) != 1:
        raise ValueError(
            f"Dataframe `{df_name}` contains zero or more than one REGIONs/ISOs: {regions_list}"
        )
    return regions_list[0]


def fun_find_common_variables_in_multiple_dataframes(
    df_dict_list: Union[list, dict],
    col: str,
) -> list:
    """Check if two dataframes have common elements in a given column (or dataframe attribute)

    Parameters
    ----------
    df_dict_list : Union[list, dict]
        List (or dictionary) with Dataframes to be checked

    col : str
        Column (or df attribute) where searching elements in common (e.g. col='MODEL' or col='columns')

    Returns
    -------
    list
        List of elements in common in the two dataframes, in a given df attribute
    """
    setlist = []
    dflist = df_dict_list.values() if isinstance(df_dict_list, dict) else df_dict_list
    for v in dflist:
        setlist = setlist + [set(getattr(v.reset_index(), (col)).unique())]
    common = set.intersection(*setlist)

    str_common = [
        x
        for x in common
        if type(x) is str and "index" not in x.lower() and "unnamed" not in x.lower()
    ]
    float_common = [x for x in common if type(x) is not str]
    return str_common + float_common


def fun_check_missing_elements_in_dataframe(
    check_dict: dict, df: pd.DataFrame, coerce_errors: bool = False
) -> list:
    # """Raises values error if there are no variables in common in two dataframes
    """Raises values error  if the  dataframe does not contain a variable/element as specified in `check_dict` .

    Parameters
    ----------
    check_dict : dict
        Dictionary {k:v} `k` is the column (or df attribute) of the dataframe where to search for `v` (variable to be checked)
    df : pd.DataFrame
       Dataframe to be checked
    coerce_errors : bool
        Wheter you want to silent the raise value error
    Returns: list
        List of missing elements
    Raises
    ------
    ValueError
        If the  dataframe does not contain a variable/element as specified in `check_dict` .
    """
    res = {}
    df = fun_xs(df, check_dict)
    df = df.reset_index()
    for k, v in check_dict.items():
        if isinstance(v, (list, tuple, set)):
            x_missing = []
            for x in v:
                if x not in getattr(df, k).unique():
                    x_missing = x_missing + [x]
                    if not coerce_errors:
                        raise ValueError(f"{x} is missing in the dataframe: {check_dict}")
            if len(x_missing) > 0:
                res[k] = x_missing
        elif v not in getattr(df, k).unique():
            res[k] = v
            txt=f"{v} is missing in the dataframe: {check_dict}"
            if not coerce_errors:
                raise ValueError(txt)
            else:
                action=input(f'{txt}. Do you wish to continue? y/n')
                if action.lower() not in ["yes", "y"]:
                    raise ValueError('Simulation Aborted by the user')
    return res


def fun_find_iso_column(
    df: pd.DataFrame, iso_list: Union[list, tuple] = ("AUT")
) -> str:
    """Find the ISO column in a dataframe `df` if this column is not named as 'ISO'

    Parameters
    ----------
    df : pd.DataFrame
        Datafram to be checked
    iso_list : Union[list,tuple], optional
        A list/tuple of ISO codes that should be present in the ISO column, by default ("AUT")

    Returns
    -------
    str
        The name of the column that contains ISO code
    """
    for col in df.columns:
        # Condition 1: all elements `i` in `iso_list`  should be present in ISO column `col`
        # col_lenght = [len(x) for x in df[col] if type(x) is str and check_iso in x]
        col_lenght = [
            len(x) for x in df[col] for i in iso_list if type(x) is str and i in x
        ]
        # Condition 2: All (or at least most of) elements in ISO column should have len==3 (3 digits ISO)
        # NOTE We test this using a quantile=0.99 because sometimes ISO column contains more than 3 digits e.g. `SRB (KOSOVO)`
        if col_lenght and np.quantile(col_lenght, 0.99) == 3:
            return col
    raise ValueError("Cannot find an ISO column in the dataframe")


def fun_drop_duplicates(_df: pd.DataFrame) -> pd.DataFrame:
    """This function checks if there are duplicated index in the dataframe.
    If the values are all the same in the duplicated index, it drops the duplicated index and returns the dataframe.
    Otherwise it will raise an error.
    """
    _df = _df.sort_index()
    for x in range(len(_df[_df.index.duplicated()])):
        max_v = (
            _df.loc[_df[_df.index.duplicated()].index[x]].max() + 1e-6
        )  # 1e-6 to avoid division by zero error
        min_v = (
            _df.loc[_df[_df.index.duplicated()].index[x]].min() + 1e-6
        )  # 1e-6 to avoid division by zero error

        check = np.round(max_v / min_v, 5)
        if check.max() != check.min():
            print(check.max(), check.min())
            filename=CONSTANTS.CURR_RES_DIR("step5")/"df_with_duplicated_index_CHECK.csv"
            _df.to_csv(filename)
            raise ValueError(
                f"Dataframe contains duplicated index with different values. Please check data for {_df[_df.index.duplicated()].index[x]} in this dataframe: {filename}"
            )
    return _df[~_df.index.duplicated()]


def fun_recalculate_existing_var_as_sum(
    df: pd.DataFrame,
    var: str,
    var_list_or_dict: Union[list, dict],
    unit: Union[str, None] = None,
    exogenous_add_value: float = 0,
    # dropna=True,
) -> pd.DataFrame:
    """Recalculate existing variable `var` in the dataframe `df` based on `var_list_or_dict`.
    It uses `fun_create_var_as_sum` function to calculate the variable.

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe
    var : str
        Variable to be re-calculated
    var_list_or_dict : Union[list, dict]
        List of variable (sum of all variables) or dictionary (weighted average across all variables)
    unit : Union[str, None], optional
        Unit of the variable, by default None
    exogenous_add_value: float
        An exogenous value to be added in your calculations
    Returns
    -------
    pd.DataFrame
        Updated dataframe
    """
    df = df.copy(deep=True)
    df = fun_drop_duplicates(df)
    df_temp = fun_create_var_as_sum(
        df,
        "temp",
        var_list_or_dict,
        unit=unit,
        exogenous_add_value=exogenous_add_value,
        # dropna=dropna
    ).xs("temp", level="VARIABLE", drop_level=False)
    if var in df.index.get_level_values("VARIABLE").unique():
        blackl = df.xs(var, level="VARIABLE", drop_level=False).index
        df = df[~df.index.isin(blackl)]
    df_temp = df_temp.rename(index={"temp": var})
    df = pd.concat([df, df_temp], sort=True, axis=0)
    return df


def fun_check_iamc_index(df: pd.DataFrame) -> None:
    """
    This function checks that the index names in your dataframe contain:
    - 'MODEL',
    - 'REGION' or 'ISO'
    - 'TARGET' or 'SCENARIO'
    If one of the above fails it raises a ValuerError.
    """
    index_df = df.index.names

    if "MODEL" not in index_df:
        raise ValueError(
            "Cannot find `MODEL` in the dataframe index names. Please check your dataframe index"
        )

    mychecklist = [
        ["REGION", "ISO"],
        ["SCENARIO", "TARGET"],
    ]
    for m in mychecklist:
        if m[0] not in index_df and m[1] not in index_df:
            raise ValueError(
                f"We could not find {m[0]} nor {m[1]} in the dataframe index. Please check your dataframe index"
            )


def fun_create_var_as_sum(
    mydf: pd.DataFrame,
    new_var_name: str,
    var_list_or_dict: Union[dict, list],
    _level: str = "VARIABLE",
    unit: Union[None, str] = None,
    exogenous_add_value: float = 0,
) -> pd.DataFrame:
    """
    This function creates a new variable named `new_var_name` in the dataframe `mydf`, defined as the sum of elements in `var_list_or_dict`.
    If `var_list_or_dict` is a dictionary, it creates the new variable `new_var_name` as the sum of `var_list_or_dict.keys()` multiplied by `var_list_or_dict.values()`.
    It creates the variable for all index names present in `mydf` (no need to loop over MODEL, SCENARIO, REGION).
    It returns the udpated dataframe.

     Parameters
    ----------
    mydf : pd.DataFrame
        Your dataframe
    new_var_name : str
        New variable name
    var_list_or_dict : Union[list, dict]
        List of variable (sum of all variables) or dictionary (weighted average across all variables)
    _level  Union[str, None], optional
        Level of your Datframe index where you want to perform operation, by default 'VARIABLE'
    unit : Union[str, None], optional
        Unit of the variable, by default None
    exogenous_add_value: float
        Exogenous value to be added, by default 0

    Returns
    -------
    pd.DataFrame
        Updated dataframe
    """
    fun_check_iamc_index(mydf)

    if new_var_name in mydf.reset_index()[_level].unique():
        raise ValueError(
            f"Variable {new_var_name} is already present in the dataframe. You might consider using the functions \n"
            "- `fun_recalculate_existing_var_as_sum`, OR \n "
            "- `fun_create_var_as_sum_only_for_models_where_missing` \n"
        )

    if type(var_list_or_dict) in [set, list]:
        var_list_or_dict = {
            x: 1 for x in var_list_or_dict
        }  # we sum all elements in var_list_or_dict. 1 means (+) sign
        # print("list or set")

    # Step 0: We get all variables in var_dict.keys() and multiply by var_dict.values() (sign of the variable). We store everything in mydf_agg
    mydf_agg = pd.DataFrame()
    missing_vars=[]
    for var, scalar in var_list_or_dict.items():
        try:
            if var in mydf.reset_index()[_level].unique():
                mydf_agg = pd.concat(
                    [mydf_agg, mydf.xs(var, level=_level).mul(scalar)], sort=True
                )
            else:
                missing_vars=missing_vars+[var]
        except Exception as e:
            print(f"`fun_create_var_as_sum` not working for {var}:{e}")
    # Step 1 We define my_new_var as mydf_agg.groupby(mylist).sum().
    # NOTE  mylist contains all indexes except _level
    mylist = list(mydf_agg.index.names)
    sum_var = mydf_agg.groupby(
        mylist,
    ).sum(numeric_only=True)
    if not len(sum_var):
        print(
            f"we could not create the new variable {new_var_name} in the dataframe," 
            f"because these variables were missing in the dataframe: {missing_vars}"
        )
        return mydf
    if len(missing_vars):
        print(f"we created the new variable {new_var_name} in the dataframe although these variables were missing in the dataframe: {missing_vars}")
    sum_var[_level] = new_var_name
    sum_var = sum_var.reset_index().set_index(mydf.index.names)

    if unit is None and "UNIT" in sum_var.index.names:
        _current_unit = sum_var.index.get_level_values("UNIT").unique()
        if len(_current_unit) == 1:
            _current_unit = _current_unit[0]
        else:
            raise ValueError(
                f"we have multiple units in the new dataframe: {_current_unit}, we were unable to change unit "
            )
        sum_var = sum_var.rename({_current_unit: unit})
    elif unit is not None:  # (and 'UNIT' is not an index)
        idxname = sum_var.index.names
        sum_var = sum_var.reset_index()
        idx_sum = [x for x in idxname if x != "UNIT"]
        sum_var = sum_var.groupby(idx_sum).sum(numeric_only=True)
        sum_var["UNIT"] = unit
        sum_var = sum_var.reset_index().set_index(idxname)
    if exogenous_add_value is not None and exogenous_add_value != 0:
        sum_var = sum_var + exogenous_add_value
    return pd.concat([mydf, sum_var], sort=True)


def fun_get_variable_unit_dictionary(df: pd.DataFrame) -> dict:
    """Returns a dictionary with {VARIABLE:UNIT} for every variable available in a given `df`

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe (with `VARIABLE` and `UNIT` in the df.index.names or in df.columns)

    Returns
    -------
    dict
        Dictionary with  {VARIABLE:UNIT}
    """
    return (
        df.reset_index()[["VARIABLE", "UNIT"]]
        .drop_duplicates(keep="first")
        .set_index("VARIABLE")
        .to_dict()["UNIT"]
    )


def rename_tuple(tuple_, dict_):
    """Replaces tuple if present in tuple dict"""
    return dict_[tuple_] if tuple_ in dict_.keys() else tuple_


def fun_rename_index_name(df, dict_mapper):
    df = df.copy(deep=True)
    df.index.names = pd.Index(
        [rename_tuple(tuple_, dict_mapper) for tuple_ in df.index.names]
    )
    return df


def fun_check_iamc_index_and_region(
    region: str, df: pd.DataFrame, rename_iso_as_region: bool = True
) -> pd.DataFrame:
    """Checks if `df.index` complies with IAMc format and if a given region is present in the `df`

    Parameters
    ----------
    region : str
        Region to be checked if present
    df : pd.DataFrame
        Dataframe to be checked
    rename_dict : dict, optional
        Rename ISO index (usually as Region), by default None

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    df = df.copy(deep=True)
    fun_check_iamc_index(df)

    df_renamed = fun_rename_index_name(df, {"ISO": "REGION"})

    # Slice for iso/region
    if region not in df_renamed.reset_index().REGION.unique():
        raise ValueError(f"cannot find {region} in `df`")
    return df_renamed if rename_iso_as_region else df


def fun_dict_level1_vs_list_level2(df: pd.DataFrame, l1: str, l2: str) -> dict:
    """Returns a dictionary with {`l1`:list(`l2`)]}` in your `df`.
    `df` should be provided IAMC format.
    Can be used to create a dictionary with all MODEL/scenarios combinations

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe
    l1: str
        Level1 of your df (this will be the key of your dictionary)
    l2: str
        Level2 of your df (this will be the (list of all l2 for a given l1) value of your dictionary )
    Returns
    -------
    dict
        Available scenarios
    """
    for x in [l1, l2]:
        if x not in df.index.names:
            raise ValueError(f"{x} not found in df.index.names")
    try:
        return {
            model: list(df.xs(model).reset_index()[l2].unique())
            for model in df.reset_index()[l1].unique()
        }
    except:
        return {
        model: list(fun_xs(df, {l1:model}).reset_index()[l2].unique())
        for model in df.reset_index()[l1].unique()
    }

def fun_available_scen(df: pd.DataFrame) -> dict:
    """Returns a dictionary with available scenarios in your dataframe `df`, for each model, .
    `df` should be provided IAMC format

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe

    Returns
    -------
    dict
        Available scenarios
    """
    return fun_dict_level1_vs_list_level2(df, "MODEL", "SCENARIO")


def fun_check_scenarios_with_missing_energy_data(df: pd.DataFrame) -> Union[dict, None]:
    """Checks if all model/scenarios combinations contain energy data (at least one variable with 'EJ/yr' as Unit).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be checked

    Returns
    -------
    Union[dict, None]
        A dictionary of {model:{scenarios}} with missing energy data
    """
    fun_check_iamc_index(df)
    all_missing = {}
    for m in df.reset_index().MODEL.unique():
        all_scen = set(fun_available_scen(df)[m])
        scen_with_energy = set(fun_available_scen(df.xs("EJ/yr", level="UNIT"))[m])
        scen_wo_energy = all_scen - scen_with_energy
        if scen_wo_energy:
            all_missing[m] = scen_wo_energy
    if len(all_missing):
        print(f"Energy data not available for {all_missing}")
        return all_missing
    print(
        "all model/scenarios contain energy data (they all have at least one variable with `EJ/yr`)"
    )
    return None


def fun_highlight_cells(val: int, exp_val: int = 184) -> pd.DataFrame:
    """Highlights cells in a Dataframe if values not equal to `exp_val`.
    We usually use this to highlight anomalies in a datafarme that contains
    number of countries for all model/scenario (we expect 184 countries).

    Parameters
    ----------
    val : any
        Cell value of a dataframe
    exp_val : int, optional
        Expected value, by default 184

    Returns
    -------
    pd.DataFrame
        _description_
    """
    color = "yellow" if val != exp_val else ""
    return f"background-color: {color}"


def fun_check_number_of_countries_by_model_scen_var(
    df_all: pd.DataFrame, var: str, highlight: bool = False
) -> pd.DataFrame:
    """Check number of countries by model, scenarios for a given variable `var` in a given dataframe `df_all`

    Parameters
    ----------
    df_all : pd.DataFrame
        Your dataframe
    var : str
        Variable to be checked
    highlight:bool
        Whether you want to highlight models/scen with less than 184 countries
        (it works with Jupyter Notebooks), by default False

    Returns
    -------
    pd.DataFrame
        A datafrane with the number of countries for a given variable (by model/scenario)
    """
    fun_check_iamc_index(df_all)
    df_summary = pd.DataFrame()
    for m in df_all.reset_index().MODEL.unique():
        df_sel = df_all.xs(m, level="MODEL").xs(var, level="VARIABLE")
        scen = df_sel.reset_index().SCENARIO.unique()[0]
        for scen in df_sel.reset_index().SCENARIO.unique():
            df_summary = pd.concat(
                [
                    df_summary,
                    pd.DataFrame(
                        [
                            {
                                "scenario": scen,
                                var: len(df_sel.xs(scen, level="SCENARIO")),
                            }
                        ],
                        index=[m],
                    ),
                ],
                axis=0,
            )
    if highlight:
        return df_summary.set_index("scenario", append=True).style.applymap(
            fun_highlight_cells, exp_val=184
        )
    return df_summary.set_index("scenario", append=True)


def fun_replace_zero_columns_with_na(
    df_merged: pd.DataFrame, selcols=range(2055, 2105, 10), replace_val=np.nan
) -> pd.DataFrame:
    """Replacing colunmns with zero with na in a given dataframe `df_merged` for selected columns  `selcols`
    if they are (mostly) equal to zero/nan.

    Parameters
    ----------
    df_merged : pd.DataFrame
        Your dataframe
    selcols : _type_, optional
        Selected columns to be replaced with nan if they are (mostly) equal to zero/nan, by default range(2055, 2105, 10)

    Returns
    -------
    pd.DataFrame
        Updated Dataframe
    """
    replacing = []
    for t in selcols:
        print("now replacing zeros for ", t)
        for model in df_merged.reset_index().MODEL.unique():
            dfm = df_merged.loc[model].reset_index()
            cond1 = (
                dfm[t - 5].all() != 0 and dfm[t + 5].all() != 0 and dfm.all()[t] == 0
            )
            # Below checking if number of variables with NA is 2 times greater than variables with value
            cond2 = dfm[t].isnull().sum() > 2 * dfm[t].count()
            if cond1 or cond2:
                myidx = df_merged.xs(model, level="MODEL", drop_level=False).index
                replacing = replacing + [model, t]
                df_merged.loc[myidx, t] = replace_val
    print("Replaced models:", replacing)
    return df_merged


def fun_get_native_vs_downs_results(df: pd.DataFrame) -> dict:
    """Provides a dictionary with a list of downscaled vs native model results.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe (it should include 'FILE' as a column or index)

    Returns
    -------
    dict
        Dictionary with a list of downscaled vs native model results
    """
    df = df.reset_index()
    df_native = df[df.FILE.str.contains("snapshot_all_region")]
    df_down = df[~df.FILE.str.contains("snapshot_all_region")]
    results_type = {"native results": list(df_native.MODEL.unique())}
    results_type["downscaled results"] = list(df_down.MODEL.unique())
    return results_type


def fun_expand_rename_df_by_level(
    df: pd.DataFrame, mapping: dict, level: str = "REGION"
) -> pd.DataFrame:
    """Expands and rename a `df` for a given index "level", based on a `mapping`.
    E.g `fun_expand_rename_df_by_level(df, {"Developed Countries":["AUS","JPN"]}, level="REGION")` will
       1) Create two dataframes where "Developed Countries"  is renamed as: "AUS" and "JPN" (level="REGION")
       Please note that "AUS" and "JPN" will have the same results as in the original "Developed Countries".
       2) return the two concatenated dataframes above

    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe to be expanded/renamed
    mapping : dict
        Your mapping with {"item to be renamed": [renamed as in this list]}
    level : str, optional
        inded Level where to rename/expand your datafarme, by default "REGION"

    Returns
    -------
    pd.DataFrame
        _description_
    """
    fun_check_iamc_index(df)
    df_all = pd.DataFrame()
    for k, v in mapping.items():
        df_renamed = pd.concat(
            [
                df.rename(index={k: c}, level=level).xs(
                    c, level=level, drop_level=False
                )
                for c in v
            ],
            axis=0,
        )
        df_all = pd.concat([df_all, df_renamed], axis=0)  # .drop('hist inventories')
    return df_all


def fun_add_units(
    df: pd.DataFrame,
    df_iam: Optional[pd.DataFrame] = None,
    extra_units_dict: Optional[dict] = None,
    missing_unit="missing",
) -> pd.DataFrame:
    """Adds a new (or updates existing) `UNIT` column, based on units as reported in the `df_iam` and in the `extra_units_dict`.
    NOTE: In case of contraddicting information from `df_iam` and `extra_units_dict`, we use the latter.

    Parameters
    ----------
    df : pd.DataFrame
        Initial Dataframe with downscaled results
    df_iam : Optional[pd.DataFrame], optional
        IAMs dataframe, by default None
    extra_units_dict : Optional[dict], optional
        Dictionary with extra units (e.g. for variable not reported by IAMs), by default None

    Returns
    -------
    pd.DataFrame
        Updated dataframe

    Raises
    ------
    ValueError
        If None of the `df_iam` nor `extra_units_dict` is provided (you need specify at least one of them to read the units associated to each variable).
    """
    if df_iam is None and extra_units_dict is None:
        raise ValueError(
            "Please provide at least a `df_iam` or an `extra_units_dict`. Currently None of them is provided"
        )

    unit_dict = fun_get_variable_unit_dictionary(df_iam) if df_iam is not None else extra_units_dict
    if extra_units_dict:
        unit_dict_reshaped = {x: k for k, v in extra_units_dict.items() for x in v}
        conflicts = set(unit_dict_reshaped.keys()).intersection(set(unit_dict.keys()))
        if conflicts:
            print(
                f"WARNING: We found conflicting units info (among `df_iam` and `extra_units_dict` ) for these variables: {unit_dict}"
                "For these we info from `extra_units_dict`"
            )
        unit_dict.update(unit_dict_reshaped)

    # NOTE unit_dict contains all the variables that we want to keep in df. These are
    if len(df.index.names) > 1:
        df = df.reset_index()
    df["UNIT"] = [unit_dict.get(x, missing_unit) for x in df.VARIABLE]
    df = fun_index_names(df, True, str)

    if "missing" in df.reset_index().UNIT.unique():
        missing_vars = df.xs(missing_unit, level="UNIT").reset_index().VARIABLE.unique()
        print(
            f"WARNING: There are  missing units in the downscaled data for those variables: {missing_vars}"
        )
    return df


def fun_drop_missing_units(
    df: pd.DataFrame, missing_unit: str = "missing"
) -> pd.DataFrame:
    """Drops variables with `missing_unit` in a given dataframe `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Initial dataframe
    missing_unit : str, optional
        How missing units  are reported in the dataframe, by default "missing"

    Returns
    -------
    pd.DataFrame
        Dataframe without variables with missing unit
    """
    vars_to_drop = df.xs(missing_unit, level="UNIT").reset_index().VARIABLE.unique()
    txt = "These variables will be dropped"
    print(f"\n {txt}: {[x for x in vars_to_drop if 'cal Difference' not in x]}")
    return df.drop(missing_unit, level="UNIT")


def fun_read_csv_or_excel(
    file: str,
    models: list,
    folder: os.PathLike = CONSTANTS.CURR_RES_DIR("step5"),
) -> pd.DataFrame:
    """Reads csv or excel files results from a given `step`, for a list of `models` and a given `file` name.

    Parameters
    ----------
    file : str
        Final name (suffix after the model name, including file extension e.g. "2023_07_20.csv")
    models : list
        List of model results to read
    step : str, optional
        Read results from a given step, by default 'step5'

    Returns
    -------
    pd.DataFrame
        Dataframe with the results
    """

    CURR_RES_DIR = folder
    if models is not None:
        if os.path.splitext(file)[1] == ".xlsx":
            df = pd.concat(
                [
                    pd.read_excel(
                        CURR_RES_DIR / f"{m}_{file}",
                        engine="openpyxl",
                        sheet_name="data",
                    )
                    for m in models
                ],
                axis=0,
            )
        else:
            if len(models) == 1 and models[0] in file:
                df = fun_read_csv({"df": CURR_RES_DIR / f"{file}"}, True, int)["df"]
            else:
                df = pd.concat(
                    [
                        fun_read_csv({"df": CURR_RES_DIR / f"{m}_{file}"}, True, int)[
                            "df"
                        ]
                        for m in models
                    ],
                    axis=0,
                )
    else:
        if os.path.splitext(file)[1] == ".xlsx":
            df = pd.read_excel(
                CURR_RES_DIR / file, engine="openpyxl", sheet_name="data"
            )
        else:
            df = fun_read_csv({"df": CURR_RES_DIR / file}, True, int)["df"]

    if "FILE" not in df.reset_index().columns:
        df["FILE"] = file
    # NOTE we need both lines below. We cannot use int directly as this would lead to error:
    # ValueError: invalid literal for int() with base 10: '2010.0'
    df = fun_index_names(df, True, float)
    df.columns= [int(x) for x in df.columns]
    return df


def fun_xs_enhanced(
    df: pd.DataFrame, d: dict, exclude_vars: bool = False, str_contains: bool = False
) -> pd.DataFrame:
    """
    Slices a DataFrame based on a dictionary of index values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be sliced.
    d : dict
        Dictionary where keys are the levels of the DataFrame's MultiIndex and values are labels to match.
    exclude_vars: bool, optional
        Whether to exclude/remove rows based on the dictionary, by default False.
    str_contains: bool, optional
        Whether to use substring matching for strings in a case-insensitive manner, by default False.

    Returns
    -------
    pd.DataFrame
        Sliced DataFrame.
    """

    # THIS FUNCTION DOES NOT SEEM TO WORK FOR `DIAGNOSTICS` VARIABLES!!!!!!
    for k, v in d.items():
        if not isinstance(v, list):
            v = [v]
        if k in df.index.names:
            condition = (
                ~df.index.get_level_values(k).isin(v)
                if exclude_vars
                else df.index.get_level_values(k).isin(v)
            )
            if str_contains:
                condition = condition | df.index.get_level_values(
                    k
                ).str.lower().str.contains("|".join(v).lower())
        else:
            # Assuming k is a column name
            condition = (
                ~df[k].str.lower().isin(v)
                if exclude_vars
                else df[k].str.lower().isin(v)
            )
        df = df[condition]
    return df


def fun_clip_df_by_dict(
    df: pd.DataFrame, clip_dict: dict, level_name: str, lower: bool = True, upper: bool=False
) -> pd.DataFrame:
    """Clips a Pandas DataFrame based on a dictionary specifying clipping values for a specific index level.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be clipped.
    clip_dict : dict
        A dictionary where keys are index values and values are the clipping values.
    level_name : str
        The name of the index level to apply clipping.
    lower : bool, optional
        Wheter you want to clip the lower value, by default True
    upper : bool, optional
        Wheter you want to clip the upper value, by default False

    Returns
    -------
    pd.DataFrame()
        The clipped DataFrame
    """
    df = df.copy(deep=True)
    for index_value, clip_value in clip_dict.items():
        mask = df.index.get_level_values(level_name) == index_value
        if lower:
            df.loc[mask, :] = df.loc[mask, :].clip(lower=clip_value)
        if upper:
            df.loc[mask, :] = df.loc[mask, :].clip(upper=clip_value)
    return df



def fun_phase_out_in_dates(
    df: pd.DataFrame,
    var: str,
    c: str,
    scenario: str,
    phase_out: bool = True,
    threshold: float = 0,
) -> pd.DataFrame:
    """Identify the date of phase-out or phase-in for a variable of interest (in a given country `c` and `scenario`).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a MultiIndex (MODEL, VARIABLE, REGION, SCENARIO, TIME).
    var : str
        Selected variable
    c : str
        Selected country (ISO3 code)
    scenario : str
        Selected scenario
    phase_out : bool, optional
         If True, identify the date of phase-out; otherwise, identify the date of phase-in, by default True
    threshold: float
        Threshold (reference) value to calculate phase in/out dates

    Returns
    -------
    pd.DataFrame
        DataFrame with phase-out (or phase-in) dates for a selected variable in a given country `c`.
    """
    df = df.copy(deep=True)
    fun_check_iamc_index(df)  # Check if dataframe complies with IAMC format
    dfreset = df.reset_index()

    # Extract the relevant subset of the DataFrame
    d_sel = {"VARIABLE": var, "REGION": c, "SCENARIO": scenario}
    for k, v in d_sel.items():
        if v not in dfreset[k].unique():
            raise ValueError(f"Cannot find `{v}` {k.lower()} in the dataframe")
    dfs = df.xs(tuple(d_sel.values()), level=tuple(d_sel.keys()), drop_level=False)
    dfs=dfs.dropna(axis=1)

    # Filter based on phase_out parameter
    # dfs=dfs.replace(0, np.nan).dropna(axis=1) # Replace zeroes with np.nan
    dfs = (
        dfs[dfs > threshold].dropna(how="all", axis=1)
        if phase_out
        else dfs[dfs <= threshold].dropna(how="all", axis=1)
    ).dropna(how="all")
    # selcol=-1 if phase_out else 0

    models = dfs.reset_index().MODEL.unique()
    if dfs.empty:
        # If subset_df is empty, return a DataFrame with NaN for each model
        # return pd.DataFrame({m: np.nan for m in models}, index=[c])
        return pd.DataFrame()

    # Create a dictionary with models and their corresponding phase-out/in dates
    d={}
    d_sel2=d_sel
    for m in models:
        d_sel2['MODEL']=m
        if len(fun_xs(dfs, d_sel2)):
            all_columns=dfs.xs(tuple(d_sel.values()), level=tuple(d_sel.keys()), drop_level=False).dropna(axis=1).columns
            all_phase_out_years=[x for x in range(2010,2101) if x not in  all_columns]
            # if len(list(set(range(2010,2021)).intersection(set(all_phase_out_years)))):
            #     all_phase_out_years=[min(set(range(2010,2021)).intersection(set(all_phase_out_years)))]
            if 2020 in all_phase_out_years:
                all_phase_out_years=[2020]
            all_phase_out_years=all_phase_out_years or [(all_columns[-1])]
            first_year_of_phase_out=all_phase_out_years[0]  
            d[m] =first_year_of_phase_out
        else:
            d[m]=np.nan
    return pd.DataFrame(d, index=[c])
