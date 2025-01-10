import pandas as pd
from typing import Callable, Dict, List, Optional, Union, List, Dict
from downscaler.utils import fun_check_iamc_index, fun_index_names
from downscaler.utils_pandas import (
    fun_xs,
    fun_index_names,
    fun_check_df_contains_single_region_country,
    fun_find_common_variables_in_multiple_dataframes,  # returns all elements in common
    fun_check_missing_elements_in_dataframe,  # raises error if a given element is not present
    fun_drop_duplicates
)


def fun_downscale_using_same_proportions_across_regions(
    var: str,
    baseyear: Union[str, int],
    df_dependent_region: pd.DataFrame,
    df_world_region: pd.DataFrame,
) -> pd.DataFrame:
    """This function performs a simple downscaling by keeping the region/world shares as in the `baseyear`, for a given variable `var`.

    This function requires:
     -  a dataframe `df_dependent_region` with baseyear data for a single regional (or iso).
     -  a dataframe `df_world_region` with future data for a world (or any other) region.

    Both dataframe should contain the selected `var`.

    It returns a dataframe with downscaled regional results for variable `var`
    (by keeping constant base year shares across the two regions).

    Parameters
    ----------
    var : str
        Variable to be downscaled e.g. 'Final Energy'
    baseyear : Union[str, int]
        Base year column e.g. '2019'
    df_dependent_region : pd.DataFrame
        Dataframe with regional data at the base year
    df_world_region : pd.DataFrame
        Dataframe with future data at the World (or any other region/iso) level.

    Returns
    -------
    pd.DataFrame
        Downscaled results at the regional level

    Raises
    ------
    ValueError
        If `var` is missing in the dataframes
    """
    
    # Copy dataframes
    df_dependent_region = df_dependent_region.copy(deep=True)
    df_world_region = df_world_region.copy(deep=True)
    
    # Use strings in columns (temporarily)
    df_dependent_region.columns=[str(x) for x in df_dependent_region.columns]
    df_world_region.columns=[str(x) for x in df_world_region.columns]
    baseyear=str(baseyear)

    res = {
        "region": df_dependent_region,
        "world": df_world_region,
    }
    reg = {}
    for x in ["region", "world"]:
        reg[x] = fun_check_df_contains_single_region_country(res[x], df_name=x)
        fun_check_iamc_index(res[x])

    check_dict = {"VARIABLE": var, "columns": baseyear}
    for df in [res["region"], res["world"]]:
        fun_check_missing_elements_in_dataframe(check_dict, df)

    # Find common models/scenarios
    common = True
    for x in ["MODEL", "SCENARIO"]:
        if len(fun_find_common_variables_in_multiple_dataframes(res, x))==0:
            common = False
    

    res["region"] = res["region"].xs(reg["region"], level="REGION")
    res["region"] = fun_xs(res["region"], {"VARIABLE": var})
    if not common:
        print(
        "We found no MODEL/SCENARIO in common between `df_dependent_region` and `df_world_region`."
        "Hence, we assume `df_dependent_region` contains historical data, and we drop MODEL/SCENARIO columns."
        )
        res["region"] = res["region"].droplevel(["MODEL", "SCENARIO"])
        res['region']=fun_drop_duplicates(res['region'])
    
    
    # # Share at the base year
    # myidx=res["world"].index.names
    # share_option1 = res["region"].reset_index().set_index(myidx)[baseyear] / res["world"].reset_index().set_index(myidx)[baseyear]
    
    share_option1 = res["region"][baseyear] / res["world"][baseyear]
    # SAME share across all time perios
    share_option1 = pd.DataFrame({x: share_option1 for x in df_world_region.columns})

    if common:
        share_option1 = share_option1.reset_index().set_index(
            df_world_region.index.names
        )

    # Use integer in columns
    df_dependent_region= fun_index_names(df_dependent_region, True, int)
    df_world_region= fun_index_names(df_world_region, True, int)
    share_option1= fun_index_names(share_option1, True, int)
    
    # Get common index
    common_idx= set(df_world_region.index.names).intersection(set(share_option1.index.names))
    df_world_region = df_world_region.reset_index().set_index(list(common_idx))
    share_option1 = share_option1.reset_index().set_index(list(common_idx))
    
    # Drop remaning columns with strings (e.g. 'FILE')
    share_option1 = share_option1.drop([x for x in share_option1.columns if isinstance(x, str)], axis=1)
    df_world_region = df_world_region.drop([x for x in df_world_region.columns if isinstance(x, str)], axis=1)
    df_dependent_region = df_dependent_region.drop([x for x in df_dependent_region.columns if isinstance(x, str)], axis=1)


    option1 = df_world_region * share_option1
    option1 = option1.rename({reg["world"]: reg["region"]})
    return fun_index_names(fun_xs(option1, {"VARIABLE": var}))


def fun_downscale_using_same_proportions_across_variables(
    df: pd.DataFrame,
    var_new: str,
    var_proxi: str,
    multiplier_value: float,
    unit: Union[str, None] = None,
) -> pd.DataFrame:
    """This function creates a new variable `var_new`, by multiplying `var_proxi` by a `multiplier_value`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with `var_proxi`
    var_new : str
        Name of the new variable
    var_proxi : str
        Reference Variable (to be multiplied)
    multiplier_value : float
        Multiplier
    unit : Union[str, None], optional
        Unit of the new variable (if None will keep the same unit as `var_proxi`), by default None

    Returns
    -------
    pd.DataFrame
        DataFrame with `var_new`
    """
    fun_check_iamc_index(df)
    df = df.copy(deep=True)
    idx_col = df.index.names
    df_mul = (
        fun_index_names(df.xs(var_proxi, level="VARIABLE"), True) * multiplier_value
    )
    df_mul = df_mul.reset_index()
    df_mul["VARIABLE"] = var_new
    if unit is not None:
        df_mul["UNIT"] = unit
    return fun_index_names(df_mul, True, None, sel_idx=idx_col)
