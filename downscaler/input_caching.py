import itertools
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from downscaler import CONSTANTS
from downscaler.fit_funcs import func_dict
from downscaler.fixtures import fen_sectors, gdp
from downscaler.utils import (
    InputFile,
    MemFunc,
    convert_to_list,
    fun_country2region,
    fun_country_map,
    fun_read_df_iam_all,
    fun_read_gdp_gdpcap,
    fun_read_gdpcap,
    make_optionally_cacheable,
    match_any_with_wildcard,
    unique,
)

# TODO:
# * Transform into yaml files and read from disk
# NOTE: For now we do not get data for Cote d'Ivoire, the reason is most likely
# the accent circonflex and something with the encoding. Traced it back to
# convert_historic_data

iea_gases = ["Natural gas", "Refinery gas", "Ethane", "Biogases"]

iea_liquids = [
    "Liquefied petroleum gases (LPG)",
    "Patent fuel",
    "Crude/NGL/feedstocks (if no detail)",
    "Crude oil",
    "Natural gas liquids",
    "Refinery feedstocks",
    "Additives/blending components",
    "Other hydrocarbons",
    "Motor gasoline excl. biofuels",
    "Aviation gasoline",
    "Gasoline type jet fuel",
    "Kerosene type jet fuel excl. biofuels",
    "Other kerosene",
    "Gas/diesel oil excl. biofuels",
    "Fuel oil",
    "Naphtha",
    "White spirit & SBP",
    "Lubricants",
    "Bitumen",
    "Paraffin waxes",
    "Petroleum coke",
    "Other oil products",
    "Biogasoline",
    "Biodiesels",
    "Bio jet kerosene",
    "Other liquid biofuels",
]

iea_solids = [
    "Gas coke",
    "Hard coal (if no detail)",
    "Brown coal (if no detail)",
    "Anthracite",
    "Coking coal",
    "Other bituminous coal",
    "Sub-bituminous coal",
    "Lignite",
    "Coke oven coke",
    "Coal tar",
    "BKB",
    "Peat",
    "Peat products",
    "Oil shale and oil sands",
    "Primary solid biofuels",
    "Non-specified primary biofuels and waste",
    "Charcoal",
]

iea_liquids_gases = iea_liquids + iea_gases

iea_flow_dict = {
    ## Data stucture  {sectors:['IEA FLOW', 'IEA PRODUCT','UNIT', scale (graph purposes)]}
    # gdp: ["", "", "Billion USD PPP"],  ## We just need this for the UNIT value
    ## PLEASE DO NOT CHANGE THE 'Final Energy' LINE (BELOW). It should not be rewritten as 'list of list
    "Final Energy": [
        "Total final consumption",
        "Total",
        "ktoe",
        1e3,
    ],  ## first entry is FLOW, second, PRODUCT of IEA database
    "Final Energy|Industry": ["Industry", "Total", "ktoe", 1e2],
    "Final Energy|Transportation": ["Transport", "Total", "ktoe", 1e2],
    "Final Energy|Residential and Commercial": [
        ["Residential", "Commercial and public services"],
        ["Total"],
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Liquids&Gases": [
        ["Total final consumption"],
        iea_liquids_gases,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Transportation|Liquids&Gases": [
        ["Transport"],
        iea_liquids_gases,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Residential and Commercial|Liquids&Gases": [
        ["Residential", "Commercial and public services"],
        iea_liquids_gases,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Industry|Liquids&Gases": [
        ["Industry"],
        iea_liquids_gases,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Solids": [
        ["Total final consumption"],
        iea_solids,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Transportation|Solids": [
        ["Transport"],
        iea_solids,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Residential and Commercial|Solids": [
        ["Residential", "Commercial and public services"],
        iea_solids,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Industry|Solids": [["Industry"], iea_solids, ["ktoe"], 1e2],
    "Final Energy|Electricity": [
        "Total final consumption",
        "Electricity",
        "ktoe",
        1e2,
    ],
    "Final Energy|Industry|Electricity": [
        "Industry",
        "Electricity",
        "ktoe",
        1e2,
    ],
    "Final Energy|Transportation|Electricity": [
        "Transport",
        "Electricity",
        "ktoe",
        1e2,
    ],
    "Final Energy|Residential and Commercial|Electricity": [
        ["Residential", "Commercial and public services"],
        ["Electricity"],
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Industry|Heat": [["Industry"], ["Heat"], ["ktoe"], 1e2],
    "Final Energy|Residential and Commercial|Heat": [
        ["Residential", "Commercial and public services"],
        ["Heat"],
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Liquids": [
        ["Total final consumption"],
        iea_liquids,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Transportation|Liquids": [
        ["Transport"],
        iea_liquids,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Residential and Commercial|Liquids": [
        ["Residential", "Commercial and public services"],
        iea_liquids,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Industry|Liquids": [
        ["Industry"],
        iea_liquids,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Gases": [
        ["Total final consumption"],
        iea_gases,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Transportation|Gases": [
        ["Transport"],
        iea_gases,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Residential and Commercial|Gases": [
        ["Residential", "Commercial and public services"],
        iea_gases,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Industry|Gases": [["Industry"], iea_gases, ["ktoe"], 1e2],
    ## we need the below later on for downscaling final energy solids by sectors and fuels 2020_11_18
    "Final Energy|Heat": [
        ["Heat output"],
        unique(iea_gases + iea_liquids + iea_solids)
        + ["Heat output from non-specified combustible fuels"],
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Residential and Commercial|Heat": [
        ["Residential", "Commercial and public services"],
        "Heat",
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Industry|Heat": [["Industry"], "Heat", ["ktoe"], 1e2],
}

dict_y_den = {
    "Final Energy": gdp,
    "Final Energy|Industry": "Final Energy",
    "Final Energy|Residential and Commercial": "Final Energy",
    "Final Energy|Transportation": "Final Energy",
    "Final Energy|Liquids": "Final Energy",
    "Final Energy|Transportation|Liquids": "Final Energy|Liquids",
    "Final Energy|Residential and Commercial|Liquids": "Final Energy|Liquids",
    "Final Energy|Industry|Liquids": "Final Energy|Liquids",
    "Final Energy|Gases": "Final Energy",
    "Final Energy|Transportation|Gases": "Final Energy|Gases",
    "Final Energy|Residential and Commercial|Gases": "Final Energy|Gases",
    "Final Energy|Industry|Gases": "Final Energy|Gases",
    "Final Energy|Solids": "Final Energy",
    "Final Energy|Transportation|Solids": "Final Energy|Solids",
    "Final Energy|Residential and Commercial|Solids": "Final Energy|Solids",
    "Final Energy|Industry|Solids": "Final Energy|Solids",
    "Final Energy|Electricity": "Final Energy",
    "Final Energy|Industry|Electricity": "Final Energy|Electricity",
    "Final Energy|Residential and Commercial|Electricity": "Final Energy|Electricity",
    "Final Energy|Transportation|Electricity": "Final Energy|Electricity",
}


# @make_optionally_cacheable
def get_selection_dict(
    iam_file: InputFile,
    model: Union[str, list] = "*",
    region: Union[str, list] = "*",
    target: Union[str, list] = "*",
    variable: Union[str, list] = "*",
    convert_to_list: MemFunc = MemFunc(convert_to_list),
    fun_read_df_iam_all: MemFunc = MemFunc(fun_read_df_iam_all),
    match_any_with_wildcard: MemFunc = MemFunc(match_any_with_wildcard),
) -> dict:
    """Match a list of patterns for model, region, target and variable to what is available
    in iam_file.
    Pattern matching is done using fnmatch from the python standard library (see
    [documentation](https://docs.python.org/3/library/fnmatch.html) for details).
    Regular expressions are **not** supported. Default for all patterns is "*", i.e.
    matches everything.
    Parameters
    ----------
    iam_file : InputFile
        IAM results file
    model : Union[str, list], optional
        pattern(s) for models, by default "*"
    region : Union[str, list], optional
        pattern(s) for regions, by default "*"
    target : Union[str, list], optional
        pattern(s) for targets, by default "*"
    variable : Union[str, list], optional
        pattern(s) for variables, by default "*"
    Returns
    -------
    dict
        Dictionary of all matched patterns. As regions, targets and variables can be
        model specific, the dictionary has the following structure:
        {'model1': {'regions': [*list of regions for model1*],
                    'targets': [*list of targets for model1*],
                    'variables': [*list of variables for model1*]},
         'model2': {'regions': [...], 'targets': [...], 'variables': [...]},
         ...
        }
    Notes
    -----
    * **Not** cached because the wildcard patterns make for bad literal comparison
    """

    # NOTE: Currently specific to step 1, would need to be generalized for step2
    #       because for variables we start with the final energy sectors.
    # NOTE: The error reporting might be improved by showing the user a list of
    #       available models, scenarios, etc...

    model_list = convert_to_list(model)
    region_list = convert_to_list(region)
    target_list = convert_to_list(target)
    variable_list = convert_to_list(variable)

    iam_data = fun_read_df_iam_all(iam_file)
    models = [
        m
        for m in iam_data.MODEL.unique()
        if match_any_with_wildcard(m, model_list) and m != "Reference"
    ]
    if not models:
        raise ValueError(f"No models found for pattern {model_list}!")
    del iam_data

    selection_dict = {}
    for m in models:
        df_slice_model = fun_read_df_iam_all(iam_file, model=m)
        regions = [
            r
            for r in df_slice_model.REGION.unique()
            if type(r) is str
            if match_any_with_wildcard(r, region_list)
        ]
        error_msg = (
            f"No regions found for pattern {region_list}!" if not regions else ""
        )

        targets = [
            t
            for t in df_slice_model.SCENARIO.unique()
            if match_any_with_wildcard(t, target_list)
        ]
        error_msg = (
            error_msg + f"\nNo targets found for pattern {target_list}!"
            if not targets
            else error_msg
        )

        # We only need the fen_sectors so we start with that list instead of iterating # over the entire variable space of the data frame
        variables = [
            v
            for v in fen_sectors
            if v in df_slice_model.VARIABLE.unique()
            and match_any_with_wildcard(v, variable_list)
        ]
        error_msg = (
            error_msg + f"\nNo variables found for pattern {variable_list}!"
            if not variables
            else error_msg
        )

        if error_msg:
            raise ValueError(error_msg)

        selection_dict[m] = {
            "regions": regions,
            "targets": targets,
            "variables": variables,
        }
    return selection_dict


# @make_optionally_cacheable
def get_parallel_input_dict(
    selection_dict,
    iam_file: InputFile,
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    country_mapping_file: InputFile,
    pyam_mapping_file: InputFile,
    ssp_file: InputFile,
    ref_target_optimal_country_list,
    ref_target_mapping,
    ssp_scenario: str = "SSP2",
    ssp_model: str = "OECD Env-Growth",
    func_type_other_sectors: str = "log-log",
    iam_base_year: int = 2010,
    max_tc: int = 2200,
) -> List[Dict]:
    """Constructs a list of input dictionaries for def run_step1_m_tar_r_v"""
    return [
        {
            "model": x[0],
            "target": x[1],
            "region": x[2],
            "variable": x[3],
            "iam_file": iam_file,
            "iea_extended_bal_file": iea_extended_bal_file,
            "historic_data_file": historic_data_file,
            "country_mapping_file": country_mapping_file,
            "pyam_mapping_file": pyam_mapping_file,
            "ssp_file": ssp_file,
            "ref_target_optimal_country_list": ref_target_optimal_country_list,
            "ref_target_mapping": ref_target_mapping,
            "ssp_scenario": ssp_scenario,
            "ssp_model": ssp_model,
            "func_type_other_sectors": func_type_other_sectors,
            "iam_base_year": iam_base_year,
            "max_tc": max_tc,
        }
        for m, spec in selection_dict.items()
        for x in itertools.product(
            [m], spec["targets"], spec["regions"], spec["variables"]
        )
    ]


# @make_optionally_cacheable
def get_available_data(
    iam_file: InputFile, fun_read_df_iam_all: MemFunc = MemFunc(fun_read_df_iam_all)
) -> Dict[str, Dict[str, List[str]]]:
    iam_data = fun_read_df_iam_all(iam_file)
    models = iam_data["MODEL"].unique().tolist()
    del iam_data

    available_data_dict = {}

    for m in models:
        if m == "Reference":
            continue
        df_slice_model = fun_read_df_iam_all(iam_file, model=m)
        available_data_dict[m] = {
            "regions": df_slice_model.REGION.unique().tolist(),
            "variables": df_slice_model.VARIABLE.unique().tolist(),
            "targets": df_slice_model.SCENARIO.unique().tolist(),
        }
    return available_data_dict


## IAM (LONG-TERM) FITS ##


# @make_optionally_cacheable
def get_single_longterm_fit(
    iam_file: InputFile,
    m: str,
    r: str,
    t: str,
    v: str,
    func_type: str = "log-log",
    fun_read_df_iam_all: MemFunc = MemFunc(fun_read_df_iam_all),
) -> dict:
    """Calculate a longterm fit for a **single** combination of model (m),
    region (r), scenario (t) and variable (v).

    The data for the fit are IAM results and the fit is referred to as
    'longterm'. The values for m, r, t and v have to be exact as this function
    features no wildcard pattern matching logic.

    Parameters
    ----------
    iam_file : InputFile
        IAM data file as an InputFile
    m : str
        Model to select from the iam_file
    r : str
        Region to select from the iam_file
    t : str
        Target to select from the iam_file
    v : str
        Variable to select from the iam_file
    fun_read_df_iam_all: MemFunc
        MemFunc object for caching purposes see the MemFunc docstring for
        details, default fun_read_df_iam_all
    Returns
    -------
    dict
        A dictionary containing the results of the fit (return value of
        scipy.stats.linregress, see
        [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html)
        for details ) as well as information on the model, scenario, region,
        variable combination. The slope and intercept are called 'beta' and
        'alpha' respectively.

    Notes
    -----
    For further optimization we could pass the dataframe as an input parameter
    and not read it every time. We could also set time as an index once and not
    every time.
    """

    iam_data = fun_read_df_iam_all(iam_file, model=m)
    num = v
    den = dict_y_den[v]
    vars = list(set((num, den, gdp, "Population")))
    df_sliced = iam_data[
        (iam_data.REGION == r)
        & (iam_data.SCENARIO == t)
        & (iam_data.VARIABLE.isin(vars))
    ]
    # dropping nan AND 0.
    na_time = df_sliced[
        (df_sliced.VALUE.isna()) | (df_sliced.VALUE == 0.0)
    ].TIME.unique()
    df_sliced = df_sliced[~df_sliced.TIME.isin(na_time)]
    df_sliced.set_index("TIME", inplace=True)
    df_xy = pd.DataFrame()
    df_xy["Y"] = (
        df_sliced[df_sliced.VARIABLE == num].VALUE
        / df_sliced[df_sliced.VARIABLE == den].VALUE
    )
    df_xy["X"] = (
        df_sliced[df_sliced.VARIABLE == gdp].VALUE
        / df_sliced[df_sliced.VARIABLE == "Population"].VALUE
    )
    info_dict = {
        "MODEL": m,
        "SCENARIO": t,
        "REGION": r,
        "VARIABLE": v,
        "fit_func": func_dict[func_type](),
    }
    info_dict["fit_func"].fit(df_xy["X"], df_xy["Y"])
    info_dict["r_squared"] = info_dict["fit_func"].r_squared
    info_dict["beta"] = info_dict["fit_func"].beta
    # info_dict["alpha"] = info_dict["fit_func"].alpha
    return info_dict


def calculate_longterm_fit(
    iam_file: InputFile,
    model: Union[str, list] = "*",
    region: Union[str, list] = "*",
    target: Union[str, list] = "*",
    variable: Union[str, list] = "*",
    get_selection_dict: MemFunc = MemFunc(get_selection_dict),
    get_single_longterm_fit: MemFunc = MemFunc(get_single_longterm_fit),
) -> pd.DataFrame:
    """Calculate the long term fits for a combination of model, region, target and variable
    base on IAM results.
    Parameters
    ----------
    iam_file : InputFile
        IAM results file
    model : Union[str, list], optional
        pattern(s) for models, by default "*"
    region : Union[str, list], optional
        pattern(s) for regions, by default "*"
    target : Union[str, list], optional
        pattern(s) for targets, by default "*"
    variable : Union[str, list], optional
        pattern(s) for variables, by default "*"
    Returns
    -------
    pd.DataFrame
        Results of the long term fits as a pandas data frame.
    Notes
    -----
    This is a prime example of an embarrassingly parallel problem and should be
    parallelized using joblib.Memory.
    """

    selection_dict = get_selection_dict(iam_file, model, region, target, variable)
    return pd.DataFrame(
        get_single_longterm_fit(iam_file, sel_model, r, t, v)
        for sel_model, spec in selection_dict.items()
        for r in spec["regions"]
        for t in spec["targets"]
        for v in spec["variables"]
    )


## HISTORIC (SHORT-TERM) FITS ##

# NOTE: Right now the data is in ktoe, maybe for the future we want to convert to EJ
# this decorator makes the function cachable, for details refer to:
# https://joblib.readthedocs.io/en/latest/generated/joblib.Memory.html and
# https://joblib.readthedocs.io/en/latest/auto_examples/memory_basic_usage.html


# @make_optionally_cacheable
def _convert_historic_data(
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    region: Optional[str] = None,
    model: Optional[str] = None,
    country_mapping_file: Optional[InputFile] = None,
    pyam_mapping_file: Optional[InputFile] = None,
    fun_country_map: MemFunc = MemFunc(fun_country_map),
) -> pd.DataFrame:
    """Convert historic data from IEA format to the downscaler internal format.

    If values for model and region are provided the dataset is sliced for that
    particular region. Otherwise this function returns the entire historic data
    set.

    Parameters
    ----------
    iea_extended_bal_file : InputFile
        Detailed IEA energy balances file with information on all sectors
    historic_data_file : InputFile
        GDP and population data from the IEA
    region: Optional[str]
        Region to be sliced, default None
    model: Optional[str]
        Model to be used to resolve the specified region, default None
    country_mapping_file: Optional[InputFile]
        Used to resolve the combination of model and region, default None
    pyam_mapping_file: Optional[InputFile]
        Used to resolve the combination of model and region, default None
    fun_country_map: MemFunc
        Function used to provide country region mapping. This parameter is only
        used so that the function is cached. Do not change the default!, default
        MemFunc(fun_country_map)


    Returns
    -------
    pd.DataFrame
        Resulting data in the following format:
        index: TIME, ISO
        columns: Final Energy, Final Energy|Transportation, etc... (as given by the
        keys in iea_flow_dict)
        units: for now we assume the following units:
            * Population: million
            * GDP|PPP: billion US Dollar (2005)
            * All sector values from the historic IEA data: ktoe (here we convert to EJ
            for the downscaling)
    """

    region_slicing_info = (model, region, country_mapping_file, pyam_mapping_file)

    # Check that if we want to perform region slicing that we have all of the
    # required specifications i.e. model, region, country_mapping_file,
    # pyam_mapping_file
    if (
        sum(x is not None for x in region_slicing_info) != len(region_slicing_info)
        and region is not None
    ):

        raise ValueError(
            "If region slicing is desired each of the following values must be provided: model, region, country_mapping_file, pyam_mapping_file"
        )

    # Read the IEA extended energy balances file and set the FLOW and PRODUCT columns
    # as index
    iea_extended_df = pd.read_csv(
        iea_extended_bal_file.file, index_col=["FLOW", "PRODUCT"]
    )
    # read the historic data file and keep only the relevant columns
    historic_df = pd.read_csv(historic_data_file.file)
    historic_df = historic_df[["ISO", "TIME", "POPULATION", "GDP|MER", "GDP|PPP"]]
    # Drop all rows where we don't have an ISO code
    iea_extended_df = iea_extended_df[iea_extended_df["ISO"].notnull()]

    # If desired perform region slicing
    if sum(x is not None for x in region_slicing_info) == len(region_slicing_info):
        map_df = fun_country_map(
            model, country_mapping_file.file, pyam_mapping_file.file
        )
        region_list = list(
            map_df[map_df["REGION"] == f"{region.split('|')[1]}r"]["ISO"]
        )
        iea_extended_df = iea_extended_df[iea_extended_df["ISO"].isin(region_list)]
        historic_df = historic_df[historic_df["ISO"].isin(region_list)]
        del map_df

    # Create a new, initially empty column called 'SECTOR'
    iea_extended_df["SECTOR"] = ""
    # Iterate over iea_flow_dict
    for iamc_var, iea_mapping in iea_flow_dict.items():
        if not isinstance(iea_mapping[0], list):
            iea_mapping[0] = [iea_mapping[0]]
        if not isinstance(iea_mapping[1], list):
            iea_mapping[1] = [iea_mapping[1]]
        for flow in iea_mapping[0]:
            for product in iea_mapping[1]:
                if (flow, product) in iea_extended_df.index.to_list():
                    iea_extended_df.loc[(flow, product), "SECTOR"] = iamc_var
    iea_extended_df = iea_extended_df[iea_extended_df["SECTOR"] != ""]

    melted_df = pd.melt(
        iea_extended_df,
        id_vars=["ISO", "SECTOR"],
        value_vars=[x for x in iea_extended_df.columns if x.isdigit()],
        var_name="TIME",
        value_name="VALUE",
    )
    melted_df["TIME"] = melted_df["TIME"].astype(int)
    melted_df = melted_df.set_index(["TIME", "ISO"])
    melted_df["VALUE"] = pd.to_numeric(melted_df["VALUE"], errors="coerce")

    historic_df = historic_df.set_index(["TIME", "ISO"])

    for sector in iea_flow_dict.keys():
        if sector in melted_df["SECTOR"].unique():
            sector_df = (
                melted_df[melted_df["SECTOR"] == sector]
                .fillna(0)
                .groupby(["TIME", "ISO"])
                .sum()
            )
            sector_df.rename(columns={"VALUE": sector}, inplace=True)
            historic_df = historic_df.join(sector_df)
            # If line above does not work, try use the line below
            # historic_df =pd.concat([historic_df, sector_df])
    # determine if we have the data given in TJ or KTOE we are doing this by using
    # the value for final energy for the USA in the year 1972
    # we check if we are within +- 20% of the reference values for both ktoe and tj
    # if we are in range of neither we raise an error

    usa_ref_ktoe = 1315460.0
    ktoe_to_ej = 0.041868 * 1e-3
    usa_ref_tj = usa_ref_ktoe * 0.041868 * 1e3

    # NOTE: For the future we may want to ask the user if the predicted unit
    # and the resulting conversion factor is correct, i.e. a dialogue like this:
    # We believe the unit to be ktoe, is this correct [Y/N]?
    # If yes we apply the correction factor if no, provide the appropriate number

    # for now we just assume we have ktoe and convert to EJ
    historic_df[[x for x in iea_flow_dict.keys() if x in historic_df.columns]] *= (
        0.041868 / 1e3
    )

    return historic_df


# @make_optionally_cacheable
def _calculate_historic_indicators(
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    region: Optional[str] = None,
    model: Optional[str] = None,
    country_mapping_file: Optional[InputFile] = None,
    pyam_mapping_file: Optional[InputFile] = None,
    _convert_historic_data: MemFunc = MemFunc(_convert_historic_data),
) -> pd.DataFrame:
    """Based on the prepared historic data calculate historic indicators.

    Historic indicator refers to the quantities that will be used by the
    downsacling, e.g. from Final Energy calculate Final Energy/GDP|PPP or for
    Final Energy|Electricity calculate Final Energy|Electricity/Final Energy.
    For Final energy this is the energy intensity for all other sectors it is
    the relative share compared to the corresponding main sector i.e. Final
    Energy|Electricity -> Final Energy or Final
    Energy|Transportation|Electricity -> Final Energy|Electricity.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Takes the input from convert_historic_data

    Returns
    -------
    pd.DataFrame
        DataFrame in the same format as given with 'normalized' columns and column names
        that reflect the normalization, i.e. Final Energy/GDP|PPP in place of Final Energy

    Notes
    -----

    _convert_historic_data is defined **and** called **inside** this function
    for caching purposes. Reason being that the caching saves the function code
    and since _convert_historic_data is used only inside this function it makes
    sense to store it here. Since the function is actually called only once and
    not inside a loop one could think about removing the function altogether.
    """

    hist_df = _convert_historic_data(
        iea_extended_bal_file,
        historic_data_file,
        region,
        model,
        country_mapping_file,
        pyam_mapping_file,
    )
    hist_ind_df = hist_df[["POPULATION", gdp]].copy(deep=True)
    hist_ind_df["GDPCAP"] = hist_ind_df[gdp] / hist_ind_df["POPULATION"]
    # here num is the sector
    for num, den in dict_y_den.items():
        if num in hist_df.columns:
            hist_ind_df[f"{num}/{den}"] = hist_df[num] / hist_df[den]
    return hist_ind_df


# @make_optionally_cacheable
def _get_bad_countries(
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    region: Optional[str] = None,
    model: Optional[str] = None,
    country_mapping_file: Optional[InputFile] = None,
    pyam_mapping_file: Optional[InputFile] = None,
    _calculate_historic_indicators: MemFunc = MemFunc(_calculate_historic_indicators),
) -> list:
    """Get the list of countries with declining gdp per capita between 1980 and 2010

    Gdp per capita falls under the category of a historic indicator since it is
    calculated by dividing gdp by population. The list of these countries is referred
    to as the 'bad countries' since they do not nicely follow the general
    macro-economic trends.

    Parameters
    ----------
    iea_extended_bal_file : InputFile
        IEA extended energy balances file
    historic_data_file : InputFile
        [description]

    Returns
    -------
    list
        List of the 'bad countries'
    """

    hist_ind_df = _calculate_historic_indicators(
        iea_extended_bal_file,
        historic_data_file,
        region,
        model,
        country_mapping_file,
        pyam_mapping_file,
    )
    hist_ind_df = hist_ind_df.reset_index(level="TIME")
    ind1980 = hist_ind_df.loc[hist_ind_df.TIME == 1980, "GDPCAP"].index
    list_bad = (
        hist_ind_df.loc[hist_ind_df.TIME == 1980, "GDPCAP"]
        >= hist_ind_df[(hist_ind_df.index.isin(ind1980)) & (hist_ind_df.TIME == 2010)][
            "GDPCAP"
        ]
    )

    return list_bad[list_bad].index.to_list()


# @make_optionally_cacheable
def _calculate_historic_fit_single_bad_country(
    iam_file: InputFile,
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    country_mapping_file: InputFile,
    pyam_mapping_file: InputFile,
    ssp_file: InputFile,
    model,
    country,
    variable,
    target,
    ssp_model,
    ssp_scenario,
    n_obs: int = 15,
    func_type: str = "log-log",
    _calculate_historic_indicators: MemFunc = MemFunc(_calculate_historic_indicators),
    fun_read_gdpcap: MemFunc = MemFunc(fun_read_gdpcap),
    fun_country2region: MemFunc = MemFunc(fun_country2region),
    get_single_longterm_fit: MemFunc = MemFunc(get_single_longterm_fit),
) -> dict:
    """Fit historic data for a single bad country.

    Parameters
    ----------
    iam_file : InputFile
        [description]
    iea_extended_bal_file : InputFile
        [description]
    historic_data_file : InputFile
        [description]
    country_mapping_file : InputFile
        [description]
    pyam_mapping_file : InputFile
        [description]
    model : [type]
        [description]
    country : [type]
        [description]
    variable : [type]
        [description]
    target : [type]
        [description]
    ssp_model : [type]
        [description]
    ssp_scenario : [type]
        [description]
    n_obs : int, optional
        [description], by default 15

    Returns
    -------
    dict
        Dictionary with the results of the fit (keys: alpha, beta, r_squared, )

    Notes
    -----
    * This function works differently than the original one in the Energy demand
    downscaling. There, zero values were either clipped or replaced with 1e-7.
    Here they are simply dropped and excluded from the fit.
    * Using f"{model}|{region[:-1]}" is a hack for now and only works for
    regions which follow the MODEL|REGION naming pattern.
    """
    hist_ind_name = f"{variable}/{dict_y_den[variable]}"
    # this is to not break backwards compatibility with the region naming
    # convention of Model|Regionr
    region = f"{model}|{fun_country2region(model, country, country_mapping_file.file, pyam_mapping_file.file)[:-1]}"
    df_hist_ind = _calculate_historic_indicators(
        iea_extended_bal_file,
        historic_data_file,
        region,
        model,
        country_mapping_file,
        pyam_mapping_file,
    )
    df_hist_ind = df_hist_ind[["GDPCAP", hist_ind_name]]
    df_hist_ind = df_hist_ind.xs(country, level="ISO", drop_level=False)
    # removing lines where the historic indicator is 0 since this kills the log-log regression
    df_hist_ind = df_hist_ind[df_hist_ind[hist_ind_name] != 0.0]
    df_hist_ind.dropna(how="any", inplace=True)
    # only add the 2100 datapoint and do the fit if we actually have data
    if len(df_hist_ind):
        gdp_proj_2100 = fun_read_gdpcap(country, ssp_file, ssp_model, ssp_scenario).loc[
            (2100, country)
        ]
        df_hist_ind.loc[(2100, country), "GDPCAP"] = gdp_proj_2100
        region = fun_country2region(
            model, country, country_mapping_file.file, pyam_mapping_file.file
        )

        longterm_region_fit = get_single_longterm_fit(
            iam_file, model, f"{model}|{region[:-1]}", target, variable, func_type
        )

        est_2100 = longterm_region_fit["fit_func"].predict_y(gdp_proj_2100)
        df_hist_ind.loc[(2100, country), hist_ind_name] = est_2100

        # If a country has less than n_obs, we still want to get at least one fit
        # we don't run the optimization but one fit we still want.
        best_fit = {"fit_func": func_dict[func_type]()}
        best_fit["fit_func"].fit(df_hist_ind["GDPCAP"], df_hist_ind[hist_ind_name])
        best_fit["r_squared"] = best_fit["fit_func"].r_squared
        # best_fit["beta"] = best_fit["fit_func"].beta
        # best_fit["alpha"] = best_fit["fit_func"].alpha

        """"
        best_fit = linregress(
            np.log(df_hist_ind["GDPCAP"]),
            np.log(df_hist_ind[hist_ind_name]),
        )._asdict()"""
        best_fit["Starting year"] = df_hist_ind.index.get_level_values("TIME").min()
        best_fit["End year"] = df_hist_ind.index.get_level_values("TIME").max()
        best_fit["No of observations"] = len(df_hist_ind)
        best_fit["obj"] = len(df_hist_ind) * best_fit["fit_func"].r_squared

        for starting_year in df_hist_ind.index.get_level_values("TIME")[:-n_obs]:
            df_hist_ind_sliced = df_hist_ind.loc[starting_year:]
            fit_dict = {"fit_func": func_dict[func_type]()}
            fit_dict["fit_func"].fit(
                df_hist_ind_sliced["GDPCAP"], df_hist_ind_sliced[hist_ind_name]
            )
            fit_dict["r_squared"] = fit_dict["fit_func"].r_squared
            # fit_dict["beta"] = fit_dict["fit_func"].beta
            # fit_dict["alpha"] = fit_dict["fit_func"].alpha
            fit_dict["Starting year"] = starting_year
            fit_dict["End year"] = df_hist_ind.index.get_level_values("TIME").max()
            fit_dict["No of observations"] = len(df_hist_ind_sliced)
            fit_dict["obj"] = len(df_hist_ind_sliced) * (fit_dict["fit_func"].r_squared)

            if fit_dict["obj"] > best_fit["obj"]:
                best_fit = fit_dict
    # for countries where we don't have data
    else:
        keys = [
            "slope",
            "intercept",
            "rvalue",
            "pvalue",
            "stderr",
            "intercept_stderr",
            "Starting year",
            "End year",
            "No of observations",
            "obj",
        ]
        best_fit = {key: np.NaN for key in keys}

    # NOTE: For the future 'functional form' will not be fixed but chosen by the user

    info_dict = {
        "MODEL": model,
        "SCENARIO": target,
        "ISO": country,
        "SECTOR": variable,
        "functional form": func_type,
    }

    return dict(info_dict, **best_fit)


# @make_optionally_cacheable
def _calculate_historic_fit_good_countries(
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    country_mapping_file: InputFile,
    pyam_mapping_file: InputFile,
    model: Optional[str] = None,
    region: Optional[str] = None,
    sector: Optional[str] = None,
    func_type: str = "log-log",
    n_obs: int = 15,
    comp_with_old_ngfs: bool = False,
    _calculate_historic_indicators: MemFunc = MemFunc(_calculate_historic_indicators),
    _get_bad_countries: MemFunc = MemFunc(_get_bad_countries),
) -> pd.DataFrame:
    """Based on historic indicators calculate linear regressions for **only** the good
    countries and all variables.

    If model, region and sector information is provided the result will be filtered for the given region and sector.

    The starting point of the regression is chosen based on an objective function which
    has the form of: number_of_observations * r**2. Where r**2 is the r-square value of
    the fit. Can set a user defined minimum amount of observations needed. There is
    also **no** base year harmonization applied at this point.

    Parameters
    ----------
    hist_ind_df : pd.DataFrame
        Uses the results from calculate_historic_indicators
    n_obs : int, optional
        Minimum number of yearly observations required for fitting the regression,
        by default 15
    comp_with_old_ngfs : bool, optional
        Restricts the data to <= 2015 this was introduced to keep backwards
        comparability with the old NGFS results where this was the max year, by default False

    Returns
    -------
    pd.DataFrame
        Returns the best fit for each country and each sector. Format:
        index: ISO (of the country)
        columns: beta, alpha, rvalue, pvalue, stderr, r_squared, obs per country,
        first year per country, last year per country, obj function value,
        functional form, SECTOR
    """
    hist_ind_df = _calculate_historic_indicators(
        iea_extended_bal_file,
        historic_data_file,
        region,
        model,
        country_mapping_file,
        pyam_mapping_file,
    )

    bad_countries = _get_bad_countries(
        iea_extended_bal_file,
        historic_data_file,
        region,
        model,
        country_mapping_file,
        pyam_mapping_file,
    )

    hist_ind_df.reset_index(level="TIME", inplace=True)
    hist_ind_df = hist_ind_df.loc[~hist_ind_df.index.isin(bad_countries)]
    # NOTE: hard-coded magic value because we don't have data for 2017
    min_year = min(hist_ind_df.TIME)
    max_year = max(hist_ind_df.TIME)

    # to preserve backwards comparability with the old NGFS results we put the
    # option of end the historic data at 2015
    if comp_with_old_ngfs:
        hist_ind_df = hist_ind_df[hist_ind_df["TIME"] <= 2015]

    # NOTE: Fragile line, should be replaced with a list of the actual sectors
    # (or variables).
    if sector is None:
        sectors = [x for x in hist_ind_df.columns if x.startswith("Final Energy")]
    else:
        sectors = [f"{sector}/{dict_y_den[sector]}"]

    # for each starting year
    df_list = {}
    country_index = {}
    # for year in years:
    for i, starting_year in enumerate(range(min_year, max_year - n_obs)):
        hist_ind_df = hist_ind_df[hist_ind_df.TIME >= starting_year]
        for sector in sectors:
            # slice for each sector
            # kick out the zeros
            # count how many observations we have
            hist_ind_df_sector = hist_ind_df[["GDPCAP", sector, "TIME"]]
            hist_ind_df_sector.dropna(how="any", inplace=True)
            hist_ind_df_sector = hist_ind_df_sector[hist_ind_df_sector[sector] != 0]
            grouped_by_country = hist_ind_df_sector.groupby("ISO")
            obs_per_country = grouped_by_country[sector].count()
            obs_per_country.name = "No of observations"
            first_year_per_country = grouped_by_country["TIME"].min()
            first_year_per_country.name = "Starting year"
            last_year_per_country = grouped_by_country["TIME"].max()
            last_year_per_country.name = "End year"

            fit_list = {}
            for name, group in grouped_by_country:
                fit_list[name] = func_dict[func_type]()
                fit_list[name].fit(group["GDPCAP"], group[sector])

            if not len(fit_list):
                continue
            fit_df = pd.DataFrame(fit_list.values(), fit_list.keys()).rename(
                columns={0: "fit_func"}
            )
            fit_df["r_squared"] = [i.r_squared for i in fit_df["fit_func"]]

            fit_df = pd.concat(
                [
                    fit_df,
                    obs_per_country,
                    first_year_per_country,
                    last_year_per_country,
                ],
                axis=1,
            )
            fit_df["obj"] = fit_df["r_squared"] * fit_df["No of observations"]
            fit_df["functional form"] = func_type
            fit_df["SECTOR"] = sector.split("/")[0]

            # add the starting and the finish year
            if i == 0:
                df_list[sector] = fit_df
                country_index[sector] = fit_df.index
            else:
                # fill all missing countries
                missing_countries = [
                    c for c in country_index[sector] if c not in fit_df.index
                ]
                for country in missing_countries:
                    fit_df.loc[country, fit_df.columns] = [
                        # np.NaN,  # beta
                        np.NaN,  # alpha
                        # 0,  # rvalue
                        # np.NaN,  # pvalue
                        # np.NaN,  # stderr
                        0,  # rsquared
                        0,  # no of observations
                        starting_year,  # starting year
                        max_year,  # final year
                        0,  # obj
                        func_type,  # func form
                        sector.split("/")[0],  # sector
                    ]
                if missing_countries:
                    fit_df.sort_index(inplace=True)
                    df_list[sector].sort_index(inplace=True)

                df_append=df_list[sector][df_list[sector]["obj"] >= fit_df["obj"]]
                df_list[sector] =pd.concat([fit_df[fit_df["obj"] > df_list[sector]["obj"]], df_append])
                # df_list[sector] = fit_df[fit_df["obj"] > df_list[sector]["obj"]].append(
                #     df_append
                # )
                df_list[sector].sort_index(inplace=True)

    return pd.concat(df_list.values())


def _get_historic_fit_all_countries_no_harmonization(
    iam_file: InputFile,
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    country_mapping_file: InputFile,
    pyam_mapping_file: InputFile,
    ssp_file: InputFile,
    model,
    variable,
    target,
    region: Optional[str] = None,
    func_type: str = "log-log",
    ssp_scenario: str = "SSP2",
    ssp_model: str = "OECD Env-Growth",
    comp_with_old_ngfs: bool = False,
    n_obs: int = 15,
) -> pd.DataFrame:
    """Get historic (short term) fits for **all** countries (**including**
    'bad countries') for a **single** combination of model, target and variable.

    Parameters
    ----------
    iam_file : InputFile
        IAM data
    iea_extended_bal_file : InputFile
        IEA energy balances data, detailed sector information (historic)
    historic_data_file : InputFile
        general historic data
    country_mapping_file : InputFile
        [description]
    pyam_mapping_file : InputFile
        [description]
    model : [type]
        [description]
    variable : [type]
        [description]
    target : [type]
        [description]
    ssp_scenario : str, optional
        [description], by default "SSP2"
    ssp_model : str, optional
        [description], by default "OECD Env-Growth"
    comp_with_old_ngfs : bool, optional
        [description], by default False
    n_obs : int, optional
        [description], by default 15

    Returns
    -------
    pd.DataFrame
        Slope and intercept values for a given model, target, variable combination for
        **all** countries.

    Notes
    -----
    * Results are for one model, one target, one variable but **all** countries.
    * The fits for the good countries are not calculated directly but obtained
      with calculate_historic_fit. The fits for the bad countries, however, are
      calculated at this point.
    * This **does not** include harmonization corrections for alpha.
    * **Not** cached because there's barely any computation happening
    """
    hist_data = _calculate_historic_fit_good_countries(
        iea_extended_bal_file,
        historic_data_file,
        country_mapping_file,
        pyam_mapping_file,
        model,
        region,
        variable,
        func_type,
        n_obs,
        comp_with_old_ngfs,
    )
    hist_data = hist_data[hist_data.SECTOR == variable]
    bad_countries = _get_bad_countries(
        iea_extended_bal_file,
        historic_data_file,
        region,
        model,
        country_mapping_file,
        pyam_mapping_file,
    )
    res_list = [
        _calculate_historic_fit_single_bad_country(
            iam_file,
            iea_extended_bal_file,
            historic_data_file,
            country_mapping_file,
            pyam_mapping_file,
            ssp_file,
            model,
            bc,
            variable,
            target,
            ssp_model,
            ssp_scenario,
            n_obs,
            func_type=func_type,
        )
        for bc in bad_countries
    ]
    if len(res_list):
        bad_country_df = pd.DataFrame(res_list).set_index("ISO")
        bad_country_df = bad_country_df.drop(["MODEL", "SCENARIO"], axis=1)
        # hist_data = hist_data.append(bad_country_df)
        hist_data = pd.concat([hist_data, bad_country_df])
    return hist_data


# @make_optionally_cacheable
def _get_historic_fit_all_countries_harmonized_alpha(
    iam_file: InputFile,
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    country_mapping_file: InputFile,
    pyam_mapping_file: InputFile,
    ssp_file: InputFile,
    model,
    variable,
    target,
    region: Optional[str] = None,
    func_type: str = "log-log",
    ssp_scenario: str = "SSP2",
    ssp_model: str = "OECD Env-Growth",
    comp_with_old_ngfs: bool = False,
    n_obs: int = 15,
    _get_historic_fit_all_countries_no_harmonization: MemFunc = MemFunc(
        _get_historic_fit_all_countries_no_harmonization
    ),
    _calculate_historic_indicators: MemFunc = MemFunc(_calculate_historic_indicators),
    fun_country_map: MemFunc = MemFunc(fun_country_map),
    fun_read_gdp_gdpcap: MemFunc = MemFunc(fun_read_gdp_gdpcap),
):
    """Get historic fits for **all** countries including alpha harmonization. (see Notes
    on details regarding alpha harmonization)

    Parameters
    ----------
    iam_file : InputFile
        [description]
    iea_extended_bal_file : InputFile
        [description]
    historic_data_file : InputFile
        [description]
    country_mapping_file : InputFile
        [description]
    pyam_mapping_file : InputFile
        [description]
    model : [type]
        [description]
    variable : [type]
        [description]
    target : [type]
        [description]
    ssp_scenario : str, optional
        [description], by default "SSP2"
    ssp_model : str, optional
        [description], by default "OECD Env-Growth"
    comp_with_old_ngfs : bool, optional
        [description], by default False
    n_obs : int, optional
        [description], by default 15

    Returns
    -------
    pd.DataFrame
        Slope and intercept values for a given model, target, variable combination for
        **all** countries with alpha harmonization.

    Notes
    -----
    * Alpha harmonization refers to correcting the intercept of the fits so that it
    lines up with the last year of the historic data.
    * For now we assume a functional form of log(y) = alpha + beta * log(x). For the future
    we probably want to implement other functional forms. For this we would need to
    change the way we correct the alpha parameter.

    """
    df_hist_fit = _get_historic_fit_all_countries_no_harmonization(
        iam_file,
        iea_extended_bal_file,
        historic_data_file,
        country_mapping_file,
        pyam_mapping_file,
        ssp_file,
        model,
        variable,
        target,
        region,
        func_type,
        ssp_scenario,
        ssp_model,
        comp_with_old_ngfs,
        n_obs,
    )
    hist_ind_name = f"{variable}/{dict_y_den[variable]}"

    df_hist_ind = _calculate_historic_indicators(
        iea_extended_bal_file,
        historic_data_file,
        region,
        model,
        country_mapping_file,
        pyam_mapping_file,
    )
    df_hist_ind = df_hist_ind[["GDPCAP", hist_ind_name]]
    df_hist_ind.reset_index(level="TIME", inplace=True)

    df_hist_ind.dropna(how="any", inplace=True)
    df_hist_ind = df_hist_ind[df_hist_ind[hist_ind_name] != 0]

    for country in df_hist_ind.index.unique():

        # We slice for the country and if we get a pd.Series instead of a pd.Dataframe
        # this means that we only had one data point in the first place. This means
        # that we don't need to do the harmonization as the fit will be perfect anyway.
        if isinstance(df_hist_ind.loc[country], pd.DataFrame):
            df_hist_ind_country = df_hist_ind.loc[country]

            final_year = df_hist_ind_country[
                df_hist_ind_country.TIME == df_hist_ind_country.TIME.max()
            ]

            ## to be updated - function specific
            if func_type == "log-log":
                #  NOTE alpha_harm = log(y_base_year)- ln(GDPCAP)*beta
                alpha = np.log(final_year[hist_ind_name]) - np.log(
                    df_hist_fit.loc[country]["fit_func"].predict_y(final_year["GDPCAP"])
                )
            else:
                alpha = final_year[hist_ind_name] - df_hist_fit.loc[country][
                    "fit_func"
                ].predict_y(final_year["GDPCAP"])

            df_hist_fit.loc[country, "fit_func"].alpha_harm = alpha[0]

    region_country_map = fun_country_map(
        model, country_mapping_file.file, pyam_mapping_file.file
    )
    # add countries without historic data but SSP data
    missing_list = [
        fun_read_gdp_gdpcap(c, ssp_file, ssp_model, ssp_scenario)
        for c in region_country_map[
            region_country_map.REGION == f"{region.split('|')[1]}r"
        ]["ISO"]
    ]
    ssp_countries = pd.concat(missing_list).index.get_level_values(1).unique()
    missing_but_ssp_data = [c for c in ssp_countries if c not in df_hist_fit.index]
    for c in missing_but_ssp_data:
        df_append=pd.DataFrame({"Starting year": [2010], "End year": [2100]}, index=[c])
        df_hist_fit = pd.concat([df_hist_fit,df_append])
        # df_hist_fit = df_hist_fit.append(df_append)
    return df_hist_fit


# @make_optionally_cacheable
def get_historic_fit_single_region_harmonized_alpha(
    iam_file: InputFile,
    iea_extended_bal_file: InputFile,
    historic_data_file: InputFile,
    country_mapping_file: InputFile,
    pyam_mapping_file: InputFile,
    ssp_file: InputFile,
    model,
    variable,
    target,
    region,
    func_type: str = "log-log",
    ssp_scenario: str = "SSP2",
    ssp_model: str = "OECD Env-Growth",
    comp_with_old_ngfs: bool = False,
    n_obs: int = 15,
    _get_historic_fit_all_countries_harmonized_alpha: MemFunc = MemFunc(
        _get_historic_fit_all_countries_harmonized_alpha
    ),
):
    """Just a thin wrapper around get_historic_fit_all_countries_harmonized_alpha to get
    the historic fits for a given region.

    Parameters
    ----------
    iam_file : InputFile
        [description]
    iea_extended_bal_file : InputFile
        [description]
    historic_data_file : InputFile
        [description]
    country_mapping_file : InputFile
        [description]
    pyam_mapping_file : InputFile
        [description]
    model : str
        [description]
    variable : str
        [description]
    target : str
        [description]
    region : str
        [description]
    ssp_scenario : str, optional
        [description], by default "SSP2"
    ssp_model : str, optional
        [description], by default "OECD Env-Growth"
    comp_with_old_ngfs : bool, optional
        [description], by default False
    n_obs : int, optional
        [description], by default 15

    Notes
    -----
    * For now we have to add an r at the end of the region to get the correct one
    """
    return _get_historic_fit_all_countries_harmonized_alpha(
        iam_file,
        iea_extended_bal_file,
        historic_data_file,
        country_mapping_file,
        pyam_mapping_file,
        ssp_file,
        model,
        variable,
        target,
        region,
        func_type,
        ssp_scenario,
        ssp_model,
        comp_with_old_ngfs,
        n_obs,
    )


if __name__ == "__main__":

    """
    iea_bal = InputFile(
        Path(__file__).parents[1] / "input_data" / "Extended_IEA_en_bal_2019_ISO.csv"
    )
    hist_file = InputFile(
        Path(__file__).parents[1] / "input_data" / "Historical_data.csv"
    )

    iea_bal = InputFile(
        Path(__file__).parents[1]
        / "input_data"
        / "default_message_test"
        / "Extended_IEA_en_bal_2019_ISO_TEST_new.csv"
    )
    hist_file = InputFile(
        Path(__file__).parents[1]
        / "input_data"
        / "default_message_test"
        / "Historical_data_TEST_bad_country.csv"
    )
    """
    iam_file = InputFile(
        Path(__file__).parents[1]
        / "input_data"
        # / "default_message_test"
        / "snapshot_all_regions_round_1p4_2021_04_27.csv",
    )
    country_mapping_file = InputFile(
        CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
    )
    pyam_mapping_file = InputFile(CONSTANTS.INPUT_DATA_DIR / "default_mapping.csv")

    # set desired fitting function here
    # fit_function = LogLogFunc()

    # res = get_historic_fit_single_region_harmonized_alpha(
    #     iam_file,
    #     iea_bal,
    #     hist_file,
    #     country_mapping_file,
    #     pyam_mapping_file,
    #     "MESSAGEix-GLOBIOM 1.0",
    #     "Final Energy|Electricity",
    #     "d_delfrag",
    #     "MESSAGEix-GLOBIOM 1.0|North America",
    #     fit_function,
    # )
    iea_bal = InputFile(
        Path(__file__).parents[1] / "input_data" / "Extended_IEA_en_bal_2019_ISO.csv"
    )

    hist_file = InputFile(
        Path(__file__).parents[1]
        / "input_data"
        / "default_message_test"
        / "Historical_data_TEST_bad_country.csv"
    )

    model = "MESSAGEix-GLOBIOM 1.0"
    x = get_historic_fit_single_region_harmonized_alpha(
        iam_file,
        iea_bal,
        hist_file,
        country_mapping_file,
        pyam_mapping_file,
        model=model,
        variable="Final Energy|Electricity",
        scenario="d_delfrag",
        region=f"{model}|North America",
        func_type="log-log",
    )
