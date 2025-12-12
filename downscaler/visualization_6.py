import re
import os
import time
from typing import Union
import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from downscaler import CONSTANTS
from downscaler.fixtures import iea_flow_dict, fun_conv_settings, primap_dict
from downscaler.input_caching import get_selection_dict
from downscaler.utils import (
    fun_read_df_countries,
    fun_read_df_iea_all,
    fun_read_IEA_fuel_dict,
    load_model_mapping,
    setindex,
    InputFile,
    fun_index_names,
    fun_ghg_emi_from_primap,
    fun_countrylist,
)
from downscaler.utils_pandas import fun_create_var_as_sum


# we run it with sensitivity = False
conv_settings = fun_conv_settings(False)


mp.use("Agg")

downscaling_columns = [
    "MODEL",
    "SCENARIO",
    "ISO",
    "VARIABLE",
    "UNIT",
    *list(range(2010, 2105, 5)),
]


prim_sec_en_color_dict = {
    "coal": "#03071E",
    "coal_ccs": "#0c1c77",
    "oil": "#6A040F",
    "gas": "#9D0208",  # "DB0086", "ff006e",
    "gas_ccs": "#b00209",
    "geothermal": "red",
    "nuclear": "#e5383b",  # "8338ec"
    "hydro": "#0077b6",  # "3a86ff",#"#5390d9",#"1688FA"
    "biomass": "#007f5f",  # "38B000",#"a0db35"
    "wind": "#00B4D8",  # "abc4ff",#"c8e7ff",#"bdd5ea",#"FAA307",
    "solar": "#FAA307",
}

fin_en_color_dict = {
    "solids": "#03071E",
    "liquids": "#9D0208",  # oil,
    "gases": "#6A040F",
    "heat": "red",  # HEAT,
    "electricity": "#a8dadc",  # "#b5179e", ## ELECTRICITY,
    "hydrogen": "#457b9d",
    "biomass": "#007f5f",  # "38B000",#"a0db35"
}

energy_color_map = {
    "Price|Final Energy|Residential and Commercial|Residential|Electricity|Index": "electricity",
    "Price|Final Energy|Residential and Commercial|Residential|Gases|Natural Gas|Index": "gases",
    "Price|Final Energy|Residential and Commercial|Residential|Liquids|Oil|Index": "liquids",
    "Price|Primary Energy|Biomass|Index": "biomass",
    "Price|Primary Energy|Coal|Index": "solids",
    "Price|Primary Energy|Gas|Index": "gases",
    "Price|Primary Energy|Oil|Index": "liquids",
    "Price|Secondary Energy|Electricity|Index": "electricity",
    "Price|Secondary Energy|Gases|Natural Gas|Index": "gases",
    "Price|Secondary Energy|Liquids|Biomass|Index": "biomass",
    "Price|Secondary Energy|Liquids|Oil|Index": "liquids",
    "Price|Secondary Energy|Solids|Coal|Index": "solids",
    "Price|Secondary Energy|Liquids|Index": "liquids",
    "Price|Final Energy|Residential and Commercial|Residential|Liquids|Biomass|Index": "biomass",
    "Price|Final Energy|Residential and Commercial|Residential|Solids|Biomass|Index": "biomass",
    "Price|Final Energy|Residential and Commercial|Residential|Solids|Coal|Index": "solids",
}


def read_and_format_historic_data() -> pd.DataFrame:
    """Reads, formats and filters historic energy data from IEA

    Parameters
    ----------

    Returns
    -------
    pd.DataFrame
        DataFrame with historic data energy data
    """

    fuel_dict_fname = "IEA_Fuel_dict.csv"
    iea_fuel_dict = fun_read_IEA_fuel_dict(
        CONSTANTS.INPUT_DATA_DIR / fuel_dict_fname
    )  # IEA fuel dict (from IEA fuel name, to standardised fuel names)
    iea_en_bal_fname = "Extended_IEA_en_bal_2019_ISO.csv"
    df_iea_all = fun_read_df_iea_all(
        CONSTANTS.INPUT_DATA_DIR / iea_en_bal_fname
    )  # Reading IEA data
    df_iea_all["IAM_FUEL"] = df_iea_all["PRODUCT"]  # IEA DATA
    df_iea_all = df_iea_all.replace(
        {"IAM_FUEL": iea_fuel_dict["FUEL"]}
    )  # IEA DATA: ADDING IAM_FUEL with standardised fuel name

    max_year = int(pd.to_numeric(df_iea_all.columns, errors="coerce").max())
    range_list = range(1960, max_year, 1)
    range_list = [str(i).zfill(2) for i in range_list]

    sectors_required = [
        "Total final consumption",
        "Transport",
        "Industry",
        "Residential",
        "Commercial and public services",
        "Total primary energy supply",
    ]
    df_iea_sel = df_iea_all[
        (
            (df_iea_all.FLOW == "Electricity output (GWh)")
            | (df_iea_all.FLOW.isin(sectors_required))
            | (df_iea_all.FLOW == "Imports")
            | (df_iea_all.FLOW == "Exports")
            | (df_iea_all.FLOW == "Total final consumption")
        )
    ]  # &df_iea_all.ISO.isin(countrylist)]

    df_iea_melt = df_iea_sel.melt(
        id_vars=["COUNTRY", "FLOW", "PRODUCT", "IAM_FUEL", "ISO"], value_vars=range_list
    )
    df_iea_melt.rename(columns={"variable": "TIME", "value": "VALUE"}, inplace=True)
    df_iea_melt["VALUE"] = pd.to_numeric(df_iea_melt["VALUE"], errors="coerce")
    df_iea_melt["TIME"] = pd.to_numeric(df_iea_melt["TIME"], errors="coerce")
    df_iea_melt = df_iea_melt.dropna().drop(
        ["COUNTRY", "IAM_FUEL"], axis="columns"
    )  # TODO: confirm with Fabio if this is safe
    df_iea_melt = df_iea_melt.set_index(["ISO", "FLOW", "PRODUCT"]).sort_index()
    df_iea_melt = df_iea_melt[df_iea_melt["TIME"].isin(range(1990, 2010))]
    return df_iea_melt


def get_df_iam_all_orig(
    file_name=CONSTANTS.INPUT_DATA_DIR / "snapshot_all_regions.csv",
) -> pd.DataFrame:
    """Reads, formats and filters IAM snapshot

    Parameters
    ----------
    file_name : Path
        path to IAM snapshot file
    Returns
    -------
    pd.DataFrame
        DataFrame with IAM projections
    """
    df_iam_all_orig = pd.read_csv(file_name, sep=",", encoding="latin-1")

    df_iam_all_orig.columns = map(str.upper, df_iam_all_orig.columns)
    year_cols_list = range(2010, 2105, 5)
    year_cols_list = [str(c) for c in year_cols_list]
    col_list = ["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"] + year_cols_list
    df_iam_all_orig = df_iam_all_orig[col_list]
    return df_iam_all_orig


def sectoral_hist_dfs_to_pkl():
    """Creates and stores pickle files with historic energy data for 6 sectors
    in input_data folder

    Parameters
    ----------

    Returns
    -------
    """
    if "step6_input_hist_data.pkl" not in os.listdir(CONSTANTS.INPUT_DATA_DIR):
        read_and_format_historic_data().to_pickle(
            CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data.pkl"
        )
    df_iea_melt = pd.read_pickle(CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data.pkl")

    hist_c_list = df_iea_melt.index.get_level_values(0).unique()
    df_iea_prim = df_iea_melt.loc[
        (hist_c_list, "Total primary energy supply"),
    ].droplevel("FLOW")
    df_iea_sec = df_iea_melt.loc[
        (hist_c_list, "Total final consumption"),
    ].droplevel("FLOW")
    df_iea_elec = df_iea_melt.loc[
        (hist_c_list, "Electricity output (GWh)"),
    ].droplevel("FLOW")
    df_iea_transport = df_iea_melt.loc[
        (hist_c_list, "Transport"),
    ].droplevel("FLOW")
    df_iea_industry = df_iea_melt.loc[
        (hist_c_list, "Industry"),
    ].droplevel("FLOW")
    df_iea_resident = df_iea_melt.loc[
        (hist_c_list, ["Residential", "Commercial and public services"]),
    ].droplevel("FLOW")

    df_iea_industry.to_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_industry.pkl"
    )
    df_iea_resident.to_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_resident.pkl"
    )
    df_iea_prim.to_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_primary.pkl"
    )
    df_iea_elec.to_pickle(CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_electr.pkl")
    df_iea_sec.to_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_secondary.pkl"
    )
    df_iea_transport.to_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_transport.pkl"
    )


def fun_historic_data(
    _sector, df_hist_data, flow_sub_list=False, as_percentage=True, sum_countries=True
) -> pd.Series:
    """Returns historical data for a given sector and country.
    If flow_sub_list != False, it returns the data divided by the denominator.
    It returns a pd.Series

    Parameters
    ----------
    _sector : str
        IAM sector
    df_hist_data : pd.DataFrame
        historic data from IEA
    flow_sub_list : list

    as_percentage : bool

    sum_countries : bool

    Returns
    -------
    pd.Series
        Series with historical data for _sector from df_hist_data

    Notes
    -------
    CAREFUL DOES NOT WORK WITH GDP  => DF_IEA_H??
    """
    if not len(df_hist_data):
        return pd.Series([0] * 20, range(1990, 2010))
    sel = iea_flow_dict[_sector]

    if type(sel[0]) == str:
        sel[0] = [sel[0]]

    if type(sel[1]) == str:
        sel[1] = [sel[1]]

    if sum_countries:
        num = (
            df_hist_data.loc[df_hist_data.index.intersection(sel[1]).unique()]
            .groupby("TIME")
            .sum()
            * 0.041868
            / 1e3
        ).squeeze()
        if flow_sub_list:
            fuel = (
                df_hist_data.loc[
                    df_hist_data.index.intersection(sel[1]).unique() + flow_sub_list
                ]
                .groupby("TIME")
                .sum()
                * 0.041868
                / 1e3
            ).squeeze()

    else:  # ADDED 2020_11_22
        if len(df_hist_data):
            num = (
                df_hist_data.loc[df_hist_data.index.intersection(sel[1]).unique()]
                .groupby("TIME")
                .sum()
                * 0.041868
                / 1e3
            ).squeeze()
        else:
            num = pd.Series()
        if flow_sub_list:
            fuel = (
                df_hist_data.loc[
                    df_hist_data.index.intersection(sel[1]).unique() + flow_sub_list
                ]
                .groupby("TIME")
                .sum()
                * 0.041868
                / 1e3
            ).squeeze()

    num = (1 / num) ** (-1)
    num = num.replace([np.inf, -np.inf], np.nan).dropna()

    #     if den==False:
    if not flow_sub_list:
        return num
    else:
        if as_percentage:
            #         print(flow_sub_list)
            return fuel / num
        else:
            return fuel


def fun_primary_secondary_energy_graphs_hist_dev(
    model,
    scen,
    c,
    ec,
    var,
    df_downsc_data,
    df_hist_data,
    df_iam_reg,
    ax,
    pyam_mapping_file,
    level="Secondary",
    _ymax=False,
    _hist=True,
):
    """This function plots primary or secondary energy graphs.
    It plots downscaled results or the original IAM results depending on the dataframe  (_df_read) provided as input
    It can also show historical data (_hist=True). How? we multiply _hist(boolean) x historical data

    Parameters
    ----------
    _sector : str
        IAM sector
    model : str
        model name
    scen : str
        scenario name
    c : str
        country name
    ec : str
        energy carrier
    var : str
        variable
    df_downsc_data: pd.DataFrame
        downscaled data
    df_hist_data : pd.DataFrame
        historical data
    df_iam_reg : pd.DataFrame
        IAM regional data
    ax : plt.axis
        axis to plot on
    level : str
        energy level
    _ymax : bool
    _hist : bool
        include historic data
    Note:
    - fun_historic_data provides results at the regional level based on a countrylist
    - Future data depend on dataframe provided as input (_df_read) which might contain regional IAM data or downscaled results at the country level
    """

    # if c is made by 3 DIGITS , we assume this is an ISO code => we read downscaled data
    if len(c) == 3:
        _df_read = df_downsc_data
        flag_region = False
    else:  ## Otherwise we read regional IAMs results data
        flag_region = True
        iam_region = c  # We initialise the region name (in case we are dealing with regional data)
        _df_read = df_iam_reg  ## Global Dataframe (regional IAM data)

    if flag_region:
        df_countries = fun_read_df_countries(
            CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
        )
        df_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file
        )
        countrylist = df_countries[
            df_countries.REGION == c.rsplit("|")[1] + "r"
        ].ISO.unique()  ## Creating a list of country in that region

    df_readT = (
        _df_read[(_df_read.SCENARIO == scen) & (_df_read.ISO == c)]
        .T.loc["VARIABLE":]
        .copy(deep=True)
    )
    col_name = df_readT.loc["VARIABLE"]
    df_readT.rename(columns=col_name, inplace=True)
    df_readT = df_readT.drop("VARIABLE")

    df_readT = df_readT[df_readT.index != "2100"]

    # Added 2021_02_03
    df_readT = df_readT.drop("UNIT")
    if "ISO" not in df_readT.index:
        df_readT.index = df_readT.index.astype("float")
    else:
        df_readT = df_readT.drop("ISO")
        df_readT.index = df_readT.index.astype("float")

    x = range(2010, 2055, 5)  # Future Time range
    x_hist = range(1990, 2010, 1)  # Historical Time range

    if level == "Primary":
        ec = ""

    if ec == "Electricity":
        conversion = 0.085984523
        # conversion from GWh to ktoe (then automatically converted to EJ from fun_historic_data)
    else:
        conversion = 1/41.868 # this is the conversion from GJ to KTOE (originally this was 1)

    if level == "Secondary":
        _sector_main = level + " Energy|" + ec + "|"
        direct_equivalent = 1
    else:
        _sector_main = level + " Energy|"
        direct_equivalent = 1 #3  # will be used for nuclear conversion

    if flag_region:
        c = countrylist  # List of country if this is a region
        var = ""
    sector_values = {
        "coal": [],
        "oil": [],
        "gas": [],
        "nuclear": [],
        "geothermal": [],
        "hydro": [],
        "biomass": [],
        "wind": [],
        "solar": [],
    }
    if level == "Primary":
        sector_values = {
            "coal|w/o CCS": [],
            "coal|w/ CCS": [],
            "oil": [],
            "gas|w/o CCS": [],
            "gas|w/ CCS": [],
            "nuclear": [],
            "geothermal": [],
            "hydro": [],
            "biomass": [],
            "wind": [],
            "solar": [],
        }
    for k in sector_values.keys():
        try:
            _sector = _sector_main + k[0].upper() + k[1:]
            if (k == "gas") & (_sector not in df_readT.columns):
                _sector = _sector_main + "Natural Gas"
            if k in ["coal|w/o CCS", "gas|w/o CCS"]:
                _sector = _sector_main + k[0].upper() + k[1:].split("|")[0]
            hist_sector = _hist * fun_historic_data(
                _sector, df_hist_data, as_percentage=False, sum_countries=flag_region
            )
            if not flag_region:
                hist_sector = hist_sector
            hist_sector = (
                hist_sector.loc[hist_sector.index.intersection(x_hist)] * conversion
            )
            if k == "nuclear":
                hist_sector /= direct_equivalent
            if len(hist_sector) != 20:
                hist_sector = pd.concat(
                    [
                        hist_sector,
                        pd.Series(
                            {
                                i: 0
                                for i in hist_sector.index.symmetric_difference(
                                    range(1990, 2010, 1)
                                )
                            }
                        ),
                    ]
                ).sort_index()
            if len(hist_sector) == 0:
                hist_sector = pd.Series(
                    list([0 for i in range(0, (len(x_hist)), 1)]), index=x_hist
                )
            if k in ["coal|w/o CCS", "gas|w/o CCS"]:
                _sector = _sector_main + k[0].upper() + k[1:]
            hist_sector = pd.concat([hist_sector, df_readT[_sector + var][x].fillna(0)])
        except KeyError:
            try:
                hist_sector = [0] * (len(x_hist))
                hist_sector = [
                    *hist_sector,
                    *df_readT[_sector + var][x].fillna(0).values,
                ]
            except KeyError:
                hist_sector = [0] * (len(x_hist) + len(x))
        sector_values[k] = hist_sector

    labels = [
        "coal",
        "oil",
        "gas",
        "nuclear",
        "geothermal",
        "hydro",
        "biomass",
        "wind",
        "solar",
    ]
    if level == "Primary":
        labels.insert(1, "coal_ccs")
        labels.insert(4, "gas_ccs")
    color_map = [
        prim_sec_en_color_dict[k] for k in labels if k in prim_sec_en_color_dict.keys()
    ]
    ax.stackplot(
        list(x_hist) + (list(x)),
        sector_values.values(),
        labels=labels,
        colors=color_map,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=(1.03, 0.35))

    if flag_region:
        ax.set_title(
            _sector_main + " [" + iam_region.rsplit("|")[1] + "]"
        )  # +" ** native IAM region **")
        ax.set_ylabel("EJ/yr")

    else:
        ax.set_title(_sector_main + " [" + c + "]")
        ax.set_ylabel("EJ/yr")
    if _ymax:
        ax.set_ylim(ymin=0, ymax=_ymax)
    return ax


def fun_final_energy_graphs_hist_dev(
    model,
    scen,
    c,
    s,
    var,
    df_downsc_data,
    df_hist_data,
    df_iam_reg,
    ax,
    pyam_mapping_file,
    _ymax=False,
    _hist=True,
):
    """
    Modified 2021_02_03 to add historical data
    This function plots final energy graphs

    Parameters
    ----------
    _sector : str
        IAM sector
    model : str
        model name
    scen : str
        scenario name
    c : str
        country name
    ec : str
        energy carrier
    var : str
        variable
    df_downsc_data: pd.DataFrame
        downscaled data
    df_hist_data : pd.DataFrame
        historical data
    df_iam_reg : pd.DataFrame
        IAM regional data
    ax : plt.axis
        axis to plot on
    level : str
        energy level
    _ymax : bool
    _hist : bool
        include historic data
    """
    flag_region = False  # We start by assuming we are dealing with country-level data
    iam_region = (
        c  # We initialise the region name (in case we are dealing with regional data)
    )

    # if c is made by 3 DIGITS , we assume this is an ISO code => we read downscaled data
    if len(c) == 3:
        _df_read = df_downsc_data[df_downsc_data.MODEL == model].copy(
            deep=True
        )  # Global Dataframe (country level data)
        flag_region = False
    else:  # Otherwise we read regional IAMs results data
        flag_region = True
        iam_region = c  # We initialise the region name (in case we are dealing with regional data)
        _df_read = df_iam_reg  # Global Dataframe (regional IAM data)
        # _df_read = _df_read.rename(columns={"REGION": "ISO"}).copy(deep=True)

    if flag_region:
        df_countries = fun_read_df_countries(
            CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
        )
        df_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file
        )
        countrylist = df_countries[
            df_countries.REGION == c.rsplit("|")[1] + "r"
        ].ISO.unique()  # Creating a list of country in that region
        flag_region = True  # If this block works, it means we are dealing with regional level-data

    df_readT = (
        _df_read[(_df_read.SCENARIO == scen) & (_df_read.ISO == c)]
        .T.loc["VARIABLE":]
        .copy(deep=True)
    )  # .pivot( columns=[2010,2020,2030,2040,2050,2060,2070,2080,2090])#.loc[c]#.plot()
    col_name = df_readT.loc["VARIABLE"]
    df_readT.rename(columns=col_name, inplace=True)
    df_readT = df_readT.drop("VARIABLE")

    df_readT = df_readT[df_readT.index != "2100"]

    # Added 2021_02_03
    df_readT = df_readT.drop("UNIT")
    if "ISO" in df_readT.index:
        df_readT = df_readT.drop("ISO")
    df_readT.index = df_readT.index.astype("float")

    # Range of historical and future data
    x = range(2010, 2055, 5)  # Future Time range
    x_hist = range(1990, 2010, 1)  # Historical Time range

    conversion = 1

    if flag_region:
        c = countrylist  # List of country if this is a region
        var = ""

    sector_values = {
        "solids": [],
        "liquids": [],
        "gases": [],
        "heat": [],
        "electricity": [],
        "hydrogen": [],
    }

    for k in sector_values.keys():
        try:
            _sector = "Final Energy|" + s + "|" + k.capitalize()
            if k == "hydrogen":
                # _sector = 'Final Energy|' + s + '|' + 'Hydrogen'  # sectors[1]
                hist_sector = list([0 for i in range(0, len(x_hist), 1)])
                hist_sector = hist_sector + [i for i in df_readT[_sector + var][x]]
                sector_values[k] = hist_sector
                continue
            else:
                hist_sector = fun_historic_data(
                    _sector,
                    df_hist_data,
                    as_percentage=False,
                    sum_countries=flag_region,
                )
                hist_sector = (
                    _hist * hist_sector.loc[hist_sector.index.intersection(x_hist)]
                )

            if len(hist_sector) != len(x_hist):
                hist_sector = pd.concat(
                    [
                        hist_sector,
                        pd.Series(
                            {
                                i: 0
                                for i in hist_sector.index.symmetric_difference(
                                    range(1990, 2010, 1)
                                )
                            }
                        ),
                    ]
                ).sort_index()
            hist_sector = pd.concat([hist_sector, df_readT[_sector + var][x].fillna(0)])
        except:
            hist_sector = [0] * (len(x_hist) + len(x))
        sector_values[k] = hist_sector

    ax.stackplot(
        list(x_hist) + (list(x)),
        sector_values.values(),
        labels=[
            "solids",
            "liquids",
            "gases",
            "heat",
            "electricity",
            "hydrogen",
        ],
        colors=[
            "#03071E",  # coal
            "#6A040F",  # gas
            "#9D0208",  # oil
            "red",  # HEAT
            "#a8dadc",  # "#b5179e", ## ELECTRICITY
            "#457b9d",
        ],
    )

    """
    ax.stackplot(list(x_hist) + (list(x)), solids, liquids, gases, heat, electricity, hydrogen,
                 labels=['solids', 'liquids', 'gases', 'heat', 'electricity', 'hydrogen', ],
                 colors=["#03071E",  # coal
                         "#6A040F",  # gas
                         "#9D0208",  # oil
                         "red",  ## HEAT
                         "#a8dadc",  # "#b5179e", ## ELECTRICITY
                         "#457b9d"]
                 )
    """
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc=(1.03, 0.35))

    if flag_region:
        ax.set_title(  # model+' - ('+scen+')'+'\n'+
            #                   _sector_main+ ' ['+iam_region+']'+" ** native IAM region **")
            "Final energy|"
            + s
            + " ["
            + iam_region.rsplit("|")[1]
            + "]"
        )  # +" ** native IAM region **")

        ax.set_ylabel("EJ/yr")

    else:
        ax.set_title(  # model+' - ('+scen+')'+'\n'+
            "Final energy|" + s + " [" + c + "]"
        )
        ax.set_ylabel("EJ/yr")

    if _ymax:
        ax.set_ylim(ymin=0, ymax=_ymax)

    #         plt.show()
    return ax


def fun_final_energy_graphs_hist_dev_solids(
    model,
    scen,
    c,
    s,
    var,
    df_read_all,
    df_hist_data,
    df_iam_reg,
    ax,
    pyam_mapping_file,
    _ymax=False,
    _hist=True,
):
    """
    Modified 2021_02_21 to match historical data for solids with IAM results
    This function plots final energy graphs.

    Parameters
    ----------
    _sector : str
        IAM sector
    model : str
        model name
    scen : str
        scenario name
    c : str
        country code
    ec : str
        energy carrier
    var : str
        variable
    df_downsc_data: pd.DataFrame
        downscaled data
    df_hist_data : pd.DataFrame
        historical data
    df_iam_reg : pd.DataFrame
        IAM regional data
    ax : plt.axis
        axis to plot on
    level : str
        energy level
    _ymax : bool
    _hist : bool
        include historic data
    """
    flag_region = False  # We start by assuming we are dealing with country-level data
    iam_region = (
        c  # We initialise the region name (in case we are dealing with regional data)
    )

    # if c is made by 3 DIGITS , we assume this is an ISO code => we read downscaled data
    if len(c) == 3:
        _df_read = df_read_all[df_read_all.MODEL == model].copy(
            deep=True
        )  # Global Dataframe (country level data)
        flag_region = False
    else:  # Otherwise we read regional IAMs results data
        flag_region = True
        iam_region = c  # We initialise the region name (in case we are dealing with regional data)
        _df_read = df_iam_reg  # Global Dataframe (regional IAM data)

    if flag_region:
        df_countries = fun_read_df_countries(
            CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
        )
        df_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file
        )
        countrylist = df_countries[
            df_countries.REGION == c.rsplit("|")[1] + "r"
        ].ISO.unique()  # Creating a list of country in that region

        flag_region = True  # If this block works, it means we are dealing with regional level-data

    df_readT = (
        _df_read[(_df_read.SCENARIO == scen) & (_df_read.ISO == c)]
        .T.loc["VARIABLE":]
        .copy(deep=True)
    )  # .pivot( columns=[2010,2020,2030,2040,2050,2060,2070,2080,2090])#.loc[c]#.plot()
    col_name = df_readT.loc["VARIABLE"]
    df_readT.rename(columns=col_name, inplace=True)
    df_readT = df_readT.drop("VARIABLE")

    df_readT = df_readT[df_readT.index != "2100"]

    # Added 2021_02_03
    df_readT = df_readT.drop("UNIT")
    if "ISO" in df_readT.index:
        df_readT = df_readT.drop("ISO")

    df_readT.index = df_readT.index.astype("float")

    # Range of historical and future data
    x = range(2010, 2055, 5)  # Future Time range
    x_hist = range(1990, 2010, 1)  # Historical Time range

    if flag_region:
        c = countrylist  # List of country if this is a region
        var = ""

    sector_values = {
        "solids": [],
        "liquids": [],
        "gases": [],
        "heat": [],
        "electricity": [],
        "hydrogen": [],
    }

    for k in sector_values.keys():
        try:
            _sector = "Final Energy|" + s + "|" + k.capitalize()
            if k == "hydrogen":
                # _sector = 'Final Energy|' + s + '|' + 'Hydrogen'  # sectors[1]
                hist_sector = list([0] * len(x_hist))
                hist_sector = hist_sector + [i for i in df_readT[_sector + var][x]]
                sector_values[k] = hist_sector
                continue
            hist_sector = fun_historic_data(
                _sector, df_hist_data, as_percentage=False, sum_countries=flag_region
            )
            hist_sector = (
                _hist * hist_sector.loc[hist_sector.index.intersection(x_hist)]
            )
            if k == "solids":
                solids_2010_ratio = (
                    df_readT[_sector + var][2010]
                    / fun_historic_data(
                        _sector,
                        df_hist_data,
                        as_percentage=False,
                        sum_countries=flag_region,
                    ).loc[2010]
                )
                hist_sector *= solids_2010_ratio

            if not flag_region:
                hist_sector = hist_sector  # .droplevel('ISO')  ## Need to drop ISO level (which is absent in case of regions)

            if len(hist_sector) != 20:
                hist_sector = pd.concat(
                    [
                        hist_sector,
                        pd.Series(
                            {
                                i: 0
                                for i in hist_sector.index.symmetric_difference(
                                    range(1990, 2010, 1)
                                )
                            }
                        ),
                    ]
                ).sort_index()
            if len(hist_sector) == 0:
                hist_sector = pd.Series(
                    list([0 for i in range(0, (len(x_hist)), 1)]), index=x_hist
                )
            hist_sector = pd.concat([hist_sector, df_readT[_sector + var][x].fillna(0)])
        except KeyError:
            hist_sector = [0] * (len(x_hist) + len(x))
        sector_values[k] = hist_sector

    ax.stackplot(
        list(x_hist) + (list(x)),
        sector_values.values(),
        colors=[
            "#03071E",  # coal
            "#6A040F",  # gas
            "#9D0208",  # oil
            "red",  # HEAT
            "#a8dadc",  # "#b5179e", ## ELECTRICITY
            "#457b9d",
        ],
    )

    ax.legend(
        [
            "solids",
            "liquids",
            "gases",
            "heat",
            "electricity",
            "hydrogen",
        ],
        loc=(1.03, 0.35),
    )

    if flag_region:
        ax.set_title("Final energy|" + s + " [" + iam_region.rsplit("|")[1] + "]")

        ax.set_ylabel("EJ/yr")

    else:
        ax.set_title(  # model+' - ('+scen+')'+'\n'+
            "Final energy|" + s + " [" + c + "]"
        )
        ax.set_ylabel("EJ/yr")

    if _ymax:
        ax.set_ylim(ymin=0, ymax=_ymax)
    return ax


def fun_emi_plot_v2(model, scen, c, var, df_emi, df_read, ax: plt.axis, emi_data_src):
    """Plots energy related CO2 emissions for given model, scenario, country
    at given matplotlib.pyplot.axis

    Parameters
    ----------
    model : str
        model name
    scen : str
        scenario name
    c : str
        country code - must be in df_emi and df_read
    var : str
        variable
    df_emi: pd.DataFrame
        historic emission data
    df_read : pd.DataFrame
        downscaled emission projections
    ax : plt.axis
        axis to plot on
    """

    ax.set_title("CO2 Emissions from fuel combustion " + "[" + c + "]")
    ax.set_ylabel("Mt CO2/yr")
    legend = []
    if emi_data_src == "IEA":
        setindex(df_emi, "FLOW")
        df_emi = (
            df_emi[
                (df_emi.ISO == c)
                & (df_emi.PRODUCT == "Total")
                & (df_emi.index == "CO2 fuel combustion")
            ][[str(i) for i in range(1990, 2017, 1)]]
            .T["CO2 fuel combustion"]
            .astype(float)
            / 1e3
        )
    elif emi_data_src == "PRIMAP":
        df_emi = df_emi[
            (df_emi["VARIABLE"] == "Emissions|CO2|Energy") & (df_emi["ISO"] == c)
        ][[i for i in range(1990, 2017, 1)]].T
    try:
        ax.plot(
            range(1990, 2017, 1),
            df_emi,
            color="blue",
            linestyle="--",
        )
        legend.append(f"historic CO2 ({emi_data_src})")
    except:
        ax.set_xlim(1986.5, 2053.5)
    ax.plot(
        range(2010, 2055, 5),
        # fun_index_names(df_read, True, int).columns,
        df_read[
            (df_read.ISO == c)
            & (df_read.SCENARIO == scen)
            & (df_read.VARIABLE == "Emissions|CO2|Energy" + var)
        ].T.loc["2010":"2050"],
        color="blue",
        linestyle="-",
    )
    legend.append("projected CO2")
    try:
        vals = df_read[
            (df_read.ISO == c)
            & (df_read.SCENARIO == scen)
            & (df_read.VARIABLE == "Carbon Sequestration|CCS|Biomass" + var)
        ].T.loc["2010":"2050"]
        if not vals.empty:
            ax.plot(
                range(2010, 2055, 5),
                df_read[
                    (df_read.ISO == c)
                    & (df_read.SCENARIO == scen)
                    & (df_read.VARIABLE == "Carbon Sequestration|CCS|Biomass" + var)
                ].T.loc["2010":"2050"],
                color="green",
                linestyle="-",
            )
            legend.append("CCS|Biomass")
    except ValueError:
        print("no CCS biomass data available")
    try:
        vals = df_read[
            (df_read.ISO == c)
            & (df_read.SCENARIO == scen)
            & (df_read.VARIABLE == "Carbon Sequestration|CCS|Fossil" + var)
        ].T.loc["2010":"2050"]
        if not vals.empty:
            ax.plot(
                range(2010, 2055, 5),
                df_read[
                    (df_read.ISO == c)
                    & (df_read.SCENARIO == scen)
                    & (df_read.VARIABLE == "Carbon Sequestration|CCS|Fossil" + var)
                ].T.loc["2010":"2050"],
                color="black",
                linestyle="-",
            )
            legend.append("CCS|Fossil")
    except ValueError:
        print("no CCS fossil data available")
    try:
        vals = df_read[
            (df_read.ISO == c)
            & (df_read.SCENARIO == scen)
            & (
                df_read.VARIABLE
                == "Carbon Sequestration|CCS|Industrial Processes" + var
            )
        ].T.loc["2010":"2050"]
        if not vals.empty:
            ax.plot(
                range(2010, 2055, 5),
                df_read[
                    (df_read.ISO == c)
                    & (df_read.SCENARIO == scen)
                    & (
                        df_read.VARIABLE
                        == "Carbon Sequestration|CCS|Industrial Processes" + var
                    )
                ].T.loc["2010":"2050"],
                color="grey",
                linestyle="-",
            )
            legend.append("CCS|Ind. Processes")
    except ValueError:
        print("no CCS industry data available")
    ax.legend(legend, loc=(1.02, 0), fontsize="small")


def fun_emi_area_plot(
    model, project, scen, c, var, df_emi, df_read, ax: plt.axis, emi_data_src
):
    """Plots energy related CO2 emissions for given model, scenario, country
    at given matplotlib.pyplot.axis

    Parameters
    ----------
    model : str
        model name
    scen : str
        scenario name
    c : str
        country code - must be in df_emi and df_read
    var : str
        variable
    df_emi: pd.DataFrame
        historic emission data
    df_read : pd.DataFrame
        downscaled emission projections
    ax : plt.axis
        axis to plot on
    """
    ax.set_title("CO2 Emissions from fuel combustion " + "[" + c + "]")
    ax.set_ylabel("Mt CO2/yr")
    legend = []
    if emi_data_src == "IEA":
        setindex(df_emi, "FLOW")
        df_emi = (
            df_emi[
                (df_emi.ISO == c)
                & (df_emi.PRODUCT == "Total")
                & (df_emi.index == "CO2 fuel combustion")
            ][[str(i) for i in range(1990, 2017, 1)]]
            .T["CO2 fuel combustion"]
            .astype(float)
            / 1e3
        )
    elif emi_data_src == "PRIMAP":
        df_emi = fun_create_var_as_sum(
            df_emi.set_index(["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"]),
            "Emissions|Total Non-CO2",
            {"Emissions|Kyoto Gases (incl. indirect AFOLU)": 1, "Emissions|CO2": -1},
            unit="Mt CO2-equiv/yr",
        ).reset_index()
        vars = [
            "Emissions|CO2|LULUCF Direct+Indirect",
            "Emissions|CO2|Energy",
            "Emissions|CO2|Industrial Processes",
            "Emissions|Total Non-CO2",
        ]
        if len(c) == 3:
            df_emi = (
                df_emi[(df_emi["VARIABLE"].isin(vars)) & (df_emi["ISO"] == c)][
                    [i for i in range(1990, 2017, 1)] + ["VARIABLE"]
                ]
                .set_index("VARIABLE")
                .T
            )

        else:
            clist = fun_countrylist(
                model,
                project,
                f"{c}r",
            )
            df_emi = (
                df_emi[(df_emi["VARIABLE"].isin(vars)) & (df_emi["ISO"].isin(clist))][
                    [i for i in range(1990, 2017, 1)] + ["VARIABLE"]
                ]
                .groupby("VARIABLE")
                .sum()
            ).T
        df_emi.index = df_emi.index.astype(int)
    try:
        # df_emi.plot.bar(ax=ax, stacked=True)
        if len(c) != 3:
            # This means regional data -> we use 'Emissions|CO2|AFOLU' instead of 'Emissions|CO2|LULUCF Direct+Indirect'
            vars = vars + ["Emissions|CO2|AFOLU"]
            vars.remove("Emissions|CO2|LULUCF Direct+Indirect")

        df_downsc = (
            df_read[
                (df_read.ISO == c)
                & (df_read.SCENARIO == scen)
                & (df_read.VARIABLE.isin(vars))
            ]
            .set_index("VARIABLE")
            .T.loc["2010":"2050"]
        )
        df_downsc.index = df_downsc.index.astype(int)
        if len(c) != 3:
            df_emi = df_emi.rename(
                {"Emissions|CO2|LULUCF Direct+Indirect": "Emissions|CO2|AFOLU"}, axis=1
            )
        df_all = pd.concat([df_emi, df_downsc]).sort_index()
        sort_var = [
            "Emissions|CO2|Energy",
            "Emissions|CO2|Industrial Processes",
            "Emissions|Total Non-CO2",
        ]
        if len(c) == 3:
            sort_var = ["Emissions|CO2|LULUCF Direct+Indirect"] + sort_var
        else:
            sort_var = ["Emissions|CO2|AFOLU"] + sort_var
        df_all = df_all[sort_var]
        df_neg = df_all.clip(-np.inf, 0)
        df_neg.columns = df_neg.columns + "_neg"
        df_all.clip(0, np.inf).merge(
            df_neg, left_index=True, right_index=True
        ).plot.area(
            ax=ax,
            stacked=True,
            color=["#5aa642", "#3f82b7", "#cb3d35", "#f18f38"],
            linewidth=1,
            # alpha=0.8
        )
        df_total = df_all.sum(axis=1)
        df_total.name = "Emissions|Net GHG"
        df_total.plot(ax=ax, color="black", linewidth=2)
        # legend.append(vars)
    except:
        ax.set_xlim(1986.5, 2053.5)
    legend.append("projected CO2")
    try:
        vals = df_read[
            (df_read.ISO == c)
            & (df_read.SCENARIO == scen)
            & (df_read.VARIABLE == "Carbon Sequestration|CCS|Biomass" + var)
        ].T.loc["2010":"2050"]
        if not vals.empty:
            ax.plot(
                range(2010, 2055, 5),
                df_read[
                    (df_read.ISO == c)
                    & (df_read.SCENARIO == scen)
                    & (df_read.VARIABLE == "Carbon Sequestration|CCS|Biomass" + var)
                ].T.loc["2010":"2050"],
                color="green",
                linestyle="dashdot",
                label="CCS|Biomass",
            )
            legend.append("CCS|Biomass")
    except ValueError:
        print("no CCS biomass data available")
    try:
        vals = df_read[
            (df_read.ISO == c)
            & (df_read.SCENARIO == scen)
            & (df_read.VARIABLE == "Carbon Sequestration|CCS|Fossil" + var)
        ].T.loc["2010":"2050"]
        if not vals.empty:
            ax.plot(
                range(2010, 2055, 5),
                df_read[
                    (df_read.ISO == c)
                    & (df_read.SCENARIO == scen)
                    & (df_read.VARIABLE == "Carbon Sequestration|CCS|Fossil" + var)
                ].T.loc["2010":"2050"],
                color="#cccccc",
                linestyle="dashed",
                label="CCS|Fossil",
            )
            legend.append("CCS|Fossil")
    except ValueError:
        print("no CCS fossil data available")
    try:
        vals = df_read[
            (df_read.ISO == c)
            & (df_read.SCENARIO == scen)
            & (
                df_read.VARIABLE
                == "Carbon Sequestration|CCS|Industrial Processes" + var
            )
        ].T.loc["2010":"2050"]
        if not vals.empty:
            ax.plot(
                range(2010, 2055, 5),
                df_read[
                    (df_read.ISO == c)
                    & (df_read.SCENARIO == scen)
                    & (
                        df_read.VARIABLE
                        == "Carbon Sequestration|CCS|Industrial Processes" + var
                    )
                ].T.loc["2010":"2050"],
                color="#7f7f7f",
                linestyle="dotted",
                label="CCS|Ind. Processes",
            )
            legend.append("CCS|Ind. Processes")
    except ValueError:
        print("no CCS industry data available")
    handles = [h for h, l in zip(*ax.get_legend_handles_labels()) if "neg" not in l]
    labels = [l for h, l in zip(*ax.get_legend_handles_labels()) if "neg" not in l]
    labels = [
        l.replace("Emissions|", "")
        .replace("Direct+Indirect", "")
        .replace("ustrial", ".")
        for l in labels
    ]
    ax.legend(handles, labels, loc=(1.02, 0), fontsize="small")


def fun_gdp_plot(model, scen, c, df_read, ax: plt.axis):
    """Plots energy related CO2 emissions for given model, scenario, country
    at given matplotlib.pyplot.axis

    Parameters
    ----------
    model : str
        model name
    scen : str
        scenario name
    c : str
        country code - must be in df_emi and df_read
    var : str
        variable
    df_emi: pd.DataFrame
        historic emission data
    df_read : pd.DataFrame
        downscaled emission projections
    ax : plt.axis
        axis to plot on
    """
    # if len(c) == 3:
    #     print("here 3 letter iso")
    ax.set_title("GDP per capita " + "[" + c + "]")
    if len(df_read[(df_read.VARIABLE == "GDP|PPP")]):
        ax.set_ylabel(df_read[(df_read.VARIABLE == "GDP|PPP")]["UNIT"].values[0])
        ax.set_xlim(1986.5, 2053.5)
        pop_vals = df_read[(df_read.VARIABLE == "Population")].T.loc["2010":"2050"]
        gdp_vals = df_read[(df_read.VARIABLE == "GDP|PPP")].T.loc["2010":"2050"]

        # pop_vals = fun_index_names(df_read[(df_read.VARIABLE == "Population")]).T.loc[
        #     "2010":"2050"
        # ]
        # gdp_vals = fun_index_names(df_read[(df_read.VARIABLE == "GDP|PPP")]).T.loc[
        #     "2010":"2050"
        # ]

        y_vals = gdp_vals.div(pop_vals.values)
        ax.plot(
            range(2010, 2055, 5),
            y_vals,
            color="blue",
            linestyle="-",
        )
        # ax.plot(
        #     fun_index_names(df_read[(df_read.VARIABLE == "Population")]).columns,
        #     y_vals,
        #     color="blue",
        #     linestyle="-",
        # )


def fun_prices_plot(
    model, scen, c, df_read, ax1: plt.axis, ax2: plt.axis, ax3: plt.axis
):
    """Plots energy related CO2 emissions for given model, scenario, country
    at given matplotlib.pyplot.axis

    Parameters
    ----------
    model : str
        model name
    scen : str
        scenario name
    c : str
        country code - must be in df_emi and df_read
    var : str
        variable
    df_emi: pd.DataFrame
        historic emission data
    df_read : pd.DataFrame
        downscaled emission projections
    ax1 : plt.axis
        axis for primary energy vars
    ax2 : plt.axis
        axis for secondary energy vars
    ax3 : plt.axis
        axis for final energy vars
    """
    price_vals = df_read[(df_read.VARIABLE.str.startswith("Price"))]
    price_vals = price_vals[
        price_vals["VARIABLE"].str.contains("Index")
        & price_vals["VARIABLE"].str.contains(" Energy")
    ]
    price_vals_prim = price_vals[price_vals["VARIABLE"].str.contains("Primary")]
    price_vals_sec = price_vals[price_vals["VARIABLE"].str.contains("Secondary")]
    price_vals_fin = price_vals[price_vals["VARIABLE"].str.contains("Final")]

    linestyle_dict = {
        "Primary Energy": "solid",
        "Secondary Energy": "dashed",
        "Final Energy": "dotted",
    }

    for level_vals, ax in zip(
        [price_vals_prim, price_vals_sec, price_vals_fin], [ax1, ax2, ax3]
    ):
        legend_labels = level_vals["VARIABLE"]
        level_vals = level_vals.T.loc["2010":"2050"]
        # colors = [fin_en_color_dict[energy_color_map[i]] for i in legend_labels]
        colors = [
            fin_en_color_dict[energy_color_map[i]]
            if i in energy_color_map
            else "#000000"
            for i in legend_labels
        ]
        linestyles = [linestyle_dict[i.split("|")[1]] for i in legend_labels]
        for i, j, k in zip(level_vals.columns, colors, linestyles):
            ax.set_title("Energy price indices " + "[" + c + "]")
            ax.set_ylabel("Index (2020 = 1)")
            ax.set_xlim(1986.5, 2053.5)
            ax.plot(
                range(2010, 2055, 5),
                level_vals[i],
                color=j,
                linestyle=k,
            )
        legend_margin_vert = 0.12 + len(level_vals.columns) * 0.08
        ax.legend(
            ["|".join(i[1:-1]) for i in legend_labels.str.split("|").values],
            bbox_to_anchor=(0.5, -legend_margin_vert),
            loc="lower center",
        )


def run_step6(
    project_name="NGFS_2022",
    input_file_downsc_data: str = "_NGFS_2022_March_first_round.csv",
    region_patterns: Union[str, list] = "",
    model_patterns: Union[str, list] = "",
    scenario_patterns: Union[str, list] = "",
    country_patterns: Union[str, list] = "",
    var_list: Union[str, list] = [""],
    pdf_out: str = "Step_6",
    hist_emi_src="PRIMAP",
):
    """Runs step6, which includes input data loading, IAM pattern matching, plotting and finally saving in PDFs

    Parameters
    ----------
    project_name='NGFS',
        project name determining path for downscaled data
    input_file_downsc_data: str = "__NGFS_November_step4_FINAL_primary_harmo.csv",
        file name of downscaled data
    region_patterns
        regex for region matching
    model_patterns
        regex for model matching
    scenario_patterns
        regex for scenario matching
    country_patterns
        regex for country matching
    var_list
        list of energy variables
    pdf_out
        file name of PDF print
    """
    pyam_mapping_file = CONSTANTS.INPUT_DATA_DIR / project_name / "default_mapping.csv"

    s0 = time.time()
    results_dir = CONSTANTS.RES_DIR / "6_Visuals"

    if not os.path.isdir(results_dir / project_name):
        os.mkdir(results_dir / project_name)
    input_dir = CONSTANTS.INPUT_DATA_DIR / project_name
    step5_dir = CONSTANTS.RES_DIR / "5_Explorer_and_New_Variables"  # / project_name

    # if "step_6_input_iam_snapshot.pkl" not in os.listdir(input_dir):
    get_df_iam_all_orig(input_dir / "snapshot_all_regions.csv").to_pickle(
        input_dir / "step_6_input_iam_snapshot.pkl"
    )

    iam_orig = pd.read_pickle(input_dir / "step_6_input_iam_snapshot.pkl").rename(
        columns={"REGION": "ISO"}
    )

    sel_dict = get_selection_dict(
        InputFile(input_dir / "snapshot_all_regions.csv"),
        model_patterns,
        region_patterns,
        scenario_patterns,
    )
    if not len(sel_dict.keys()):
        print(
            "no matching model name. model must be one of: ", iam_orig["MODEL"].unique()
        )
        return

    if "step6_input_hist_data_primary.pkl" not in os.listdir(CONSTANTS.INPUT_DATA_DIR):
        sectoral_hist_dfs_to_pkl()

    df_iea_prim = pd.read_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_primary.pkl"
    )
    df_iea_sec = pd.read_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_secondary.pkl"
    )
    df_iea_elec = pd.read_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_electr.pkl"
    )
    df_iea_transport = pd.read_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_transport.pkl"
    )
    df_iea_industry = pd.read_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_industry.pkl"
    )
    df_iea_resident = pd.read_pickle(
        CONSTANTS.INPUT_DATA_DIR / "step6_input_hist_data_resident.pkl"
    )

    var_dict_primary_secondary = {
        i.replace("MED", "MEDIUM"): f"{x}_BLEND"
        for i, x in conv_settings["Secondary"].items()
    }
    var_dict_primary_secondary.update({"": ""})
    var_dict_final = {
        i.replace("MED", "MEDIUM"): f"{x}_BLEND"
        for i, x in conv_settings["Final"].items()
    }
    var_dict_final.update({"": ""})
    file_name = input_file_downsc_data

    var_list = [""]  # No range of projections (single pathway)
    if hist_emi_src == "IEA":
        df_emi = pd.read_csv(
            CONSTANTS.INPUT_DATA_DIR / "IEA 2019 CO2 emissions from fuels_ISO.csv",
            sep=",",
            encoding="latin",
            dtype={"REGION": str, "ISO": str},
        )
    elif hist_emi_src == "PRIMAP":
        df_emi = fun_ghg_emi_from_primap(
            None, primap_dict, range(1970, 2017)
        ).reset_index()
        df_emi["ISO"] = df_emi["REGION"]
    else:
        raise ValueError(
            f"`hist_emi_src` must be equal to `IEA` or `PRIMAP`. Your input: {hist_emi_src}."
        )
    df_countries = fun_read_df_countries(
        CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv"
    )

    df_downsc = pd.DataFrame()
    for i, model in enumerate(sel_dict.keys()):
        if not i:
            df_downsc = pd.read_csv(
                step5_dir / (model + file_name), sep=",", encoding="latin-1"
            )
            df_downsc["MODEL"] = model
        else:
            df_read_add = pd.read_csv(
                step5_dir / (model + file_name), sep=",", encoding="latin-1"
            )
            df_read_add["MODEL"] = model
            df_downsc = pd.concat([df_downsc, df_read_add])

    df_downsc.rename(columns={"ISO": "REGION"}, inplace=True)
    df_downsc = df_downsc.replace({"REGION": r"^D\."}, {"REGION": ""}, regex=True)

    try:
        df_downsc.rename(
            columns={
                "2010.0": "2010",
                "2015.0": "2015",
                "2020.0": "2020",
                "2025.0": "2025",
                "2030.0": "2030",
                "2035.0": "2035",
                "2040.0": "2040",
                "2045.0": "2045",
                "2050.0": "2050",
                "2055.0": "2055",
                "2060.0": "2060",
                "2065.0": "2065",
                "2070.0": "2070",
                "2075.0": "2075",
                "2080.0": "2080",
                "2085.0": "2085",
                "2090.0": "2090",
                "2095.0": "2095",
                "2100.0": "2100",
            },
            inplace=True,
        )
        if "UNIT" not in df_downsc.columns:
            df_downsc["UNIT"] = "EJ/yr"
        df_downsc = df_downsc[downscaling_columns]

    except:
        pass
    if "FILE" in df_downsc.columns:
        df_downsc = df_downsc.drop("FILE", axis=1)
    fig, axs = plt.subplots(4, 3, figsize=(25, 15))
    fig.tight_layout(w_pad=13.0, rect=[0.01, 0.03, 0.95, 0.90])

    for model in sel_dict.keys():
        print(model)
        df_downsc_filtered = df_downsc[df_downsc.MODEL == model].copy(deep=True)

        df_downsc_filtered["ISO"] = df_downsc_filtered["REGION"]

        scens = sel_dict[model]["targets"]

        if not len(scens):
            print("no matching scenario names for model: ", model)
            continue
        print(scens)

        df_iam_all = iam_orig.loc[iam_orig["MODEL"] == model]
        pyam_mapping_file = (
            CONSTANTS.INPUT_DATA_DIR / project_name / "default_mapping.csv"
        )
        df_model_countries, regions = load_model_mapping(
            model, df_countries, pyam_mapping_file
        )

        if not len(sel_dict[model]["regions"]):
            print("no matching region names for model: ", model)
            continue
        print(sel_dict[model]["regions"])
        for region in sel_dict[model]["regions"]:
            region = region.replace(model + "|", "")
            countrylist = (
                df_model_countries[df_model_countries["REGION"] == region + "r"]
                .ISO.unique()
                .tolist()
            )

            if not len(countrylist):
                print("no matching countries found for region: ", region)
                continue

            c_list = countrylist
            r = re.compile(f".{country_patterns}.")
            c_list = list(filter(r.match, c_list))
            if not len(c_list):
                print(f"no matching country names for model {model} in region {region}")
                continue
            print(c_list)
            countrylist = [
                i
                for i in countrylist
                if i in df_iea_prim.index.get_level_values(0).unique()
            ]
            # df_iea_prim_reg = df_iea_prim.loc[countrylist]
            # df_iea_sec_reg = df_iea_sec.loc[countrylist]
            # df_iea_elec_reg = df_iea_elec.loc[countrylist]
            # df_iea_transport_reg = df_iea_transport.loc[countrylist]
            # df_iea_industry_reg = df_iea_industry.loc[countrylist]
            # df_iea_resident_reg = df_iea_resident.loc[countrylist]
            df_iea_prim_reg = df_iea_prim.loc[df_iea_prim.index.get_level_values(0).isin(countrylist)]
            df_iea_sec_reg = df_iea_sec.loc[df_iea_sec.index.get_level_values(0).isin(countrylist)]
            df_iea_elec_reg = df_iea_elec.loc[df_iea_elec.index.get_level_values(0).isin(countrylist)]
            df_iea_transport_reg = df_iea_transport.loc[df_iea_transport.index.get_level_values(0).isin(countrylist)]
            df_iea_industry_reg = df_iea_industry.loc[df_iea_industry.index.get_level_values(0).isin(countrylist)]
            df_iea_resident_reg = df_iea_resident.loc[df_iea_resident.index.get_level_values(0).isin(countrylist)]

            df_downsc_filtered_reg = df_downsc_filtered[
                df_downsc_filtered["REGION"].isin(c_list)
            ]
            c_list = list([model + "|" + region]) + c_list

            df_iam_all_reg = df_iam_all[df_iam_all["ISO"] == c_list[0]]
            pdf = PdfPages(
                results_dir
                / project_name
                / (model + pdf_out + region + "_" + str(int(s0)) + ".pdf")
            )

            for c in c_list:
                df_iea_prim_c = pd.DataFrame()
                df_iea_sec_c = pd.DataFrame()
                df_iea_elec_c = pd.DataFrame()
                df_iea_transport_c = pd.DataFrame()
                df_iea_industry_c = pd.DataFrame()
                df_iea_resident_c = pd.DataFrame()

                if len(c) == 3:
                    df_downsc_filtered_country = df_downsc_filtered_reg[
                        df_downsc_filtered_reg["REGION"] == c
                    ]
                    if len(df_downsc_filtered_country) == 0:
                        print(c, " No downscaled countrydata we skip this country")
                        continue
                    try:
                        df_iea_prim_c = df_iea_prim_reg.loc[c]
                        df_iea_sec_c = df_iea_sec_reg.loc[c]
                        df_iea_elec_c = df_iea_elec_reg.loc[c]
                        df_iea_transport_c = df_iea_transport_reg.loc[c]
                        df_iea_industry_c = df_iea_industry_reg.loc[c]
                        df_iea_resident_c = df_iea_resident_reg.loc[c]
                    except:
                        print("There is some historic data missing for this country!")
                        pass
                else:
                    df_downsc_filtered_country = df_downsc_filtered_reg
                    df_iea_prim_c = df_iea_prim_reg.droplevel(
                        "ISO",
                    )
                    df_iea_sec_c = df_iea_sec_reg.droplevel(
                        "ISO",
                    )
                    df_iea_elec_c = df_iea_elec_reg.droplevel(
                        "ISO",
                    )
                    df_iea_transport_c = df_iea_transport_reg.droplevel(
                        "ISO",
                    )
                    df_iea_industry_c = df_iea_industry_reg.droplevel(
                        "ISO",
                    )
                    df_iea_resident_c = df_iea_resident_reg.droplevel(
                        "ISO",
                    )

                if c == c_list[0]:
                    var_list_final = [var_list[0]]
                else:
                    var_list_final = var_list

                for scen in scens:

                    df_downsc_filtered_country_scen = df_downsc_filtered_country.loc[
                        df_downsc_filtered_country["SCENARIO"] == scen
                    ]
                    df_iam_all_scen = df_iam_all_reg[df_iam_all_reg["SCENARIO"] == scen]

                    for var in var_list_final:
                        print(model, scen, c, c_list.index(c), "/", len(c_list))

                        if len(c) != 3:
                            for i in range(3):
                                axs[0][i].clear()
                            for i in range(1, 4):
                                for j in range(3):
                                    axs[i][j].set_visible(False)
                            fun_prices_plot(
                                model,
                                scen,
                                c,
                                df_iam_all_scen,
                                axs[0][0],
                                axs[0][1],
                                axs[0][2],
                            )
                            try:
                                fig.suptitle(
                                    df_countries[df_countries.ISO == c].index[0]
                                    + "\n"
                                    + model
                                    + " - ("
                                    + scen
                                    + ","
                                    + var
                                    + " Convergence)",
                                    fontsize=16,
                                )
                            except:
                                fig.suptitle(
                                    c
                                    + " ** Native region"
                                    + "\n"
                                    + model
                                    + " - ("
                                    + scen
                                    + ")",
                                    fontsize=16,
                                )
                            plt.close()
                            pdf.savefig(fig, dpi=1500)
                            for i in range(4):
                                for j in range(3):
                                    if i == 3:
                                        if (j == 0) | (j == 2):
                                            continue
                                    axs[i][j].set_visible(True)

                        try:
                            axs[0][0].clear()
                            fun_primary_secondary_energy_graphs_hist_dev(
                                model,
                                scen,
                                c,
                                "",
                                var_dict_primary_secondary[var],
                                df_downsc_filtered_country_scen,
                                df_iea_prim_c,
                                df_iam_all_scen,
                                axs[0][0],
                                pyam_mapping_file,
                                level="Primary",
                                _ymax=False,
                            )

                            axs[0][1].clear()
                            fun_primary_secondary_energy_graphs_hist_dev(
                                model,
                                scen,
                                c,
                                "Liquids",
                                var_dict_primary_secondary[var],
                                df_downsc_filtered_country_scen,
                                df_iea_sec_c,
                                df_iam_all_scen,
                                axs[0][1],
                                pyam_mapping_file,
                                level="Secondary",
                                _ymax=False,
                                _hist=True,
                            )

                            axs[0][2].clear()
                            fun_final_energy_graphs_hist_dev(
                                model,
                                scen,
                                c,
                                "Transportation",
                                var_dict_final[var],
                                df_downsc_filtered_country_scen,
                                df_iea_transport_c,
                                df_iam_all_scen,
                                axs[0][2],
                                pyam_mapping_file,
                                _ymax=False,
                                _hist=True,
                            )

                            ## adding co2 emissions
                            try:
                                axs[1][0].clear()
                                if len(c) == 3:

                                    fun_emi_area_plot(
                                        model,
                                        project_name,
                                        scen,
                                        c,
                                        var_dict_primary_secondary[var],
                                        df_emi,
                                        df_downsc_filtered_country_scen,
                                        axs[1][0],
                                        emi_data_src=hist_emi_src,
                                    )
                                else:
                                    iam_vars = (
                                        df_iam_all_scen.reset_index().VARIABLE.unique()
                                    )
                                    if (
                                        "Emissions|Kyoto Gases" in iam_vars
                                        and "Emissions|CO2" in iam_vars
                                    ):
                                        df_iam_all_scen = fun_create_var_as_sum(
                                            fun_index_names(df_iam_all_scen, True, str),
                                            "Emissions|Total Non-CO2",
                                            {
                                                "Emissions|Kyoto Gases": 1,
                                                "Emissions|CO2": -1,
                                            },
                                            unit="Mt CO2-equiv/yr",
                                        ).reset_index()
                                    fun_emi_area_plot(
                                        model,
                                        project_name,
                                        scen,
                                        c,
                                        var_dict_primary_secondary[var],
                                        df_emi,
                                        df_iam_all_scen,
                                        axs[1][0],
                                        emi_data_src=hist_emi_src,
                                    )
                            except:
                                pass

                            axs[1][1].clear()
                            fun_primary_secondary_energy_graphs_hist_dev(
                                model,
                                scen,
                                c,
                                "Gases",
                                var_dict_primary_secondary[var],
                                df_downsc_filtered_country_scen,
                                df_iea_sec_c,
                                df_iam_all_scen,
                                axs[1][1],
                                pyam_mapping_file,
                                level="Secondary",
                                _ymax=False,
                                _hist=True,
                            )

                            axs[1][2].clear()
                            fun_final_energy_graphs_hist_dev(
                                model,
                                scen,
                                c,
                                "Industry",
                                var_dict_final[var],
                                df_downsc_filtered_country_scen,
                                df_iea_industry_c,
                                df_iam_all_scen,
                                axs[1][2],
                                pyam_mapping_file,
                                _ymax=False,
                                _hist=True,
                            )

                            axs[2][0].clear()
                            # if c=='SPM':
                            #     print('check this')
                            if len(c) == 3:
                                fun_gdp_plot(
                                    model,
                                    scen,
                                    c,
                                    df_downsc_filtered_country_scen,
                                    axs[2][0],
                                )
                            else:
                                fun_gdp_plot(model, scen, c, df_iam_all_scen, axs[2][0])

                            axs[2][1].clear()
                            fun_primary_secondary_energy_graphs_hist_dev(
                                model,
                                scen,
                                c,
                                "Electricity",
                                var_dict_primary_secondary[var],
                                df_downsc_filtered_country_scen,
                                df_iea_elec_c,
                                df_iam_all_scen,
                                axs[2][1],
                                pyam_mapping_file,
                                level="Secondary",
                                _ymax=False,
                                _hist=True,
                            )

                            axs[2][2].clear()
                            ## 2021_02_21 Update
                            if model == "MESSAGEix-GLOBIOM 1.0":
                                # We harmonise Solids historical data to match MESSAGE base year data
                                fun_final_energy_graphs_hist_dev_solids(
                                    model,
                                    scen,
                                    c,
                                    "Residential and Commercial",
                                    var_dict_final[var],
                                    df_downsc_filtered_country_scen,
                                    df_iea_resident_c,
                                    df_iam_all_scen,
                                    axs[2][2],
                                    pyam_mapping_file,
                                    _ymax=False,
                                    _hist=True,
                                )
                            else:
                                fun_final_energy_graphs_hist_dev(
                                    model,
                                    scen,
                                    c,
                                    "Residential and Commercial",
                                    var_dict_primary_secondary[var],
                                    df_downsc_filtered_country_scen,
                                    df_iea_resident_c,
                                    df_iam_all_scen,
                                    axs[2][2],
                                    pyam_mapping_file,
                                    _ymax=False,
                                    _hist=True,
                                )

                            axs[3][1].clear()
                            fun_primary_secondary_energy_graphs_hist_dev(
                                model,
                                scen,
                                c,
                                "Solids",
                                var_dict_primary_secondary[var],
                                df_downsc_filtered_country_scen,
                                df_iea_sec_c,
                                df_iam_all_scen,
                                axs[3][1],
                                pyam_mapping_file,
                                level="Secondary",
                                _ymax=False,
                                _hist=True,
                            )

                            try:
                                fig.suptitle(
                                    df_countries[df_countries.ISO == c].index[0]
                                    + "\n"
                                    + model
                                    + " - ("
                                    + scen
                                    + ","
                                    + var
                                    + " Convergence)",
                                    fontsize=16,
                                )
                            except:
                                fig.suptitle(
                                    c
                                    + " ** Native region"
                                    + "\n"
                                    + model
                                    + " - ("
                                    + scen
                                    + ")",
                                    fontsize=16,
                                )

                        except:
                            print("plotting was not sucessful!")
                            pass

                        plt.close()
                        pdf.savefig(fig, dpi=1500)
            pdf.close()
    print("elapsed: ", (time.time() - s0) / 60, " minutes")


if __name__ == "__main__":
    run_step6(
        project_name="NGFS_2022",
        input_file_downsc_data="_NGFS_2022_Round_2nd.csv",
        model_patterns="*GCAM*",
        scenario_patterns="d_rap",
        region_patterns="*Taiw*",
        country_patterns="*",  # "AUT",
    )
