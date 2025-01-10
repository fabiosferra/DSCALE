import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from downscaler import CONSTANTS
from downscaler.utils_pandas import fun_index_names, fun_read_csv

from downscaler.utils import (
    fun_regions,
    fun_countrylist,
    fun_aggregate_countries,
    fun_ghg_emi_from_primap,
    fun_fill_na_with_previous_next_values,
    fun_create_var_as_sum,
)


def main(project: str, scenario="HISTCR") -> pd.DataFrame:
    """Creates a `snapshot_all_regions.csv` file based on historical data, and saved the data in a given `project` folder.
    It aggregates the observed historical country-level data to the regional level, based on a `default_mapping.csv` from the
    project folder.

    Parameters
    ----------
    project : str
        You project folder
    scenario : str, optional
        Historical data from PRIMAP, by default "HISTCR"

    Returns
    -------
    pd.DataFrame
        Snapshot with regional data
    """

    # Constants
    idx = ["MODEL", "VARIABLE", "UNIT", "SCENARIO"]
    p = project
    p_dir = CONSTANTS.INPUT_DATA_DIR / p

    # Assign short name for function
    f = fun_countrylist
    fa = fun_aggregate_countries
    fr = fun_read_csv

    # Read IEA, PRIMAP, and GDP data
    primap = fun_ghg_emi_from_primap(None, scenario=scenario).droplevel("FILE")
    # IEA file below was created with the function `fun_read_energy_iea_data_in_iamc_format`
    iea_path = CONSTANTS.INPUT_DATA_DIR / "input_reference_iea_2019_old.csv"
    gdp_path = CONSTANTS.INPUT_DATA_DIR / f"{p}/GDP_NGFS_merged.csv"
    input_dict = {}
    for k, v in {"iea": iea_path, "gdp": gdp_path}.items():
        input_dict.update({k: fr({k: v}, True, int)[k]})
    df_iea_gdp = pd.concat([x for x in (input_dict.values())], sort=True)
    # We need the line below otherwise we have multiple final energy data.
    df_iea_gdp = df_iea_gdp.groupby(df_iea_gdp.index.names).sum()

    # Calculating w/o CCS variables below:
    new_vars = [
        "Primary Energy|Coal|w/o CCS",
        "Primary Energy|Gas|w/o CCS",
        "Primary Energy|Oil|w/o CCS",
    ]
    for v in new_vars:
        df_iea_gdp = fun_create_var_as_sum(
            df_iea_gdp, v, ["|".join(v.split("|")[:-1])], unit="EJ/yr"
        )
        vw = v.replace("w/o CCS", "w/ CCS")
        df_iea_gdp = fun_create_var_as_sum(
            df_iea_gdp, vw, {"|".join(v.split("|")[:-1]): 0}, unit="EJ/yr"
        )
    # Concatenate IEA, GDP, and PRIMAP data
    df = pd.concat([df_iea_gdp, primap], axis=0, sort=True).reset_index()
    df.loc[:, "SCENARIO"] = scenario
    df = fun_index_names(df, True, int)
    df.to_csv(p_dir / "hist_country_level_data.csv")
    df = fun_fill_na_with_previous_next_values(
        df.replace(0, np.nan), use_previous_value=True, use_next_value=False
    )
    df = df.iloc[:, df.columns.isin(range(2005, 2105, 5))]

    # Get models list from default mapping
    default_mapping = pd.read_csv(p_dir / "default_mapping.csv")
    models = [x for x in default_mapping.columns if x.endswith(".REGION")]
    models = [x.replace(".REGION", "") for x in models]

    # Apply regional mapping to historical data based on models
    df_all = pd.DataFrame()
    for m in models:
        rs = fun_regions(m, p)
        # Calculate aggregated regions and combined in one dataframe
        combi = [fa(df, r, idx, f(m, p, r), True) for r in rs]
        dfco = pd.concat(combi, axis=0).reset_index()
        # Filter out ISO regions
        dfco = fun_index_names(dfco.iloc[[len(x) != 3 for x in dfco.REGION]], True, int)
        dfco = dfco.reset_index()
        dfco.loc[:, "REGION"] = [x[:-1] for x in dfco.REGION]
        dfco["MODEL"] = m
        dfco = fun_index_names(dfco, True, int)
        # Rename AFOLU variable
        rename_dict = {
            "Emissions|CO2|LULUCF Direct+Indirect": "Emissions|CO2|AFOLU",
            "Emissions|Kyoto Gases (incl. indirect AFOLU)": "Emissions|Kyoto Gases",
        }
        df_all = pd.concat([df_all, dfco.rename(rename_dict, level="VARIABLE")])

    df_all.to_csv(p_dir / "snapshot_v1" / "snapshot_all_regions_hist_data.csv")
    print("Done!")
    return df_all


if __name__ == "__main__":
    main(project="SIMPLE_hindcasting")
