from typing import Union

import numpy as np
import pandas as pd
import warnings

from downscaler import CONSTANTS

RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR("downscaler/Step_5_Scenario_explorer_5.py")

import matplotlib.pyplot as plt
from downscaler.utils_pandas import (
    fun_operations_compared_to_base_year,
    fun_index_names,
    fun_expand_rename_df_by_level,
)

from downscaler.utils_pandas import (
    fun_create_var_as_sum,
    fun_dict_level1_vs_list_level2,
)
from downscaler.utils import (
    fun_add_multiply_dfmultindex_by_dfsingleindex,
    fun_eu27,
    fun_read_lulucf_country_data,
    fun_read_df_iam_from_multiple_df,
    fun_interpolate,
    fun_growth_index,
    fun_rename_index_name,
    fun_flatten_list,
)


def main(
    folder: str,
    model: str,
    baseyear: int,
    show_plots: bool = True,
    sel_method: str = "offset_harmo",
    marker_region: str = "marker",
    countrylist: list = fun_eu27(),
    agg_region: Union[None, str] = "EU27",
    ylim: Union[None, tuple] = None,
    indirect_emi_average: Union[range, list, None] = None,
    save_to_csv: bool = True,
    grassi_dynamic: bool = False,
    grassi_scen_mapping=None,
    # {
    #         "SSP2 1.9": ["d_delfrag", "o_1p5c", "o_lowdem"],
    #         "SSP2 2.6": ["d_strain", "o_2c"],
    #         "SSP2 3.4": ["h_ndc"],
    #         "SSP2 4.5": ["h_cpol"],
    #         "SSP2 6.0": ["SSP2 6.0"],
    #         "SSP2 BAU": ["SSP2 BAU"],
    #     }
) -> pd.DataFrame:
    """Harmonizes "Emissions|CO2|AFOLU" direct emissions (to match on historical country-level data at the `baseyear`) and adds indirect emissions (assumed to stay at the same level of `baseyear` over time).
    "Emissions|CO2|AFOLU" will not match regional IAM results -> there will be always a mismatch (offset) constnat over time, equal to the difference between historical data and IAMs results. (if needed can be harmonized at a later stage -> NOTE: this would make sense only if `agg_region` is None).
    
    NOTE: For a long description see: https://github.com/iiasa/downscaler_repo/issues/197
    While downscaling Land Use emissions at the country level I am using the dataset below (from Giacomo Grassi), which contains data for the period 2000-2020:  
    https://zenodo.org/records/7541525/files/Global%20models%20land%20CO2%20data%202000-2020.xlsx?download=1 
    Or the main page (with the description): https://zenodo.org/records/7650360 
    For Direct emissions I am using the average of 3 BMs Bookkeeping Models (for the `LULUCF net` category) , 
    whereas for Indirect emissions I am using DGVMs (`Non intact forest` category).
    The sum of Direct and Indirect emissions do not always match the national Inventories (Table 5: https://zenodo.org/records/7190601) 
    as they are coming from different sources.  
    
    Parameters
    ----------
    folder : str
        input data folder with IAMs results and historical AFOLU inventories (combined LULUCF fluxed, Direct + Indirect )
    model : str
        selected model (e.g. MESSAGE)
    baseyear : Union[int, str]
        Baseyear of historical emissions (e.g. 2020)
    show_plots : bool, optional
        Wheter to show plots with different harmonization methods, by default True
    sel_method : str, optional
        Selected harmonization method, by default "offset_harmo"
    marker_region : str, optional
        Select region that contains string `marker_region` (e.g. we can use `marker_region='marker'` if we want to get data from regions named [`Western_Europe_marker`, and 'Eastern_Europe_marker'] )., by default "marker"
    countrylist : list, optional
        List of countries for which we want to get AFOLU emissions, by default fun_eu27()
    agg_region : Union[None, str], optional
        Name of aggregated region., by default "EU27"
    ylim: Union[None, tuple], optional
        Upper/lower bounds on y axis e.g. `ylim(-20,20)` when showing plt.plot graphs
    indirect_emi_average Union[range, list, None], by default None
        Selected years, for which we calculate average indirect emissions (by default (if equal to None) we use the average of the last 10 years).
        If we take the last 10 years, Indirect emissions are essentially negative for all countries except for: Egypt: 0.11 MtCO2 and Lesotho 0.07 MtCO2 (and a few other countries with positive emissions but very close to zero (after the 2nd decimal point) )
    save_to_csv: bool, by default True:
        Wheter you want to save results to csv
    grassi_dynamic: bool, by default False:
        Wheter you want to apply Grassi regional growth rate (over time) to your Indirect emissions at the country level.
        This option is supported only if  `agg_region` is None.
    Returns
    -------
    pd.DataFrame
        If agg_region is not None, it returns Direct+Indirect emissions for an aggregated region (sum across all countries defined in `countrylist`). Otherwise it returns Direct+Indirect emissions for all individual countries.

    """
    var = "Emissions|CO2|AFOLU"
    if len(countrylist) <= 1:
        print(
            f"{agg_region} region contains {len(countrylist)} country - we skip this region"
        )
        return pd.DataFrame()
    res_direct = fun_harmonize_direct_afolu_emissions(
        folder, model, var, baseyear, countrylist, marker_region, agg_region
    )
    unit = res_direct["original"].columns[0][-1]
    if show_plots:
        [
            fun_lulucf_emissions_graph(res_direct[x], f"Direct_{x}", unit, ylim=ylim)
            for x in res_direct
        ]

    # NOTE Until now we have calculated the direct emissions ("Emissions|CO2|AFOLU").
    # Below we have two methods (depending on agg_region=True/False) for calculating Indirect emissions and the total.

    if indirect_emi_average is None:
        indirect_emi_average = range(2010, 2021)

    (RESULTS_DATA_DIR / folder).mkdir(exist_ok=True)

    if agg_region is not None:
        if grassi_dynamic:
            raise ValueError(
                "The `grassi_dynamic` option is supported only for individual countries.\n"
                "Reason being: we don't know which regional data(e.g. 'Developed Countries') from Grassi we should use for your {agg_region} `agg_region`.\n  "
                "Hence `agg_region` should be None if you want to apply a `grassi_dynamic` correction."
            )

        cols = [str(i) for i in indirect_emi_average]

        # Recaculate 2020 values as an average of last 10 years
        hist_lulucf = fun_get_hist_lulucf(folder, countrylist, agg_region)
        if (
            isinstance(hist_lulucf, pd.DataFrame)
            and "Net LULUCF CO2 flux|Indirect"
            in hist_lulucf.reset_index().VARIABLE.unique()
        ):
            res_indirect = (
                hist_lulucf.xs("Net LULUCF CO2 flux|Indirect", level="VARIABLE")[cols]
                .mean(axis=1)
                .sum()
            )
        else:
            res_indirect = 0

        # Direct + Indirect emissions
        res_tot = {key: res_indirect + val for key, val in res_direct.items()}

        # Select harmonization method and returns df
        df = (
            fun_get_df_from_selected_method(sel_method, agg_region, res_tot, unit)
            .reset_index()
            .set_index(["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"])
        ).rename(
            {"Harmonized base year emissions": "Emissions|CO2|LULUCF Direct + Indirect"}
        )
        if save_to_csv:
            df.drop("hist inventories", level="MODEL", axis=0).to_csv(
                f"{RESULTS_DATA_DIR}/{folder}/{model}_AFOLU_emissions.csv"
            )
    else:
        # if indirect_emi_average is not None:
        #     raise ValueError(f"`indirect_emi_average` is only used if `agg_region` is not None."
        #                      f"If you want to downscale AFOLU for Individual countries please pass `indirect_emi_average=None`. You have passed {indirect_emi_average} "
        #                      )
        if sel_method not in res_direct.keys():
            raise ValueError(
                f"Selected method {sel_method}, is not available. Please select one of the following: {res_direct.keys()}"
            )
        # Returns total afolu by country for a selected harmonization method
        df = fun_total_afolu_by_country(
            folder,
            baseyear,
            indirect_emi_average,
            sel_method,
            countrylist,
            agg_region,
            res_direct,
            unit,
            grassi_dynamic=grassi_dynamic,
            grassi_scen_mapping=grassi_scen_mapping,
        )
        if save_to_csv: # we save results in the wrapper_all_steps.py
            df.drop("hist inventories", level="MODEL", axis=0).to_csv(
                f"{RESULTS_DATA_DIR}/{folder}/{model}_AFOLU_emissions.csv"
            )
        print(f"downscaling afolu for {model} {marker_region} done")
    return df


def fun_total_afolu_by_country(
    folder: str,
    baseyear: str,
    indirect_emi_average: list,
    sel_method: str,
    countrylist: list,
    agg_region: str,
    res_direct: dict,
    unit: str,
    grassi_dynamic: bool = False,
    grassi_scen_mapping: Union[dict, None] = None,
) -> pd.DataFrame:
    """Returns dataframe with total AFOLU emissions ("Emissions|CO2|LULUCF Direct+Indirect").

    Parameters
    ----------
    folder : str
        input data folder with IAMs results and historical AFOLU inventories (combined LULUCF fluxed, Direct + Indirect )
    baseyear : Union[int, str]
        Baseyear of historical emissions (e.g. 2020)
    sel_method : str, optional
        Selected harmonization method, by default "offset_harmo"
    countrylist : list, optional
        List of countries for which we want to get AFOLU emissions, by default fun_eu27()
    agg_region : Union[None, str], optional
        Name of aggregated region., by default "EU27"
    res_direct : dict
        Dictionary with direct regional AFOLU emissions, harmonized to match base year emissions (by harmonization method (keys)),
    unit : str
        Emissions unit

    Returns
    -------
    pd.DataFrame
        Dataframe with "Emissions|CO2|AFOLU" (Direct) and Total (Direct+Indirect) AFOLU emissions
    """
    # Direct emissions at the country level
    add_cols, df_direct_country_harmo = fun_direct_afolu_by_country(
        folder, baseyear, sel_method, countrylist, agg_region, res_direct, unit
    )

    # Indirect emissions at the country level
    df_indirect = fun_indirect_afolu_by_country(
        folder, indirect_emi_average, countrylist, agg_region, add_cols
    )
    df_indirect = fun_add_multiply_dfmultindex_by_dfsingleindex(
        0 * df_direct_country_harmo,
        df_indirect.reset_index().set_index("ISO"),
        operator="+",
    ).rename(index={"Emissions|CO2|AFOLU": "Emissions|CO2|LULUCF Indirect"})

    if grassi_scen_mapping and not grassi_dynamic:
        txt = (
            f"\n WARNING: `grassi_dynamic` is equal to {grassi_dynamic}. Hence we do not apply any Grassi dynamic regional growth rates. "
            f"We disregard the `grassi_scen_mapping` that you passed: {grassi_scen_mapping} \n \n"
        )
        print(txt)
        warnings.warn(txt)

    if grassi_dynamic:
        if grassi_scen_mapping is None:
            raise ValueError(
                "If you want to apply a Grassi Dynamic correction you need to pass a `grassi_reg_mapping` dict (you passed None)"
            )
        # Check for missing scenarios in `grassi_scen_mapping`
        iam_scenarios = df_indirect.reset_index().SCENARIO.unique()
        iam_scenarios = [x for x in iam_scenarios if x != "historical"]
        grassi_scenarios = fun_flatten_list(list(grassi_scen_mapping.values()))
        missing_scenario = set(iam_scenarios) - set(grassi_scenarios)
        if len(missing_scenario):
            raise ValueError(
                f"Some scenarios are not included in the `grassi_scen_mapping`: {missing_scenario}.\n"
                "Please check your `grassi_scen_mapping`. "
            )

        growth_rates = fun_get_grassi_growth_rate_compared_to_baseyar(
            df_indirect,
            grassi_scen_mapping,
        )

        # NOTE: below no need to select 2020 as indirect emissions are constant over time
        df_indirect = (df_indirect * growth_rates).dropna(how="all", axis=0)

    return fun_create_var_as_sum(
        pd.concat([df_indirect, df_direct_country_harmo]),
        "Emissions|CO2|LULUCF Direct+Indirect",
        ["Emissions|CO2|AFOLU", "Emissions|CO2|LULUCF Indirect"],
        _level="VARIABLE",
        unit="Mt CO2/yr",
    ).rename_axis(index={"ISO": "REGION"})


def fun_get_grassi_growth_rate_compared_to_baseyar(
    df_indirect: pd.DataFrame, grassi_scen_mapping: dict, baseyear: int = 2020
) -> pd.DataFrame:
    """Returns Grassi regional indexed (`baseyar`=1) growth rates over time for each country belonging to the same region,
    by scenario (based on `grassi_scen_mapping`)

    Parameters
    ----------
    df_indirect : pd.DataFrame
        DataFrame with indirec emissions at the country level based on historical data
    grassi_scen_mapping : dict
        Dictionary with {RCP SCENARIO :[scenario list associated to that RCP]}. e.g. `{"SSP2 2.6": ["d_strain", "o_2c"]}`
    baseyear : int, optional
        Baseyear index (a the base year growth rate=1), by default 2020

    Returns
    -------
    pd.DataFrame
        Regional Grassi growth rates by scenario, at the country level (as as the regional growth rate)
    """
    dyn = pd.read_csv(CONSTANTS.INPUT_DATA_DIR / "Grassi_regions_dynamic.csv")
    dyn = fun_index_names(dyn, True, int)
    dyn = fun_interpolate(dyn, False, range(baseyear, 2105, 5), True)
    dyn = fun_growth_index(dyn, baseyear)  # * res_indirect

    # grassi_reg_mapping = {"Developed Countries": ["AUS", "JPN", "NZL", "TKL"]}
    grassi_reg_mapping = (
        pd.read_csv(CONSTANTS.INPUT_DATA_DIR / "Grassi_regional_mapping.csv")
        .dropna()
        .set_index(["R5_region", "ISO"])
    )
    grassi_reg_mapping = fun_dict_level1_vs_list_level2(
        grassi_reg_mapping,
        l1="R5_region",
        l2="ISO",
    )

    dyn = fun_expand_rename_df_by_level(dyn, grassi_reg_mapping, level="REGION")
    dyn = fun_expand_rename_df_by_level(dyn, grassi_scen_mapping, level="SCENARIO")
    ren_model_dict = {
        dyn.reset_index().MODEL.unique()[0]: df_indirect.reset_index().MODEL.unique()[0]
    }
    ren_unit_dict = {
        dyn.reset_index().UNIT.unique()[0]: df_indirect.reset_index().UNIT.unique()[0]
    }
    dyn = dyn.rename(ren_model_dict, level="MODEL").rename(ren_unit_dict, level="UNIT")
    return fun_rename_index_name(dyn, {"REGION": "ISO"})


def fun_indirect_afolu_by_country(
    folder: str, selcols: list, countrylist: list, agg_region: bool, add_cols: list, baseyear:int=2020
):
    selcols = [int(x) for x in selcols]
    df_indirect = fun_get_hist_lulucf(folder, countrylist, agg_region).xs(
        "Net LULUCF CO2 flux|Indirect", level="VARIABLE"
    )

    df_indirect.columns = [int(x) for x in df_indirect.columns]
    
    [
        df_indirect.insert(
            len(df_indirect.columns),
            x,
            # df_indirect[int(selcols)]
            df_indirect[selcols].mean(axis=1),
        )
        for x in add_cols
    ]

    if baseyear in range(2010,2030,5):
        # if baseyear coincides with one of the period reported by IAMs (e.g. 2020):
        # - we drop the baseyear from historical data
        # - we assume the baseyear values coincides with historical average (to avoid  structural breaks in the baseyear vs future indirect emissions data and beyond)
        df_indirect.loc[:,baseyear]=df_indirect.loc[:, baseyear+5]

    return df_indirect


def fun_direct_afolu_by_country(
    folder, baseyear, sel_method, countrylist, agg_region, res_direct, unit, average_hist_years=10
):
    df_direct_region = (
        fun_get_df_from_selected_method(sel_method, agg_region, res_direct, unit)
        .reset_index()
        .set_index(["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"])
    ).rename({"Harmonized base year emissions": "Emissions|CO2|AFOLU"})

    df_direct_country = fun_get_hist_lulucf(folder, countrylist, agg_region).xs(
        "Net LULUCF CO2 flux|Direct", level="VARIABLE"
    )
    df_direct_country.columns = [int(x) for x in df_direct_country.columns]
    # # Exclude baseyear so that it will be replaced by
    # selcols = [x for x in df_direct_country.columns if x != baseyear]
    # df_direct_country = df_direct_country[selcols]
    add_cols = [
        x for x in df_direct_region.columns if x not in df_direct_country.columns
    ]
    [
        df_direct_country.insert(
            len(df_direct_country.columns), x, # df_direct_country[int(baseyear)]
            # below we use last 10 years (average_hist_years) before the baseyear(2020) to calculate average direct land use emissions
            # this  will be used to calculate the ratio (% in each country), to allocate direct land use emissions in each country.
            #  df_direct_country[range(baseyear-average_hist_years,baseyear+1)].mean(axis=1)
            # Using standard deviation below
            df_direct_country[range(baseyear-average_hist_years,baseyear+1)].std(axis=1)
        )
        for x in add_cols
    ]

    # ratio
    ratio = np.abs(df_direct_country.droplevel(["MODEL", "UNIT"])) / (
        np.abs(df_direct_country.droplevel(["MODEL", "UNIT"])).sum()
    )

    ratio.index.names = ["REGION"]

    region_trend_to_be_added = fun_operations_compared_to_base_year(
        df_direct_region, baseyear, operator="sub"
    )
    region_trend_to_be_added = region_trend_to_be_added.droplevel("REGION")
    region_trend_to_be_added.loc[:, "REGION"] = 'myISO'
    region_trend_to_be_added=region_trend_to_be_added.set_index('REGION', append=True)
    region_trend_to_be_added_all = pd.concat(
            [region_trend_to_be_added.rename({"myISO":c}) for c in countrylist], axis=0
        )
    region_trend_to_be_added_all = region_trend_to_be_added_all.reset_index().set_index(
        df_direct_region.index.names
    )
    region_trend_to_be_added_all = fun_add_multiply_dfmultindex_by_dfsingleindex(
        region_trend_to_be_added_all, ratio, operator="*"
    )

    region_trend_to_be_added_all = region_trend_to_be_added_all.loc[
        :, region_trend_to_be_added_all.columns.isin(range(baseyear, 2200))
    ]

    df_direct_country_harmo = fun_add_multiply_dfmultindex_by_dfsingleindex(
        region_trend_to_be_added_all.rename_axis(index={"REGION": "ISO"}),
        df_direct_country.droplevel(["MODEL", "UNIT"]).loc[
            :, region_trend_to_be_added_all.columns
        ],
        operator="+",
    )

    return add_cols, df_direct_country_harmo


def fun_get_df_from_selected_method(sel_method, agg_region, res_tot, unit):
    df = (
        res_tot[sel_method]
        .rename({0: ("hist inventories", "historical", unit)}, axis=1)
        .stack()
        .unstack(level=0)
    )
    df.index = pd.MultiIndex.from_tuples(df.index, names=["MODEL", "SCENARIO", "UNIT"])
    df["VARIABLE"] = "Harmonized base year emissions"
    df["REGION"] = agg_region
    return df


def fun_harmonize_direct_afolu_emissions(
    folder,
    model,
    var,
    baseyear,
    countrylist,
    marker_region: str,
    agg_region: Union[None, str],
):
    var_hist = "Net LULUCF CO2 flux|Direct"
    # read LULUCF data
    df_lulucf = fun_get_hist_lulucf(folder, countrylist, agg_region)
    if not len(df_lulucf):
        df_lulucf = 0
    else:
        df_lulucf = df_lulucf.xs(var_hist, level="VARIABLE", drop_level=False)

    # read IAM data
    df_iam = fun_read_df_iam_from_multiple_df(
        model, CONSTANTS.INPUT_DATA_DIR / folder / "multiple_df"
    )
    df_iam = fun_get_model_results_for_marker_regions(
        df_iam, marker_region=marker_region
    )

    # Sum up all marker regions
    df_iam = df_iam.groupby(["MODEL", "SCENARIO", "VARIABLE", "UNIT"]).sum()

    offset = fun_offset_value(df_iam, var, df_lulucf, var_hist, baseyear=baseyear)
    ratio = fun_ratio_value(df_iam, var, df_lulucf, var_hist, baseyear=baseyear)

    input_dict = {
        "original": (df_iam, var, df_lulucf, var_hist),
        "offset_harmo": (df_iam + offset, var, df_lulucf, var_hist),
        "ratio_harmo": (df_iam * ratio, var, df_lulucf, var_hist),
        "ratio_harmo_keep_sign": (
            np.sign(df_iam) * df_iam * ratio,
            var,
            df_lulucf,
            var_hist,
        ),
    }

    return {key: fun_combine_datasets(*val) for key, val in input_dict.items()}


def fun_get_model_results_for_marker_regions(
    df_iam: pd.DataFrame,
    marker_region: str = "marker",
) -> pd.DataFrame:
    """Slices regional IAM results (`df_iam`) for selected marker regions (`marker_region`)

    Parameters
    ----------
    df_iam : pd.DataFrame
        Regional IAMs results for all regions
    marker_region : str, optional
        String to detect marker regions (e.g. all regions that contain `marker` in their name), by default `marker`

    Returns
    -------
    pd.DataFrame
        Sliced dataframe with marker regions
    """

    df_iam = df_iam.reset_index()

    # first try to get regions with exact same name as `marker_region`
    regions = [
        x for x in df_iam.REGION.unique() if type(x) is str if x == marker_region
    ]
    # then if regions in empty, search for regions that contains `marker_region` in their string
    if not regions:
        regions = [
            x for x in df_iam.REGION.unique() if type(x) is str if marker_region in x
        ]
    df_iam = df_iam[df_iam.REGION.isin(regions)]
    return df_iam.set_index(["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"])


def fun_get_hist_lulucf(folder, countrylist, agg_region):
    df_lulucf = fun_read_lulucf_country_data(
        f"{CONSTANTS.INPUT_DATA_DIR}/{folder}/combined_lulucf_fluxes.csv"
    )
    df_lulucf = df_lulucf.reset_index()[
        df_lulucf.reset_index().ISO.isin(countrylist)
    ].set_index(df_lulucf.index.names)
    if agg_region is None:
        return df_lulucf
    else:
        return df_lulucf.groupby(["MODEL", "VARIABLE", "UNIT"]).sum()


def fun_offset_value(
    df_iam: pd.DataFrame,
    var: str,
    lulucf: Union[int, pd.DataFrame],
    var_hist: str,
    baseyear: int = 2020,
)->Union[int,pd.DataFrame]:
    if not isinstance(lulucf, int):
        return lulucf.xs(var_hist, level="VARIABLE").sum()[str(baseyear)] - (
            df_iam.xs(var, level="VARIABLE")
            .groupby(["MODEL", "SCENARIO", "UNIT"])
            .sum()[str(baseyear)]
            .mean()  # mean across different IAM scenarios
        )
    return lulucf


def fun_ratio_value(df_iam: pd.DataFrame,
    var: str,
    lulucf: Union[int, pd.DataFrame],
    var_hist: str,
    baseyear: int = 2020,)->Union[int,pd.DataFrame]:
    if not isinstance(lulucf, int):
        return lulucf.xs(var_hist, level="VARIABLE").sum()[str(baseyear)] / (
            df_iam.xs(var, level="VARIABLE")
            .groupby(["MODEL", "SCENARIO", "UNIT"])
            .sum()[str(baseyear)]
            .mean()  # mean across different IAM scenarios
        )
    return lulucf


def fun_combine_datasets(df_iam, var_iam, df_lulucf, var_hist):
    if not isinstance(df_lulucf, int):
        combi = pd.concat(
            [
                df_iam.xs(var_iam, level="VARIABLE")
                .groupby(["MODEL", "SCENARIO", "UNIT"])
                .sum()
                .T,
                df_lulucf.xs(var_hist, level="VARIABLE").sum(),
            ],
            axis=1,
            sort=True,
        ).interpolate(axis=0, limit_direction="backward")
    else:
        combi = (
            df_lulucf
            + df_iam.xs(var_iam, level="VARIABLE")
            .groupby(["MODEL", "SCENARIO", "UNIT"])
            .sum()
            .T
        ).interpolate(axis=0, limit_direction="backward")

    combi.index = [int(x) for x in combi.index]
    return combi


def fun_lulucf_emissions_graph(combi, title, unit, ylim=None):
    combi.plot(legend=None, title=f"{title} - EU27")
    plt.ylabel(unit)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


if __name__ == "__main__":
    models = [
        #     "MESSAGEix-GLOBIOM 1.0",
        #     "REMIND-MAgPIE 2.1-4.2",
        #     "REMIND-MAgPIE 2.1-4.3",
        #     "AIM_CGE 2.2",
        #     "GCAM-PR 5.3",
        #     "IMAGE 3.2",
        "GCAM 5.3+ NGFS",
        "REMIND-MAgPIE 3.0-4.4",
        "AIM_CGE 2.2",
    ]
    for model in models:
        main(
            folder="EU_climate_advisory_board_2023",
            model=model,
            baseyear=2020,
            marker_region="marker",
            countrylist=fun_eu27(),
            agg_region=None,
            show_plots=False,
            ylim=(-800, 800),
        )
