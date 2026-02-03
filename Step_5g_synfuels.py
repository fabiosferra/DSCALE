# Syn fun_drop_duplicates
from downscaler.utils import *
from downscaler.utils_dictionary import *
def fun_harmonize_df_with_IAM(df, df_iam, x, k, verbose: bool = True) -> pd.DataFrame:
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
        ratio = ratio.replace(np.inf, np.nan)
    except:
        ratio=1
    if not isinstance(ratio, int):
        ratio=ratio.T
        
    k_updated = (
        df.xs(k, level="SECTOR")[x].unstack("TIME").reset_index().set_index(u + ["ISO"])
        * ratio
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

def run_sector_harmo_enhanced(
    df: pd.DataFrame,
    d: dict,
    x: str,
    df_iam: Optional[pd.DataFrame] = None,
    w:int=1, # 1 is our standard assumption (we do not sum sub-sector -> sum_anyway=False ). 0 means we sum anyway
    verbose:bool = True,
    # d2:dict= None,
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
    if df_iam is None and verbose:
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

            # v_updated_clipped=pd.DataFrame()
            # v_updated = v_updated.reset_index().set_index(df.index.names)
            # for vv in v:
            #     df_v_updated=pd.DataFrame(v_updated.xs(vv, level='SECTOR', drop_level=False)).reset_index().set_index(df.index.names)
            #     df1_min=df.xs(fun_invert_dictionary(d)[vv][0], level='SECTOR', drop_level=False).rename({fun_invert_dictionary(d)[vv][0]:vv})
            #     df2_min=pd.DataFrame()
            #     if d2 is not None:
            #         df2_min=df.xs(fun_invert_dictionary(d2)[vv][0], level='SECTOR', drop_level=False).rename({fun_invert_dictionary(d2)[vv][0]:vv})
            #     # NOTE: example: 'Final Energy|Transportation|Liquids' (in `df_v_updated`) should be smaller than 'Final Energy|Liquids' (e.g. in df1_min) as well as  'Final Energy|Transportation' (e.g. in df2_min)   
            #     df_v_updated=fun_min_across_datasets([df_v_updated, df1_min, df2_min,]).reset_index().set_index(v_updated.index.names)
            #     try:
            #         v_updated=fun_min_across_datasets([df_v_updated, v_updated]).reset_index().set_index(v_updated.index.names)
            #     except:
            #         agvhj=1
            #         print('fix this')

            if len(v_updated) == 0:
                raise ValueError(
                    "Unable to rescale sectors, `v_updated` is empty, please check your data"
                )
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

def reshape_df_after_harmo_in_IAMC_format(df_iam, df):
    if 'METHOD' in df.columns:
        df = df.drop('METHOD', axis=1)
    elif 'METHOD' in df.index.names:
        df = df.droplevel('METHOD')


    # Rename variables
    mydict={x:x.replace('LONG_TERM','') for x in df.index.get_level_values('VARIABLE')}
    df=df.rename(mydict, axis=0)

    # Fix units
    units = fun_get_variable_unit_dictionary(df_iam)
    d = {k: [v] for k, v in units.items()}
    df = fun_add_units(df, df_iam, fun_invert_dictionary(d))

    # Rename target as scenario
    return fun_rename_index_name(df, {"TARGET": "SCENARIO"})

# GOAL/Backrgound:
# There are some synfuels data in the IAMs results e.g. 'Final Energy|Transportation|Liquids|Electricity'. Eelecricity is not a fuel in DSCALE, but rather as carrier.
# We need to find a solution for this.
# We create this `synfuel` variable as the difference between total liquids and the sum of sub-sectors (excluding electricity).
# Then we harmonize the results by using the `run_sector_harmo_enhanced_iamc` (that ensures consistency with regional IAMs results if provided)
# NOTE: if you want to add historical harmonization you need to be sure that `df_iea` and `df_iam` are also fully aligned (otherwise you may create inconsistencies)
# NOTE: Given that synfuel='Electricity' this code does not work for energy_carrier= Electricity

# Define project, scenarios and file name with downscaled results
project="REMIND_fuel_mix_testing"
scenarios=['NPE-core'] # to be updated
file='REMIND-MAgPIE 3.3-4.8_REMIND_fuel_mix_testing_2026_02_02_CA_TEST_ALL_2022_harmo_step5e_None.csv' # your file name with downscaled results
synfuel='Electricity' # name of your synfuel variable (e.g. 'Final Energy|Transportation|Liquids|Electricity')

# Select regions to be harmonized (if None, all regions/countries will be harmonized) and number of iterations
sel_reg='LAMr' # None # # Region to be harmonized (e.g. LAMr for Latin America). Select None if you want to apply this to all regions/countries
iterations=15 # number of harmonization iterations (with 15/20 you get fully consistent results, but it takes longer)

# Read data
df_iam=fun_read_df_iams(project)
df_iam=fun_index_names(df_iam, True, int)
mydir=CONSTANTS.CURR_RES_DIR('step5')
df=fun_read_csv({0:mydir/file}, True, int)[0]

# Get regional mapping
myres={}
models=fun_get_models(project)
for m in models:
   temp=fun_regional_country_mapping_as_dict(m, project)
   myres[m]={f"{m}|{k}":v  for k,v in temp.items()}
myres2=fun_invert_dictionary(fun_append_list_of_dicts(list(myres.values())))

# Select None if you want to apply this to all countries/regions!
if sel_reg is not None:
    countrylist=myres[models[0]][f'{models[0]}|{sel_reg}']
else:
    countrylist=None # This will apply the harmonization to all regions/countries

model=models[0]

# TODO: Expand this using a loop if needed
# - [ ] e.g. ADD A LOOP HERE: for sector in ['Transportation', 'Industry', 'Residential and Commercial']:

mydict={'Final Energy|Transportation|Liquids':
      ['Final Energy|Transportation|Liquids|Bioenergy',
       'Final Energy|Transportation|Liquids|Biomass',
       'Final Energy|Transportation|Liquids|Coal',
       'Final Energy|Transportation|Liquids|Electricity',
       'Final Energy|Transportation|Liquids|Natural Gas',
       'Final Energy|Transportation|Liquids|Oil'
       ]}


# Sectorial harmonization (for each model,region,scenario)
regmap=fun_get_regions(project, model) # regional mapping
if countrylist is not None:
    regmap={k:v for k,v in regmap.items() if len(set(v)&(set(countrylist)))>0} # select only regions with countries in countrylist

for r,countrylist in regmap.items():
    for scen in scenarios:
        for iter in range(iterations):
            res={}
            ds=[mydict]
            print('Harmonzing', model, r, scen,'ITERATION', iter+1 )
            df1=fun_xs(df, {'SCENARIO':scen, 'REGION':countrylist}) # df
            df2=fun_rename_index_name(fun_xs(df_iam, {'REGION':f"{model}|{r[:-1]}", 'SCENARIO':scen}).drop(2005, axis=1), {'SCENARIO':'TARGET'}) # df_iam
            
            # Slice for specific variables
            all_vars=list(set(fun_flatten_list(list(mydict.values())+list([mydict.keys()]))))
            df1=fun_xs(df1, {'VARIABLE':[x for x in all_vars if synfuel not in x]}) # downscaled results, excluding synfuel variable, that will be created as the difference between total liquids and the sum of sub-sectors
            df2=fun_xs(df2, {'VARIABLE':all_vars}) # IAM results

            # # If you want to enhance it, here you may consider using 
            # # - `fun_harmonize_hist_data_by_preserving_sum_across_countries` to harmonize with historical data
            
            # Create Synfuel variable as the difference between total liquids and the sum of sub-sectors
            temp= fun_create_var_as_sum(df1, 'sum of subs', [x for x in fun_flatten_list(mydict.values())
                                                            if synfuel not in x # we exclude electricity as this is your `synfuel` variable (created as the difference between total liquids and the sum of sub-sectors)
                                                            ], unit='EJ/yr') # sum of sub-sectors
            temp = fun_divide_variable_by_another(temp, 
                                                f'Final Energy|Transportation|Liquids|{synfuel}',  # New `synfuel` variable to be created as the difference of:
                                                'Final Energy|Transportation|Liquids', 
                                                'sum of subs', 
                                                operator='-', # subtraction 
                                                unit='EJ/yr', 
                                                concatenate=True).clip(1e-9) # we clip to avoid negative values (if you want to keep negative values, remove the .clip(0.01) part)



            # Harmonize results (top-down harmonization)
            temp = run_sector_harmo_enhanced_iamc(project, temp, ds, df2, verbose=False, no_iter=1) # To enhance consistency, increase the number of iterations (no_iter) 
            temp =reshape_df_after_harmo_in_IAMC_format(df_iam, temp)
            
            # Drop the temporary variable `sum of subs` (created to calculate the Synfuel variable)
            temp=temp.drop('sum of subs', level='VARIABLE') # raise error if we find negative values for Final energy variables


            # We may consider adding historical harmonization here (if this is not done in step5e)
            # - `fun_harmonize_hist_data_by_preserving_sum_across_countries` to harmonize with historical data
            # var='Final Energy|Transportation|Liquids'
            # df_iea=fun_most_recent_iea_data()
            # # fun_harmonize_hist_data_by_preserving_sum_across_countries(
            # #         fun_index_names(temp, True, int),
            # #         fun_xs(fun_xs_fuzzy(df_iea, [var]).dropna(how='all', axis=1), {'REGION': countrylist}).dropna().reset_index().set_index(temp.index.names), 
            # #         var)
            # pd.concat(
            #             [
            #                 fun_harmonize_hist_data_by_preserving_sum_across_countries(
            #                 fun_index_names(temp, True, int),
            #                 fun_xs(fun_xs_fuzzy(df_iea, [var]).dropna(how='all', axis=1), {'REGION': countrylist}).dropna().reset_index().set_index(temp.index.names), 
            #                 var).sum(), 

            #                 fun_xs_fuzzy(fun_index_names(temp, True, int), [var]).sum(),
            #                 fun_xs_fuzzy(df_iam, [var, scen, '|lam']).sum()
            #             ], axis=1
            #         ).plot()
            
            # Sanity check
            fun_check_negative_energy_variables(temp) # Check if we find negative values for Final energy variables 
            updated_results = temp.reset_index().set_index(df.index.names)

            # Updated `df` after harmonization
            # updated_results=pd.concat(list(res.values()))
            updated_results=fun_index_names(updated_results, True, int)
            common_variables=list(set(updated_results.index)&set(df.index)) # common variables between `df` and `updated_results` (e.g. Final Energy|Transportation|Liquids)
            df=pd.concat([df.drop(common_variables), updated_results])


print('Harmonization completed')


# -------------
# These are just some Sanity Check of results
# ------------- 

# 1) Check inconsistencies after harmonization for individual countries (to minimize inconsistencies, please increase the number of iterations above)
print(show_inconsistencies(fun_xs(temp, {'REGION':iea_countries}), ds))

# 2) Compare sum of country level results with IAMs for `Transportation|Liquids|Biomass`
plt.style.use('seaborn-white')
for x in ['Final Energy|Transportation|Liquids|Biomass', 'Final Energy|Transportation|Liquids|Electricity']:
    # Calculate the downscaled and IAM values
    down_liquids = fun_xs_fuzzy(fun_xs(df, {'REGION': countrylist}), [x, scen])
    iam_liquids = fun_xs_fuzzy(fun_xs(df_iam, {'REGION': [f'{models[0]}|LAM']}), [x, scen])
    
    # Combine the data
    check = pd.concat([iam_liquids.sum().T, down_liquids.sum().T], axis=1)
    check = check.rename(columns={0: 'IAM', 1: 'downs'})  # Rename columns

    # Create a new figure for each plot
    plt.figure()

    # Plot each column with a different line style
    plt.plot(check['IAM'], linestyle='-', label='IAM', alpha=0.8, lw=3)
    plt.plot(check['downs'], linestyle='--', label='Downscaled (sum across countries)', alpha=1)

    # Add legend and labels
    plt.legend()
    plt.title(f'{x} - {scen} - LAM')
    plt.xlabel('Index')  # Set an appropriate x-label if needed
    plt.ylabel('Values')  # Set an appropriate y-label if needed

    # Show plot
    plt.show()

# -------------
# TODO SAVE RESULTS
# ------------- 
# df.to_csv('your_file_name.csv')

