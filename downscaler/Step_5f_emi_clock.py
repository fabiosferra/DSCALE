from typing import Union
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from downscaler import CONSTANTS

from pandas.testing import assert_frame_equal
from downscaler.utils_pandas import fun_xs, fun_create_var_as_sum, fun_rename_index_name
from downscaler.utils import (
    fun_read_results,
    run_sector_harmo_enhanced_iamc,
    fun_international_variables,
    fun_read_df_iams,
    fun_get_models,
    fun_wildcard,
    fun_get_variable_unit_dictionary,
    fun_add_units,
    fun_invert_dictionary,
    fun_get_regions,
    fun_index_names,
    fun_get_scenarios,
    show_inconsistencies,
    fun_flatten_list,
    fun_inconsistencies,
    fun_rename_index_name,
    fun_harmonize_df_iam_with_hist_data,
    # shrink,
    # clip_emissions_below_reference,
)
from downscaler.fixtures import step5f_dict1, step5f_dict2, step5f_dict3, step5f_temp_vars_dict, step5f_temp_vars_dict_df_iam, iea_countries


def main(
    project,
    csv_out: str,
    csv_in: Optional[str] = None,
    step: str = "step5",
    models: Optional[list] = None,
    countrylist: Optional[list] = None,
    scenarios: Optional[list] = ["h_cpol", "h_ndc", "o_2c", "o_1p5c"],
):
    files=None
    
    # Get models
    avmodels=fun_get_models(project)
    models = fun_wildcard(models, avmodels)
    
    # Read NGFS full data 
    i = project, step, files, None, models, f"{csv_in}_2022_harmo_step5e_None" # Careful about the harmonisation date!
    df = fun_read_results(*i)[0]
    
    # Read sectorial CO2 emissions (from step5b)
    if csv_in:
        i = project, step, files, None, models, f"{csv_in}_Emissions_by"
    else:
        i = project, step, files, None, models, "Emissions_by"
    df_co2 = fun_read_results(*tuple(list(i)))[0]
    d = {"VARIABLE": "Emissions|CO2|Industrial Processes"}
    df_co2 = fun_xs(df_co2, d, exclude_vars=True)

    # Append sectorial CO2 emissions
    df = pd.concat([df, df_co2], axis=0)

    # Append non-co2 by gases
    i = project, step, files, None, models, "non"
    non_co2 = fun_read_results(*tuple(list(i)))[0].dropna(how="all")
    df = pd.concat([df.droplevel("FILE"), non_co2.droplevel("REGION")])

    # Read Regional IAMs results
    df_iam = fun_read_df_iams(project, models)
    df_iam = fun_index_names(df_iam, True, int)

    # Harmonize IAMs data with historical EMISSIONS data
    model=models[0]
    regmap=fun_get_regions(project, model) # regional mapping
    if countrylist is not None:
        regmap={k:v for k,v in regmap.items() if len(set(v)&(set(countrylist)))>0} # select only regions with countries in countrylist
    res={}
    vars_to_be_harmo = [
                        'Emissions|CO2|Energy',
                        'Emissions|CO2|Energy|Demand|Transportation',
                        'Emissions|CO2|Energy|Demand|Industry',
                        'Emissions|CO2|Energy|Supply|Heat',
                        'Emissions|CO2|Energy|Supply|Electricity',
                        'Emissions|CO2|Energy|Demand|Residential and Commercial'
                    ]
    for r,countries in regmap.items():
        res[r]=fun_harmonize_df_iam_with_hist_data(f"{model}|{r[:-1]}", countries, df_iam,df, 
                    main_var = 'Emissions|CO2|Energy',
                    vars_to_be_harmo =vars_to_be_harmo,
                tc=2050 # From 2050 ownwards, we will keep original IAMs data (data no longer harmonized)
                )
    add=pd.concat(list(res.values()))
    df_iam= pd.concat([df_iam.drop(vars_to_be_harmo, level="VARIABLE"), add])


    # Slice for emissions only 
    emi_unit = ["Mt CO2/yr", "Mt CO2-equiv/yr"]
    emi_unit=emi_unit+list(non_co2.reset_index().UNIT.unique()) # add non-co2 units
    df=fun_xs(df, {"UNIT": emi_unit})

    # Exclude statistical differences variables
    excl = [x for x in df.reset_index().VARIABLE.unique() if "Statistica" in x]
    excl = ["Emissions|Total Non-CO2"] + excl
    df = fun_xs(df, {"VARIABLE": excl}, exclude_vars=True)

    # Create temporary variables, perform sectorial harmonization and remove temporary variables
    for k, v in step5f_temp_vars_dict.items():
        df = fun_create_var_as_sum(df, k, v)
    for k,v in step5f_temp_vars_dict_df_iam.items():
        df_iam = fun_create_var_as_sum(df_iam, k, v)
    
    if countrylist is not None:
        df = fun_xs(df, {"REGION": countrylist})

    # Slice for scenarios
    df = fun_xs(df, {"SCENARIO": scenarios})

    # Apply np.abs values to emissions data that are supposed to be positive (and they are not)
    res={}
    updated_data_list=[]
    for r,clist in regmap.items(): # do not use countrylist here!
        for scen in scenarios:
            dfrs=fun_xs(df, {'SCENARIO':scen, 'REGION':clist}) # dfrs means df region and scenario
            variables=dfrs.index.get_level_values('VARIABLE').unique()
            # Select variables that we want to `shrink`
            variables= [x for x in variables if 'Demand' in x or 'Supply' in x]
            for variable in variables:
                df1= fun_xs(dfrs, {'VARIABLE':variable})# slice variable
                for t in df1.columns:
                    if np.round(df1[t],5).min()*np.round(df1[t],5).max() < 0:
                        # print('Correction applied to', r, 'for variable', variable, 'at time', t)
                        if t<=2070: # we keep track only of data until 2070 (we do not share data beyond 2070)
                            info=f'Absolute value Correction applied to {r} for variable {variable} at time {t}'
                            print(info)
                            updated_data_list+=[info]
                        
                        # Apply shrinkage to the data (minimize `trade` volume)
                        # df1[t] = shrink(df1[t])
                        
                        df1[t] = np.abs(df1[t]) # apply absolute value
                        res[f"{r,scen,variable}"] = df1
    
    df_update=pd.concat(list(res.values())) # Get Updated results after negative values correction 
    df_update=fun_index_names(df_update, True, int)
    df=pd.concat([df.drop(df_update.index), df_update]) # Update df with corrected values


    
    # print(f"\n ### Clip values exceeding h_cpol scenario ###")
    # # NOTE Approch below will not work if
    # # 1) adjustments are applied to the subsectors,whereas the issue is present in the main sector (e.g. `Emissions|CO2|Energy`)
    # # 2) and for some reasons it won't work even if you adjust the main sectors.
    # variables=df.index.get_level_values('VARIABLE').unique()
    # variables= [x for x in variables if 'Demand' in x or 'Supply' in x]
    # variables_new=['Emissions|CO2', 
    #            'Emissions|CO2|Energy', 
    #            'Emissions|CO2|LULUCF Direct+Indirect', 
    #            'Emissions|Kyoto Gases (incl. indirect LULUCF)']
    # for var in variables+variables_new: # Adjust all sub-sectors variables
    #     res={} # dictionary to store updated results
    #     adjusted_countries=[] # list of countries with values above h_cpol
    #     clist_not_working=[] # list of countries not working
    #     for c in df.index.get_level_values('REGION').unique():
    #         if len(c)==3:
    #             try:
    #                 # Check if max value is above h_cpol
    #                 maxval=fun_xs(df, {'VARIABLE':var, 'REGION':c}).max()
    #                 refval=fun_xs(df, {'VARIABLE':var, 'REGION':c, 'SCENARIO':'h_cpol'})
    #                 check=(maxval/refval)[range(2010,2055,5)].iloc[0].max()
    #                 if check>1.001:
    #                     adjusted_countries+=[c] # Update list of adjusted countries
    #                     # Clip values below h_cpol
    #                     res[c]=clip_emissions_below_reference(df, c, var, 'h_cpol')
    #             except:
    #                 clist_not_working+=[c]
    #     print(f"{len(adjusted_countries)} Countries with values above h_cpol for {var}:", adjusted_countries)
    #     print( len(clist_not_working),"countries not working:", clist_not_working)
    #     if len(list(res.values()))>0:
    #         df_update=pd.concat(list(res.values())) # Get Updated results after negative values correction 
    #         df_update=fun_index_names(df_update, True, int)
    #         df=pd.concat([df.drop(df_update.index), df_update])


    # Sectorial harmonization (for each model,region,scenario)
    model=models[0]
    regmap=fun_get_regions(project, model) # regional mapping
    if countrylist is not None:
        regmap={k:v for k,v in regmap.items() if len(set(v)&(set(countrylist)))>0} # select only regions with countries in countrylist
    res={}
    ds=[step5f_dict1, step5f_dict2, step5f_dict3]
    for r,countrylist in regmap.items():
        for scen in scenarios:
            print('Harmonzing', model, r, scen)
            df1=fun_xs(df, {'SCENARIO':scen, 'REGION':countrylist}) # df
            df2=fun_rename_index_name(fun_xs(df_iam, {'REGION':f"{model}|{r[:-1]}", 'SCENARIO':scen}).drop(2005, axis=1), {'SCENARIO':'TARGET'}) # df_iam
            res[f"{model, r,scen}"] = run_sector_harmo_enhanced_iamc(project, df1, ds, df2, verbose=False)
            # If you want to enhance it, here you may consider using 
            # - `fun_harmonize_hist_data_by_preserving_sum_across_countries` to harmonize with historical data
            # - and then again ` run_sector_harmo_enhanced_iamc(project, df1, ds, df2, verbose=False)` to make sure results are consistent

    
    # Update results after harmonization
    df=pd.concat(list(res.values()))
    df=fun_index_names(df, True, int)

    # Remove temporary variables (this block does not seem to work properly - maybe could be commented out)
    d = {"VARIABLE": list(step5f_temp_vars_dict.keys())}
    df = fun_xs(df, d, exclude_vars=True)

    # Calculates international bunkers (World - sum across regions)
    var = "Emissions|CO2|Energy|Demand|Transportation"
    bunkers = fun_international_variables(project, var, models)
    bunkers = bunkers.rename({var: f"{var}|International Bunkers"})
    bunkers = fun_rename_index_name(bunkers, {"SCENARIO": "TARGET"})
    bunkers = fun_index_names(bunkers, True, int)
    if 'METHOD' in df.columns:
        df = df.drop('METHOD', axis=1)
    elif 'METHOD' in df.index.names:
        df = df.droplevel('METHOD')
    df = pd.concat([df, bunkers.reset_index().set_index(df.index.names)])

    # Add regional LULUCF data
    v = "Emissions|CO2|AFOLU"
    f = fun_rename_index_name
    df_iam = f(df_iam, {"SCENARIO": "TARGET"}).reset_index().set_index(df.index.names)
    df = pd.concat([df, df_iam.xs(v, level="VARIABLE", drop_level=False)])

    # Rename variables
    mydict={x:x.replace('LONG_TERM','') for x in df.index.get_level_values('VARIABLE')}
    df=df.rename(mydict, axis=0)
    
    # Fix units
    units = fun_get_variable_unit_dictionary(df_iam)
    d = {k: [v] for k, v in units.items()}
    df = fun_add_units(df, df_iam, fun_invert_dictionary(d))

    # select time period and add missing units
    df = df.rename({"missing": "Mt CO2/yr"}, level="UNIT")
    df = df.loc[:, [str(x) for x in range(2010, 2055, 5)]]

    # Save to CSV
    df.to_csv(CONSTANTS.CURR_RES_DIR(step) / csv_out)

    # Check if there are inconsistencies:
    for scen in scenarios:
        check=sum(fun_inconsistencies(fun_xs(df, {'TARGET':[scen]}), [step5f_dict1,step5f_dict2], by_sector=False, by_country=True).values())
        if check>0:
            show_inconsistencies(fun_xs(fun_rename_index_name(df , {'TARGET':'SCENARIO'}), {'SCENARIO':scen, "REGION":iea_countries}), [step5f_dict1,step5f_dict2], absolute=True)
            raise ValueError(f"{scen}: Consistency issues found: {check}")
        else:
            print(f"\n {scen}: Results are consistent!")

    # Check if sum across countries matches df_iam
    emiclockvar=set(fun_flatten_list(list(ds[0].values()))+fun_flatten_list(list(ds[1].values()))+list(ds[0].keys())+(list(ds[1].keys())))
    iamvar=set(df_iam.reset_index().VARIABLE.unique())
    checkvar=list(iamvar&emiclockvar)
    print("\n")
    print("## Check if sum across countries matches df_iam ##")
    for r,countrylist in regmap.items():
        for var in checkvar:
            dfsum=fun_xs(df, {"VARIABLE":var, "REGION":countrylist}).groupby(["TARGET"]).sum()
            iam=df_iam.xs((var, f"{model}|{r[:-1]}"), level=("VARIABLE","REGION")).groupby(["TARGET"]).sum()
            iam=fun_rename_index_name(fun_xs(iam, {"TARGET":scenarios}), {"SCENARIO":"TARGET"})[range(2010,2055,5)]
            dfsum=fun_index_names(dfsum, True, int)
            try:
                assert_frame_equal(np.round(dfsum.T,5), np.round(iam.T,5))
            except Exception as e:
                print(r, var, e, "\n")
                 
    
    # Drop tenmporary variables
    df=df.drop(list(step5f_temp_vars_dict.keys()), level="VARIABLE")

    # Save refined CSV (excl. temporary variables)
    df.to_csv(CONSTANTS.CURR_RES_DIR(step) / csv_out.replace(".csv","_sent.csv"))

    return df

# NOTE: you may need to re-run non-co2 emissions (step5c) using the   `run_multiple_file.py` before running this script (this will take approximately 10 mins)
if __name__ == "__main__":
    main(
        project="REMIND_2025_for_testing",
        csv_in = '2025_12_15_test', # NGFS data
        csv_out="Emissions_REMIND_2025_12_15_test.csv", # name of the output file
        step="step5",
        models=['*REMIND*'],  # None
        # countrylist=['CHN', 'HKG'], # Choose big countries (to avoid issues with missing data)
        #countrylist=['ABW', 'ARG', 'ATG', 'BHS', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'CHL', 'COL', 'CRI', 'CUB', 'CYM', 'DMA', 'DOM', 'ECU', 'FLK', 'GLP', 'GRD', 'GTM', 'GUF', 'GUY', 'HND', 'HTI', 'JAM', 'KNA', 'LCA', 'MEX', 'MSR', 'MTQ', 'NIC', 'PAN', 'PER', 'PRY', 'SLV', 'SUR', 'TCA', 'TTO', 'URY', 'VCT', 'VEN'] ,
        scenarios=["NPE-core"],# "h_ndc", "o_2c", "o_1p5c"],
              )


