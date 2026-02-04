"""
This script calculates the synthetic fuels per-country, by looking at the difference between the
sum of current fuels (e.g. biomass + oil) and the total energy carrier (e.g. liquids), 
and ascertaining the 'missing' synfuels as a result

MORE DETAILS:
We create this `synfuel` variable as the difference between total liquids and the sum of sub-sectors
(excluding electricity). Then we harmonize the results by using the `run_sector_harmo_enhanced_iamc`
(that ensures consistency with regional IAMs results if provided) 

NOTE: if you want to add historical harmonization you need to be sure that `df_iea` and `df_iam` 
are also fully aligned (otherwise you may create inconsistencies)
NOTE: Given that synfuel='Electricity' this code does not work for energy_carrier= Electricity
"""

# -----------------------------------
# PACKAGE IMPORTS -------------------
# -----------------------------------
import logging
import pandas as pd
import matplotlib.pyplot as plt

from downscaler import CONSTANTS

from downscaler.utils import (
    fun_check_negative_energy_variables,
    fun_divide_variable_by_another,
    fun_flatten_list,
    fun_get_models,
    fun_get_regions,
    fun_invert_dictionary,
    fun_read_df_iams,
    fun_regional_country_mapping_as_dict,
    fun_xs_fuzzy,
    iea_countries,
    run_sector_harmo_enhanced_iamc,
    show_inconsistencies,
)
from downscaler.utils_dictionary import fun_append_list_of_dicts

from downscaler.utils_pandas import (
    fun_add_units,
    fun_create_var_as_sum,
    fun_get_variable_unit_dictionary,
    fun_index_names,
    fun_read_csv,
    fun_rename_index_name,
    fun_xs,
)

# -----------------------------------
# LOGGING CONFIG -------------------
# -----------------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------------
# CONFIG  -------------------
# -----------------------------------

# Define project, scenarios and file name with downscaled results
project="REMIND_fuel_mix_testing"
scenarios=['NPE-core'] # to be updated
file='REMIND-MAgPIE 3.3-4.8_REMIND_fuel_mix_testing_fuel_mix_testing_03022026_2022_harmo_step5e_None.csv' # your file name with downscaled results
synfuel='Electricity' # name of your synfuel variable (e.g. 'Final Energy|Transportation|Liquids|Electricity')

# Select regions to be harmonized (if None, all regions/countries will be harmonized) and number of iterations
sel_reg='LAMr' # None # # Region to be harmonized (e.g. LAMr for Latin America). Select None if you want to apply this to all regions/countries
iterations=15 # number of harmonization iterations (with 15/20 you get fully consistent results, but it takes longer)

# Decide if to show the sanity check plots
sanity_plot = False

# -----------------------------------
# FUNCTION DEFINITIONS---------------
# -----------------------------------

def reshape_df_after_harmo_in_IAMC_format(df_iam, df):
    """TODO: Add docstring for function"""
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

# -----------------------------------
# MAIN SCRIPT  -------------------
# -----------------------------------

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



for sector in ['Transportation', 'Industry', 'Residential and Commercial']:
    for carrier in ['Liquids','Gases']:
        mydict={f'Final Energy|{sector}|{carrier}':
            [
                f'Final Energy|{sector}|{carrier}|Biomass',
                f'Final Energy|{sector}|{carrier}|Coal',
                f'Final Energy|{sector}|{carrier}|Electricity',
                f'Final Energy|{sector}|{carrier}|Natural Gas',
                f'Final Energy|{sector}|{carrier}|Oil'
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
                                                        f'Final Energy|{sector}|{carrier}|{synfuel}',  # New `synfuel` variable to be created as the difference of:
                                                        f'Final Energy|{sector}|{carrier}', 
                                                        'sum of subs', 
                                                        operator='-', # subtraction 
                                                        unit='EJ/yr', 
                                                        concatenate=True).clip(1e-9) # we clip to avoid negative values (if you want to keep negative values, remove the .clip(0.01) part)



                    # Harmonize results (top-down harmonization)
                    temp = run_sector_harmo_enhanced_iamc(project, temp, ds, df2, verbose=False, no_iter=1) # To enhance consistency, increase the number of iterations (no_iter) 
                    temp =reshape_df_after_harmo_in_IAMC_format(df_iam, temp)
                    
                    # Drop the temporary variable `sum of subs` (created to calculate the Synfuel variable)
                    temp=temp.drop('sum of subs', level='VARIABLE') # raise error if we find negative values for Final energy variables
                  
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
        # These are just some Sanity Check of results that can be added
        # ------------- 

        if sanity_plot:
            # 1) Check inconsistencies after harmonization for individual countries (to minimize inconsistencies, please increase the number of iterations above)
            print(show_inconsistencies(fun_xs(temp, {'REGION':iea_countries}), ds))

            # 2) Compare sum of country level results with IAMs for `Transportation|Liquids|Biomass`
            plt.style.use('seaborn-white')
            for x in [f'Final Energy|{sector}|{carrier}|Biomass', f'Final Energy|{sector}|{carrier}|Electricity']:
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
# SAVE RESULTS
# ------------- 
df.to_csv(f'{mydir}/REMIND-MAgPIE 3.3-4.8_REMIND_fuel_mix_testing_fuel_mix_testing_03022026_2022_harmo_step5g_synfuel.csv')

