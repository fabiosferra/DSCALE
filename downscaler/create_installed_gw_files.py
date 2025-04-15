import pandas as pd
import os
from pathlib import Path
from downscaler import CONSTANTS, IFT
from downscaler.fixtures import lifetime_dict
from downscaler.utils import fun_capital_projections, fun_load_platts, fun_read_df_countries, setindex

# CAPITAL VINTAGING PROJECTIONS 
RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR("step2")
df_platts = fun_load_platts()

df_countries = fun_read_df_countries(
        CONSTANTS.INPUT_DATA_DIR / "MESSAGE_CEDS_region_mapping_2020_02_04.csv",
    )
# Reading Platts fuel dictionary for standard fuel name mapping
df_fuel_dict = pd.read_csv(CONSTANTS.INPUT_DATA_DIR / "Fuel_dict.csv", sep=",")

setindex(df_fuel_dict, "PLATTS")
fuel_dict = df_fuel_dict.to_dict()

df_platts["FUEL_IAM"] = df_platts["FUEL"].map(fuel_dict["FUEL"])

df_gw_all_fuels=pd.DataFrame()
for f in ['GEO','OIL','COAL','GAS']: # fuel list
    status=['OPR','PLN','CON'] # selected status
    string='df_gw_all_'+f
    print("working on",f)
    fun_capital_projections(RESULTS_DATA_DIR, df_platts, pd.DataFrame(range(2005,2105,5)).sort_values([0], ascending=False).set_index([0]), df_countries, False, f,status
                      ,lifetime_dict[f], _show_plot=False, _save_csv=True)
