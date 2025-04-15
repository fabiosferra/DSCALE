
from downscaler.Step_5e_visuals import fun_create_iea_file_IAMC_format
from downscaler import CONSTANTS

# Step1 - Create IEA file in IAMC format (`input_reference_iea_2022.csv`)
file='IEA_Balances_Rev2022.csv'
df = fun_create_iea_file_IAMC_format(CONSTANTS.INPUT_DATA_DIR/file)

# Step2 - Create Historical Trade Variable (`IEA_hist_trade_variables_v2022.csv`)
from downscaler.fixtures import *
from downscaler.utils import *
mydir=CONSTANTS.INPUT_DATA_DIR/"tmp/IEA_FORMAT"
# Read all files and concatenate
df_read=pd.concat([pd.read_csv(mydir/x) for x in os.listdir(mydir)])
# Slice flows
df_read=fun_index_names(df_read[df_read.FLOW.isin(["IMPORTS","EXPORTS"])], True, int)
mydict_general={'Trade|Primary Energy|Biomass|Volume':iea_biomass, 'Trade|Primary Energy|Coal|Volume':iea_coal, 'Trade|Primary Energy|Gas|Volume':iea_gas, 'Trade|Primary Energy|Oil|Volume':iea_oil}

df_all=pd.DataFrame()
for var, myflow in mydict_general.items():
    # Slice for selected products
    mydict=SliceableDict({v:k for k,v in iea_product_long_short_dict.items() }).slice(*tuple(myflow))
    df=df_read.rename(iea_product_long_short_dict, level="PRODUCT")
    df=fun_xs(df, {"PRODUCT":myflow})
    # Calculate sum
    df=(df.dropna(how="all").groupby("ISO").sum()*(-1))
    # Convert from TJ to EJ
    df=df * 1e-6
    # Add model, scenario, variable, unit  columns and concat 
    df=df.assign(VARIABLE=var).assign(SCENARIO='Historic data').assign(MODEL="IEA").assign(UNIT="EJ/yr")
    df_all=pd.concat([df_all,df])
df_all.index.names=["REGION"]
df_all=df_all.reset_index().set_index(['MODEL', 'SCENARIO', 'REGION', 'VARIABLE', 'UNIT'])
df_all.to_csv(CONSTANTS.INPUT_DATA_DIR/"IEA_hist_trade_variables_v2022.csv")