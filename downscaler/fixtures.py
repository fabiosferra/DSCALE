import pandas as pd


## Reference countries for comparison below:
ref_countries = ["DEU", "USA", "JPN", "DEU", "FRA", "GBR", "CHN", "IND"]

## SSP selection:
## we select the GDP AND POPULATION
gdp = "GDP|PPP"  # (unit =  2005$, contrary to IEA data which is in 2010$)
pop = "Population"
# pop_ssp= 'Population' # Population, Population by age structure etc.
pop_iea = "POPULATION"

## Final energy sectors
fen_sectors = [
    "Final Energy",
    "Final Energy|Liquids",
    "Final Energy|Transportation|Liquids",
    "Final Energy|Residential and Commercial|Liquids",
    "Final Energy|Industry|Liquids",
    "Final Energy|Gases",
    "Final Energy|Transportation|Gases",
    "Final Energy|Residential and Commercial|Gases",
    "Final Energy|Industry|Gases",
    ## TO DO: ## do a new function that checks if you are using the same sector for  both liquids and gases
    "Final Energy|Solids",  ## new 2020_09_30
    #'Final Energy|Transportation|Solids',
    "Final Energy|Industry|Solids",
    "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Electricity",
    "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
    "Final Energy|Industry|Electricity",
]

## Secondary energy sectors
sen_sectors = [
    "Secondary Energy|Electricity|Hydro",
    "Secondary Energy|Electricity|Geothermal",
    "Secondary Energy|Electricity|Coal",
    "Secondary Energy|Electricity|Oil",
    "Secondary Energy|Electricity|Gas",
    "Secondary Energy|Electricity|Nuclear",
    "Secondary Energy|Electricity|Wind",
    "Secondary Energy|Electricity|Solar",
    "Secondary Energy|Electricity|Biomass",
]

pen_sectors = fen_sectors + sen_sectors

## NOTE: Here the order is different from fen_sectors.
#  NOTE: If we change the order of sectors here the test_case (test1 for final energy) will fail
sectors_energy_demand = [
    "Final Energy",
    "Final Energy|Solids",
    "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Transportation|Solids",
    "Final Energy|Industry|Solids",
    "Final Energy|Liquids",
    "Final Energy|Transportation|Liquids",
    "Final Energy|Residential and Commercial|Liquids",
    "Final Energy|Industry|Liquids",
    "Final Energy|Gases",
    "Final Energy|Transportation|Gases",
    "Final Energy|Residential and Commercial|Gases",
    "Final Energy|Industry|Gases",
    "Final Energy|Electricity",
    "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
    "Final Energy|Industry|Electricity",
]

sectors_energy_demand_split_res = [
    "Final Energy",
    "Final Energy|Solids",
    # "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Residential|Solids",
    "Final Energy|Commercial|Solids",
    "Final Energy|Transportation|Solids",
    "Final Energy|Industry|Solids",
    "Final Energy|Liquids",
    "Final Energy|Transportation|Liquids",
    # "Final Energy|Residential and Commercial|Liquids",
    "Final Energy|Residential|Liquids",
    "Final Energy|Commercial|Liquids",
    "Final Energy|Industry|Liquids",
    "Final Energy|Gases",
    "Final Energy|Transportation|Gases",
    # "Final Energy|Residential and Commercial|Gases",
    "Final Energy|Residential|Gases",
    "Final Energy|Commercial|Gases",
    "Final Energy|Industry|Gases",
    "Final Energy|Electricity",
    "Final Energy|Transportation|Electricity",
    # "Final Energy|Residential and Commercial|Electricity",
    "Final Energy|Residential|Electricity",
    "Final Energy|Commercial|Electricity",
    "Final Energy|Industry|Electricity",
]


## We initialise sectors to be equal to fen_sectors
sectors = fen_sectors

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

# NOTE `iea_other` calculated as follow:
# [x for x in df_iea_melt.PRODUCT.unique() if "other" in x.lower() if x not in iea_gases+iea_liquids+iea_solids]
iea_other = ["Other recovered gases", "Geothermal", "Other sources"]
iea_liquids_gases = iea_liquids + iea_gases

## Individual fuel lists for primary energy downscaling (non electric sector):

iea_gas = [
    "Natural gas"
]  ## 2021_03_26 update ( same definition as in the df_iea_melt['IAM_FUEL'])
iea_natural_gas = iea_gas

## Updated 2021_03_26 ( same definition as in the df_iea_melt['IAM_FUEL'])
iea_oil = [
    "Crude/NGL/feedstocks (if no detail)",
    "Crude oil",
    "Natural gas liquids",
    "Refinery feedstocks",
    "Additives/blending components",
    "Other hydrocarbons",
    "Refinery gas",
    "Ethane",
    "Liquefied petroleum gases (LPG)",
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
]

## 2021_03_26 update ( same definition as in the df_iea_melt['IAM_FUEL'])
iea_coal = [
    "Hard coal (if no detail)",
    "Brown coal (if no detail)",
    "Anthracite",
    "Coking coal",
    "Other bituminous coal",
    "Sub-bituminous coal",
    "Lignite",
    "Patent fuel",
    "Coke oven coke",
    "Gas coke",
    "Coal tar",
    "BKB",
    "Gas works gas",
    "Coke oven gas",
    "Blast furnace gas",
    "Other recovered gases",
    "Peat",
    "Peat products",
    "Oil shale and oil sands",
]

iea_biomass = [
    "Biogases",
    "Primary solid biofuels",
    "Non-specified primary biofuels and waste",
    "Biogasoline",
    "Biodiesels",
    "Bio jet kerosene",
    "Other liquid biofuels",
]

iea_flow_dict = {
    ## Data stucture  {sectors:['IEA FLOW', 'IEA PRODUCT','UNIT', scale (graph purposes)]}
    gdp: ["", "", "Billion USD PPP"],  ## We just need this for the UNIT value
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
    "Final Energy|Industry|Solids|Biomass": [
        ["Industry"],
        (list(set(iea_biomass).intersection(iea_solids))),
        ["ktoe"],
        1e2,
    ],  ## intersection of two lists
    "Final Energy|Industry|Solids|Coal": [
        ["Industry"],
        (list(set(iea_coal).intersection(iea_solids))),
        ["ktoe"],
        1e2,
    ],  ## intersection of two lists
    "Final Energy|Residential and Commercial|Solids|Coal": [
        ["Residential", "Commercial and public services"],
        (list(set(iea_coal).intersection(iea_solids))),
        ["ktoe"],
        1e2,
    ],  ## intersection of two lists
    "Final Energy|Residential and Commercial|Solids|Biomass": [
        ["Residential", "Commercial and public services"],
        (list(set(iea_biomass).intersection(iea_solids))),
        ["ktoe"],
        1e2,
    ],  ## intersection of two lists
    "Final Energy|Heat": [
        ["Heat output"],
        list(set(iea_gases + iea_liquids + iea_solids))
        + ["Heat output from non-specified combustible fuels"],
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Residential and Commercial|Heat": [
        ["Residential", "Commercial and public services"],
        ["Heat"],
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Industry|Heat": [["Industry"], ["Heat"], ["ktoe"], 1e2],
    "Final Energy|Hydrogen": [
        ["Hydrogen"],
        list(set(iea_gases + iea_liquids + iea_solids))
        + ["Heat output from non-specified combustible fuels"],
        ["ktoe"],
        1e2,
    ],
    ## Block Added 2021_02_03:
    "Secondary Energy|Electricity": [
        ["Electricity output (GWh)"],
        ["Total"],
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Hydro": [
        ["Electricity output (GWh)"],
        ["Hydro"],
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Coal": [
        ["Electricity output (GWh)"],
        iea_coal,
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Oil": [
        ["Electricity output (GWh)"],
        iea_oil,
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Gas": [
        ["Electricity output (GWh)"],
        iea_gas,
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Nuclear": [
        ["Electricity output (GWh)"],
        ["Nuclear"],
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Wind": [
        ["Electricity output (GWh)"],
        ["Wind"],
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Solar": [
        ["Electricity output (GWh)"],
        ["Solar photovoltaics", "Solar thermal"],
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Biomass": [
        ["Electricity output (GWh)"],
        iea_biomass,
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Electricity|Geothermal": [
        ["Electricity output (GWh)"],
        ["Geothermal"],
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Liquids": [
        ["Total final consumption"],
        iea_liquids,
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Hydro": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(["Hydro"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Coal": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(iea_coal)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Oil": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(iea_oil)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Gas": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(["Natural gas"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Natural Gas": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(["Natural gas"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Nuclear": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(["Nuclear"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Wind": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(["Wind"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Solar": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(["Solar photovoltaics", "Solar thermal"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Biomass": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(iea_biomass)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Liquids|Geothermal": [
        ["Total final consumption"],
        list(set(iea_liquids).intersection(["Geothermal"])),
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Gases": [
        ["Total final consumption"],
        iea_gases,
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Hydro": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(["Hydro"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Coal": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(iea_coal)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Oil": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(iea_oil)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Gas": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(["Natural gas"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Natural Gas": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(["Natural gas"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Geothermal": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(["Geothermal"])),
        ["GWh"],
        1e2,
    ],
    "Secondary Energy|Gases|Nuclear": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(["Nuclear"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Wind": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(["Wind"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Solar": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(["Solar photovoltaics", "Solar thermal"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Gases|Biomass": [
        ["Total final consumption"],
        list(set(iea_gases).intersection(iea_biomass)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids": [
        ["Total final consumption"],
        iea_solids,
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Hydro": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(["Hydro"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Coal": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(iea_coal)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Oil": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(iea_oil)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Gas": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(["Natural gas"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Natural Gas": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(["Natural gas"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Nuclear": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(["Nuclear"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Wind": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(["Wind"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Solar": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(["Solar photovoltaics", "Solar thermal"])),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Biomass": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(iea_biomass)),
        ["ktoe"],
        1e2,
    ],
    "Secondary Energy|Solids|Geothermal": [
        ["Total final consumption"],
        list(set(iea_solids).intersection(["Geothermal"])),
        ["GWh"],
        1e2,
    ],
    "Primary Energy": [["Total primary energy supply"], ["Total"], ["ktoe"], 1e2],
    "Primary Energy|Hydro": [
        ["Total primary energy supply"],
        ["Hydro"],
        ["ktoe"],
        1e2,
    ],
    "Primary Energy|Coal": [
        ["Total primary energy supply"],
        iea_coal,
        ["ktoe"],
        1e2,
    ],
    "Primary Energy|Oil": [["Total primary energy supply"], iea_oil, ["ktoe"], 1e2],
    "Primary Energy|Gas": [["Total primary energy supply"], iea_gas, ["ktoe"], 1e2],
    "Primary Energy|Nuclear": [
        ["Total primary energy supply"],
        ["Nuclear"],
        ["ktoe"],
        1e2,
    ],
    "Primary Energy|Wind": [
        ["Total primary energy supply"],
        ["Wind"],
        ["ktoe"],
        1e2,
    ],
    "Primary Energy|Solar": [
        ["Total primary energy supply"],
        ["Solar photovoltaics", "Solar thermal"],
        ["ktoe"],
        1e2,
    ],
    "Primary Energy|Biomass": [
        ["Total primary energy supply"],
        iea_biomass,
        ["ktoe"],
        1e2,
    ],
    "Trade|Primary Energy|Coal|Volume": [
        ["Imports", "Exports"],
        iea_coal,
        ["ktoe"],
        1e2,
    ],
    "Trade|Primary Energy|Oil|Volume": [
        ["Imports", "Exports"],
        iea_oil,
        ["ktoe"],
        1e2,
    ],
    "Trade|Primary Energy|Gas|Volume": [
        ["Imports", "Exports"],
        iea_gases,
        ["ktoe"],
        1e2,
    ],
    "Trade|Primary Energy|Biomass|Volume": [
        ["Imports", "Exports"],
        iea_biomass,
        ["ktoe"],
        1e2,
    ],
    "Final Energy|Residential": [["Residential"], ["Total"], ["ktoe"], 100.0],
    "Final Energy|Residential|Solids": [["Residential"], iea_solids, ["ktoe"], 100.0,],
    "Final Energy|Residential|Electricity": [
        ["Residential"],
        ["Electricity"],
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Residential|Heat": [["Residential"], ["Heat"], ["ktoe"], 100.0],
    "Final Energy|Residential|Liquids": [
        ["Residential"],
        iea_liquids,
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Residential|Gases": [["Residential"], iea_gases, ["ktoe"], 100.0,],
    "Final Energy|Residential|Solids|Coal": [
        ["Residential"],
        list(set(iea_solids).intersection(iea_coal)),
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Residential|Solids|Biomass": [
        ["Residential"],
        list(set(iea_solids).intersection(iea_biomass)),
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Commercial": [
        ["Commercial and public services"],
        ["Total"],
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Commercial|Solids": [
        ["Commercial and public services"],
        iea_solids,
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Commercial|Electricity": [
        ["Commercial and public services"],
        ["Electricity"],
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Commercial|Heat": [
        ["Commercial and public services"],
        ["Heat"],
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Commercial|Liquids": [
        ["Commercial and public services"],
        iea_liquids,
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Commercial|Gases": [
        ["Commercial and public services"],
        iea_gases,
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Commercial|Solids|Coal": [
        ["Commercial and public services"],
        list(set(iea_solids).intersection(iea_coal)),
        ["ktoe"],
        100.0,
    ],
    "Final Energy|Commercial|Solids|Biomass": [
        ["Commercial and public services"],
        list(set(iea_solids).intersection(iea_biomass)),
        ["ktoe"],
        100.0,
    ],
}

## dict_y_den detetrmines the denominator of Y=NUM/DEN. This is the denominator of Y (of the log-log graph)
dict_y_den = {
    "Final Energy": gdp,
    "Final Energy|Industry": "Final Energy",
    "Final Energy|Residential and Commercial": "Final Energy",
    "Final Energy|Transportation": "Final Energy",
    "Final Energy|Liquids&Gases": "Final Energy",
    "Final Energy|Transportation|Liquids&Gases": "Final Energy|Liquids&Gases",
    "Final Energy|Residential and Commercial|Liquids&Gases": "Final Energy|Liquids&Gases",
    "Final Energy|Industry|Liquids&Gases": "Final Energy|Liquids&Gases",
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
    "Final Energy|Industry|Solids|Biomass": "Final Energy|Industry|Solids",
    "Final Energy|Industry|Solids|Coal": "Final Energy|Industry|Solids",
    "Final Energy|Residential and Commercial|Solids|Coal": "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Residential and Commercial|Solids|Biomass": "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Hydrogen": "Final Energy",
    "Final Energy|Heat": "Final Energy",
    "Final Energy|Residential and Commercial|Heat": "Final Energy|Heat",
    "Final Energy|Residential and Commercial|Heat|Biomass": "Final Energy|Residential and Commercial|Heat",
    "Final Energy|Residential and Commercial|Heat|Coal": "Final Energy|Residential and Commercial|Heat",
    "Final Energy|Residential and Commercial|Heat|Oil": "Final Energy|Residential and Commercial|Heat",
    "Final Energy|Residential and Commercial|Heat|Gas": "Final Energy|Residential and Commercial|Heat",
    "Final Energy|Residential and Commercial|Heat|Natural Gas": "Final Energy|Residential and Commercial|Heat",
    "Final Energy|Industry|Heat": "Final Energy|Heat",
    "Final Energy|Industry|Heat|Biomass": "Final Energy|Industry|Heat",
    "Final Energy|Industry|Heat|Coal": "Final Energy|Industry|Heat",
    "Final Energy|Industry|Heat|Oil": "Final Energy|Industry|Heat",
    "Final Energy|Industry|Heat|Gas": "Final Energy|Industry|Heat",
    "Final Energy|Industry|Heat|Natural Gas": "Final Energy|Industry|Heat",
    ## Block Added 2021_02_03:
    "Secondary Energy|Electricity|Hydro": "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Coal": "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Oil": "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Gas": "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Nuclear": "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Wind": "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Solar": "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Biomass": "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Geothermal": "Secondary Energy|Electricity",
    "Secondary Energy|Liquids|Hydro": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Coal": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Oil": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Gas": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Natural Gas": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Nuclear": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Wind": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Solar": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Biomass": "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Geothermal": "Secondary Energy|Liquids",
    "Secondary Energy|Gases|Hydro": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Coal": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Oil": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Gas": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Natural Gas": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Nuclear": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Wind": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Solar": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Biomass": "Secondary Energy|Gases",
    "Secondary Energy|Gases|Geothermal": "Secondary Energy|Gases",
    "Secondary Energy|Solids|Hydro": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Coal": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Oil": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Gas": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Natural Gas": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Nuclear": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Wind": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Solar": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Biomass": "Secondary Energy|Solids",
    "Secondary Energy|Solids|Geothermal": "Secondary Energy|Solids",
    "Primary Energy": "Primary Energy",
    "Primary Energy|Hydro": "Primary Energy",
    "Primary Energy|Coal": "Primary Energy",
    "Primary Energy|Oil": "Primary Energy",
    "Primary Energy|Gas": "Primary Energy",
    "Primary Energy|Nuclear": "Primary Energy",
    "Primary Energy|Wind": "Primary Energy",
    "Primary Energy|Solar": "Primary Energy",
    "Primary Energy|Biomass": "Primary Energy",
    "Primary Energy|Geothermal": "Primary Energy",
    "Final Energy|Residential": "Final Energy",
    "Final Energy|Residential|Liquids": "Final Energy|Liquids",
    "Final Energy|Residential|Gases": "Final Energy|Gases",
    "Final Energy|Residential|Solids": "Final Energy|Solids",
    "Final Energy|Residential|Electricity": "Final Energy|Electricity",
    "Final Energy|Residential|Solids|Coal": "Final Energy|Residential|Solids",
    "Final Energy|Residential|Solids|Biomass": "Final Energy|Residential|Solids",
    "Final Energy|Residential|Heat": "Final Energy|Heat",
    "Final Energy|Residential|Heat|Biomass": "Final Energy|Residential|Heat",
    "Final Energy|Residential|Heat|Coal": "Final Energy|Residential|Heat",
    "Final Energy|Residential|Heat|Oil": "Final Energy|Residential|Heat",
    "Final Energy|Residential|Heat|Gas": "Final Energy|Residential|Heat",
    "Final Energy|Residential|Heat|Natural Gas": "Final Energy|Residential|Heat",
    "Final Energy|Commercial": "Final Energy",
    "Final Energy|Commercial|Liquids": "Final Energy|Liquids",
    "Final Energy|Commercial|Gases": "Final Energy|Gases",
    "Final Energy|Commercial|Solids": "Final Energy|Solids",
    "Final Energy|Commercial|Electricity": "Final Energy|Electricity",
    "Final Energy|Commercial|Solids|Coal": "Final Energy|Commercial|Solids",
    "Final Energy|Commercial|Solids|Biomass": "Final Energy|Commercial|Solids",
    "Final Energy|Commercial|Heat": "Final Energy|Heat",
    "Final Energy|Commercial|Heat|Biomass": "Final Energy|Commercial|Heat",
    "Final Energy|Commercial|Heat|Coal": "Final Energy|Commercial|Heat",
    "Final Energy|Commercial|Heat|Oil": "Final Energy|Commercial|Heat",
    "Final Energy|Commercial|Heat|Gas": "Final Energy|Commercial|Heat",
    "Final Energy|Commercial|Heat|Natural Gas": "Final Energy|Commercial|Heat",
}

IAM_fuel_dict = {
    "|Nuclear": "NUC",
    "|Coal": "COAL",
    "|Gas": "GAS",
    "|Oil": "OIL",
    "|Non-biomass renewables": "REN",
    "|Biomass": "BIO",
    "|Solar": "SOL",
    "|Wind": "WIND",
    "|Hydro": "HYDRO",
    "|Geothermal": "GEO",
    "|Hydrogen": "H2",
    "": "TOTAL",
}  ## IAM fuel dictionary


fuel_list_steb2b = [
    "COAL",
    "NUC",
    "GAS",
    "BIO",
    "SOL",
    "WIND",
    "HYDRO",
    "OIL",
    "GEO",
]

file_name_dict = {
    "SOL": "CostCurve_PV",
    "WIND": "CostCurve_onshore_Wind",
    "HYDRO": "CostCurveHYD_lpjml_gfdl-esm2m_ewembi_rcp26_rcp26soc_co2_qtot_global_daily_2031_2070_merge_monmean_Hydro",
    "BIO": "MaxProd_1stGen_Bio",
}

# LIFETIME ASSUMPTIONS FOR POWER PLANTS (years)
lifetime_dict = {
    "COAL": 50,
    "GAS": 40,
    "OIL": 30,
    "GEO": 30,
}

sectors_step2b = [
    "Final Energy",
    "Final Energy|Solids",  ## new 2020_09_30
    "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Transportation|Solids",
    "Final Energy|Industry|Solids",
    "Final Energy|Liquids",
    "Final Energy|Transportation|Liquids",
    "Final Energy|Residential and Commercial|Liquids",
    "Final Energy|Industry|Liquids",
    "Final Energy|Gases",
    "Final Energy|Transportation|Gases",
    "Final Energy|Residential and Commercial|Gases",
    "Final Energy|Industry|Gases",
    ## TO DO: ## do a new function that checks if you are using the same sector for  both liquids and gases
    "Final Energy|Electricity",
    "Final Energy|Transportation|Electricity",
    "Final Energy|Residential and Commercial|Electricity",
    "Final Energy|Industry|Electricity",
]  # This is the main sector




# default electricity weights criteria 2021_09_03 (for electricity downscaling => step 2b)
data = {
    "IAM_FUEL": ["SOL", "WIND", "BIO", "HYDRO", "COAL", "GAS", "OIL", "GEO", "NUC"],
    "df_cost_criteria": [
        0.35,
        0.35,
        0.50,
        0.50,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ],  ## SUPPLY COST CURVES
    "df_gw_all_fuels": [
        0.00,
        0.00,
        0.00,
        0.00,
        0.50,
        0.50,
        0.50,
        0.50,
        0.00,
    ],  ## STRANDED ASSETS
    "df_base_year_share": [
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.50,
        0.850,
    ],  ## BASE YEAR SHARE
    "df_gov": [0.15, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.150],
}  ## GOVERNANCE

df_weights = pd.DataFrame(data)

vars_to_be_harmo_step2b = [
    "Secondary Energy|Electricity|Biomass",
    "Secondary Energy|Electricity|Biomass|w/o CCS",
    "Secondary Energy|Electricity|Coal",
    "Secondary Energy|Electricity|Coal|w/o CCS",
    "Secondary Energy|Electricity|Gas",
    "Secondary Energy|Electricity|Gas|w/o CCS",
    "Secondary Energy|Electricity|Geothermal",
    "Secondary Energy|Electricity|Hydro",
    "Secondary Energy|Electricity|Nuclear",
    "Secondary Energy|Electricity|Oil",
    "Secondary Energy|Electricity|Oil|w/o CCS",
    "Secondary Energy|Electricity|Solar",
    "Secondary Energy|Electricity|Wind",
]
vars_to_be_harmo_step2a = [
    "Secondary Energy|Electricity",
    "Secondary Energy|Gases",
    "Secondary Energy|Gases|Biomass",
    "Secondary Energy|Gases|Coal",
    "Secondary Energy|Gases|Natural Gas",
    "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Biomass",
    "Secondary Energy|Liquids|Coal",
    # "Secondary Energy|Liquids|Fossil",
    "Secondary Energy|Liquids|Gas",
    "Secondary Energy|Liquids|Oil",
    "Secondary Energy|Solids|Biomass",
    "Secondary Energy|Solids",
    "Secondary Energy|Solids|Coal",
    "Primary Energy",
    "Primary Energy|Biomass",
    "Primary Energy|Coal",
    "Primary Energy|Coal|w/ CCS",
    "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Fossil",
    "Primary Energy|Fossil|w/ CCS",
    "Primary Energy|Fossil|w/o CCS",
    "Primary Energy|Gas",
    "Primary Energy|Gas|w/ CCS",
    "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Oil",
    "Primary Energy|Oil|w/o CCS",
    "Primary Energy|Non-Biomass Renewables|Geothermal",
    "Primary Energy|Non-Biomass Renewables|Hydro",
    "Primary Energy|Non-Biomass Renewables|Solar",
    "Primary Energy|Non-Biomass Renewables|Wind",
    "Secondary Energy|Electricity|Geothermal",  # we clip geothermal and hydro in step 2a
    "Secondary Energy|Electricity|Hydro",  # we clip geothermal and hydro in step 2a
    "Primary Energy|Nuclear"
    # "Primary Energy|Solar",
    # "Primary Energy|Wind",
]

vars_to_be_harmo_step2 = vars_to_be_harmo_step2a + vars_to_be_harmo_step2b

# below we exclude GDP and Population because we do provide data for 184 countries (some countries that we do not report have GDP/pop projections)
# we exclude revenues bacause we add them afterwards (in step 5b)
vars_to_be_harmo_step5 = [
    "Carbon Sequestration|CCS",
    "Carbon Sequestration|CCS|Biomass",
    "Carbon Sequestration|CCS|Fossil",
    "Carbon Sequestration|CCS|Industrial Processes",
    "Emissions|CO2|Energy",
    "Final Energy",
    "Final Energy|Electricity",
    "Final Energy|Gases",
    "Final Energy|Heat",
    "Final Energy|Hydrogen",
    "Final Energy|Industry",
    "Final Energy|Industry|Electricity",
    "Final Energy|Industry|Gases",
    "Final Energy|Industry|Heat",
    "Final Energy|Industry|Hydrogen",
    "Final Energy|Industry|Liquids",
    "Final Energy|Industry|Solids",
    "Final Energy|Industry|Solids|Biomass",  # message did not work properly FSU
    "Final Energy|Industry|Solids|Coal",  # message did not work properly SSA
    "Final Energy|Liquids",
    "Final Energy|Residential and Commercial",
    "Final Energy|Residential and Commercial|Electricity",
    "Final Energy|Residential and Commercial|Gases",
    "Final Energy|Residential and Commercial|Heat",
    "Final Energy|Residential and Commercial|Liquids",
    "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Residential and Commercial|Solids|Biomass",
    "Final Energy|Residential and Commercial|Solids|Coal",  # this one did not work properly for remind (very small 0.01 ej difference in EU 28)
    'Final Energy|Residential and Commercial|Hydrogen',
    "Final Energy|Solids",
    "Final Energy|Transportation",
    "Final Energy|Transportation|Electricity",
    "Final Energy|Transportation|Gases",  ## message did not work properly CHN (small differences)
    "Final Energy|Transportation|Hydrogen",
    "Final Energy|Transportation|Liquids",
] + vars_to_be_harmo_step2

list_of_fuels = [
    "Biomass",
    "Coal",
    "Gas",
    "Geothermal",
    "Hydro",
    "Nuclear",
    "Oil",
    "Solar",
    "Wind",
    "Natural Gas",
    "Other",
]
list_of_ec = ["Liquids", "Solids", "Gases", "Hydrogen", "Heat", "Electricity"]
list_of_sectors = ["Residential and Commercial", "Industry", "Transportation"]

secondary_energy_list = [
    "Secondary Energy|Gases",
    "Secondary Energy|Gases|Biomass",
    "Secondary Energy|Gases|Coal",
    "Secondary Energy|Gases|Natural Gas",
    "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Biomass",
    "Secondary Energy|Liquids|Coal",
    "Secondary Energy|Liquids|Gas",
    "Secondary Energy|Liquids|Oil",
    "Secondary Energy|Gases|Other",
    "Secondary Energy|Solids",
    "Secondary Energy|Solids|Biomass",
    "Secondary Energy|Solids|Coal",
]

ec_list = ["Final Energy", "Electricity", "Solids", "Liquids", "Gases"]
hydrogen_list = ["Final Energy|Transportation", "Final Energy|Industry", 'Final Energy|Residential and Commercial']
clipped_fuels_2010 = [
    "Secondary Energy|Electricity|GeothermalENSHORT_REF",
    "Secondary Energy|Electricity|HydroENSHORT_REF",
    "Secondary Energy|Electricity|GeothermalENLONG_RATIO",
    "Secondary Energy|Electricity|HydroENLONG_RATIO",
]


sectors_adj = [
    "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Transportation|Solids",
    "Final Energy|Industry|Solids",
    "Final Energy|Transportation|Liquids",
    "Final Energy|Residential and Commercial|Liquids",
    "Final Energy|Industry|Liquids",
    "Final Energy|Transportation|Gases",
    "Final Energy|Residential and Commercial|Gases",
    "Final Energy|Industry|Gases",
]

main_emi_sectors = {
    "Emissions|CO2|Energy|Supply|Electricity": {
        "Emissions|CO2|Energy|Supply|Electricity|Coal": 1,
        "Emissions|CO2|Energy|Supply|Electricity|Gas": 1,
        "Emissions|CO2|Energy|Supply|Electricity|Oil": 1,
        # "Carbon Sequestration|CCS|Biomass|Energy|Supply|Electricity": -1,
    },
    "Emissions|CO2|Energy|Demand|Residential and Commercial": [
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Gases",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Liquids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Solids",
    ],
    "Emissions|CO2|Energy|Demand|Industry": [
        "Emissions|CO2|Energy|Demand|Industry|Gases",
        "Emissions|CO2|Energy|Demand|Industry|Liquids",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Biomass",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Coal",
    ],
    "Emissions|CO2|Energy|Demand|Transportation": [
        "Emissions|CO2|Energy|Demand|Transportation|Air Travel",
        "Emissions|CO2|Energy|Demand|Transportation|Buses/Trucks",
        "Emissions|CO2|Energy|Demand|Transportation|Cars",
        "Emissions|CO2|Energy|Demand|Transportation|Off-road",
        "Emissions|CO2|Energy|Demand|Transportation|Ships",
    ],
}


var_list_step5b = [
    (
        "Emissions|CO2|Energy",
        # "Carbon Sequestration|CCS|Biomass",
        "Carbon Sequestration|CCS|Biomass|Energy|Supply",  ## UPDATED 2022_08_04
        None,
        "Emissions|CO2|Energy",
    ),
    (
        "Emissions|CO2|Energy|Supply|Electricity",
        "Carbon Sequestration|CCS|Biomass|Energy|Supply|Electricity",
        None,
        "Emissions|CO2|Energy|Supply|Electricity",
    ),
    (
        "Emissions|CO2|Energy|Demand|Industry",
        "Carbon Sequestration|CCS|Biomass|Energy|Demand|Industry",
        None,
        "Emissions|CO2|Energy|Demand|Industry",
    ),
    (
        "Emissions|CO2|Energy|Demand|Transportation",
        None,
        None,
        "Emissions|CO2|Energy|Demand|Transportation",
    ),
    (
        "Emissions|CO2|Energy|Demand|Residential and Commercial",
        None,
        None,
        "Emissions|CO2|Energy|Demand|Residential and Commercial",
    ),
    (
        "Emissions|CO2|Energy|Supply|Heat",
        None,
        None,
        "Emissions|CO2|Energy|Supply|Heat",
    ),
    (
        "Emissions|CO2|Industrial Processes",
        None,
        None,
        "Emissions|CO2|Industrial Processes",
    ),
]

energy_dict_step5b = {  # tup[0]: (Energy variable, f_list (fuels or energy carrier) )
    "Emissions|CO2|Energy": ("Primary Energy", ["Coal", "Gas", "Oil"]),
    "Emissions|CO2|Energy|Supply|Electricity": (
        "Secondary Energy|Electricity",
        ["Coal", "Gas", "Oil"],
    ),
    "Emissions|CO2|Energy|Demand|Industry": (
        "Final Energy|Industry",
        ["Liquids", "Gases", "Solids|Biomass", "Solids|Coal"],
    ),
    "Emissions|CO2|Energy|Demand|Transportation": (
        "Final Energy|Transportation",
        [
            "Liquids",
            "Solids",
            "Gases",
        ],
    ),
    "Emissions|CO2|Energy|Demand|Residential and Commercial": (
        "Final Energy|Residential and Commercial",
        [
            "Liquids",
            "Solids",
            "Gases",
        ],
    ),
    "Emissions|CO2|Energy|Supply|Heat": ("Final Energy|Heat", [""]),
    "Emissions|CO2|Industrial Processes": ("Final Energy|Industry", [""]),
}

revenue_var_list = {
    "Revenue|Government|Tax|Carbon|Demand|Buildings": [
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Liquids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Solids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Gases",
    ],
    "Revenue|Government|Tax|Carbon|Demand|Industry": [
        "Emissions|CO2|Energy|Demand|Industry|Gases",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Biomass",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Coal",
    ],
    "Revenue|Government|Tax|Carbon|Supply": [
        "Emissions|CO2|Energy|Supply|Electricity|Coal",
        "Emissions|CO2|Energy|Supply|Electricity|Gas",
        "Emissions|CO2|Energy|Supply|Electricity|Oil",
        "Emissions|CO2|Energy|Supply|Heat",
    ],
    "Revenue|Government|Tax|Carbon|Demand|Transport": [
        "Emissions|CO2|Energy|Demand|Transportation|Liquids",
        "Emissions|CO2|Energy|Demand|Transportation|Gases",
    ],
}

iea_var_dict = {
    "Emissions|CO2|Energy": {"flow": ["CO2 fuel combustion"], "product": ["Total"]},
    "Emissions|CO2|Energy excl BECCS": {
        "flow": ["CO2 fuel combustion"],
        "product": ["Total"],
    },
    "Emissions|CO2|Energy excl BECCS v2": {
        "flow": ["CO2 fuel combustion"],
        "product": ["Total"],
    },
    "Emissions|CO2|Energy excl BECCS v3": {
        "flow": ["CO2 fuel combustion"],
        "product": ["Total"],
    },
    "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS": {
        "flow": [
            "Main activity electricity and heat production",
            "Unallocated autoproducers",
        ],
        "product": ["Total"],
    },
    "Emissions|CO2|Energy|Demand|Industry": {
        "flow": ["Manufacturing industries and construction"],
        "product": ["Total"],
    },
    "Emissions|CO2|Energy|Demand|Transportation": {
        "flow": ["Transport"],
        "product": ["Total"],
    },
    "Emissions|CO2|Energy|Demand|Transportation v2": {
        "flow": ["Transport"],
        "product": ["Total"],
    },
    "Emissions|CO2|Energy|Demand|Residential and Commercial": {
        "flow": ["Residential", "Commercial and public services"],
        "product": ["Total"],
    },
}

emissions_harmo_dict_step5b = {  # NOTE here the order matters for top-down harmonization
    "Emissions|CO2|Energy excl BECCS": [
        "Emissions|CO2|Energy|Coal",
        "Emissions|CO2|Energy|Gas",
        "Emissions|CO2|Energy|Oil",
    ],
    "Emissions|CO2|Energy excl BECCS v2": [
        # NOTE # We do not use "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS" to avoid mismatch with REGIONAL iam results. because:
        # this variable has been correctly calculated until now (as total electricty excluding BECCS seqeustration)
        # However, this variable is not present in IAM results. Therefore when we do the top down harmonization this variable will change to reach sectorial consistency in each country (but will be not harmonized to match regional iam results =>  mandatory_hamronization=False - which means no regional harmonization if the variable if not presen in IAM results).
        "Emissions|CO2|Energy|Demand|Industry",
        "Emissions|CO2|Energy|Demand|Residential and Commercial",
        # "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS v2",  ## we don't put coal, gas and oil because this variable will be not harmonized with regional Electricity emissions exc.l beccs data (we don't have this variable)
        "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS",
        #  "Emissions|CO2|Energy|Supply|Heat", # already in electricity
        ## transport definition 1 below
        "Emissions|CO2|Energy|Demand|Transportation",
    ],
    "Emissions|CO2|Energy|Demand|Transportation": [
        "Emissions|CO2|Energy|Demand|Transportation|Liquids",
        "Emissions|CO2|Energy|Demand|Transportation|Gases",
    ],
    "Emissions|CO2|Energy|Demand|Transportation v2": [
        "Emissions|CO2|Energy|Demand|Transportation|Air Travel",
        "Emissions|CO2|Energy|Demand|Transportation|Buses/Trucks",
        "Emissions|CO2|Energy|Demand|Transportation|Cars",
        "Emissions|CO2|Energy|Demand|Transportation|Off-road",
        "Emissions|CO2|Energy|Demand|Transportation|Ships",
    ],
    "Emissions|CO2|Energy|Demand|Industry": [
        "Emissions|CO2|Energy|Demand|Industry|Liquids",
        "Emissions|CO2|Energy|Demand|Industry|Gases",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Biomass",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Coal",
    ],
    "Emissions|CO2|Energy|Demand|Residential and Commercial": [
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Liquids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Solids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Gases",
    ],
    "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS": [
        "Emissions|CO2|Energy|Supply|Electricity|Coal",
        "Emissions|CO2|Energy|Supply|Electricity|Gas",
        "Emissions|CO2|Energy|Supply|Electricity|Oil",
        "Emissions|CO2|Energy|Supply|Heat",  ## this wil be compared to IEA emissions inlcuding heat
    ],
}

new_var_dict_step5b = {  # NOTE here the order matters for top-down harmonization
    "Emissions|CO2|Energy excl BECCS": [
        "Emissions|CO2|Energy|Coal",
        "Emissions|CO2|Energy|Gas",
        "Emissions|CO2|Energy|Oil",
    ],
    "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS": [
        "Emissions|CO2|Energy|Supply|Electricity|Coal",
        "Emissions|CO2|Energy|Supply|Electricity|Gas",
        "Emissions|CO2|Energy|Supply|Electricity|Oil",
        "Emissions|CO2|Energy|Supply|Heat",  ## this wil be compared to IEA emissions inlcuding heat
    ],
    "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS v2": [
        "Emissions|CO2|Energy|Supply|Electricity|Coal",
        "Emissions|CO2|Energy|Supply|Electricity|Gas",
        "Emissions|CO2|Energy|Supply|Electricity|Oil",
        "Emissions|CO2|Energy|Supply|Heat",  ## this wil be compared to IEA emissions inlcuding heat
    ],
    "Emissions|CO2|Energy|Demand|Transportation": [
        "Emissions|CO2|Energy|Demand|Transportation|Liquids",
        "Emissions|CO2|Energy|Demand|Transportation|Gases",
    ],
    "Emissions|CO2|Energy|Demand|Transportation v2": [
        "Emissions|CO2|Energy|Demand|Transportation|Air Travel",
        "Emissions|CO2|Energy|Demand|Transportation|Buses/Trucks",
        "Emissions|CO2|Energy|Demand|Transportation|Cars",
        "Emissions|CO2|Energy|Demand|Transportation|Off-road",
        "Emissions|CO2|Energy|Demand|Transportation|Ships",
    ],
    "Emissions|CO2|Energy|Demand|Industry": [
        "Emissions|CO2|Energy|Demand|Industry|Liquids",
        "Emissions|CO2|Energy|Demand|Industry|Gases",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Biomass",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Coal",
    ],
    "Emissions|CO2|Energy|Demand|Residential and Commercial": [
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Liquids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Solids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Gases",
    ],
    "Emissions|CO2|Energy excl BECCS v2": [
        # NOTE # We do not use "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS" to avoid mismatch with REGIONAL iam results. because:
        # this variable has been correctly calculated until now (as total electricty excluding BECCS seqeustration)
        # However, this variable is not present in IAM results. Therefore when we do the top down harmonization this variable will change to reach sectorial consistency in each country (but will be not harmonized to match regional iam results =>  mandatory_hamronization=False - which means no regional harmonization if the variable if not presen in IAM results).
        "Emissions|CO2|Energy|Demand|Industry",
        "Emissions|CO2|Energy|Demand|Residential and Commercial",
        "Emissions|CO2|Energy|Supply|Electricity and heat excl BECCS",  ## we don't put coal, gas and oil because this variable will be not harmonized with regional Electricity emissions exc.l beccs data (we don't have this variable)
        #  "Emissions|CO2|Energy|Supply|Heat", # already in electricity
        ## transport definition 1 below
        "Emissions|CO2|Energy|Demand|Transportation",
    ],
}

col_iam_dict = {
    "ISO": "REGION",
    "TARGET": "SCENARIO",
    "SECTOR": "VARIABLE",
}
iamc_idx_names = ["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"]

step1_col_dict = {
    "VARIABLE": "SECTOR",
    "REGION": "ISO",
    "SCENARIO": "TARGET",
}

# NOTE here the order matters
step5b_sect_consistency_downs_data = {
    "Emissions|CO2|Energy EXCL BECCS": [
        "Emissions|CO2|Energy|Coal",
        "Emissions|CO2|Energy|Gas",
        "Emissions|CO2|Energy|Oil",
    ],
    "Emissions|CO2|Energy EXCL BECCS v2": [
        "Emissions|CO2|Energy|Supply|Electricity EXCL BECCS",
        "Emissions|CO2|Energy|Demand|Industry",
        "Emissions|CO2|Energy|Demand|Residential and Commercial",
        "Emissions|CO2|Energy|Supply|Heat",
        "Emissions|CO2|Energy|Demand|Transportation",
    ],
    "Emissions|CO2|Energy|Supply|Electricity EXCL BECCS": {
        "Emissions|CO2|Energy|Supply|Electricity|Coal",
        "Emissions|CO2|Energy|Supply|Electricity|Gas",
        "Emissions|CO2|Energy|Supply|Electricity|Oil",
    },
    "Emissions|CO2|Energy|Demand|Industry": [
        "Emissions|CO2|Energy|Demand|Industry|Liquids",
        "Emissions|CO2|Energy|Demand|Industry|Gases",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Biomass",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Coal",
    ],
    "Emissions|CO2|Energy|Demand|Transportation": [
        "Emissions|CO2|Energy|Demand|Transportation|Liquids",
        "Emissions|CO2|Energy|Demand|Transportation|Gases",
    ],
    "Emissions|CO2|Energy|Demand|Transportation v2": [
        "Emissions|CO2|Energy|Demand|Transportation|Air Travel",
        "Emissions|CO2|Energy|Demand|Transportation|Buses/Trucks",
        "Emissions|CO2|Energy|Demand|Transportation|Cars",
        "Emissions|CO2|Energy|Demand|Transportation|Off-road",
        "Emissions|CO2|Energy|Demand|Transportation|Ships",
    ],
    # NOTE: We need to add Heat here otherwise heat will be not re-hamronized after top-down harmonization
    # (as we iterate across `step5b_sect_consistency_downs_data.items()` )
    "Emissions|CO2|Energy|Supply|Heat": ["Emissions|CO2|Energy|Supply|Heat"],
    "Emissions|CO2|Energy|Demand|Residential and Commercial": [
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Liquids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Solids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Gases",
    ],
}

step5b_sect_consistency_reg_iam_data = {
    # "aa": "bb"
    "Emissions|CO2|Energy|Supply|Electricity EXCL BECCS": {
        "Emissions|CO2|Energy|Supply|Electricity": 1,
        "Carbon Sequestration|CCS|Biomass|Energy|Supply|Electricity": 1,  # plus sign
    },
    ## total sector
    "Emissions|CO2|Energy EXCL BECCS": {
        "Emissions|CO2|Energy": 1,
        "Carbon Sequestration|CCS|Biomass|Energy|Supply": 1,  # plus sign
    },
    "Emissions|CO2|Energy EXCL BECCS v2": {
        "Emissions|CO2|Energy": 1,
        "Carbon Sequestration|CCS|Biomass|Energy|Supply": 1,  # plus sign
    },
    "Emissions|CO2|Energy|Demand|Transportation v2": [
        "Emissions|CO2|Energy|Demand|Transportation"
    ],
    "Emissions|CO2|Energy|Supply|Heat": ["Emissions|CO2|Energy|Supply|Heat"],
    # just for checking sectorial consistency
    "Emissions|CO2|Energy EXCL BECCS v2": [
        "Emissions|CO2|Energy|Supply|Electricity EXCL BECCS",
        "Emissions|CO2|Energy|Demand|Industry",
        "Emissions|CO2|Energy|Demand|Residential and Commercial",
        "Emissions|CO2|Energy|Supply|Heat",
        "Emissions|CO2|Energy|Demand|Transportation",
    ],
}

emi_factor = {
    "Coal": 95.7,
    "Gas": 56.1,
    "Oil": 67.5,
}
eff_factor = {"Coal": 0.35, "Gas": 0.55, "Oil": 0.3}

emi_factor.update(
    {
        "Solids": emi_factor["Coal"],
        "Liquids": emi_factor["Oil"],
        "Gases": emi_factor["Gas"],
        "Solids|Coal": emi_factor["Coal"],
        "Solids|Biomass": 0,
        "Beccs": -emi_factor[
            "Coal"
        ],  ## assumed to be same as coal (with a negative value)
    }
)

eff_factor.update(
    {
        "Solids": eff_factor["Coal"],
        "Liquids": eff_factor["Oil"],
        "Gases": eff_factor["Gas"],
        "Solids|Coal": eff_factor["Coal"],
        "Solids|Biomass": 0.35,
        "Beccs": 0.55,
    }
)

# NOTE: top 20 legend for MACC graph
legend_dict = {
    " ": "white",
    0: "white",
    "Emissions|non-CO2|Energy": "#00C1A3",
    "Emissions|CO2|Energy|Supply|Heat": "#00B0F6",
    "Emissions|CO2|Energy|Supply|Electricity": "#35A2FF",
    "Emissions|CO2|Energy|Demand|Residential and Commercial": "#C77CFF",
    "Emissions|CO2|Energy|Demand|Industry": "#FF62BC",
    "Emissions|CO2|Industrial Processes": "#F8766D",
    "Emissions|non-CO2|Industry": "#EA8331",
    "Emissions|CO2|Energy|Demand|Transportation": "#A3A500",
    "Transport": "#7CAE00",
    "Carbon Sequestration|CCS|Biomass|Energy|Supply|Electricity": "#39B600",
    "Carbon Sequestration|CCS|Biomass|Energy|Supply|Liquids": "#00BB4E",
    "Carbon Sequestration|CCS|Biomass|Energy|Supply|Hydrogen": "#00BF7D",
    "Agriculture": "green",
    "Emissions|non-CO2|Building (Waste)": "grey",
}

var_dict_top20 = {
    "Building": "Emissions|non-CO2|Building (Waste)",
    "Industry": "Emissions|non-CO2|Industry",
    "Energy": "Emissions|non-CO2|Energy",
}

time_dict_top20 = {
    "short": {"from": 2025, "to": 2030},
    "long": {"from": 2040, "to": 2050},
}


fast_list = [
    "CD_Links_SSP1_v2INDCi_1000-con-prim-dir-ncr",
    "CD_Links_SSP1_v2NPi2020_1000-con-prim-dir-ncr",
    "CD_Links_SSP1_v2NPi2020_400-con-prim-dir-ncr",
    "CD_Links_SSP2_v2INDCi_1000-con-prim-dir-ncr",
    "CD_Links_SSP2_v2NPi2020_1000-con-prim-dir-ncr",
    "CD_Links_SSP2_v2NPi2020_400-con-prim-dir-ncr",
    "CD_Links_SSP3_v2INDCi_1000-con-prim-dir-ncr",
    "CD_Links_SSP3_v2NPi2020_1000-con-prim-dir-ncr",
    "CD_Links_SSP1_v2NPi2020_400-con-prim-dir-ncr",
]

slow_list = [
    "CD_Links_SSP1_v2INDCi_1600-con-prim-dir-ncr",
    "CD_Links_SSP1_v2NPi2020_1600-con-prim-dir-ncr",
    "CD_Links_SSP2_v2INDCi_1600-con-prim-dir-ncr",
    "CD_Links_SSP2_v2NPi2020_1600-con-prim-dir-ncr",
    "CD_Links_SSP3_v2INDCi_1600-con-prim-dir-ncr",
    "CD_Links_SSP3_v2NPi2020_1600-con-prim-dir-ncr",
]

med_list = [
    "CD_Links_SSP1_v2INDC2030i_forever-con-prim-dir-ncr",
    "CD_Links_SSP1_v2NPiREF-con-prim-dir-ncr",
    "CD_Links_SSP2_v2INDC2030i_forever-con-prim-dir-ncr",
    "CD_Links_SSP2_v2NPiREF-con-prim-dir-ncr",
    "CD_Links_SSP3_v2INDC2030i_forever-con-prim-dir-ncr",
    "CD_Links_SSP3_v2NPiREF-con-prim-dir-ncr",
]


def fun_conv_settings(sensitivity: bool):
    if sensitivity:
        # return {
        #     "Final": {"FAST": 2050, "MED": 2100, "SLOW": 2150},
        #     "Secondary": {"FAST": 2050, "MED": 2150, "SLOW": 2250},
        #     "Primary": {"FAST": 2050, "MED": 2150, "SLOW": 2250},
        # }
        myrange=range(2050,2305,5)
        return {
            "Final": {str(x):x for x in myrange},
            "Secondary": {str(x):x for x in myrange},
            "Primary":  {str(x):x for x in myrange},
        }
    return {
        "Final": {"FAST": 2100, "MED": 2150, "SLOW": 2200},
        "Secondary": {"FAST": 2200, "MED": 2250, "SLOW": 2300},
        "Primary": {"FAST": 2200, "MED": 2250, "SLOW": 2300},
    }

step1_var = [
    "GDP|PPP",
    "Population",
    "Final Energy",
    "Final Energy|Electricity",
    "Final Energy|Gases",
    "Final Energy|Heat",
    "Final Energy|Hydrogen",
    "Final Energy|Industry",
    "Final Energy|Industry|Electricity",
    "Final Energy|Industry|Gases",
    "Final Energy|Industry|Heat",
    "Final Energy|Industry|Hydrogen",
    "Final Energy|Industry|Liquids",
    "Final Energy|Industry|Solids",
    "Final Energy|Industry|Solids|Biomass",
    "Final Energy|Industry|Solids|Coal",
    "Final Energy|Liquids",
    "Final Energy|Residential and Commercial",
    "Final Energy|Residential and Commercial|Electricity",
    "Final Energy|Residential and Commercial|Gases",
    "Final Energy|Residential and Commercial|Heat",
    "Final Energy|Residential and Commercial|Liquids",
    "Final Energy|Residential and Commercial|Solids",
    "Final Energy|Residential and Commercial|Solids|Biomass",
    "Final Energy|Residential and Commercial|Solids|Coal",
    'Final Energy|Residential and Commercial|Hydrogen',
    "Final Energy|Solids",
    "Final Energy|Transportation",
    "Final Energy|Transportation|Electricity",
    "Final Energy|Transportation|Gases",
    "Final Energy|Transportation|Hydrogen",
    "Final Energy|Transportation|Liquids",
]

step2_var = [
    "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Biomass",
    "Secondary Energy|Electricity|Coal",
    "Secondary Energy|Electricity|Gas",
    "Secondary Energy|Electricity|Geothermal",
    "Secondary Energy|Electricity|Hydro",
    "Secondary Energy|Electricity|Nuclear",
    "Secondary Energy|Electricity|Oil",
    "Secondary Energy|Electricity|Solar",
    "Secondary Energy|Electricity|Wind",
    "Secondary Energy|Gases",
    "Secondary Energy|Gases|Biomass",
    "Secondary Energy|Gases|Coal",
    "Secondary Energy|Gases|Natural Gas",
    "Secondary Energy|Liquids",
    "Secondary Energy|Liquids|Biomass",
    "Secondary Energy|Liquids|Coal",
    "Secondary Energy|Liquids|Fossil",
    "Secondary Energy|Liquids|Gas",
    "Secondary Energy|Liquids|Oil",
    "Secondary Energy|Solids|Biomass",
    "Secondary Energy|Solids|Coal",
    "Secondary Energy|Solids",
    "Primary Energy",
    "Primary Energy|Biomass",
    "Primary Energy|Coal",
    "Primary Energy|Fossil",
    "Primary Energy|Gas",
    "Primary Energy|Nuclear",
    "Primary Energy|Oil",
    "Primary Energy|Non-Biomass Renewables|Geothermal",
    "Primary Energy|Non-Biomass Renewables|Hydro",
    "Primary Energy|Non-Biomass Renewables|Solar",
    "Primary Energy|Non-Biomass Renewables|Wind",
]

step3_var = [
    "Carbon Sequestration|CCS",
    "Carbon Sequestration|CCS|Biomass",
    "Carbon Sequestration|CCS|Industrial Processes",
    "Primary Energy|Coal|w/ CCS",
    "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Fossil|w/ CCS",
    "Primary Energy|Fossil|w/o CCS",
    "Primary Energy|Gas|w/ CCS",
    "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Oil|w/o CCS",
    "Carbon Sequestration|CCS",
    "Carbon Sequestration|CCS|Biomass",
    "Carbon Sequestration|CCS|Industrial Processes",
    "Emissions|CO2|Energy",
]

step5b_var = [
    "Revenue|Government|Tax|Carbon",
    "Revenue|Government|Tax|Carbon|Supply",
    "Revenue|Government|Tax|Carbon|Demand|Buildings",
    "Revenue|Government|Tax|Carbon|Demand|Industry",
    "Revenue|Government|Tax|Carbon|Demand|Transport",
]

price_var = [
    "Price|Final Energy|Residential and Commercial|Residential|Electricity|Index",
    "Price|Final Energy|Residential and Commercial|Residential|Gases|Natural Gas|Index",
    "Price|Final Energy|Residential and Commercial|Residential|Liquids|Oil|Index",
    "Price|Primary Energy|Biomass|Index",
    "Price|Primary Energy|Coal|Index",
    "Price|Primary Energy|Gas|Index",
    "Price|Primary Energy|Oil|Index",
    "Price|Secondary Energy|Electricity|Index",
    "Price|Secondary Energy|Gases|Natural Gas|Index",
    "Price|Secondary Energy|Liquids|Biomass|Index",
    "Price|Secondary Energy|Liquids|Oil|Index",
    "Price|Secondary Energy|Solids|Coal|Index",
]

seq_var = [
    "Carbon Sequestration|CCS",
    "Carbon Sequestration|CCS|Biomass",
    "Carbon Sequestration|CCS|Biomass|Energy|Demand|Industry",
    "Carbon Sequestration|CCS|Biomass|Energy|Supply",
    "Carbon Sequestration|CCS|Biomass|Energy|Supply|Electricity",
    "Carbon Sequestration|CCS|Biomass|Energy|Supply|Hydrogen",
    "Carbon Sequestration|CCS|Biomass|Energy|Supply|Liquids",
    "Carbon Sequestration|CCS|Fossil",
    "Carbon Sequestration|CCS|Fossil|Energy|Demand|Industry",
    "Carbon Sequestration|CCS|Fossil|Energy|Supply",
    "Carbon Sequestration|CCS|Fossil|Energy|Supply|Electricity",
    "Carbon Sequestration|CCS|Fossil|Energy|Supply|Hydrogen",
    "Carbon Sequestration|CCS|Fossil|Energy|Supply|Liquids",
    "Carbon Sequestration|CCS|Industrial Processes",
]


ec_list_step5 = [
    "Liquids",
    "Solids",
    "Electricity",
    "Gases",
    "Hydrogen",
    "Heat",
]

sectors_list = ["Industry", "Transportation", "Residential and Commercial"]

fuel_list_step5 = [
    "Coal",
    "Oil",
    "Gas",
    "Natural Gas",
    "Biomass",
    "Nuclear",
    "Non-Biomass Renewables|Hydro",
    "Non-Biomass Renewables|Solar",
    "Non-Biomass Renewables|Wind",
    "Non-Biomass Renewables|Geothermal",
]


check_consistency_dict = {
    "step1_demand": {
        "Final Energy": [
            "Final Energy|Electricity",
            "Final Energy|Liquids",
            "Final Energy|Solids",
            "Final Energy|Gases",
            "Final Energy|Heat",
            "Final Energy|Hydrogen",
        ]
    },
    "step1_supply": {
        "Final Energy": [
            "Final Energy|Residential and Commercial",
            "Final Energy|Transportation",
            "Final Energy|Industry",
        ]
    },
    "step2": {
        "Primary Energy": [
            "Primary Energy|Biomass",
            "Primary Energy|Coal|w/ CCS",
            "Primary Energy|Coal|w/o CCS",
            "Primary Energy|Gas|w/ CCS",
            "Primary Energy|Gas|w/o CCS",
            "Primary Energy|Geothermal",
            "Primary Energy|Hydro",
            "Primary Energy|Nuclear",
            "Primary Energy|Oil",
            "Primary Energy|Solar",
            "Primary Energy|Wind",
        ]
    },
}


step2_primary_energy = {
    "Primary Energy": [
        "Primary Energy|Biomass",
        "Primary Energy|Coal",
        "Primary Energy|Gas",
        "Primary Energy|Geothermal",
        "Primary Energy|Hydro",
        "Primary Energy|Nuclear",
        "Primary Energy|Oil",
        "Primary Energy|Solar",
        "Primary Energy|Wind",
    ]
}


step2_electricity = {
    "Secondary Energy|Electricity": [
        "Secondary Energy|Electricity|Biomass",
        "Secondary Energy|Electricity|Coal",
        "Secondary Energy|Electricity|Gas",
        "Secondary Energy|Electricity|Geothermal",
        "Secondary Energy|Electricity|Hydro",
        "Secondary Energy|Electricity|Nuclear",
        "Secondary Energy|Electricity|Oil",
        "Secondary Energy|Electricity|Solar",
        "Secondary Energy|Electricity|Wind",
    ]
}


prova = {
    "Share of energy imports on Primary Energy": {
        "xvar": "Share of hydrogen on Final Energy",
        "by": "none",
        "row": 0,
        "col": 0,
        "same_level": False,
        "kind": "line",
        "ylim": (-0.2, 0.8),
    },
}

# Graphs/visualizations
dashboard_bar_plots = {
    "Primary Energy": {
        "by": "fuel",
        "row": 0,
        "col": 0,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 80),
    },
    # "Primary Energy": {
    #     "kind": "scatter",
    #     "xvar": "Final Energy",
    #     "svar": "Emissions|CO2|Energy", # scatter point size
    #     "row": 0,
    #     "col": 0,
    #     "ylim": None,
    # },
    "Final Energy": {
        "by": "sector",
        "row": 0,
        "col": 1,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 60),
    },
    "Secondary Energy|Electricity": {
        "by": "fuel",
        "row": 0,
        "col": 2,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 50),
    },
    # "Carbon Sequestration|CCS|Biomass": {
    #     "by": "none", # Just get "Carbon Sequestration|CCS|Biomass"
    #     "row": 1,
    #     "col": 0,
    #     "same_level": False,
    #     "kind": "area",
    #     "ylim": (0, 1e3),
    # },
    "Carbon Removal": {
        "by": "cdr",  # get all variables starting with "Carbon Removal|" #(CDR)
        "row": 1,
        "col": 0,
        "same_level": False,
        "kind": "area",
        "ylim": (0, 1.5e3),
    },
    # "Share of energy imports on Primary Energy": {
    #     "by": "none",
    #     "row": 1,
    #     "col": 1,
    #     "same_level": False,
    #     "kind": "line",
    #     "ylim": (-0.2, 0.8),
    # },
    "Share of energy imports on Primary Energy": {
        "xvar": "Share of hydrogen on Final Energy",
        "by": "none",
        "row": 1,
        "col": 1,
        "same_level": False,
        "kind": "line",
        "ylim": (-0.2, 0.8),
    },
    "Emissions": {
        "by": "gas",
        "row": 1,
        "col": 2,
        "same_level": True,  # False,
        "kind": "bar",
        "ylim": (-2e3, 6e3),
    },
    "Final Energy|Industry": {
        "by": "ec",
        "row": 2,
        "col": 0,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 25),
    },
    "Final Energy|Transportation": {
        "by": "ec",
        "row": 2,
        "col": 1,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 25),
    },
    "Final Energy|Residential and Commercial": {
        "by": "ec",
        "row": 2,
        "col": 2,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 25),
    },
}


dashboard_scatter_plots = {
    "Primary Energy": {
        "kind": "scatter",
        "xvar": "Final Energy",
        "svar": "Emissions|CO2|Energy",  # scatter point size
        "row": 0,
        "col": 0,
        "ylim": None,
    },
}

list_emission_by_sectors_step5b = [
    "Emissions|CO2|Energy|Coal",
    "Emissions|CO2|Energy|Gas",
    "Emissions|CO2|Energy|Oil",
    "Emissions|CO2|Energy|Supply|Electricity|Coal",
    "Emissions|CO2|Energy|Supply|Electricity|Gas",
    "Emissions|CO2|Energy|Supply|Electricity|Oil",
    "Emissions|CO2|Energy|Demand|Industry|Liquids",
    "Emissions|CO2|Energy|Demand|Industry|Gases",
    "Emissions|CO2|Energy|Demand|Industry|Solids|Biomass",
    "Emissions|CO2|Energy|Demand|Industry|Solids|Coal",
    "Emissions|CO2|Energy|Demand|Transportation|Liquids",
    "Emissions|CO2|Energy|Demand|Transportation|Gases",
    "Emissions|CO2|Energy|Demand|Transportation|Air Travel",
    "Emissions|CO2|Energy|Demand|Transportation|Buses/Trucks",
    "Emissions|CO2|Energy|Demand|Transportation|Cars",
    "Emissions|CO2|Energy|Demand|Transportation|Off-road",
    "Emissions|CO2|Energy|Demand|Transportation|Ships",
    "Emissions|CO2|Energy|Demand|Residential and Commercial|Liquids",
    "Emissions|CO2|Energy|Demand|Residential and Commercial|Solids",
    "Emissions|CO2|Energy|Demand|Residential and Commercial|Gases",
    "Emissions|CO2|Energy|Supply|Heat",
    "Emissions|CO2|Industrial Processes",
]

ngfs_2023_nomenclature = {
    "Revenue|Government|Tax|Carbon|Demand|Residential and Commercial": "Revenue|Government|Tax|Carbon|Demand|Buildings",
    "Revenue|Government|Tax|Carbon|Demand|Transportation": "Revenue|Government|Tax|Carbon|Demand|Transport",
}

step5e_harmo = [
    "Emissions|CO2|LULUCF Direct+Indirect",  # harminize afolu emissions
    "Emissions|CO2|Industrial Processes",
    "Emissions|CO2|Energy",
    'Emissions|Total Non-CO2',
    # "Emissions|CO2",  # NOTE Calculated as the sum of subsectors
    "Primary Energy",
    "Primary Energy|Biomass",
    # "Primary Energy|Biomass|w/o CCS", # not available in dataframe
    "Primary Energy|Coal",
    "Primary Energy|Coal|w/o CCS",
    "Primary Energy|Fossil",
    "Primary Energy|Fossil|w/o CCS",
    "Primary Energy|Gas",
    "Primary Energy|Gas|w/o CCS",
    "Primary Energy|Geothermal",
    "Primary Energy|Hydro",
    "Primary Energy|Nuclear",
    # "Primary Energy|Ocean",
    "Primary Energy|Oil",
    "Primary Energy|Oil|w/o CCS",
    # "Primary Energy|Other",
    "Primary Energy|Solar",
    "Primary Energy|Wind",
    "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Biomass",
    "Secondary Energy|Electricity|Coal",
    "Secondary Energy|Electricity|Gas",
    "Secondary Energy|Electricity|Geothermal",
    "Secondary Energy|Electricity|Hydro",
    "Secondary Energy|Electricity|Nuclear",
    "Secondary Energy|Electricity|Oil",
    "Secondary Energy|Electricity|Solar",
    "Secondary Energy|Electricity|Wind",
]

primap_dict = {
    "Emissions|Kyoto Gases excl. LULUCF": {
        "entity": "KYOTOGHG (AR4GWP100)",
        "category (IPCC2006_PRIMAP)": "M.0.EL",
    },  #  National Total excluding LULUCF
    "Emissions|CO2|LULUCF Direct+Indirect": {
        "entity": "KYOTOGHG (AR4GWP100)",
        "category (IPCC2006_PRIMAP)": "M.LULUCF",  # LULUCF
    },
    "Emissions|CO2": {
        "entity": "CO2",
        "category (IPCC2006_PRIMAP)": "M.0.EL",  #  National Total excluding LULUCF
    },
    "Emissions|CO2|Energy": {
        "entity": "CO2",
        "category (IPCC2006_PRIMAP)": "1",  # Energy
    },
    "Emissions|CO2|Industrial Processes": {
        "entity": "CO2",
        "category (IPCC2006_PRIMAP)": "2",  # Industrial Processes and Product Use (IPPU)
    },
}

sel_plot_vars_step5e = [
    "GHG incl. International transport",
    "GHG incl. International transport (intra-eu only)",
    "Emissions|Kyoto Gases (incl. indirect AFOLU)",
    "Carbon Sequestration|CCS",
    "Carbon Sequestration|CCS|Biomass",
    "Carbon Sequestration|CCS|Fossil",
    "Carbon Sequestration|CCS|Industrial Processes",
    "Emissions|CO2",
    # "Emissions|CO2|AFOLU",
    # "Emissions|CO2|Energy",
    # "Emissions|CO2|Industrial Processes",
    # "Emissions|CH4",
    # "Emissions|HFC",
    # "Emissions|N2O",
    "Emissions|Total Non-CO2",
]

countries_w_2030_emi_policies = [
    "AUT",
    "BEL",
    "BGR",
    "CYP",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GRC",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LTU",
    "LUX",
    "LVA",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "SWE",
    "JPN",
    "KOR",
    "CAN",
    "ZAF",
    "GBR",
]
countries_w_2050_emi_policies = [
    "AUT",
    "BEL",
    "BGR",
    "CYP",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GRC",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LTU",
    "LUX",
    "LVA",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "SWE",
    "JPN",
    "KOR",
    "CAN",
    "ZAF",
    "GBR",
]

timing_net_zero = {
    "CHN": 2060,
    "KAZ": 2060,
    "SWE": 2045,
    "AUT": 2040,
    "ISL": 2040,
}

# NOTE In the dictionary below we omit some variables on purpose, so that they will be dropped at the beginning of step5e (e.g.  'Emissions|Kyoto Gases (incl. indirect AFOLU)' , 'Emissions|Total Non-CO2'), ... and the statistical variables].
# Do not modify - if needed create a new  dictionary
extra_units_dict = {
    "EJ/yr": ["Secondary Energy|Electricity|Trade"],
    "Mt CO2/yr": [
        "Emissions|CO2|LULUCF Direct+Indirect",
        "Emissions|CO2|LULUCF Indirect",
    ],
}

colors = {
            "Solar": "#FAA307",
            "Wind": "#00AFD6",
            "Biomass": "#007F5F",
            "Hydro": "#0077B6",
            "Geothermal": "#FF0000",
            "Nuclear": "#C00000",
            "Gas": "#B00108",
            "Gases": "#B00108",
            "Oil": "#6A040F",
            "Liquids": "#6A040F",
            "Coal": "#000118",
            "Solids": "#000118",
            "Hydrogen": "#457B9D",
            "Electricity": "#A8DADC",
            "Heat": "#FF0000",
            "Transportation": "#6A040F",
            "Transport": "#6A040F",
            "Residential and Commercial": "#8172B3",
            "Industry": "#C44E52",
        }

NGFS_dashboard_bar_plots = {
    "Primary Energy": {
        "by": "fuel",
        "row": 0,
        "col": 0,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 6),
    },
    "Final Energy": {
        "by": "sector",
        "row": 0,
        "col": 1,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 4),
    },
    "Secondary Energy|Electricity": {
        "by": "fuel",
        "row": 0,
        "col": 2,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 2),
    },
    "Final Energy|Industry": {
        "by": "ec",
        "row": 2,
        "col": 0,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 2),
    },
    "Final Energy|Transportation": {
        "by": "ec",
        "row": 2,
        "col": 1,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 2),
    },
    "Final Energy|Residential and Commercial": {
        "by": "ec",
        "row": 2,
        "col": 2,
        "same_level": True,
        "kind": "area",
        "ylim": (0, 2),
    },
}

step5f_dict1 = {
    "Emissions|CO2|Energy EXCL BECCS": [
        "Emissions|CO2|Energy|Coal",
        "Emissions|CO2|Energy|Gas",
        "Emissions|CO2|Energy|Oil",
    ],
   
    "Emissions|CO2|Energy|Supply|Electricity EXCL BECCS": {
        "Emissions|CO2|Energy|Supply|Electricity|Coal",
        "Emissions|CO2|Energy|Supply|Electricity|Gas",
        "Emissions|CO2|Energy|Supply|Electricity|Oil",
    },
    "Emissions|CO2|Energy|Demand|Industry": [
        "Emissions|CO2|Energy|Demand|Industry|Liquids",
        "Emissions|CO2|Energy|Demand|Industry|Gases",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Biomass",
        "Emissions|CO2|Energy|Demand|Industry|Solids|Coal",
    ],
    "Emissions|CO2|Energy|Demand|Transportation": [
        "Emissions|CO2|Energy|Demand|Transportation|Liquids",
        "Emissions|CO2|Energy|Demand|Transportation|Gases",
    ],
    "Emissions|CO2|Energy|Demand|Residential and Commercial": [
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Liquids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Solids",
        "Emissions|CO2|Energy|Demand|Residential and Commercial|Gases",
    ],
}


step5f_dict2 = {

    "Emissions|CO2|Energy EXCL BECCS": [
        # "Emissions|CO2|Energy|Supply|Electricity EXCL BECCS",
        "Emissions|CO2|Energy|Supply|Electricity|Gas",
        "Emissions|CO2|Energy|Supply|Electricity|Coal",
        "Emissions|CO2|Energy|Supply|Electricity|Oil",

        "Emissions|CO2|Energy|Demand|Industry",
        "Emissions|CO2|Energy|Demand|Residential and Commercial",
        "Emissions|CO2|Energy|Supply|Heat",
        "Emissions|CO2|Energy|Demand|Transportation",
    ],
    "Emissions|CO2|Energy|Demand|Transportation": [
        "Emissions|CO2|Energy|Demand|Transportation|Air Travel",
        "Emissions|CO2|Energy|Demand|Transportation|Buses/Trucks",
        "Emissions|CO2|Energy|Demand|Transportation|Cars",
        "Emissions|CO2|Energy|Demand|Transportation|Off-road",
        "Emissions|CO2|Energy|Demand|Transportation|Ships",
    ],
}

step5f_temp_vars_dict={"Emissions|CO2|Energy EXCL BECCS": {"Emissions|CO2|Energy":1,
                                                        "Carbon Sequestration|CCS|Biomass":-1},

                   "Emissions|CO2|Energy|Supply|Electricity EXCL BECCS":['Emissions|CO2|Energy|Supply|Electricity|Coal',
                                                                         'Emissions|CO2|Energy|Supply|Electricity|Gas',
                                                                         'Emissions|CO2|Energy|Supply|Electricity|Oil',
                                                                         ]
    }

# Below is created using fun_load_iea() based on "Historical_data.csv"
iea_countries=[
    'ALB', 'DZA', 'AGO', 'ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BHR', 'BGD', 'BLR', 'BEL', 'BEN', 'BOL', 'BIH', 'BWA', 
    'BRA', 'BRN', 'BGR', 'KHM', 'CMR', 'CAN', 'CHL', 'CHN', 'COL', 'COG', 'CRI', 'CIV', 'HRV', 'CUB', 'ANT', 'CYP', 
    'CZE', 'PRK', 'COD', 'DNK', 'DOM', 'ECU', 'EGY', 'SLV', 'ERI', 'EST', 'ETH', 'FIN', 'FRA', 'GAB', 'GEO', 'DEU', 
    'GHA', 'GIB', 'GRC', 'GTM', 'HTI', 'HND', 'HKG', 'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'ISR', 'ITA', 
    'JAM', 'JPN', 'JOR', 'KAZ', 'KEN', 'KOR', 'KWT', 'KGZ', 'LVA', 'LBN', 'LBY', 'LTU', 'LUX', 'MYS', 'MLT', 'TWN',
    'MUS', 'MEX', 'MDA', 'MNG', 'MNE', 'MAR', 'MOZ', 'MMR', 'NAM', 'NPL', 'NLD', 'NZL', 'NIC', 'NER', 'NGA', 'MKD', 
    'NOR', 'OMN', 'PAK', 'PAN', 'PRY', 'PER', 'PHL', 'POL', 'PRT', 'QAT', 'ROU', 'RUS', 'SAU', 'SEN', 'SRB', 'SGP', 
    'SVK', 'SVN', 'ZAF', 'SSD', 'ESP', 'LKA', 'SDN', 'SUR', 'SWE', 'CHE', 'SYR', 'TJK', 'TZA', 'THA', 'TGO', 'TTO', 
    'TUN', 'TUR', 'TKM', 'UKR', 'ARE', 'GBR', 'USA', 'URY', 'UZB', 'VEN', 'VNM', 'YEM', 'ZMB', 'ZWE', 'MLI', 'UGA',
      ]

check_IEA_countries=[x for x in iea_countries if x not in ["GIB","ANT"]]

# Below we get 220 countries
all_countries=['ABW', 'AFG', 'AGO', 'ALB', 'ARE', 'ARG', 'ARM', 'ASM', 'ATG',
       'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BFA', 'BGD', 'BGR',
       'BHR', 'BHS', 'BIH', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB',
       'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CHE', 'CHL', 'CHN', 'CIV',
       'CMR', 'COD', 'COG', 'COK', 'COL', 'COM', 'CPV', 'CRI', 'CUB',
       'CUW', 'CYM', 'CYP', 'CZE', 'DEU', 'DJI', 'DMA', 'DNK', 'DOM',
       'DZA', 'ECU', 'EGY', 'ERI', 'ESH', 'ESP', 'EST', 'ETH', 'FIN',
       'FJI', 'FLK', 'FRA', 'FRO', 'FSM', 'GAB', 'GBR', 'GEO', 'GHA',
       'GIB', 'GIN', 'GLP', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GRL',
       'GTM', 'GUF', 'GUM', 'GUY', 'HKG', 'HND', 'HRV', 'HTI', 'HUN',
       'IDN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM',
       'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR', 'KNA', 'KOR',
       'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LIE', 'LKA', 'LSO',
       'LTU', 'LUX', 'LVA', 'MAC', 'MAR', 'MDA', 'MDG', 'MDV', 'MEX',
       'MHL', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MOZ', 'MRT',
       'MSR', 'MTQ', 'MUS', 'MWI', 'MYS', 'NAM', 'NCL', 'NER', 'NGA',
       'NIC', 'NIU', 'NLD', 'NOR', 'NPL', 'NZL', 'OMN', 'PAK', 'PAN',
       'PER', 'PHL', 'PLW', 'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY',
       'PSE', 'PYF', 'QAT', 'REU', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN',
       'SEN', 'SGP', 'SLB', 'SLE', 'SLV', 'SOM', 'SPM', 'SRB', 'SSD',
       'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SXM', 'SYC', 'SYR',
       'TCA', 'TCD', 'TGO', 'THA', 'TJK', 'TKL', 'TKM', 'TLS', 'TON',
       'TTO', 'TUN', 'TUR', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'USA',
       'UZB', 'VCT', 'VEN', 'VGB', 'VIR', 'VNM', 'VUT', 'WLF', 'WSM',
       'YEM', 'ZAF', 'ZMB', 'ZWE']

g20_countries = ['ARG', 'AUS', 'BRA', 'CAN', 'CHN', 'FRA', 
                 'DEU', 'IND', 'IDN', 'ITA', 'JPN', 'MEX', 
                 'RUS', 'SAU', 'ZAF', 'KOR', 'TUR', 'GBR', 'USA']

# We get this dictionary from `fun_read_large_iea_file` in debug mode. Just before `df.columns=[x.upper() for x in df.columns]` run the code below:
# print(df.reset_index().set_index('FLOW')[['Flow']].to_dict())#.unique()
# print(df.reset_index().set_index('PRODUCT')[['Product']].to_dict())#.unique()
iea_flow_long_short_dict = {
    "INDPROD": "Production",
    "IMPORTS": "Imports",
    "EXPORTS": "Exports",
    "MARBUNK": "International marine bunkers",
    "AVBUNK": "International aviation bunkers",
    "STOCKCHA": "Stock changes",
    "TES": "Total energy supply",
    "TRANSFER": "Transfers",
    "STATDIFF": "Statistical differences",
    "TOTTRANF": "Transformation processes",
    "MAINELEC": "Main activity producer electricity plants",
    "AUTOELEC": "Autoproducer electricity plants",
    "MAINCHP": "Main activity producer CHP plants",
    "AUTOCHP": "Autoproducer CHP plants",
    "MAINHEAT": "Main activity producer heat plants",
    "AUTOHEAT": "Autoproducer heat plants",
    "THEAT": "Heat pumps",
    "TBOILER": "Electric boilers",
    "TELE": "Chemical heat for electricity production",
    "TBLASTFUR": "Blast furnaces",
    "TGASWKS": "Gas works",
    "TCOKEOVS": "Coke ovens",
    "TPATFUEL": "Patent fuel plants",
    "TBKB": "BKB/peat briquette plants",
    "TREFINER": "Oil refineries",
    "TPETCHEM": "Petrochemical plants",
    "TCOALLIQ": "Coal liquefaction plants",
    "TGTL": "Gas-to-liquids (GTL) plants",
    "TBLENDGAS": "For blended natural gas",
    "TCHARCOAL": "Charcoal production plants",
    "TNONSPEC": "Non-specified (transformation)",
    "TOTENGY": "Energy industry own use",
    "EMINES": "Coal mines",
    "EOILGASEX": "Oil and gas extraction",
    "EBLASTFUR": "Blast furnaces",
    "EGASWKS": "Gas works",
    "EBIOGAS": "Gasification plants for biogases",
    "ECOKEOVS": "Coke ovens",
    "EPATFUEL": "Patent fuel plants",
    "EBKB": "BKB/peat briquette plants",
    "EREFINER": "Oil refineries",
    "ECOALLIQ": "Coal liquefaction plants",
    "ELNG": "Liquefaction (LNG) / regasification plants",
    "EGTL": "Gas-to-liquids (GTL) plants",
    "EPOWERPLT": '"Own use in electricity, CHP and heat plants"',
    "EPUMPST": "Pumped storage plants",
    "ENUC": "Nuclear industry",
    "ECHARCOAL": "Charcoal production plants",
    "ENONSPEC": "Non-specified (energy)",
    "DISTLOSS": "Losses",
    "TFC": "Total final consumption",
    "TOTIND": "Industry",
    "MINING": "Mining and quarrying",
    "CONSTRUC": "Construction",
    "MANUFACT": "Manufacturing",
    "IRONSTL": "Iron and steel",
    "CHEMICAL": "Chemical and petrochemical",
    "NONFERR": "Non-ferrous metals",
    "NONMET": "Non-metallic minerals",
    "TRANSEQ": "Transport equipment",
    "MACHINE": "Machinery",
    "FOODPRO": "Food and tobacco",
    "PAPERPRO": '"Paper, pulp and printing"',
    "WOODPRO": "Wood and wood products",
    "TEXTILES": "Textile and leather",
    "INONSPEC": "Industry not elsewhere specified",
    "TOTTRANS": "Transport",
    "WORLDAV": "World aviation bunkers",
    "DOMESAIR": "Domestic aviation",
    "ROAD": "Road",
    "RAIL": "Rail",
    "PIPELINE": "Pipeline transport",
    "WORLDMAR": "World marine bunkers",
    "DOMESNAV": "Domestic navigation",
    "TRNONSPE": "Transport not elsewhere specified",
    "RESIDENT": "Residential",
    "COMMPUB": "Commercial and public services",
    "AGRICULT": "Agriculture/forestry",
    "FISHING": "Fishing",
    "ONONSPEC": "Final consumption not elsewhere specified",
    "NONENUSE": "Non-energy use",
    "NEINTREN": "Non-energy use industry/transformation/energy",
    "NEIND": "Memo: Non-energy use in industry",
    "NECONSTRUC": "Memo: Non-energy use in construction",
    "NEMINING": "Memo: Non-energy use in mining and quarrying",
    "NEIRONSTL": "Memo: Non-energy use in iron and steel",
    "NECHEM": "Memo: Non-energy use in chemical/petrochemical",
    "NENONFERR": "Memo: Non-energy use in non-ferrous metals",
    "NENONMET": "Memo: Non-energy use in non-metallic minerals",
    "NETRANSEQ": "Memo: Non-energy use in transport equipment",
    "NEMACHINE": "Memo: Non-energy use in machinery",
    "NEFOODPRO": "Memo: Non-energy use in food/beverages/tobacco",
    "NEPAPERPRO": "Memo: Non-energy use in paper/pulp and printing",
    "NEWOODPRO": "Memo: Non-energy use in wood and wood products",
    "NETEXTILES": "Memo: Non-energy use in textiles and leather",
    "NEINONSPEC": "Memo: Non-energy use in industry not elsewhere specified",
    "NETRANS": "Non-energy use in transport",
    "NEOTHER": "Non-energy use in other",
    "ELOUTPUT": "Electricity output (GWh)",
    "ELMAINE": "Electricity output (GWh)-main activity producer electricity plants",
    "ELAUTOE": "Electricity output (GWh)-autoproducer electricity plants",
    "ELMAINC": "Electricity output (GWh)-main activity producer CHP plants",
    "ELAUTOC": "Electricity output (GWh)-autoproducer CHP plants",
    "HEATOUT": "Heat output",
    "HEMAINC": "Heat output-main activity producer CHP plants",
    "HEAUTOC": "Heat output-autoproducer CHP plants",
    "HEMAINH": "Heat output-main activity producer heat plants",
    "HEAUTOH": "Heat output-autoproducer heat plants",
}

iea_product_long_short_dict = {
    "HARDCOAL": "Hard coal (if no detail)",
    "BROWN": "Brown coal (if no detail)",
    "ANTCOAL": "Anthracite",
    "COKCOAL": "Coking coal",
    "BITCOAL": "Other bituminous coal",
    "SUBCOAL": "Sub-bituminous coal",
    "LIGNITE": "Lignite",
    "PATFUEL": "Patent fuel",
    "OVENCOKE": "Coke oven coke",
    "GASCOKE": "Gas coke",
    "COALTAR": "Coal tar",
    "BKB": "BKB",
    "GASWKSGS": "Gas works gas",
    "COKEOVGS": "Coke oven gas",
    "BLFURGS": "Blast furnace gas",
    "OGASES": "Other recovered gases",
    "PEAT": "Peat",
    "PEATPROD": "Peat products",
    "OILSHALE": "Oil shale and oil sands",
    "NATGAS": "Natural gas",
    "CRNGFEED": "Crude/NGL/feedstocks (if no detail)",
    "CRUDEOIL": "Crude oil",
    "NGL": "Natural gas liquids",
    "REFFEEDS": "Refinery feedstocks",
    "ADDITIVE": "Additives/blending components",
    "NONCRUDE": "Other hydrocarbons",
    "REFINGAS": "Refinery gas",
    "ETHANE": "Ethane",
    "LPG": "Liquefied petroleum gases (LPG)",
    "NONBIOGASO": "Motor gasoline excl. biofuels",
    "AVGAS": "Aviation gasoline",
    "JETGAS": "Gasoline type jet fuel",
    "NONBIOJETK": "Kerosene type jet fuel excl. biofuels",
    "OTHKERO": "Other kerosene",
    "NONBIODIES": "Gas/diesel oil excl. biofuels",
    "RESFUEL": "Fuel oil",
    "NAPHTHA": "Naphtha",
    "WHITESP": "White spirit & SBP",
    "LUBRIC": "Lubricants",
    "BITUMEN": "Bitumen",
    "PARWAX": "Paraffin waxes",
    "PETCOKE": "Petroleum coke",
    "ONONSPEC": "Other oil products",
    "INDWASTE": "Industrial waste",
    "MUNWASTER": "Municipal waste (renewable)",
    "MUNWASTEN": "Municipal waste (non-renewable)",
    "PRIMSBIO": "Primary solid biofuels",
    "BIOGASES": "Biogases",
    "BIOGASOL": "Biogasoline",
    "BIODIESEL": "Biodiesels",
    "BIOJETKERO": "Bio jet kerosene",
    "OBIOLIQ": "Other liquid biofuels",
    "RENEWNS": "Non-specified primary biofuels and waste",
    "CHARCOAL": "Charcoal",
    "MANGAS": "Elec/heat output from non-specified manufactured gases",
    "HEATNS": "Heat output from non-specified combustible fuels",
    "NUCLEAR": "Nuclear",
    "HYDRO": "Hydro",
    "GEOTHERM": "Geothermal",
    "SOLARPV": "Solar photovoltaics",
    "SOLARTH": "Solar thermal",
    "TIDE": '"Tide, wave and ocean"',
    "WIND": "Wind",
    "OTHER": "Other sources",
    "ELECTR": "Electricity",
    "HEAT": "Heat",
    "TOTAL": "Total",
    "MRENEW": "Memo: Renewables",
}

step1cols=["Population",
            "Y_DEN",
            "ENLONG",
            "BETA_ENLONG",
            "ALPHA_ENLONG",
            "ALPHA",
            "R_SQUARED",
            "BETA",
            "ENSHORT",
            "HIST_START_YEAR",
            "HIST_END_YEAR",
            "ENSHORT_RATIO",
            "ENLONG_RATIO",
            "GDP",
            "GDPCUM",
            "ENSHORT_INIT",
            "ENSHORT_REF",
            "TARGET",
            "SECTOR",
            "FUNC",
            "OPT_RATIO",
            "COUNTRYLIST",
            "MAX_TC",
            "ENSHORT_HIST",
            "CONV_WEIGHT"]