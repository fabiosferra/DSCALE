# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:31:51 2019

@author: sferra
"""


## Creating a dictionary with units conversion
unit_conv_dict = {
    ## We convert energy units into EJ/yr (multiply by):
    "EJ/yr": 1,
    "MTOE": 0.041868,
    # We convert GDP units into USD 2005 (multiply by):
    "billion US$2005/yr": 1,
    "billion US$2010/yr OR local currency": 0.91,  ## WDI US deflator => https://databank.worldbank.org/reports.aspx?source=2&series=NY.GDP.DEFL.ZS&country=
    # We convert population units in Million (multiply by):
    "million": 1,
    "Gwa": 1 / 31.71,  # (multiply by):
}

## Calculations for Energy Intensity (either Percentage share or EN/GDP)
ei_type = {
    "Primary Energy": "gdp/va",
    "Final Energy": "gdp/va",
    "Final Energy|Electricity": "share",
    "Electricity Generation": "share",
    "Final Energy|Industry|Electricity": "share",
}

# print(ei_type['Final Energy'])

## Creating a dictionary to map variables names ('IAM': 'IEA')
iea_dict = {
    "Primary Energy": "TPES",
    "Final Energy": "TFC",
    "Final Energy|Electricity": "ELCONS",
    "Electricity Generation": "ELGEN",
    "Extraction_oil_dummy": "empty",
    "Final Energy|Industry|Electricity": "Final Energy|Industry|Electricity",
}

## Creating a dictionary to map variable names ('IAM': 'IEA')
iea_dict_flow = {
    "Transport": "Transportation",
    "Primary Energy": "Total primary consumption",
    "Final Energy": " Total final consumption",
    "Final Energy|Electricity": " Total final consumption",  # https://stackoverflow.com/questions/35552874/get-first-letter-of-a-string-from-column
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    "Electricity Generation": "Total final consumption",  #' Production' #TO BE ADJUSTED LATER!!!!!!!!!!!!!
    "Extraction_oil_dummy": "Production"
    ##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

iea_dict_product = {
    "Primary Energy": "Total",
    "Final Energy": "Total",
    "Final Energy|Electricity": "Electricity",
    "Electricity Generation": " Electricity",
    "Extraction_oil_dummy": "Crude, NGL and feedstocks",
}
