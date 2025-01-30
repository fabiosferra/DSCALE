# DSCALE
Downscaling Scenarios to the Country level for Assessment of Low carbon Emissions

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the specific language governing
permissions and limitations under the License.

## Description

This tool downscales regional IAMs results to country level. It focuses on Energy and emissions variables including:
- Final Energy (by sector/energy carriers)
- Secondary Energy (by fuel)
- Primary Energy (by fuel, with and w/o CCS technologies)
- Emissions from Energy, Industrial process
- Land Use emissions, Non-CO2 emissions, total GHG emissions

## How to install
The easiest way to install DSCALE is using:
```console
# clone directly from github
git clone git@github.com:fabiosferra/DSCALE
# navigate into the newly created folder
cd DSCALE
# install using pip
pip install -e .[tests,dev]
```
Please note that the `-e` option as well as the `[tests,dev]` are optional and mostly for
development.


## How to run

0. If you have already installed the downscaler please skip this, and go to (`1. get the data`). Otherwise, please setup/install downscaler code:
	- a. clone DSCALE
	- b. create virtual environment
	- c. pip install -e .[dev,test] inside the cloned folder

1. get the data
    - a. download Regional IAMs data e.g. from https://data.ece.iiasa.ac.at/eu-climate-advisory-board/#/downloads; place in e.g. DSCALE\input_data\project_folder\snapshot_v1\...
    - b. Make sure you have all input data (historical data etc.) required for the downscaling. Please note that some of the data are proprietary and cannot be made publicly available. If you have questions please contact sferra@iiasa.ac.at 
    - c. get region mapping file(s) placed in same project folder, e.g. DSCALE\input_data\project_folder\snapshot_v1\...
    

2.  Run the downscaling for a given `project`.
	- a. add a configuration (YAML) file, in your `project_folder`. This file specifies the list of models/regions/targats that you want to downscale and the downscaling steps that you want to run, like in the example below, e.g.:
       ```yaml
            project_folder: "SIMPLE_hindcasting"
            file_suffix: "2023_07_20"
            n_jobs: 6
            step0: False
            model_folders: "snapshot_v1" # if a string, will split all dataframes contained in that folder by model
            snapshot_with_all_models: null #
            country_marker_list: null
            previous_projects_folders: null
            ref_target: "HISTCR_IEA Energy Statistics_(r2022)"
            default_ssp_scenario: "SSP2"
            list_of_models: ["*"]
            list_of_regions: ["*"]
            list_of_targets: ["*"]
            _gdp_pop_down: True
            gdp_model: "NGFS"
            pop_model: "NGFS"
            add_gdp_pop_data: True
            harmonize_eea_data_until: 2010 # 2018
            step1: True
            step1b: True
            step2: True
            step2_pick_one_pathway: False
            step3: True
            step5: True # additional variables
            run_step5_with_policies: False
            step5b: False # sectorial emissions and revenues
            step5c: True # non-co2
            step5c_bis: True # hydrogen share and trade variables
            step5c_tris: True # afolu
            step5d: True # eu27
            step5e: True # harmonize with historical data
            step4: True # policies
            step5e_after_policy: True # harmonize with historical data (after policy adjustments)
            step6: True # by default False
            co2_energy_only: False
            grassi_dynamic: True
            grassi_scen_mapping: { "SSP2 4.5": ["HISTCR"] }
        ```
    - c. RUN `call.py` for your project for all steps by changing the initial bit of `call.py`, e.g.
    ```python
        d= {
        'config_file_name':"project_multiple_regions/config.yaml", # Add path to the `config.yaml` file 
        'list_of_models': ['*MESSAGE*']	, # Run only the MESSAGE model
        'list_of_regions': ['*'], # Run all regions
        'list_of_targets':["*"],# Run all scenarios 
        'file_suffix':'2025_01_30_test', # Suffix of your file name (should contain a date)
        'n_jobs':6, # Run of CPUs for job parallelization 
        "coerce_errors":True # Runs
        }
    ```
3.  Results will be saved in the `results` folder. This folder is divided in different sub-folders reflecting the different downscaling steps. If you run all steps you will find the final data in the `5_Explorer_and_New_Variables` folder
    ```
    └── 📁results
        └── 📁1_Final_Energy
        └── 📁2_Primary_and_Secondary_Energy
        └── 📁3_CCS_and_Emissions
        └── 📁4_Policy_Adjustments
        └── 📁5_Explorer_and_New_Variables
        └── 📁6_Visuals
    ```
Please note that `6_Visuals` folder contains graphs of the downscaled results (that will be created if you run  `step6`).

4. Please open the log file, located in the `input_data/project/logs` to check if your run was successful.
```
└── 📁input_data
    └── 📁project
        └── 📁logs
            └── log_config.yaml_2023-07-11_17-51-22.log
```
If your run was successful, the log file will look like the example below:
```
on 1: 2024-07-12 17:02:56 INFO     Running model *MESSAGE*
on 1: 2024-07-12 17:02:59 INFO     Sucessfully ran file: snapshot_all_regions_RAW_MESSAGEix-GLOBIOM 2.0-M-R12-NGFS.csv
```
Otherwise, please read the log file to get help on how to solve the issue.

### Caching Information

This version of the downscaler uses the joblib library to achieve caching for
performance improvements. In `utils.py`, there is a decorator called
`make_optionally_cacheable`. Adding this decorator to a function enables optional
caching.

Optional means that there is a global varable, defined in `downscaler/__init__.py`
called `USE_CACHING` which defaults to `True`.

By importing `downscaler` the user can change whether or not caching should be used. The
following example illustrates this:

```python
import downscaler
from downscaler.utils import make_optionally_cacheable

@make_optionally_cacheable
def myfunc():
    ...
    return

# setting caching to False
downscaler.USE_CACHING = False
myfunc()
# setting caching to True
downscaler.USE_CACHING = True
myfunc()
```

If dependencies are updated or other issue with the caching arise it might be a good
idea to delete the cache, which is located by default in `input_data/.downscaler_cache`.

