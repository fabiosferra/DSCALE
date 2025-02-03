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


## Warning

This code will not work if you don't you have all input data (historical data etc.) required for the downscaling. Please note that some of the data are proprietary and cannot be made publicly available. If you have questions please contact sferra@iiasa.ac.at 

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
	- a. add a configuration (YAML) file, in your `project_folder`. This file specifies the list of models/regions/targats that you want to downscale and the downscaling steps that you want to run, like in the example below (with default downscaling assumptions), e.g.:
       ```yaml
        # NOTE: Please take a look at scenario_config.csv to set the convergence assumptions for each scenario
        project_folder: "NGFS_2023" 
        file_suffix: 2025_01_31 
        n_jobs: 6
        step0: False
        model_folders: "snapshot_v1" # if a string, will split all dataframes contained in that folder by model
        snapshot_with_all_models: null #"snapshot_to_be_split"
        country_marker_list: null
        previous_projects_folders: null
        ref_target: "h_cpol"
        default_ssp_scenario: "SSP2"
        list_of_models: ["*"]
        list_of_regions: ["*"]
        list_of_targets: ["*"]
        _gdp_pop_down: True
        gdp_model: "NGFS"
        pop_model: "NGFS"
        long_term: "ENLONG_RATIO"
        method: "wo_smooth_enlong"
        add_gdp_pop_data: True
        harmonize_eea_data_until: 2018
        step1: True
        step1b: True
        step2: True
        run_sensitivity: False
        step2_pick_one_pathway: False
        step3: True
        step5: True # additional variables
        run_step5_with_policies: False
        step5b: True # sectorial emissions and revenues
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
        # aggregate_non_iea_countries: True
        grassi_scen_mapping:
        {
            "SSP2 1.9":
            [
                "o_1p5c",
                "o_1p5c_d50",
                "o_1p5c_d95high",
                "o_lowdem",
                "o_lowdem_d95high",
                "o_lowdem_d50",
                "d_rap",
            ],
            "SSP2 2.6":
            [
                "d_delfrag",
                "d_delfrag_d50",
                "d_delfrag_d95high",
                "o_2c",
                "o_2c_d50",
                "o_2c_d95high",
            ],
            "SSP2 3.4":
            [
                "d_strain",
                "h_ndc",
                "h_ndc_d50",
                "h_ndc_d95high",
                "d_strain_d50",
                "d_strain_d95high",
            ],
            "SSP2 4.5": ["h_cpol", "h_cpol_d50", "h_cpol_d95high"],
        }
        # Negative energy variables issues below:
        known_issues:
        {
            "NGFS_2023":
            {
                "MESSAGEix-GLOBIOM 1.1-M-R12":
                {
                    "MESSAGEix-GLOBIOM 1.1-M-R12|South Asia": "Primary Energy|Biomass|Modern",
                },
            },
        }

        ```
    - c. RUN `run_multiple_files.py` for your project for all steps by changing the last bit of `run_multiple_files.py`, e.g.
    ```python
        main_with_yaml_config(
        config_file_name="NGFS_2023/config.yaml", 
        list_of_models=["*MESSAGE*"],
        list_of_regions=['*Pacific OECD*', '*Western Euro*', '*China*','*Sub*'],
        file_suffix='2025_01_31_Test_replicate_paper',
        list_of_targets=["h_cpol", "o_1p5c"],
        run_sensitivity_from_step2_to_5=False,
        random_electricity_weights=False,
        n_jobs=6,
        step0=False,
        # step1=False,
        # step1b=False,
        # step2=False,
        # step3=False,
        # step5=False,  # additional variables
        # step5b=False,  # sectorial emissions and revenues
        # step5c=False,  # non-co2
        # step5c_bis=False,  # hydrogen share aynd trade variables
        # step5c_tris=False,  # afolu
        # step5d=False,  # eu27 and aggregate results from multiple files
        # step5e=False,  # harmonize with historical data
        # step4=False,
        # step5e_after_policy=False,
        step6=False,
    )

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

### Run sensitivity analysis for the `Pacific OECD` region in the `h_cpol` (current policy) scenario
To run a sensitivity analysis, for the current policy scenario of the Pacific OECD region, please update the `run_multiple_files` as follows:
    ```python
            file_suffix='2025_01_31_Test_replicate_paper_sensitivity',
            list_of_regions=['*Pacific OECD*'],
            list_of_targets=["h_cpol"],
            func_type = ["log-log","s-curve"],
            random_electricity_weights=True,
        run_sensitivity_from_step2_to_5=True,
    ```
The sensitivity analysis results for final energy will be saved in the  `📁1_Final_Energy` folder.
The results for secondary energy electricity will be saved in the `📁2_Primary_and_Secondary_Energy` folder.

### Hindcasting tuns 
To run the hindcasting, you need historical energy data from IEA and emissiosn data from PRIMAP. 
Please run `create_regional_files_for_hindcasting.py` to create regional data (to be used as input for the downscaling), based on regional mappings from the GCAM, MESSAGE and REMIND models.
Please save the data in a new project folder (e.g. SIMPLE_hincasting), inside the `input_data` folder
Please run the downscaling (using `run_multuple_files.py`) by referring to this new project folder. Please create a new  config.yaml (within this project folder)  with the following settings:
    ```
    default_ssp_scenario: "SSP2"
    list_of_models: ["*"]
    list_of_regions: ["*"]
    list_of_targets: ["*"]
    _gdp_pop_down: True
    gdp_model: "NGFS"
    pop_model: "NGFS"
    add_gdp_pop_data: True
    harmonize_eea_data_until: 2010
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
    project_folder: "SIMPLE_hindcasting" 
    file_suffix: "2023_07_20" 
    n_jobs: 6
    step0: False
    model_folders: "snapshot_v1" 
    snapshot_with_all_models: null 
    country_marker_list: null
    previous_projects_folders: null
    ref_target: "HISTCR"
    default_ssp_scenario: "SSP2"
    list_of_models: ["*"]
    list_of_regions: ["*"]
    list_of_targets: ["*"]
    _gdp_pop_down: True
    gdp_model: "NGFS"
    pop_model: "NGFS" 
    add_gdp_pop_data: True
    harmonize_eea_data_until: 2010
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


### Visualization scripts
- Please use the file `Paper1_technical_paper.py` to create graphs as they appear in the paper. Please change the last bit to produce specific graphs, like the example below:
    ```python
    if __name__ == "__main__":
    # FOR HINDCASTING PLEASE USE `hindcasting_results_comparison.py`
    # main() # EUAB
    # NGFS Below:
    main(
        # figure_step2b_boxplot=True, # Step 2b sensitivity analysis - Boxplot
        # figure_step2b_violin_plot=True, # Step 2b sensitivity analysis - Violin plot - # CRAMERY MYPALETTE
        models=["*MESSAGE*"],
        project='NGFS_2023',
        figure_2=True, # Energy Intensity plot
        figure_4=True, # Figure 4 (stacked plot)
        # figure_5=True, # Figure 5 (without parallel coordinates)
        # figure_5_parallel=True, # Figure 5 (parallel coordinates)
        # figure_72_boxplot=True, # Figure 7.2 (Step 2b sensitivity analysis) - Boxplot
        # figure_step2b_violin_plot=True, # Step 2b sensitivity analysis (JUST WEIGHTS) - Violin plot - # CRAMERY MYPALETTE
        # figure_72_violin=True, # Figure 7.2 (Step 2b sensitivity analysis: WEIGHTS/DEMAND/CONVERGENCE) - Violin
        # figure_73_weights_scatter=True, # Figure 7.3 (Step 2b sensitivity analysis) - Scatter plot for SUPPLEMENTARY INFORMATION
        # figure_8=True, # Figure 8 (hindcasting)
        )
    ```
    Please make sure that the `file_dict` in the `Paper1_technical_paper.py` contains the name of your file:
    ```python
    file_dict={'NGFS_2023':"MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2023_2025_01_31_Test_replicate_paper_2018_harmo_step5e_WITH_POLICY_None.csv",
                    }   
    ```
    - if you want to create figure 3 of the manuscript, please use the `Step_5e_visuals.py` file with the settings below:
    ```python
        if __name__ == "__main__":
            main(
                files=None,
                models=["*MESSAGE*"],
                project='NGFS_2023',
                step='step1',
                sel_vars=[
                    "Final Energy", # stacked countries plot
                    ],
                markers_dict=markers_dict,
                iea_countries_only=True, 
                countrylist=["AUS" ],
                sel_scen=['h_cpol', "*1p5C*"],
                palette=None,#'flare',
                create_dashbord=None,  # "NGFS_2022_dashboard.pdf"
                eu_ab_plot=False,
                ngfs_plot=False, # Simple Line plots across model/scenario
                florian_plot=False,
                top_20_macc_plot= None,# 'test_macc.csv', #'Main_table.csv',
                log_log_graphs=True,
                stacked_countries_plot=False,
                phase_out_plot=False,
                carbon_budget_plot=False,
                step1_sensitivity=True, 
                step1_extended_sensitivity = True,
                analyse_drivers=False,
                split_df_by_model_and_save_to_csv=False,
                # If you want to plot individual method in each graph, please provide a list of `sel_step1_methods`
                sel_step1_methods=None,  # If None will plot all methods in a single graph. Otherwise provide a list of methods
                # sel_step1_methods=[
                #     "wo_smooth_enlong_ENLONG_RATIO",
                #     "GDPCAP_False_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO",
                #     "GDPCAP_False_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO",
                #     "GDPCAP_True_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO",
                #     "GDPCAP_True_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO",
                #     "TIME_False_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO",
                #     "TIME_False_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO",
                #     "TIME_True_ENSHORT_REF_to_ENLONG_RATIO_2050_thenENLONG_RATIO",
                #     "TIME_True_ENSHORT_REF_to_ENLONG_RATIO_2100_thenENLONG_RATIO",
                # ],
            )
    ```

- If you want to create country dashboards (a PDF file for each country with the main downscaled results), please
specify `step6=True` In the `run_mutliple_file.py`


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

