import os
import pandas as pd
from downscaler import CONSTANTS
from downscaler.run_multiple_files import main_with_yaml_config
from downscaler.utils import fun_read_csv, fun_get_models, fun_xs, check_missing_iea_countries, fun_index_names, fun_fill_na_with_previous_next_values,fun_check_missing_data, fun_fuzzy_match, fun_check_missing_variables, fun_flatten_list, fun_regional_country_mapping_as_dict, fun_wildcard
from downscaler.fixtures import iea_countries, check_IEA_countries

# NOTE: Change dictionary below as appropriate
d= {
    'config_file_name':"project_multiple_regions/config.yaml", # Add path to the `config.yaml` file 
    'list_of_models': ['*MESSAGE*']	, # Run only the MESSAGE model
    'list_of_regions': ['*'], # Run all regions
    'list_of_targets':["*"],# Run all scenarios 
    'file_suffix':'2025_01_30_test', # Suffix of your file name (should contain a date)
    'n_jobs':6, # Run of CPUs for job parallelization 
    "coerce_errors":True # Runs
}


steps=['step0', 'step1', 'step1b','step2', 'step3',
       'step5','step5b','step5c','step5c_bis', 
       'step5c_tris','step5d','step5e','step4',
       'step5e_after_policy','step6']

steps={s:False for s in steps}
project=d['config_file_name'].split('/')[0]

# NOTE: Get list of models assuming all files are in `.csv` format. We do this in two steps:
# 1) First we get the list of models from the `snapshot_v1` folder
# 2) Then we get the list of models from the `multiple_df` folder
# Finally we combine them and remove duplicates
# if d['list_of_models'] ==['*'] :

#     models=fun_get_models(project, sub_folder='snapshot_v1', nrows=None) # 1) Read all rows with nrows=None form `snapshot_v1` folder
#     if os.path.exists(CONSTANTS.INPUT_DATA_DIR/project/"multiple_df"):
#         models2=fun_get_models(project, sub_folder='multiple_df') # 2) Read only the first row from `multiple_df` folder
#         models=list(set(models+models2))
# else:
#     models=d['list_of_models']


# A) Run step0 - prepare regional IAMs results
main_with_yaml_config(**{**d,**steps,**{'step0':True}})


# B)  If we have a "GDP_NGFS_merged.csv" file, Run step3 (downscale GDP) for all models and store results in `res` dictionary. 
# We will create a new `SSP_projections.csv` file  based on "GDP_NGFS_merged.csv" 
project_folder=CONSTANTS.INPUT_DATA_DIR/project

# Get list of models
models_available=fun_get_models(project)
models=fun_wildcard(d['list_of_models'], models_available)


if os.path.exists(CONSTANTS.INPUT_DATA_DIR/project/"GDP_NGFS_merged.csv"):
    # we delete the `SSP_projections.csv` file if it already exists
    if os.path.exists(project_folder/'SSP_projections.csv'):
        os.unlink(project_folder/'SSP_projections.csv')

    ## Fill missing GDP/Population data in "GDP_NGFS_merged.csv" and save it to csv
    df= fun_index_names(pd.read_csv(CONSTANTS.INPUT_DATA_DIR/project/"GDP_NGFS_merged.csv"))
    df= fun_fill_na_with_previous_next_values(df)
    df.to_csv(CONSTANTS.INPUT_DATA_DIR/project/"GDP_NGFS_merged.csv")

    res={}
    for model in models:
        # Here we run step3 (to downscale the GDP) for
        main_with_yaml_config(**{**d,**steps,**{'step3':True}, **{'list_of_models':[model], "list_of_regions":["*"]}})
        file=CONSTANTS.CURR_RES_DIR('step3')/f"GDP_{d['file_suffix']}_updated_gdp_harmo.csv"
        res[model]=fun_read_csv({'aa':file}, True, int)['aa']
    df=pd.concat(list(res.values()))
    df.to_csv(project_folder/'SSP_projections.csv')
    mode="a"  
    for model in models:
        # Check mssing data in SSP_projections.csv:
        check_countries= list(set(fun_flatten_list(fun_regional_country_mapping_as_dict(model, project).values()))&(set(check_IEA_countries)))
        iea_countries_excluded=set(check_countries)^set(iea_countries)
        # We exclude IEA countries non included in the model mapping
        missing=check_missing_iea_countries(df.xs(model, level="MODEL", drop_level=False), 
                                            iea_countries_excluded)

        # Complement missing SOCIOECONOMIC data with the following files:
        backup_dict={
            'MESSAGE':'MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2023_2024_02_29_2018_harmo_step5e_WITH_POLICY_None.csv',
            'GCAM': 'GCAM 6.0 NGFS_NGFS_2023_2024_02_29_2018_harmo_step5e_WITH_POLICY_None.csv',
            'REMIND':'REMIND-MAgPIE 3.2-4.6_NGFS_2023_2024_02_21_2018_harmo_step5e_WITH_POLICY_None.csv'
            }

        # TODO:
        # rename models##!!!!!!!!!!! This file does not seem to work anylonger, please double check!!!!!!!!
        ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if len(missing):
            # raise ValueError(f"Data are missing in `SSP_projections.csv`: {missing}")
            txt=f"Data are missing in `GDP_NGFS_merged.csv`: {missing}"
            # backup_project_data='NGFS_2023'
            action=input(f'{txt}. Would you like to add missing data from these files {backup_dict}? y/n?')
            if action.lower() in ["yes", "y"]:
                res={}
                folder=CONSTANTS.CURR_RES_DIR('step5')
                for model in fun_get_models(project):
                    backup_model=fun_fuzzy_match(list(backup_dict.keys()), model, cutoff=0.1)[0]
                    df_backup=fun_read_csv({model:folder/backup_dict[backup_model]},True, int)[model]
                    df_add=pd.concat([fun_xs(df_backup, {'REGION':c, 'VARIABLE':v}) for c,v in missing.items()])
                    model_name_old=df_add.reset_index().MODEL.unique()[0]
                    model_name_new=fun_fuzzy_match(list(df.reset_index().MODEL.unique()), model, cutoff=0.1)[0]
                    res[backup_model]=df_add.rename({model_name_old:model_name_new})
                df_add_all_models=pd.concat(list(res.values()))
                if set(df.reset_index().UNIT.unique())!=set(df_add_all_models.reset_index().UNIT.unique()):
                    raise ValueError('Unable to add missing data: units are different')
                df=pd.concat([df, df_add_all_models])
                df= fun_fill_na_with_previous_next_values(df)
                df.to_csv(project_folder/'SSP_projections.csv', mode=mode)


    # Check for missing scenarios (considering that scenarios may vary across models, reason why 'MODEL' is not specified in `fun_check_missing_data`)
    for x in ['GDP|PPP', 'Population']:
        missing=fun_check_missing_data(df.xs(x, level='VARIABLE'), 'ISO', 'SCENARIO')
        if len(missing)>0:
            raise ValueError(f'Missing {x} data for {missing}')
        
    # Final check to ensure that all models have all variables (after adding missing data)
    for model in models:
        check_countries= list(set(fun_flatten_list(fun_regional_country_mapping_as_dict(model, project).values()))&(set(check_IEA_countries)))
        fun_check_missing_variables(df.xs(model, level="MODEL", drop_level=False),
                                    ["Population","GDP|PPP"], # Variales to be checked 
                                    set(check_countries)&set(iea_countries) # Countries to be checked
                                    )

#C) Run all remaining steps
res={}
# steps={s:s!="step0" for s in steps}
main_with_yaml_config(**{**d,
                        #  **steps, # by commenting this out we run the steps found in config.yaml
                         **{'list_of_models':models}})