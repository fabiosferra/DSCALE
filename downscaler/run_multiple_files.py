import logging
import shutil
from datetime import datetime
from pathlib import Path
import msvcrt
import yaml
from alive_progress import alive_bar

from downscaler import CONSTANTS, Step0_interpolate_regional_data, Wrapper_all_steps
from downscaler.utils import unique, fun_fuzzy_match


def main(
    input_parameters: dict,
):
    """Runs multiple files (for multiple IAMs)

    Parameters
    ----------
    input_parameters : dict
        Downscaling parameters
    """
    if input_parameters["step0"]:
        Step0_interpolate_regional_data.main(**input_parameters)
    # NOTE: We use `snapshot_all_models.csv` instead of `snapshot_all_regions.csv`
    datadir = (
        CONSTANTS.INPUT_DATA_DIR / input_parameters["project_folder"] / "multiple_df"
    )
    n_files = len(list(datadir.iterdir()))

    log_folder = CONSTANTS.INPUT_DATA_DIR / input_parameters["project_folder"] / "logs"
    log_folder.mkdir(exist_ok=True)

    logging.basicConfig(
        filename=log_folder
        / f"log_{input_parameters['log_file']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    input_parameters.pop("log_file")
    with alive_bar(n_files) as bar:
        ## Blueprint for enhancing run_multiples_files.py
        models=input_parameters['list_of_models']
        # Older (Philip's code) below: it will copy-paste and and run all files found in `datadir` folder:
        #     for file in datadir.iterdir():
        #         try:
        #             snapshot_all_regions = file.parents[1] / "snapshot_all_regions.csv"
        #             shutil.copy(str(file), str(snapshot_all_regions))
        #             logging.info(f"Running file {file}")
        #             Wrapper_all_steps.main(**input_parameters)
        #             logging.info(f"Sucessfully ran file: {file}")
        #         except Exception as e:
        #             logging.exception(f"Error in file: {file}")
        #         finally:
        #             snapshot_all_regions.unlink()
        #             bar()
        mydict={str(x).split('\\')[-1]:x for x in list(datadir.iterdir())}
        mylist=[str(x).split('\\')[:-1] for x in  list(mydict.values())]
        myfolder=list(set(['\\'.join(x) for x in mylist]))[0]
        if models!=["*"]:
            # New code will copy-paste and run only seclected files, based on models selected in `input_parameters['list_of_models']`
            models=input_parameters['list_of_models']
            mylist=[x for x in list(datadir.iterdir())]
            mylist=[str(x) for x in mylist]
            # myfolder=list(set(['\\'.join(x.split('\\')[:-1]) for x in mylist]))[0]
            mylist=[x.split('\\')[-1] for x in mylist]
            mydict={m:fun_fuzzy_match(mylist,m, cutoff=0)[:1][0] for m in models}
        for m,file in mydict.items():
            try:
                snapshot_all_regions = Path(myfolder).parents[0] / "snapshot_all_regions.csv"
                shutil.copy(str(Path(str(myfolder))/file), str(snapshot_all_regions))
                logging.info(f"Running model {m}")
                Wrapper_all_steps.main(**input_parameters)
                logging.info(f"Sucessfully ran file: {file}")
            except Exception as e:
                logging.exception(f"Error in model: {m}")
            finally:
                snapshot_all_regions.unlink()
                bar()
        


def main_with_yaml_config(config_file_name: str, coerce_errors:bool=False, **kwargs):
    with open(Path(CONSTANTS.INPUT_DATA_DIR) / config_file_name) as f:
        input_parameters = yaml.safe_load(f)
    if kwargs:
        # user input to confirm overwriting parameters
        changed = {k: v for k, v in kwargs.items() if k not in input_parameters or v != input_parameters[k]}
        true_list = [k for k, v in kwargs.items() if v is True]
        true_list = true_list + [
            k
            for k, v in input_parameters.items()
            if v == True and "step" in k.lower()
            if k not in changed
        ]
        if not coerce_errors:
            txt = f"Warning - do you wish to manually overwrite the {config_file_name} parameters as follows? \n {changed}"
            txt = f"{txt} \n This means you are about to run: {unique(true_list)}. Do you wish to proceed y/n?"
            # action = input(txt)
            # if action.lower() not in ["yes", "y"]:
            print(txt)
            action = msvcrt.getch()
            if action.lower().decode() not in ["yes", "y"]:
                raise ValueError(f"Simulation aborted by the user (user input={action})")
        print("running...")
        for key, value in kwargs.items():
            input_parameters[key] = value
    if Path(config_file_name).parent != Path(input_parameters["project_folder"]):
        raise ValueError("Project file and config are in conflict")
    input_parameters.update({"log_file": config_file_name.split("/")[1]})
    input_parameters["coerce_errors"]=coerce_errors
    return main(input_parameters)


if __name__ == "__main__":
    main_with_yaml_config(
        # config_file_name="NGFS_2023_s_curve/config.yaml", # LOG-LOG/SCURVE Sensitivity analysis
        config_file_name="myproject_2024_v2/config.yaml", # NGFS 2024 not working
        # config_file_name="myproject_2024/config.yaml", # NGFS 2024
        # config_file_name="SIMPLE_hindcasting_enhanced_GDP/config.yaml", # HINDCASTING
        list_of_models=["*MESSAGE*"],
        # list_of_regions=['AUS', 'CAN', 'CHN', 'HKG', 'MAC', 'NZL', 'TWN'],
        # list_of_models=['*REMIND*'],
        # list_of_regions=['*Can*'],
        # list_of_models=['*'],
        # list_of_regions=['EU27'],
        # list_of_regions=['ALB', 'BIH', 'CHE', 'ISL', 'MKD', 'MNE', 'NOR', 'SRB', 'TUR'],
        file_suffix='2024_11_15_TEST',
        # list_of_targets=["h_cpol"],
        # list_of_targets=["h_cpol",
        #                  #'h_ndc', 'o_1p5c'
        #                  ],
        # list_of_models=["*"],
        # list_of_regions=['SYR'],
        # file_suffix='2024_04_12_ALL_countries',
        # list_of_regions=["*Rest Centra*",'*Western Eu*', '*South Asia*', '*Sub*'],
        # list_of_regions=["*Asia*"],#["*Rest Centrally Planned*"],
        # file_suffix="2024_02_21",
        # add_gdp=False,## Just for this test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # add_gdp_pop_data=False, ## Just for this test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # run_sensitivity=True, # to make step1b graphs (log-log)
        # config_file_name="NGFS_2023/config.yaml",
        # file_suffix="2023_11_07_test",
        # list_of_targets=["h_cpol",
        #                 # "*h_ndc*",  
        #                 #  "*d_delfrag*", 
        #                 #"o_1p5c", 
        #                 #  "*o_2*"
        #                  ],
        # n_sectors=1,#['Final Energy|Residential and Commercial|Gases', 'Final Energy|Transportation|Gases'], # Can be a number or a list of sectors
        run_sensitivity_from_step2_to_5=False,
        random_electricity_weights=False,
        n_jobs=6,
        step0=False,
        step1=True,
        step1b=False,
        step2=False,
        # # # # # # # # step2_pick_one_pathway=False,
        # # # # # # # # # # fun_finalize_step2_all_targets=False,
        step3=False,
        step5=False,  # additional variables
        step5b=False,  # sectorial emissions and revenues
        step5c=False,  # non-co2
        step5c_bis=False,  # hydrogen share aynd trade variables
        step5c_tris=False,  # afolu
        step5d=False,  # eu27 and aggregate results from multiple files
        step5e=False,  # harmonize with historical data
        step4=False,
        step5e_after_policy=False,
        step6=False,
    )
