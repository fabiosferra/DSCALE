import os
from pathlib import Path
from typing import Union

import pandas as pd
import numpy as np

from downscaler import CONSTANTS
from downscaler.Energy_demand_sectors_harmonization_1b import fun_read_df_iam_iamc
from downscaler.fixtures import ec_list_step5 as ec_list
from downscaler.fixtures import (
    fuel_list_step5,
    sectors_list,
    vars_to_be_harmo_step5,
    extra_units_dict,
)
from downscaler.input_caching import InputFile, get_selection_dict
from downscaler.utils_pandas import fun_index_names, fun_add_units
from downscaler.utils_pandas import fun_create_var_as_sum, fun_drop_missing_units
from downscaler.utils import (
    InputFile,
    fun_add_criteria,
    fun_add_gdp_step5,
    fun_add_non_biomass_ren_nomenclature,
    fun_add_twn_results,
    fun_append_gdp,
    fun_append_regional_iam,
    fun_create_input_data_step5,
    fun_create_new_var_dict_step5,
    fun_create_variable_sum,
    fun_downloadable_file,
    fun_downscale_var_using_proxi,
    fun_drop_duplicates,
    fun_exclude_variables_from_df,
    fun_explorer_file,
    fun_get_csv_path_suf,
    fun_get_target_regions_step5,
    fun_make_var_dict,
    fun_native_non_native_countries,
    fun_no_trade_after_2060,
    fun_print_missing_or_duplicated_var,
    fun_print_variable_summary,
    fun_remove_non_biomass_ren_nomenclature,
    fun_rename_sector_and_region,
    fun_save_to_excel,
    fun_select_countries_that_report_variable,
    fun_select_criteria,
    fun_shape_df_step5,
    fun_step5_bottom_up_harmo,
    fun_validation,
    setindex,
    fun_make_var_dict,
    fun_create_var_as_sum,
    fun_drop_columns,
)

RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR(
    str(Path(os.path.abspath("")) / Path(__file__).name)
)
RESULTS_DATA_DIR.mkdir(exist_ok=True)


def main(
    project_name: str = "NGFS",
    add_model_in_region_name: bool = False,
    region_patterns: Union[str, list] = "*",
    model_patterns: Union[str, list] = "*",
    target_patterns: Union[str, list] = "*",
    csv_str: str = "NGFS_November",
    add_gdp: bool = True,
    add_twn: bool = True,
    step4: bool = True,
    _scen_dict=None,
    run_sensitivity: bool = False,
    sensitivity_suffix: str = "",
    input_arguments=None,
    save_to_excel=False,
    **kwargs,
) -> pd.DataFrame:  # sourcery skip: raise-specific-error
    fuel_list = fuel_list_step5
    var_dict_demand, new_var_dict = fun_make_var_dict(
        sectors_list, ec_list, demand_dict=True
    )

    new_var_dict = fun_create_new_var_dict_step5(fuel_list, new_var_dict, sectors_list)

    (
        region_patterns,
        model_patterns,
        target_patterns,
        input_file,
        pyam_mapping_file,
        df_iam_all_models,
        RESULTS_DATA_DIR,
        PREV_STEP_RES_DIR,
        df_ssp,
        selection_dict,
    ) = fun_create_input_data_step5(
        project_name,
        add_model_in_region_name,
        region_patterns,
        model_patterns,
        target_patterns,
        add_twn,
        get_selection_dict,
    )

    ## Read GDP and POP data from Step3
    # csv_suffix = f"results/3_CCS_and_Emissions/GDP_{csv_str}_updated_gdp.csv"
    csv_suffix = f"results/3_CCS_and_Emissions/GDP_{csv_str}_updated_gdp_harmo.csv"  ## harmonized data
    model = "GCAM5.3_NGFS"
    if add_gdp:
        df_gdp = fun_add_gdp_step5(csv_suffix, model)

    for model in selection_dict:
        targets, df_countries, regions = fun_get_target_regions_step5(
            pyam_mapping_file, selection_dict, model
        )
        csv_suffix, suf, my_path = fun_get_csv_path_suf(
            csv_str,
            step4,
            run_sensitivity,
            sensitivity_suffix,
            PREV_STEP_RES_DIR,
            model,
        )
        df = pd.read_csv(my_path / csv_suffix, sep=",")
        df["MODEL"] = model
        # df = fun_index_names(df) # this line could replace the below
        setindex(df, ["MODEL", "SCENARIO", "ISO", "VARIABLE", "UNIT"])
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
        df=fun_drop_columns(df)
        # if "Unnamed: 0" in df.columns:
        #     df = df.drop("Unnamed: 0", axis=1)

        if "CRITERIA" not in df.columns:
            df["CRITERIA"] = "standard"
        criteria = [x for x in df.CRITERIA.unique() if "CRITERIA" not in x]
        df_all = df.set_index("CRITERIA", append=True)

        for ra in criteria:
            df = fun_select_criteria(df_all, ra)
            ## ADDING TWN TO THE LIST OF NATIVE REGIONS IN GCAM (IF TWN IS MISSING)
            ## List of native countries (in the model)
            native_countries, countries_wo_native = fun_native_non_native_countries(
                model, df_countries, regions
            )

            twn = [i for i in native_countries if i == "TWN"]
            if model == "GCAM5.3_NGFS" and not twn:
                print("we dO add TWN!")
                native_countries = native_countries + ["TWN"]

            ## Appending df and df_gdp
            if add_gdp:
                df = fun_append_gdp(model, df_gdp, df, countries_wo_native)

            # Here we add `MODEL_downscaled`
            df, model_dict = fun_downloadable_file(model, df, regions, df_countries)

            ## Excluding Primary Energy emissions variables
            df = fun_exclude_variables_from_df(
                df, ["Primary Energy Emissions"], "VARIABLE"
            )
            df = fun_drop_duplicates(df)
            fun_print_missing_or_duplicated_var(df)

            ## Shifting df if we find 'UNIT' in 2085. Block added on 2021_05_26
            index_shift = df.loc[df["2085"] == "UNIT", :].index
            if (len(index_shift)) != 0:
                raise ValueError(
                    "Need to shift cells: UNIT found in 2085 time period",
                )

            df_iam = fun_read_df_iam_iamc(input_file)
            df_iam.columns = [str(x) for x in df_iam.columns]
            ## 2022_03_15 use method from step1b
            cols = ["MODEL", "UNIT"] + list(df.columns)
            if new_var_dict:
                df_iam, df = fun_step5_bottom_up_harmo(
                    add_model_in_region_name,
                    new_var_dict,
                    model,
                    df_countries,
                    regions,
                    df,
                    df_iam,
                    cols,
                )

            df = fun_rename_sector_and_region(df)

            ## Creating 'Secondary Energy|Eletricity|Trade' as the difference between Secondary and Final electricity (NO need to create a loop   (already applies to all scenario/regions))
            df = fun_create_variable_sum(
                df,
                _new_var_name="Secondary Energy|Electricity|Trade",
                _models_list=[model + "_downscaled", model],
                _add_var_list=["Secondary Energy|Electricity"],
                _subtract_var_list=["Final Energy|Electricity"],
            )
            ## Replacing trade values after 2060 (no electricity generation data)
            df = fun_no_trade_after_2060(df)

            ## NOTE: Here we do not consider:
            # 1.'Secondary Energy|Electricity|Storage Losses',
            # 2.'Secondary Energy|Electricity|Transmission Losses',
            ## These variables could be added (subtracted) after downscaling the two variable above.
            ## However apparently there is no need to subtract those two variables, otherwise we get: Secondary Energy|Electricity|trade = 'Secondary Energy|Electricity|Transmission Losses'

            ## Here we downscale variables that can be calculated by using share within region (based on a proxy variable)
            ## Exampe: CCS|Industrial Processes computed by using Final Energy|Industry as proxi
            ## Adding a simple calculation for CCS|Industry (based on Share of a country in industry final energy in the region), so we can also calculate and report the Carbon Sequestration|CCS (Total)
            ## Can you add a simple calculation for CCS|Industry 2021_05_26 h 16.28
            tup_dict = {
                "Carbon Sequestration|CCS|Industrial Processes": "Final Energy|Industry"
            }

            df = fun_downscale_var_using_proxi(
                df,
                df_iam_all_models,
                model,
                targets,
                df_countries,
                regions,
                tup_dict,
            )
            ##(My new variable name, defined as the sum of those elements => [e.g. Electricity, Gases, Hydrogen, ... ])
            tup_dict = {
                "Carbon Sequestration|CCS": [
                    "Biomass",
                    "Fossil",
                    "Industrial Processes",
                ]
            }
            ## NOTE: we need a loop here because we need to provide 1 single tuple as input to this function.
            # REASON: Otherwise the function will return data only for the last tuple in tup_lists (the function does not update/keep in memory the results from the previous tuples, therefore it only returns the last tuple.
            # We solve the issue by using a loop outside the function)
            # for tup_list in tup_lists:
            for i, j in tup_dict.items():
                # this one would work the same as the below
                df = fun_create_var_as_sum(
                    df, i, [f"{i}|{x}" for x in j], unit="Mt CO2/yr"
                )

            ## WE NEED THE BLOCK BELOW AGAIN AS WE HAVE CREATED DUPLICATED VALUES  FOR NATIVE REGIONS (FOR THE NEW VARIABLES ABOVE, E.G. ['Final Energy|Industry', 'Final Energy|Transportation', 'Final Energy|Residential and Commercial', 'Primary Energy', 'Secondary Energy|Electricity', 'Secondary Energy|Gases', 'Secondary Energy|Liquids'])
            ## Dropping duplicated value (usually gdp or pop data)

            df = fun_drop_duplicates(df)

            # NOTE: the Population below comes from raw SSP data
            # we add Population for TWN for REMIND and MESSAGE
            ## Checking if TWN population is in the df  (for some reasons is missing in REMIND and MESSAGE)
            for target in targets:
                if add_twn:
                    df = fun_add_twn_results(
                        _scen_dict, df_ssp, model, targets, df, target
                    )

            df = fun_drop_duplicates(df)

            # We report only countries for which we have Final Energy information available
            df = fun_select_countries_that_report_variable(df, "Final Energy")

            # Append native IAM regional results
            df = fun_append_regional_iam(
                model,
                df_countries,
                df,
                native_countries,
                df_iam,
            )

            ## SAVE CSV FILE IN EXPLORER FOLDER
            ## 1) INCLUDING ALL COUNTRIES
            folder = RESULTS_DATA_DIR
            csv_suffix = csv_suffix.replace(suf, "")
            path_to_file = folder / (csv_suffix)  ## Downloadable version
            # NOTE - can use the below
            # path_to_file= Path(RESULTS_DATA_DIR/project_name/csv_suffix.replace(f'{model}_','').replace('.csv','')/f'{model}.csv')

            # Nomenclature primary energy:
            # IAM results contain the Non-Biomass Renewables nomenclature:
            # EXAMPLE: df_iam.xs('Primary Energy|Non-Biomass Renewables|Solar', level='VARIABLE')
            df, primary_ren = fun_add_non_biomass_ren_nomenclature(df)

            ## here add regional harmonization step
            print("here we do regional harmonization")
            myindex_names = df.index.names
            if "TIME" not in df_iam_all_models.index.names:
                df_iam_all_models = df_iam_all_models.set_index("TIME")
            mycols = [str(x) for x in df_iam_all_models.index.unique()]
            df = df.iloc[:, df.columns.isin(mycols)]
            df = fun_validation(
                CONSTANTS,
                RESULTS_DATA_DIR,
                selection_dict,
                project_name=project_name,
                model_patterns=model_patterns,
                region_patterns=region_patterns,
                target_patterns=target_patterns,
                vars=vars_to_be_harmo_step5,
                # csv_str=f"MODEL_{csv_str}",
                pd_dataframe_or_csv_str=df,
                harmonize=True,
                _add_model_name_to_region=add_model_in_region_name,
            )

            df = fun_add_units(df, df_iam)
            df = fun_drop_missing_units(df)
            df = fun_shape_df_step5(model, df, model_dict, myindex_names)

            # Workaround: add Final Energy|Heat if missing.
            ec = [ "Liquids",  "Solids", "Electricity","Gases", "Hydrogen", "Heat",]
            sectors= ["Industry", "Transportation", "Residential and Commercial"]
            var_dict_supply = fun_make_var_dict(ec, sectors, demand_dict=False)
            for model in df.reset_index().MODEL.unique():
                if "downscaled" in model:
                    # NOTE as an elaternative you could loop over all variables in var_dict_supply (instead of just "Final Energy|Heat") 
                    if "Final Energy|Heat" not in df.xs(model, level="MODEL").reset_index().VARIABLE.unique():
                        heat=fun_create_var_as_sum(df.xs(model, level="MODEL", drop_level=False), 
                                                   "Final Energy|Heat", 
                                                   var_dict_supply["Final Energy|Heat"],
                                                   unit="EJ/yr")
                        df=pd.concat([df,heat.xs("Final Energy|Heat",level="VARIABLE", drop_level=False)])                
            
            ## Here Saving version for Downloadable file (with standard ISO for all countries and model+"_downscaled" for downscaled countries, otherwise just "model")
            if run_sensitivity:
                path_to_file = str(path_to_file).replace(".csv", f"{suf}.csv")
                df = fun_add_criteria(ra, df)
            mode = "a"
            if ra == criteria[0]:
                mode = "w"

            df.to_csv(path_to_file, mode=mode, header=mode == "w")

            ## Below Saving version for explorer (with D.ISO for downscaled variables, and standard model name (instead of model+'_downscale') for model)
            df = fun_explorer_file(model, df, regions, df_countries)
            path_to_file = folder / ("Explorer_" + csv_suffix)

            if run_sensitivity:
                path_to_file = str(path_to_file).replace(".csv", f"{suf}.csv")
            df.to_csv(
                path_to_file, mode=mode, header=not os.path.exists(path_to_file)
            )  ## ALL COUNTRIES

            # Check for how many countries we have the information contained in test_variables:

            fun_print_variable_summary(model, df, selection_dict)

            if save_to_excel:
                fun_save_to_excel(
                    project_name,
                    str(path_to_file),
                    model,
                    df,
                    input_arguments,
                )

    print("Downscaling Step5 done")
    return df


if __name__ == "__main__":
    project_name = "NGFS_2022"
    model_patterns = ["*REMIND*"]
    region_patterns = ["*"]
    target_patterns = ["*"]
    csv_str = "NGFS_2022_Round_3rd_version_June"
    run_sensitivity = False

    main(
        project_name=project_name,
        add_model_in_region_name=False,
        region_patterns=region_patterns,
        model_patterns=model_patterns,
        target_patterns=target_patterns,
        csv_str=csv_str,
        add_gdp=False,
    )

    try:
        selection_dict = get_selection_dict(
            InputFile(
                CONSTANTS.INPUT_DATA_DIR / project_name / "snapshot_all_regions.csv"
            ),
            model=model_patterns,
            region=region_patterns,
            target=target_patterns,
            variable=["*"],
        )
    except ValueError as e:
        raise ValueError(f"{e}\nConsider using the asterisk '*' for the pattern.")

    fun_validation(
        CONSTANTS,
        RESULTS_DATA_DIR,
        selection_dict,
        project_name=project_name,
        model_patterns=model_patterns,
        region_patterns=region_patterns,
        target_patterns=target_patterns,
        vars=vars_to_be_harmo_step5,
        # vars=["Carbon Sequestration|CCS|Fossil"],
        pd_dataframe_or_csv_str=f"MODEL_{csv_str}",
        harmonize=False,
        save_to_csv=False,
    )
