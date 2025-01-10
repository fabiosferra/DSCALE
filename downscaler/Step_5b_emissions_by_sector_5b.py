import os
from pathlib import Path
from typing import Union

import pandas as pd

from downscaler import CONSTANTS
from downscaler.fixtures import eff_factor, emi_factor
from downscaler.input_caching import get_selection_dict
from downscaler.utils import (
    InputFile,
    downs_sectorial_revenues,
    downscale_tot_revenues_harmonization,
    fun_add_criteria,
    fun_downloadable_file,
    fun_downscale_emi_and_save_csv,
    fun_explorer_file,
    fun_finalise_and_drop_duplicates,
    fun_get_countrylist_reg_name,
    fun_get_step5_results,
    fun_get_targets_df_countries_regions,
    fun_save_to_excel,
    fun_select_criteria,
    fun_step5b_input_data,
    fun_reg_harmo_step5b,
    fun_xs,
)


def main(
    project_file: str,
    model_patterns: list,
    region_patterns: list,
    target_patterns: list,
    downs_res_file_step5: str,
    csv_out: str,  # if empty string it appends data to existing csv file. Otherwise it creates a new file. CAREFUL: we always have two datasets for each model 'Explorer_Model...csv' and 'Model...csv'!
    model_in_region_name: bool = False,
    add_only_revenues: bool = True,  # add revenues to final dataset, not sectorial emissions
    add_revenues:bool=True,
    save_to_excel: bool = True,
    input_arguments: Union[dict, None] = None,
):

    RESULTS_DATA_DIR = CONSTANTS.CURR_RES_DIR(
        str(Path(os.path.abspath("")) / Path(__file__).name)
    )

    (
        selection_dict,
        cols,
        iamc_index,
        pyam_mapping_file,
        df_iam_all_models,
        df_iea_act,
        df_iam,
        df_emi,
        models,
    ) = fun_step5b_input_data(
        project_file,
        model_patterns,
        region_patterns,
        target_patterns,
        get_selection_dict,
    )

    for model in models:

        df_downs_all, myiso_dict, mymodel_dict = fun_get_step5_results(
            downs_res_file_step5, model, RESULTS_DATA_DIR
        )
        # Add standard criteria, only if there is no "CRITERIA" column in the dataframe
        df_downs_all = fun_add_criteria("standard", df_downs_all)
        if "CRITERIA" in df_downs_all.columns:
            criteria = df_downs_all.CRITERIA.unique()
        else:
            criteria = df_downs_all.index.get_level_values("CRITERIA").unique()
        criteria = [x for x in criteria if "CRITERIA" not in x]

        df_final_all = pd.DataFrame()
        for ra in criteria:
            df_downs = fun_select_criteria(df_downs_all, ra)
            targets, df_countries, regions = fun_get_targets_df_countries_regions(
                selection_dict, pyam_mapping_file, model
            )

            if type(downs_res_file_step5) is not str:
                downs_res_file = "test_results.csv"
            else:
                downs_res_file = downs_res_file_step5
            csv_name_out = downs_res_file.replace("MODEL", model)
            csv_name_out = csv_name_out.replace(".csv", f"{csv_out}.csv")

            df = fun_downscale_emi_and_save_csv(
                project_file,
                region_patterns,
                target_patterns,
                downs_res_file,
                csv_out,
                model_in_region_name,
                selection_dict,
                cols,
                iamc_index,
                df_iam_all_models,
                df_iea_act,
                df_iam,
                df_emi,
                model,
                df_downs,
                myiso_dict,
                mymodel_dict,
                targets,
                df_countries,
                regions,
                RESULTS_DATA_DIR / csv_name_out,
                emi_factor,
                eff_factor,
                ra,
                add_only_revenues,
            )

            df = fun_select_criteria(df, ra)
            if add_revenues: 
                if not [
                    x
                    for x in df_iam.reset_index().VARIABLE.unique()
                    if 'Revenue|' in x
                ]:
                    raise ValueError("`df_iam` does not contain any variables with `Revenue|`."
                                     "If you don't want to downscale revenues please select `add_revenues=False`")
                # Downscale revenues
                for region in regions:
                    countrylist, reg_name = fun_get_countrylist_reg_name(
                        model_in_region_name, model, df_countries, region
                    )
                    if len(countrylist) > 1:
                        # if not len(fun_xs(df, {"REGION":countrylist})):
                        #     txt1=f"`df` does not contain any country from the"
                        #     txt2=f"{region} region ({countrylist} are missing). "
                        #     txt3= 'We skip this region'
                        #     print(f"{txt1}{txt2}{txt3}")
                        #     # raise ValueError(f"{txt1}{txt2}")

                        # else:
                        # Downscale carbon revenues: (at the end of region loop)
                        # NOTE currently it seems we do not save to csv
                        df_rev = downs_sectorial_revenues(
                            downs_res_file,
                            csv_out,
                            model_in_region_name,
                            iamc_index,
                            df_iam,
                            model,
                            myiso_dict,
                            mymodel_dict,
                            region,
                            countrylist,
                            df.reset_index(),
                        )

                        ## NOTE: `df_rev` will be a tuple if `downs_sectorial_revenues` did not work  
                        if isinstance(df_rev, pd.DataFrame):
                            df_rev = fun_add_criteria(ra, df_rev)
                            cols_drop = [2005]
                            cols_drop = cols_drop + [str(x) for x in cols_drop]
                            for x in cols_drop:
                                if x in df_rev.columns:
                                    df_rev = df_rev.drop(x, axis=1)

                            df_rev.to_csv(RESULTS_DATA_DIR / csv_name_out, mode="a")

                # downscaling total revenues and sectorial harmonization
                # NOTE currently it seems we do not save to csv
                (
                    df_downs,
                    myiso_dict,
                    mymodel_dict,
                    df_rev_all,
                ) = downscale_tot_revenues_harmonization(
                    project_file,
                    model_patterns,
                    region_patterns,
                    target_patterns,
                    # downs_res_file,
                    downs_res_file_step5,
                    iamc_index,
                    RESULTS_DATA_DIR,
                    selection_dict,
                    model,
                    RESULTS_DATA_DIR / csv_name_out,  # file_csv,
                    ra,
                )
            else:
                df_rev_all=pd.DataFrame()
                # my_index_names=['MODEL', 'SCENARIO', 'REGION', 'VARIABLE', 'UNIT']

            df_final = fun_finalise_and_drop_duplicates(
                add_only_revenues,
                cols,
                model,
                df_downs,
                myiso_dict,
                mymodel_dict,
                ['MODEL', 'SCENARIO', 'REGION', 'VARIABLE', 'UNIT'],
                df_rev_all,
            )
            df_final = fun_add_criteria(ra, df_final)
            df_final_all = pd.concat([df_final_all, df_final], axis=0)

        # Save results to excel file
        if save_to_excel:
            # Downloadable file below
            fun_save_to_excel(
                project_file,
                str(RESULTS_DATA_DIR / downs_res_file),
                model,
                fun_downloadable_file(model, df_final_all, regions, df_countries)[0],
                input_arguments,
            )

            # Explorer fil below
            fun_save_to_excel(
                project_file,
                str(RESULTS_DATA_DIR / downs_res_file).replace(
                    "MODEL", "Explorer_MODEL"
                ),
                model,
                fun_explorer_file(model, df_final_all, regions, df_countries),
                input_arguments,
            )

    print("downscaling step 5b done")
    return fun_downloadable_file(model, df_final_all, regions, df_countries)[0]


if __name__ == "__main__":
    project_file = "NGFS_2023"
    models = ["*GCAM*"]
    # region_patterns = ["*South Asia*"]  # "*Ind*",
    region_patterns = ["*"]  # ["*Sub-Saha*"] #
    # ["*Centrally Planned*"]#["*Eastern*"]  # ["*"]  # ["*Sub-Saha*"]  # "*Ind*",
    target_patterns = ["*"]#["*d_del*", "*o_1p5c*"]  # ["*o_1p5c*"] #
    # target_patterns = ["*h_ndc*"]
    # file_suffix = "NGFS_2022_Round_2nd_version2_FINAL_emi_clock_3_August2022_FINAL"
    # file_suffix = "NGFS_2022_Round_2nd_version2_FINAL_emi_clock_9_August2022"
    # file_suffix = "NGFS_2022_Round_2nd_version2_FINAL_emi_clock_28_Sept_2022_prova"
    file_suffix="2023_04_25"
    # file_prefix_list = ["Explorer_", ""]
    file_prefix_list=[""]
    # file_prefix_list = ["Explorer_"]
    # for file_prefix in file_prefix_list:
    #     main(
    #         project_file,
    #         models,
    #         region_patterns,
    #         target_patterns,
    #         f"{file_prefix}MODEL_{file_suffix}.csv",
    #         # "_Emissions_by_sectors_and_revenues",  # if empty string it appends data to existing csv file. Otherwise it creates a new file. CAREFUL: we always have two datasets for each model 'Explorer_Model...csv' and 'Model...csv'!
    #         "_5b",  # if empty string it appends data to existing csv file. Otherwise it creates a new file. CAREFUL: we always have two datasets for each model 'Explorer_Model...csv' and 'Model...csv'!
    #         True,
    #         add_only_revenues=False,  # add revenues to final dataset, not sectorial emissions
    #         save_to_excel=True,
    #     )

    try:
        selection_dict = get_selection_dict(
            InputFile(
                CONSTANTS.INPUT_DATA_DIR / project_file / "snapshot_all_regions.csv"
            ),
            model=models,
            region=region_patterns,
            target=target_patterns,
            variable=["*"],
        )
    except ValueError as e:

        raise ValueError(
            f"{e}\nConsider using the asterisk '*' for the pattern."
        ) from e

    # The below is essentially a validation. However we do not use `fun_validation` because:
    # 1) we need to create ad-hoc variables in df_iam_all_models
    # 2) the definition of these two variables is different in df_iam_all_models and downscaled results
    fun_reg_harmo_step5b(
        project_file,
        models,
        region_patterns,
        target_patterns,
        file_suffix,
        selection_dict,
        _harmonize=False,
        project_name=project_file,
    )
