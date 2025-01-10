import pandas as pd
from downscaler import CONSTANTS
from downscaler.fixtures import legend_dict, main_emi_sectors
from downscaler.fixtures import time_dict_top20 as time_dict
from downscaler.input_caching import get_selection_dict
from downscaler.utils import (
    fun_plot_all_maccs_graphs,
    fun_run_analysis_all_mitigation_settings,
)


def main(top: int = None, read_ghg_from_csv: bool = False):
    input_dict = {
        "get_selection_dict": get_selection_dict,
        "project_file": "NGFS_2022",
        "model": "MESSAGEix-GLOBIOM 1.1-M-R12",
        "main_emi_sectors": main_emi_sectors,
        "file_emi": "emissions_clock/Explorer_MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2022_Round_2nd_version2_FINAL_9_August2022_5b_base_year_harmo.csv",
        "file_energy": "emissions_clock/Explorer_MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2022_Round_2nd_version2_FINAL_9_August2022.csv",
        # "file_emi": "2022/MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2022_Round_2nd_2022_12_07_sensitivity_SLOW_step4_FINAL_Emissions_by_sectors_and_revenues.csv",
        # "file_energy": "2022/MESSAGEix-GLOBIOM 1.1-M-R12_NGFS_2022_Round_2nd_2022_12_07_sensitivity_SLOW_step4_FINAL.csv",
        "baseline": "h_ndc",
        "stab": "o_1p5c",
        "time_dict": time_dict,
        "top": top,
        "_show_abs_emi": True,
        "c": None,  # Run single country analysis e.g.  c=['BRA']
    }

    # df_test=pd.read_csv(CONSTANTS.CURR_RES_DIR('step5')/'emissions_clock/Main_table.csv', index_col=['MODEL','VARIABLE','REGION'])
    # df_test=pd.read_csv(CONSTANTS.CURR_RES_DIR('step5')/'test_macc.csv', index_col=['MODEL','VARIABLE','REGION'])
    # fun_plot_all_maccs_graphs(df_test,#.xs('CHN', level='REGION', drop_level=False), 
    #                           legend_dict, _step="pre", complement_to_one=True,indicator='eff')
    
    # How we calcultate mitigation: below 2010 or below h_ndc
    input_spec = {"o_1p5c vs 2010": 2010, "o_1p5c vs h_ndc": False}

    # NOTE: the unit in df_summary is MtCO2 unless otherwise specified
    summary_dict = {}
    for ra in ["standard", "upper", "lower"]:
        df_summary = fun_run_analysis_all_mitigation_settings(
            top, read_ghg_from_csv, input_dict, input_spec, "o_1p5c vs 2010", ra
        )
        summary_dict[ra] = df_summary

    # Blueprint
    # 1) - we add criteria loop for ra in ["standard", "upper", "lower"]
    # 2) - we add convergence loop: med/slow/fast
    # 3) - we add standard results (because "med" criteria are different when we run sensitivity analysis)
    # 4) - we add emissions range in parenthesis to standard results

    # CHECK:
    # fun_plot_all_maccs_graphs(summary_dict["standard"].xs('ALB', level='REGION', drop_level=False),
    #                               legend_dict, _step="pre")
    # Plot MACC graphs
    fun_plot_all_maccs_graphs(summary_dict["standard"], legend_dict, _step="pre", indicator='gov', complement_to_one=True)


if __name__ == "__main__":
    main(top=None, read_ghg_from_csv=False)
