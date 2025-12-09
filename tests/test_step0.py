import pickle
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

import downscaler
from downscaler import CONSTANTS
from downscaler.Step0_interpolate_regional_data import main
from downscaler.utils import fun_eu28

downscaler.USE_CACHING = False
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def test_step0(tmpdir, monkeypatch):
    monkeypatch.setattr(CONSTANTS, "RES_DIR", lambda *args, **kwargs: tmpdir)
    monkeypatch.setattr(
        CONSTANTS, "INPUT_DATA_DIR", Path(__file__).parent / "test_data" / "test_step0"
    )
    obs = main(
        "",
        model_folders=["REMIND 3.0", "WITCH 5.0", "MESSAGEix-GLOBIOM_1.1"],
        snapshot_with_all_models=None,  # "snapshot_to_be_split",
        country_marker_list=fun_eu28(),
        previous_projects_folders=None,
        model_reg_folder="ENGAGE_model_mapping",
        col_sheet_dict={},
        scenario_marker_dict={"WITCH 5.0": "GP_Glasgow"},
        rename_df_mapping_dict={
            "REMIND 3.0": {
                "Countries from the Reforming Ecomonies of the Former Soviet Union": "Countries from the Reforming Economies of the Former Soviet Union"
            },
        },
        save_to_csv=False,
        coerce_errors=True,
    )

    obs_default_mapping = obs[0]
    obs_df = obs[1]

    exp_default_mapping = pd.read_csv(
        TEST_DATA_DIR / "test_step0" / "exp_default_mapping.csv",
        index_col=["ISO"],
    )

    exp_df = pd.read_csv(
        TEST_DATA_DIR / "test_step0" / "exp_df.csv",
        index_col=["MODEL", "SCENARIO", "REGION", "VARIABLE", "UNIT"],
    )
    cols = [x for x in exp_default_mapping.columns if "Country" not in x]
    assert_frame_equal(obs_default_mapping[cols], exp_default_mapping[cols])
    assert_frame_equal(obs_df, exp_df)

    with open(
        TEST_DATA_DIR / "test_step0" / "exp_check_variables.pickle", "rb"
    ) as handle:
        exp_check_variables = pickle.load(handle)
    with open(
        TEST_DATA_DIR / "test_step0" / "exp_check_regions.pickle", "rb"
    ) as handle:
        exp_check_regions = pickle.load(handle)
    assert obs[2] == exp_check_variables # Test changed when we added 'Final Energy|Residential and Commercial|Hydrogen'
    assert obs[3] == exp_check_regions

    # To save data use the code below:
    # Save obs[2] to a new .pkl file
    # output_file_path = TEST_DATA_DIR / "test_step0" / "exp_check_variables.pickle"
    # with open(output_file_path, "wb") as file:
    #     pickle.dump(obs[2], file)
