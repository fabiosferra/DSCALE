from pathlib import Path

import pytest

import downscaler
from downscaler.utils import fun_get_ssp

downscaler.USE_CACHING = False
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.mark.parametrize(
    "scenario, defaut_ssp, _scen_dict, exp",
    [
        ("c", None, {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP4"),
        ("aa", "SSP2", {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP2"),
        ("aa", "SSP2", None, "SSP2"),
        ("Baseline ssp5", "SSP2", None, "SSP5"),
        ("Baseline ssp5", "SSP2", {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP5"),
        ("Baseline", "SSP2", {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP2"),
        ("aa ssp4", None, {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP4"),
        ("c ssp2", "SSP2", {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP2"),
        ("c ssp2", "SSP2", {"A": "SSP1", "b": "SSP5", "c ssp2": "SSP4"}, "SSP4"),
        ("c sSp3", "SSP2", {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP3"),
        ("c SSP11A", "SSP2", {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP1"),
        ("c SSP1ssp1", "SSP2", {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "SSP1"),
    ],
)
def test_get_ssp(scenario, defaut_ssp, _scen_dict, exp):
    # Test that getting the ssp scenario returns the correct value
    assert fun_get_ssp(scenario, defaut_ssp, _scen_dict) == exp


@pytest.mark.parametrize(
    "scenario, defaut_ssp, _scen_dict, error_msg",
    [
        ("aa", None, {"A": "SSP1", "b": "SSP5", "c": "SSP4"}, "Unable.*aa"),
        ("aa", None, None, "specify.*default_ssp.*scen_dict"),
        (
            "Baseline",
            "SSP7",
            {"A": "SSP1", "b": "SSP5", "c": "SSP4"},
            "default_ssp.*only allowed.*is: SSP7",
        ),
        (
            "Baseline",
            "SSP2",
            {"A": "SSP8", "b": "SSP5", "c": "SSP4"},
            "_scen_dict.values().*provided: SSP8",
        ),
        (
            "c ssp2 ssp3",
            "SSP2",
            {"A": "SSP1", "b": "SSP5", "c": "SSP4"},
            r"multiple SSPs.*\['SSP2', 'SSP3'\]",
        ),
    ],
)
def test_get_ssp_raises(scenario, defaut_ssp, _scen_dict, error_msg):

    with pytest.raises(ValueError, match=error_msg):
        fun_get_ssp(scenario, defaut_ssp, _scen_dict)
