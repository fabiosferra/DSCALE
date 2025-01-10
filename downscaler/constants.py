from pathlib import Path
from typing import Union

IFT = Union[Path, str]  # IFT is Input File Type
RFT = Union[Path, str]  # RFT is Results File Type


class constants:

    INPUT_DATA_DIR = Path(__file__).absolute().parents[1] / "input_data"
    RES_DIR = Path(__file__).absolute().parents[1] / "results"
    RESULT_FOLDER_MAPPING = {
        "Energy_demand_downs_1.py": {"folder": "1_Final_Energy", "step": 1},
        "Energy_demand_downs_1_new.py": {"folder": "1_Final_Energy", "step": 1},
        "step1": {"folder": "1_Final_Energy", "step": 1},
        "Energy_demand_sectors_harmonization_1b.py": {
            "folder": "1_Final_Energy",
            "step": 1,
        },
        "Primary_Energy_2a.py": {"folder": "2_Primary_and_Secondary_Energy", "step": 2},
        "step2": {"folder": "2_Primary_and_Secondary_Energy", "step": 2},
        
        "Electricity_2b.py": {"folder": "2_Primary_and_Secondary_Energy", "step": 2},
        "step2b": {"folder": "2_Primary_and_Secondary_Energy", "step": 2},
        "trade_fossil_2c.py": {"folder": "2_Primary_and_Secondary_Energy", "step": 2},
        "CCS_CO2_GDP_Prices_January_3.py": {"folder": "3_CCS_and_Emissions", "step": 3},
        "step3": {"folder": "3_CCS_and_Emissions", "step": 3},
        "harmonise_non_co2_and_lulucf_4a.py": {
            "folder": "4_Policy_Adjustments",
            "step": 4,
        },
        "step4": {
            "folder": "4_Policy_Adjustments",
            "step": 4,
        },
        "Policies_emissions_4.py": {"folder": "4_Policy_Adjustments", "step": 4},
        "step5": {
            "folder": "5_Explorer_and_New_Variables",
            "step": 5,
        },
        "Step_5_Scenario_explorer_5.py": {
            "folder": "5_Explorer_and_New_Variables",
            "step": 5,
        },
        "Scenario_explorer_5.ipynb": {
            "folder": "5_Explorer_and_New_Variables",
            "step": 5,
        },
        "Step_5b_emissions_by_sector_5b.py": {
            "folder": "5_Explorer_and_New_Variables",
            "step": 5,
        },
        "Visual_6.ipynb": {"folder": "6_Visuals", "step": 6},
        "step6": {"folder": "6_Visuals", "step": 6},
        "Energy_demans_downs.ipynb": {"folder": "1_Final_Energy", "step": 1},
        "2a_Primary_Energy.ipynb": {
            "folder": "2_Primary_and_Secondary_Energy",
            "step": 2,
        },
        "2a_Primary_Energy_clean.ipynb": {
            "folder": "2_Primary_and_Secondary_Energy",
            "step": 2,
        },
        "2b_Electricity.ipynb": {"folder": "2_Primary_and_Secondary_Energy", "step": 2},
        "CCS_CO2_GDP_Prices_January.ipynb": {
            "folder": "3_CCS_and_Emissions",
            "step": 3,
        },
        "Policies_emissions.ipynb": {"folder": "4_Policy_Adjustments", "step": 4},
        "step4": {"folder": "4_Policy_Adjustments", "step": 4},
    }
    CACHE_DIR = INPUT_DATA_DIR / ".downscaler_cache"
    CACHE_DIR.mkdir(exist_ok=True)

    def check_for_file(self, fname: str) -> None:
        """Check if the file name is part of the key in RESULT_FOLDER_MAPPING

        Parameters
        ----------
        fname : str
            Name of the file

        Returns
        -------
        None
            If the file is found in the keys

        Raises
        ------
        ValueError
            If the file is not found in the keys
        """
        if fname not in self.RESULT_FOLDER_MAPPING.keys():
            raise ValueError(
                f"{fname} not known only use one of the following: {self.RESULT_FOLDER_MAPPING.keys()}"
            )

    def CURR_RES_DIR(self, fname: str) -> Path:
        """Returns the directory for the results of the current step

        Parameters
        ----------
        fname : str
            [description]

        Returns
        -------
        Path
            Path to the result folder of the current step
        """
        fname = str(Path(fname).name)
        self.check_for_file(fname)
        return self.RES_DIR / self.RESULT_FOLDER_MAPPING[fname]["folder"]

    def PREV_RES_DIR(self, fname: str) -> Union[Path, None]:
        """Returns the result dictionary for the previous step

        Parameters
        ----------
        fname : str
            Name of the current module (e.g. 3_CCS_CO2_GDP...)

        Returns
        -------
        Path
            Path to the result folder of the previous step

        Note
        ----
        This will not work for the first step.
        """
        fname = str(Path(fname).name)
        self.check_for_file(fname)

        prev_step = self.RESULT_FOLDER_MAPPING[fname]["step"] - 1
        if prev_step == -1:
            raise ValueError(
                "Step 1 is the first step. No previous results can be found."
            )
        for res_map in self.RESULT_FOLDER_MAPPING.values():
            if res_map["step"] == prev_step:
                prev_folder = res_map["folder"]
                break
        else:
            raise ValueError()

        return self.RES_DIR / prev_folder
