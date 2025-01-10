import os
from downscaler import CONSTANTS
from typing import Optional
from downscaler.utils import fun_read_csv,fun_create_var_as_sum
sectors=['Industry', 'Transportation','Residential and Commercial','Residential', 'Commercial']
ec=['Electricity','Gases','Heat','Hydrogen','Liquids','Solids']
rename_dict={f"Final Energy|{s}":[f'Final Energy|{s}|{e}' for e in ec] for s in sectors}


def main(project:str, model:Optional[str]=None, file_suffix:Optional[str]=None):
    # """
    # This function recalculates sectorial energy demand. More specifically:
    # 1) Renames sectorial energy demand coming from step1b with the suffix `_step1b`.
    #     Example: 'Final Energy|Industry' -> 'Final Energy|Industry_step1b'.  
    #     NOTE: Here sum of sub-sectors (Industrt|electricity, Industry|gases etc). does not necessarily match the total Industry
    # 2) Creates 'Final Energy|Industry' as the sum of ['Final Energy|Industry|Electricity', 'Final Energy|Industry|Gases' etc.]
    #     Updated csv files will be saved in folder `1_Final_Energy\SHAPE_2023\project\updated_final_energy`

    # Parameters
    # ----------
    # project : str
    #     Your project
    # model : Optional[str], optional
    #     Your model, e.g. 'REMIND-MAgPIE 3.2-4.6'. If provided will search only for files containing that model
    # file_suffix : Optional[str], optional
    #     your file suffix, e.g.'2024_01_24'. If provided will search only for files containing that file_suffix

    # """    
    # NOTE: I need to comment out description above otherwise code is not running 

    folder=CONSTANTS.CURR_RES_DIR('step1')/project/'IAMC_DATA'
    folder_out=folder/'updated_final_energy'
    folder_out.mkdir(exist_ok=True)
    file_list= os.listdir(folder)
    file_list=[x for x in file_list if 'native.csv' not in x]
    if model:
        file_list=[x for x in file_list if model in x ]
    if file_suffix:
        file_list=[x for x in file_list if file_suffix in x]
    for x in file_list:
        df=fun_read_csv({'a':folder/x}, True, int)['a']
        variables=df.reset_index().VARIABLE.unique()
        df=df.rename({k:f"{k}_step1b" for k in rename_dict})
        for k,v in rename_dict.items():
            if k in variables:
                df=fun_create_var_as_sum(df, k, v)
        
        df.to_csv(folder_out/f"{x.replace('.csv','_updated_final_energy.csv')}")
        print(f"{x} done")
    print('All done!')
if __name__ == "__main__":
    main(project='SHAPE_2023',
         model='IMAGE 3.3', 
         file_suffix='2023_12_12')