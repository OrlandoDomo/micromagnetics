import pandas as pd
import numpy as np
import discretisedfield as df
import discretisedfield.tools as dft
import json

from config_reader import config

ABS_PATH = config['abs_path']
OVF_FILES_PATH = f"ovf_files/{config['simulation_results_folder']}"
CSV_NAME = config['csv_name']

def export_relax_data():
    
    table_columns = {
        'D': pd.Series(dtype=np.float64),
        'Ms': pd.Series(dtype=np.float64),
        'DMI': pd.Series(dtype=np.float64),
        'Ku': pd.Series(dtype=np.float64),
        'S2k_bot': pd.Series(dtype=np.float64),
        'S2k_top': pd.Series(dtype=np.float64)
    }

    new_table = pd.DataFrame(table_columns)

    i=0

    Ds = range(config['d_min'], config['d_max'], config['d_step'])
    Mss = range(config['ms_min'], config['ms_max'], config['ms_step'])
    
    Kus = range(config['ku_min'], config['ku_max'], config['ku_step'])
    dmis = range(config['dmi_min'], config['dmi_max'], config['dmi_step'])
    
    for dmi in dmis:
        for ku in Kus:
            for D in Ds:
                for Ms in Mss:
                    
                    if ku == 10 or ku == 20:
                        ovf_path_file = f'{ABS_PATH}/{OVF_FILES_PATH}/dmi={dmi/10}/m_D={D}_Ms={Ms}_T=0_dmi={dmi/10}_Ku={ku/100:.1f}.ovf'
                    else:
                        ovf_path_file = f'{ABS_PATH}/{OVF_FILES_PATH}/dmi={dmi/10}/m_D={D}_Ms={Ms}_T=0_dmi={dmi/10}_Ku={ku/100:.2f}.ovf'
                        
                    try:
                        read_field = df.Field.from_file(ovf_path_file)
                    except:
                        print(f'Could not read {ovf_path_file}')
                        continue
                    
                    s2k_bot = dft.topological_charge(read_field.sel(z=0.5e-9), absolute=True)
                    s2k_top = dft.topological_charge(read_field.sel(z=2.5e-9), absolute=True)
                    
                    new_table.loc[i] = [D, Ms, dmi/10, ku/100, s2k_bot, s2k_top]
                    i += 1

    new_table.to_csv(rf'{ABS_PATH}\data\csv_data\{CSV_NAME}.csv')
    
def export_hyst_data():

    table_columns = {
        'D': pd.Series(dtype=np.float64),
        'Ms': pd.Series(dtype=np.float64),
        'DMI': pd.Series(dtype=np.float64),
        'Ku': pd.Series(dtype=np.float64),
        'sk_stability': pd.Series(dtype=np.float64),
    }

    new_table = pd.DataFrame(table_columns)
    T = 0
    i = 0

    Ds = range(config['d_min'], config['d_max'], config['d_step'])
    Mss = range(config['ms_min'], config['ms_max'], config['ms_step'])
    
    Kus = range(config['ku_min'], config['ku_max'], config['ku_step'])
    dmis = range(config['dmi_min'], config['dmi_max'], config['dmi_step'])

    for dmi in dmis:
        data_path = f'../data/saf_results_hyst/dmi={dmi/10}/'
        for ku in Kus:
            for D in Ds:
                for Ms in Mss:
                    
                    filename = f'topological_charge_hyst_D={D}_Ms={Ms}_T={T}_dmi={dmi/10}_Ku={ku/100}.json'
                    try:
                        with open(data_path+filename) as json_file:
                            json_file = json.load(json_file)
                    except:
                        skyrmion_stability = 'none'
                        continue
            
                    sk_values = json_file[f'({D},{Ms})']['s_k']
                    s2k_values = json_file[f'({D},{Ms})']['s2_k']
                    h_values = json_file[f'({D},{Ms})']['H']
                    init = 0
                    final = 34
                    topological_charge_init = s2k_values[init][0]
                    topological_charge_final = s2k_values[final][0]

                    top_charge_diff = np.abs(topological_charge_final - topological_charge_init)
                
                    if top_charge_diff < 1e-1 and abs(abs(s2k_values[final][0])-1) < 0.25:
                        skyrmion_stability = 'stable'
                    else:
                        skyrmion_stability = 'metastable'

                    new_table.loc[i] = [D, Ms, dmi/10, ku/100, skyrmion_stability]
                    i += 1    

    new_table.to_csv(rf'{ABS_PATH}\data\csv_data\saf_hyst-hi_res.csv')

if __name__ == '__main__':
    
    export_hyst_data()