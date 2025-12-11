import pandas as pd
import numpy as np
import discretisedfield as df
import discretisedfield.tools as dft

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

    Ds = range(150, 825, 75)
    Mss = range(260, 460, 20)
    Kus = list(np.round(np.linspace(0.02,0.2,10), 2))
    
    for dmi in [0.5,1.0]:
        for ku in Kus:
            for D in Ds:
                for Ms in Mss:
                    
                    if ku == 0.1 or ku == 0.2:
                        ovf_path_file = f'../ovf_files/saf_results/dmi={dmi}/m_D={D}_Ms={Ms}_T=0_dmi={dmi}_Ku={ku:.1f}.ovf'
                    else:
                        ovf_path_file = f'../ovf_files/saf_results/dmi={dmi}/m_D={D}_Ms={Ms}_T=0_dmi={dmi}_Ku={ku:.2f}.ovf'
                        
                    try:
                        read_field = df.Field.from_file(ovf_path_file)
                    except:
                        #print(f'Could not read {ovf_path_file}')
                        continue
                    
                    s2k_bot = dft.topological_charge(read_field.sel(z=0.5e-9), absolute=True)
                    s2k_top = dft.topological_charge(read_field.sel(z=2.5e-9), absolute=True)
                    
                    new_table.loc[i] = [D, Ms, dmi, ku, s2k_bot, s2k_top]
                    i += 1

    new_table.to_csv(rf'C:\SPIN-UNI\Orlando\machine_learning\data\saf_skyrmion_results_final.csv')
    
if __name__ == '__main__':
    
    dmi = 0.5
    ku = 0.05
    export_relax_data()