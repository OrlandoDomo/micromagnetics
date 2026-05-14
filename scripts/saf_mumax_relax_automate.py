import discretisedfield as df
import matplotlib.pyplot as plt
import discretisedfield.tools as dft

import os
import re
import time

from logger import get_logger
from config_reader import config
from pathlib import Path
from subprocess import run, PIPE, STDOUT

LOGGER = get_logger(config['log_relax_name'])
LOGGER.info('Logging timestamps are respect to America/Lima timezone')

IMAGES_PATH = f'images/{config["simulation_folder_name"]}'
OVF_FILES_PATH = f'ovf_files/{config["simulation_folder_name"]}'
DATA_PATH = f'data/{config["simulation_folder_name"]}'
ABS_PATH = config['abs_path']

def find_ovf_files(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile(".*ovf")
  ovf_file_path = list(filter(r.match, dir_list))[0]
  
  return ovf_file_path

def move_ovf_file(ovf_file, filename):
  
  dmi = float(re.search(r"dmi=([\d.]+)", filename).group(1))
  src = rf'{ABS_PATH}\scripts\saf_mumax.out\{ovf_file}'
  dest = rf'{ABS_PATH}\{OVF_FILES_PATH}\dmi={dmi}\m_{filename}.ovf'
  
  os.rename(src, dest)

def create_image_saf(filename):
  
  try:
    ovf_file = find_ovf_files('saf_mumax.out')
  except:
    LOGGER.error(f"Didnt find m_{filename}.ovf; probably the program failed before")
  
  read_field = df.Field.from_file(f'saf_mumax.out/{ovf_file}')
  topological_charge = dft.topological_charge(read_field.sel(z=0.5e-9))
  Sk = f'{topological_charge:.2f}'.replace('.','')
  
  move_ovf_file(ovf_file, filename)
  
  px = 1/plt.rcParams['figure.dpi']
  size = 800
  
  fig, axs = plt.subplots(
    figsize=(size*px, size*px),
    nrows=1,
    ncols=1
  )
  
  read_field.sel(z=0.5e-9).z.mpl.scalar(ax=axs, cmap='bwr', colorbar=False, vmax=1, vmin=-1)
  plt.axis('off')

  image_filename = f'bottomlayer_{filename}_Sk={Sk}.png'
  
  fig.savefig(
    f"../{IMAGES_PATH}/{image_filename}",
    #bbox_inches='tight',
    transparent="True",
    pad_inches=0
  )
  
  plt.close('all')

def run_main(D, Ms, T, dmi, Ku, rkky):
  
  scriptfile = 'saf_mumax.txt'
  filename = f'D={D}_Ms={Ms}_T={T}_dmi={dmi}_Ku={Ku}_RKKY=-{rkky}'
  
  if os.path.isfile(f"../{OVF_FILES_PATH}/dmi={dmi}/m_{filename}.ovf"):
    LOGGER.info(f"\t{filename}.ovf file already exists, skipping")
    return 0
  
  with open('mumax_templates/saf_relax_script_template.txt', 'r') as file:
    SAF_SCRIPT = file.read()

  with open(scriptfile, 'w') as f:
    f.write(SAF_SCRIPT.format(
        Msat=Ms,
        D=D,
        T=T,
        DMI=dmi,
        Ku=Ku,
        rkky=rkky
    ))

  start_time = time.time()
  run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
  sim_time = time.time() - start_time
  
  with open('saf_mumax.out/table.txt', 'r') as input:
    output = open(f'../{DATA_PATH}/table_{filename}.txt', 'w')
    output.write(input.read())
  
  LOGGER.info(f'\t Simulation time={sim_time:.2f} s')
  
  create_image_saf(filename)
  
if __name__ == '__main__':
  Ds = range(config['d_min'], config['d_max'], config['d_step'])
  Mss = range(config['ms_min'], config['ms_max'], config['ms_step'])
  
  Kus = range(config['ku_min'], config['ku_max'], config['ku_step'])
  dmis = range(config['dmi_min'], config['dmi_max'], config['dmi_step'])
  rkkys = range(config['rkky_min'], config['rkky_max'], config['rkky_step'])
  
  T = config['temperature']
  LOGGER.info('Starting simulation')
  
  for rkky in rkkys:
    if rkky == 15:  continue
    LOGGER.info(f'=== Running simulations for RKKY=-{rkky/10} J/m ===') # default was rkky = -1.5
    #Path(rf'{ABS_PATH}\{OVF_FILES_PATH}\rkky={rkky/10}').mkdir(parents=True, exist_ok=True)
    for dmi in dmis:
      Path(rf'{ABS_PATH}\{OVF_FILES_PATH}\dmi={dmi/10}').mkdir(parents=True, exist_ok=True)
      LOGGER.info(f'Running simulations for DMI={dmi/10} J/m2') # dmi = 0.x - 2.0
      LOGGER.info(f'==========================================')
      for Ku in Kus:
        LOGGER.info(f'---For Ku={Ku/100} MJ/m3---') # Ku = 0.0x - 0.2      
        for D in Ds:
          for Ms in Mss:
            params = {
              'D': D,
              'Ms': Ms,
              'T': T,
              'dmi': dmi/10,
              'Ku': Ku/100,
              'rkky': rkky/10
            }
            try:
              LOGGER.info(f'Running simulation for D={D} nm and Msat={Ms} kA/m')
              run_main(**params)
              #run_main(D=D, Ms=Ms, T=T, dmi=dmi/10, Ku=Ku/100)
            except Exception as e:
              LOGGER.warning(f'Could not run job for D={D} nm and Msat={Ms} kA/m because of {e}')
