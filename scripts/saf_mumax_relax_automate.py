import discretisedfield as df
import matplotlib.pyplot as plt
import discretisedfield.tools as dft

import os
import re
import time

from logger import get_logger
from pathlib import Path
from subprocess import run, PIPE, STDOUT

LOGGER = get_logger()
LOGGER.info('Logging timestamps are respect to America/Lima timezone')

IMAGES_PATH = '../images/saf_results_relax'
OVF_FILES_PATH = 'ovf_files/saf_results_relax'
DATA_PATH = '../data/saf_results_relax'

def find_ovf_files(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile(".*ovf")
  ovf_file_path = list(filter(r.match, dir_list))[0]
  
  return ovf_file_path

def move_ovf_file(ovf_file, filename):
  
  src = rf'C:\SPIN-UNI\Orlando\micromagnetics\scripts\saf_mumax.out\{ovf_file}'
  dest = rf'C:\SPIN-UNI\Orlando\micromagnetics\{OVF_FILES_PATH}\m_{filename}.ovf'
  
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
    f"{IMAGES_PATH}/saf-skyrmion-images/{image_filename}",
    #bbox_inches='tight',
    transparent="True",
    pad_inches=0
  )
  
  plt.close('all')

def run_main(D, Ms, T, dmi, Ku):
  
  scriptfile = 'saf_mumax.txt'
  filename = f'D={D}_Ms={Ms}_T={T}_dmi={dmi}_Ku={Ku}'
  
  if os.path.isfile(f"../{OVF_FILES_PATH}/dmi={dmi}/m_{filename}.ovf"):
    LOGGER.info(f"{filename}.ovf file already exists, skipping")
    return 0
  
  with open('mumax_templates/saf_relax_script_template.txt', 'r') as file:
    SAF_SCRIPT = file.read()

  with open(scriptfile, 'w') as f:
    f.write(SAF_SCRIPT.format(
        Msat=Ms,
        D=D,
        T=T,
        DMI=dmi,
        Ku=Ku
    ))

  start_time = time.time()
  run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
  sim_time = time.time() - start_time
  
  with open('saf_mumax.out/table.txt', 'r') as input:
    output = open(f'{DATA_PATH}/table_{filename}.txt', 'w')
    output.write(input.read())
  
  LOGGER.info(f'\t Simulation time={sim_time:.2f} s')
  
  create_image_saf(filename)
  
if __name__ == '__main__':
  T = 0
  
  Ds = range(150, 825, 75)
  Mss = range(260, 460, 20)
  
  DMIs = range(5,25,10)
  Kus = range(1,21,1)
  LOGGER.info('Starting simulation')
  
  for dmi in DMIs:
    Path(rf'C:\SPIN-UNI\Orlando\micromagnetics\ovf_files\saf_results\dmi={dmi/10}').mkdir(parents=True, exist_ok=True)
    LOGGER.info(f'Running simulations for DMI={dmi/10} J/m2') # dmi = 0.x - 2.0
    LOGGER.info(f'==========================================')
    for Ku in Kus:
      LOGGER.info(f'---For Ku={Ku/100} MJ/m3---') # Ku = 0.0x - 0.2
      for D in Ds:
        for Ms in Mss:
          try:
            LOGGER.info(f'Running simulation for D={D} nm and Msat={Ms} kA/m')
            run_main(D=D, Ms=Ms, T=T, dmi=dmi/10, Ku=Ku/100)
          except Exception as e:
            LOGGER.warning(f'Could not run job for D={D} nm and Msat={Ms} kA/m because of {e}')
