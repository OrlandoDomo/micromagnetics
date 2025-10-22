import discretisedfield as df
import imageio.v3 as iio
import matplotlib.pyplot as plt
import discretisedfield.tools as dft

import os
import re
import time
import logging

from subprocess import run, PIPE, STDOUT

def get_logger():
  logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  logging_date_format = "%Y/%m/%d %H:%M:%S %p"
  logger = logging.getLogger(__name__)
  logging.basicConfig(
    handlers = [
      logging.FileHandler(
        filename = "mumax-run.log", 
        encoding = 'utf-8', 
        mode = 'w',
        delay = True
      )
    ],
    format = logging_format,
    datefmt = logging_date_format,
    level = logging.INFO
  )
  return logger

#logger = logging.getLogger(__name__)
logger = get_logger()
logger.info('Logging timestamps are respect to America/Lima timezone')

IMAGES_PATH = '../images/mumax_skyrmion_training'
OVF_FILES_PATH = 'ovf_files_k=0.65'
DATA_PATH = 'data_k=0.65'

def find_ovf_files(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile(".*ovf")
  ovf_file_path = list(filter(r.match, dir_list))[0]
  
  return ovf_file_path

def move_ovf_file(ovf_file, D , Ms, T, dmi):
  
  filename = f'm_D={D}_Ms={Ms}_T={T}_dmi={dmi}'
  
  src = rf'C:\SPIN-UNI\Orlando\micromagnetics\scripts\saf_mumax.out\{ovf_file}'
  dest = rf'C:\SPIN-UNI\Orlando\micromagnetics\{OVF_FILES_PATH}\{filename}.ovf'
  
  os.rename(src, dest)
  

def create_image_saf(D, Ms, T, dmi):
  
  try:
    ovf_file = find_ovf_files('saf_mumax.out')
  except:
    logger.error(f"Didnt find ovf_file for D={D}, Ms={Ms}; probably the program failed one run before")
  
  read_field = df.Field.from_file(f'saf_mumax.out/{ovf_file}')
  topological_charge = dft.topological_charge(read_field.sel(z=0.5e-9))
  Sk = f'{topological_charge:.2f}'.replace('.','')
  move_ovf_file(ovf_file, D ,Ms, T, dmi)
  px = 1/plt.rcParams['figure.dpi']
  size = 800
  
  fig, axs = plt.subplots(
    figsize=(size*px, size*px),
    nrows=1,
    ncols=1
  )
  
  read_field.sel(z=0.5e-9).z.mpl.scalar(ax=axs, cmap='coolwarm', colorbar=False, vmax=1, vmin=-1)
  plt.axis('off')

  filename = f'bottomlayer_D={D}_Ms={Ms}_T={T}_dmi={dmi}_Sk={Sk}.png'
  
  fig.savefig(
    f"{IMAGES_PATH}/saf-skyrmion-images/k=0.65_dmi={dmi}/{filename}",
    #bbox_inches='tight',
    transparent="True",
    pad_inches=0
  )
  
  plt.close('all')

def run_main(D, Ms, T, dmi):
  
  scriptfile = 'saf_mumax.txt'
  
  with open('saf_skyrmion_temp_relax.txt', 'r') as file:
    SAF_SCRIPT = file.read()

  with open(scriptfile, 'w') as f:
    f.write(SAF_SCRIPT.format(
        Msat=Ms,
        D=D,
        T=T,
        DMI=dmi
    ))

  start_time = time.time()
  run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
  sim_time = time.time() - start_time
  
  with open('saf_mumax.out/table.txt', 'r') as input:
    output = open(f'../{DATA_PATH}/table_D={D}_Ms={Ms}_T={T}_dmi={dmi}.txt', 'w')
    output.write(input.read())
  
  logger.info(f'\t Simulation time={sim_time:.2f} s')
  
  create_image_saf(D, Ms, T, dmi)
  
if __name__ == '__main__':
  logging.basicConfig(filename='mumax-logger.log', level=logging.INFO)
  
  Ds = range(160, 880, 80)
  Mss = range(500, 1100, 50)
  #Ts = range(0, 150, 50)
  
  #DMIs = range(5,25,10)
  
  logger.info('Starting simulation')
  T = 0
  
  #for dmi in DMIs:
  dmi=1
  logger.info(f'Running simulations for DMI={dmi}')
  for D in Ds:  
    for Ms in Mss:
      try:
        logger.info(f'Running simulation for D={D} nm and Msat={Ms} kA/m')
        run_main(D, Ms, T, dmi)
      except Exception as e:
        logger.warning(f'Could not run job for D={D} nm and Msat={Ms} kA/m because of {e}')
