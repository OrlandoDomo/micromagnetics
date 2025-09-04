import discretisedfield as df
import imageio.v3 as iio
import matplotlib.pyplot as plt
import discretisedfield.tools as dft

import os
import re
import time
import logging

from subprocess import run, PIPE, STDOUT

logger = logging.getLogger(__name__)
IMAGES_PATH = '../images/mumax_skyrmion_training'

def find_ovf_files(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile(".*ovf")
  ovf_file_path = list(filter(r.match, dir_list))[0]
  
  return ovf_file_path

def move_ovf_file(ovf_file, D ,Ms, T):
  
  filename = f'm_D={D}_Ms={Ms}_T={T}'
  
  src = rf'C:\SPIN-UNI\Orlando\micromagnetics\scripts\saf_mumax.out\{ovf_file}'
  dest = rf'C:\SPIN-UNI\Orlando\micromagnetics\ovf_files\{filename}.ovf'
  
  os.rename(src, dest)
  

def create_image_saf(D, Ms, T):
  
  ovf_file = find_ovf_files('saf_mumax.out')
  
  read_field = df.Field.from_file(f'saf_mumax.out/{ovf_file}')
  topological_charge = dft.topological_charge(read_field.sel(z=0.5e-9))
  Sk = f'{topological_charge:.2f}'.replace('.','')
  move_ovf_file(ovf_file, D ,Ms, T)
  px = 1/plt.rcParams['figure.dpi']
  size = 750
  
  fig, axs = plt.subplots(
    figsize=(size*px, size*px),
    nrows=1,
    ncols=1
  )
  
  read_field.sel(z=0.5e-9).z.mpl.scalar(ax=axs, cmap='coolwarm', colorbar=False)
  plt.axis('off')

  filename = f'bottomlayer_D={D}_Ms={Ms}_T={T}_Sk={Sk}.png'#
  
  fig.savefig(
    f"{IMAGES_PATH}/mumax-T_{T}K/{filename}",
    #bbox_inches='tight',
    transparent="True",
    pad_inches=0
  )
  
  plt.close('all')

def run_main(D, Ms, T):
  
  scriptfile = 'saf_mumax.txt'
  
  with open('saf_skyrmion_temp_relax.txt', 'r') as file:
    SAF_SCRIPT = file.read()

  with open(scriptfile, 'w') as f:
    f.write(SAF_SCRIPT.format(
        Msat=Ms,
        D=D,
        T=T
    ))

  start_time = time.time()
  run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
  print(f'Finished simulation mumax')
  sim_time = time.time() - start_time
  
  
  with open('saf_mumax.out/table.txt', 'r') as input:
    output = open(f'../data/table_D={D}_Ms={Ms}_T={T}.txt', 'w')
    output.write(input.read())
  
  logger.info(f'\t Simulation time={sim_time:.2f} s')
  
  create_image_saf(D, Ms, T)
  
if __name__ == '__main__':
  logging.basicConfig(filename='mumax-logger.log', level=logging.INFO)
  
  Ds = range(160,880,80)
  Mss = range(260, 460, 20)
  Ts = range(0, 150, 50)
  
  logger.info('Starting simulation')
  
  for T in Ts:
    for D in Ds:
      for Ms in Mss:
        try:
          logger.info(f'Running simulation for D={D} nm and Msat={Ms} kA/m')
          run_main(D, Ms, T)
        except Exception as e:
          logger.warning(f'Could not run job for D={D} nm and Msat={Ms} kA/m because of {e}')
  