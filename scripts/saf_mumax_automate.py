import discretisedfield as df
import imageio.v3 as iio
import matplotlib.pyplot as plt
import discretisedfield.tools as dft
import pandas as pd

import os
import re
import time
import json

from subprocess import run, PIPE, STDOUT
from io import BytesIO
from pathlib import Path
from logger import get_logger
from config_reader import config

logger = get_logger("saf-hysteresis")
logger.info('Logging timestamps are respect to America/Lima timezone')

IMAGES_PATH = 'images/saf_results_hysteresis'
DATA_PATH = 'data/saf_results_hysteresis'
OVF_FILES_PATH = 'ovf_files/saf_results'
ABS_PATH = config['abs_path']

def find_ovf_files(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile(".*ovf")
  ovf_file_path = list(filter(r.match, dir_list))

  return ovf_file_path

def create_gif_saf(D, Ms, T, dmi, Ku):
  
  ovf_files = find_ovf_files('saf_mumax_hysteresis.out')
  data = pd.read_table("saf_mumax_hysteresis.out/table.txt")
  hvalues = data["B_extz (T)"].to_list()
  images = list()
  
  topological_charges = {}
  sk_charges = []
  s2k_charges = []
  
  for i, ovf_file in enumerate(ovf_files):
    read_field = df.Field.from_file(f'saf_mumax_hysteresis.out/{ovf_file}')
    fig, axs = plt.subplots(
        figsize=(12, 6),
        nrows=1,
        ncols=2
    )
    
    sk = dft.topological_charge(read_field.sel(z=0.5e-9))
    s2k = dft.topological_charge(read_field.sel(z=0.5e-9), absolute=True)
    
    sk_top = dft.topological_charge(read_field.sel(z=2.5e-9))
    s2k_top = dft.topological_charge(read_field.sel(z=2.5e-9), absolute=True)
    
    sk_charges.append((sk, sk_top))
    s2k_charges.append((s2k, s2k_top))
    
    read_field.sel(z=0.5e-9).z.mpl.scalar(ax=axs[0],cmap='bwr', vmax=1, vmin=-1)
    read_field.sel(z=0.5e-9).resample((25, 25)).mpl.vector(
        ax=axs[0], use_color=False, color="black"
    )
    read_field.sel(z=2.5e-9).z.mpl.scalar(ax=axs[1],cmap='bwr', vmax=1, vmin=-1)
    read_field.sel(z=2.5e-9).resample((25, 25)).mpl.vector(
        ax=axs[1], use_color=False, color="black"
    )
    
    axs[0].set_title(r"Bottom Layer: $z = 0.5 \times 10^{-9}$ m")
    axs[1].set_title(r"Top Layer: $z = 2.5 \times 10^{-9}$ m")
    
    fig.suptitle(
        rf"H={hvalues[i]:.2f} T",
        fontsize='xx-large'
    )
    fig.tight_layout()
    buffer = BytesIO()

    fig.savefig(buffer, format='png')
    buffer.seek(0)
    
    images.append(iio.imread(buffer))
    plt.close(fig)

  try:
    topological_charges[f'({D},{Ms})'] = {'H':hvalues, 's_k':sk_charges, 's2_k':s2k_charges}
    with open(f'../{DATA_PATH}/dmi={dmi}/topological_charge_hyst_D={D}_Ms={Ms}_T={T}_dmi={dmi}_Ku={Ku}.json', 'w', encoding='utf-8') as f:
      json.dump(topological_charges, f, ensure_ascii=False, indent=4)
  except:
    logger.warning('No se pudo dumpear el json con la data de carga topologica')
  
  #iio.imwrite(f'{IMAGES_PATH}/saf_skyrmion_hysteresis-{D}nm-{Ms}kA_m.gif', images, fps=2)
  kwargs = {
      'fps': 2,
      'macro_block_size': None
  }
  iio.imwrite(f'../{IMAGES_PATH}/dmi={dmi}/saf_skyrmion_hysteresis-{D}nm-{Ms}kA_m-{Ku}MJ_m3.mp4', images, **kwargs)

def run_main(D, Ms, T, dmi, Ku, if_hysteresis_exists=False):
  scriptfile = 'saf_mumax_hysteresis.txt'

  mp4_filename = f'saf_skyrmion_hysteresis-{D}nm-{Ms}kA_m-{Ku}MJ_m3.mp4'
  
  if if_hysteresis_exists:
    if os.path.isfile(f"../{IMAGES_PATH}/dmi={dmi}/{mp4_filename}"):
      logger.info(f"{mp4_filename}.mp4 file already exists, skipping")
      return 0

  with open('mumax_templates/saf_hysteresis_script_template.txt', 'r') as file:
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
  logger.info(f'\t Simulation time={sim_time:.2f} s')
  
  ovf_filename = f'm_D={D}_Ms={Ms}_T={T}_dmi={dmi}_Ku={Ku}'
  
  try:
    src = rf'{ABS_PATH}\scripts\saf_mumax_hysteresis.out\m0_relaxed.ovf'
    dest = rf'{ABS_PATH}\{OVF_FILES_PATH}\dmi={dmi}\{ovf_filename}.ovf'
    os.rename(src, dest)
  except:
    logger.info(f'ovf_file {ovf_filename} already exists but program will not stop.')
  
  create_gif_saf(D, Ms, T, dmi, Ku)
  logger.info(f"\tFinished converting image for hysteresis at D={D} nm and Ms={Ms} kA/m")
  
  with open('saf_mumax_hysteresis.out/table.txt', 'r') as input:
    output = open(f'{DATA_PATH}/dmi={dmi}/table_hyst_D={D}_Ms={Ms}_T={T}_dmi={dmi}_Ku={Ku}.txt', 'w')
    output.write(input.read())
  
if __name__ == '__main__':
  
  Ds = range(config['d_min'], config['d_max'], config['d_step'])
  Mss = range(config['ms_min'], config['ms_max'], config['ms_step'])
  #Ds = range(150, 825, 75)
  #Mss = range(260, 460, 20)
  Kus = range(config['ku_min'], config['ku_max'], config['ku_step'])
  dmis = range(config['dmi_min'], config['dmi_max'], config['dmi_step'])
  #Kus = range(2,22,2)
  #dmis = range(10,25,5)

  T = config['temperature']
  
  for dmi in dmis:
    Path(rf'{ABS_PATH}\{OVF_FILES_PATH}\dmi={dmi/10}').mkdir(parents=True, exist_ok=True)
    Path(rf'{ABS_PATH}\{IMAGES_PATH}\dmi={dmi/10}').mkdir(parents=True, exist_ok=True)
    Path(rf'{ABS_PATH}\{DATA_PATH}\dmi={dmi/10}').mkdir(parents=True, exist_ok=True)
    
    logger.info(f'Running simulations for DMI={dmi/10} J/m2') # dmi = 0.x - 2.0
    logger.info(f'==========================================')
    for Ku in Kus:
      logger.info(f'---For Ku={Ku/100} MJ/m3---') # Ku = 0.0x - 0.2
      for D in Ds:
        for Ms in Mss:    
          params = {
            'D': D,
            'Ms': Ms,
            'T': T,
            'dmi': dmi/10,
            'Ku': Ku/100,
          }
          try:
            logger.info(f'Running simulation for D={D} nm and Msat={Ms} kA/m')
            run_main(**params)
            #run_main(D=D, Ms=Ms, T=T, dmi=dmi/10, Ku=Ku/100)
          except Exception as e:
            logger.warning(f'Could not run job for D={D} nm and Msat={Ms} kA/m because of {e}')
