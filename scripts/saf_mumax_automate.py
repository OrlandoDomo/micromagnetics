import discretisedfield as df
import imageio.v3 as iio
import matplotlib.pyplot as plt
import discretisedfield.tools as dft
import pandas as pd

import os
import re
import time
import logging
import json

from subprocess import run, PIPE, STDOUT
from io import BytesIO

def get_logger():
  logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  logging_date_format = "%Y/%m/%d %H:%M:%S %p"
  logger = logging.getLogger(__name__)
  logging.basicConfig(
    handlers = [
      logging.FileHandler(
        filename = "mumax-run-hysteresis-batistel.log", 
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

logger = get_logger()
logger.info('Logging timestamps are respect to America/Lima timezone')

IMAGES_PATH = '../images/mumax_skyrmion_training/saf-skyrmion-hysteresis_batistel'
OVF_FILES_PATH = '../ove_files/ovf_files_hyst_ku_change'
DATA_PATH = '../data/data_hyst_batistel'

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
    
    sk_charges.append(sk)
    s2k_charges.append(s2k)
    
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
    with open(f'{DATA_PATH}/topological_charge_hyst_D={D}_Ms={Ms}_T={T}_dmi={dmi}_Ku={Ku}.json', 'w', encoding='utf-8') as f:
      json.dump(topological_charges, f, ensure_ascii=False, indent=4)
  except:
    logger.warning('No se pudo dumpear el json con la data de cargaa topologicas')
  
  #iio.imwrite(f'{IMAGES_PATH}/saf_skyrmion_hysteresis-{D}nm-{Ms}kA_m.gif', images, fps=2)
  kwargs = {
      'fps': 2,
      'macro_block_size': None
  }
  iio.imwrite(f'{IMAGES_PATH}/saf_skyrmion_hysteresis-{D}nm-{Ms}kA_m-{Ku}MJ_m3.mp4', images, **kwargs)

def run_main(D, Ms, T, dmi,Ku):
  scriptfile = 'saf_mumax_hysteresis.txt'

  with open('saf_skyrmion_hyst.txt', 'r') as file:
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
  
  create_gif_saf(D, Ms, T, dmi, Ku)
  logger.info(f"\tFinished converting image for hysteresis at D={D} nm and Ms={Ms} kA/m")
  
  with open('saf_mumax_hysteresis.out/table.txt', 'r') as input:
    output = open(f'{DATA_PATH}/table_hyst_D={D}_Ms={Ms}_T={T}_dmi={dmi}_Ku={Ku}.txt', 'w')
    output.write(input.read())
  
if __name__ == '__main__':
  Ds = range(150, 825, 75)
  Mss = range(260, 460, 20)
  #Kus = range(1,11,1)

  dmi = 1.0
  T = 0
  Ku = 0.1  
  
  for D in Ds:
    for Ms in Mss:
      try:
        logger.info(f'Running simulation for D={D} nm and Msat={Ms} kA/m')
        run_main(D, Ms, T, dmi, Ku)
      except Exception as e:
        logger.warning(f'Could not run job for D={D} nm and Msat={Ms} kA/m because of {e}')
  