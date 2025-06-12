import matplotlib.pyplot as plt
import micromagneticdata as md
import discretisedfield as df
import numpy as np
import os
import re

RESULTS_PATH = "../results"
IMAGES_PATH = "../images/saf_skyrmion"

def visualize_field(
    field_path_file:str,
    D:float,
    Ms:float
):
  read_field = df.Field.from_file(field_path_file)

  number_cells = int(read_field.mesh.n[0]/2)
  try:
    number_of_non_zeros = np.where(read_field.array[:,number_cells,5,2]/1e3 > 0)[0].shape[0]
    approximated_radius = (number_of_non_zeros*3)/2
  except:
    print('Could not calculate radius')
    approximated_radius = 0
  
  fig, axs = plt.subplots(
    figsize=(12, 6),
    nrows=1,
    ncols=2
  )
  read_field.sel(z=5e-9).z.mpl.scalar(ax=axs[0],cmap='coolwarm')
  read_field.sel(z=17e-9).z.mpl.scalar(ax=axs[1],cmap='coolwarm')

  axs[0].set_title(r"Bottom Layer: $z = 5 \times 10^{-9}$ m")
  axs[1].set_title(r"Top Layer: $z = 17 \times 10^{-9}$ m")

  fig.suptitle(
    rf"Skyrmions states D={D} nm, Ms= {Ms} kA/m, $r={approximated_radius:.1f}$ nm",
    fontsize='xx-large'
  )
  fig.tight_layout()
  fig.savefig(f"{IMAGES_PATH}/skyrmion-{D}nm-{Ms}kA_m.png")
  plt.close()

def find_omf_file(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile("saf.*omf")
  omf_file_path = list(filter(r.match, dir_list))[0]
  print(f"Founde .omf file as {omf_file_path}")
  return omf_file_path

if __name__ == "__main__":
  
  Ds = [300, 375, 450, 525, 600] # nm
  Mss = [380, 400, 420, 440] # kA/m

  simulation_name = "saf_skyrmion"
  data = md.Data(name=simulation_name, dirname=RESULTS_PATH)
  number_drivers = data.n

  for driver in range(number_drivers):
    drive_path = f"{RESULTS_PATH}/{simulation_name}/drive-{driver}/"
    omf_file_path = find_omf_file(drive_path)
    visualize_field(
      f"{drive_path}/{omf_file_path}",
      D=Ds[driver//4],
      Ms=Mss[driver%4]
    )


