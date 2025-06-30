from results_analysis import find_omf_file
import discretisedfield as df
import discretisedfield.tools as dft
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "../results/saf_skyrmion-0.8nm-d_1"
IMAGES_PATH = "../images/saf_skyrmion-0.8nm-z_0.8nm-d_1-results"

def get_topological_charge(field_path_file):
  read_field = df.Field.from_file(field_path_file)

  topological_charge = dft.topological_charge(read_field.sel(z=12e-9))

  return topological_charge

def create_phase_diagram():
  Ds = range(150, 750, 75)
  Mss = range(260, 460, 20)

  system_name = 'saf_skyrmion'

  data = []
  for D in Ds:
    value = []
    for Ms in Mss:

      dirname = f'saf_skyrmion-{D}nm-{Ms}kA_m'
      drive_path = f"{RESULTS_PATH}/{dirname}/{system_name}/drive-0/"
      try:
        omf_file_path = find_omf_file(drive_path)
      except:
        value.append(0)
        continue
      
      k = get_topological_charge(f"{drive_path}/{omf_file_path}")
      
      print(f'Topological charge k={k:.2f} for D,Ms=({D},{Ms})')
      value.append(k)
    data.append(value)
  
  data = np.flip(data, axis=0)
  data_matrix = np.array(data)
  fig, ax = plt.subplots()
  im = ax.imshow(data, cmap='cool', interpolation='nearest')
  
  ax.set_xticks(
    range(len(Mss)),
    labels=[f'{Ms}' for Ms in Mss]
  )
  ax.set_yticks(
    range(len(Ds)),
    labels=[f'{D}' for D in reversed(Ds)]
  )

  for i in range(len(Ds)):
    for j in range(len(Mss)):
      label = f'{data_matrix[i, j]:.2f}'
      text = ax.text(j, i, label, ha='center', va='center', color='b')

  ax.set_title(r"Phase diagram, DMI=0.5 mJ/m$^2$")
  ax.set_ylabel("D [nm]")
  ax.set_xlabel("Ms [kA/m]")
  
  fig.colorbar(im, ax=ax, label=r"$k$")
  fig.tight_layout()
  
  fig.savefig(f"{IMAGES_PATH}/phase_diagram.png")
  #plt.show()


if __name__ == '__main__':
  create_phase_diagram()