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
  Ds = range(150, 450, 75)
  Mss = range(260, 460, 20)

  system_name = 'saf_skyrmion'

  data = []
  for D in Ds:
    value = []
    for Ms in Mss:

      if D == 750 and Ms == 440:
        value.append(0)  
        continue

      dirname = f'saf_skyrmion-{D}nm-{Ms}kA_m'
      drive_path = f"{RESULTS_PATH}/{dirname}/{system_name}/drive-0/"
      omf_file_path = find_omf_file(drive_path)
      
      k = get_topological_charge(f"{drive_path}/{omf_file_path}")
      
      print(f'Topological charge k={k:.2f} for D,Ms=({D},{Ms})')
      value.append(k)
    data.append(value)
  
  data_matrix = np.array(data)
  fig, ax = plt.subplots()
  im = ax.imshow(data, cmap='cool')

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
      text = ax.text(j, i, f'{data_matrix[len(Ds)-1-i, j]:.2f}', ha='center', va='center', color='b')

  ax.set_title(r"Phase diagram, DMI=1 mJ/m$^2$")
  ax.set_ylabel("D [nm]")
  ax.set_xlabel("Ms [kA/m]")
  
  fig.colorbar(im, ax=ax, label=r"$k$")
  fig.tight_layout()
  
  fig.savefig(f"{IMAGES_PATH}/phase_diagram.png")
  #plt.show()


if __name__ == '__main__':
  create_phase_diagram()