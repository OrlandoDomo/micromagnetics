from results_analysis import find_omf_file, visualize_field

RESULTS_PATH = "../results/saf_skyrmion-0.8nm"
IMAGES_PATH = "../images/saf_skyrmion-0.8nm-z_0.8nm-results"

def create_plots():
  Ds = range(150, 825, 75)
  Mss = range(260, 460, 20)

  system_name = 'saf_skyrmion'

  for D in Ds:
    for Ms in Mss:
      dirname = f'saf_skyrmion-{D}nm-{Ms}kA_m'
      drive_path = f"{RESULTS_PATH}/{dirname}/{system_name}/drive-0/"
      try:
        omf_file_path = find_omf_file(drive_path)
      except:
        print(f'Didnt found file for D,Ms = {D},{Ms}')
        continue
      visualize_field(
        f"{drive_path}/{omf_file_path}",
        D=D,
        Ms=Ms,
        images_path=IMAGES_PATH
      )

if __name__ == '__main__':
  create_plots()