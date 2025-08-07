import matplotlib.pyplot as plt
import discretisedfield as df
import imageio.v3 as iio
import re
import os

from io import BytesIO

RESULTS_PATH_HYSTERESIS = "../results/saf_skyrmion-0.8nm-d_1-hysteresis"
HYSTERESIS_IMAGES = '../images/hysteresis-old/saf_skyrmion-300nm-280kA_m-hysteresis'

def find_omf_file(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile("saf.*omf")
  omf_file_path = list(filter(r.match, dir_list))

  return omf_file_path

def main(D, Ms):
  omf_files = find_omf_file(f'{RESULTS_PATH_HYSTERESIS}/saf_skyrmion-{D}nm-{Ms}kA_m-hysteresis/saf_skyrmion_hysteresis/drive-0')
  images_path = f'../images/hysteresis'

  images = list()
  for i, omf_file in enumerate(omf_files):
    field_path_file = f'{RESULTS_PATH_HYSTERESIS}/saf_skyrmion-{D}nm-{Ms}kA_m-hysteresis/saf_skyrmion_hysteresis/drive-0/{omf_file}'
    read_field = df.Field.from_file(field_path_file)

    fig, axs = plt.subplots(
        figsize=(12, 6),
        nrows=1,
        ncols=2
    )
    read_field.sel(z=5e-9).z.mpl.scalar(ax=axs[0],cmap='coolwarm')
    read_field.sel(z=5e-9).resample((25, 25)).mpl.vector(
        ax=axs[0], use_color=False, color="black"
    )
    read_field.sel(z=12e-9).z.mpl.scalar(ax=axs[1],cmap='coolwarm')
    read_field.sel(z=12e-9).resample((25, 25)).mpl.vector(
        ax=axs[1], use_color=False, color="black"
    )
    
    axs[0].set_title(r"Bottom Layer: $z = 5 \times 10^{-9}$ m")
    axs[1].set_title(r"Top Layer: $z = 12 \times 10^{-9}$ m")
    
    fig.suptitle(
        rf"State at step={i+1}",
        fontsize='xx-large'
    )
    
    buffer = BytesIO()

    fig.savefig(buffer, format='png')
    buffer.seek(0)
    
    images.append(iio.imread(buffer))
    plt.close(fig)

  iio.imwrite(f'{images_path}/hysteresis_stages-{D}nm-{Ms}kA_m.gif', images, fps=2)

if __name__ == '__main__':
  Ds = range(150, 450, 75)
  Mss = range(260, 460, 20)
  for D in Ds:
    for Ms in Mss:
      main(D, Ms)