import matplotlib.pyplot as plt
import discretisedfield as df
import imageio.v3 as iio
import re
import os

from io import BytesIO

RESULTS_PATH_HYSTERESIS = "../results/saf_skyrmion-0.8nm-d_1-hysteresis-llg-old"

def find_omf_file(driver_path):
  dir_list = os.listdir(driver_path)
  r = re.compile("saf.*omf")
  omf_file_path = list(filter(r.match, dir_list))

  return omf_file_path

def main(D, Ms):
  omf_files = find_omf_file(f'{RESULTS_PATH_HYSTERESIS}/saf_skyrmion-{D}nm-{Ms}kA_m-200steps-2ns-100K/saf_skyrmion_hysteresis_llg/drive-2')
  images_path = f'../images'

  images = list()
  for i, omf_file in enumerate(omf_files):
    field_path_file = f'{RESULTS_PATH_HYSTERESIS}/saf_skyrmion-{D}nm-{Ms}kA_m-200steps-2ns-100K/saf_skyrmion_hysteresis_llg/drive-2/{omf_file}'
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
    
    time = 0.01*(i+1)
    fig.suptitle(
        rf"t={time:.2f} ns",
        fontsize='xx-large'
    )
    fig.tight_layout()

    buffer = BytesIO()

    fig.savefig(buffer, format='png')
    buffer.seek(0)
    
    images.append(iio.imread(buffer))
    plt.close(fig)

  iio.imwrite(f'{images_path}/saf_skyrmion-{D}nm-{Ms}kA_m-2ns-100K.gif', images, fps=2)

if __name__ == '__main__':
  main(D=300, Ms=280)