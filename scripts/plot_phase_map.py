import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

TOLERANCE = 0.25

def plot_phase_diagram_scatter(dmi, ku, csv_path):
  
  df = pl.read_csv(csv_path)
  
  df = pl.read_csv(csv_path).with_columns([
    pl.when(abs(pl.col("S2k_bot") - 1) < TOLERANCE)
    .then(1)
    .otherwise(0)
    .alias("Sk")
  ])
  
  filtered_df = df.filter(
    (pl.col("DMI") == dmi) & (pl.col("Ku") == ku)
  )
  
  fig, ax = plt.subplots(figsize=(10, 6))

  ax.scatter(
    filtered_df[ filtered_df['skyrmion_bool'] == True]['Ms'],
    filtered_df[ filtered_df['skyrmion_bool'] == True]['D'],
    marker='o',
    color='red',
    s=200,
    label='Skyrmion'
  )

  ax.scatter(
    filtered_df[ filtered_df['skyrmion_bool'] == False]['Ms'],
    filtered_df[ filtered_df['skyrmion_bool'] == False]['D'],
    marker='x',
    color='blue',
    s=200,
    label='Other Phase'
  )

  Ms_values = filtered_df['Ms'].values
  D_values = filtered_df['D'].values
  
  ax.set_xlabel('M$_s$ [kA/m]', fontsize=12)
  ax.set_ylabel('D [nm]', fontsize=12)
  ax.set_title(f'Phase Diagram Ku={ku} DMI={dmi}', fontsize=14, weight='bold')
  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  ax.grid(True, linestyle='--', alpha=0.4)
  ax.set_xticks(sorted(set(Ms_values)))
  ax.set_yticks(sorted(set(D_values)))

  plt.tight_layout()
  plt.show()
       
def plot_phase_diagram(dmi, ku, csv_path, save_path):

  df = pl.read_csv(csv_path).with_columns([
    pl.when(abs(pl.col("S2k_bot") - 1) < TOLERANCE)
    .then(1)
    .otherwise(0)
    .alias("Sk")
  ])
  
  filtered_df = df.filter(
    (pl.col("DMI") == dmi) & (pl.col("Ku") == ku)
  )
    
  Mss = range(260,460,10)
  Ds = range(150,825,15)
    
  data = []
  for D in Ds:
    value = []
    for Ms in Mss:
      skyrmion = filtered_df.filter(
        (pl.col("Ms") == Ms) & (pl.col("D") == D)
      ).select(pl.col("Sk")).item()
      
      value.append(skyrmion)       
    data.append(value)
  
  data = np.flip(data, axis=0)
  data_matrix = np.array(data)
  
  fig, ax = plt.subplots(figsize=(20, 12))
  im = ax.imshow(data, cmap=ListedColormap(['blue', 'red']), aspect='auto')

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
      label = f'{data_matrix[i, j]:.0f}'
      text = ax.text(j, i, label, ha='center', va='center', color='black')

  ax.set_title(rf"Phase diagram, DMI={dmi} and Ku={ku}")
  ax.set_ylabel("D [nm]")
  ax.set_xlabel("Ms [kA/m]")
  
  fig.colorbar(im, ax=ax, label="Skyrmion", ticks=[0,1])
  fig.tight_layout()
  plt.savefig(save_path)

if __name__ == '__main__':
  dmi = 0.5
  ku = 0.08
  csv_path = '../data/csv_data/saf_relax-hi_res.csv'
  save_path = '../images/saf_results_relax_figures/test.png'
  plot_phase_diagram(dmi, ku, csv_path, save_path)
  