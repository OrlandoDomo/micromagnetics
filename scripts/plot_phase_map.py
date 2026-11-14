import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import ListedColormap
from ml.predicting import main as predicting_main
from config_reader import config_ml
from pathlib import Path

def main():

  output_dir = Path('../results/phase_maps')
  class_names = {
    0: "Ferro",
    1: "Skyrmion",
    2: "Exotico",
    3: "Complejo",
    4: "Skyrmionium",
    5: "Laberinto",
  }
  dmi = config_ml['DMI_predict_unseen']
  ku = config_ml['Ku_predict_unseen']

  csv_path_eval = config_ml['csv_path_eval']
  model_save_path = config_ml['model_save_path']
  img_save_path = f'phase_map-dmi={dmi}-ku={ku}.png'
  metrics_save_path = f'metrics-dmi={dmi}-ku={ku}.png'

  comparing_args = {
    'dmi': dmi,
    'ku': ku,
    'csv_path': csv_path_eval,
    'model_path': model_save_path,
    'save_path': output_dir / img_save_path,
    'metrics_save_path': output_dir / metrics_save_path,
    'dataset_name': 'Unseen Classification',
    'class_names': class_names
  }

  predicting_main(**comparing_args)

if __name__ == '__main__':
  main()