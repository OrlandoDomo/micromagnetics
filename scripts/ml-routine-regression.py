import typst
import torch
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from ml.training_regression import (
  PhaseDatasetRegression,
  train_model as training_regression
)

from ml.predicting_regression import main as predicting_main
from ml.models import DenseNetwork_BatchNorm, DenseNetwork_DropOut

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datetime import datetime as dt
from pathlib import Path

from config_reader import config_ml
from logger import get_logger

LOGGER = get_logger(__name__, "ml-routine")
LOGGER.info('Logging timestamps are respect to America/Lima timezone')

def main(
  csv_path="../data/csv_data/saf_results_sk.csv",
  csv_path_eval="../data/csv_data/saf_results_sk-validation.csv",
  batch_size=32,
  epochs=1000,
  lr=0.001,
  patience=50
):
  LOGGER.info("Workflow start")

  df = pl.read_csv(csv_path).with_columns(
    pl.col("Sk_bot").round(3).alias("Sk_bot")
  )
  
  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_target = df.select('Sk_bot').to_numpy().flatten()

  # Split data
  X_train, X_val, y_train, y_val = train_test_split(X_raw, Y_target, test_size=0.2, random_state=42)

  # Create datasets
  train_dataset = PhaseDatasetRegression(X_train, y_train, augment=True, fit_scaler=True)
  val_dataset = PhaseDatasetRegression(X_val, y_val, augment=False, scaler=train_dataset.scaler)
  
  # Create dataloaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  LOGGER.info(f"Using device: {device}")

  df_test = pl.read_csv(csv_path_eval)
  
  X_raw_test = df_test.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels_test = df_test.select('Sk_bot').to_numpy().flatten()
  
  val_dataset_test = PhaseDatasetRegression(X_raw_test, Y_labels_test, augment=False, scaler=train_dataset.scaler)
  val_loader_test = DataLoader(val_dataset_test, batch_size=batch_size, shuffle=False)

  predicting_args = {
    'dmi': config_ml['DMI_predict'],
    'ku': config_ml['Ku_predict'],
    'csv_path': csv_path
  }

  sys_inputs = {
    'lr': str(lr),
    'batch-size': str(batch_size),
    'epochs': str(epochs),
    'dmi-value': str(predicting_args['dmi']),
    'ku-value': str(predicting_args['ku'])
  }
  
  now = dt.now().strftime("%d_%m-%H_%M")
  parent_folder = f'../results/training/train-regression-{now}'
  Path(parent_folder).mkdir(parents=True, exist_ok=True)
  
  models_train = {'default': DenseNetwork_DropOut, 'batchnorm': DenseNetwork_BatchNorm}
  for model_arch in models_train.values():
    model = model_arch(n_features=8)
    LOGGER.info(f"Training {model.name} model...")
    # Train
    model, best_epoch, stats, metrics_fig = training_regression(
      model, train_loader, val_loader, device,
      epochs=epochs, lr=lr,
      patience=patience
    )
    
    # Save model
    model_save_path = f'{parent_folder}/{model.name}-regression.pt'
    metrics_img = f'{parent_folder}/{model.name}-metrics.png'
    metrics_fig.savefig(metrics_img, format='png')
    plt.close(metrics_fig)
              
    torch.save({
      'model_state_dict': model.state_dict(),
      'scaler': train_dataset.scaler,
      'model_type': model.type
    }, model_save_path)
    
    LOGGER.info(f"Model saved to {model_save_path}")

    sys_inputs[f'{model.type}-best-epoch'] = str(best_epoch)
    sys_inputs[f'{model.type}-metrics-plot'] = metrics_img
    sys_inputs[f'{model.type}-model-save-path'] = model_save_path

    predicting_args['model_path'] = model_save_path
    predicting_args['save_path'] = f'{parent_folder}/{model.name}-phase-map.png'
    predicting_args['fixed_vmin'] = config_ml['vmin']
    predicting_args['fixed_vmax'] = config_ml['vmax']

    predicting_main(**predicting_args)

    sys_inputs[f'{model.type}-phase-diagram-img'] = predicting_args['save_path']
    
    comparing_args = {
      'dmi': config_ml['DMI_predict_unseen'],
      'ku': config_ml['Ku_predict_unseen'],
      'csv_path': csv_path_eval,
      'model_path': model_save_path,
      'save_path': f'{parent_folder}/{model.name}-unseen-phase-map.png',
      'fixed_vmin': config_ml['vmin'],
      'fixed_vmax': config_ml['vmax']
    }
    
    predicting_main(**comparing_args)

    sys_inputs[f'{model.type}-phase-diagram-img-unseen'] = comparing_args['save_path']

  typst.compile(
    input='report_template_regression.typ',
    output=f'{parent_folder}/report.pdf',
    root='..',
    sys_inputs=sys_inputs
  )

if __name__ == '__main__':
  main()