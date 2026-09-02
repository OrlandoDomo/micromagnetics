import typst
import torch
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from ml.training import (
  PhaseDatasetClassification,
  train_model as training_classification
)

from ml.predicting import main as predicting_main
from ml.models import DenseNetwork_BatchNorm, DenseNetwork_DropOut
from ml.compare import compare_new_data

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime as dt
from pathlib import Path

from config_reader import config_ml
from logger import get_logger

THRESHOLD = config_ml['bc_threshold']
LOGGER = get_logger(__name__, "ml-routine")
LOGGER.info('Logging timestamps are respect to America/Lima timezone')

def main(
  csv_path="../data/csv_data/saf_hyst_results.csv",
  csv_path_eval="../data/csv_data/saf_hyst-hi_res.csv",
  batch_size=64,
  epochs=1000,
  lr=0.001,
  patience=100
):
  LOGGER.info("Workflow start")

  df = pl.read_csv(csv_path).with_columns([
    pl.when(pl.col("sk_stability") == 'stable')
      .then(1)
      .otherwise(0)
      .alias("Sk")
  ])
  
  LOGGER.info(f"Class distribution: {df['Sk'].value_counts().to_dict()}")

  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels = df.select('Sk').to_numpy().flatten()

  # Split data
  X_train, X_val, y_train, y_val = train_test_split(X_raw, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels)

  # Determine weights
  # num_pos = np.sum(y_train == 1)
  # num_neg = np.sum(y_train == 0)
  # pos_weight_val = torch.tensor([num_neg / num_pos], dtype=torch.float32)

  class_counts = np.bincount(y_train)
  class_weights = 1.0 / class_counts
  sample_weights = np.array([class_weights[t] for t in y_train])

  sampler = WeightedRandomSampler(
      weights=torch.from_numpy(sample_weights).type(torch.FloatTensor),
      num_samples=len(sample_weights),
      replacement=True
  )
  
  # Create datasets
  train_dataset = PhaseDatasetClassification(X_train, y_train, augment=True, fit_scaler=True)
  val_dataset = PhaseDatasetClassification(X_val, y_val, augment=False, scaler=train_dataset.scaler)
  
  # Create dataloaders
  #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  LOGGER.info(f"Using device: {device}")

  #pos_weight_val = pos_weight_val.to(device)

  df_test = pl.read_csv(csv_path_eval).with_columns([
    pl.when(pl.col("sk_stability") == 'stable')
      .then(1)
      .otherwise(0)
      .alias("Sk")
  ])
  
  X_raw_test = df_test.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels_test = df_test.select('Sk').to_numpy().flatten()
  
  val_dataset_test = PhaseDatasetClassification(X_raw_test, Y_labels_test, augment=False, scaler=train_dataset.scaler)
  val_loader_test = DataLoader(val_dataset_test, batch_size=batch_size, shuffle=False)

  predicting_args = {
    'DMI': config_ml['DMI_predict'],
    'Ku': config_ml['Ku_predict'],
    'resolution': config_ml['resolution'],
    'task':'classification'
  }

  sys_inputs = {
    'lr': str(lr),
    'batch-size': str(batch_size),
    'epochs': str(epochs),
    'threshold': str(THRESHOLD),
    'dmi-value': str(predicting_args['DMI']),
    'ku-value': str(predicting_args['Ku']),
    'pm-resolution': str(predicting_args['resolution']),
  }
  
  now = dt.now().strftime("%d_%m-%H_%M")
  parent_folder = f'../results/training/train-hyst-classification-{now}'
  Path(parent_folder).mkdir(parents=True, exist_ok=True)
  
  models_train = {'default': DenseNetwork_DropOut, 'batchnorm': DenseNetwork_BatchNorm}
  for model_arch in models_train.values():
    model = model_arch(n_features=8)
    LOGGER.info(f"Training {model.name} model...")
    # Train
    model, best_epoch, history, metrics_fig = training_classification(
      model, train_loader, val_loader, device,
      epochs=epochs, lr=lr, pos_weight=None,
      patience=patience
    )
    
    # Save model
    save_path = f'{parent_folder}/{model.name}-classification.pt'
    metrics_img = f'{parent_folder}/{model.name}-metrics.png'
    metrics_fig.savefig(metrics_img, format='png')
    plt.close(metrics_fig)
              
    torch.save({
      'model_state_dict': model.state_dict(),
      'scaler': train_dataset.scaler,
      'model_type': model.type
    }, save_path)
    
    LOGGER.info(f"Model saved to {save_path}")

    sys_inputs[f'{model.type}-best-epoch'] = str(best_epoch)
    sys_inputs[f'{model.type}-metrics-plot'] = metrics_img
    sys_inputs[f'{model.type}-model-save-path'] = save_path

    predicting_args['model_path'] = save_path
    predicting_args['save_path'] = f'{parent_folder}/{model.name}-phase-map.png'

    predicting_main(**predicting_args)

    sys_inputs[f'{model.type}-phase-diagram-img'] = predicting_args['save_path']

    oor_image = f'{parent_folder}/{model.name}-oor-test.png'
    oor_fig, _ = compare_new_data(
      model=model,
      val_loader=val_loader_test,
      device=device,
      fig_path=oor_image
    )
    plt.close(oor_fig)
    sys_inputs[f'{model.type}-oor-img'] = oor_image

  typst.compile(
    input='report_template.typ',
    output=f'{parent_folder}/report.pdf',
    root='..',
    sys_inputs=sys_inputs
  )

if __name__ == '__main__':
  main()