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

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime as dt
from pathlib import Path

from config_reader import config_ml
from logger import get_logger

THRESHOLD = config_ml['bc_threshold_stability']
LOGGER = get_logger(__name__, "ml-routine")
LOGGER.info('Logging timestamps are respect to America/Lima timezone')

PHASE_MAP = {
  0: "Meta Stable",
  1: "Stable"
}

def main(
  csv_path="../data/csv_data/saf_hyst_results.csv",
  csv_path_eval="../data/csv_data/saf_hyst-hi_res.csv",
  batch_size=64,
  epochs=1000,
  lr=0.001,
  patience=100
):
  LOGGER.info("Workflow start")

  df = pl.read_csv(csv_path)
  
  LOGGER.info(f"Class distribution: {df['phase_label'].value_counts().to_dict()}")

  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels = df.select('phase_label').to_numpy().flatten()

  # Split data
  X_train, X_val, y_train, y_val = train_test_split(X_raw, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels)

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
    'ku-value': str(predicting_args['ku']),
    'threshold': str(THRESHOLD)
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
    model_save_path = f'{parent_folder}/{model.name}-classification.pt'
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
    predicting_args['metrics_save_path'] = f'{parent_folder}/{model.name}-metrics-prediction.png'
    predicting_args['dataset_name'] = 'Predicted Classification'

    predicting_main(**predicting_args)

    sys_inputs[f'{model.type}-phase-diagram-img'] = predicting_args['save_path']
    sys_inputs[f'{model.type}-predicted-metrics-img'] = predicting_args['metrics_save_path']

    comparing_args = {
      'dmi': config_ml['DMI_predict_unseen'],
      'ku': config_ml['Ku_predict_unseen'],
      'csv_path': csv_path_eval,
      'model_path': model_save_path,
      'save_path': f'{parent_folder}/{model.name}-phase-map-unseen.png',
      'metrics_save_path': f'{parent_folder}/{model.name}-metrics-unseen.png',
      'dataset_name': 'Unseen Classification'
    }
    
    predicting_main(**comparing_args)

    sys_inputs[f'{model.type}-phase-diagram-img-unseen'] = comparing_args['save_path']
    sys_inputs[f'{model.type}-metrics-img-unseen'] = comparing_args['metrics_save_path']

  typst.compile(
    input='report_template.typ',
    output=f'{parent_folder}/report.pdf',
    root='..',
    sys_inputs=sys_inputs
  )

if __name__ == '__main__':
  main(
    csv_path="../data/csv_data/saf_hyst_results-labeled.csv",
    csv_path_eval="../data/csv_data/saf_hyst_results-hi_res-labeled.csv",
    class_names=PHASE_MAP
  )