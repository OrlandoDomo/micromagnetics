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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from datetime import datetime as dt
from pathlib import Path

from config_reader import config_ml
from logger import get_logger

LOGGER = get_logger(__name__, "ml-routine")
LOGGER.info('Logging timestamps are respect to America/Lima timezone')
THRESHOLD = config_ml['bc_threshold']
TOLERANCE = config_ml['sk_tolerance']

PHASE_MAP_MULTI = {
  0: "Ferro",
  1: "Skyrmion",
  2: "Exotico",
  3: "Complejo",
  4: "Skyrmionium",
  5: "Laberinto",
}

PHASE_MAP_BIN = {
  0: "Others",
  1: "Skyrmion"
}

def main(
  csv_path="../data/csv_data/saf_relax-results.csv",
  csv_path_eval="../data/csv_data/saf_relax-hi_res.csv",
  batch_size=64,
  epochs=100,
  lr=0.001,
  patience=50,
  class_names=PHASE_MAP_MULTI
):
  LOGGER.info("Workflow start")

  df = pl.read_csv(csv_path)
  
  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels = df.select('phase_label').to_numpy().flatten()

  # Split data
  X_train, X_val, y_train, y_val = train_test_split(X_raw, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels)

  num_classes = len(np.unique(Y_labels))
  LOGGER.info(f'Classes are {np.unique(Y_labels)}')
  
  if num_classes == 2:
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    class_weights_val = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    LOGGER.info(f"Binary mode detected. pos_weight: {class_weights_val.item():.4f}")
  else:
    computed_weights = compute_class_weight(
      class_weight='balanced',
      classes=np.unique(y_train),
      y=y_train
    )
    class_weights_val = torch.tensor(computed_weights, dtype=torch.float32)
    LOGGER.info(f"Multiclass mode ({num_classes} classes). Class weights: {class_weights_val.numpy()}")
  
  # Create datasets
  train_dataset = PhaseDatasetClassification(X_train, y_train, augment=True, fit_scaler=True)
  val_dataset = PhaseDatasetClassification(X_val, y_val, augment=False, scaler=train_dataset.scaler)
  
  # Create dataloaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  LOGGER.info(f"Using device: {device}")

  predicting_args = {
    'dmi': config_ml['DMI_predict'],
    'ku': config_ml['Ku_predict'],
    'csv_path': csv_path,
    'class_names': class_names
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
  parent_folder = f'../results/training/train-classification-{now}'
  Path(parent_folder).mkdir(parents=True, exist_ok=True)
  
  models_train = {'default': DenseNetwork_DropOut, 'batchnorm': DenseNetwork_BatchNorm}
  for model_arch in models_train.values():
    if num_classes == 2:
      model = model_arch(n_features=8)
    else :
      model = model_arch(n_features=8, num_classes=num_classes)

    LOGGER.info(f"Training {model.name} model...")
    # Train
    model, best_epoch, history, metrics_fig = training_classification(
      model, train_loader, val_loader, device,
      epochs=epochs, lr=lr, class_weights=class_weights_val,
      patience=patience,
      num_classes=num_classes,
      class_names=class_names
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
      'dataset_name': 'Unseen Classification',
      'class_names': class_names
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
    csv_path="../data/csv_data/saf_relax-results-labeled.csv",
    csv_path_eval="../data/csv_data/saf_relax-dmi=0.6-8_ku=0.08-labeled.csv",
    epochs=config_ml['epochs'],
    patience=config_ml['patience'],
    class_names=PHASE_MAP_BIN
  )