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
from torch.utils.data import DataLoader
from datetime import datetime as dt
from pathlib import Path

from config_reader import config_ml
from logger import get_logger

LOGGER = get_logger(__name__, "ml-routine-matrix")
LOGGER.info('Logging timestamps are respect to America/Lima timezone')
TOLERANCE = config_ml['sk_tolerance']

def main(
  csv_path="../data/csv_data/saf_relax-results.csv",
  csv_path_eval="../data/csv_data/saf_relax-hi_res.csv",
  batch_size=64,
  epochs=100,
  lr=0.001,
  patience=50
):
  LOGGER.info("Starting Matrix Workflow: Conditions x Architectures")

  # ==========================================
  # 1. LOAD RAW DATA ONCE
  # ==========================================
  # Train/Val Data
  df = pl.read_csv(csv_path).with_columns([
      pl.when(abs(pl.col("S2k_bot") - 1) < TOLERANCE).then(1).otherwise(0).alias("Sk")
  ])
  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels = df.select('Sk').to_numpy().flatten()
  X_train, X_val, y_train, y_val = train_test_split(X_raw, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels)

  num_pos = np.sum(y_train == 1)
  num_neg = np.sum(y_train == 0)
  pos_weight_val = torch.tensor([num_neg / num_pos], dtype=torch.float32)

  # OOD Eval Data
  df_test = pl.read_csv(csv_path_eval).with_columns([
      pl.when(abs(pl.col("S2k_bot") - 1) < TOLERANCE).then(1).otherwise(0).alias("Sk")
  ])
  X_raw_test = df_test.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels_test = df_test.select('Sk').to_numpy().flatten()

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  pos_weight_val = pos_weight_val.to(device)

  # ==========================================
  # 2. SETUP DIRECTORY & TYPST DICTIONARY
  # ==========================================
  now = dt.now().strftime("%d_%m-%H_%M")
  parent_folder = f'../results/training/ablation-study-{now}'
  Path(parent_folder).mkdir(parents=True, exist_ok=True)

  predicting_args = {
      'DMI': config_ml['DMI_predict'],
      'Ku': config_ml['Ku_predict'],
      'resolution': config_ml['resolution'],
      'task': 'classification'
  }

  # Base payload for Typst
  sys_inputs = {
      'lr': str(lr),
      'batch-size': str(batch_size),
      'epochs': str(epochs),
      'dmi-value': str(predicting_args['DMI']),
      'ku-value': str(predicting_args['Ku']),
      'pm-resolution': str(predicting_args['resolution']),
  }

  # Define the 3 experimental conditions
  conditions = [
      {'name': 'Baseline', 'augment': False, 'eng_feat': False},
      {'name': 'Augmented', 'augment': True, 'eng_feat': False},
      {'name': 'Engineered', 'augment': False, 'eng_feat': True}
  ]
  
  models_train = {'default': DenseNetwork_DropOut, 'batchnorm': DenseNetwork_BatchNorm}

  # ==========================================
  # 3. NESTED EXECUTION LOOP
  # ==========================================
  for cond in conditions:
    LOGGER.info(f"\n{'='*40}\nSTARTING CONDITION: {cond['name'].upper()}\n{'='*40}")

    # 3a. Rebuild Datasets for this specific condition
    train_dataset = PhaseDatasetClassification(
      X_train, y_train, augment=cond['augment'], eng_feat=cond['eng_feat'], fit_scaler=True
    )
    val_dataset = PhaseDatasetClassification(
      X_val, y_val, augment=False, eng_feat=cond['eng_feat'], scaler=train_dataset.scaler
    )
    test_dataset = PhaseDatasetClassification(
      X_raw_test, Y_labels_test, augment=False, eng_feat=cond['eng_feat'], scaler=train_dataset.scaler
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dims = 8 if cond['eng_feat'] else 4

    # 3b. Loop through architectures
    for model_arch in models_train.values():
      model = model_arch(n_features=input_dims).to(device)
      m_name = model.name
      cond_name = cond['name']
      
      LOGGER.info(f"--- Training {m_name} under {cond_name} ---")

      # --- TRAIN ---
      model, best_epoch, history, metrics_fig = training_classification(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, pos_weight=pos_weight_val,
        patience=patience
      )
      
      save_path = f'{parent_folder}/{m_name}_{cond_name}_best.pt'
      metrics_img = f'{parent_folder}/{m_name}_{cond_name}_metrics.png'
      
      metrics_fig.savefig(metrics_img, format='png')
      plt.close(metrics_fig) # Prevent RAM leak
      
      torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': train_dataset.scaler,
        'model_type': model.type,
        'n_features': input_dims
      }, save_path)

      # --- PREDICT PHASE DIAGRAM ---
      predicting_args['model_path'] = save_path
      predict_img = f'{parent_folder}/{m_name}_{cond_name}_phase_map.png'
      predicting_args['save_path'] = predict_img
      predicting_main(**predicting_args)

      # --- EVALUATE OOD ---
      oor_img = f'{parent_folder}/{m_name}_{cond_name}_oor_test.png'
      oor_fig, _ = compare_new_data(
        model=model, val_loader=test_loader, device=device, fig_path=oor_img
      )
      plt.close(oor_fig) # Prevent RAM leak

      # --- LOG TO TYPST ---
      # Keys follow the format: {ModelType}-{ConditionName}-{Metric}
      # Example: DropoutNet-Baseline-metrics-plot
      sys_inputs[f'{model.type}-{cond_name}-best-epoch'] = str(best_epoch)
      sys_inputs[f'{model.type}-{cond_name}-metrics-plot'] = metrics_img
      sys_inputs[f'{model.type}-{cond_name}-phase-map-img'] = predict_img
      sys_inputs[f'{model.type}-{cond_name}-oor-img'] = oor_img

  # ==========================================
  # 4. COMPILE MASTER REPORT
  # ==========================================
  LOGGER.info("\nCompiling Master Typst Report...")
  typst.compile(
    input='ablation_report_template.typ',
    output=f'{parent_folder}/ablation_report.pdf',
    root='..',
    sys_inputs=sys_inputs
  )
  LOGGER.info(f"Done! Report saved to: {parent_folder}")

if __name__ == '__main__':
  main()