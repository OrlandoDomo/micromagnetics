import typst
import torch
import random
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

LOGGER = get_logger(__name__, "ml-evaluate-models")
LOGGER.info('Logging timestamps are respect to America/Lima timezone')
TOLERANCE = config_ml['sk_tolerance']

def set_seed(seed):
  """Locks all random state generators for reproducibility."""
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

def main(
  csv_path="../data/csv_data/saf_relax-results.csv",
  csv_path_eval="../data/csv_data/saf_relax-hi_res.csv",
  batch_size=64,
  epochs=100,
  lr=0.001,
  patience=50,
  n_runs=5 # Number of statistical runs
):
  LOGGER.info(f"Starting Robust Evaluation Workflow: {n_runs} Runs per setup.")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # ==========================================
  # 1. LOAD RAW DATA ONCE
  # ==========================================
  df = pl.read_csv(csv_path).with_columns([
    pl.when(abs(pl.col("S2k_bot") - 1) < TOLERANCE).then(1).otherwise(0).alias("Sk")
  ])
  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels = df.select('Sk').to_numpy().flatten()

  df_test = pl.read_csv(csv_path_eval).with_columns([
    pl.when(abs(pl.col("S2k_bot") - 1) < TOLERANCE).then(1).otherwise(0).alias("Sk")
  ])
  X_raw_test = df_test.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels_test = df_test.select('Sk').to_numpy().flatten()

  # ==========================================
  # 2. SETUP DIRECTORY & TYPST DICTIONARY
  # ==========================================
  now = dt.now().strftime("%d_%m-%H_%M")
  parent_folder = f'../results/training/evaluation-study-{now}'
  Path(parent_folder).mkdir(parents=True, exist_ok=True)

  predicting_args = {
    'DMI': config_ml['DMI_predict'],
    'Ku': config_ml['Ku_predict'],
    'resolution': config_ml['resolution'],
    'task': 'classification'
  }

  sys_inputs = {
    'lr': str(lr),
    'batch-size': str(batch_size),
    'epochs': str(epochs),
    'dmi-value': str(predicting_args['DMI']),
    'ku-value': str(predicting_args['Ku']),
    'pm-resolution': str(predicting_args['resolution']),
    'total-runs': str(n_runs)
  }

  conditions = [
    {'name': 'Baseline', 'augment': False, 'eng_feat': False},
    {'name': 'Augmented', 'augment': True, 'eng_feat': False},
    {'name': 'Engineered', 'augment': False, 'eng_feat': True}
  ]
  models_train = {'dnn_do': DenseNetwork_DropOut, 'dnn_batch': DenseNetwork_BatchNorm}
  random_seeds = [42, 100, 2024, 777, 99][:n_runs]

  # Tracking Dictionary: results['dnn_do']['Baseline'] = [0.91, 0.93, ...]
  results = {m_key: {c['name']: [] for c in conditions} for m_key in models_train.keys()}

  # ==========================================
  # 3. MASTER EVALUATION LOOP
  # ==========================================
  for run_idx, seed in enumerate(random_seeds):
    LOGGER.info(f"\n{'#'*50}\nSTARTING RUN {run_idx + 1}/{n_runs} (Seed: {seed})\n{'#'*50}")
    set_seed(seed)

    # Split data uniquely for this seed
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, Y_labels, test_size=0.2, random_state=seed, stratify=Y_labels
    )
    
    # Recalculate weights based on new split
    num_pos, num_neg = np.sum(y_train == 1), np.sum(y_train == 0)
    pos_weight_val = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)

    for cond in conditions:
      cond_name = cond['name']
      LOGGER.info(f"--- Condition: {cond_name.upper()} ---")

      train_dataset = PhaseDatasetClassification(X_train, y_train, augment=cond['augment'], eng_feat=cond['eng_feat'], fit_scaler=True)
      val_dataset = PhaseDatasetClassification(X_val, y_val, augment=False, eng_feat=cond['eng_feat'], scaler=train_dataset.scaler)
      test_dataset = PhaseDatasetClassification(X_raw_test, Y_labels_test, augment=False, eng_feat=cond['eng_feat'], scaler=train_dataset.scaler)

      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
      test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

      input_dims = 8 if cond['eng_feat'] else 4

      for m_key, model_arch in models_train.items():
        model = model_arch(n_features=input_dims).to(device)
        
        # --- TRAIN ---
        model, best_epoch, history, metrics_fig = training_classification(
          model, train_loader, val_loader, device,
          epochs=epochs, lr=lr, pos_weight=pos_weight_val, patience=patience
        )
        
        save_path = f'{parent_folder}/{m_key}_{cond_name}_best.pt'
        
        # CRITICAL: Save n_features so predicting.py doesn't crash!
        torch.save({
          'model_state_dict': model.state_dict(),
          'scaler': train_dataset.scaler,
          'model_type': model.type,
          'n_features': input_dims 
        }, save_path)

        # --- EVALUATE OOD ---
        oor_img = f'{parent_folder}/{m_key}_{cond_name}_oor_test.png'
        
        # Assumes your compare_new_data returns the Figure AND the metrics dict
        oor_fig, ood_metrics = compare_new_data(
          model=model, val_loader=test_loader, device=device, fig_path=oor_img
        )
        
        # Track numerical result
        run_f1 = ood_metrics['f1_score']
        results[m_key][cond_name].append(run_f1)
        
        # --- VISUALS (ONLY ON RUN 1) ---
        if run_idx == 0:
          metrics_img = f'{parent_folder}/{m_key}_{cond_name}_metrics.png'
          metrics_fig.savefig(metrics_img, format='png')
          
          predicting_args['model_path'] = save_path
          predict_img = f'{parent_folder}/{m_key}_{cond_name}_phase_map.png'
          predicting_args['save_path'] = predict_img
          predicting_main(**predicting_args)
          
          sys_inputs[f'{model.type}-{cond_name}-metrics-plot'] = metrics_img
          sys_inputs[f'{model.type}-{cond_name}-phase-map-img'] = predict_img
          sys_inputs[f'{model.type}-{cond_name}-oor-img'] = oor_img
          sys_inputs[f'{model.type}-{cond_name}-best-epoch'] = str(best_epoch)
        
        # Prevent RAM Leak
        plt.close(metrics_fig)
        plt.close(oor_fig)

  # ==========================================
  # 4. AGGREGATE RESULTS & COMPILE REPORT
  # ==========================================
  LOGGER.info("\nAggregating Statistical Results...")
  for m_key in models_train.keys():
    for cond_name in [c['name'] for c in conditions]:
      f1_array = results[m_key][cond_name]
      mean_f1 = np.mean(f1_array)
      std_f1 = np.std(f1_array)
      
      # Send the text string "0.912 ± 0.015" to Typst
      formatted_stat = f"{mean_f1:.3f} ± {std_f1:.3f}"
      sys_inputs[f'{m_key}-{cond_name}-ood-f1'] = formatted_stat
      LOGGER.info(f"{m_key} ({cond_name}) OOD F1: {formatted_stat}")

  LOGGER.info("Compiling Master Typst Report...")
  typst.compile(
    input='evaluate_models_report_template.typ',
    output=f'{parent_folder}/final_evaluation_report.pdf',
    root='..',
    sys_inputs=sys_inputs
  )
  LOGGER.info(f"Done! Report saved to: {parent_folder}")

if __name__ == '__main__':
    main()