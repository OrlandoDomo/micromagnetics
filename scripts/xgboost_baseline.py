import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


def plot_xgboost_metrics(y_true, y_pred, dataset_name="Evaluation", save_path=None):
  """Generates scatter and residual plots matching your Typst report style."""
  sns.set_theme(style="whitegrid")
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  fig.suptitle(f'XGBoost Regression Evaluation: {dataset_name}', fontsize=14, fontweight='bold')

  # Metrics
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_true, y_pred)

  # 1. Actual vs Predicted Scatter
  axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='none', color='#2b5c8f', s=35)
  min_val = min(np.nanmin(y_true), np.nanmin(y_pred))
  max_val = max(np.nanmax(y_true), np.nanmax(y_pred))
  axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Ideal Fit (y = x)')
  axes[0].set_title('Actual vs. Predicted Values', fontsize=11)
  axes[0].set_xlabel('Ground Truth ($S_k$)', fontsize=10)
  axes[0].set_ylabel('Predicted ($S_k$)', fontsize=10)
  axes[0].legend(loc='upper left', frameon=True)

  # 2. Residual Distribution
  residuals = y_true - y_pred
  sns.histplot(residuals, kde=True, ax=axes[1], color='#2b5c8f', stat='density')
  axes[1].set_title('Residuals Error Distribution ($y - \hat{y}$)', fontsize=11)
  axes[1].set_xlabel('Prediction Error', fontsize=10)

  # Metric box overlay
  metrics_text = f"MAE:  {mae:.4f}\nMSE:  {mse:.4f}\nRMSE: {rmse:.4f}\nR²:    {r2:.4f}"
  props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc')
  axes[1].text(0.95, 0.95, metrics_text, transform=axes[1].transAxes, fontsize=10,
                family='monospace', verticalalignment='top', horizontalalignment='right', bbox=props)

  plt.tight_layout()

  if save_path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

  return mae, mse, rmse, r2


def run_xgboost_experiment(
  csv_path="../data/csv_data/saf_relax-results.csv",
  csv_path_eval="../data/csv_data/saf_relax-dmi=0.6-8_ku=0.08.csv",
  target_col='S2k_bot',
  feature_cols=['D', 'Ms', 'DMI', 'Ku']
):
  print("=== Starting XGBoost Baseline Experiment ===")
  
  # 1. Load Training Data
  df_train = pl.read_csv(csv_path).with_columns(
    pl.col(target_col).round(3).alias(target_col)
  )
  X_raw = df_train.select(feature_cols).to_numpy()
  y_raw = df_train.select(target_col).to_numpy().flatten()

  X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

  # 2. Initialize and Train XGBoost Regressor
  model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
  )

  print("Training XGBoost Regressor with early stopping...")
  model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
  )
  print(f"Best iteration stopped at epoch: {model.best_iteration}")

  # Output folder setup
  now = dt.now().strftime("%d_%m-%H_%M")
  output_folder = Path(f'../results/training/xgboost-baseline-{now}')
  output_folder.mkdir(parents=True, exist_ok=True)

  # 3. In-Distribution Validation Evaluation
  y_val_pred = model.predict(X_val)
  val_mae, val_mse, val_rmse, val_r2 = plot_xgboost_metrics(
    y_val, y_val_pred,
    dataset_name="Validation (In-Distribution)",
    save_path=output_folder / "xgboost-in-distribution-metrics.png"
  )

  # 4. Out-of-Distribution (Unseen Data) Evaluation
  df_eval = pl.read_csv(csv_path_eval).with_columns(
    pl.col(target_col).round(3).alias(target_col)
  )
  X_eval = df_eval.select(feature_cols).to_numpy()
  y_eval = df_eval.select(target_col).to_numpy().flatten()

  y_eval_pred = model.predict(X_eval)
  eval_mae, eval_mse, eval_rmse, eval_r2 = plot_xgboost_metrics(
    y_eval, y_eval_pred,
    dataset_name="Unseen Data (Out-of-Distribution)",
    save_path=output_folder / "xgboost-unseen-metrics.png"
  )

  # 5. Print Comparison Summary
  print("\n" + "="*50)
  print("           XGBOOST METRICS SUMMARY           ")
  print("="*50)
  print(f"In-Distribution Val  | R²: {val_r2:.4f}  | MAE: {val_mae:.4f}  | RMSE: {val_rmse:.4f}")
  print(f"Unseen Dataset Eval | R²: {eval_r2:.4f}  | MAE: {eval_mae:.4f}  | RMSE: {eval_rmse:.4f}")
  print("="*50)
  print(f"Metrics plots saved in: {output_folder}")

  return model, df_eval.with_columns(pl.Series(f"{target_col}_pred", y_eval_pred))


if __name__ == '__main__':
  run_xgboost_experiment(
    csv_path="../data/csv_data/saf_results_sk.csv",
    csv_path_eval="../data/csv_data/saf_results_sk-validation.csv",
    target_col='Sk_bot'
  )