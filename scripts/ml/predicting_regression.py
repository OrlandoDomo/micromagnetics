import torch
import numpy as np
import seaborn as sns
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .training_regression import PhaseDatasetRegression
from .models import (
  DenseNetwork_BatchNorm,
  DenseNetwork_DropOut
)
from logger import get_logger

LOGGER = get_logger(__name__, "predicting_regression")

def load_and_predict(
  csv_path: Union[str, Path],
  checkpoint_path: Union[str, Path],
  target_col: str = "Sk_bot",
  pred_col: str = "Sk_bot_pred",
  batch_size: int = 64,
  device: Optional[torch.device] = None
) -> pl.DataFrame:
    
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  df = pl.read_csv(csv_path).with_columns(
    pl.col(target_col).round(3).alias("Sk_bot")
  )
  feature_cols = ['D', 'Ms', 'DMI', 'Ku']
  X_raw = df.select(feature_cols).to_numpy()

  if target_col in df.columns:
    y_raw = df.select("Sk_bot").to_numpy().flatten()
  else:
    y_raw = np.zeros(len(df), dtype=np.float32)

  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
  model_type = checkpoint.get('model_type', 'dnn_do')
  saved_scaler = checkpoint['scaler']

  if model_type == 'dnn_do':
    model = DenseNetwork_DropOut(n_features=8)
  elif model_type == 'dnn_batch':
    model = DenseNetwork_BatchNorm(n_features=8)
  else:
    raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  model.eval()

  test_dataset = PhaseDatasetRegression(
    features=X_raw,
    target=y_raw,
    augment=False,         
    scaler=saved_scaler,   
    fit_scaler=False,      
    eng_feat=True          
  )

  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  all_preds = []
  with torch.no_grad():
    for inputs, _ in test_loader:
      inputs = inputs.to(device)
      preds = model(inputs).squeeze(1).cpu().numpy()
      all_preds.extend(preds)

  df_predicted = df.with_columns(pl.Series(pred_col, all_preds))
  return df_predicted

def plot_phase_diagram_comparison(
  df: pl.DataFrame,
  actual_col: str = "Sk_bot",
  pred_col: str = "Sk_bot_pred",
  x_col: str = "Ms",
  y_col: str = "D",
  fixed_params: Optional[Dict[str, float]] = None,
  tol: float = 1e-4,
  axes: Optional[Union[np.ndarray, list, Tuple[plt.Axes, plt.Axes]]] = None,
  cmap: str = "RdBu_r",
  fixed_vmin: float = -1.0,
  fixed_vmax: float = 1.0,
  show_values: bool = True,
  value_fmt: str = '.2f',
  font_size: int = 7,
  figsize: Tuple[int, int] = (12, 5),
  save_path: Optional[Union[str, Path]] = None,
  dpi: int = 300
) -> Union[np.ndarray, list, Tuple[plt.Axes, plt.Axes]]:
    
  if fixed_params is None:
    fixed_params = {"DMI": 0.5, "Ku": 0.08}

  # 1. Filter DataFrame using floating-point tolerance
  filter_exprs = [
    (pl.col(param) - value).abs() < tol
    for param, value in fixed_params.items()
  ]
  filtered_df = df.filter(filter_exprs)

  if len(filtered_df) == 0:
    raise ValueError(
        f"No rows matched slice condition: {fixed_params} with tolerance {tol}. "
        "Verify parameter range or increase `tol`."
    )

  # 2. Pivot filtered grid data into 2D matrices (sorted top-to-bottom for Y-axis)
  pivot_actual = (
    filtered_df.pivot(on=x_col, index=y_col, values=actual_col, aggregate_function="mean")
    .sort(y_col, descending=True)
  )
  pivot_pred = (
    filtered_df.pivot(on=x_col, index=y_col, values=pred_col, aggregate_function="mean")
    .sort(y_col, descending=True)
  )

  # Format axis tick labels
  y_labels = [f"{int(round(float(val)))}" for val in pivot_actual[y_col].to_list()]
  x_labels = [f"{int(round(float(col)))}" for col in pivot_actual.columns[1:]]

  matrix_actual = pivot_actual.select(pivot_actual.columns[1:]).to_numpy()
  matrix_pred = pivot_pred.select(pivot_pred.columns[1:]).to_numpy()

  # 3. Canvas Setup: Create figure if axes are not provided
  if axes is None:
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
  elif len(axes) != 2:
    raise ValueError("The `axes` argument must contain exactly two Matplotlib Axes objects.")

  ax_actual, ax_pred = axes[0], axes[1]
  slice_title = ", ".join([f"{k} = {v}" for k, v in fixed_params.items()])

  # Compute shared color limits and TwoSlopeNorm centering at 0.00
  # vmin = min(np.nanmin(matrix_actual), np.nanmin(matrix_pred))
  # vmax = max(np.nanmax(matrix_actual), np.nanmax(matrix_pred))

  norm = mcolors.TwoSlopeNorm(vmin=fixed_vmin, vcenter=0.0, vmax=fixed_vmax)

  # 4. Plot Actual Heatmap
  sns.heatmap(
    matrix_actual,
    ax=ax_actual,
    xticklabels=x_labels,
    yticklabels=y_labels,
    cmap=cmap,
    norm=norm,
    vmin=fixed_vmin,
    vmax=fixed_vmax,
    annot=show_values,
    fmt=value_fmt,
    annot_kws={"size": font_size},
    cbar_kws={'label': actual_col}
  )
  ax_actual.set_title(f"Actual Phase ({actual_col})\n[{slice_title}]", fontsize=11)
  ax_actual.set_xlabel(x_col, fontsize=10)
  ax_actual.set_ylabel(y_col, fontsize=10)

  # 5. Plot Predicted Heatmap
  sns.heatmap(
    matrix_pred,
    ax=ax_pred,
    xticklabels=x_labels,
    yticklabels=y_labels,
    cmap=cmap,
    norm=norm,
    vmin=fixed_vmin,
    vmax=fixed_vmax,
    annot=show_values,
    fmt=value_fmt,
    annot_kws={"size": font_size},
    cbar_kws={'label': pred_col}
  )
  ax_pred.set_title(f"Predicted Phase ({pred_col})\n[{slice_title}]", fontsize=11)
  ax_pred.set_xlabel(x_col, fontsize=10)

  for ax in [ax_actual, ax_pred]:
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

  plt.tight_layout()
  
  plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
  LOGGER.info(f"Phase diagram comparison saved successfully to: {save_path}")

def plot_dataset_metrics(
  y_true: Union[np.ndarray, torch.Tensor],
  y_pred: Union[np.ndarray, torch.Tensor],
  dataset_name: str = "Unseen Evaluation Data",
  save_path: Optional[Union[str, Path]] = None,
  dpi: int = 300
) -> plt.Figure:

  if isinstance(y_true, torch.Tensor):
    y_true = y_true.detach().cpu().numpy()
  if isinstance(y_pred, torch.Tensor):
    y_pred = y_pred.detach().cpu().numpy()

  sns.set_theme(style="whitegrid")
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  fig.suptitle(f'Regression Evaluation: {dataset_name}', fontsize=14, fontweight='bold')

  # Calculate Regression Metrics
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_true, y_pred)

  # 1. Actual vs. Predicted Scatter
  axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='none', color='#2b5c8f', s=35)
  
  min_val = min(np.nanmin(y_true), np.nanmin(y_pred))
  max_val = max(np.nanmax(y_true), np.nanmax(y_pred))
  axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, label='Ideal Fit (y = x)')

  axes[0].set_title('Actual vs. Predicted Values', fontsize=11)
  axes[0].set_xlabel('Ground Truth ($S_{k}$)', fontsize=10)
  axes[0].set_ylabel('Predicted ($S_{k}$)', fontsize=10)
  axes[0].legend(loc='upper left', frameon=True)

  # 2. Residual Distribution Plot
  residuals = y_true - y_pred
  sns.histplot(residuals, kde=True, ax=axes[1], color='#2b5c8f', stat='density')
  axes[1].set_title('Residuals Error Distribution ($y - \hat{y}$)', fontsize=11)
  axes[1].set_xlabel('Prediction Error', fontsize=10)

  # Metrics Overlay Textbox
  metrics_text = f"MAE:  {mae:.4f}\nMSE:  {mse:.4f}\nRMSE: {rmse:.4f}\nR²:    {r2:.4f}"
  props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc')
  axes[1].text(0.95, 0.95, metrics_text, transform=axes[1].transAxes, fontsize=10,
              family='monospace', verticalalignment='top', horizontalalignment='right', bbox=props)

  plt.tight_layout()

  # Save to disk if path is provided
  if save_path is not None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

  return fig

def main(
  csv_path,
  model_path,
  target_col='Sk_bot',
  plot_metrics=True,
  save_path=None,
  metrics_save_path=None,
  dataset_name=None,
  dmi=0.5,
  ku=0.08,
  fixed_vmin: float = -1.0,
  fixed_vmax: float = 1.0
):
  
  LOGGER.info("Running inference on test dataset...")
  df_result = load_and_predict(
    csv_path=csv_path,
    checkpoint_path=model_path,
    target_col=target_col,
    pred_col="Sk_bot_pred"
  )

  LOGGER.info("Generating comparison phase diagram...")
  plot_phase_diagram_comparison(
    df=df_result,
    actual_col="Sk_bot",
    pred_col="Sk_bot_pred",
    x_col="Ms",
    y_col="D",
    fixed_params={"DMI": dmi, "Ku": ku},
    cmap="PiYG",
    save_path=save_path,
    show_values=True,
    value_fmt=".2f",
    fixed_vmin=fixed_vmin,
    fixed_vmax=fixed_vmax
  )

  if plot_metrics:
    LOGGER.info("Generating comparison metrics plots...")
    y_true_eval = df_result.select("Sk_bot").to_numpy()
    y_pred_eval = df_result.select("Sk_bot_pred").to_numpy()

    plot_dataset_metrics(
      y_true=y_true_eval,
      y_pred=y_pred_eval,
      dataset_name=dataset_name,
      save_path=metrics_save_path,
      dpi=300
    )

if __name__ == '__main__':
  main()