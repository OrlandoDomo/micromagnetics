import torch
import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
                             

from .models import (
  DenseNetwork_BatchNorm,
  DenseNetwork_DropOut
)
from .models import DenseNetwork_DropOut, DenseNetwork_BatchNorm
from .training import PhaseDatasetClassification
from config_reader import config_ml
from logger import get_logger

LOGGER = get_logger(__name__, "predicting")
TOLERANCE = config_ml['sk_tolerance']
THRESHOLD = config_ml['bc_threshold']

PHASE_MAP = {
  0: "Others",
  1: "Skyrmion"
}

def load_and_predict_classification(
  csv_path: Union[str, Path],
  checkpoint_path: Union[str, Path],
  num_classes: int = 2,
  pred_col: str = "phase_label_pred",
  batch_size: int = 64,
  device: Optional[torch.device] = None
) -> pl.DataFrame:
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  df = pl.read_csv(csv_path)
  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels = df.select('phase_label').to_numpy().flatten()

  checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
  model_type = checkpoint.get('model_type', 'dnn_do')
  saved_scaler = checkpoint['scaler']

  num_classes = 1 if num_classes == 2 else num_classes
  print(f'THERE ARE {num_classes} classes in load_and_predict_classification')

  if model_type == 'dnn_do':
    model = DenseNetwork_DropOut(n_features=8, num_classes=num_classes)
  elif model_type == 'dnn_batch':
    model = DenseNetwork_BatchNorm(n_features=8, num_classes=num_classes)
  else:
    raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  model.eval()

  test_dataset = PhaseDatasetClassification(
    features=X_raw,
    labels=Y_labels,
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
      logits = model(inputs)
      
      # Determine predictions based on output dimensions
      if logits.shape[1] == 1:
        # Binary classification
        preds = (torch.sigmoid(logits) > THRESHOLD).long().squeeze(1)
      else:
        # Multi-class classification
        preds = torch.argmax(logits, dim=1)

      all_preds.extend(preds.cpu().numpy())

  df_predicted = df.with_columns(pl.Series(pred_col, all_preds))
  return df_predicted


def plot_phase_diagram_classification(
  df: pl.DataFrame,
  actual_col: str = "phase_label",
  pred_col: str = "phase_label_pred",
  x_col: str = "Ms",
  y_col: str = "D",
  fixed_params: Optional[Dict[str, float]] = None,
  class_names: Optional[Dict[int, str]] = None,
  tol: float = 1e-4,
  axes: Optional[Union[np.ndarray, list, Tuple[plt.Axes, plt.Axes]]] = None,
  cmap_name: str = "tab10",
  show_values: bool = True,
  font_size: int = 8,
  figsize: Tuple[int, int] = (14, 6),
  save_path: Optional[Union[str, Path]] = None,
  dpi: int = 300
) -> Union[np.ndarray, list, Tuple[plt.Axes, plt.Axes]]:
  
  if fixed_params is None:
    fixed_params = {"DMI": 0.5, "Ku": 0.08}

  filter_exprs = [
    (pl.col(param) - value).abs() < tol
    for param, value in fixed_params.items()
  ]
  filtered_df = df.filter(filter_exprs)

  if len(filtered_df) == 0:
    raise ValueError(
      f"No rows matched slice condition: {fixed_params} with tolerance {tol}."
    )

  pivot_actual = (
    filtered_df.pivot(on=x_col, index=y_col, values=actual_col, aggregate_function="first")
    .sort(y_col, descending=True)
  )
  pivot_pred = (
    filtered_df.pivot(on=x_col, index=y_col, values=pred_col, aggregate_function="first")
    .sort(y_col, descending=True)
  )

  y_labels = [f"{int(round(float(val)))}" for val in pivot_actual[y_col].to_list()]
  x_labels = [f"{int(round(float(col)))}" for col in pivot_actual.columns[1:]]

  matrix_actual = pivot_actual.select(pivot_actual.columns[1:]).to_numpy().astype(int)
  matrix_pred = pivot_pred.select(pivot_pred.columns[1:]).to_numpy().astype(int)

  # Handle Discrete Palette & Class Color Boundaries
  unique_classes = np.sort(np.unique(np.concatenate([matrix_actual.flatten(), matrix_pred.flatten()])))
  num_classes = len(unique_classes)

  base_cmap = plt.cm.get_cmap(cmap_name, num_classes)
  
  # Define exact color boundaries for discrete integer steps [min-0.5, ..., max+0.5]
  bounds = np.append(unique_classes - 0.5, unique_classes[-1] + 0.5)
  norm = mcolors.BoundaryNorm(bounds, base_cmap.N)

  if axes is None:
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
  elif len(axes) != 2:
    raise ValueError("The `axes` argument must contain exactly two Matplotlib Axes objects.")

  ax_actual, ax_pred = axes[0], axes[1]
  slice_title = ", ".join([f"{k} = {v}" for k, v in fixed_params.items()])

  # Plot Actual Classification Heatmap
  sns.heatmap(
    matrix_actual,
    ax=ax_actual,
    xticklabels=x_labels,
    yticklabels=y_labels,
    cmap=base_cmap,
    norm=norm,
    annot=show_values,
    fmt="d",
    annot_kws={"size": font_size},
    cbar=False
  )
  ax_actual.set_title(f"Actual Phase ({actual_col})\n[{slice_title}]", fontsize=11)
  ax_actual.set_xlabel(x_col, fontsize=10)
  ax_actual.set_ylabel(y_col, fontsize=10)

  # Plot Predicted Classification Heatmap
  sns.heatmap(
    matrix_pred,
    ax=ax_pred,
    xticklabels=x_labels,
    yticklabels=y_labels,
    cmap=base_cmap,
    norm=norm,
    annot=show_values,
    fmt="d",
    annot_kws={"size": font_size},
    cbar=False
  )
  ax_pred.set_title(f"Predicted Phase ({pred_col})\n[{slice_title}]", fontsize=11)
  ax_pred.set_xlabel(x_col, fontsize=10)

  # Force tick labels horizontal (rotation=0)
  for ax in [ax_actual, ax_pred]:
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

  fig = ax_actual.get_figure()
  sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
  sm.set_array([])

  cbar = fig.colorbar(sm, ax=[ax_actual, ax_pred], orientation='vertical', pad=0.02, aspect=25, ticks=unique_classes)
  cbar.set_label("Phase Class", fontsize=10)

  if class_names is not None:
    tick_labels = [class_names.get(c, str(c)) for c in unique_classes]
    cbar.set_ticklabels(tick_labels)

  if save_path is not None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    LOGGER.info(f"Classification phase diagram saved successfully to: {save_path}")

  return axes

def plot_dataset_metrics(
  y_true: Union[np.ndarray, torch.Tensor],
  y_pred: Union[np.ndarray, torch.Tensor],
  dataset_name: str = "Unseen Evaluation Data",
  save_path: Optional[Union[str, Path]] = None,
  class_names: Optional[Dict[int, str]] = None,
  dpi: int = 300
) -> plt.Figure:

  if isinstance(y_true, torch.Tensor):
    y_true = y_true.detach().cpu().numpy()
  if isinstance(y_pred, torch.Tensor):
    y_pred = y_pred.detach().cpu().numpy()

  sns.set_theme(style="whitegrid")
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  fig.suptitle(f'Classification Metrics: {dataset_name}', fontsize=16)
  
  # 1. Confusion Matrix
  cm = confusion_matrix(y_true, y_pred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
    xticklabels=class_names, yticklabels=class_names)
  axes[0].set_title('Confusion Matrix')
  axes[0].set_xlabel('Predicted Phase')
  axes[0].set_ylabel('True Phase')
  
  # 2. Precision, Recall, F1-Score Bar Chart
  report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
  # Extract metrics, excluding 'accuracy', 'macro avg', 'weighted avg'
  metrics_data = {k: v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
  
  classes = list(metrics_data.keys())
  precision = [metrics_data[c]['precision'] for c in classes]
  recall = [metrics_data[c]['recall'] for c in classes]
  f1 = [metrics_data[c]['f1-score'] for c in classes]
  
  x = np.arange(len(classes))
  width = 0.25
  
  axes[1].bar(x - width, precision, width, label='Precision', color='#4c72b0')
  axes[1].bar(x, recall, width, label='Recall', color='#dd8452')
  axes[1].bar(x + width, f1, width, label='F1-Score', color='#55a868')
  
  axes[1].set_title('Metrics per Class')
  axes[1].set_xticks(x)
  axes[1].set_xticklabels(classes)
  axes[1].set_ylim(0, 1.1)
  axes[1].legend()
  
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
    save_path,
    metrics_save_path=None,
    dataset_name=None,
    plot_metrics=True,
    dmi=0.5,
    ku=0.08,
    class_names=PHASE_MAP
):

  num_classes = len(class_names)

  LOGGER.info("Running classification inference...")
  df_result = load_and_predict_classification(
    csv_path=csv_path,
    checkpoint_path=model_path,
    pred_col="phase_label_pred",
    num_classes=num_classes
  )

  LOGGER.info("Generating classification comparison phase diagram...")
  plot_phase_diagram_classification(
    df=df_result,
    actual_col="phase_label",
    pred_col="phase_label_pred",
    x_col="Ms",
    y_col="D",
    fixed_params={"DMI": dmi, "Ku": ku},
    class_names=class_names,
    cmap_name="Accent",
    show_values=True,
    save_path=save_path,
    dpi=300
  )

  if plot_metrics:
    LOGGER.info("Generating comparison metrics plots...")
    y_true_eval = df_result.select("phase_label").to_numpy()
    y_pred_eval = df_result.select("phase_label_pred").to_numpy()

    plot_dataset_metrics(
      y_true=y_true_eval,
      y_pred=y_pred_eval,
      dataset_name=dataset_name,
      save_path=metrics_save_path,
      class_names=class_names,
      dpi=300
    )

if __name__ == '__main__':
  main()