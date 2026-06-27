import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from sklearn.metrics import (
  accuracy_score, precision_score, recall_score,
  f1_score, roc_curve, auc, precision_recall_curve,
  confusion_matrix, ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader
from matplotlib.colors import ListedColormap
from .models import (
  DenseNetwork_BatchNorm,
  DenseNetwork_DropOut
)
from .predicting import load_model, predict_single
from .training import PhaseDatasetClassification
from config_reader import config_ml

TOLERANCE = config_ml['sk_tolerance']
THRESHOLD = config_ml['bc_threshold']

def compare_new_data(model, val_loader, device, fig_path):
    
  plt.figure(figsize=(12, 8))

  model.eval()
  all_probs = []
  all_labels = []

  with torch.no_grad():
    for X_batch, y_batch in val_loader:
      X_batch = X_batch.to(device)
      y_batch = y_batch.to(device)
      
      outputs = model(X_batch)
      probs = torch.sigmoid(outputs).view(-1)  # shape [batch_size]
      
      all_probs.append(probs.cpu())
      all_labels.append(y_batch.cpu())

  all_probs = torch.cat(all_probs)
  all_labels = torch.cat(all_labels)
  
  # Convert probabilities to binary predictions
  all_preds = (all_probs >= THRESHOLD).long()
  
  # Metrics
  acc = accuracy_score(all_labels, all_preds)
  prec = precision_score(all_labels, all_preds)
  rec = recall_score(all_labels, all_preds)
  f1 = f1_score(all_labels, all_preds)
  
  # Create figure with 1 row, 3 columns
  fig, axes = plt.subplots(1, 3, figsize=(18,5))
  
  # Compute and plot confusion matrix
  cm = confusion_matrix(all_labels, all_preds)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
  axes[0].set_title(f"Confusion Matrix\nF1 Score: {f1:.4f}")
  
  # ROC curve
  fpr, tpr, _ = roc_curve(all_labels, all_probs)
  roc_auc = auc(fpr, tpr)
  axes[1].plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
  axes[1].plot([0,1],[0,1],'--', color='gray')
  axes[1].set_xlabel("False Positive Rate")
  axes[1].set_ylabel("True Positive Rate")
  axes[1].set_title("ROC Curve")
  axes[1].legend(loc="lower right")
  
  # Precision-Recall curve
  precision, recall, _ = precision_recall_curve(all_labels, all_probs)
  axes[2].plot(recall, precision, color='green')
  axes[2].set_xlabel("Recall")
  axes[2].set_ylabel("Precision")
  axes[2].set_title("Precision-Recall Curve")
  
  fig.suptitle(f"Metrics for {model.name}\nAccuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}", fontsize=16, y=1.05)
  
  plt.tight_layout()
  if fig_path:    
    plt.savefig(fig_path, dpi=300)
  else:
    return fig
  #plt.show()
  
def plot_error_regions_raw_features(
  dataloader, 
  model, 
  scaler,                 # your scaler object with inverse_transform()
  feature_cols_indices,   # indices of features to plot (in scaled tensor)
  device='cpu', 
  feature_names=None
):
  model.eval()
  model.to(device)

  all_inputs_scaled = []
  all_labels = []
  all_preds = []

  with torch.no_grad():
    for batch_inputs, batch_labels in dataloader:
      batch_inputs = batch_inputs.to(device)
      batch_labels = batch_labels.to(device)

      outputs = model(batch_inputs)
      probs = torch.sigmoid(outputs).squeeze()
      preds = (probs > 0.5).long()

      all_inputs_scaled.append(batch_inputs.cpu().numpy())
      all_labels.append(batch_labels.cpu().numpy())
      all_preds.append(preds.cpu().numpy())

  inputs_scaled = np.concatenate(all_inputs_scaled, axis=0)
  labels = np.concatenate(all_labels, axis=0)
  preds = np.concatenate(all_preds, axis=0)

  # Inverse scale all features
  inputs_raw = scaler.inverse_transform(inputs_scaled)

  # Extract the two features of interest (raw values)
  x = inputs_raw[:, feature_cols_indices[0]]
  y = inputs_raw[:, feature_cols_indices[1]]

  correct = (labels == preds)

  plt.figure(figsize=(8,6))
  plt.scatter(x[correct], y[correct], c='green', label='Correct', alpha=0.6, s=40)
  plt.scatter(x[~correct], y[~correct], c='red', label='Errors', alpha=0.8, s=50, edgecolors='k')

  plt.xlabel(feature_names[0] if feature_names else f"Feature {feature_cols_indices[0]} (raw)")
  plt.ylabel(feature_names[1] if feature_names else f"Feature {feature_cols_indices[1]} (raw)")
  plt.title("Model Predictions: Correct vs Errors (Raw Features)")
  plt.legend()
  plt.grid(True)
  plt.show()

def main(model_path, csv_path, batch_size, fig_path):
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  
  # Load model
  print(f"Loading model from {model_path}...")
  model, scaler, model_type = load_model(model_path, device)
  model = model.to(device)
  print(f"Model type: {model_type}")

  df = pl.read_csv(csv_path).with_columns([
    pl.when(abs(pl.col("S2k_bot") - 1) < TOLERANCE)
      .then(1)
      .otherwise(0)
      .alias("Sk")
  ])
  
  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels = df.select('Sk').to_numpy().flatten()
  
  val_dataset = PhaseDatasetClassification(X_raw, Y_labels, augment=False, scaler=scaler)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
  compare_new_data(
    model=model,
    val_loader=val_loader,
    device=device,
    fig_path=fig_path
  )
  
  plot_error_regions_raw_features(
    dataloader=val_loader,
    model=model,
    scaler=scaler,
    feature_cols_indices=[1,0],
    feature_names=['Ms','D'],
    device='cuda'
  )
    
  return 1
    
if __name__ == '__main__':
  
  main(
    model_path='ml/saved_models/DenseNN-BatchNorm_model_bs-64.pt',
    csv_path='../data/csv_data/saf_relax-hi_res.csv',
    batch_size=64,
    fig_path='../data/comparing_hi_res-batchnorm-tol_025.png'
  )