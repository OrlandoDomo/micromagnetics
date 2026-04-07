import torch
import os
import torch.optim as optim
import polars as pl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from logger import get_logger

from .models import (
  DenseNetwork_BatchNorm,
  DenseNetwork_DropOut
)
from config_reader import config_ml

LOGGER = get_logger(config_ml['training_log'])
LOGGER.info('Logging timestamps are respect to America/Lima timezone')

TOLERANCE = config_ml['sk_tolerance']
THRESHOLD = config_ml['bc_threshold']

class PhaseDataset(Dataset):
  def __init__(self, features, labels, jitter_std=0.01, augment=True, scaler=None, fit_scaler=False):
    self.raw_data = features.astype(np.float32)
    self.labels = labels.astype(np.int64)
    self.jitter_std = jitter_std
    self.augment = augment
    
    self.Aexchange = 1e-11
    self.scaler = scaler
      
    if fit_scaler:
      self.scaler = self._fit_interal_scaler()

  def _engineer_features(self, raw_params):
    D, Ms_raw, DMI_raw, Ku_raw = raw_params
    eps = 1e-10
    mu0 = 4 * np.pi * 1e-7

    DMI = DMI_raw*1e-3
    Ms = Ms_raw*1e3
    Ku = Ku_raw*1e6

    Q = (2*Ku) / (mu0 * Ms**2 + eps)
    kappa = (np.pi * DMI)/(4*np.sqrt(self.Aexchange * Ku + eps))
    dmi = DMI/(np.sqrt(self.Aexchange * Ku + eps))
    lex = np.sqrt(self.Aexchange/(Ku+eps))

    return np.array([D, Ms_raw, DMI_raw, Ku_raw, Q, kappa, dmi, lex])
  
  def _fit_interal_scaler(self):
    all_engineered = np.array([self._engineer_features(row) for row in self.raw_data])
    scaler = StandardScaler()
    scaler.fit(all_engineered)
    return scaler

  def __len__(self):
    return len(self.raw_data)
  
  def __getitem__(self, idx):
    raw = self.raw_data[idx].copy()
    label = torch.tensor(self.labels[idx], dtype=torch.long)
    
    if self.augment:
      noise = np.random.normal(0, self.jitter_std, raw.shape) * raw
      raw += noise

    eng_feat = self._engineer_features(raw)

    if self.scaler:
      eng_feat = self.scaler.transform(eng_feat.reshape(1,-1)).flatten()

    return torch.tensor(eng_feat, dtype=torch.float32), label
  
def visualize_metrics(model, history, device, val_loader):
  model.eval()
  all_preds = []
  all_true = []

  with torch.no_grad():
    for inputs, labels in val_loader:
      logits = model(inputs.to(device))
      preds = (torch.sigmoid(logits) > THRESHOLD).float().cpu().numpy()
      
      all_preds.extend(preds.flatten())
      all_true.extend(labels.numpy())

  plt.figure(figsize=(10, 5))
    
  # Plotting the Loss
  plt.subplot(1, 2, 1)
  plt.plot(history['train_loss'], 
    color='tab:red', 
    label='Training Loss'
  )
  plt.plot(history['val_loss'], 
    color='tab:red',
    linestyle='--', 
    linewidth=2, 
    label='Val Loss'
  )
  plt.title(f'Binary Cross-Entropy Loss for {model.name}')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.grid(True, which='both', linestyle='--', alpha=0.5)
  plt.legend()

  f1 = f1_score(all_true, all_preds)
  plt.subplot(1, 2, 2)
  sns.heatmap(confusion_matrix(all_true, all_preds), annot=True, fmt='d', cmap='Blues')
  plt.title(f"Confusion Matrix\nF1: {f1:.3f}")
  plt.ylabel('Actual')
  plt.xlabel('Predicted') 
  
  plt.tight_layout()
  #plt.show()
  plt.savefig(f"../data/{model.type}-bs_64.png")


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, pos_weight=None, patience=10):
  model = model.to(device)
  criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  optimizer = optim.Adam(model.parameters(), lr=lr)
   
  best_val_loss = float('inf')
  best_model_state = None
  epochs_without_improvement = 0
  best_epoch = 0

  history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': []
  }
  
  for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_data in train_loader:
      inputs, labels = batch_data
      inputs = inputs.to(device)
      labels = labels.to(device).float().unsqueeze(1)
      optimizer.zero_grad()

      logits = model(inputs)
      loss = criterion(logits, labels)  
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

      predicted = (torch.sigmoid(logits) > THRESHOLD).float()
      train_total += labels.size(0)
      train_correct += (predicted == labels).sum().item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
      for batch_data in val_loader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        logits = model(inputs)
        loss = criterion(logits, labels)
        val_loss += loss.item()
        
        predicted = (torch.sigmoid(logits) > THRESHOLD).float()
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    
    # Calculate Metrics
    epoch_loss = train_loss / len(train_loader)
    epoch_val_loss = val_loss / len(train_loader)

    epoch_f1 = f1_score(all_labels, all_preds)
    epoch_acc = (np.array(all_preds).flatten() == np.array(all_labels)).mean()

    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    LOGGER.info(f'Epoch {epoch+1}/{epochs}:')
    LOGGER.info(f'  Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%')
    LOGGER.info(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    LOGGER.info(f'  F1: {epoch_f1:.4f}')
    
    # Save History
    history['train_loss'].append(epoch_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_f1'].append(epoch_f1)
    history['val_acc'].append(epoch_acc)
    
    # Track best model and early stopping
    current_val_loss = val_loss / len(val_loader)
    if current_val_loss < best_val_loss:
      best_val_loss = current_val_loss
      best_model_state = model.state_dict().copy()
      best_epoch = epoch + 1
      epochs_without_improvement = 0
      LOGGER.info(f'\t New best model! (Val Loss: {best_val_loss:.4f})')
    else:
      epochs_without_improvement += 1
      LOGGER.info(f'\t No improvement for {epochs_without_improvement} epoch(s)')
    
    # Early stopping check
    if epochs_without_improvement >= patience:
      LOGGER.info(f'==Early stopping triggered after {epoch+1} epochs==')
      LOGGER.info(f' Best model was at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}')
      break
  
  visualize_metrics(model, history, device, val_loader)

  # Load best model weights before returning
  if best_model_state is not None:
    model.load_state_dict(best_model_state)
    LOGGER.info(f'Loaded best model from epoch {best_epoch}')
  
  return model, best_val_loss, history

def main(csv_path, model_name, epochs, batch_size, lr, patience):

  df = pl.read_csv(csv_path).with_columns([
    pl.when(abs(pl.col("S2k_bot") - 1) < 0.3)
      .then(1)
      .otherwise(0)
      .alias("Sk")
    #pl.format(
    #  "../images/saf_results_relax/bottomlayer_D={}_Ms={}_T=0_dmi={}_Ku={}.png",
    #  pl.col("D").cast(pl.Int64),
    #  pl.col("Ms").cast(pl.Int64),
    #  pl.col("DMI").cast(pl.Float32),
    #  pl.col("Ku").cast(pl.Float32)
    #).alias("image")
  ])
  
  LOGGER.info(f"Loaded {len(df)} samples from {csv_path}")
  LOGGER.info(f"Created Sk labels based on: |S2k_bot - 1| < {TOLERANCE}")
  LOGGER.info(f"Class distribution: {df['Sk'].value_counts().to_dict()}")
  LOGGER.info(f"  Sk=1: {(df['Sk']==1).sum()} samples ({100*(df['Sk']==1).sum()/len(df):.1f}%)")
  LOGGER.info(f"  Sk=0: {(df['Sk']==0).sum()} samples ({100*(df['Sk']==0).sum()/len(df):.1f}%)")
  
  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_labels = df.select('Sk').to_numpy().flatten()

  # Split data
  X_train, X_val, y_train, y_val = train_test_split(X_raw, Y_labels, test_size=0.2, random_state=42, stratify=Y_labels)

  # Determine weights
  num_pos = np.sum(y_train == 1)
  num_neg = np.sum(y_train == 0)
  pos_weight_val = torch.tensor([num_neg / num_pos], dtype=torch.float32)
  
  # Create datasets
  train_dataset = PhaseDataset(X_train, y_train, augment=True, fit_scaler=True)
  val_dataset = PhaseDataset(X_val, y_val, augment=False, scaler=train_dataset.scaler)
  
  # Create dataloaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  LOGGER.info(f"Using device: {device}")

  pos_weight_val = pos_weight_val.to(device)
  
  if model_name == 'default':
    model = DenseNetwork_DropOut(n_features=8)
  elif model_name == 'batchnorm':
    model = DenseNetwork_BatchNorm(n_features=8)
  else:
    raise ValueError(f"Unknown model type: {model}")

  LOGGER.info(f"Training {model.name} model...")
  
  # Train
  model, best_loss, history = train_model(
    model, train_loader, val_loader, device,
    epochs=epochs, lr=lr, pos_weight=pos_weight_val,
    patience=patience
  )

  # Save model
  os.makedirs('ml/saved_models', exist_ok=True)
  save_path = f'ml/saved_models/{model.name}-bs_{batch_size}.pt'
  
  torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': train_dataset.scaler,
    'model_type': model.type
  }, save_path)
  
  LOGGER.info(f"Model saved to {save_path}")

if __name__ == '__main__':
  
  args = {
    'csv_path': "../data/csv_data/saf_relax-results.csv",
    'model_name': config_ml['model'],
    'epochs': config_ml['epochs'],
    'batch_size': config_ml['batch_size'],
    'lr': 0.001,
    'patience': 10
  }
  
  main(**args)