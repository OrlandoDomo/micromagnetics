import torch
import copy
import torch.optim as optim
import polars as pl
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from logger import get_logger
from pathlib import Path
from datetime import datetime as dt

from .models import (
  DenseNetwork_BatchNorm,
  DenseNetwork_DropOut
)
from config_reader import config_ml

LOGGER = get_logger(__name__, config_ml['training_log'])
LOGGER.info('Logging timestamps are respect to America/Lima timezone')

class PhaseDatasetRegression(Dataset):
  def __init__(self, features, target, jitter_std=0.01, augment=True, scaler=None, fit_scaler=False, eng_feat=True):
    self.raw_data = features.astype(np.float32)
    self.target = target
    self.jitter_std = jitter_std
    self.augment = augment
    self.eng_feat = eng_feat
    
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
    if self.eng_feat:
      all_feats = np.array([self._engineer_features(row) for row in self.raw_data])
    else:
      all_feats = self.raw_data
      
    scaler = StandardScaler()
    scaler.fit(all_feats)
    return scaler

  def __len__(self):
    return len(self.raw_data)
  
  def __getitem__(self, idx):
    raw = self.raw_data[idx].copy()
    target = torch.tensor(self.target[idx], dtype=torch.long)
    
    if self.augment:
      noise = np.random.normal(0, self.jitter_std, raw.shape) * raw
      raw += noise

    if self.eng_feat:
      feat = self._engineer_features(raw)
    else:
      feat = raw

    if self.scaler:
      feat = self.scaler.transform(feat.reshape(1,-1)).flatten()

    return torch.tensor(feat, dtype=torch.float32), target
  
def visualize_metrics(model, stats, device, val_loader):
  
  model.eval()
  all_preds = []
  all_targets = []

  with torch.no_grad():
    for batch_data in val_loader:
      inputs, target = batch_data
      inputs = inputs.to(device)
      target = target.to(device).float().unsqueeze(1)

      predicted = model(inputs)

      all_preds.append(predicted)
      all_targets.append(target)

  y_true = torch.cat(all_targets).cpu().numpy()
  y_pred = torch.cat(all_preds).cpu().numpy()
  r2 = r2_score(y_true, y_pred)    

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))    
  # Plotting the Loss
  ax1.plot(stats['train_mse'], label='Train MSE')
  ax1.plot(stats['val_mse'], label='Val MSE', ls='--')
  ax1.set_title('MSE Loss')
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('MSE')
  ax1.legend()

  ax2.plot(stats['val_mae'], label='Val MAE', color='green')
  ax2.set_title('MAE Loss')
  ax2.set_xlabel('Epochs')
  ax2.set_ylabel('MAE')
  ax2.legend()

  fig.suptitle(fr"$R^2$ Score: {r2}")
  
  fig.tight_layout()
  #plt.show()
  #fig.savefig(f"../data/{model.type}-regression-bs_64.png")
  return fig
  
def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, patience=10):
  model = model.to(device)
  mse_criterion = torch.nn.MSELoss()
  mae_criterion = torch.nn.L1Loss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
   
  best_loss = float('inf')
  best_model_state = None
  epochs_without_improvement = 0
  best_epoch = 0

  stats = {
        'train_mse': [],
        'val_mse': [],
        'val_mae': [],
        'val_r2': []
  }
  
  for epoch in range(epochs):
    # Training
    model.train()
    running_train_loss = 0.0
    for batch_data in train_loader:
      inputs, target = batch_data
      inputs = inputs.to(device)
      target = target.to(device).float().unsqueeze(1)
      
      optimizer.zero_grad()
      predicted = model(inputs)
      loss = mse_criterion(predicted, target)
      loss.backward()
      optimizer.step()
      
      running_train_loss += loss.item()
      
    # Validation
    model.eval()
    total_mse_loss = 0.0
    total_mae_loss = 0.0
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
      for batch_data in val_loader:
        inputs, target = batch_data
        inputs = inputs.to(device)
        target = target.to(device).float().unsqueeze(1)

        predicted = model(inputs)
        mse_loss = mse_criterion(predicted, target)
        mae_loss = mae_criterion(predicted, target)
        
        total_mse_loss += mse_loss.item()
        total_mae_loss += mae_loss.item()

        all_preds.append(predicted)
        all_targets.append(target)
    
    # Calculate Metrics
    epoch_loss = running_train_loss / len(train_loader)
    epoch_mse_loss = total_mse_loss / len(val_loader)
    epoch_mae_loss = total_mae_loss / len(val_loader)

    y_true = torch.cat(all_targets).cpu().numpy()
    y_pred = torch.cat(all_preds).cpu().numpy()
    epoch_r2 = r2_score(y_true, y_pred)

    LOGGER.info(f'Epoch {epoch+1}/{epochs}:')
    LOGGER.info(f'  Train Loss: {epoch_loss:.4f} | MSE Loss: {epoch_mse_loss:.2f} | MAE Loss: {epoch_mae_loss:.2f}')
    
    # Save stats
    stats['train_mse'].append(epoch_loss)
    stats['val_mse'].append(epoch_mse_loss)
    stats['val_mae'].append(epoch_mae_loss)
    stats['val_r2'].append(epoch_r2)
    
    # Track best model and early stopping
    if epoch_mse_loss < best_loss:
      best_loss = epoch_mse_loss
      best_model_state = copy.deepcopy(model.state_dict())
      best_epoch = epoch + 1
      epochs_without_improvement = 0
      LOGGER.info(f'\t New best model! (MSE Loss: {best_loss:.4f})')
    else:
      epochs_without_improvement += 1
      LOGGER.info(f'\t No improvement for {epochs_without_improvement} epoch(s)')
    
    # Early stopping check
    if epochs_without_improvement >= patience:
      LOGGER.info(f'==Early stopping triggered after {epoch+1} epochs==')
      LOGGER.info(f' Best model was at epoch {best_epoch} with Val Loss: {best_loss:.4f}')
      break
  
  # Load best model weights before returning
  if best_model_state is not None:
    model.load_state_dict(best_model_state)
    LOGGER.info(f'Loaded best model from epoch {best_epoch}')
  
  fig = visualize_metrics(model, stats, device, val_loader)
  return model, best_epoch, stats, fig

def main(csv_path, model_name, epochs, batch_size, lr, patience):

  df = pl.read_csv(csv_path)
  
  X_raw = df.select(['D', 'Ms', 'DMI', 'Ku']).to_numpy()
  Y_target = df.select('S2k_bot').to_numpy().flatten()

  # Split data
  try:
    X_train, X_val, y_train, y_val = train_test_split(X_raw, Y_target, test_size=0.2, random_state=42)
  except Exception as e:
    LOGGER.error(f'Error :\n[{e}]')
    return 0

  # Create datasets
  train_dataset = PhaseDatasetRegression(X_train, y_train, augment=True, fit_scaler=True)
  val_dataset = PhaseDatasetRegression(X_val, y_val, augment=False, scaler=train_dataset.scaler)
  
  # Create dataloaders
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  LOGGER.info(f"Using device: {device}")
  
  if model_name == 'default':
    model = DenseNetwork_DropOut(n_features=8)
  elif model_name == 'batchnorm':
    model = DenseNetwork_BatchNorm(n_features=8)
  else:
    raise ValueError(f"Unknown model type: {model_name}")

  LOGGER.info(f"Training {model.name} model...")
  
  # Train
  model, best_epoch, stats, fig = train_model(
    model, train_loader, val_loader, device,
    epochs=epochs, lr=lr,
    patience=patience
  )

  now = dt.now().strftime("%d_%m-%H_%M")
  parent_folder = f'../results/training/train-regression-{now}'
  Path(parent_folder).mkdir(parents=True, exist_ok=True)
  # Save model
  #os.makedirs('ml/saved_models', exist_ok=True)
  save_path = f'{parent_folder}/{model.name}-regression.pt'
  metrics_img = f'{parent_folder}/{model.name}-metrics.png'
  fig.savefig(metrics_img, format='png')
  
  torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': train_dataset.scaler,
    'model_type': model.type
  }, save_path)
  
  LOGGER.info(f"Model saved to {save_path}")

  sys_inputs = {
    'lr': str(lr),
    'batch-size': str(batch_size),
    'epochs': str(epochs),
    'best-epoch': str(best_epoch),
    'metrics-plot': metrics_img,
    'model-save-path': save_path
  }

  #typst.compile(
  #  input='report_template.typ',
  #  output=f'{parent_folder}/report.pdf',
  #  root='..',
  #  sys_inputs=sys_inputs
  #)

  return sys_inputs, parent_folder

if __name__ == '__main__':
  
  args = {
    'csv_path': "../data/csv_data/saf_relax-results.csv",
    'model_name': config_ml['model'],
    'epochs': config_ml['epochs'],
    'batch_size': config_ml['batch_size'],
    'lr': 0.001,
    'patience': 50
  }
  
  main(**args)