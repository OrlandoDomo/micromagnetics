import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os
from models import (
  Option1_2DCNN_Separate_Dynamic,
  Option2_2DCNN_4Channels_Dynamic,
  Option3_3DCNN_Dynamic,
  DenseNetwork
)

# Global tolerance for Sk label creation
TOLERANCE = 0.3  # Sk = 1 if |S2k_top - 1| < TOLERANCE


class PhaseDataset(Dataset):
  """Dataset for phase diagram data"""
  def __init__(self, df, model_type='option1', scaler=None, fit_scaler=False):
    self.df = df.copy()
    self.model_type = model_type
    
    # Standardize continuous features
    if scaler is None:
      self.scaler = StandardScaler()
    else:
      self.scaler = scaler
      
    if fit_scaler:
      self.df[['D', 'Ms', 'DMI', 'K']] = self.scaler.fit_transform(
        self.df[['D', 'Ms', 'DMI', 'K']]
      )
    else:
      self.df[['D', 'Ms', 'DMI', 'K']] = self.scaler.transform(
        self.df[['D', 'Ms', 'DMI', 'K']]
      )
    
    # Define actual parameter ranges
    # D: 150 to 825 in steps of 75 -> 10 values
    # Ms: 260 to 460 in steps of 20 -> 11 values (but we'll use 10)
    # K: 0.02 to 0.20 in steps of 0.02 -> 10 values
    # DMI: 3 values (0.5, 1.0, 1.5)
    
    self.D_values = list(range(150, 826, 75))  # [150, 225, ..., 825]
    self.Ms_values = list(range(260, 461, 20))  # [260, 280, ..., 460]
    self.K_values = [round(0.02 * i, 2) for i in range(1, 11)]  # [0.02, 0.04, ..., 0.20]
    self.DMI_values = [0.5, 1.0, 1.5]
    
    # Get actual unique values from data
    self.D_unique = sorted(self.df['D'].unique())
    self.Ms_unique = sorted(self.df['Ms'].unique())
    self.DMI_unique = sorted(self.df['DMI'].unique())
    self.K_unique = sorted(self.df['K'].unique())
    
    # Create mappings to grid indices
    self.D_to_idx = {v: i for i, v in enumerate(self.D_unique)}
    self.Ms_to_idx = {v: i for i, v in enumerate(self.Ms_unique)}
    self.DMI_to_idx = {v: i for i, v in enumerate(self.DMI_unique)}
    self.K_to_idx = {v: i for i, v in enumerate(self.K_unique)}
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    label = torch.tensor(row['Sk'], dtype=torch.float32)
    
    if self.model_type == 'option1':
      # Create 10x11 phase diagram + separate params
      d_idx = self.D_to_idx[row['D']]
      ms_idx = self.Ms_to_idx[row['Ms']]
      
      # Create phase diagram with actual grid size
      n_d = len(self.D_unique)
      n_ms = len(self.Ms_unique)
      phase_diagram = torch.zeros((1, n_d, n_ms))
      phase_diagram[0, d_idx, ms_idx] = 1.0
      
      params = torch.tensor([row['DMI'], row['K']], dtype=torch.float32)
      return (phase_diagram, params), label
      
    elif self.model_type == 'option2':
      # 4-channel 2D
      d_idx = self.D_to_idx[row['D']]
      ms_idx = self.Ms_to_idx[row['Ms']]
      
      n_d = len(self.D_unique)
      n_ms = len(self.Ms_unique)
      x = torch.zeros((4, n_d, n_ms))
      x[0, d_idx, ms_idx] = row['D']
      x[1, d_idx, ms_idx] = row['Ms']
      x[2, d_idx, ms_idx] = row['DMI']
      x[3, d_idx, ms_idx] = row['K']
      return x, label
      
    elif self.model_type == 'option3':
      # 3D volume
      d_idx = self.D_to_idx[row['D']]
      ms_idx = self.Ms_to_idx[row['Ms']]
      dmi_idx = self.DMI_to_idx[row['DMI']]
      k_idx = self.K_to_idx[row['K']]
      
      n_d = len(self.D_unique)
      n_ms = len(self.Ms_unique)
      n_dmi = len(self.DMI_unique)
      n_k = len(self.K_unique)
      
      x = torch.zeros((1, n_d, n_ms, n_dmi * n_k))  # Flatten DMI×K dimension
      x[0, d_idx, ms_idx, dmi_idx * n_k + k_idx] = 1.0
      return x, label
      
    else:  # dense
      # Feature engineering
      features = [
        row['D'], row['Ms'], row['DMI'], row['K'],
        row['D'] * row['Ms'],
        row['D'] * row['DMI'],
        row['Ms'] * row['K'],
        row['DMI'] * row['K'],
        row['D']**2,
        row['Ms']**2
      ]
      return torch.tensor(features, dtype=torch.float32), label


def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, class_weights=None, patience=10):
  """
  Train the model with early stopping
  
  Args:
    model: PyTorch model
    train_loader: Training data loader
    val_loader: Validation data loader
    device: torch device
    epochs: Maximum number of epochs
    lr: Learning rate
    class_weights: Dictionary of class weights for imbalanced data
    patience: Number of epochs to wait for improvement before stopping
  """
  criterion = nn.BCELoss()
  
  optimizer = optim.Adam(model.parameters(), lr=lr)
  
  # Apply class weights if provided
  if class_weights is not None:
    # Create weighted loss
    pos_weight = class_weights[1] / class_weights[0]
    print(f"Using class weights - positive class weight: {pos_weight:.2f}")
  
  best_val_loss = float('inf')
  best_model_state = None
  epochs_without_improvement = 0
  best_epoch = 0
  
  for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_data in train_loader:
      inputs, labels = batch_data
      
      # Handle tuple inputs (Option1) vs tensor inputs (Option2, Option3, Dense)
      if isinstance(inputs, (tuple, list)):  # Option1
        inputs = tuple(inp.to(device) for inp in inputs)
      else:
        inputs = inputs.to(device)
      labels = labels.to(device).unsqueeze(1)
      
      optimizer.zero_grad()
      
      # Forward pass
      if isinstance(inputs, (tuple, list)):
        outputs = model(*inputs)
      else:
        outputs = model(inputs)
      
      # Apply class weighting manually
      if class_weights is not None:
        # Create weight tensor for each sample based on its label
        weights = torch.ones_like(labels)
        weights[labels == 1] = class_weights[1] / class_weights[0]
        loss = (criterion(outputs, labels) * weights).mean()
      else:
        loss = criterion(outputs, labels)
        
      loss.backward()
      optimizer.step()
      
      train_loss += loss.item()
      predicted = (outputs > 0.5).float()
      train_total += labels.size(0)
      train_correct += (predicted == labels).sum().item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
      for batch_data in val_loader:
        inputs, labels = batch_data
        
        # Handle tuple inputs (Option1) vs tensor inputs (Option2, Option3, Dense)
        if isinstance(inputs, (tuple, list)):
          inputs = tuple(inp.to(device) for inp in inputs)
        else:
          inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        # Forward pass
        if isinstance(inputs, (tuple, list)):
          outputs = model(*inputs)
        else:
          outputs = model(inputs)
          
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        
        predicted = (outputs > 0.5).float()
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    
    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
    
    # Track best model and early stopping
    current_val_loss = val_loss / len(val_loader)
    if current_val_loss < best_val_loss:
      best_val_loss = current_val_loss
      best_model_state = model.state_dict().copy()
      best_epoch = epoch + 1
      epochs_without_improvement = 0
      print(f'  → New best model! (Val Loss: {best_val_loss:.4f})')
    else:
      epochs_without_improvement += 1
      print(f'  → No improvement for {epochs_without_improvement} epoch(s)')
    
    # Early stopping check
    if epochs_without_improvement >= patience:
      print(f'\n⚠ Early stopping triggered after {epoch+1} epochs')
      print(f'Best model was at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}')
      break
  
  # Load best model weights before returning
  if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f'\n✓ Loaded best model from epoch {best_epoch}')
  
  return model, best_val_loss


def main(args):
  # Load data
  df = pd.read_csv(args.data_path)
  
  # Rename Ku to K for consistency
  if 'Ku' in df.columns:
    df = df.rename(columns={'Ku': 'K'})
  
  # Create Sk label based on S2k_top
  # Sk = 1 if |S2k_top - 1| < TOLERANCE, else Sk = 0
  df['Sk'] = (np.abs(df['S2k_top'] - 1) < TOLERANCE).astype(int)
  
  print(f"Loaded {len(df)} samples from {args.data_path}")
  print(f"Created Sk labels based on: |S2k_top - 1| < {TOLERANCE}")
  print(f"Class distribution: {df['Sk'].value_counts().to_dict()}")
  print(f"  Sk=1: {(df['Sk']==1).sum()} samples ({100*(df['Sk']==1).sum()/len(df):.1f}%)")
  print(f"  Sk=0: {(df['Sk']==0).sum()} samples ({100*(df['Sk']==0).sum()/len(df):.1f}%)")
  
  # Keep only needed columns
  df = df[['D', 'Ms', 'DMI', 'K', 'Sk']]
  
  # Split data
  train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Sk'])
  
  # Calculate class weights for imbalance
  class_counts = df['Sk'].value_counts().to_dict()
  total = len(df)
  class_weights = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
  
  # Create datasets
  train_dataset = PhaseDataset(train_df, model_type=args.model, fit_scaler=True)
  val_dataset = PhaseDataset(val_df, model_type=args.model, scaler=train_dataset.scaler)
  
  # Create dataloaders
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  
  # Initialize model with correct dimensions
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  
  # Get grid dimensions from dataset
  n_d = len(train_dataset.D_unique)
  n_ms = len(train_dataset.Ms_unique)
  n_dmi = len(train_dataset.DMI_unique)
  n_k = len(train_dataset.K_unique)
  
  print(f"Grid dimensions: D={n_d}, Ms={n_ms}, DMI={n_dmi}, K={n_k}")
  
  if args.model == 'option1':
    model = Option1_2DCNN_Separate_Dynamic(n_d, n_ms).to(device)
  elif args.model == 'option2':
    model = Option2_2DCNN_4Channels_Dynamic(n_d, n_ms).to(device)
  elif args.model == 'option3':
    model = Option3_3DCNN_Dynamic(n_d, n_ms, n_dmi * n_k).to(device)
  elif args.model == 'dense':
    model = DenseNetwork(n_features=10).to(device)
  else:
    raise ValueError(f"Unknown model type: {args.model}")
  
  print(f"\nTraining {args.model} model...")
  
  # Train
  model, best_loss = train_model(
    model, train_loader, val_loader, device,
    epochs=args.epochs, lr=args.lr, class_weights=class_weights,
    patience=args.patience
  )
  
  # Save model
  os.makedirs('saved_models', exist_ok=True)
  save_path = f'saved_models/{args.model}_model.pt'
  
  torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': train_dataset.scaler,
    'model_type': args.model,
    'grid_dims': {
      'n_d': n_d,
      'n_ms': n_ms,
      'n_dmi': n_dmi,
      'n_k': n_k,
      'D_unique': train_dataset.D_unique,
      'Ms_unique': train_dataset.Ms_unique,
      'DMI_unique': train_dataset.DMI_unique,
      'K_unique': train_dataset.K_unique
    }
  }, save_path)
  
  print(f"\nModel saved to {save_path}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train phase diagram classifier')
  parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file')
  parser.add_argument('--model', type=str, required=True, 
                      choices=['option1', 'option2', 'option3', 'dense'],
                      help='Model architecture to use')
  parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--patience', type=int, default=10, 
                      help='Early stopping patience (epochs without improvement)')
  
  args = parser.parse_args()
  main(args)