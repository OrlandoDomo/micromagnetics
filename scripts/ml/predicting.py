import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from .models import (
  DenseNetwork_BatchNorm,
  DenseNetwork_DropOut
)

THRESHOLD = 0.5

def load_model(model_path, device='cpu'):
  """Load trained model from checkpoint"""
  checkpoint = torch.load(model_path, map_location=device, weights_only=False)
  model_type = checkpoint['model_type']
  scaler = checkpoint['scaler']
  
  # Initialize model
  if model_type == 'dnn_do':
    model = DenseNetwork_DropOut(n_features=8)
  elif model_type == 'dnn_batch':
    model = DenseNetwork_BatchNorm(n_features=8)
  else:
    raise ValueError(f"Unknown model type: {model_type}")
  
  model.load_state_dict(checkpoint['model_state_dict'])

  model.to(device)
  model.eval()
  
  return model, scaler, model_type

def predict_single(model, scaler, D, Ms, DMI, Ku, device='cpu'):

  model.eval()
  
  # Standardize inputs
  #input_df = pl.DataFrame([[D, Ms, DMI, Ku]], columns=['D', 'Ms', 'DMI', 'Ku'])
  Aexchange = 1e-11
  raw_features = [D, Ms, DMI, Ku]
  eps = 1e-10
  mu0 = 4 * np.pi * 1e-7

  DMI = DMI*1e-3
  Ms = Ms*1e3
  Ku = Ku*1e6

  Q = (2*Ku) / (mu0 * Ms**2 + eps)
  kappa = (np.pi * DMI)/(4*np.sqrt(Aexchange * Ku + eps))
  dmi = DMI/(np.sqrt(Aexchange * Ku + eps))
  lex = np.sqrt(Aexchange/(Ku + eps))
  
  raw_features.extend([Q, kappa, dmi, lex])
  features = np.array(raw_features)

  scaled_features = scaler.transform(features.reshape(1,-1)).flatten()
  
  with torch.no_grad():
    x = torch.tensor(np.array([scaled_features]), dtype=torch.float32).to(device)
    output = torch.sigmoid(model(x))
  
  probability = output.item()
  prediction = 1 if probability > THRESHOLD else 0
  
  return prediction, probability


def create_phase_diagram(
  model, scaler, model_type, DMI, Ku, 
  D_range=(150, 825), Ms_range=(260, 460), 
  resolution=100, device='cpu', save_path=None
):
  
  # Create grid - Ms on x-axis, D on y-axis
  Ms_values = np.linspace(Ms_range[0], Ms_range[1], resolution)
  D_values = np.linspace(D_range[0], D_range[1], resolution)
  Ms_grid, D_grid = np.meshgrid(Ms_values, D_values)
  
  # Predict for each point
  predictions = np.zeros((resolution, resolution))
  probabilities = np.zeros((resolution, resolution))
  
  print(f"Generating phase diagram for DMI={DMI}, K={Ku}...")
  print(f"Ms range (x-axis): {Ms_range[0]} to {Ms_range[1]}")
  print(f"D range (y-axis): {D_range[0]} to {D_range[1]}")
  print(f"Resolution: {resolution}x{resolution} = {resolution**2} points")
  
  for i in range(resolution):
    for j in range(resolution):
      D = D_grid[i, j]
      Ms = Ms_grid[i, j]
      pred, prob = predict_single(model, scaler, D, Ms, DMI, Ku, device)
      predictions[i, j] = pred
      probabilities[i, j] = prob
    
    if (i + 1) % 10 == 0:
      print(f"Progress: {i+1}/{resolution} rows completed")
  
  # Create figure
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
  
  # Plot 1: Binary predictions
  cmap_binary = ListedColormap(['blue', 'red'])
  im1 = ax1.imshow(predictions, extent=[Ms_range[0], Ms_range[1], D_range[0], D_range[1]],
                   origin='lower', cmap=cmap_binary, aspect='auto', interpolation='nearest')
  ax1.set_xlabel('Ms', fontsize=12)
  ax1.set_ylabel('D', fontsize=12)
  ax1.set_title(f'Phase Diagram (DMI={DMI}, Ku={Ku})\nBinary Prediction', fontsize=14)
  cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1])
  cbar1.set_label('Sk', fontsize=12)
  ax1.grid(True, alpha=0.3)
  
  # Plot 2: Probability map
  im2 = ax2.imshow(probabilities, extent=[Ms_range[0], Ms_range[1], D_range[0], D_range[1]],
                   origin='lower', cmap='RdYlBu_r', aspect='auto', interpolation='bilinear',
                   vmin=0, vmax=1)
  ax2.set_xlabel('Ms', fontsize=12)
  ax2.set_ylabel('D', fontsize=12)
  ax2.set_title(f'Phase Diagram (DMI={DMI}, K={Ku})\nProbability Map', fontsize=14)
  cbar2 = plt.colorbar(im2, ax=ax2)
  cbar2.set_label('P(Sk=1)', fontsize=12)
  ax2.grid(True, alpha=0.3)
  
  plt.tight_layout()
  
  # Save if path provided
  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Phase diagram saved to {save_path}")
  
  return fig

def main(model_path, D_min=150, D_max=825, Ms_min=260, Ms_max=460, DMI=0.5, Ku=0.08, resolution=10, save_path=None):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  
  # Load model
  print(f"Loading model from {model_path}...")
  model, scaler, model_type = load_model(model_path, device)
  model = model.to(device)
  print(f"Model type: {model_type}")

  # Create phase diagram
  fig = create_phase_diagram(
    model, scaler, model_type,
    DMI=DMI, Ku=Ku,
    D_range=(D_min, D_max),
    Ms_range=(Ms_min, Ms_max),
    resolution=resolution,
    device=device,
    save_path=save_path
  )
  
  if not save_path:
    plt.show()
  
if __name__ == '__main__':
  
  args = {
    'model_path': 'ml/saved_models/DenseNN-BatchNorm_model_bs-64.pt',
    'DMI': 0.5,
    'Ku': 0.08,
    'resolution': 45,
    'save_path': '../data/phase_daigram_dnn_bn_hi-res.png'
  }

  main(**args)