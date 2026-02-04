import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
from models import (
  Option1_2DCNN_Separate_Dynamic,
  Option2_2DCNN_4Channels_Dynamic,
  Option3_3DCNN_Dynamic,
  DenseNetwork
)


def load_model(model_path, device='cpu'):
  """Load trained model from checkpoint"""
  checkpoint = torch.load(model_path, map_location=device, weights_only=False)
  model_type = checkpoint['model_type']
  scaler = checkpoint['scaler']
  grid_dims = checkpoint['grid_dims']
  
  # Initialize model with correct dimensions
  if model_type == 'option1':
    model = Option1_2DCNN_Separate_Dynamic(grid_dims['n_d'], grid_dims['n_ms'])
  elif model_type == 'option2':
    model = Option2_2DCNN_4Channels_Dynamic(grid_dims['n_d'], grid_dims['n_ms'])
  elif model_type == 'option3':
    model = Option3_3DCNN_Dynamic(grid_dims['n_d'], grid_dims['n_ms'], 
                                   grid_dims['n_dmi'] * grid_dims['n_k'])
  elif model_type == 'dense':
    model = DenseNetwork(n_features=10)
  else:
    raise ValueError(f"Unknown model type: {model_type}")
  
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  model.eval()
  
  return model, scaler, model_type, grid_dims


def predict_single(model, scaler, model_type, grid_dims, D, Ms, DMI, K, device='cpu'):
  """
  Predict Sk for a single point
  
  Args:
    model: trained model
    scaler: fitted StandardScaler
    model_type: type of model architecture
    grid_dims: dictionary with grid dimensions and unique values
    D, Ms, DMI, K: parameter values
    device: torch device
    
  Returns:
    predicted Sk value (0 or 1)
    probability
  """
  model.eval()
  
  # Standardize inputs - use DataFrame to avoid feature name warning
  input_df = pd.DataFrame([[D, Ms, DMI, K]], columns=['D', 'Ms', 'DMI', 'K'])
  scaled_input = scaler.transform(input_df)[0]
  D_scaled, Ms_scaled, DMI_scaled, K_scaled = scaled_input
  
  # Get actual unique values and create mappings
  D_unique = grid_dims['D_unique']
  Ms_unique = grid_dims['Ms_unique']
  DMI_unique = grid_dims['DMI_unique']
  K_unique = grid_dims['K_unique']
  
  # Find closest values in grid
  d_idx = np.argmin(np.abs(np.array(D_unique) - D))
  ms_idx = np.argmin(np.abs(np.array(Ms_unique) - Ms))
  dmi_idx = np.argmin(np.abs(np.array(DMI_unique) - DMI))
  k_idx = np.argmin(np.abs(np.array(K_unique) - K))
  
  n_d = len(D_unique)
  n_ms = len(Ms_unique)
  n_dmi = len(DMI_unique)
  n_k = len(K_unique)
  
  with torch.no_grad():
    if model_type == 'option1':
      phase_diagram = torch.zeros((1, 1, n_d, n_ms)).to(device)
      phase_diagram[0, 0, d_idx, ms_idx] = 1.0
      
      params = torch.tensor([[DMI_scaled, K_scaled]], dtype=torch.float32).to(device)
      output = model(phase_diagram, params)
      
    elif model_type == 'option2':
      x = torch.zeros((1, 4, n_d, n_ms)).to(device)
      x[0, 0, d_idx, ms_idx] = D_scaled
      x[0, 1, d_idx, ms_idx] = Ms_scaled
      x[0, 2, d_idx, ms_idx] = DMI_scaled
      x[0, 3, d_idx, ms_idx] = K_scaled
      output = model(x)
      
    elif model_type == 'option3':
      x = torch.zeros((1, 1, n_d, n_ms, n_dmi * n_k)).to(device)
      x[0, 0, d_idx, ms_idx, dmi_idx * n_k + k_idx] = 1.0
      output = model(x)
      
    else:  # dense
      features = [
        D_scaled, Ms_scaled, DMI_scaled, K_scaled,
        D_scaled * Ms_scaled,
        D_scaled * DMI_scaled,
        Ms_scaled * K_scaled,
        DMI_scaled * K_scaled,
        D_scaled**2,
        Ms_scaled**2
      ]
      x = torch.tensor([features], dtype=torch.float32).to(device)
      output = model(x)
  
  probability = output.item()
  prediction = 1 if probability > 0.5 else 0
  
  return prediction, probability


def create_phase_diagram(model, scaler, model_type, grid_dims, DMI, K, 
                          D_range=(150, 825), Ms_range=(260, 460), 
                          resolution=100, device='cpu', save_path=None):
  """
  Create a continuous phase diagram for given DMI and K values
  
  Args:
    model: trained model
    scaler: fitted StandardScaler
    model_type: type of model architecture
    grid_dims: dictionary with grid dimensions and unique values
    DMI: DMI value
    K: K value
    D_range: tuple of (min, max) for D axis (y-axis, default: 150 to 825)
    Ms_range: tuple of (min, max) for Ms axis (x-axis, default: 260 to 460)
    resolution: number of points along each axis
    device: torch device
    save_path: path to save the figure (optional)
    
  Returns:
    figure object
  """
  # Create grid - Ms on x-axis, D on y-axis
  Ms_values = np.linspace(Ms_range[0], Ms_range[1], resolution)
  D_values = np.linspace(D_range[0], D_range[1], resolution)
  Ms_grid, D_grid = np.meshgrid(Ms_values, D_values)
  
  # Predict for each point
  predictions = np.zeros((resolution, resolution))
  probabilities = np.zeros((resolution, resolution))
  
  print(f"Generating phase diagram for DMI={DMI}, K={K}...")
  print(f"Ms range (x-axis): {Ms_range[0]} to {Ms_range[1]}")
  print(f"D range (y-axis): {D_range[0]} to {D_range[1]}")
  print(f"Resolution: {resolution}x{resolution} = {resolution**2} points")
  
  for i in range(resolution):
    for j in range(resolution):
      D = D_grid[i, j]
      Ms = Ms_grid[i, j]
      pred, prob = predict_single(model, scaler, model_type, grid_dims, D, Ms, DMI, K, device)
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
  ax1.set_title(f'Phase Diagram (DMI={DMI}, K={K})\nBinary Prediction', fontsize=14)
  cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1])
  cbar1.set_label('Sk', fontsize=12)
  ax1.grid(True, alpha=0.3)
  
  # Plot 2: Probability map
  im2 = ax2.imshow(probabilities, extent=[Ms_range[0], Ms_range[1], D_range[0], D_range[1]],
                   origin='lower', cmap='RdYlBu_r', aspect='auto', interpolation='bilinear',
                   vmin=0, vmax=1)
  ax2.set_xlabel('Ms', fontsize=12)
  ax2.set_ylabel('D', fontsize=12)
  ax2.set_title(f'Phase Diagram (DMI={DMI}, K={K})\nProbability Map', fontsize=14)
  cbar2 = plt.colorbar(im2, ax=ax2)
  cbar2.set_label('P(Sk=1)', fontsize=12)
  ax2.grid(True, alpha=0.3)
  
  plt.tight_layout()
  
  # Save if path provided
  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Phase diagram saved to {save_path}")
  
  return fig


def main(args):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  
  # Load model
  print(f"Loading model from {args.model_path}...")
  model, scaler, model_type, grid_dims = load_model(args.model_path, device)
  print(f"Model type: {model_type}")
  print(f"Grid dimensions: D={grid_dims['n_d']}, Ms={grid_dims['n_ms']}, "
        f"DMI={grid_dims['n_dmi']}, K={grid_dims['n_k']}")
  
  if args.mode == 'single':
    # Single prediction
    pred, prob = predict_single(
      model, scaler, model_type, grid_dims,
      args.D, args.Ms, args.DMI, args.K, device
    )
    print(f"\nPrediction for D={args.D}, Ms={args.Ms}, DMI={args.DMI}, K={args.K}:")
    print(f"  Sk = {pred}")
    print(f"  Probability(Sk=1) = {prob:.4f}")
    
  elif args.mode == 'diagram':
    # Create phase diagram
    fig = create_phase_diagram(
      model, scaler, model_type, grid_dims,
      DMI=args.DMI, K=args.K,
      D_range=(args.D_min, args.D_max),
      Ms_range=(args.Ms_min, args.Ms_max),
      resolution=args.resolution,
      device=device,
      save_path=args.save_path
    )
    
    if not args.save_path:
      plt.show()
  
  else:
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Predict Sk and create phase diagrams')
  parser.add_argument('--model_path', type=str, required=True, 
                      help='Path to saved model checkpoint')
  parser.add_argument('--mode', type=str, required=True, choices=['single', 'diagram'],
                      help='Prediction mode: single point or full diagram')
  
  # Single prediction arguments
  parser.add_argument('--D', type=float, help='D value for single prediction')
  parser.add_argument('--Ms', type=float, help='Ms value for single prediction')
  parser.add_argument('--DMI', type=float, help='DMI value')
  parser.add_argument('--K', type=float, help='K value')
  
  # Phase diagram arguments
  parser.add_argument('--D_min', type=float, default=150, help='Minimum D value for diagram')
  parser.add_argument('--D_max', type=float, default=825, help='Maximum D value for diagram')
  parser.add_argument('--Ms_min', type=float, default=260, help='Minimum Ms value for diagram')
  parser.add_argument('--Ms_max', type=float, default=460, help='Maximum Ms value for diagram')
  parser.add_argument('--resolution', type=int, default=100, 
                      help='Grid resolution for phase diagram')
  parser.add_argument('--save_path', type=str, help='Path to save phase diagram image')
  
  args = parser.parse_args()
  main(args)