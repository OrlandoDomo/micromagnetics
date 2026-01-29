import torch
import numpy as np
import pickle
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import model architecture
from model import PhaseCNN, PhaseMLP

def load_model(model_path='skyrmion_cnn_model.pth', model_type='mlp', device=None):
    """
    Load the trained model from .pth file
    
    Args:
        model_path: Path to the .pth file
        model_type: Type of model ('mlp' or 'cnn')
        device: Device to load model on ('cuda' or 'cpu'). If None, auto-detect.
    
    Returns:
        Loaded model in evaluation mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model architecture based on type
    if model_type.lower() == 'mlp':
        model = PhaseMLP(input_features=4, num_classes=2)
    else:
        model = PhaseCNN(input_features=4, num_classes=2)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    print(f"Model loaded from {model_path}")
    print(f"Using device: {device}")
    
    return model, device


def load_scaler(scaler_path='scaler.pkl'):
    """
    Load the fitted scaler from pickle file
    
    Args:
        scaler_path: Path to the scaler pickle file
    
    Returns:
        Loaded scaler object
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Scaler loaded from {scaler_path}")
    return scaler

def load_feature_ranges(ranges_path='feature_ranges.pkl'):
    """
    Load the feature ranges from training data
    
    Args:
        ranges_path: Path to the feature ranges pickle file
    
    Returns:
        Dictionary with feature ranges
    """
    try:
        with open(ranges_path, 'rb') as f:
            ranges = pickle.load(f)
        print(f"Feature ranges loaded from {ranges_path}")
        print("Available ranges:")
        for key, (min_val, max_val) in ranges.items():
            print(f"  {key}: [{min_val:.4f}, {max_val:.4f}]")
        return ranges
    except FileNotFoundError:
        print(f"⚠️  Warning: {ranges_path} not found. Using default ranges.")
        return {
            'D': (100, 1000),
            'Ms': (100, 1000),
            'Ku': (0.05, 0.5),
            'DMI': (0.1, 0.5)
        }

def predict_phase(model, scaler, new_data, device='cpu'):
    """
    Predict phase for new data
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        new_data: numpy array of shape (n_samples, 4) with columns [D, Ms, Ku, DMI]
        device: Device to run inference on
    
    Returns:
        List of predictions ['Skyrmion' or 'Other Phase']
    """
    model.eval()
    
    # Scale the input data
    new_data_scaled = scaler.transform(new_data)
    new_data_tensor = torch.FloatTensor(new_data_scaled).to(device)
    
    with torch.no_grad():
        outputs = model(new_data_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
    
    phase_names = ['Other Phase', 'Skyrmion']
    predictions = [phase_names[p] for p in predicted.cpu().numpy()]
    probs = probabilities.cpu().numpy()
    
    return predictions, probs


def predict_with_confidence(model, scaler, new_data, device='cpu'):
    """
    Predict phase with confidence scores
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        new_data: numpy array of shape (n_samples, 4) with columns [D, Ms, Ku, DMI]
        device: Device to run inference on
    
    Returns:
        List of tuples (prediction, confidence_score)
    """
    predictions, probs = predict_phase(model, scaler, new_data, device)
    
    results = []
    for pred, prob in zip(predictions, probs):
        confidence = prob[1] if pred == 'Skyrmion' else prob[0]
        results.append((pred, confidence))
    
    return results

def plot_phase_map_2d(model, scaler, d_range=None, ms_range=None, ku_fixed=None, dmi_fixed=None, 
                      resolution=100, device='cpu',feature_ranges=None):
    """
    Plot a 2D phase map with D on X-axis and Ms on Y-axis
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        d_range: Tuple (d_min, d_max) for D values. If None, use feature_ranges.
        ms_range: Tuple (ms_min, ms_max) for Ms values. If None, use feature_ranges.
        ku_fixed: Fixed value for Ku parameter. If None, use mean from feature_ranges.
        dmi_fixed: Fixed value for DMI parameter. If None, use mean from feature_ranges.
        resolution: Number of points in each dimension (default: 100)
        device: Device to run inference on
        save_path: Path to save the plot (default: 'phase_map_2d.png')
        feature_ranges: Dictionary with feature ranges from training
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
        predictions_2d: 2D array of predictions
    """
    # Use feature ranges if provided
    if feature_ranges is None:
        feature_ranges = load_feature_ranges()
    
    if d_range is None:
        d_range = feature_ranges['D']
    if ms_range is None:
        ms_range = feature_ranges['Ms']
    if ku_fixed is None:
        ku_fixed = (feature_ranges['Ku'][0] + feature_ranges['Ku'][1]) / 2
    if dmi_fixed is None:
        dmi_fixed = (feature_ranges['DMI'][0] + feature_ranges['DMI'][1]) / 2
    # Create meshgrid
    d_values = np.linspace(d_range[0], d_range[1], resolution)
    ms_values = np.linspace(ms_range[0], ms_range[1], resolution)
    Ms, D = np.meshgrid(ms_values, d_values)
    
    # Flatten for prediction
    d_flat = D.flatten()
    ms_flat = Ms.flatten()
    
    # Create input array with fixed Ku and DMI
    n_points = len(d_flat)
    input_data = np.column_stack([
        d_flat,
        ms_flat,
        np.full(n_points, ku_fixed),
        np.full(n_points, dmi_fixed)
    ])
    
    print(f"Generating phase map with {n_points} points...")
    print(f"D range: [{d_range[0]:.4f}, {d_range[1]:.4f}]")
    print(f"Ms range: [{ms_range[0]:.4f}, {ms_range[1]:.4f}]")
    print(f"Ku (fixed): {ku_fixed:.4f}")
    print(f"DMI (fixed): {dmi_fixed:.4f}")
    
    # Get predictions
    predictions, probabilities = predict_phase(model, scaler, input_data, device)
    
    # Convert predictions to binary (0 or 1)
    predictions_binary = np.array([1 if p == 'Skyrmion' else 0 for p in predictions])
    predictions_2d = predictions_binary.reshape(D.shape)
    
    # Get skyrmion probabilities for coloring
    skyrmion_probs = probabilities[:, 1].reshape(D.shape)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Binary phase map
    #cmap_binary = ListedColormap(['#3498db', '#e74c3c'])  # Blue for Other, Red for Skyrmion
    im1 = ax1.contourf(Ms, D, predictions_2d, levels=1, cmap='summer', alpha=0.8)
    ax1.contour(Ms, D, predictions_2d, levels=1, colors='black', linewidths=1.5, alpha=0.3)
    
    ax1.set_xlabel('Ms', fontsize=12, fontweight='bold')
    ax1.set_ylabel('D', fontsize=12, fontweight='bold')
    ax1.set_title(f'Phase Map: Binary Classification\n(Ku={ku_fixed:.4f}, DMI={dmi_fixed:.4f})', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar for binary map
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0.25, 0.75])
    cbar1.ax.set_yticklabels(['Other Phase', 'Skyrmion'])
    
    # Count phases
    n_skyrmion = np.sum(predictions_2d == 1)
    n_other = np.sum(predictions_2d == 0)
    ax1.text(0.02, 0.98, f'Skyrmion: {n_skyrmion}/{n_points} ({100*n_skyrmion/n_points:.1f}%)\n'
                          f'Other: {n_other}/{n_points} ({100*n_other/n_points:.1f}%)',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Probability map
    im2 = ax2.contourf(Ms, D, skyrmion_probs, levels=20, cmap='viridis', alpha=0.9)
    ax2.contour(Ms, D, skyrmion_probs, levels=[0.5], colors='black', linewidths=2, 
                linestyles='--', alpha=0.5)
    
    ax2.set_xlabel('Ms', fontsize=12, fontweight='bold')
    ax2.set_ylabel('D', fontsize=12, fontweight='bold')
    ax2.set_title(f'Skyrmion Probability Map\n(Ku={ku_fixed:.2f}, DMI={dmi_fixed:.2f})', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar for probability map
    cbar2 = plt.colorbar(im2, ax=ax2, label='P(Skyrmion)')
    
    plt.tight_layout()
    save_path = f'..\images\phase_maps_predictions\phase_map_skyrmion_DMI={dmi_fixed}_Ku={ku_fixed}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')
    print(f"\nPhase map saved")
    
    return fig, (ax1, ax2), predictions_2d

if __name__ == "__main__":
    # Load model and scaler
    model, device = load_model('skyrmion_cnn_model.pth', model_type='cnn')
    scaler = load_scaler('scaler.pkl')
    feature_ranges = load_feature_ranges('feature_ranges.pkl')
    
    # Example: Predict for new samples
    # Format: [D, Ms, Ku, DMI]
    Ds = np.arange(150,750,75)
    Mss = np.arange(260,440,20)
    dmi = 0.5
    ku = 0.05
    
    input_df = pd.DataFrame(list(itertools.product(Ds, Mss)))
    input_df.columns = ['D','Ms']
    
    input_df['Ku'] = ku
    input_df['DMI'] = dmi
    
    input_data = input_df.to_numpy()
    
    print("\nCreating single phase map...")
    fig, axes, pred_map = plot_phase_map_2d(
        model=model,
        scaler=scaler,
        d_range=(150,750),
        ms_range=(260,440),
        ku_fixed=ku,
        dmi_fixed=dmi,
        resolution=10,           # 200x200 grid
        device=device,
        feature_ranges=feature_ranges
    )
    #plt.show()