import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import pickle

from model import PhaseCNN, PhaseMLP
from predicting import predict_phase

TOLERANCE = 0.25
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom Dataset class
class MagneticPhaseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_preds = []
        train_labels = []
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Detailed logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            train_sky_pred = np.sum(np.array(train_preds) == 1)
            train_sky_true = np.sum(np.array(train_labels) == 1)
            val_sky_pred = np.sum(np.array(val_preds) == 1)
            val_sky_true = np.sum(np.array(val_labels) == 1)
            
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Train predictions: {train_sky_pred}/{len(train_preds)} skyrmions ({100*train_sky_pred/len(train_preds):.1f}%)')
            print(f'  Train true labels: {train_sky_true}/{len(train_labels)} skyrmions ({100*train_sky_true/len(train_labels):.1f}%)')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Val predictions: {val_sky_pred}/{len(val_preds)} skyrmions ({100*val_sky_pred/len(val_preds):.1f}%)')
            print(f'  Val true labels: {val_sky_true}/{len(val_labels)} skyrmions ({100*val_sky_true/len(val_labels):.1f}%)')
            print('-' * 60)
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return train_losses, val_losses, train_accs, val_accs


# Main workflow
def main(csv_path='data.csv', model_type='mlp', use_oversampling=True):
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}\n")
    
    # Extract features and labels
    feature_columns = ['D', 'Ms', 'Ku', 'DMI']
    X = df[feature_columns].values
    y = df['S2k_bot'].values
    
    # Check S2k statistics
    print(f"S2k statistics:")
    print(f"  Min: {y.min():.6f}, Max: {y.max():.6f}")
    print(f"  Mean: {y.mean():.6f}, Std: {y.std():.6f}")
    
    # Convert S2k to binary: abs(S2k - 1) < 0.3 -> 1 (Skyrmion), else -> 0 (Other)
    tolerance = TOLERANCE
    y_binary = (np.abs(y - 1.0) < tolerance).astype(int)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Binary classification with tolerance={tolerance}:")
    print(f"  Skyrmion (|S2k-1| < {tolerance}): {np.sum(y_binary==1)} samples")
    print(f"  Other phase: {np.sum(y_binary==0)} samples")
    print(f"  Class balance: {100*np.sum(y_binary==1)/len(y_binary):.2f}% skyrmions")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply oversampling for minority class if requested
    if use_oversampling:
        print(f"\n{'='*60}")
        print("Applying Random Oversampling to balance classes...")
        
        # Find minority and majority class indices
        skyrmion_indices = np.where(y_binary == 1)[0]
        other_indices = np.where(y_binary == 0)[0]
        
        n_skyrmion = len(skyrmion_indices)
        n_other = len(other_indices)
        
        # Oversample minority class to match majority
        if n_skyrmion < n_other:
            # Randomly sample with replacement from skyrmion class
            oversample_indices = np.random.choice(skyrmion_indices, 
                                                   size=n_other - n_skyrmion, 
                                                   replace=True)
            all_indices = np.concatenate([other_indices, skyrmion_indices, oversample_indices])
        else:
            oversample_indices = np.random.choice(other_indices, 
                                                   size=n_skyrmion - n_other, 
                                                   replace=True)
            all_indices = np.concatenate([other_indices, oversample_indices, skyrmion_indices])
        
        # Shuffle
        np.random.shuffle(all_indices)
        
        X_scaled = X_scaled[all_indices]
        y_binary = y_binary[all_indices]
        
        print(f"After oversampling:")
        print(f"  Total samples: {len(y_binary)}")
        print(f"  Skyrmion: {np.sum(y_binary==1)} samples ({100*np.sum(y_binary==1)/len(y_binary):.2f}%)")
        print(f"  Other: {np.sum(y_binary==0)} samples ({100*np.sum(y_binary==0)/len(y_binary):.2f}%)")
        print(f"{'='*60}\n")
    
    # Create dataset with binary labels
    dataset = MagneticPhaseDataset(X_scaled, y_binary)
    
    # Split into train, validation, test
    train_size = int(0.6 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Model type: {model_type.upper()}")
    
    if model_type.lower() == 'mlp':
        model = PhaseMLP(input_features=4, num_classes=2).to(device)
    else:
        model = PhaseCNN(input_features=4, num_classes=2).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Handle class imbalance with weighted loss (if not using oversampling)
    # Calculate class weights (inverse frequency)
    n_other = np.sum(y_binary == 0)
    n_skyrmion = np.sum(y_binary == 1)
    
    if n_skyrmion > 0 and not use_oversampling:
        weight_other = len(y_binary) / (2 * n_other)
        weight_skyrmion = len(y_binary) / (2 * n_skyrmion)
        class_weights = torch.FloatTensor([weight_other, weight_skyrmion]).to(device)
        
        print(f"\nClass weights for balanced loss:")
        print(f"  Other phase weight: {weight_other:.4f}")
        print(f"  Skyrmion weight: {weight_skyrmion:.4f}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    if n_skyrmion == 0:
        print("\n No skyrmion samples found! Cannot train classifier.")
        return None, None
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=10000, device=device
    )
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to 'training_history.png'")
    
    # Test model
    print("\nEvaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    # Detailed prediction analysis
    print("\n" + "="*60)
    print("TEST SET PREDICTION ANALYSIS:")
    print("="*60)
    print(f"Total test samples: {len(all_preds)}")
    print(f"Predicted skyrmions: {np.sum(np.array(all_preds)==1)} ({100*np.sum(np.array(all_preds)==1)/len(all_preds):.2f}%)")
    print(f"True skyrmions: {np.sum(np.array(all_labels)==1)} ({100*np.sum(np.array(all_labels)==1)/len(all_labels):.2f}%)")
    print(f"\nSkyrmion probability statistics:")
    print(f"  Mean P(skyrmion): {all_probs[:, 1].mean():.4f}")
    print(f"  Max P(skyrmion): {all_probs[:, 1].max():.4f}")
    print(f"  Min P(skyrmion): {all_probs[:, 1].min():.4f}")
    print(f"  Std P(skyrmion): {all_probs[:, 1].std():.4f}")
    
    # Show distribution of predictions
    print(f"\nPrediction confidence distribution:")
    sky_probs = all_probs[:, 1]
    print(f"  P(sky) < 0.1: {np.sum(sky_probs < 0.1)} samples")
    print(f"  0.1 ≤ P(sky) < 0.3: {np.sum((sky_probs >= 0.1) & (sky_probs < 0.3))} samples")
    print(f"  0.3 ≤ P(sky) < 0.5: {np.sum((sky_probs >= 0.3) & (sky_probs < 0.5))} samples")
    print(f"  0.5 ≤ P(sky) < 0.7: {np.sum((sky_probs >= 0.5) & (sky_probs < 0.7))} samples")
    print(f"  0.7 ≤ P(sky) < 0.9: {np.sum((sky_probs >= 0.7) & (sky_probs < 0.9))} samples")
    print(f"  P(sky) ≥ 0.9: {np.sum(sky_probs >= 0.9)} samples")
    
    # Check if model learned anything
    if all_probs[:, 1].max() < 0.3:
        print("\n WARNING: Model never predicts high skyrmion probability!")
        print("   This suggests the model hasn't learned to distinguish skyrmions.")
        print("   Possible issues:")
        print("   1. Features (D, Ms, Ku, DMI) may not be sufficient to predict S2k")
        print("   2. The relationship might be too complex for this architecture")
        print("   3. Data quality issues or mislabeling")
        print("   4. Need more skyrmion samples")
    
    print("="*60)
    
    # Print classification report
    target_names = ['Other Phase', 'Skyrmion']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
        
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(f"[[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f" [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    # Save model
    torch.save(model.state_dict(), 'skyrmion_cnn_model.pth')
    print("\nModel saved to 'skyrmion_cnn_model.pth'")
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to 'scaler.pkl'")
    
    # Save feature ranges for plotting
    feature_ranges = {
        'D': (X[:, 0].min(), X[:, 0].max()),
        'Ms': (X[:, 1].min(), X[:, 1].max()),
        'Ku': (X[:, 2].min(), X[:, 2].max()),
        'DMI': (X[:, 3].min(), X[:, 3].max())
    }
    with open('feature_ranges.pkl', 'wb') as f:
        pickle.dump(feature_ranges, f)
    print("Feature ranges saved to 'feature_ranges.pkl'")
    print("\nFeature ranges for plotting:")
    for key, (min_val, max_val) in feature_ranges.items():
        print(f"  {key}: [{min_val:.4f}, {max_val:.4f}]")
    
    return model, scaler

if __name__ == "__main__":
    # Run the main workflow
    # Replace 'data.csv' with your actual CSV file path
    csv_file_path = '..\data\saf_skyrmion_results_final.csv'
    model, scaler = main(csv_file_path, model_type='cnn', use_oversampling=False)
    
    if model is not None:
        print("\n" + "="*60)
        print("DIAGNOSTIC: Testing model on a few skyrmion samples")
        print("="*60)
        
        # Load the data again to get some skyrmion examples
        df = pd.read_csv(csv_file_path)
        y = df['S2k_bot'].values
        tolerance = 0.2
        y_binary = (np.abs(y - 1.0) < tolerance).astype(int)
        
        skyrmion_indices = np.where(y_binary == 1)[0][:5]  # First 5 skyrmions
        other_indices = np.where(y_binary == 0)[0][:5]      # First 5 others
        
        if len(skyrmion_indices) > 0:
            X_test = df[['D', 'Ms', 'Ku', 'DMI']].values
            
            print("\nTesting on actual skyrmion samples from training data:")
            test_samples_sky = X_test[skyrmion_indices]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            preds, probs = predict_phase(model, scaler, test_samples_sky, device)
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                print(f"  Skyrmion sample {i+1}: Predicted={pred}, P(sky)={prob[1]:.4f}")
            
            print("\nTesting on actual other phase samples from training data:")
            test_samples_other = X_test[other_indices]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            preds, probs = predict_phase(model, scaler, test_samples_sky, device)
            for i, (pred, prob) in enumerate(zip(preds, probs)):
                print(f"  Other sample {i+1}: Predicted={pred}, P(sky)={prob[1]:.4f}")
    
    # Alternative configurations:
    # model, scaler = main('data.csv', model_type='mlp', use_oversampling=False)  # MLP with class weights
    # model, scaler = main('data.csv', model_type='cnn', use_oversampling=True)   # CNN with oversampling