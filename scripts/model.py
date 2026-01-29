"""
Model architecture for Skyrmion phase classification
Import this in both training and prediction scripts
"""

import torch.nn as nn

class PhaseCNN(nn.Module):
    def __init__(self, input_features=4, num_classes=2):
        super(PhaseCNN, self).__init__()
        
        # Reshape 1D features to work with Conv1d
        # We'll treat each feature as a channel
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the flattened size after convolutions
        # After conv1: (4+2*1-2)+1 = 5, after pool: 2
        # After conv2: (2+2*1-2)+1 = 3, after pool: 1
        # After conv3: (1+2*1-2)+1 = 2, after pool: 1
        self.fc1 = nn.Linear(64 * 1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Reshape from (batch, features) to (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    

class PhaseMLP(nn.Module):
    """
    Simple MLP that might work better for low-dimensional tabular data
    """
    def __init__(self, input_features=4, num_classes=2):
        super(PhaseMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu(self.bn3(self.fc3(x))))
        x = self.dropout4(self.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x