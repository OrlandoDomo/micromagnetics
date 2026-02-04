import torch
import torch.nn as nn
import torch.nn.functional as F


class Option1_2DCNN_Separate_Dynamic(nn.Module):
  """
  2D CNN treating (D, Ms) as phase diagram with DMI and K as separate inputs
  Input: phase_diagram (batch, 1, n_d, n_ms) + params (batch, 2)
  Dynamic grid size based on actual data dimensions
  """
  def __init__(self, n_d=10, n_ms=11):
    super(Option1_2DCNN_Separate_Dynamic, self).__init__()
    
    # CNN branch for phase diagram
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    # Dense branch for DMI, K
    self.fc_params = nn.Linear(2, 16)
    
    # Combined layers
    self.fc1 = nn.Linear(128 + 16, 64)
    self.dropout1 = nn.Dropout(0.3)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 1)
    
  def forward(self, phase_diagram, params):
    # CNN branch
    x = F.relu(self.conv1(phase_diagram))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    
    # Dense branch
    y = F.relu(self.fc_params(params))
    
    # Combine
    combined = torch.cat([x, y], dim=1)
    z = F.relu(self.fc1(combined))
    z = self.dropout1(z)
    z = F.relu(self.fc2(z))
    output = torch.sigmoid(self.fc3(z))
    
    return output


class Option2_2DCNN_4Channels_Dynamic(nn.Module):
  """
  2D CNN treating all 4 parameters as channels in (D, Ms) space
  Input: (batch, 4, n_d, n_ms) where channels are [D, Ms, DMI, K]
  Dynamic grid size based on actual data dimensions
  """
  def __init__(self, n_d=10, n_ms=11):
    super(Option2_2DCNN_4Channels_Dynamic, self).__init__()
    
    self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
    self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    self.fc1 = nn.Linear(128, 64)
    self.dropout = nn.Dropout(0.3)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 1)
    
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    output = torch.sigmoid(self.fc3(x))
    
    return output


class Option3_3DCNN_Dynamic(nn.Module):
  """
  3D CNN treating data as 4D volume (D, Ms, DMI, K)
  Input: (batch, 1, n_d, n_ms, n_dmi*n_k)
  Dynamic grid size based on actual data dimensions
  """
  def __init__(self, n_d=10, n_ms=11, n_dmi_k=30):
    super(Option3_3DCNN_Dynamic, self).__init__()
    
    # Using 3D convolutions
    self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
    self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
    self.pool = nn.MaxPool3d((2, 2, 1))
    self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 2))
    self.global_pool = nn.AdaptiveAvgPool3d(1)
    
    self.fc1 = nn.Linear(128, 64)
    self.dropout = nn.Dropout(0.3)
    self.fc2 = nn.Linear(64, 1)
    
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = self.global_pool(x)
    x = x.view(x.size(0), -1)
    
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    output = torch.sigmoid(self.fc2(x))
    
    return output


class DenseNetwork(nn.Module):
  """
  Simple dense network with feature engineering
  Input: (batch, n_features)
  """
  def __init__(self, n_features=10):
    super(DenseNetwork, self).__init__()
    
    self.fc1 = nn.Linear(n_features, 128)
    self.dropout1 = nn.Dropout(0.3)
    self.fc2 = nn.Linear(128, 64)
    self.dropout2 = nn.Dropout(0.3)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, 1)
    
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout1(x)
    x = F.relu(self.fc2(x))
    x = self.dropout2(x)
    x = F.relu(self.fc3(x))
    output = torch.sigmoid(self.fc4(x))
    
    return output