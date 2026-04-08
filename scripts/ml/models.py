import torch.nn as nn
import torch.nn.functional as F

class DenseNetwork_BatchNorm(nn.Module):
  def __init__(self, n_features=8):
    super(DenseNetwork_BatchNorm, self).__init__()
    self.name = 'DenseNN-BatchNorm'
    self.type = 'dnn_batch'
    self.net = nn.Sequential(
      nn.Linear(n_features, 128),
      nn.ReLU(),
      nn.BatchNorm1d(128),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.BatchNorm1d(64),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
    )
  
  def forward(self, x):
    return self.net(x)

class DenseNetwork_DropOut(nn.Module):
  def __init__(self, n_features=8, dropout_rate=0.3):
    super(DenseNetwork_DropOut, self).__init__()
    self.name = 'DenseNN-DropOut'
    self.type = 'dnn_do'
    self.net = nn.Sequential(
      nn.Linear(n_features, 128),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(dropout_rate),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
    )

  def forward(self, x):
    return self.net(x)