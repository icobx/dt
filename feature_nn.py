import torch
import torch.nn as nn


class FeatureNN(nn.Module):

    def __init__(self, feature_dim, hidden_dim=128, n_hidden_layers=1, dropout=0.5):
        super(FeatureNN, self).__init__()

        dropouts = [nn.Dropout(p=dropout) for _ in range(n_hidden_layers+1)]
        dense_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]

        sequence = [None]*(len(dropouts)+len(dense_layers))

        sequence[::2] = dropouts
        sequence[1::2] = dense_layers

        self.seq = nn.Sequential(nn.Linear(feature_dim, hidden_dim), *sequence, nn.Linear(hidden_dim, 1))

    def forward(self, features):
        return torch.squeeze(self.seq(features), 1)
