import torch
import torch.nn as nn

class Position_Wise_FeedForward_Networks(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout=0.5):
        super(Position_Wise_FeedForward_Networks).__init__()
        
        self.layer1 = nn.Linear(hidden_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.layer2(output)
        
        return output