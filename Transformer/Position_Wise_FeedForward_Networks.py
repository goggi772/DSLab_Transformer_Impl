import torch
import torch.nn as nn

class Position_Wise_FeedForward_Networks(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout=0.5):
        super(Position_Wise_FeedForward_Networks).__init__()
        
        self.fc_1 = nn.Linear(hidden_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [batch size, seq length, hidden dim]
        output = self.fc_1(x)
        # x = [batch size, seq length, FFN dim]
        output = self.relu(output)
        output = self.dropout(output)
        
        output = self.fc_2(output)
        # x = [batch size, seq length, hidden dim]
        return output