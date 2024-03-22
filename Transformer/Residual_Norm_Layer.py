import torch
import torch.nn as nn

class Residual_Norm_Layer(nn.Module):
    def __init__(self, dim, dropout=0.5):
        super(Residual_Norm_Layer, self).__init__()
        
        self.layer = nn.LayerNorm(dim)  # layer normalization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.layer(x)))
        return self.layer(x + self.dropout(sublayer(x)))    # residual connection진행 후 layer norm진행
