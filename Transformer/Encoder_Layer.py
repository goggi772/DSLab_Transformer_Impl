import torch
import torch.nn as nn
import Multi_Head_Attention
import Position_Wise_FeedForward_Networks
import Residual_Norm_Layer

class Encoder_Layer(nn.Module):         # 하나의 encoder layer 구현
    def __init__(self, head_n, hidden_dim, ffn_dim, dropout=0.5):
        super(Encoder_Layer, self).__init__()
        
        '''
            Encoder layer 구조인 Multi Head attention -> Residual & Norm -> FFN layer -> Residual & Norm을 구현
        '''
        
        self.attention = Multi_Head_Attention(head_n, hidden_dim, dropout)
        self.residual_norm1 = Residual_Norm_Layer(hidden_dim, dropout)
        
        self.FFN = Position_Wise_FeedForward_Networks(hidden_dim, ffn_dim, dropout)
        self.residual_norm2 = Residual_Norm_Layer(hidden_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, x_mask):
        next_x, score = self.attention(x, x, x, x_mask)     # 인자 x는 순서대로 Query, Key, Value를 의미(x_mask는 mask여부)
        x = self.residual_norm1(x, next_x)  # residual connection, layer norm한 결과
        
        next_x = self.FFN(x)
        x = self.residual_norm2(x, next_x)
        
        return x, score
    
        