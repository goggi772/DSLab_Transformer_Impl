import torch
import torch.nn as nn
import Multi_Head_Attention
import Position_Wise_FeedForward_Networks
import Residual_Norm_Layer

class Decoder_Layer(nn.Module):
    def __init__(self, head_n, hidden_dim, ffn_dim, dropout=0.5):
        super(Decoder_Layer, self).__init__()
        
        '''
            Decoder layer 구조인 Masked Multi Head Attention -> Residual & Norm -> Multi Head Attention -> 
            Residual & Norm -> FFN layer -> Residual & Norm 구현
        '''
        
        self.masked_attention = Multi_Head_Attention(head_n, hidden_dim, dropout)
        self.residual_norm1 = Residual_Norm_Layer(hidden_dim, dropout)
        
        self.attention = Multi_Head_Attention(head_n, hidden_dim, dropout)
        self.residual_norm2 = Residual_Norm_Layer(hidden_dim, dropout)
        
        self.FFN = Position_Wise_FeedForward_Networks(hidden_dim, ffn_dim, dropout)
        self.residual_norm3 = Residual_Norm_Layer(hidden_dim, dropout)
        
    def forward(self, x, target, x_mask, target_mask):
        next_target, _ = self.masked_attention(target, target, target, target_mask)     # input embedding으로부터 Query, Key, Value 받음
        target = self.residual_norm1(target, next_target)
        
        next_target, score = self.attention(target, x, x, x_mask)       # Encoder로부터 Key, Value(x, x)를 받고 Masked Multi Head Attention으로부터 Query를(target) 입력받음
        target = self.residual_norm2(target, next_target)
        
        next_target = self.FFN(target)
        target = self.residual_norm3(target, next_target)
        
        return target, score
        