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
        
    def forward(self, src, trg, src_mask, trg_mask):
        
        #trg = [batch size, trg len, hid dim]
        #src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        next_trg, _ = self.masked_attention(trg, trg, trg, trg_mask)     # input embedding으로부터 Query, Key, Value 받음
        trg = self.residual_norm1(trg, next_trg)
        
        next_trg, score = self.attention(trg, src, src, src_mask)       # Encoder로부터 Key, Value(src, src)를 받고 Masked Multi Head Attention으로부터 Query를(trg) 입력받음
        trg = self.residual_norm2(trg, next_trg)
        
        next_trg = self.FFN(trg)
        trg = self.residual_norm3(trg, next_trg)
        
        return trg, score
        