import torch
import torch.nn as nn
import Positional_Encoding
import Encoder_Layer

'''
    구현한 Encoder Layer를 설정한 Layer개수만큼 쌓음
'''

class Encoder(nn.Module):
    def __init__(self, layer_n, source_voca_size, max_len, head_n, hidden_dim, ffn_dim, device, dropout=0.5):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(source_voca_size, hidden_dim)        # input embedding
        self.p_encoding = Positional_Encoding(max_len, hidden_dim, device)  # Positional encoding
        
        self.encoder_layers = nn.ModuleList([Encoder_Layer(head_n, hidden_dim, ffn_dim, dropout) for _ in range(layer_n)]) # layer_n의 값에 따라 encoder layer가 쌓임
        
    def forward(self, src, src_mask):
        src = self.embedding(src)
        src = self.p_encoding(src)
        
        for encoder_layer in self.encoder_layers:
            src, score = encoder_layer(src, src_mask)
            
        return src, score