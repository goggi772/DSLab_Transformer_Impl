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
        
    def forward(self, x, x_mask):
        x = self.embedding(x)
        x = self.p_encoding(x)
        
        for encoder_layer in self.encoder_layers:
            x, score = encoder_layer(x, x_mask)
            
        return x, score