import torch
import torch.nn as nn
import Positional_Encoding
import Decoder_Layer

'''
    구현한 Decoder Layer를 설정한 Layer개수만큼 쌓음
'''

class Decoder(nn.Module):
    def __init__(self, layer_n, trg_voca_size, max_len, head_n, hidden_dim, ffn_dim, device, dropout=0.5):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(trg_voca_size, max_len)        # decoder의 input embedding
        self.position_embedding = Positional_Encoding(max_len, hidden_dim, device)  # positional encoding
        
        self.decoder_layers = nn.ModuleList([Decoder_Layer(head_n, hidden_dim, ffn_dim, dropout) for _ in range(layer_n)])
        
        self.linear = nn.Linear(hidden_dim, trg_voca_size)   # 최종 출력 layer
        
    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.embedding(trg)
        trg = self.position_embedding(trg)
        
        for decoder_layer in self.decoder_layers:
            trg, score = decoder_layer(src, trg, src_mask, trg_mask)
            
        output = self.linear(trg)
        
        return output, score