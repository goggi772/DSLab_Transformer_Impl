import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.src_pad_idx = src_pad_idx      # source에 mask를 적용하는 index
        self.trg_pad_idx = trg_pad_idx      # target에 mask를 적용하는 index
        
        self.device = device
        
    def create_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src = [batch size, 1, 1, src len]
        
        return src_mask
    
    def create_trg_mask(self, trg):
        # trg = [batch size, trg length]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg = [batch size, 1, 1, trg length]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_len * trg_len크기의 텐서를 생성하고(torch.ones) 아래 삼각 행렬을 모두 0으로 만든(torch.tril) 후 bool로 변환
        # trg = [trg length, trg length]
        trg_mask = trg_pad_mask & trg_sub_mask
        # trg mask = [batch size, 1, trg length, trg length]
        return trg_mask
    
    def forward(self, src, trg):
        # src = [batch size, src length]
        # trg = [batch size, trg length]
        src_mask = self.create_src_mask(src)
        trg_mask = self.create_trg_mask(trg)
        # src mask = [batch size, 1, 1, src length]
        # trg mask = [batch size, 1, trg length, trg length]
        
        en_src = self.encoder(src, src_mask)
        # en_src = [batch size, src len, hiden dim]
        
        output, score = self.decoder(trg, en_src, trg_mask, src_mask)
        # output = [batch size, trg length, output dim]
        # score = [batch size, n heads, trg length, src length]
        
        return output, score
        
        
        
        
        