import torch
import torch.nn as nn
import torch.nn.functional as func

"""
    Transformer의 Multi-Head Attention을 이루는 Scaled Dot Product Attention 구현
"""

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention).__init__()
    
    def forward(self, Query, Key, Value, mask=None):  # Masked Multi Head Attention을 위해 mask여부 결정
        dk = torch.tensor(Query.shape[-1])  # Query와 Key의 차원 크기인 dk에 루트를 씌운값으로 QK_T를 Scaled
        QueryKey_T = torch.matmul(Query, Key.transpose(-1, -2))   # Query와Key의 전치행렬을 행렬곱셈함
        
        scaled = QueryKey_T / torch.sqrt(dk)    # softmax하기 전에 scale하기 위해 루트dk값으로 나눠줌
        
        if mask is not None:
            scaled.masked_fill_(mask==0, -1e14)     # 0인 부분을 아주 작은값인 -1e14로 바꾸어 attention score를 작게 만듦
        
        Att = func.softmax(scaled, dim=-1)      # softmax결과(attention score)
        output = torch.matmul(Att, Value)       # softmax한 값과 Value 곱셈
        
        return output, Att