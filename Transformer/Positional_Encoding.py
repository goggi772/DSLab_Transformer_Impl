import torch
import torch.nn as nn

class Positional_Encoding(nn.Module):
    def __init__(self, max_len, hidden_dim, device):
        super(Positional_Encoding, self).__init__()
        
        self.encoding = torch.zeros(max_len, hidden_dim)   # 시퀀스의 최대 길이와 차원크기인 hidden_dim크기를 가지는 텐서
        position = torch.arange(max_len).view(1, -1)    # pos 텐서
        index = torch.arange(0, hidden_dim, step=2)     # 2i를 hidden_dim로 나누어 줘야함 그래서 step이 2
        
        self.encoding[:, 0::2] = torch.sin(position / 10000 ** (index / hidden_dim)).to(device)    # index가 2i일때 sin함수로 위치 생성
        self.encoding[:, 1::2] = torch.cos(position / 10000 ** (index / hidden_dim)).to(device)    # index가 2i+1일때 cos함수로 위치 생성
        
        
    def forward(self, x):
        seq_len = x.shape[1]    # 데이터 x -> [배치 사이즈, 시퀀스 길이, 전체 차원]
        return x + self.encoding[:seq_len, :]   # 입력된 시퀀스의 길이만큼만 encoding에서 가져와 더함
        
        
        