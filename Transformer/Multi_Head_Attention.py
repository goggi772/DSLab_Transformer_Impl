import torch
import torch.nn as nn
import Scaled_Dot_Product_Attention

class Multi_Head_Attention(nn.Module):
    def __init__(self, heads_n, hidden_dim, dropout = 0.5):
        super(Multi_Head_Attention, self).__init__()
        
        assert hidden_dim % heads_n == 0    # Multi head를 적용하려면 hidden dimension이 head의 개수로 나누어 떨어져야만 함
        
        self.heads_n = heads_n
        self.head_dim = hidden_dim // heads_n
        
        self.Query_Weight = nn.Linear(hidden_dim, hidden_dim)   # Q, K, V의 가중치 
        self.Key_Weight = nn.Linear(hidden_dim, hidden_dim)
        self.Value_Weight = nn.Linear(hidden_dim, hidden_dim)
        
        self.attention = Scaled_Dot_Product_Attention()
        
        self.O_Weight = nn.Linear(hidden_dim, hidden_dim)   # 가중치 O
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Query, Key, Value, mask=None):
        batch_size = Query.shape[0]     # 입력 데이터의 첫번째 크기가 batch size이므로 Query의 shape[0]으로 batch size설정
        
        Query = self.Query_Weight(Query)    # 원래 Q, K, V의 shape은 [batch size, seq length, hidden dim]
        Key = self.Key_Weight(Key)
        Value = self.Value_Weight(Value)
        
        Query = Query.view(batch_size, -1, self.heads_n, self.head_dim)     # Q, K, V의 shape을 n개의 헤드수로 쪼개 [batch size, seq length, the number of head, hidden dim]으로 변경
        Key = Key.view(batch_size, -1, self.heads_n, self.head_dim)
        Value = Value.view(batch_size, -1, self.heads_n, self.head_dim)
        
        Query = Query.transpose(1, 2)   # [batch size, the number of head, seq length, hidden dim]
        Key = Key.transpose(1, 2)
        Value = Value.transpose(1, 2)
        
        output, score = self.attention(Query, Key, Value, mask)     # n개의 head수로 나눠진 Q, K, V 텐서를 scaled dot product attention을 함
        
        output = self.dropout(output)
        
        output = output.transpose(1, 2).contiguous.view(batch_size, -1, self.heads_n * self.head_dim)   # Multihead를 다시 원래대로 바꿈 [batch size, seq length, 전체차원]
        
        output = self.O_Weight(output)
        
        return output, score
        
        