import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import Data_Preprocess

def pad_tensor(batch):
    pad_idx = Data_Preprocess.pad_idx
    
    src = pad_sequence([torch.tensor(src).long() for src, _ in batch], batch_first=True, padding_value=pad_idx)
    trg = pad_sequence([torch.tensor(trg).long() for _, trg in batch], batch_first=True, padding_value=pad_idx)
    
    return src, trg
