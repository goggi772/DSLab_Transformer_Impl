import torch
from torch.utils.data import Dataset

class CustomDatasets(Dataset):
    def __init__(self, data, data_preprocess):
        self.data = data
        self.data_preprocess = data_preprocess
        self.sentence = self.preprocess()
        
    def preprocess(self):
        return [(self.data_preprocess.src_encode(de), self.data_preprocess.trg_encode(en))
                    for de, en in self.data if len(en) > 0 and len(de) > 0]
    
    def src_max_length(self):
        return max([len(src) for src, _ in self.sentence])
    
    def trg_max_length(self):
        return max([len(trg) for _, trg in self.sentence])
    
    def __getitem__(self, idx):
        return self.sentence[idx]
    
    def __len__(self):
        return len(self.sentence)