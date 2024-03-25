import torch

class Data_Preprocess:
    pad_idx = 0
    unk_idx = 1
    sos_idx = 2
    eos_idx = 3
    def __init__(self, src_tokenizer, trg_tokenizer, src_tok2idx, trg_tok2idx, src_idx2tok, trg_idx2tok):
        self.src_tokenizer = src_tokenizer      # spacy의 tokenizer를 입력받음
        self.trg_tokenizer = trg_tokenizer
        
        self.src_tok2idx = src_tok2idx      # vocab에 저장된 token to index
        self.trg_tok2idx = trg_tok2idx
        
        self.src_idx2tok = src_idx2tok      # vocav에 저장된 index to token
        self.trg_idx2tok = trg_idx2tok
        
    def src_encode(self, src_text):     # text로 표현된 입력을 index로 인코딩하여 리턴하는 함수
        src = [self.src_tok2idx.get(token.text, Data_Preprocess.unk_idx) for token in self.src_tokenizer.tokenizer(src_text)]
        return src
    
    def trg_encode(self, trg_text):
        trg = [self.trg_tok2idx['<sos>']] \
            + [self.trg_tok2idx.get(token.text, Data_Preprocess.unk_idx) for token in self.trg_tokenizer.tokenizer(trg_text)] \
                + [self.trg_tok2idx['<eos>']]
        return trg
    
    def src_decode(self, src_idx):      # index로 표현된 입력을 text로 디코딩하여 리턴하는 함수
        src = list(map(lambda x: self.src_idx2tok[x], src_idx))
        return " ".join(src)
    
    def trg_decode(self, trg_idx):
        trg = list(map(lambda x: self.trg_idx2tok[x], trg_idx))[1:-1]   # <sos>, <eos>토큰은 제외하고 디코딩
        return " ".join(trg)