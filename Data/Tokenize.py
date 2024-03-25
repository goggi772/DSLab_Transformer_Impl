import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import Multi30k
from functools import partial
from torchtext.vocab import build_vocab_from_iterator
import spacy

import random
import numpy as np
import math

SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.deterministic = True

train, valid, test = Multi30k()

en_tokenizer = spacy.load('en_core_web_sm')
de_tokenizer = spacy.load('de_core_news_sm')

# 입력된 text를 tokenize함
def tokenize(tokenizer, text):
    return [token.text for token in tokenizer.tokenizer(text)]

specials = ['<pad>', '<unk>', '<sos>', '<eos>']

en_vocab = build_vocab_from_iterator(map(partial(tokenize, en_tokenizer), [eng for _, eng in train]), min_freq=2, specials=specials)
de_vocab = build_vocab_from_iterator(map(partial(tokenize, de_tokenizer), [de for de, _ in train]), min_freq=2, specials=specials)
# build_vocav_from_iterator - dictionary를 만듦
# (tokenize된 배열의 dic, min_req=최소 빈도수/2개 이하로 나온토큰은 저장x, specials=special토큰 추가)
# partial = en_tokenizer, de_tokenizer를 인자로 받은 새로운 tokenize 함수 생성


print(en_vocab.get_itos()[:10])
print(de_vocab.get_itos()[:10])

