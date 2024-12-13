import re
from random import randrange, randint
import math
from random import random
from random import shuffle

from torch.nn.functional import embedding

from O1BERT import batch_size, max_seq_len
import torch
from torch import nn

text = (
       'Hello, how are you? I am Romeo.n'
       'Hello, Romeo My name is Juliet. Nice to meet you.n'
       'Nice meet you too. How are you today?n'
       'Great. My baseball team won the competition.n'
       'Oh Congratulations, Julietn'
       'Thanks you Romeo'
   )
d_model = 512
batch_size = 100
token_list = []
max_pred = 20
# re.sub substitute, .lower() lower case, split
sentences = re.sub("[.,!?-]",'',text.lower()).split('n')
word_list = list(set(" ".join(sentences).split()))

word_dict = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
for i, w in enumerate(word_list):
    word_dict[w] = i
number_dict = {i : w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)



def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2 :
        tokens_a_index, tokens_b_index = randrange(len(sentences)),  randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]

        # input_ids: The model uses this sequence as the input, representing the actual tokens of the combined sentences.
        # word_dict['[CLS]'] = 1 and word_dict['[SEP]'] = 2
        input_ids = word_dict['[CLS]'] + tokens_a + word_dict['[SEP]'] + tokens_b + word_dict['[SEP]']
        # segment_ids: This tells the model which parts of the input belong to Sentence A (0) and which belong to Sentence B (1).
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (1 + len(tokens_b) + 1)

        # MASK IN
        # # 15 % of tokens in one sentence
        n_pred = min(max_pred, max(1,int(round(len(input_ids) * 0.15))))
        # Grade position
        # OK this i is index
        cand_made_pos = [ i for i, token in enumerate(input_ids)
                            if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_made_pos)
        masked_token, masked_pos = [], []
        for pos in cand_made_pos[:n_pred]:
            masked_pos.append(pos)
            masked_token.append(cand_made_pos[pos])
            if random() < 0.8:
                input_ids[pos] = word_dict['[MASK]']
            elif random() < 0.5:
                index = randint(0,vocab_size-1)
                input_ids[pos] = word_dict[number_dict[index]]

        # maxlen = 128(maximum sequence length)
        # Zero Padding
        # maxLen (Maximum Sequence length > Input Text Value)
        n_pad = maxlen - len(input_ids)
        # Fill up the masked Token List with [0]
        input_ids.extend([0]*n_pad)
        # segment_ids used to have position for each words
        segment_ids.extend([0]*n_pad)

        # max_pred (maximum token to mask) > masked token list
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            # Add masked_token with missing length
            masked_token.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2 :
            batch.append([input_ids,segment_ids, masked_token, masked_pos, True])
            positive += 1
            # IsNext : Sentence B directly follows sentence A in the corpus.

        elif tokens_a_index + 1 != tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_token, masked_pos, False])
            negative += 1
            # NotNext : Sentence B is randomly selected and does not follow sentence A.




    # Embedding layers
    class Embedding(nn.Module):
        def __init__(self, vocab_size, maxlen,n_segments):
            super(Embedding, self).__init__()
            self.tok_embedding = nn.Embedding(vocab_size, d_model)
            self.pos_embedding = nn.Embedding(maxlen, d_model)
            self.seg_embedding = nn.Embedding(n_segments, d_model)

        def forward(self, x, seg):
            seq_len = x.size(1)
            pos = torch.arange(seq_len, dtype = torch.long)
            pos = pos.unsqueeze(0).expand_as(x)
            embedding = self.tok_embedding(x) + self.pos_embedding(pos) + self.seg_embedding(seg)
            return embedding


    # Attention Mask
    # Encoder layer
    #     Multi-head attention
    #         Scaled dot product attention
    #     Position-wise feed-forward network
    # BERT (assembling all the components)



