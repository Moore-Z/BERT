import math

import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocabu_size: int):
     super().__init__()
     self.d_model = d_model
     self.vocabu_size = vocabu_size
     self.embedding = nn.Embedding(vocabu_size,d_model)

# nn.Embedding is not a matrix multiplication operation but
# rather an index lookup operation. Let's clarify the details:
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len : int, dropout: float) -> None:
        super().__init__(self)
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(self.seq_len,self.d_model)
        # Create a vector of shape
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))
        #Apply the sin to even pos and cos to odd pos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, Seq_len, d_model)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float=10**6):
        super().__init__(self)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / math.sqrt(variance**2 + self.eps) + self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout:float)->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu_(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float, h: int):
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not dividable by h"
        self.d_k = d_model // h
        self.attention_score = None
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        attention_score = (query @ key.transpose(-2,-1)) * math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_score = attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score

    def forward (self, q, k, v, mask):
        query = self.w_q(q) # q (Batch, Seq_Len, d_model) Multiply (d_model, d_model) ==> (Batch, Seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, Seq_Len, d_model) ==> (Batch, Seq_Len, h (Number of Head), d_k(number of row per head) )
        # => (Batch, h, Seq_len, d_K)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # x = (Batch, h, Seq_len, d_K)
        x, self.attention_score = self.attention(query, key, value, mask, self.dropout)
        # (Batch, h, Seq_len, d_K)  ==> (Batch, Seq_Len, d_model) 这里的-1 是placeholder ，告诉了view method
        # 我最后一位的是要d_model 的length， 自己给我算这一位的size
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(self.norm(sublayer(x)))

class EncoderBlock(nn.Module):
    def __init__(self, feedforward: FeedForwardBlock, self_attention_block: MultiHeadAttentionBlock, dropout: float)->None:
        super().__init__()
        self.feedForward = feedforward
        self.self_attention_block = self_attention_block
        self.residualconnects = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))

    def forward(self, x, src_mask):
        # The lambda is used here to delay the evaluation of the function call
        # This x is the residual connect block's parameter
        x = self.residualconnects[0](x, lambda x : self.self_attention_block(x, x, x, src_mask))
        x = self.residualconnects[1](x, lambda x : self.feedForward(x))
        return x

class Encoder(nn.ModuleList):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)