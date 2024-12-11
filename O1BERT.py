import math
import torch
import torch.nn as nn
import torch.optim as optim

############################################
# BERT Model From Scratch (Educational)    #
############################################

# -------------------------
# Configuration Parameters
# -------------------------
vocab_size = 30522  # typical BERT vocab size (You can reduce this if you want)
max_seq_len = 64  # maximum sequence length
hidden_size = 256  # hidden embedding size (usually BERT uses 768, but we use smaller for demo)
num_heads = 4  # number of attention heads (BERT Base uses 12)
num_layers = 4  # number of transformer layers (BERT Base uses 12)
ffn_hidden = 4 * hidden_size  # feed-forward intermediate size (BERT uses 4x hidden)
dropout_prob = 0.1
lr = 1e-4
batch_size = 16
num_iterations = 100  # small number just to illustrate training steps


# -------------------------
# Embeddings
# -------------------------
class BERTEmbeddings(nn.Module):
    """
    Combines token embeddings, positional embeddings, and segment embeddings
    into a single embedding matrix. This closely follows what BERT does:
    output = LayerNorm( Dropout( TokenEmbedding + PositionalEmbedding + SegmentEmbedding ) )
    """

    def __init__(self, vocab_size, hidden_size, max_position_embeddings=512, type_vocab_size=2, dropout=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# -------------------------
# Multi-Head Attention
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    This implements multi-head attention from scratch.
    We have W_q, W_k, W_v for each head. We then compute attention and
    concatenate heads back together.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be divisible by number of heads.")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Projection matrices for query, key, value
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Final linear layer after concatenation
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, hidden_dim = hidden_states.size()

        # project hidden states to queries, keys, values
        Q = self.query(hidden_states)  # [B, L, H]
        K = self.key(hidden_states)  # [B, L, H]
        V = self.value(hidden_states)  # [B, L, H]

        # reshape into [B, num_heads, L, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # apply attention mask (if any)
        if attention_mask is not None:
            # attention_mask should be broadcastable to [B, num_heads, L, L]
            # typically masks use -inf for masked positions
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # [B, num_heads, L, head_dim]

        # concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.out(context)
        output = self.dropout(output)
        return output


# -------------------------
# Feed Forward Network
# -------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, ffn_hidden, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()  # BERT uses GELU

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# -------------------------
# Transformer Encoder Layer
# -------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_hidden, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.layernorm1 = nn.LayerNorm(hidden_size, eps=1e-12)

        self.ffn = PositionwiseFeedForward(hidden_size, ffn_hidden, dropout)
        self.layernorm2 = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, x, attention_mask=None):
        # Self-attention block
        attn_out = self.self_attn(x, attention_mask=attention_mask)
        x = x + attn_out
        x = self.layernorm1(x)

        # Feed Forward block
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.layernorm2(x)
        return x


# -------------------------
# BERT Encoder (Stack of Layers)
# -------------------------
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, ffn_hidden, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size, hidden_size, max_position_embeddings=max_seq_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(hidden_size, num_heads, ffn_hidden, dropout) for _ in range(num_layers)]
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # attention_mask should be of shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        # In BERT, attention masks are typically additive masks with 0 for keep and -inf for masked positions.
        embedding_output = self.embeddings(input_ids, token_type_ids)
        x = embedding_output
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


# -------------------------
# BERT For Masked LM
# -------------------------
class BERTForMaskedLM(nn.Module):
    """
    BERT with a masked language modeling head on top.
    The MLM head is typically:
    - Take final hidden states
    - Apply layer norm and maybe a linear projection
    - Use a matrix tied with the word embeddings for prediction
    """

    def __init__(self, encoder, vocab_size, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.mlm_classifier = nn.Linear(hidden_size, vocab_size, bias=False)
        # Tie the classifier weight with the embedding layer weight
        self.mlm_classifier.weight = self.encoder.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output = self.encoder(input_ids, token_type_ids, attention_mask)
        # Apply layer norm before predictions (some implementations do a prediction head transform)
        sequence_output = self.LayerNorm(sequence_output)
        prediction_scores = self.mlm_classifier(sequence_output)
        return prediction_scores


# -------------------------
# Training Example
# -------------------------
# In a real scenario, you'd prepare a dataset of text, tokenize it,
# create input_ids with masks, and so forth. Here, we'll create dummy data.

def create_dummy_data(batch_size, seq_len, vocab_size, mask_prob=0.15):
    """
    Creates dummy input_ids and masked language model targets.
    We'll randomly choose tokens. Then we randomly mask some of them.
    input_ids: [batch_size, seq_len]
    labels: same shape, with -100 for unmasked positions (per PyTorch convention).
    """
    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    # Create a mask of booleans that indicate which positions to mask
    mask = torch.rand(size=(batch_size, seq_len)) < mask_prob
    labels = input_ids.clone()
    # Set unmasked positions to -100 so they're not included in the loss
    labels[~mask] = -100
    # Replace masked positions in input_ids with the [MASK] token id
    # Here we pretend [MASK] = 103 (BERT's actual mask token),
    # but it doesn't matter much in this dummy example
    MASK_ID = 103
    input_ids[mask] = MASK_ID

    # In a real scenario, you'd also handle special tokens, token_type_ids, etc.
    token_type_ids = torch.zeros_like(input_ids)
    # Construct an attention mask (no padding here, so just all ones)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, token_type_ids, attention_mask, labels


# Initialize model
model = BERTForMaskedLM(
    BERTEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_seq_len,
        dropout=dropout_prob),
    vocab_size=vocab_size,
    hidden_size=hidden_size
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()  # Cross-entropy for MLM

# Simple training loop on dummy data
model.train()
for step in range(num_iterations):
    input_ids, token_type_ids, attention_mask, labels = create_dummy_data(batch_size, max_seq_len, vocab_size)
    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    # Convert attention_mask to a form suitable for BERT:
    # BERT expects additive attention mask [B,1,1,L]
    # We'll use 0.0 for keep and -inf for masked positions
    extended_attention_mask = (attention_mask.unsqueeze(1).unsqueeze(2)).float()
    # 1.0 means keep, 0 means masked position
    # Convert to additive mask: 0 -> -inf
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    optimizer.zero_grad()
    prediction_scores = model(input_ids, token_type_ids, extended_attention_mask)
    # prediction_scores: [B, L, vocab_size]
    loss = criterion(prediction_scores.view(-1, vocab_size), labels.view(-1))
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

print("Training complete (demo). Model parameters have been updated.")
