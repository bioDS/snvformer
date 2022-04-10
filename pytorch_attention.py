import torch as torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from torch import Tensor

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class D2LAdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(D2LAdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of `queries`: (`batch_size`, no. of
        # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        # no. of key-value pairs, `num_hiddens`). Sum them up with
        # broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of `self.w_v`, so we remove the last
        # one-dimensional entry from the shape. Shape of `scores`:
        # (`batch_size`, no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

class D2LDotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(D2LDotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        # print("query shape {}".format(queries.shape))
        # print("keys  shape {}".format(keys.transpose(1,2).shape))
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        # print("scores shape: {}".format(scores.shape))
        quit
        self.attention_weights = masked_softmax(scores, valid_lens)
        self.attention_weights.requires_grad_()
        self.attention_weights.retain_grad()
        return torch.bmm(self.dropout(self.attention_weights), values)
    # def lrp_forward(self, queries, keys, values, valid_lens=None):
    #     d = queries.shape[-1]
    #     # Set `transpose_b=True` to swap the last two dimensions of `keys`
    #     scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
    #     if self.use_sparsemax:
    #         self.attention_weights = masked_sparsemax(scores, valid_lens)
    #     else:
    #         self.attention_weights = masked_softmax(scores, valid_lens)
    #     return torch.bmm(self.dropout(self.attention_weights).detach(), values)

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads."""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class LinformerAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, embed_dim, seq_len, linform_k,
                 num_heads, dropout, device='cpu', bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = D2LDotProductAttention(dropout)
        # self.attention = D2LAdditiveAttention(key_size, query_size, num_hiddens, dropout)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        # Linformer components
        self.E_i = nn.Linear(seq_len, linform_k, device=device)
        self.F_i = nn.Linear(seq_len, linform_k, device=device)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Linformer approximation
        keys = self.E_i(keys.swapaxes(1,2)).swapaxes(1,2)
        values = self.F_i(values.swapaxes(1,2)).swapaxes(1,2)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)

        output_concat = transpose_output(output, self.num_heads)
        return self.attention.attention_weights, self.W_o(output_concat)

    # def lrp_forward(self, queries, keys, values, valid_lens):
    #     queries = transpose_qkv(self.W_q(queries), self.num_heads)
    #     keys = transpose_qkv(self.W_k(keys), self.num_heads)
    #     values = transpose_qkv(self.W_v(values), self.num_heads)

    #     # Linformer approximation
    #     keys = self.E_i(keys.swapaxes(1,2)).swapaxes(1,2)
    #     values = self.F_i(values.swapaxes(1,2)).swapaxes(1,2)

    #     if valid_lens is not None:
    #         valid_lens = torch.repeat_interleave(
    #             valid_lens, repeats=self.num_heads, dim=0)

    #     output = self.attention.lrp_forward(queries, keys, values, valid_lens)

    #     output_concat = transpose_output(output, self.num_heads)
    #     return self.attention.attention_weights, self.W_o(output_concat)

class D2LMultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, embed_dim,
                 num_heads, dropout, device='cpu', bias=False, **kwargs):
        super(D2LMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = D2LDotProductAttention(dropout)
        # self.attention = D2LAdditiveAttention(key_size, query_size, num_hiddens, dropout)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias, device=device)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.attention.attention_weights, self.W_o(output_concat)

# need at least log(n) hidden units to encode a sequence of length d
# d2l suggests using the same dimensions as the word embedding
# either make sure word embedding is large enough, or make this larger.
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    # def __init__(self, num_hiddens, dropout, max_len=1000):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
    
class AddNorm(nn.Module):
    """Add residual then normalise"""
    def __init__(self, normalized_shape, device='cpu', dropout=0.0, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape, device=device)
        
    def forward(self, residual, X):
        return self.ln(self.dropout(X) + residual)

    def lrp_forward(self, residual, X):
        # return self.ln(self.dropout(X) + residual)
        mean = torch.mean(X)
        X = X - mean
        var = torch.var(X)
        eps = 1e-5
        sqrt = torch.sqrt(eps + var).detach()
        return X/sqrt

    
class PositionWiseFFN(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, device='cpu') -> None:
        super().__init__()
        self.dense1 = nn.Linear(num_in, num_hidden, device=device) 
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(num_in, num_hidden, device=device)
    
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))