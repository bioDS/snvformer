import torch
from torch import nan_to_num_, nn, tensor
from pytorch_attention import *

class TransformerBlock(nn.Module):
    def __init__(self, seq_len, embed_dim, num_heads, vocab_size, batch_size, device, use_linformer, linformer_k) -> None:
        super(TransformerBlock, self).__init__()
        if use_linformer:
            self.attention = LinformerAttention(embed_dim, seq_len, linformer_k, num_heads, dropout=0.05, device=device)
        else:
            self.attention = D2LMultiHeadAttention(embed_dim, num_heads, dropout=0.05, device=device)
        self.addnorm = AddNorm(embed_dim)
        self.positionwise_ffn = PositionWiseFFN(embed_dim, embed_dim, embed_dim)

    def forward(self, X):
        """X is (batch_size, seq_len, embed_dim)"""
        residual1 = X
        # print("X shape: {}".format(X.shape))
        # weights is the 'A' matrix (after softmax but before \cdot V)
        weights, at = self.attention(X, X, X, valid_lens=None)
        self.A = weights
        ra_out = self.addnorm(at, residual1)

        residual2 = ra_out
        ffn_out = self.positionwise_ffn(ra_out)
        return self.addnorm(ffn_out, residual2)

class One_Hot_Embedding:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        pass

    def embed(self, t: tensor):
        # new_t = t.long().to(t.device)
        # return torch.nn.functional.one_hot(new_t)
        return torch.nn.functional.one_hot(t.long(), num_classes=self.num_classes)

class TransformerModel(nn.Module):
    def __init__(self, seq_len, max_seq_pos, embed_dim, num_heads, num_layers, vocab_size, batch_size, device, output_type, use_linformer=False, linformer_k=16) -> None:
        super().__init__()
        self.device = device
        # self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.pos_size = embed_dim - vocab_size
        if self.pos_size % 2 == 1:
            self.pos_size = self.pos_size - 1
        one_hot_embed_size = embed_dim - self.pos_size
        embedding = One_Hot_Embedding(one_hot_embed_size)
        self.embedding = embedding.embed
        self.pos_encoding = ExplicitPositionalEncoding(self.pos_size, max_len=max_seq_pos+1)
        self.blocks = []
        for _ in range(num_layers):
            new_block = TransformerBlock(seq_len, embed_dim, num_heads, vocab_size, batch_size, device, use_linformer, linformer_k)
            self.blocks.append(new_block)
        self.blocks = nn.ModuleList(self.blocks)
        self.output_type = output_type
        self.true = tensor([0.0,1.0])
        if output_type == 'tok':
            dense = nn.Linear(embed_dim, 2)
            self.softmax = nn.Softmax(1)
            self.final_layer = nn.Sequential(dense, self.softmax)
        elif output_type == 'binary':
            dense = nn.Linear(embed_dim*seq_len, 2)
            self.softmax = nn.Softmax(1)
            self.final_layer = nn.Sequential(dense, self.softmax)
            # pass
        elif output_type == "continuous":
            # dense1 = nn.Linear(num_hiddens, 1, device=device)
            # dense2 = nn.Linear(seq_len, 1, device=device)
            # self.final_layer = nn.Sequential(dense1, dense2)
            self.final_layer = nn.Linear(seq_len*embed_dim, 1)
        else:
            raise ValueError("output_type must be 'binary' or 'continuous'")

    def forward(self, x, pos):
        # ex = self.embedding(x.t()).swapaxes(0,1)
        ex = self.embedding(x)
        ep = self.pos_encoding(torch.zeros([x.shape[0], x.shape[1], self.pos_size], device=x.device), pos)
        at = torch.cat([ex, ep], dim=2)
        for block in self.blocks:
            at = block(at)
        if self.output_type == "tok":
            # cls_tok = at[:,0,0]
            cls_tok = at[:,0,:]
            # return cls_tok # allows classifications > 1 (and we tend to only get these)
            return self.softmax(cls_tok)
            # cls_tok = torch.mean(at[:,0,:], dim=1)
            # return self.final_layer(cls_tok)
            # return torch.outer(cls_tok, self.true)
        else:
            at = at.view(ex.shape[0], -1)
            return self.final_layer(at)
