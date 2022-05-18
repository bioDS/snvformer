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

class FlattenedOutput(nn.Module):
    def __init__(self, embed_dim, seq_len, num_phenos) -> None:
        super().__init__()
        dense = nn.Linear(embed_dim*(seq_len+num_phenos), 2)
        self.softmax = nn.Softmax(1)
        self.final_layer = nn.Sequential(dense, self.softmax)

    def forward(self, enc_out):
        enc_out = enc_out.view(enc_out.shape[0], -1)
        return self.final_layer(enc_out)

class TokenOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # dense = nn.Linear(embed_dim, 2)
        self.softmax = nn.Softmax(1)
        # self.final_layer = nn.Sequential(dense, self.softmax)

    def forward(self, enc_out):
        cls_tok = enc_out[:,0,:]
        return self.softmax(cls_tok)

# pre-trainable model
class Encoder(nn.Module):
    def __init__(self, seq_len, num_phenos, max_seq_pos, embed_dim, num_heads, num_layers, vocab_size, batch_size, device, use_linformer=False, linformer_k=16) -> None:
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
            new_block = TransformerBlock(seq_len+num_phenos, embed_dim, num_heads, vocab_size, batch_size, device, use_linformer, linformer_k)
            self.blocks.append(new_block)
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, phenos, x, pos):
        # ex = self.embedding(x.t()).swapaxes(0,1)
        ex = self.embedding(x)
        if (ex.shape[2] != 32):
            print("ex.shape[2]: {}".format(ex.shape[2]))
        ep = self.pos_encoding(torch.zeros([x.shape[0], x.shape[1], self.pos_size], device=x.device), pos)
        phenos = torch.unsqueeze(phenos, 2).expand(-1,-1, self.pos_size + ex.shape[2])
        at = torch.cat([ex, ep], dim=2)
        at = torch.cat([phenos, at], dim=1)
        for block in self.blocks:
            at = block(at)
        return at

# Fine-tunable full model
class TransformerModel(nn.Module):
    def __init__(self, seq_len, num_phenos, max_seq_pos, embed_dim, num_heads, num_layers, vocab_size, batch_size, device, output_type, use_linformer=False, linformer_k=16) -> None:
        super().__init__()
        self.encoder = Encoder(seq_len, num_phenos, max_seq_pos, embed_dim, num_heads, num_layers, vocab_size, batch_size, device, use_linformer, linformer_k)

        # self.output_type = output_type
        # self.true = tensor([0.0,1.0])
        if output_type == 'tok':
            self.output = TokenOutput()
        elif output_type == 'binary':
            self.output = FlattenedOutput(embed_dim, seq_len, num_phenos)
        else:
            raise ValueError("output_type must be 'binary', or 'tok'")

    def forward(self, phenos, x, pos):
        enc_out = self.encoder(phenos, x, pos)
        return self.output(enc_out)