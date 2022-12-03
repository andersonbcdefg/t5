import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def alibi_matrix(n, n_heads, causal):
    dists = np.array([[-abs(j - i) for i in range(n)] for j in range(n)])
    slopes = [2**-x for x in np.linspace(1, 8, n_heads)]
    alibi = torch.tensor(np.array([dists * slope for slope in slopes]), dtype=torch.float32)
    if causal:
        mask = torch.tril(torch.ones(n, n)).unsqueeze(0)
        alibi = alibi.masked_fill_(mask == 0, float("-inf"))
    return alibi.unsqueeze(0)

class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, test=False): # b l h d
        rms = torch.mean(X * X, dim=-1)**0.5
        out = X / rms.unsqueeze(-1)
        if test:
            return rms, out
        return out

class PreNormAndAdd(nn.Module):
    def __init__(self, sublayer):
        super().__init__()
        self.norm = RMSNorm()
        self.sublayer = sublayer

    def forward(self, X, encoder_out=None):
        if encoder_out is None:
            return X + self.sublayer(self.norm(X))
        else:
            return X + self.sublayer(self.norm(X), encoder_out)


class SelfAttention(nn.Module):
    def __init__(self, causal, seq_len, embed_dim, d_qkv, n_heads, dropout):
        super().__init__()
        self.d_qkv = d_qkv
        self.scale = d_qkv**0.5
        self.to_qkv = nn.Linear(embed_dim, n_heads * 3 * d_qkv)
        self.alibi_matrix = alibi_matrix(seq_len, n_heads, causal)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(n_heads * d_qkv, embed_dim)

    def forward(self, X): # b l h d
        Q, K, V = rearrange(self.to_qkv(X), "b l (h ddd) -> b l h ddd", ddd = 3 * self.d_qkv).chunk(3, dim=-1)
        attn_scores = (rearrange(Q, "b l h d -> b h l d") @ rearrange(K, "b l h d -> b h d l")) / self.scale + self.alibi_matrix
        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        attn_out = attn_weights @ rearrange(V, "b l h d -> b h l d")
        return self.resid_dropout(self.out_proj(rearrange(attn_out, "b h l d -> b l (h d)")))

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, d_qkv, n_heads, dropout):
        super().__init__()
        self.d_qkv = d_qkv
        self.scale = self.d_qkv**0.5
        self.to_q = nn.Linear(embed_dim, n_heads * d_qkv)
        self.to_kv = nn.Linear(embed_dim, n_heads * 2 * d_qkv)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(n_heads * d_qkv, embed_dim)

    def forward(self, X, encoder_out):
        Q = rearrange(self.to_q(X), "b l (h d) -> b l h d", d = self.d_qkv)
        K, V = rearrange(self.to_kv(encoder_out), "b l (h dd) -> b l h dd", dd = 2 * self.d_qkv).chunk(2, dim=-1)
        attn_scores = (rearrange(Q, "b l h d -> b h l d") @ rearrange(K, "b l h d -> b h d l")) / self.scale
        attn_weights = self.attn_dropout(F.softmax(attn_scores, dim=-1))
        attn_out = attn_weights @ rearrange(V, "b l h d -> b h l d") # b h l d
        out = self.resid_dropout(self.out_proj(rearrange(attn_out, "b h l d -> b l (h d)")))
        return X 

class FFN(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """
    def __init__(self, embed_dim, d_ff, dropout):
        super().__init__()
        self.w_proj = nn.Linear(embed_dim, d_ff, bias=False)
        self.v_proj = nn.Linear(embed_dim, d_ff, bias=False)
        self.out_proj = nn.Linear(d_ff, embed_dim, bias=False)

    def forward(self, X):
        return self.out_proj(F.gelu(self.w_proj(X)) * self.v_proj(X))


class EncoderBlock(nn.Module):
    def __init__(self, seq_len, embed_dim, d_qkv, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = PreNormAndAdd(SelfAttention(False, seq_len, embed_dim, d_qkv, n_heads, dropout))
        self.ffn = PreNormAndAdd(FFN(embed_dim, d_ff, dropout))

    def forward(self, X):
        return self.ffn(self.attn(X))


class DecoderBlock(nn.Module):
    def __init__(self, seq_len, embed_dim, d_qkv, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = PreNormAndAdd(SelfAttention(True, seq_len, embed_dim, d_qkv, n_heads, dropout))
        self.cross_attn = PreNormAndAdd(CrossAttention(embed_dim, d_qkv, n_heads, dropout))
        self.ffn = PreNormAndAdd(FFN(embed_dim, d_ff, dropout))
    
    def forward(self, X, encoder_out):
        return self.ffn(self.cross_attn(self.self_attn(X), encoder_out))

class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, enc_seq_len, n_enc_layers, dec_seq_len, n_dec_layers, 
                    embed_dim, d_qkv, n_heads, d_ff, dropout, tie_weights=True):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(*[
            EncoderBlock(enc_seq_len, embed_dim, d_qkv, n_heads, d_ff, dropout) for _ in range(n_enc_layers)
        ])
        self.encoder_final_norm = RMSNorm()
        self.decoder = nn.ModuleList([
            DecoderBlock(dec_seq_len, embed_dim, d_qkv, n_heads, d_ff, dropout) for _ in range(n_dec_layers)
        ])
        self.decoder_final_norm = RMSNorm()
        self.token_unembed = nn.Linear(embed_dim, vocab_size)
        if tie_weights:
            self.token_unembed.weight = self.token_embed.weight

    def forward(self, X, Y):
        X = self.token_embed(X)
        Y = self.token_embed(Y)
        encoder_out = self.encoder_final_norm(self.encoder(X))
        decoder_out = Y
        for layer in self.decoder:
            decoder_out = layer(decoder_out, encoder_out)
        return self.token_unembed(self.decoder_final_norm(decoder_out))