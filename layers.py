import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class PreNormAndAdd(nn.Module):
    def __init__(self, embed_dim, sublayer):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sublayer = sublayer

    def forward(self, X):
        return X + self.sublayer(self.norm(X))

class Attention(nn.Module):
    def __init__(self, embed_dim, d_kv, n_heads, attn_mask=None, dropout=0.1):
        super().__init__()
        self.d_kv = d_kv
        self.to_qkv = nn.Linear(embed_dim, n_heads * 3 * d_kv)

    def forward(self, X):
        Q, K, V = rearrange(self.to_qkv(X), "b l (h ddd) -> b l h ddd", ddd = 3 * self.d_kv).chunk(3, dim=-1)
        attn_scores = torch.einsum("bkhd, blhd -> bhkl", Q, K) / self.d_kv ** 0.5
        if attn_mask == "causal":
            mask = torch.tril(torch.ones(attn_scores.shape[-1], attn_scores.shape[-1])).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill_(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)