import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ViT(nn.Module):
    def __init__(self, classes):
        super(ViT, self).__init__()
        self.patch_size = 28
        self.num_channels = 1
        self.embed_dim = 384
        self.num_heads = 4
        self.num_layers = 4
        self.mlp_ratio = 2
        self.num_classes = classes

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_size * self.patch_size * self.num_channels, self.embed_dim)
        )

        self.position_embedding = nn.Parameter(torch.zeros(1, (28 // self.patch_size) * (28 // self.patch_size) + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.transformer_blocks = nn.ModuleList(
            [ModifiedBlock(self.embed_dim, self.num_heads, self.mlp_ratio)] +
            [TransformerBlock(self.embed_dim, self.num_heads, self.mlp_ratio)
            for _ in range(self.num_layers - 1)])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim * self.mlp_ratio)
        self.fc2 = nn.Linear(self.embed_dim * self.mlp_ratio, self.num_classes)

    def forward(self, x):
        B, _, _, _ = x.shape

        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x[:, 0])
        x = nn.GELU()(self.fc1(x))
        x = self.fc2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attention = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_weights = self.attention((q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5))
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)

        return self.out_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.fc(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ModifiedBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(ModifiedBlock, self).__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = self.mha(x)
        x = x + self.mlp(self.norm2(x))
        return x
