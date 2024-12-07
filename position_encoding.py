import math
import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, H, W, embed_dim):
        super().__init__()
        self.H = H
        self.W = W
        self.embed_dim = embed_dim

        self.row_embed = nn.Parameter(torch.randn(H, embed_dim // 2))
        self.col_embed = nn.Parameter(torch.randn(W, embed_dim // 2))
    
    def forward(self):
        row_embed = self.row_embed.unsqueeze(1).repeat(1, self.W, 1)
        col_embed = self.col_embed.unsqueeze(0).repeat(self.H, 1, 1)
        pe = torch.cat([col_embed, row_embed], dim=-1) # (H, W, embed_dim)
        return pe


## https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
## Non standardized implementation
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, H, W, embed_dim):
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError("Embed dimension must be divisible by 4 for 2D positional encoding")
        self.H = H
        self.W = W
        self.embed_dim = embed_dim

        # one-dimension
        # \frac{1}{10000^{2*i / d}} = exp^{log10000 \cdot (-2*i / d)} 
        #                           = exp^{-frac{log 1000}{d} \cdot 2*i}
        # 2i = 0, 2, ..., d - 2

        div_term = torch.exp(
            torch.arange(0, embed_dim // 2, 2, dtype=torch.float32) * (-math.log(10000.0) / (embed_dim // 2))
        )
        y_pos = torch.arange(0, H, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(0, W, dtype=torch.float32).unsqueeze(0).repeat(H, 1)

        pe_y = torch.zeros(H, W, embed_dim // 2)
        pe_y[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe_y[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        pe_x = torch.zeros(H, W, embed_dim // 2)
        pe_x[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe_x[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)
        self.pe = torch.cat([pe_y, pe_x], dim=-1)

    
    def forward(self):
        return self.pe


if __name__ == '__main__':
    H, W = 32, 32
    embed_dim = 256

    learnable_pos_encoding_layer = LearnablePositionalEncoding(H, W, embed_dim)
    learnable_pos_encoding = learnable_pos_encoding_layer()
    print(learnable_pos_encoding.size())

    sin_pos_encoding_layer = SinusoidalPositionalEncoding(H, W, embed_dim)
    sin_pos_encoding = sin_pos_encoding_layer()
    print(sin_pos_encoding.size())
