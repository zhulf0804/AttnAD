import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x (tensor): [batch_size, query_len, embed_dim]
        Returns
            out (tensor): [batch_size, query_len, embed_dim]
        """
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout_rate):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim, dropout_rate)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)


    def forward(self, x):
        """
        Args:
            x (tensor): [batch_size, query_len, embed_dim]
        Returns
            out (tensor): [batch_size, query_len, embed_dim]
        """
        attn_out = self.attn(x, x, x)
        out = x + self.dropout1(attn_out)
        out = self.layernorm1(out)

        ffn_out = self.ffn(out)
        out = out + self.dropout2(ffn_out)
        out = self.layernorm2(out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout_rate, depth):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout_rate) 
            for _ in range(depth)]
        )
    
    def forward(self, x):
        """
        Args:
            x (tensor): [batch_size, query_len, embed_dim]
        Returns
            out (tensor): [batch_size, query_len, embed_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim =768
    num_heads = 12 
    ffn_dim = 3072
    dropout_rate = 0.1
    depth = 6

    batch_size = 6
    query_len = 1024

    transformer_encoder = TransformerEncoder(embed_dim, num_heads, ffn_dim, dropout_rate, depth).to(device)
    x = torch.randn(batch_size, query_len, embed_dim).to(device)

    out = transformer_encoder(x)
    print(x.size(), out.size())
