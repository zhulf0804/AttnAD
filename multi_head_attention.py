import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding size needs to be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, values, mask=None):
        """
        Args:
            queries (tensor):[batch_size, query_len, embed_dim]
            keys (tensor): [batch_size, key_len, embed_dim]
            values (tensor): [batch_size, key_len, embed_dim]
            mask (tensor): [batch_size, query_len, key_len]
        Returns:
            out (tensor): [batch_size, query_len, embed_dim]
        """
        N = queries.shape[0] # batch size
        query_len, key_len, value_len = queries.shape[1], keys.shape[1], values.shape[1]

        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)

        queries = queries.view(N, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(N, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(N, value_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(queries, keys.transpose(3, 2)) / (self.head_dim ** 0.5) # (N, num_heads, query_len, key_len)
        if mask is not None:
            # mask shape: [N, 1, 1, key_len] or [N, 1, query_len, key_len]
            # Expand mask if necessary
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, key_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [N, 1, query_len, key_len]
            # Masking: fill positions with -inf where mask == 0
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1)
        attention = self.attention_dropout(attention)

        out = torch.matmul(attention, values) # (N, num_heads, query_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.embed_dim)
        out = self.fc_out(out)
        out = self.proj_dropout(out)
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 256
    query_len = 10
    key_len = 20
    num_heads = 8
    batch_size = 32
    dropout_rate = 0.1

    multi_head_attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate).to(device)
    x = torch.rand((batch_size, query_len, embed_dim)).to(device)
    y = torch.rand((batch_size, key_len, embed_dim)).to(device)
    out = multi_head_attention(x, y, y)
    print(x.size(), out.size())
