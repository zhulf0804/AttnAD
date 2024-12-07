import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.fc_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, queries, keys, values, mask=None):
        """
        Args:
            queries (tensor):[batch_size, seq_len, embed_dim]
            keys (tensor): [batch_size, seq_len, embed_dim]
            values (tensor): [batch_size, seq_len, embed_dim]
            mask (tensor): [batch_size, seq_len, seq_len]
        Returns:
            out (tensor): [batch_size, seq_len, embed_dim]
        """
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        
        enery = torch.bmm(queries, keys.transpose(1, 2)) / (self.embed_dim ** 0.5)
        if mask is not None:
            enery = enery.mask_fill(mask == 0, float('-inf'))
        attention = torch.softmax(enery, dim=-1)
        
        out = torch.bmm(attention, values)
        out = self.fc_out(out)
        return out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 256
    seq_len = 10
    batch_size = 32

    self_attention = Attention(embed_dim).to(device)
    x = torch.rand((batch_size, seq_len, embed_dim)).to(device)
    out = self_attention(x, x, x)
    print(x.size(), out.size())
