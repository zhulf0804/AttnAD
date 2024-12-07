import torch
import torch.nn as nn


class BEVQueryGenerator(nn.Module):
    def __init__(self, bev_h, bev_w, embed_dim):
        super(BEVQueryGenerator, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dim = embed_dim
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dim))

    def forward(self):
        return self.bev_queries


if __name__ == '__main__':
    bev_h, bev_w = 256, 256
    embed_dim = 128
    bev_query_generator = BEVQueryGenerator(bev_h, bev_w, embed_dim)
    bev_query = bev_query_generator()
    print(bev_query.size())
