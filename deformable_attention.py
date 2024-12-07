import torch
import torch.nn as nn
import torch.nn.functional as F


# SingleScaleDeformableAttention
class DeformableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_points):
        super(DeformableAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, query, reference_points, value):
        """
        query: [batch_size, num_queries, embed_dim]
        reference_points: [batch_size, num_queries, 2], values in [0, 1], representing (x, y)
        value: [batch_size, H, W, embed_dim]
        spatial_shapes: (H, W)
        """
        batch_size, num_queries, _ = query.shape
        H, W = value.shape[1], value.shape[2]
        N = batch_size

        value = self.value_proj(value.view(batch_size, H * W, self.embed_dim))  # [N, H*W, embed_dim]
        value = value.view(N, H, W, self.embed_dim).permute(0, 3, 1, 2)  # [N, embed_dim, H, W]

        sampling_offsets = self.sampling_offsets(query)  # [N, num_queries, num_heads * num_points * 2]
        sampling_offsets = sampling_offsets.view(N, num_queries, self.num_heads, self.num_points, 2)

        attention_weights = self.attention_weights(query)  # [N, num_queries, num_heads * num_points]
        attention_weights = attention_weights.view(N, num_queries, self.num_heads, self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1)

        reference_points = reference_points.unsqueeze(2).unsqueeze(3)  # [N, num_queries, 1, 1, 2]
        sampling_locations = reference_points + sampling_offsets / torch.tensor([W, H], device=query.device)  # [N, num_queries, num_heads, num_points, 2]
        # 将采样位置从归一化坐标系映射到 [-1, 1]，以适配 grid_sample
        sampling_locations_normalized = sampling_locations * 2 - 1 
        sampling_locations_normalized = sampling_locations_normalized.view(N, num_queries * self.num_heads * self.num_points, 1, 2)

        sampled_value = F.grid_sample(value, sampling_locations_normalized, mode='bilinear', padding_mode='zeros', align_corners=False)  # [N, embed_dim, num_queries * num_heads * num_points, 1]

        sampled_value = sampled_value.squeeze(-1)  # [N, embed_dim, num_queries * num_heads * num_points]
        sampled_value = sampled_value.view(N, self.embed_dim, num_queries, self.num_heads, self.num_points)  # [N, embed_dim, num_queries, num_heads, num_points]
        sampled_value = sampled_value.permute(0, 2, 3, 4, 1)  # [N, num_queries, num_heads, num_points, embed_dim]

        attention_weights = attention_weights.unsqueeze(-1)  # [N, num_queries, num_heads, num_points, 1]
        weighted_values = (sampled_value * attention_weights).sum(dim=3)  # [N, num_queries, num_heads, embed_dim]

        weighted_values = weighted_values.reshape(N, num_queries, -1)  # [N, num_queries, num_heads * embed_dim]

        output = self.output_proj(weighted_values)  # [N, num_queries, embed_dim]
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    num_queries = 10
    embed_size = 256
    num_heads = 8
    num_points = 4
    H, W = 32, 32

    deformable_attention = DeformableAttention(embed_size, num_heads, num_points).to(device)

    query = torch.randn(batch_size, num_queries, embed_size).to(device)
    reference_points = torch.rand(batch_size, num_queries, 2).to(device)  # 在 [0, 1] 区间
    value = torch.randn(batch_size, H, W, embed_size).to(device)

    output = deformable_attention(query, reference_points, value)
    print(query.shape, output.shape)  # 应为 [batch_size, num_queries, embed_dim]
