import torch
import torch.nn as nn
from transformer_encoder import TransformerEncoder


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=3, out_channels=embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        """
        Args:
            x (tensor): [batch_size, 3, image_size, image_size]
        Returns:
            out (tensor): [batch_size, (image_size // patch_size)^2, embed_dim]
        """
        out = self.proj(x)
        out = out.flatten(2)
        out = out.transpose(1, 2)
        return out


class ClassToken(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    def forward(self, x):
        """
        Args:
            x (tensor): [batch_size, num_patches, embed_dim]
        Returns:
            out (tensor): [batch_size, num_patches + 1, embed_dim]
        """
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        out = torch.cat([cls_tokens, x], dim=1)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
    def forward(self, x):
        """
        Args:
            x (tensor): [batch_size, num_patches + 1, embed_dim]
        Returns:
            out (tensor): [batch_size, num_patches + 1, embed_dim]
        """
        return x + self.pos_embedding


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, dropout_rate, num_heads, ffn_dim, depth, num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        self.class_token = ClassToken(embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.positional_encoding = PositionalEncoding(num_patches, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.transformer = TransformerEncoder(embed_dim, num_heads, ffn_dim, dropout_rate, depth)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )


    def forward(self, images):
        """
        Args:
            images (tensor): [batch_size, 3, image_size, image_size]
        Returns:
            out (tensor): [batch_size, num_classes]
        """
        x = self.patch_embedding(images)
        x = self.class_token(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        x = self.transformer(x)

        cls_token_final = x[:, 0]
        out = self.mlp_head(cls_token_final)

        return out 


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim =768
    num_heads = 12 
    ffn_dim = 3072
    dropout_rate = 0.1
    depth = 6
    image_size = 224
    patch_size = 16
    num_classes = 10

    batch_size = 6

    images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    model = ViT(image_size, patch_size, embed_dim, dropout_rate, num_heads, ffn_dim, depth, num_classes).to(device)
    out = model(images)
    print(images.size(), out.size())
    