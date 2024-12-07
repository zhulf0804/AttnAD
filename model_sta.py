import torch
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis
from attention import Attention
from multi_head_attention import MultiHeadAttention
from deformable_attention import DeformableAttention
from lss import LSSModel
from vit import ViT


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_dim = 256
    seq_len = 10
    batch_size = 1
    dropout_rate = 0.1
    x = torch.rand((batch_size, seq_len, embed_dim)).to(device)
    
    ## Attention
    print("***************************** 1. Attention *****************************")
    attention = Attention(embed_dim).to(device)
    summary(attention, input_size=[(batch_size, seq_len, embed_dim), (batch_size, seq_len, embed_dim), (batch_size, seq_len, embed_dim)], device=str(device))    
    # params: (256 * 256 + 256) * 4 = 263168
    flops = FlopCountAnalysis(attention, (x, x, x))
    print(f'Total FLOPs: {flops.total()}') 
    # FLOPs: 10 * 256 * 256 * 4 + 10 * 10 * 256 + 10 * 256 * 10 = 2672640

    ## MultiHead Attention
    print("***************************** 2. MultiHeadAttention *****************************")
    num_heads = 8
    multi_head_attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate).to(device)

    summary(multi_head_attention, input_size=[(batch_size, seq_len, embed_dim), (batch_size, seq_len, embed_dim), (batch_size, seq_len, embed_dim)], device=str(device))    
    flops = FlopCountAnalysis(multi_head_attention, (x, x, x))
    print(f'Total FLOPs: {flops.total()}') 

    ## Deformable Attention
    print("***************************** 3. DeformableAttention *****************************")
    num_heads = 8
    num_points = 4
    H, W = 32, 32 # (32 * 32 = 1024 values)
    reference_points = torch.rand(batch_size, seq_len, 2).to(device) 
    value = torch.randn(batch_size, H, W, embed_dim).to(device)
    deformable_attention = DeformableAttention(embed_dim, num_heads, num_points).to(device)
    summary(deformable_attention, input_size=[(batch_size, seq_len, embed_dim), (batch_size, seq_len, 2), (batch_size, H, W, embed_dim)], device=str(device))
    flops = FlopCountAnalysis(deformable_attention, (x, reference_points, value))
    print(f'Total FLOPs: {flops.total()}') 

    ## LSS Model
    print("***************************** 4. LSS *****************************")
    num_bins = 80
    depth_start, depth_end = 1.0, 50.0
    bev_grid = [[-50.0, 50.0], [-50.0, 50.0]]
    bev_resolution = 0.5
    num_cameras = 6
    
    batch_size = 32
    H_f, W_f, C_f = 48, 64, 128
    num_classes = 22

    device = torch.device('cpu')
    features = torch.randn((batch_size, num_cameras, C_f, H_f, W_f), dtype=torch.float32).to(device)
    intrinsics = torch.randn((batch_size, num_cameras, 3, 3), dtype=torch.float32).to(device)
    extrinsics = torch.randn((batch_size, num_cameras, 4, 4), dtype=torch.float32).to(device)
    lss_model = LSSModel(num_bins, depth_start, depth_end, bev_grid, bev_resolution, C_f, num_classes)
    summary(lss_model, input_size=[(batch_size, num_cameras, C_f, H_f, W_f), (batch_size, num_cameras, 3, 3), (batch_size, num_cameras, 4, 4)], device=str(device))
    flops = FlopCountAnalysis(lss_model, (features, intrinsics, extrinsics))
    print(f'Total FLOPs: {flops.total()}') 

    ## ViT Model
    print("***************************** 5. ViT *****************************")
    image_size = 224
    patch_size = 16
    depth = 6
    ffn_dim = embed_dim
    images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    vit_model = ViT(image_size, patch_size, embed_dim, dropout_rate, num_heads, ffn_dim, depth, num_classes).to(device)
    summary(vit_model, input_size=[(batch_size, 3, image_size, image_size)], device=str(device))
    flops = FlopCountAnalysis(vit_model, (images))
    print(f'Total FLOPs: {flops.total()}') 
