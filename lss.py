import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthPredictor(nn.Module):
    def __init__(self, input_channels, num_bins):
        super().__init__()
        self.num_bins = num_bins
        self.conv = nn.Conv2d(input_channels, num_bins, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features (tensor): [batch_size, num_cameras, C_f, H_h, W_f]
        Returns:
            depth_probs (tensor): [batch_size, num_cameras, num_bins, H_f, W_f]
        """
        batch_size, num_cameras, C_f, H_f, W_f = features.size()
        features = features.view(-1, C_f, H_f, W_f)
        depth_logits = self.conv(features)
        depth_probs = F.softmax(depth_logits, dim=1)
        depth_probs = depth_probs.view(batch_size, num_cameras, self.num_bins, H_f, W_f)
        return depth_probs


class LiftModule(nn.Module):
    def __init__(self, num_bins, depth_start, depth_end):
        super().__init__()
        self.num_bins = num_bins
        self.depth_start = depth_start
        self.depth_end = depth_end

    def compute_camera_rays(self, intrinsics, H_f, W_f):
        """
        Returns:
            rays (tensor): [batch_size, num_cameras, H_f, W_f, 3]
        """
        device = intrinsics.device
        batch_size, num_cameras, _, _ = intrinsics.shape
        ys, xs = torch.meshgrid(
            torch.arange(0, H_f, device=device),
            torch.arange(0, W_f, device=device)
        )
        xs = xs.float() # (H_f, W_f)
        ys = ys.float()
        ones = torch.ones_like(xs)
        pixel_coords = torch.stack([xs, ys, ones], dim=-1) # (H_f, W_f, 3)
        pixel_coords = pixel_coords.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # (1, 1, H_f, W_f, 3, 1)

        intrinsics_inv = torch.inverse(intrinsics) # (batch_size, num_cameras, 3, 3)
        intrinsics_inv = intrinsics_inv.unsqueeze(2).unsqueeze(2) #  (batch_size, num_cameras, 1, 1, 3, 3)

        rays = torch.matmul(intrinsics_inv, pixel_coords).squeeze(-1)
        return rays
        
    
    def forward(self, features, depth_probs, intrinsics):
        """
        Args:
            features (tensor): [batch_size, num_cameras, C_f, H_f, W_f]
            depth_probs (tensor): [batch_size, num_cameras, num_bins, H_f, W_f]
            intrinsics (tensor): [batch_size, num_cameras, 3, 3]
        Returns:
            coords_cam (tensor): [batch_size, num_cameras, num_bins, H_f, W_f, 3]
            lifted_features (tensor): [batch_size, num_cameras, num_bins, C_f, H_f, W_f]
        """
        device = features.device
        batch_size, num_cameras, C_f, H_f, W_f = features.size()

        rays = self.compute_camera_rays(intrinsics, H_f, W_f)
        rays = rays.unsqueeze(2) # (batch_size, num_cameras, 1, H_f, W_f, 3)
    
        depths = torch.linspace(self.depth_start, self.depth_end, self.num_bins, device=device)
        depths = depths.view(1, 1, self.num_bins, 1, 1, 1)
        coords_cam = rays * depths

        # features: (batch_size, num_cameras, num_bins, C_f, H_f, W_f)
        features = features.unsqueeze(2).expand(-1, -1, self.num_bins, -1, -1, -1) 
        depth_probs = depth_probs.unsqueeze(3) # (batch_size, num_cameras, num_bins, 1, H_f, W_f)
        lifted_features = features * depth_probs 

        return coords_cam, lifted_features


class SplatModule(nn.Module):
    def __init__(self, bev_grid, bev_resolution):
        super().__init__()
        self.bev_grid = bev_grid # [[x_min, x_max], [y_min, y_max]]
        self.bev_resolution = bev_resolution
        self.x_min, self.x_max = bev_grid[0]
        self.y_min, self.y_max = bev_grid[1]
        self.bev_W = int((self.x_max - self.x_min) / bev_resolution)
        self.bev_H = int((self.y_max - self.y_min) / bev_resolution)

    def forward(self, coords_world, lifted_features):
        """
        Args: 
            coords_world (tensor): [batch_size, num_cameras, num_bins, H_f, W_f, 3]
            lifted_features (tensor): [batch_size, num_cameras, num_bins, C_f, H_f, W_f]
        Returns:
            bev_features (tensor): [batch_size, C_f, bev_H, bev_W]
        """
        device = coords_world.device
        batch_size, num_cameras, num_bins, C_f, H_f, W_f = lifted_features.size()

        x = coords_world[..., 0]
        y = coords_world[..., 1]
        x_indices = (((x - self.x_min) / (self.x_max - self.x_min)) * self.bev_W).long()
        y_indices = ((y - self.y_min) / (self.y_max - self.y_min) * self.bev_H).long()
        valid_mask = (x_indices >= 0) & (x_indices < self.bev_W) & (y_indices >= 0) & (y_indices < self.bev_H)

        bev_features = torch.zeros((batch_size, C_f, self.bev_H, self.bev_W), device=device)
        for b in range(batch_size):
            for n in range(num_cameras):
                valid_mask_b_n = valid_mask[b, n]
                N_valid = valid_mask_b_n.sum().item()
                if N_valid == 0:
                    continue

                x_idx_flat = x_indices[b, n].reshape(-1)[valid_mask[b, n].reshape(-1)]
                y_idx_flat = y_indices[b, n].reshape(-1)[valid_mask[b, n].reshape(-1)]
                linear_idx = y_idx_flat * self.bev_W + x_idx_flat

                feats_b_n = lifted_features[b, n].permute(1, 0, 2, 3)
                feats_b_n_flat = feats_b_n.reshape(C_f, -1)
                feats = feats_b_n_flat[:, valid_mask_b_n.reshape(-1)]

                bev_feat_flat = bev_features[b].view(C_f, -1)
                bev_feat_flat.scatter_add_(1, linear_idx.unsqueeze(0).expand(C_f, -1), feats)
        return bev_features


class BEVProcessor(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, bev_features):
        """
        Args:
            bev_features (tensor): [batch_size, C_f, bev_H, bev_W]
        Returns:
            outputs (tensor): [batch_size, num_classes, bev_H, bev_W]
        """
        outputs = self.conv(bev_features)
        return outputs


class LSSModel(nn.Module):
    def __init__(self, num_bins, depth_start, depth_end, bev_grid, bev_resolution, output_channels, num_classes):
        super().__init__()
        self.depth_predictor = DepthPredictor(output_channels, num_bins)
        self.lift_module = LiftModule(num_bins, depth_start, depth_end)
        self.splat_module = SplatModule(bev_grid, bev_resolution)
        self.bev_processor = BEVProcessor(output_channels, num_classes)

    def transform_to_world(self, coords_cam, extrinsics):
        """
        Args:
            coords_cam (tensor): [batch_size, num_cameras, num_bins, H_f, W_f, 3]
            extrinsics (tensor): [batch_size, num_cameras, 4, 4]
        Returns:
            coords_world (tensor): [batch_size, num_cameras, num_bins, H_f, W_f, 3]
        """
        device = coords_cam.device
        batch_size, num_cameras, num_bins, H_f, W_f, _ = coords_cam.size()
        coords_cam_hom = torch.cat(
            [coords_cam, torch.ones_like(coords_cam[..., :1])], dim=-1
        )
        extrinsics = extrinsics.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        coords_world_hom = torch.matmul(extrinsics, coords_cam_hom.unsqueeze(-1)).squeeze(-1)
        coords_world = coords_world_hom[..., :3] / coords_world_hom[..., 3:]
        return coords_world

    def forward(self, features, intrinsics, extrinsics):
        """
        Args:
            features (tensor): [batch_size, num_cameras, C_f, H_f, W_f]
            intrinsics (tensor): [batch_size, num_cameras, 3, 3]
            extrinsics (tensor): [batch_size, num_cameras, 4, 4]
        """
        depth_probs = self.depth_predictor(features)
        coords_cam, lifted_features = self.lift_module(features, depth_probs, intrinsics)

        coords_world = self.transform_to_world(coords_cam, extrinsics)
        bev_feats = self.splat_module(coords_world, lifted_features)
        outputs = self.bev_processor(bev_feats)
        return outputs

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 0. model
    num_bins = 80
    depth_start, depth_end = 1.0, 50.0
    bev_grid = [[-50.0, 50.0], [-50.0, 50.0]]
    bev_resolution = 0.5
    
    batch_size = 32
    H_f, W_f, C_f = 48, 64, 128
    num_classes = 22

    lss_model = LSSModel(num_bins, depth_start, depth_end, bev_grid, bev_resolution, C_f, num_classes).to(device)

    # 1. data
    num_cameras = 6
    features = torch.randn((batch_size, num_cameras, C_f, H_f, W_f), dtype=torch.float32).to(device)
    intrinsics = torch.randn((batch_size, num_cameras, 3, 3), dtype=torch.float32).to(device)
    extrinsics = torch.randn((batch_size, num_cameras, 4, 4), dtype=torch.float32).to(device)

    # 2. inference
    outputs  = lss_model(features, intrinsics, extrinsics)
    print(outputs.size())
