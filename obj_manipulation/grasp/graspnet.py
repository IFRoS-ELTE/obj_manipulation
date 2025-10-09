from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from obj_manipulation.grasp.pointnet import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    PointNetSetAbstractionMsg,
)

if TYPE_CHECKING:
    from torch import FloatTensor


# Code adapted from https://github.com/elchun/contact_graspnet_pytorch
class ContactGraspNet(nn.Module):
    """Contact-GraspNet based on the PointNet++ architecture."""

    def __init__(self):
        super().__init__()
        self.bin_vals = self._get_bins_vals()
        self.gripper_depth = 0.1034  # TODO: Update based on actual gripper

        # Initialize PointNet for feature extraction
        pn_out_channels = self._build_pointnet()

        # Initialize output head for grasp direction
        self.grasp_dir_head = nn.Sequential(
            nn.Conv1d(pn_out_channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 3, kernel_size=1),
        )

        # Initialize output head for grasp approach direction
        self.grasp_approach_head = nn.Sequential(
            nn.Conv1d(pn_out_channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 3, kernel_size=1),
        )

        # Initialize output head for grasp width (discretized across 10 bins)
        self.grasp_offset_head = nn.Sequential(
            nn.Conv1d(pn_out_channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 10, kernel_size=1),
        )
        
        # Initialize output head for contact points (binary success/failure)
        self.binary_seg_head = nn.Sequential(
            nn.Conv1d(pn_out_channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, kernel_size=1),
        )
    
    def _get_bins_vals(self) -> FloatTensor:
        """Creates bin values for grasping widths according to set opening bounds."""
        # Determine bin values at centers of boundaries
        bin_bounds = torch.tensor([
            0,
            0.00794435329,
            0.0158887021,
            0.0238330509,
            0.0317773996,
            0.0397217484,
            0.0476660972,
            0.055610446,
            0.0635547948,
            0.0714991435,
            0.08,
        ])
        bin_vals = (bin_bounds[:-1] + bin_bounds[1:]) / 2
        
        # Clip max values to avoid grasps that require the gripper's full width
        gripper_width, gripper_opening_margin = 0.08, 0.005
        bin_vals = torch.minimum(bin_vals, torch.tensor(gripper_width - gripper_opening_margin))
        return bin_vals
    
    def _build_pointnet(self) -> int:
        """Build PointNet for feature extraction with fixed architecture and return number of 
        output channels.
        """
        # Initialize PointNet set abstraction layers
        self.sa1 = PointNetSetAbstractionMsg(
            in_channel=3,
            mlp_list=[
                [32, 32, 64],
                [64, 64, 128],
                [64, 96, 128],
            ],
            n_points=2048,
            radius_list=[0.02, 0.04, 0.08],
            max_samples_list=[32, 64, 128],
        )
        sa1_out_channels = sum([64, 128, 128])

        self.sa2 = PointNetSetAbstractionMsg(
            in_channel=sa1_out_channels,
            mlp_list=[
                [64, 64, 128],
                [128, 128, 256],
                [128, 128, 256],
            ],
            n_points=512,
            radius_list=[0.04, 0.08, 0.16],
            max_samples_list=[64, 64, 128],
        )
        sa2_out_channels = sum([128, 256, 256])

        self.sa3 = PointNetSetAbstractionMsg(
            in_channel=sa2_out_channels,
            mlp_list=[
                [64, 64, 128],
                [128, 128, 256],
                [128, 256, 256],
            ],
            n_points=128,
            radius_list=[0.08, 0.16, 0.32],
            max_samples_list=[64, 64, 128],
        )
        sa3_out_channels = sum([128, 256, 256])

        self.sa4 = PointNetSetAbstraction(
            in_channel=sa3_out_channels + 3,
            mlp=[256, 512, 1024],
            n_points=-1,            # Unused
            radius=0.0,             # Unused
            max_samples=-1,         # Unused
            group_all=True,
        )
        sa4_out_channels = 1024

        # Initialize PointNet feature propagation layers
        self.fp3 = PointNetFeaturePropagation(
            in_channel=sa4_out_channels + sa3_out_channels,
            mlp=[128, 128, 128],
        )
        fp3_out_channels = 128

        self.fp2 = PointNetFeaturePropagation(
            in_channel=fp3_out_channels + sa2_out_channels,
            mlp=[256, 128],
        )
        fp2_out_channels = 128

        self.fp1 = PointNetFeaturePropagation(
            in_channel=fp2_out_channels + sa1_out_channels,
            mlp=[256, 256],
        )
        fp1_out_channels = 256
        return fp1_out_channels

    def forward(self, xyz_pc: FloatTensor) -> Dict[str, FloatTensor]:
        """Predict grasp points from input point cloud.
        
        Args:
            xyz_pc: [B x N x 3] tensor of batched full 3-dim point clouds.
        
        Returns:
            dict
            - "pred_grasps": [B x N x 4 x 4] tensor of batched homogeneous matrices representing
            grasp poses in the same coordinate system as input point clouds.
            - "pred_scores": [B x N x 1] tensor of batched grasp success probabilities.
            - "pred_points": [B x N x 3] tensor of batched contact points used for grasp predictions.
        """
        # Convert inputs from channel-last to format channel-first
        xyz_pc = xyz_pc.permute(0, 2, 1)

        l0_xyz = xyz_pc[:, :3, :]
        l0_points = l0_xyz.clone()

        # -- PointNet Backbone -- #
        # Set Abstraction Layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Feature Propagation Layers
        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        l0_points = l1_points
        pred_points = l1_xyz

        # -- Output Heads -- #
        # Grasp Direction Head + Normalization
        grasp_dir = self.grasp_dir_head(l0_points)
        grasp_dir = F.normalize(grasp_dir, p=2, dim=1)

        # Grasp Approach Head + Gram-Schmidt Orthonormalization
        approach_dir = self.grasp_approach_head(l0_points)
        dot_product = torch.sum(approach_dir * grasp_dir, dim=1, keepdim=True)
        projection = dot_product * grasp_dir
        approach_dir = F.normalize(approach_dir - projection, p=2, dim=1)

        # Grasp Width Head (Logits for the 10 bins)
        grasp_width_logits = self.grasp_offset_head(l0_points)

        # Binary Segmentation Head
        binary_seg = self.binary_seg_head(l0_points)

        # -- Construct Output -- #
        # Get grasp width
        bin_vals = self.bin_vals.to(device=xyz_pc.device)
        grasp_width_indices = torch.argmax(grasp_width_logits, dim=1, keepdim=True)
        grasp_width = bin_vals[grasp_width_indices]  # Shape = (B, 1, N)
        
        # Get 6 DoF grasp pose (as homogeneous matrices)
        pred_grasps = self.build_6d_grasp(
            approach_dir.permute(0, 2, 1),
            grasp_dir.permute(0, 2, 1),
            pred_points.permute(0, 2, 1),
            grasp_width.permute(0, 2, 1),
        )  # Shape = (B, N, 4, 4)

        # Get success pred scores
        pred_scores = torch.sigmoid(binary_seg).permute(0, 2, 1)

        # Get pred points
        pred_points = pred_points.permute(0, 2, 1)

        # Combine outputs into dictionary
        pred = {
            "pred_grasps": pred_grasps,
            "pred_scores": pred_scores,
            "pred_points": pred_points,
        }
        return pred

    def build_6d_grasp(
        self,
        approach_dir: FloatTensor,
        grasp_dir: FloatTensor,
        contact_pts: FloatTensor,
        grasp_width: FloatTensor,
        gripper_depth: float = 0.1034,
    ) -> FloatTensor:
        """Build 6-DoF grasps + width from point-wise network predictions

        Args:
            approach_dir: [B x N x 3] tensor of batched approach direction vectors.
            grasp_dir: [B x N x 3] tensor of batched grasp direction vectors.
            contact_pts [B x N x 3] tensor of batched contact points for predictions.
            grasp_width: [B x N x 1] tensor of batched grasp width predictions.
            gripper_depth: Distance from gripper coordinate frame to gripper baseline.

        Returns:
            [B x N x 4 x 4] tensor of batched homogeneous matrices representing grasp poses
            in the same coordinate system as input point clouds.
        """
        # We are trying to build a stack of 4x4 homogeneous transform matricies of size B x N x 4 x 4.
        # To do so, we calculate the rotation and translation portions according to the paper.
        device = approach_dir.device
        n_batches, n_points = approach_dir.shape[:2]
        
        # Calculate R from approach and grasp orthogonal directions
        grasp_R = torch.stack(
            [grasp_dir, torch.cross(approach_dir, grasp_dir, dim=2), approach_dir],
            dim=3,
        )  # Shape = (B, N, 3, 3)

        # Calculate t based on contact point, gripper width and depth (check paper for visualization).
        grasp_t = contact_pts + (grasp_width / 2) * grasp_dir - gripper_depth * approach_dir
        grasp_t = grasp_t.unsqueeze(3)  # Shape = (B, N, 3, 1)
        
        # Combine R and t into 4x4 homogeneous matrices
        ones = torch.ones((n_batches, n_points, 1, 1), device=device)
        zeros = torch.zeros((n_batches, n_points, 1, 3), device=device)
        homog_vec = torch.cat([zeros, ones], dim=3)  # Shape = (B, N, 1, 4)
        grasps = torch.cat(
            [torch.cat([grasp_R, grasp_t], dim=3), homog_vec],
            dim=2,
        )  # Shape = (B, N, 4, 4)

        return grasps