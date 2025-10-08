from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from obj_manipulation.grasp.utils_pointnet import (
    get_squared_distances,
    group_points,
    index_points,
    sample_and_group,
    sample_and_group_all_points,
    sample_farthest_points,
)

if TYPE_CHECKING:
    from torch import FloatTensor


# Code adapted from https://github.com/elchun/contact_graspnet_pytorch
class PointNetSetAbstraction(nn.Module):
    """PointNet set abtraction module responsible for generating local feature vectors for an input
    point cloud. 
    
    A subset of points located far away from each other are sampled, then neighboring
    points are collected from the local region of each sample. These points are passed through the
    same MLP layers to generate local features, then features from different samples are aggregated
    through max pooling.
    """

    def __init__(
        self,
        in_channel: int,
        mlp: List[int],
        n_points: int,
        radius: float,
        max_samples: int,
        group_all: bool,
    ):
        super().__init__()
        self.n_points = n_points
        self.radius = radius
        self.max_samples = max_samples
        self.group_all = group_all
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        channels = [in_channel] + mlp
        for i in range(len(channels) - 1):
            self.mlp_convs.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm2d(channels[i + 1]))

    def forward(
        self,
        xyz_pc: FloatTensor,
        points: Optional[FloatTensor],
    ) -> Tuple[FloatTensor, FloatTensor]:
        """Sample point groups, pass them through PointNet and apply max pooling over samples.
        
        Args:
            xyz_pc: [B x C x N] tensor of batched C-dim input point groups.
            points: [B x D x N] tensor fo batched D-dim input point groups.

        Returns:
            tuple
            - xyz_query_pc: [B x C x S] tensor of sampled points from xyz_pc.
            - new_points: [B x D' x S] tensor of points feature data.
        """
        # Convert from PyTorch format (channel-first) to that used in sampling (channel-last).
        xyz_pc = xyz_pc.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # Sample point groups
        xyz_query_pc, new_points = sample_and_group_all_points(xyz_pc, points) if self.group_all \
            else sample_and_group(xyz_pc, points, self.n_points, self.radius, self.max_samples)
        # xyz_query_pc.shape = (B, self.n_points, C)
        # new_points.shape = (B, self.n_points, self.max_samples, C+D)
        
        # Forward propagate new points through PointNet
        new_points = new_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points =  F.relu(bn(conv(new_points)))

        # Max pool over samples to reduce dimension
        new_points = torch.amax(new_points, dim=2)
        xyz_query_pc = xyz_query_pc.permute(0, 2, 1)
        return xyz_query_pc, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """PointNet set abstraction module responsible for generating local feature vectors at
    different scales (through grouping at different radii) for an input point cloud. 
    
    Same as PointNetSetAbstract except that it maintains multiple radii for collecting points, each
    with its own MLP layers. Features from all scales are concatenated together at the end and
    returned.
    """
    
    def __init__(
        self,
        in_channel: int,
        mlp_list: List[List[int]],
        n_points: int,
        radius_list: List[float],
        max_samples_list: List[int],
    ):
        super().__init__()
        self.n_points = n_points
        self.radius_list = radius_list
        self.max_samples_list = max_samples_list
        
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for mlp in mlp_list:
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            channels = [in_channel + 3] + mlp
            for i in range(len(channels) - 1):
                convs.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1))
                bns.append(nn.BatchNorm2d(channels[i + 1]))
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(
        self,
        xyz_pc: FloatTensor,
        points: Optional[FloatTensor],
    ) -> Tuple[FloatTensor, FloatTensor]:
        """Sample point groups at multiple scales (radii), pass them through PointNet, apply
        max pooling over samples and then combine features from all scales.
        
        Args:
            xyz_pc: [B x C x N] tensor of batched C-dim input point groups.
            points: [B x D x N] tensor fo batched D-dim input point groups.

        Returns:
            tuple
            - xyz_query_pc: [B x C x S] tensor of sampled points from xyz_pc.
            - new_points: [B x D' x S] tensor of concatenated points feature data.
        """
        # Convert from PyTorch format (channel-first) to that used in sampling (channel-last).
        xyz_pc = xyz_pc.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # Sample common points to use at all radii
        fps_idx = sample_farthest_points(xyz_pc, self.n_points)
        xyz_query_pc = index_points(xyz_pc, fps_idx)  # Shape = (B, self.n_points, C)
        
        # Generate points at all radii (collect points -> pass through MLP -> max pool)
        new_points_list = []
        for radius, max_samples, convs, bns in zip(
            self.radius_list, self.max_samples_list, self.conv_blocks, self.bn_blocks
        ):
            new_points = group_points(xyz_pc, xyz_query_pc, points, radius, max_samples)
            new_points = new_points.permute(0, 3, 2, 1)  # Shape = (B, D, max_samples, self.n_points)
            for conv, bn in zip(convs, bns):
                new_points = F.relu(bn(conv(new_points)))
            new_points = torch.amax(new_points, dim=2)
            new_points_list.append(new_points)

        # Combine features from all radii along channel dimension
        new_points = torch.cat(new_points_list, dim=1)
        xyz_query_pc = xyz_query_pc.permute(0, 2, 1)
        return xyz_query_pc, new_points


class PointNetFeaturePropagation(nn.Module):
    """PointNet feature propagation module that operates on collections of points. This is
    different from abstraction set modules that operate on collections of point groups.
    
    Features from sampled point cloud are interpolated according to the proximity of their sample
    points to the original points in the input point cloud. Then, these interpolated features are
    passed through the MLP network to produce new features. 
    """
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        channels = [in_channel] + mlp
        for i in range(len(channels) - 1):
            self.mlp_convs.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=1))
            self.mlp_bns.append(nn.BatchNorm1d(channels[i + 1]))

    def forward(
        self,
        xyz_pc1: FloatTensor,
        xyz_pc2: FloatTensor,
        points1: Optional[FloatTensor],
        points2: FloatTensor,
    ) -> FloatTensor:
        """Interpolate features from points2 and combine with points1 if available and pass
        concatenated features through the shared MLP network.
        
        Args:
            xyz_pc1: [B x C x N] tensor of batched C-dim input points groups.
            xyz_pc2: [B x C x S] tensor of batched C-dim sampled input point groups.
            points1: [B x D x N] tensor of batched input feature groups corresponding to xyz_pc1. 
            points2: [B x D x S] tensor of batched input feature groups corresponding to xyz_pc2.
        
        Returns:
            [B x D' x N] tensor of upsampled points data.
        """
        group_dim = xyz_pc1.shape[2]
        group_sample_dim = xyz_pc2.shape[2]

        # Convert inputs from channel-first to channel-last format
        xyz_pc1 = xyz_pc1.permute(0, 2, 1)
        xyz_pc2 = xyz_pc2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # Interpolate point2 based on proximity of points in xyz_pc2 to points in xyz_pc1
        if group_sample_dim == 1:  # All points belong to the same single group
            interpolated_points = points2.repeat(1, group_dim, 1)
        else:
            # Get closest samples to each point in xyz_pc1
            dists = get_squared_distances(xyz_pc1, xyz_pc2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # Shape = (B, N, 3)
            
            # Get sample weights that are inv. proportional to distances
            weights = 1.0 / (dists + 1e-8)  # eps for stability
            weights /= torch.sum(weights, dim=2, keepdim=True)

            # Get interpolated points from points2 based on closest points in xyz_pc2 to points in xyz_pc1
            points2_idx = index_points(points2, idx)
            interpolated_points = torch.sum(points2_idx * weights.unsqueeze(dim=3), dim=2)  # Shape = (B, N, D)

        # Combine interpolated points with points1
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        # Forward propagate combined points through network 
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        
        return new_points
