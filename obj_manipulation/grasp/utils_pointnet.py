from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from torch import FloatTensor, IntTensor, Optional


# Code adapted from https://github.com/elchun/contact_graspnet_pytorch
def get_squared_distances(src: FloatTensor, dst: FloatTensor) -> FloatTensor:
    """Calculate the squared pairwise Euclidean distance between points.

    Args:
        src: [B x N x C] tensor of batched C-dim point groups. 
        dst: [B x M x C] tensor of batched C-dim point groups.
    
    Returns:
        [B x N x M] tensor containing point-wise squared Euclidean distances for all points.
    """
    return torch.sum((src.unsqueeze(2) - dst.unsqueeze(1) ** 2), dim=3)


def index_points(points: FloatTensor, idx: IntTensor) -> FloatTensor:
    """Index points at specified indices and return subset of chosen points.
    
    Args:
        points: [B x N x C] tensor of batched C-dim input point groups.
        idx: [B x S] or [B x S x M] int tensor of sample index data.
    
    Return:
        [B x S x C] or [B x S x M x C] tensor of indexed points data.
    """
    indices = idx.unsqueeze(-1).expand_as(points)  # Repeat along last dimension
    shape = indices.shape
    indices = indices.view(indices.shape[0], -1, indices.shape[-1])
    return torch.gather(points, dim=1, index=indices).view(shape)


def sample_farthest_points(xyz_pc: FloatTensor, n_points: int) -> IntTensor:
    """Find indices of batch of points located farthest from each other in the point cloud.  
    
    Args:
        xyz_pc: [B x N x C] tensor of batched C-dim input point groups.
        n_points: Number of points to sample from each point group. 
    
    Returns:
        [B x n_points] int tensor of sampled point indices.
    """
    device = xyz_pc.device
    batch_dim, group_dim = xyz_pc.shape[:2]

    centroid_indices = torch.zeros((batch_dim, n_points), dtype=torch.int, device=device)
    distance_min = torch.ones((batch_dim, group_dim), device=device) * 1e10
    farthest_indices = torch.randint(0, group_dim, (batch_dim,), dtype=torch.int, device=device)
    batch_indices = torch.arange(batch_dim, dtype=torch.int, device=device)
    for i in range(n_points):
        # Add farthest point from points selected so far  
        centroid_indices[:, i] = farthest_indices
        
        # Update nearest distance for all points
        centroid = xyz_pc[batch_indices, farthest_indices, :].view(batch_dim, 1, 3)
        centroid_dist = torch.sum((xyz_pc - centroid) ** 2, dim=2)
        mask = centroid_dist < distance_min
        distance_min[mask] = centroid_dist[mask]
        
        # Update farthest point indices
        farthest_indices = torch.argmax(distance_min, dim=-1)
    
    return centroid_indices


def query_ball_point(
    xyz_pc: FloatTensor,
    xyz_query_pc: FloatTensor,
    radius: float,
    max_samples: int,
) -> IntTensor:
    """Query point groups in the local neighborhood of given points from point cloud. 
    
    Args:
        xyz_pc: [B x N x C] tensor of batched full C-dim point clouds.
        xyz_query_pc: [B x S x C] tensor of batched C-dim point group centroids.
        radius: Radius of local neighborhood to collect points from.
        max_samples: Maximum number of points to sample from each local region.
    
    Returns:
        [B x S x max_samples] int tensor of indices of point groups around each centroid.
    """
    device = xyz_pc.device
    batch_dim, group_dim = xyz_pc.shape[:2]
    query_group_dim = xyz_query_pc.shape[1]

    # Generate group indices that encompass all points in the point cloud
    group_idx = torch.arange(group_dim, dtype=torch.int, device=device)
    group_idx = group_idx.repeat((batch_dim, query_group_dim, 1))

    # Keep only the first max_samples within the set radius
    sqr_dists = get_squared_distances(xyz_query_pc, xyz_pc)
    group_idx[sqr_dists > radius ** 2] = group_dim
    group_idx = group_idx.sort(dim=-1)[0][:, :, :max_samples]
    
    # Replace out-of-boundary points with the repetitions of the first point in the group
    group_first = group_idx[:, :, :1].repeat((1, 1, max_samples))
    mask = group_idx == group_dim
    group_idx[mask] = group_first[mask]
    return group_idx


def group_points(
    xyz_pc: FloatTensor,
    xyz_query_pc: FloatTensor,
    points: Optional[FloatTensor],
    radius: float,
    max_samples: int,
) -> FloatTensor:
    """Collect point groups from the local neighborhoods of given query point clouds.
    
    Args:
        xyz_pc: [B x N x C] tensor of batched full C-dim point clouds.
        xyz_query_pc: [B x S x C] tensor of sampled points (group centroids).
        points: [B x N x D] tensor of batched D-dim input point groups.
        radius: Radius of local neighborhood to collect points from.
        max_samples: Maximum number of points to sample from each local region.
    
    Returns:
        [B x S x max_samples x (C or C+D)] tensor of sampled point groups around each query.
    """
    # Collect point groups from query points' local neighborhoods
    group_idx = query_ball_point(xyz_pc, xyz_query_pc, radius, max_samples)  # Shape = (B, S, max_samples)
    xyz_query_group_pc = index_points(xyz_pc, group_idx)  # Shape = (B, S, max_samples, C)
    xyz_query_group_pc -= xyz_query_pc.unsqueeze(dim=2)  # Convert to relative coordinates to centroids 
    if points is None:
        return xyz_query_group_pc
    
    # If input points are given, combine them with the 3D query group points
    grouped_points = index_points(points, group_idx)  # Shape = (B, S, max_samples, D)
    new_points = torch.cat([xyz_query_group_pc, grouped_points], dim=-1)  # Shape = (B, S, max_samples, C+D)
    return new_points


def sample_and_group(
    xyz_pc: FloatTensor,
    points: Optional[FloatTensor],
    n_points: int,
    radius: float,
    max_samples: int,
) -> Tuple[FloatTensor, FloatTensor]:
    """Sample farthest points and collect point groups from their neighborhoods.
    
    Args:
        xyz_pc: [B x N x C] tensor of batched full C-dim point clouds.
        points: [B x N x D] tensor of batched D-dim input point groups.
        n_points: Number of points (centroids) to sample from each input point group. 
        radius: Radius of local neighborhood to collect points from.
        max_samples: Maximum number of points to sample from each local region.
    
    Returns:
        tuple
        - xyz_query_pc: [B x n_points x C] tensor of sampled points (group centroids).
        - new_points: [B x n_points x max_samples x (C or C+D)] tensor of sampled point groups around each centroid.
    """
    # Sample farthest points as point group centroids
    fps_idx = sample_farthest_points(xyz_pc, n_points)  # Shape = (B, n_points)
    xyz_query_pc = index_points(xyz_pc, fps_idx)  # Shape = (B, n_points, C)
    
    # Collect point groups from their local neighborhoods
    new_points = group_points(xyz_pc, xyz_query_pc, points, radius, max_samples)
    return xyz_query_pc, new_points


def sample_and_group_all_points(
    xyz_pc: FloatTensor,
    points: Optional[FloatTensor],
) -> Tuple[FloatTensor, FloatTensor]:
    """Sample a single point group that contains all points in input point cloud.
    Special version of sample_and_group() n_samples = 1, radius = inf and max_samples = N.

    Args:
        xyz_pc: [B x N x C] tensor of batched full C-dim point clouds.
        points: [B x N x D] tensor of batched D-dim input point groups.
        
    Returns:
        tuple
        - xyz_query_pc: [B x 1 x C] tensor of zeros (virtual centroid of all points).
        - new_points: [B x 1 x N x (C or C+D)] tensor of all points.
    """
    batch_dim, _, point_dim = xyz_pc.shape

    # Create query centroid at zero and group all points
    xyz_query_pc = torch.zeros((batch_dim, 1, point_dim), device=xyz_pc.device)
    xyz_query_group_pc = xyz_pc.unsqueeze(dim=1)
    if points is None:
        return xyz_query_pc, xyz_query_group_pc
    
    # If input points are given, combine them with the query group points
    new_points = torch.cat([xyz_query_group_pc, points.unsqueeze(dim=1)], dim=-1)
    return xyz_query_pc, new_points