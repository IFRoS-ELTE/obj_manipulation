from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch

if TYPE_CHECKING:
    from torch import FloatTensor, IntTensor


# Code adapted from https://github.com/chrisdxie/uois
def get_euclid_distance(x: FloatTensor, y: FloatTensor) -> FloatTensor:
    """Compute the pairwise Euclidean distances between each datapoint in x and y.
    
    Args:
        x: [n x d] tensor of datapoints.
        y: [m x d] tensor of datapoints.
    
    Returns:
        [n x m] tensor of pairwise Euclidean distances.
    """
    return torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)


def get_euclid_distance_sqr(x: FloatTensor, y: FloatTensor) -> FloatTensor:
    """Compute the pairwise squared Euclidean distances between each datapoint in x and y.
    
    Args:
        x: [n x d] tensor of datapoints.
        y: [m x d] tensor of datapoints.
    
    Returns:
        [n x m] tensor of squared pairwise Euclidean distances.
    """
    return torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2)


def get_gaussian_density(x: FloatTensor, y: FloatTensor, sigma: Union[float, FloatTensor]) -> FloatTensor:
    """Compute pairwise Gaussian kernel density without normalizing constant.
    
    Args:
        x: [n x d] tensor of datapoints.
        y: [m x d] tensor of datapoints.
        sigma: Gaussian kernel bandwidth, either scalar or [1 x m] tensor.
    
    Returns:
        [n x m] tensor of pairwise, unnormalized Gaussian kernel densities. 
    """
    return torch.exp(-0.5 / (sigma ** 2) * get_euclid_distance_sqr(x, y))


class MeanShift(ABC):
    """ Base abstract class for Mean Shift algorithms w/ diff kernels."""

    def __init__(
        self,
        num_seeds: int = 100,
        max_iters: int = 10,
        epsilon: float = 1e-2,
        sigma: float = 1.0,
        subsample_factor: int = 1,
        batch_size: Optional[int] = None,
    ):
        self.num_seeds = num_seeds
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.sigma = sigma
        self.subsample_factor = subsample_factor
        self.batch_size = 1024 if batch_size is None else batch_size
        # This should be a function that computes distances w/ func signature: (x, y)
        self.distance: Optional[Callable[[FloatTensor, FloatTensor], FloatTensor]] = None 
        # This should be a function that computes a kernel w/ func signature: (x, y, sigma)
        self.kernel: Optional[Callable[[FloatTensor, FloatTensor, float], FloatTensor]] = None
        
    def connected_components(self, seeds: FloatTensor) -> IntTensor:
        """Compute simple connected components algorithm.

        Args:
            seeds: [n x d] tensor of datapoints.
        
        Returns:
            [n] int tensor of cluster labels. 
        """
        n = seeds.shape[0]
        device = seeds.device
        
        # Sampling/Grouping
        n_clusters = torch.zeros(1, dtype=torch.int, device=device)
        cluster_labels = torch.full(n, -1, dtype=torch.int, device=device)
        for i in range(n):
            if cluster_labels[i] == -1:
                # Find all points close to it and label it the same
                distances = self.distance(seeds, seeds[i:i+1]) # Shape: (n, 1)
                adjacents = distances[:, 0] <= self.epsilon
                adjacent_labels = cluster_labels[adjacents]

                # If at least one component already has a label, then use the mode of the label
                assigned_labels_mask = adjacent_labels != -1
                if assigned_labels_mask.any():
                    label = torch.mode(adjacent_labels[assigned_labels_mask])
                else:
                    label = n_clusters.clone()
                    n_clusters += 1  # Increment number of clusters
                cluster_labels[adjacents] = label
            
            # Exit early if all points have been assigned to a cluster
            if not (cluster_labels == -1).any():
                break

        return cluster_labels

    def seed_hill_climbing(self, data: FloatTensor, seeds: FloatTensor) -> FloatTensor:
        """Run mean shift hill climbing algorithm on the seeds. 
        The seeds climb the distribution given by the KDE of data.

        Args:
            data: [n x d] tensor of datapoints.
            seeds: [m x d] tensor of initial seed locations to start from.
        
        Returns:
            [m x d] tensor containing final positions of seeds.
        """
        m = seeds.shape[0]
        batch_size = min(self.batch_size, m)

        for _ in range(self.max_iters):
            # Create a new object for Z
            new_seeds = seeds.clone()

            # Compute the update in batches if seeds are very many
            for i in range(0, m, batch_size):
                weights = self.kernel(seeds[i: i + batch_size], data, self.sigma)  # Shape: (batch_size, n)
                weights = weights / weights.sum(dim=1, keepdim=True)  # Shape: (batch_size, n)
                new_seeds[i: i + batch_size] = torch.mm(weights, data)
            seeds = new_seeds

        return seeds

    def select_seeds(self, data: FloatTensor) -> FloatTensor:
        """Randomly select seeds that far away from each other.

        Args:
            data: [n x d] tensor of datapoints.

        Returns:
            [min(num_seeds, n) x d] tensor of selected seeds.
        """
        # Initialize seeds matrix
        n, d = data.shape
        num_seeds = min(self.num_seeds, n)
        device = data.device
        seeds = torch.empty((num_seeds, d), device=device)

        # Keep track of distances
        distances = torch.empty((n, num_seeds), device=device)

        # Select first seed
        selected_seed_index = torch.randint(0, n, (1,), dtype=torch.int, device=device)
        seeds[0] = data[selected_seed_index].clone()

        distances[:, 0] = self.distance(data, seeds[:1])[:, 0]
        num_chosen_seeds = 1

        # Select rest of seeds
        for i in range(num_chosen_seeds, num_seeds):
            # Sample points that have the furthest distance from the nearest seed with high probability
            # Note: Return values of torch.min() have changed in later versions
            distance_to_nearest_seed = torch.min(distances[:, :i], dim=1)[0]  # Shape: (n,)
            selected_seed_index = torch.multinomial(distance_to_nearest_seed, 1)
            seeds[i] = data[selected_seed_index].clone()

            # Calculate distance to this selected seed
            distances[:, i] = self.distance(data, seeds[i: i + 1])[:, 0]
        
        return seeds

    def mean_shift(self, data: FloatTensor) -> IntTensor:
        """Run the mean shift algorithm.

        Args:
            data: [n x d] tensor of datapoints.
        
        Returns:
            [n] int tensor of cluster labels.
        """
        # Get initial seed positions
        data_subsampled = data[::self.subsample_factor]  # Shape (n // subsample_factor, d)
        seeds = self.select_seeds(data_subsampled)

        # Run mean shift algorithm and get connected clusters labels
        seeds = self.seed_hill_climbing(data_subsampled, seeds)
        seed_cluster_labels = self.connected_components(seeds)

        # Get distances for all points to updated seeds
        distances = self.distance(data, seeds)

        # Get clusters labels for all points according to closest seed
        closest_seed_indices = torch.argmin(distances, dim=1)  # Shape (n,)
        cluster_labels = seed_cluster_labels[closest_seed_indices]

        return cluster_labels


class GaussianMeanShift(MeanShift):
    """Specialization of the mean shift algorithm for Gaussian kernels with Euclidean distance."""

    def __init__(
        self,
        num_seeds: int = 100,
        max_iters: int = 10,
        epsilon: float = 0.05,
        sigma: float = 1.0,
        subsample_factor: int = 1,
        batch_size: Optional[int] = None,
    ):
        super().__init__(
            num_seeds, max_iters, epsilon, sigma, subsample_factor, batch_size=batch_size
        )
        self.distance = get_euclid_distance
        self.kernel = get_gaussian_density