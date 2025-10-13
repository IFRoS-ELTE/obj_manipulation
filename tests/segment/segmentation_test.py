import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from obj_manipulation.segment import InstanceSegmentationFull
from obj_manipulation.segment.utils import (
    standardize_image_rgb,
    standardize_image_xyz,
    unstandardize_image_rgb,
    load_config,
)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_color_mask(object_index, nc=None):
    """Colors each index differently. Useful for visualizing semantic masks.

    Args:
        object_index: a [H x W] numpy array of ints from {0, ..., nc-1}
        nc: total number of colors. If None, this will be inferred by masks.

    Returns:
        [H x W x 3] numpy array of dtype np.uint8.
    """
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255
        
    return color_mask


def main():
    # Seed PyTorch and NumPy to ensure that repeatable results
    seed_everything(seed=0)

    # Load configuration
    config_path = Path(__file__).parents[2] / "obj_manipulation/segment/config/config.toml"
    assert config_path.exists()
    config = load_config(config_path)

    # Initialize instance segmentation module and load its weights
    ins_seg = InstanceSegmentationFull(config)
    ins_seg.load()
    ins_seg.eval_mode()

    # Get path to all available examples
    osd_img_files = sorted((Path(__file__).parent / "examples").glob("OSD_*.npy"))
    ocid_img_files = sorted((Path(__file__).parent / "examples").glob("OCID_*.npy"))
    n_imgs = len(osd_img_files) + len(ocid_img_files)
    assert n_imgs > 0
    
    # Load test examples into tensors on GPU
    device = ins_seg.device
    rgb_imgs = torch.zeros((n_imgs, 3, 480, 640), device=device) 
    xyz_imgs = torch.zeros((n_imgs, 3, 480, 640), device=device) 
    label_imgs = np.zeros((n_imgs, 480, 640), dtype=np.uint8)
    for i, img_file in enumerate(osd_img_files + ocid_img_files):
        data = np.load(img_file, allow_pickle=True, encoding='bytes').item()
        rgb_imgs[i] = standardize_image_rgb(data["rgb"], device=device)
        xyz_imgs[i] = standardize_image_xyz(data["xyz"], device=device)
        label_imgs[i] = data["label"]
    
    # Warm-up GPU
    ins_seg.segement(xyz_imgs[0], rgb_imgs[0])

    # Run instance segmentation module on examples
    st_time = time.time()
    cluster_img_list = []
    for rgb_img, xyz_img in zip(rgb_imgs, xyz_imgs):
        cluster_img, _ = ins_seg.segement(xyz_img, rgb_img)
        cluster_img_list.append(cluster_img)
    total_time = time.time() - st_time
    print(f'Total time taken for Segmentation: {total_time:.3f} seconds')
    print(f'FPS: {(n_imgs / total_time):.3f} images/sec')

    # Get results as NumPy arrays
    cluster_img_list = [t.cpu().numpy() for t in cluster_img_list]
    rgb_imgs_np = np.zeros((n_imgs, 480, 640, 3), dtype=np.uint8)
    for i, rgb_img in enumerate(rgb_imgs):
        rgb_imgs_np[i] = unstandardize_image_rgb(rgb_img)

    # Plot a comparison of predicted and GT image masks
    for i in range(n_imgs):
        num_objs = max(np.unique(cluster_img_list[i]).max(), np.unique(label_imgs[i]).max()) + 1
        rgb = rgb_imgs_np[i]
        depth = xyz_imgs[i, 2].cpu().numpy()
        seg_mask_plot = get_color_mask(cluster_img_list[i], nc=num_objs)
        gt_masks = get_color_mask(label_imgs[i], nc=num_objs)
        
        images = [rgb, depth, seg_mask_plot, gt_masks]
        titles = [
            f'Image {i+1}',
            'Depth',
            f"Refined Masks. #objects: {np.unique(cluster_img_list[i]).shape[0]-1}",
            f"Ground Truth. #objects: {np.unique(label_imgs[i]).shape[0]-1}"
        ]

        plt.figure(i+1, figsize=(4*5, 5))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.imshow(images[i])
            plt.title(titles[i])
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()