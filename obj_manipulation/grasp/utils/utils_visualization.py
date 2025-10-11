from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from numpy import ndarray


def visualize_grasps(
    xyz_pc: ndarray,
    rgb_pc: ndarray,
    pred_grasps: List[ndarray],
    pred_scores: List[ndarray],
    pred_widths: List[ndarray],
    gripper_depth: float,
    plot_cam_coords: bool = False,
    T_world_cam: ndarray = np.eye(4),
) -> None:
    """Visualizes colored point cloud and predicted grasps. Thick grasp is most confident per object.

    Args:
        xyz_pc: [N x 3] array containing the full scene point cloud.
        rgb_pc: [N x 3] array (uint8) containing the colors of each point in xyz_pc.
        pred_grasps: List of [M x 4 x 4] arrays of homogeneous matrices representing grasp poses in
            the camera coordinate system.
        pred_scores: List of [M x 1] arrays of grasp success probabilities.
        pred_widths: List of [M x 1] arrays of grasp widths for each grasp predicition.
        gripper_depth: Gripper depth in meters. Distance from TCP to grasp point along approach axis.
        plot_cam_coords: Plot camera coordinate frame.
        T_world_cam: Homogeneous matrix representing pose of camera coordinate frame w.r.t. world frame.
    """
    print('Visualizing grasps...')

    # Add colored raw point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_pc)
    pcd.colors = o3d.utility.Vector3dVector(rgb_pc.astype(np.float64) / 255)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Add camera coordinate frame
    if plot_cam_coords:
        plot_coordinates(vis, np.zeros(3), np.eye(3,3), origin_color=(0.5, 0.5, 0.5))
        # This is world in cam frame (everything is visualized in the camera frame)
        T_cam_world = np.linalg.inv(T_world_cam)
        t, r = T_cam_world[:3, 3], T_cam_world[:3, :3]
        plot_coordinates(vis, t, r)

    # Get colors for best grasp (at position 0) and rest of grasps
    cm = plt.get_cmap('rainbow')
    cm2 = plt.get_cmap('viridis')
    colors = [cm(1.0 * i / len(pred_grasps))[:3] for i in range(len(pred_grasps))]
    colors2 = [cm2(0.5 * pred_scores[i][0, 0])[:3] for i in range(len(pred_grasps))]

    for i, (grasp_tfs, grasp_ws) in enumerate(zip(pred_grasps, pred_widths)):
        if np.any(grasp_tfs):
            if len(pred_grasps) > 1:
                # All grasps share same color except best one
                draw_grasps(vis, grasp_tfs, grasp_ws, gripper_depth, colors=[colors[i]])
                draw_grasps(vis, grasp_tfs[:1], grasp_ws[:1], gripper_depth, colors=[colors2[i]])
            else:
                # Vary grasp colors according to their scores except best one (uses red)
                min_score, max_score = pred_scores[i][-1, 0], pred_scores[i][0, 0]
                colors3 = [cm2((score - min_score) / (max_score - min_score))[:3] 
                           for score in pred_scores[i][:, 0]]
                draw_grasps(vis, grasp_tfs, grasp_ws, gripper_depth, colors=colors3)
                draw_grasps(vis, grasp_tfs[:1], grasp_ws[:1], gripper_depth, colors=[(1, 0, 0)])

    vis.run()
    vis.destroy_window()
    return


def plot_coordinates(vis, t: ndarray, r: ndarray, origin_color: Optional[List[int]] = None) -> None:
    """Plots coordinate frame in Open3D visualization window.

    Args:
        t: [3] translation vector.
        r: [3 x 3] rotation matrix.
        origin_color: Color to use for sphere at the origin of coordinate frame.
        
    """
    # Visualize sphere at the origin
    if origin_color is not None:
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        ball.paint_uniform_color(np.array(origin_color))
        vis.add_geometry(ball)

    # Create a line for each axis of the coordinate frame
    lines = []
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Red, Green, Blue
    for i in range(3):
        line_st = t.tolist()
        line_end = (t + 0.2 * r[:, i]).tolist()

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([line_st, line_end])
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        line.colors = o3d.utility.Vector3dVector(np.array([colors[i]]))
        line.paint_uniform_color(colors[i])  # Set line color
        lines.append(line)

    # Visualize the lines in the Open3D visualizer
    for line in lines:
        vis.add_geometry(line)


def draw_grasps(
    vis,
    grasp_tfs: ndarray,
    grasp_widths: ndarray,
    gripper_depth: float,
    colors: List[Tuple[int]] = [(0.0, 1.0, 0.0)],
) -> None:
    """Draws wireframe grasps from given grasp pose and with given gripper dimensions.

    Args:
        gripper_tfs: [N x 4 x 4] array of grasp homegeneous transformations.
        gripper_widths: [N x 1] array of grasp widths.
        gripper_depth: Gripper depth in meters. Distance from TCP to grasp point along approach axis.
        colors: List of colors for each grasp or common color for all grasps.
    """    
    # Get gripper line plots for all grasps
    line_plots = get_gripper_lineplots(grasp_widths, gripper_depth)
    line_plots = transform_lineplots(grasp_tfs, line_plots)
    n_grasps, n_points = line_plots.shape[:2]
    connections_base = np.stack(
        [np.arange(0, n_points - 1), np.arange(1, n_points)], axis=1
    )
    connections_offsets = np.arange(0, n_grasps)[:, None, None] * n_points
    connections = connections_base + connections_offsets
    
    # Plot gripper line plots
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_plots.reshape(-1, 3))
    line_set.lines = o3d.utility.Vector2iVector(connections.reshape(-1, 2))
    if len(colors) == 1:
        colors = np.vstack(colors).astype(np.float64)
        colors = np.repeat(colors, n_grasps, axis=0)
    elif len(colors) == n_grasps:
        colors = np.vstack(colors).astype(np.float64)
    else:
        raise ValueError('Number of colors must be 1 or equal to number of grasps')
    colors = np.repeat(colors, n_points - 1, axis=0)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)


def get_gripper_lineplots(grasp_widths: ndarray, gripper_depth: float) -> ndarray:
    """Create lineplots for gripper visualization according to its width and depth.
    
    Args:
        gripper_widths: [N x 1] array of grasp widths.
        gripper_depth: Gripper depth in meters. Distance from TCP to grasp point along approach axis.
    
    Returns:
        [N x 7 x 3] array containing the 3D coordinates of the 7 points that form the gripper line
        plot for each width.
    """
    n_grasps = grasp_widths.shape[0]
    grasp_dir = np.array([1.0, 0.0, 0.0])
    approach_dir = np.array([0.0, 0.0, 1.0])

    tcp = np.zeros((n_grasps, 3))
    mid = tcp + approach_dir * gripper_depth / 2
    mid_l = mid - grasp_dir * grasp_widths / 2
    mid_r = mid + grasp_dir * grasp_widths / 2
    gri_l = mid_l + approach_dir * gripper_depth / 2
    gri_r = mid_r + approach_dir * gripper_depth / 2
    
    line_plots = np.stack([tcp, mid, mid_l, gri_l, mid_l, mid_r, gri_r], axis=1)
    return line_plots


def transform_lineplots(grasp_tfs: ndarray, line_plots: ndarray) -> ndarray:
    """Apply homogeneous transformation to gripper line plots.
    
    Args:
        gripper_tfs: [N x 4 x 4] array of grasp homegeneous transformations.
        line_plots: [N x 7 x 3] array of gripper line plots expressed in gripper frame.
    
    Returns:
        [N x 7 x 3] array of gripper line plots expressed in camera frame.
    """
    line_plots = line_plots @ grasp_tfs[:, :3, :3].transpose(0, 2, 1)
    line_plots += np.expand_dims(grasp_tfs[:, :3, 3], axis=1)
    return line_plots
