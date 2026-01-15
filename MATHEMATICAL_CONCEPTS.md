# Mathematical Concepts and Algorithms

This document provides a comprehensive mathematical explanation of the algorithms and concepts used in the object manipulation system. It covers grasp estimation, segmentation, point cloud processing, and motion planning with detailed equations and formulations.

---

## Table of Contents

1. [Grasp Estimation](#1-grasp-estimation)
2. [Segmentation Algorithms](#2-segmentation-algorithms)
3. [Point Cloud Processing](#3-point-cloud-processing)
4. [Motion Planning and Transformations](#4-motion-planning-and-transformations)
5. [Camera Models and Projections](#5-camera-models-and-projections)
6. [References](#6-references)

---

## 1. Grasp Estimation

The grasp estimation module uses learning-based methods (Contact-GraspNet) to predict feasible grasp poses for unseen objects based on RGB-D input.

### 1.1 Depth-to-3D Coordinate Conversion

Given a depth image $D$ and camera intrinsic parameters, we convert 2D pixel coordinates $(u, v)$ to 3D coordinates $(X, Y, Z)$ in the camera frame:

$$
\begin{aligned}
X &= \frac{(u - c_x) \cdot Z}{f_x} \\
Y &= \frac{(v - c_y) \cdot Z}{f_y} \\
Z &= D(u, v)
\end{aligned}
$$

where:
- $(f_x, f_y)$ are the focal lengths
- $(c_x, c_y)$ is the principal point
- $D(u, v)$ is the depth value at pixel $(u, v)$

**Camera Intrinsic Matrix:**

$$
K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

**Implementation:** `obj_manipulation/grasp/utils/utils.py:15-43`

---

### 1.2 6-DoF Grasp Pose Construction

A 6-DoF grasp pose is represented as a homogeneous transformation matrix $T \in SE(3)$:

$$
T = \begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

where $R \in SO(3)$ is a 3×3 rotation matrix and $\mathbf{t} \in \mathbb{R}^3$ is the translation vector.

#### 1.2.1 Gram-Schmidt Orthonormalization

Given two non-orthogonal direction vectors $\mathbf{d}_1$ (grasp direction) and $\mathbf{d}_2$ (approach direction), we orthonormalize them:

$$
\begin{aligned}
\mathbf{v}_1 &= \frac{\mathbf{d}_1}{||\mathbf{d}_1||_2} \\
\text{proj}_{\mathbf{v}_1}(\mathbf{d}_2) &= (\mathbf{d}_2 \cdot \mathbf{v}_1) \mathbf{v}_1 \\
\mathbf{v}_2 &= \frac{\mathbf{d}_2 - \text{proj}_{\mathbf{v}_1}(\mathbf{d}_2)}{||\mathbf{d}_2 - \text{proj}_{\mathbf{v}_1}(\mathbf{d}_2)||_2}
\end{aligned}
$$

#### 1.2.2 Rotation Matrix Construction

The rotation matrix $R$ is constructed using the grasp direction, approach direction, and their cross product:

$$
R = \begin{bmatrix}
| & | & | \\
\mathbf{g} & \mathbf{g} \times \mathbf{a} & \mathbf{a} \\
| & | & |
\end{bmatrix}
$$

where:
- $\mathbf{g}$ is the normalized grasp direction (closing direction of gripper)
- $\mathbf{a}$ is the normalized approach direction (perpendicular to gripper jaws)
- $\mathbf{g} \times \mathbf{a}$ is the binormal direction (along gripper jaws)

#### 1.2.3 Grasp Translation Calculation

The grasp position accounts for gripper geometry:

$$
\mathbf{t} = \mathbf{p}_c + \frac{w}{2} \mathbf{g} - d \mathbf{a}
$$

where:
- $\mathbf{p}_c$ is the contact point on the object surface
- $w$ is the gripper width (0.08 m for DH AG95)
- $d$ is the gripper depth (0.1034 m)
- $\mathbf{g}$ is the grasp direction vector
- $\mathbf{a}$ is the approach direction vector

**Implementation:** `obj_manipulation/grasp/graspnet.py:259-303`

---

### 1.3 Farthest Point Sampling (FPS)

FPS is a greedy algorithm for selecting a subset of $N$ points from a point cloud $\mathcal{P}$ that maximizes coverage:

**Algorithm:**
1. Initialize: Select random starting point $\mathbf{p}_0 \in \mathcal{P}$
2. For $i = 1$ to $N-1$:
   $$
   \mathbf{p}_i = \arg\max_{\mathbf{p} \in \mathcal{P}} \min_{j < i} ||\mathbf{p} - \mathbf{p}_j||_2
   $$
3. Return $\{\mathbf{p}_0, \mathbf{p}_1, \ldots, \mathbf{p}_{N-1}\}$

**Computational Complexity:** $O(N \cdot |\mathcal{P}|)$

**Implementation:** `obj_manipulation/grasp/utils/utils_pointnet.py:57-87`

---

### 1.4 Pairwise Squared Euclidean Distance

For efficient computation of distances between point sets $A \in \mathbb{R}^{n \times d}$ and $B \in \mathbb{R}^{m \times d}$:

$$
D_{ij} = ||\mathbf{a}_i - \mathbf{b}_j||_2^2 = \sum_{k=1}^{d} (a_{ik} - b_{jk})^2
$$

**Matrix Formulation:**

$$
D = -2 A B^T + \mathbf{1}_n \text{diag}(B B^T)^T + \text{diag}(A A^T) \mathbf{1}_m^T
$$

where $\mathbf{1}_n$ and $\mathbf{1}_m$ are column vectors of ones.

**Implementation:** `obj_manipulation/grasp/utils/utils_pointnet.py:12-22`

---

## 2. Segmentation Algorithms

### 2.1 Gaussian Mean-Shift Clustering

Mean-shift is a non-parametric clustering algorithm that finds modes of a probability density function.

#### 2.1.1 Gaussian Kernel Density Estimation

Given a set of points $\{\mathbf{x}_i\}_{i=1}^{N}$, the kernel density estimate at point $\mathbf{x}$ is:

$$
f(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} K_H(\mathbf{x} - \mathbf{x}_i)
$$

where $K_H$ is the Gaussian kernel with bandwidth $H$:

$$
K_H(\mathbf{u}) = \frac{1}{(2\pi)^{d/2} |H|^{1/2}} \exp\left(-\frac{1}{2} \mathbf{u}^T H^{-1} \mathbf{u}\right)
$$

For isotropic kernels with bandwidth $\sigma$:

$$
K(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{||\mathbf{x} - \mathbf{y}||_2^2}{2\sigma^2}\right)
$$

#### 2.1.2 Mean-Shift Update Rule

The mean-shift algorithm iteratively updates each point towards the local mode:

$$
\mathbf{x}^{(t+1)} = \frac{\sum_{i=1}^{N} \mathbf{x}_i K(\mathbf{x}^{(t)}, \mathbf{x}_i)}{\sum_{i=1}^{N} K(\mathbf{x}^{(t)}, \mathbf{x}_i)}
$$

This is equivalent to gradient ascent on the density estimate.

#### 2.1.3 Convergence Criterion

The algorithm stops when the shift magnitude falls below a threshold:

$$
||\mathbf{x}^{(t+1)} - \mathbf{x}^{(t)}||_2 < \epsilon
$$

where $\epsilon$ is typically set to 0.01.

#### 2.1.4 Connected Components Grouping

After convergence, points are grouped into clusters based on their converged positions:

$$
C_k = \{\mathbf{x}_i : ||\text{mode}(\mathbf{x}_i) - \text{mode}(\mathbf{x}_j)||_2 < \delta, \forall \mathbf{x}_j \in C_k\}
$$

where $\delta$ is the grouping threshold and $\text{mode}(\mathbf{x})$ is the converged position of point $\mathbf{x}$.

**Implementation:** `obj_manipulation/segment/cluster.py:113-138` (core algorithm), full method at lines 178-204

---

### 2.2 Morphological Operations

Morphological operations are used for mask refinement and noise removal.

#### 2.2.1 Dilation

Dilation expands the boundaries of foreground regions:

$$
(A \oplus B)(\mathbf{p}) = \max_{\mathbf{q} \in B} A(\mathbf{p} - \mathbf{q})
$$

where $A$ is the binary image and $B$ is the structuring element.

#### 2.2.2 Erosion

Erosion shrinks the boundaries:

$$
(A \ominus B)(\mathbf{p}) = \min_{\mathbf{q} \in B} A(\mathbf{p} + \mathbf{q})
$$

#### 2.2.3 Opening

Opening removes small objects and smooths boundaries:

$$
A \circ B = (A \ominus B) \oplus B
$$

#### 2.2.4 Closing

Closing fills small holes and connects nearby regions:

$$
A \bullet B = (A \oplus B) \ominus B
$$

**Implementation:** `obj_manipulation/segment/utils.py:87-121`

---

### 2.3 Point Cloud Filtering

#### 2.3.1 Pass-Through Filtering

Points are filtered based on axis-aligned bounding box constraints:

$$
\mathcal{P}_{\text{filtered}} = \{\mathbf{p} \in \mathcal{P} : a_i \leq p_i \leq b_i, \forall i \in \{x, y, z\}\}
$$

where $[a_i, b_i]$ defines the valid range for each axis.

#### 2.3.2 Voxel Grid Downsampling

Points are grouped into 3D voxels of size $\ell$, and each voxel is represented by the centroid:

$$
\mathbf{c}_v = \frac{1}{|\mathcal{P}_v|} \sum_{\mathbf{p} \in \mathcal{P}_v} \mathbf{p}
$$

where $\mathcal{P}_v$ is the set of points in voxel $v$.

---

## 3. Point Cloud Processing

### 3.1 Ball Query

Ball query finds all points within radius $r$ of a query point $\mathbf{q}$:

$$
\mathcal{N}(\mathbf{q}, r) = \{\mathbf{p} \in \mathcal{P} : ||\mathbf{p} - \mathbf{q}||_2 \leq r\}
$$

This is used to define local neighborhoods for feature extraction.

**Implementation:** `obj_manipulation/grasp/utils/utils_pointnet.py`

---

### 3.2 Local Coordinate Transformation

Points in a neighborhood are transformed to a local coordinate frame centered at the query point:

$$
\mathbf{p}_{\text{local}} = \mathbf{p} - \mathbf{q}
$$

This makes features translation-invariant.

---

### 3.3 Hierarchical Point Cloud Abstraction

PointNet++ uses a hierarchical structure with Set Abstraction (SA) layers:

#### 3.3.1 Set Abstraction Layer

Each SA layer performs three operations:
1. **Sampling:** Use FPS to select $N'$ centroids from $N$ points
2. **Grouping:** Use ball query to form neighborhoods around centroids
3. **PointNet:** Extract features from each neighborhood using a mini-PointNet

**Formal Definition:**

$$
\mathcal{F}_l = \text{PointNet}(\{\mathcal{N}(\mathbf{c}_i, r_l)\}_{i=1}^{N_l})
$$

where:
- $\mathcal{F}_l$ is the feature set at layer $l$
- $\mathbf{c}_i$ are the sampled centroids
- $r_l$ is the ball query radius
- $N_l$ is the number of centroids at layer $l$

---

### 3.4 Multi-Scale Grouping (MSG)

To handle varying point densities, multiple scales are processed in parallel:

$$
\mathcal{F}_{\text{MSG}} = \text{Concat}(\text{PointNet}(\mathcal{N}_1), \text{PointNet}(\mathcal{N}_2), \ldots, \text{PointNet}(\mathcal{N}_k))
$$

where $\mathcal{N}_i$ represents neighborhoods at different scales (radii).

---

## 4. Motion Planning and Transformations

### 4.1 Homogeneous Transformations

Rigid body transformations in 3D space are represented using the Special Euclidean group $SE(3)$:

$$
T = \begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix} \in SE(3)
$$

where $R \in SO(3)$ is the rotation matrix and $\mathbf{t} \in \mathbb{R}^3$ is the translation vector.

#### 4.1.1 Composition of Transformations

The composition of two transformations $T_1$ and $T_2$ is:

$$
T_3 = T_1 \cdot T_2 = \begin{bmatrix}
R_1 R_2 & R_1 \mathbf{t}_2 + \mathbf{t}_1 \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

#### 4.1.2 Inverse Transformation

The inverse of a transformation is:

$$
T^{-1} = \begin{bmatrix}
R^T & -R^T \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
$$

**Implementation:** `scripts/xarm_move.py`

---

### 4.2 Euler Angles and Quaternions

#### 4.2.1 Euler Angles (Roll-Pitch-Yaw)

Rotation can be decomposed into three sequential rotations:

$$
R(\phi, \theta, \psi) = R_z(\psi) R_y(\theta) R_x(\phi)
$$

where:
- $\phi$ (roll) is rotation about the x-axis
- $\theta$ (pitch) is rotation about the y-axis
- $\psi$ (yaw) is rotation about the z-axis

**Rotation Matrices:**

$$
R_x(\phi) = \begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\phi & -\sin\phi \\
0 & \sin\phi & \cos\phi
\end{bmatrix}
$$

$$
R_y(\theta) = \begin{bmatrix}
\cos\theta & 0 & \sin\theta \\
0 & 1 & 0 \\
-\sin\theta & 0 & \cos\theta
\end{bmatrix}
$$

$$
R_z(\psi) = \begin{bmatrix}
\cos\psi & -\sin\psi & 0 \\
\sin\psi & \cos\psi & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

#### 4.2.2 Quaternion Representation

A unit quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ represents rotation:

$$
\mathbf{q} = \cos\frac{\alpha}{2} + (q_x \mathbf{i} + q_y \mathbf{j} + q_z \mathbf{k}) \sin\frac{\alpha}{2}
$$

where $\alpha$ is the rotation angle and $(q_x, q_y, q_z)$ is the unit rotation axis.

**Constraint:** $q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1$

#### 4.2.3 Euler to Quaternion Conversion

$$
\begin{aligned}
q_w &= \cos\frac{\phi}{2}\cos\frac{\theta}{2}\cos\frac{\psi}{2} + \sin\frac{\phi}{2}\sin\frac{\theta}{2}\sin\frac{\psi}{2} \\
q_x &= \sin\frac{\phi}{2}\cos\frac{\theta}{2}\cos\frac{\psi}{2} - \cos\frac{\phi}{2}\sin\frac{\theta}{2}\sin\frac{\psi}{2} \\
q_y &= \cos\frac{\phi}{2}\sin\frac{\theta}{2}\cos\frac{\psi}{2} + \sin\frac{\phi}{2}\cos\frac{\theta}{2}\sin\frac{\psi}{2} \\
q_z &= \cos\frac{\phi}{2}\cos\frac{\theta}{2}\sin\frac{\psi}{2} - \sin\frac{\phi}{2}\sin\frac{\theta}{2}\cos\frac{\psi}{2}
\end{aligned}
$$

**Implementation:** Uses ROS `tf.transformations` library

---

### 4.3 Coordinate Frame Transformations

#### 4.3.1 Camera to World Frame

Transform a point from camera frame $\mathcal{F}_C$ to world frame $\mathcal{F}_W$:

$$
\mathbf{p}_W = T_{WC} \mathbf{p}_C = R_{WC} \mathbf{p}_C + \mathbf{t}_{WC}
$$

where $T_{WC}$ is the camera-to-world transformation matrix.

#### 4.3.2 World to Robot Base Frame

Transform from world frame to robot base frame $\mathcal{F}_B$:

$$
\mathbf{p}_B = T_{BW} \mathbf{p}_W
$$

#### 4.3.3 Forward Kinematics

The position of the end-effector in the base frame is:

$$
T_{BE} = T_{B1} T_{12} T_{23} \cdots T_{n-1,E}
$$

where $T_{i,i+1}$ is the transformation from joint $i$ to joint $i+1$.

---

### 4.4 Inverse Kinematics (IK)

Given a desired end-effector pose $T_d$, find joint angles $\mathbf{q} = [q_1, q_2, \ldots, q_n]^T$ such that:

$$
T_{BE}(\mathbf{q}) = T_d
$$

This is solved using:
1. **Analytical methods** (for specific robot geometries)
2. **Numerical methods** (e.g., Jacobian-based methods)
3. **MoveIt!** motion planning framework (used in this project)

---

### 4.5 Trajectory Planning

#### 4.5.1 Cartesian Path Planning

Given waypoints $\{\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_n\}$, generate a smooth trajectory:

$$
\mathbf{p}(t) = (1-t) \mathbf{p}_i + t \mathbf{p}_{i+1}, \quad t \in [0, 1]
$$

#### 4.5.2 Joint Space Planning

Generate a trajectory in joint space using the Open Motion Planning Library (OMPL):

$$
\mathbf{q}(t) : [0, T] \rightarrow \mathcal{C}
$$

where $\mathcal{C}$ is the configuration space of the robot.

**Constraints:**
- Collision-free: $\mathbf{q}(t) \in \mathcal{C}_{\text{free}}$
- Velocity limits: $||\dot{\mathbf{q}}(t)||_{\infty} \leq v_{\max}$
- Acceleration limits: $||\ddot{\mathbf{q}}(t)||_{\infty} \leq a_{\max}$

**Implementation:** MoveIt! with OMPL planners

---

## 5. Camera Models and Projections

### 5.1 Pinhole Camera Model

The pinhole camera model relates 3D world points to 2D image coordinates:

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R | \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

where:
- $K$ is the intrinsic matrix
- $[R | \mathbf{t}]$ is the extrinsic matrix (camera pose)
- $s$ is the scaling factor (depth)
- $(u, v)$ are pixel coordinates
- $(X_w, Y_w, Z_w)$ are world coordinates

### 5.2 Depth Projection

For RGB-D cameras (Intel RealSense), depth is directly measured:

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \begin{bmatrix} Z_c(u - c_x)/f_x \\ Z_c(v - c_y)/f_y \\ Z_c \end{bmatrix}
$$

where $(X_c, Y_c, Z_c)$ are coordinates in the camera frame.

### 5.3 Distortion Models

Real cameras have lens distortion modeled by radial and tangential components:

$$
\begin{aligned}
x_{\text{distorted}} &= x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1 xy + p_2(r^2 + 2x^2) \\
y_{\text{distorted}} &= y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y^2) + 2p_2 xy
\end{aligned}
$$

where:
- $r^2 = x^2 + y^2$ (radial distance)
- $k_1, k_2, k_3$ are radial distortion coefficients
- $p_1, p_2$ are tangential distortion coefficients

---

## 6. Key Parameters and Constants

### 6.1 Gripper Parameters (DH Robotics AG95)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Gripper Width | 0.08 m | Maximum opening width |
| Gripper Depth | 0.1034 m | Distance from wrist to gripper tip |
| Opening Margin | 0.005 m | Safety margin for grasp width |

### 6.2 Camera Parameters (Intel RealSense)

Typical values for Intel RealSense D435:
- Resolution: 640 × 480
- Focal length: ~380 pixels
- Depth range: 0.3 - 3.0 m
- Field of view: 69° × 42°

### 6.3 Point Cloud Processing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| FPS samples | 512 - 2048 | Number of points sampled by FPS |
| Ball query radius | 0.02 - 0.08 m | Neighborhood radius for feature extraction |
| Voxel size | 0.005 - 0.01 m | Voxel grid resolution |

### 6.4 Mean-Shift Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Bandwidth (σ) | 0.05 - 0.1 | Gaussian kernel bandwidth |
| Epsilon (ε) | 0.01 | Convergence threshold |
| Max iterations | 100 | Maximum iterations per point |

---

## 7. References

### 7.1 Grasp Estimation
- Sundermeyer, M., et al. (2021). "Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes." *ICRA 2021*.
- Mousavian, A., et al. (2019). "6-DOF GraspNet: Variational Grasp Generation for Object Manipulation." *ICCV 2019*.

### 7.2 Point Cloud Processing
- Qi, C. R., et al. (2017). "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." *CVPR 2017*.
- Qi, C. R., et al. (2017). "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." *NeurIPS 2017*.

### 7.3 Segmentation
- Comaniciu, D., & Meer, P. (2002). "Mean shift: A robust approach toward feature space analysis." *TPAMI 2002*.
- Kirillov, A., et al. (2023). "Segment Anything." *ICCV 2023*.

### 7.4 Motion Planning
- Sucan, I. A., & Chitta, S. (2018). "MoveIt! Motion Planning Framework." *IEEE RAM*.
- Kavraki, L. E., et al. (1996). "Probabilistic roadmaps for path planning in high-dimensional configuration spaces." *IEEE TRA*.

### 7.5 Robotics Mathematics
- Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.
- Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control*. Pearson.
- Siciliano, B., et al. (2009). *Robotics: Modelling, Planning and Control*. Springer.

---

## 8. Notation Reference

| Symbol | Description |
|--------|-------------|
| $\mathbb{R}^n$ | n-dimensional Euclidean space |
| $SO(3)$ | Special Orthogonal Group (3D rotations) |
| $SE(3)$ | Special Euclidean Group (3D rigid transformations) |
| $\mathcal{P}$ | Point cloud |
| $\mathbf{p}$ | 3D point |
| $R$ | Rotation matrix |
| $\mathbf{t}$ | Translation vector |
| $T$ | Transformation matrix |
| $K$ | Camera intrinsic matrix |
| $\mathbf{q}$ | Joint angles or quaternion (context-dependent) |
| $||\cdot||_2$ | Euclidean (L2) norm |
| $\nabla$ | Gradient operator |
| $\times$ | Cross product |
| $\circ$ | Composition operator |

---

*This document is based on the implementation in the `obj_manipulation` repository for the IFRoSLab project at ELTE University.*
