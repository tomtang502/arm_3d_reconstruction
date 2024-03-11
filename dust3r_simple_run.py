import sys, os
dust3r_local_dir = os.path.dirname(os.path.realpath(__file__)) + "/dust3r"
sys.path.append(dust3r_local_dir)
from scipy.spatial.transform import Rotation as R
from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from utils.graph_util import graph_single_struct, graph_double_struct
from observation_poses_config import ExperimentConfigs
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
from trace_viz import *

import cv2, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from geometric_util import *

using_saved = True
pts_path = 'dust3r_saved_output/pts_tor.pt'
cam_poses_path = 'dust3r_saved_output/cam_poses.pt'
"""
cv2.caliberatehandeye
his x and z might be flipped
"""
def get_file_paths(folder_path):
    file_paths = []  # List to store file paths
    # Walk through all files and directories in the specified folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)  # Create full file path
            file_paths.append(file_path)  # Add file path to list
    # Sort the list based on filenames
    file_paths.sort(key=lambda path: os.path.basename(path))
    return file_paths

if not using_saved:
    device="cuda"
    model_path = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = load_model(model_path, device)
    folder_path = 'arm_captured_images/xyz2each_10imgs_sa'
    file_paths = get_file_paths(folder_path)#[:4]
    images = load_images(file_paths, size=512)

    schedule = 'cosine'
    lr = 0.01 # 0.01
    niter = 320
    batch_size = 4

    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    print(type(poses))
    print(poses.shape)
    print(poses)
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    pts_list=[]
    for i in range(len(imgs)):
        sel_pts=pts3d[i][confidence_masks[i]]
        pts_list.append(sel_pts)
    pts_tor=torch.cat(pts_list).detach().cpu()
    poses = poses.detach().cpu()
    # Save each tensor separately
    torch.save(pts_tor, pts_path)
    torch.save(poses, cam_poses_path)
else:
    # Load each tensor separately
    # import trimesh

    # # Load the scene from a file
    # scene = trimesh.load('dust3r_saved_output/scene.glb')
    # #focals = scene.get_focals()
    # poses = scene.get_im_poses().detach().cpu()
    # pts_tor = scene.get_pts3d().detach().cpu()
    pts_tor = torch.load(pts_path)
    poses = torch.load(cam_poses_path)

def extract_positions_directions(transform_matrices):
    positions = transform_matrices[:, :3, 3]
    forward_directions = transform_matrices[:, :3, 2] # Negate to if cam face -Z direction
    return positions, forward_directions

positions, directions = extract_positions_directions(poses)

scale_factor=(0.05)/(0.5*(torch.norm(positions[4]-positions[3])+torch.norm(positions[3]-positions[2])))
# eef_poses = [
#     [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
#     [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
#     [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
#     [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
#     [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
#     [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2
    
#     [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157], # left side 1
#     [2.28403, 0.05687, 0.21994, 0.40305, 0.46293, 0.37193], # left side 2
#     [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
#     [-2.28403, 0.05687, -0.21994, 0.40305, -0.46293, 0.37193] # right side 2
# ]
# eef_poses = [
#     [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
#     [3.008, -0.5914, 0.633, -0.111, -0.2634, 0.485], # right back 1
#     [-2.31092, -0.24056, -0.99504, -0.21117, -0.12188, 0.41125], #right back 2
#     [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
#     [-3.008, -0.668, -0.549, -0.16574, 0.268, 0.5065], # left back 0
#     [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
#     [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
#     [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
#     [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2
#     [2.39645, -0.54843, 0.08445, -0.00101, 0.49278, 0.42958], # left back corner

#     [1.721, 0.179, 0.2195, 0.326, 0.5645, 0.1535], # left side 0
#     [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157], # left side 1
#     [2.28403, 0.05687, 0.21994, 0.40305, 0.46293, 0.37193], # left side 2
#     [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
#     [-1.721, 0.179, -0.2195, 0.326, -0.5645, 0.1535], # right side 1
#     [-2.28403, 0.05687, -0.21994, 0.40305, -0.46293, 0.37193] # right side 2
# ]

exp_config = ExperimentConfigs()

experiment_tag = "xyz2linear_2backsym_3side_sa"
pose_data = exp_config.get_config(experiment_tag)
eef_poses = pose_data.poses 

print(len(eef_poses), len(eef_poses[0]), poses.shape)
eef_pos_np = np.array(eef_poses)[:, 3:]
eef_poses_6d=torch.tensor(eef_poses)
eef_mats=pose_to_transform(eef_poses_6d)

poses_viz = poses_convert_to_viz(poses.numpy())
print("poses_viz shape:", poses_viz.shape)
world2cam = poses_viz

world2cam = np.linalg.pinv(np.array(world2cam))

poses_L = [p for p in poses]
cam_xyz_L = np.stack(transform_list_of_matrices_to_xyz(poses_L))
#cam_xyz_L = flip_axis(cam_xyz_L)
print(cam_xyz_L)
graph_single_struct(cam_xyz_L)
#graph_single_struct(pts_tor)

scale_factor = scale_factor.numpy()

world2cam = poses
world2cam[:,:3,3]=world2cam[:,:3,3]*scale_factor
print(world2cam.shape)
world2cam = world2cam.numpy()
# world-space coordinates to camera pose
#poses_scaled_np=np.array(poses_scaled.cpu().detach())
# robot base to manipulator end-effector pose
eef_mats_np=np.array(eef_mats.cpu().detach())
R_base2world, t_base2world, R_gripper2cam, t_gripper2cam =cv2.calibrateRobotWorldHandEye(world2cam[:,:3,:3], 
                                                                                         world2cam[:,:3,3],
                                                                                         eef_mats_np[:,:3,:3],
                                                                                         eef_mats_np[:,:3,3])
"""
    The handeye calib solves matmul(T_A, T_X) = matmul(T_Z, T_B)
    T_A: world to cam transformation (poses_scaled_np from dust3r)
    T_X: base to world transformation (this is formed by the output R_base2wofig.update_layout(scene=dict(
#                     xaxis=dict(range=[-1., 1.]),
#                     yaxis=dict(range=[-1., 1.]),
#                     zaxis=dict(range=[-1., 1.])
# ))r2cam and t_gripper2cam)
    T_B: base to gripper transformation (eef_mats_np or end effector from arm)
    ** All of those are transformation matrices **
    ** The "output" here refer to the out put generated by cv2.calibrateRobotWorldHandEye

    Evaluation criteria
    Compute the residual error for the rotation angles that comes from rotation matrix AR, ZB
    
    Parameters:
    R (base to world): torch.Tensor
        The estimated rotation matrix of shape (3, 3).
    A (world to cam), B (base to gripper): torch.Tensor
        3D rotation tensors of shape (N, 3, 3) representing sequences of 
        transformations A and B respectively.
    Z (gripper to cam): torch.Tensor 3D rotation tensors of shape (N, 3, 3) 
        representing sequences of rotation respectively.
    
    Returns:
    error : float
        The average residual error.
"""
print(residual_error(R_base2world, world2cam[:,:3,:3], eef_mats_np[:,:3,:3], R_gripper2cam))


def create_transformation_matrix(rotation_matrix, translation_vector):
    """
    Create a transformation matrix from a rotational matrix and a translational vector.
    
    Args:
    - rotation_matrix: 3x3 numpy array representing the rotational matrix
    - translation_vector: 3x1 numpy array representing the translational vector
    
    Returns:
    - transformation_matrix: 4x4 numpy array representing the transformation matrix
    """
    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)
    
    # Assign the rotational matrix to the top-left 3x3 submatrix
    transformation_matrix[:3, :3] = rotation_matrix
    
    # Assign the translational vector to the rightmost column
    transformation_matrix[:3, 3] = translation_vector
    
    return transformation_matrix

"""
Given a list of word to cam transformation matrix
"""
T_A = world2cam # world to cam

R_X, t_X = R_base2world, t_base2world
print(t_X.shape)
T_X = create_transformation_matrix(R_X, np.squeeze(t_X))

R_Z, t_Z = R_gripper2cam, t_gripper2cam
T_Z = create_transformation_matrix(R_Z, np.squeeze(t_Z))

T_B = eef_mats_np


"""
we are give T_A, trying to get T_B
We can use matmul(T_A, T_X) = matmul(T_Z, T_B), which gives T_B = inv(T_Z)T_A@T_X
T_A: world to cam transformation (poses_scaled_np from dust3r)
T_X: base to world transformation (this is formed by the output R_base2world and t_base2world)
T_Z: gripper to cam transformation (this is formed by the output R_gripper2cam and t_gripper2cam)
T_B: base to gripper transformation (eef_mats_np or end effector from arm)
"""
# Transform point cloud points
pts = pts_tor.numpy()
print("point cloud dim:", pts_tor.shape)
pts_b = []
for p in pts[::100]:
    pts_b.append(apply_transform_pt(p, np.linalg.pinv(T_X)))
pts_b = np.stack(pts_b)
print("After transformed with skipping points:", pts_b.shape)

# Transform camera
positions_cam=poses[:,:3,3]
cam_pts_b = []
for p in positions_cam:
    #combined_transformation_matrix = np.matmul(np.linalg.pinv(T_Z), np.matmul(p, T_X))
    cam_pts_b.append(apply_transform_pt(p, np.linalg.pinv(T_X)))
cam_pts_b = np.stack(cam_pts_b)
print(cam_pts_b.shape)
graph_single_struct(cam_pts_b)
# cam_pts_b = flip_axis(np.stack(cam_pts_b))
# print(cam_pts_b.shape)

def align_point_cloud(point_cloud):
    """
    Aligns a point cloud with its principal axes.

    Parameters:
    - point_cloud: A torch tensor of shape (n, 3) representing the point cloud.

    Returns:
    - A torch tensor of shape (4, 4) representing the transformation matrix.
    """
    # Center the point cloud
    mean = point_cloud.mean(dim=0)
    centered_point_cloud = point_cloud - mean

    # Compute the covariance matrix
    cov_matrix = torch.matmul(centered_point_cloud.T, centered_point_cloud) / (centered_point_cloud.shape[0] - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Create the rotation matrix (3x3)
    rotation_matrix = eigenvectors

    # Create the transformation matrix (4x4)
    transformation_matrix = torch.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix

    # Add translation to align the original point cloud
    translation = -torch.matmul(rotation_matrix.T, mean)
    transformation_matrix[:3, 3] = translation

    return transformation_matrix

pts_b_tor = torch.tensor(pts_b).float()
transformation_matrix = align_point_cloud(pts_b_tor)
# To transform the point cloud, append 1s to the original point cloud to make it (n, 4)

homogeneous_point_cloud = torch.cat((pts_b_tor, torch.ones(pts_b.shape[0], 1)), dim=1)
aligned_point_cloud = torch.matmul(transformation_matrix, homogeneous_point_cloud.T).T
#graph_single_struct(aligned_point_cloud)
#graph_double_struct(cam_pts_b, pts_b)

# reassign variable name to align with Will's name format 
tr_pts = pts_b
# cam_pts=cam_pts_b
# fig = go.Figure()
# # X
# fig.add_trace(go.Scatter3d(x=eef_pos_np[:, 0], y=eef_pos_np[:, 1], z=eef_pos_np[:, 2],
#                            mode='markers', marker=dict(size=5, color='green'), name='eef Positions'))

# # Camera positions
# fig.add_trace(go.Scatter3d(x=cam_pts[:, 0], y=cam_pts[:, 1], z=cam_pts[:, 2],
#                            mode='markers', marker=dict(size=5, color='blue'), name='Camera Positions'))

# fig.add_trace(go.Scatter3d(x=tr_pts[::5, 0],  # X coordinates
#     y=tr_pts[::5, 1],  # Y coordinates
#     z=tr_pts[::5, 2],  # Z coordinates
#                            mode='markers', marker=dict(size=0.5, color='red'),name="point cloud"))

# # fig.update_layout(scene=dict(
# #                     xaxis=dict(range=[-1., 1.]),
# #                     yaxis=dict(range=[-1., 1.]),
# #                     zaxis=dict(range=[-1., 1.])
# # ))

# fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
#                   title="Camera Positions and Orientations")

# fig.update_layout(width=1000, height=700)
# fig.show()
