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

import matplotlib.pyplot as plt
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

import cv2, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
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

# imput as a list of [row, pitch, yaw]
def rpy_to_rot_matrix(ori):
    r = R.from_euler('XYZ', ori, degrees=False)
    return torch.tensor(r.as_matrix())
    
def pose_to_transform(pose_batch):
    # Unpack the pose components
    pos = pose_batch[:, 3:6]

    # Convert RPY to rotation matrices
    rotation_matrices = rpy_to_rot_matrix(pose_batch[:, :3])

    # Create the transformation matrices
    transform_matrices = torch.zeros((pose_batch.shape[0], 4, 4), dtype=torch.float)
    transform_matrices[:, :3, :3] = rotation_matrices
    transform_matrices[:, :3, 3] = pos
    transform_matrices[:, 3, 3] = 1.0
    return transform_matrices

def residual_error(R, A, B):
    """
    Compute the residual error for the rotation matrix R.
    
    Parameters:
    R : torch.Tensor
        The estimated rotation matrix of shape (3, 3).
    A, B : torch.Tensor
        3D rotation tensors of shape (N, 3, 3) representing sequences of 
        transformations A and B respectively.
    
    Returns:
    error : float
        The average residual error.
    """
    N = A.shape[0]
    errors = []
    for i in range(N):
        RA = R.mm(A[i])
        BR = B[i].mm(R)
        error = (RA - BR).norm()
        errors.append(error.item())
    return sum(errors) / N

device="cuda"
model_path = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
model = load_model(model_path, device)
folder_path = 'arm_captured_images/2bs2sslb3_sa'
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
pts_tor=torch.cat(pts_list)

def extract_positions_directions(transform_matrices):
    positions = transform_matrices[:, :3, 3]
    forward_directions = transform_matrices[:, :3, 2] # Negate to if cam face -Z direction
    return positions, forward_directions

positions, directions = extract_positions_directions(poses)
positions=positions.detach().cpu()
#directions=directions.detach().cpu()

# fig = go.Figure()

# # Camera positions
# fig.add_trace(go.Scatter3d(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
#                            mode='markers', marker=dict(size=5, color='blue'), name='Camera Positions'))

# fig.add_trace(go.Scatter3d(x=pts_tor[::10, 0].detach().cpu(),  # X coordinates
#     y=pts_tor[::10, 1].detach().cpu(),  # Y coordinates
#     z=pts_tor[::10, 2].detach().cpu(),  # Z coordinates
#                            mode='markers', marker=dict(size=0.5, color='red'),name="point cloud"))

# # Directions
# for position, direction in zip(positions, directions):
#     fig.add_trace(go.Cone(x=[position[0]], y=[position[1]], z=[position[2]],
#                           u=[direction[0]], v=[direction[1]], w=[direction[2]],
#                           anchor="tail", showscale=False, sizeref=0.02))

# fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
#                   title="Camera Positions and Orientations")

# fig.update_layout(width=1000, height=700)
# fig.show()


scale_factor=(0.05)/(0.5*(torch.norm(positions[4]-positions[3])+torch.norm(positions[3]-positions[2])))
eef_poses = [
    [3.008, -0.668, 0.549, -0.16574, -0.268, 0.5065], # right back 0
    [-2.51890, -0.54126, -0.08807, -0.00074, -0.49342, 0.42817], # right back corner
    [-3.008, -0.5914, -0.633, -0.111, 0.2634, 0.485], # left back 1
    [-3.008, -0.5914, -0.633, -0.061, 0.2634, 0.485], # left back 1 x + 5cm
    [-3.008, -0.5914, -0.633, -0.011, 0.2634, 0.485], # left back 1 x + 10cm
    [2.31092, -0.24056, 0.99504, -0.21117, 0.12188, 0.41125], # left back 2
    
    [1.8205, 0.179, 0.2795, 0.256, 0.5694, 0.157], # left side 1
    [2.28403, 0.05687, 0.21994, 0.40305, 0.46293, 0.37193], # left side 2
    [-1.8205, 0.179, -0.2795, 0.256, -0.5694, 0.157], # right side 0
    [-2.28403, 0.05687, -0.21994, 0.40305, -0.46293, 0.37193] # right side 2
]
print(eef_poses)
eef_poses_6d=torch.tensor(eef_poses)
eef_mats=pose_to_transform(eef_poses_6d)
poses_scaled=poses.clone()
poses_scaled[:,:3,3]=poses_scaled[:,:3,3]*scale_factor
# world-space coordinates to camera pose
poses_scaled_np=np.array(torch.linalg.pinv(poses_scaled).cpu().detach())
# robot base to manipulator end-effector pose
eef_mats_np=np.array(torch.linalg.pinv(eef_mats).cpu().detach())

rot,tr=cv2.calibrateHandEye(eef_mats_np[:,:3,:3],eef_mats_np[:,:3,3],
        poses_scaled_np[:,:3,:3],poses_scaled_np[:,:3,3],method=cv2.CALIB_HAND_EYE_DANIILIDIS)

print("here", rot, tr)
print(residual_error(torch.tensor(rot).float(), torch.tensor(eef_mats_np[:,:3,:3]).float(),
      torch.tensor(poses_scaled_np[:,:3,:3]).float()))