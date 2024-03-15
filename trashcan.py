

# reconstruction.write("output/dc_saved_output/2bs2sslb3_sa_apriltag")
import os
import numpy as np
import open3d as o3d
import torch
import pycolmap
import utils.geometric_util as geomu
from utils.graph_util import graph_single_struct, graph_double_struct
from configs.experiments_data_config import ArmDustrExpData
from utils.scale_calib import *

"""
Use this script to manually caliberate due to uncertainty of which point colmap will take
"""

exp_config = ArmDustrExpData()
exp_name = '7obj_divangs'
reconstruction = pycolmap.Reconstruction(f"output/colmap_saved_output/{exp_name}")
print(reconstruction.summary())

pose_data = exp_config.get_obs_config(exp_name)
eef_poses = pose_data.poses + pose_data.additional_colmap_pose
eef_poses_tor=geomu.pose_to_transform(torch.tensor(eef_poses))

col_cam_poses = []
selected_idx = []
idx_map = dict()
col_cam_poses_map = dict()
i = 0
for image_id, image in reconstruction.images.items():
    name = image.name[:-len('.jpg')]
    idx = ord(name.split('_')[2]) - ord('a')
    print(idx, name)
    selected_idx.append(idx)
    idx_map[idx] = i
    i += 1
    img_pose = np.array(image.cam_from_world.matrix())
    pose_tmat = torch.tensor(geomu.colmap_pose2transmat(img_pose))
    col_cam_poses.append(pose_tmat)
    col_cam_poses_map[idx] = pose_tmat.clone()

print(selected_idx)
eef_poses_tor_selected = eef_poses_tor[selected_idx, :, :]

ply_path = os.path.join(exp_config.get_ptc_output_path(exp_name, exp_type=1), "dense.ply")
point_cloud = o3d.io.read_point_cloud(ply_path)
ptc_xyz = np.asarray(point_cloud.points)
# o3d.visualization.draw_geometries([point_cloud])
print(f"dense shape: {ptc_xyz.shape}")
if point_cloud.colors:
    # Extract color information
    ptc_colors = np.asarray(point_cloud.colors)
else:
    ptc_colors = None
    print("This point cloud has no color information.")

ptc_xyz = np.array([[1,2,3], [4.,5,6]])
ptc_tor = torch.tensor(ptc_xyz)
poses_tor = torch.tensor(np.stack(col_cam_poses))

#xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor_o))
xyz_eef = np.stack(geomu.tmatw2c_to_xyz(eef_poses_tor_selected))
# print(xyz_eef.shape)
# graph_single_struct(xyz_eef)
# graph_double_struct(xyz_eef, xyz_eef)
# print(poses_tor.shape)

### Solving for scale and then do caliberation (CHANGE HERE)
# linear_idx_x = [21, 16]
# x_d = 0.1
linear_idx_x = [5, 6, 7]
x_d = 0.05
im_poses_tor_o, ptc_tor_o = rescale_pose_ptc_col(col_cam_poses_map, poses_tor, ptc_tor, 
                                                 linear_idx_x, x_d)




### Use rearrange to make sure no consecutive linear matrices (CHANGE HERE)
#reorder_idxs = [22, 21, 20, 19, 18, 17, 16, 0, 13, 1, 14, 2, 15, 3, 4, 8, 9]
reorder_idxs = [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
reorder_idxs = [idx_map[idx] for idx in reorder_idxs]

print(reorder_idxs)
eef_poses_tor_calib, im_poses_tor_o_calib = geomu.rearrange(eef_poses_tor_selected.float(), 
                                                            im_poses_tor_o.float(), 
                                                            reorder_idxs)
print(eef_poses_tor_calib.shape, im_poses_tor_o.shape)
# xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor_o))
# xyz_eef = np.stack(geomu.tmatw2c_to_xyz(eef_poses_tor_calib))


###########################
xyz_R = im_poses_tor_o[:, :3, :3].float()
xyz_RT = -np.transpose(xyz_R, [0,2,1])
xyz = im_poses_tor_o[:, :3, -1].float()
for i in range(26):
    xyz[i] = xyz_RT[i]@xyz[i]

xyz_eef = eef_poses_tor_calib[:, :3, -1].float()
###########################


graph_double_struct(xyz, xyz_eef)


xyz = xyz - xyz.mean(axis=0)
xyz_eef = xyz_eef - xyz_eef.mean(axis=0)
A = xyz.T @ xyz_eef
U, S, Vt = np.linalg.svd(A)
R = np.dot(Vt.T, U.T)

# Special case: Ensuring a right-handed coordinate system
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = np.dot(Vt.T, U.T)

# Step 5: Apply the rotation and translation
A_aligned = np.dot(xyz, R.T)


graph_double_struct(A_aligned, xyz_eef)


# # Angle theta
# theta = np.pi / 2 - 0.001

# # Rotation matrix
# R = np.array([
#     [np.cos(theta), 0, np.sin(theta)],
#     [0, 1, 0],
#     [-np.sin(theta), 0, np.cos(theta)]
# ])

# # Translation vector
# t = np.array([-0.005, 0.123, 0])

# # Combine into 4x4 transformation matrix
# e2g_mat = np.eye(4)  # Start with an identity matrix
# e2g_mat[:3, :3] = R  # Set the rotation part
# e2g_mat[:3, 3] = t   # Set the translation part


# im_poses_tor_o_calib, ptc_tor = transpose_poses_ptc(im_poses_tor_o_calib.float(), ptc_tor_o.float(), T)
# xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor_o_calib))
# xyz_eef = np.stack(geomu.tmatw2c_to_xyz(eef_poses_tor_selected))
# graph_double_struct(xyz, xyz_eef)
# T = caculate_calib_trans_mat(end_effector_matrices, im_poses_tor_o_calib)

# im_poses_tor, ptc_tor = transpose_poses_ptc(im_poses_tor_o_calib.float(), ptc_tor_o.float(), T)
# xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor))
# xyz_eef = np.stack(geomu.tmatw2c_to_xyz(eef_poses_tor_selected))
# graph_double_struct(xyz, xyz_eef)
# # Placing tensors in a dictionary
# tensors_to_save = {
#     'tensor1': tensor1,
#     'tensor2': tensor2,
#     'tensor3': tensor3
# }

# # Saving the dictionary of tensors to a file
# torch.save(tensors_to_save, 'tensors.pth')