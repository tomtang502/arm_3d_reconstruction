

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
exp_name = '8obj_4cluster'
reconstruction = pycolmap.Reconstruction(f"output/colmap_saved_output/{exp_name}")
print(reconstruction.summary())

col_cam_poses = []
selected_idx = []
idx_map = dict()
col_cam_poses_map = dict()
i = 0
for image_id, image in reconstruction.images.items():
    name = image.name
    print(name)
    idx = ord(name.strip('.jpg').split('_')[2]) - ord('a')
    print(idx)
    selected_idx.append(idx)
    idx_map[idx] = i
    i += 1
    img_pose = np.array(image.cam_from_world.matrix())
    pose_tmat = torch.tensor(geomu.colmap_pose2transmat(img_pose))
    col_cam_poses.append(pose_tmat)
    col_cam_poses_map[idx] = pose_tmat.clone()

ply_path = os.path.join(exp_config.get_ptc_output_path(exp_name, exp_type=1), "dense.ply")
point_cloud = o3d.io.read_point_cloud(ply_path)
ptc_xyz = np.asarray(point_cloud.points)
if point_cloud.colors:
    # Extract color information
    ptc_colors = np.asarray(point_cloud.colors)
else:
    ptc_colors = None
    print("This point cloud has no color information.")

ptc_tor = torch.tensor(ptc_xyz)
poses_tor = torch.tensor(np.stack(col_cam_poses))
xyz = np.stack(geomu.tmatw2c_to_xyz(poses_tor))
#graph_double_struct(xyz, ptc_tor)


pose_data = exp_config.get_obs_config(exp_name)
eef_poses = pose_data.poses + pose_data.additional_colmap_pose
eef_poses_tor=geomu.pose_to_transform(torch.tensor(eef_poses))
eef_poses_tor_cm = eef_poses_tor[selected_idx]
print(selected_idx)

### Solving for scale and then do caliberation (CHANGE HERE)
# linear_idx_x = [21, 16]
# x_d = 0.1
linear_idx_x = [21, 22, 23]
x_d = 0.05
im_poses_tor_o, ptc_tor_o = rescale_pose_ptc_col(col_cam_poses_map, poses_tor, ptc_tor, 
                                                 linear_idx_x, x_d)
xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor_o))
graph_double_struct(xyz, ptc_tor_o)



### Use rearrange to make sure no consecutive linear matrices (CHANGE HERE)
reorder_idxs = [22, 17, 20, 23, 19]
reorder_idxs = [idx_map[i] for i in reorder_idxs]

print(eef_poses_tor_cm, im_poses_tor_o)
eef_poses_tor_calib, im_poses_tor_o_calib = geomu.rearrange(eef_poses_tor_cm.float(), 
                                                            im_poses_tor_o.float(), 
                                                            reorder_idxs)
print(eef_poses_tor_calib, im_poses_tor_o_calib)
T = caculate_calib_trans_mat(eef_poses_tor_calib, im_poses_tor_o_calib)
im_poses_tor, ptc_tor = transpose_poses_ptc(im_poses_tor_o.float(), ptc_tor_o.float(), T)
xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor))
graph_double_struct(xyz, ptc_tor)
# # Placing tensors in a dictionary
# tensors_to_save = {
#     'tensor1': tensor1,
#     'tensor2': tensor2,
#     'tensor3': tensor3
# }

# # Saving the dictionary of tensors to a file
# torch.save(tensors_to_save, 'tensors.pth')