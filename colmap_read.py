

# reconstruction.write("output/dc_saved_output/2bs2sslb3_sa_apriltag")
import os
import numpy as np
import torch
import pycolmap
import utils.geometric_util as geomu
from utils.graph_util import graph_single_struct, graph_double_struct
from configs.experiments_data_config import ArmDustrExpData
from utils.scale_calib import *

exp_config = ArmDustrExpData()
exp_name = '2bs2sslb3_sa_apriltag'
reconstruction = pycolmap.Reconstruction(f"output/dc_saved_output/{exp_name}")
print(reconstruction.summary())

col_cam_poses = []
selected_idx = []
col_cam_poses_map = dict()
for image_id, image in reconstruction.images.items():
    selected_idx.append(image_id)
    img_pose = np.array(image.cam_from_world.matrix())
    pose_tmat = torch.tensor(geomu.colmap_pose2transmat(img_pose))
    col_cam_poses.append(pose_tmat)
    col_cam_poses_map[image_id] = pose_tmat.clone()
print(col_cam_poses_map)
ptc = []
for point3D_id, point3D in reconstruction.points3D.items():
    ptc.append(point3D.xyz)
ptc_tor = torch.tensor(np.array(ptc))
poses_tor = torch.tensor(np.stack(col_cam_poses))
xyz = np.stack(geomu.tmatw2c_to_xyz(poses_tor))
graph_double_struct(xyz, ptc_tor)


pose_data = exp_config.get_obs_config(exp_name)
eef_poses = pose_data.poses
eef_poses_tor=geomu.pose_to_transform(torch.tensor(eef_poses))
eef_poses_tor_cm = eef_poses_tor[selected_idx]
linear_idx_x, add_pts, x_d = pose_data.linearidx['x'] 
linear_idx_x = [i for i in selected_idx if i in linear_idx_x]

print(linear_idx_x, selected_idx)

### Solving for scale and then do caliberation
print(x_d)
im_poses_tor_o, ptc_tor_o = rescale_pose_ptc_col(col_cam_poses_map, poses_tor, ptc_tor, linear_idx_x, x_d)
xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor_o))

graph_double_struct(xyz, ptc_tor_o)




eef_poses_tor_calib, im_poses_tor_o_calib = filter_out_calib_poses(eef_poses_tor, im_poses_tor_o, 
                                                                 add_pts)
T = caculate_calib_trans_mat(eef_poses_tor_calib, im_poses_tor_o_calib)
im_poses_tor, ptc_tor = transpose_poses_ptc(im_poses_tor_o, ptc_tor_o, T)
