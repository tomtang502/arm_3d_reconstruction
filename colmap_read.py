

# reconstruction.write("output/dc_saved_output/2bs2sslb3_sa_apriltag")
import os
import numpy as np
import open3d as o3d
import torch
import pycolmap
import utils.geometric_util as geomu
from utils.graph_util import graph_single_struct, graph_double_struct, plotty_graph_multistruct
from configs.experiments_data_config import ArmDustrExpData
from utils.scale_calib import *

"""
Use this script to manually caliberate due to uncertainty of which point colmap will take
"""

exp_config = ArmDustrExpData()
exp_name = '7obj_4cluster'
col_out_path = f"output/colmap_saved_output/{exp_name}"
reconstruction = pycolmap.Reconstruction(col_out_path)
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
    name = image.name[:-len('.jpg')].split('_')[2]
    idx = 0
    if len(name) == 1:
        idx = ord(name) - ord('a')
    else:
        idx = 26 + ord(name[1]) - ord('a')
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

ptc_tor = torch.tensor(ptc_xyz)
poses_tor = torch.tensor(np.stack(col_cam_poses))
poses_tor = torch.linalg.pinv(poses_tor)

#xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor_o))
xyz_eef = np.stack(geomu.tmatw2c_to_xyz(eef_poses_tor_selected))

### Solving for scale and then do caliberation (CHANGE HERE)
# linear_idx_x = [21, 16]
# x_d = 0.1
linear_idx_x = [5, 6, 7]
x_d = 0.05
im_poses_tor_o, ptc_tor_o = rescale_pose_ptc_col(col_cam_poses_map, poses_tor, ptc_tor, 
                                                 linear_idx_x, x_d)

### Use rearrange to make sure no consecutive linear matrices (CHANGE HERE)
#reorder_idxs = [22, 21, 20, 19, 18, 17, 16, 0, 13, 1, 14, 2, 15, 3, 4, 8, 9]
reorder_idxs = [17, 16, 7, 9, 8, 6, 18, 2, 5]
reorder_idxs = [idx_map[idx] for idx in reorder_idxs]
eef_poses_tor_calib, im_poses_tor_o_calib = geomu.rearrange(eef_poses_tor_selected.float(), 
                                                            im_poses_tor_o.float(), 
                                                            reorder_idxs)

# ###########################
# Verify that colmap has cam to world pose
# xyz_R = im_poses_tor_o[:, :3, :3].float()
# xyz_RT = -np.transpose(xyz_R, [0,2,1])
# xyz = im_poses_tor_o[:, :3, -1].float()
# for i in range(26):
#     xyz[i] = xyz_RT[i]@xyz[i]

# xyz_eef = eef_poses_tor_calib[:, :3, -1].float()
# ###########################

T = caculate_calib_trans_mat(eef_poses_tor_calib, im_poses_tor_o_calib)

im_poses_tor, ptc_tor = transpose_poses_ptc(im_poses_tor_o.float(), ptc_tor_o.float(), T)
#ptc_tor = ptc_tor[::10]
xyz = np.stack(geomu.tmatw2c_to_xyz(im_poses_tor))
xyz_eef = np.stack(geomu.tmatw2c_to_xyz(eef_poses_tor_selected))
plotty_graph_multistruct([xyz, xyz_eef, ptc_tor], ['cam_pose', 'eef', 'point cloud'], 
                         [2, 2, 0.5])


# Placing tensors in a dictionary
tensors_to_save = {
    'poses': im_poses_tor,
    'dense_pt': ptc_tor,
    'idx': torch.tensor(selected_idx)
}

# Saving the dictionary of tensors to a file
torch.save(tensors_to_save, os.path.join(col_out_path, 'colmap_out.pth'))