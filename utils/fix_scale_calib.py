import torch
import numpy as np
from scipy.linalg import inv
import sys, os
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_folder)
import utils.park_martin as pm_calib
from utils.geometric_util import transform_points, apply_transform_pt

"""
Calculae the transformation matrix, which used method described by Park, Frank C and Martin, Bryan J,
see utils/park_martin.py for detail
[IN] eef_poses_tor: The tensor of end-effector poses
[IN] im_poses_tor: The tensor of corresponding world to camera transfomation matrices generated 
    by dust3r.
[OUT] The Transformation matrix to transform from world coordinate to base coordinate.
NOTICE: No linear motion should be included consecutively in both tensor, so call 
    filter_out_calib_poses before passing the two input in this function. Make sure the 
    scaling is solved befor run this function (ie. The two input should be consistant under 
    Left-Invariant Riemannian Metric).
"""
def caculate_calib_trans_mat(eef_poses_tor, im_poses_tor):
    A, B = [], []
    for i in range(1,len(im_poses_tor)):
        p = eef_poses_tor[i-1], im_poses_tor[i-1]
        n = eef_poses_tor[i], im_poses_tor[i]
        A.append(np.dot(inv(p[0]), n[0]))
        B.append(np.dot(inv(p[1]), n[1]))

    X = np.eye(4)
    Rx, tx = pm_calib.calibrate(A, B)
    X[0:3, 0:3] = Rx
    X[0:3, -1] = tx


    tmp_list=[]
    for i in range(len(im_poses_tor)):
        rob = eef_poses_tor[i]
        obj = im_poses_tor[i]
        tmp = np.dot(rob, np.dot(X, inv(obj)))
        tmp_list.append(tmp)
    tmp_array = np.stack(tmp_list)
    tmp_tor=torch.tensor(tmp_array)

    world_pose=tmp_tor.mean(dim=0)
    return world_pose

def filter_out_calib_poses(eef_poses_tor_o, im_poses_tor_o, additional_axis_cam_pose_idx):

    selected_idx = [i for i in range(eef_poses_tor_o.shape[0]) 
                    if i not in additional_axis_cam_pose_idx]
    eef_poses_tor=eef_poses_tor_o[selected_idx]
    im_poses_tor=im_poses_tor_o[selected_idx]
    return eef_poses_tor, im_poses_tor

"""
Compute the scale factor given the following
[IN] cam_poses is an array of world to camera transformation marix.
[IN] linear_idx_axis index into the w3c (cam_poses) to get corresponding transformation 
    matrices that have cam poses moving parallel to the specified axis.
[IN] axis_d is the distance between each consecutive two cam poses in real world.
    (here the variable name use x becuase it is the pose example we have in cam_pose config)
[OUT] Scale factor to scale the w2c transformation matrix.
"""
def get_scale_factor(cam_poses, linear_idx_x, x_d):
    sum_dist = 0
    n = len(linear_idx_x)
    for i in range(n - 1):
        idx1 = linear_idx_x[i]
        idx2 = linear_idx_x[i+1]
        pt2pt_trans = np.matmul(torch.linalg.pinv(cam_poses[idx1]), cam_poses[idx2])
        vec = apply_transform_pt([0.0, 0.0, 0.0], pt2pt_trans)
        #print(idx, cam_xyz_L[idx, cam_xyz_L[idx + 1,0])
        sum_dist += np.linalg.norm(vec)
    scale_factor = x_d/(sum_dist/float(n-1))
    return scale_factor

# This assume input selected by linear_idx_x is on a line
def get_scale_factor_col(cam_poses_map, linear_idx_x, x_d):
    sum_dist = 0
    n = len(linear_idx_x)
    for i in range(n - 1):
        idx1 = linear_idx_x[i]
        idx2 = linear_idx_x[i+1]
        pt2pt_trans = np.matmul(torch.linalg.pinv(cam_poses_map[idx1]), cam_poses_map[idx2])
        vec = apply_transform_pt([0.0, 0.0, 0.0], pt2pt_trans)
        #print(idx, cam_xyz_L[idx, cam_xyz_L[idx + 1,0])
        sum_dist += np.linalg.norm(vec)
    scale_factor = x_d/(sum_dist/float(n-1))
    return scale_factor

def rescale_pose_ptc(im_poses_o, ptc, linear_idx_x, x_d):
    scale_factor = get_scale_factor(im_poses_o, linear_idx_x, x_d)
    print(f"scale factor for 3d reconstruction by dustr: {scale_factor}")
    im_poses_o[:,:3,3]=im_poses_o[:,:3,3]*scale_factor
    ptc = ptc*scale_factor
    return im_poses_o, ptc

def rescale_pose_ptc_col(cam_poses_map, poses_tor, ptc, linear_idx_x, x_d):
    scale_factor = get_scale_factor_col(cam_poses_map, linear_idx_x, x_d)
    print(f"scale factor for 3d reconstruction by colmap: {scale_factor}")
    for k in cam_poses_map:
        cam_poses_map[k][:3,3] = cam_poses_map[k][:3,3]*scale_factor
    poses_tor[:,:3,3]=poses_tor[:,:3,3]*scale_factor
    ptc = ptc*scale_factor
    return poses_tor, ptc

def rescale_pose_tag(cam_poses_map, poses_tor, linear_idx_x, x_d):
    scale_factor = get_scale_factor_col(cam_poses_map, linear_idx_x, x_d)
    print(f"scale factor for 3d reconstruction by colmap: {scale_factor}")
    for k in range(len(cam_poses_map)):
        if isinstance(cam_poses_map[k], torch.Tensor):
            cam_poses_map[k][:3, 3] = cam_poses_map[k][:3, 3] * scale_factor
    poses_tor[:,:3,3]=poses_tor[:,:3,3]*scale_factor
    return poses_tor

def transpose_poses_ptc(poses, ptc, trans_mat):
    poses_trans = trans_mat.float()@poses
    ptc_trans = transform_points(ptc, trans_mat.float())
    return poses_trans, ptc_trans
