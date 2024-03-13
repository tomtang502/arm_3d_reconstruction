import torch
import numpy as np
from scipy.linalg import inv
import sys, os
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_folder)
import utils.park_martin as pm_calib
from utils.geometric_util import get_scale_factor, transform_points

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

def rescale_pose_ptc(im_poses_o, ptc, linear_idx_x, x_d):
    scale_factor = get_scale_factor(im_poses_o, linear_idx_x, x_d)
    print(f"scale factor for 3d reconstruction by dustr: {scale_factor}")
    im_poses_o[:,:3,3]=im_poses_o[:,:3,3]*scale_factor
    ptc = ptc*scale_factor
    return im_poses_o, ptc

def transpose_poses_ptc(poses, ptc, trans_mat):
    poses_trans = trans_mat.float()@poses
    ptc_trans = transform_points(ptc, trans_mat.float())
    return poses_trans, ptc_trans