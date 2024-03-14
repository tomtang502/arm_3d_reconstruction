import sys, os
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dust3r_local_dir = os.path.join(project_folder, "dust3r")
sys.path.append(dust3r_local_dir)
sys.path.append(project_folder)
from scipy.spatial.transform import Rotation as R
from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from configs.experiments_data_config import ArmDustrExpData
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

import cv2, torch, re
import numpy as np
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
from utils.geometric_util import *

data_config = ArmDustrExpData()
cam_poses_file_suffix = data_config.dust3r_cam_poses_file_suffix
ptc_file_suffix = data_config.ptc_file_suffix
img_size = data_config.img_size
std_model_pth = data_config.standard_model_pth 
"""
Several utils for running dust3r and extracting camera poses and points cloud.
Their functionality is literally explained by their name.
"""

def running_dust3r(exp_name, out_name=None, out_dir=data_config.dustr_out_pth, batch_size=4, 
                   schedule="cosine", lr=0.01, niter=320, device="cuda", model_path=std_model_pth):
    model = load_model(model_path, device)
    file_paths = data_config.get_images_paths(exp_name)
    images = load_images(file_paths, size=img_size)

    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    print(f"Status: dust3r finished running with a loss of {loss}")

    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    pts_list=[]
    for i in range(len(imgs)):
        sel_pts=pts3d[i][confidence_masks[i]]
        pts_list.append(sel_pts)
    pts_tor=torch.cat(pts_list).detach().cpu()
    poses = poses.detach().cpu()
    # Save each tensor separately
    if out_name == None:
        file_name = exp_name
    else:
        file_name = out_name
    output_pose_pth = os.path.join(out_dir, f"{file_name}{cam_poses_file_suffix}")
    output_pc_pth = os.path.join(out_dir, f"{file_name}{ptc_file_suffix}")
    torch.save(poses, output_pose_pth)
    torch.save(pts_tor, output_pc_pth)
    return output_pose_pth, output_pc_pth

def load_pose_ptc(output_pose_pth, output_pc_pth):
    pts_tor = torch.load(output_pc_pth)
    poses = torch.load(output_pose_pth)
    return poses, pts_tor

def load_pose_from_exp_name(exp_name):
    ptc_pth = data_config.get_ptc_output_path(exp_name)
    poses_pth = data_config.get_cam_pose_path(exp_name)
    if os.path.isfile(ptc_pth) and os.path.isfile(poses_pth):
        return load_pose_ptc(poses_pth, ptc_pth)
    else:
        print(f"{exp_name} has not been processed by dust3r yet, START processing...")
        if exp_name not in data_config.expnames:
            raise Exception(f"{exp_name} not configured, please configuring it in configs/experiments_data_config.py")
        else:
            output_pose_pth, output_pc_pth = running_dust3r(exp_name)
            return load_pose_ptc(output_pose_pth, output_pc_pth)

def extract_positions(transform_matrices):
    positions = transform_matrices[:, :3, 3]
    forward_directions = transform_matrices[:, :3, 2] # Negate to if cam face -Z direction
    return positions, forward_directions

if __name__ == "__main__":
    # This is for submodule test and can be treated as a sample usage
    exp_name = '8obj_divangs'
    output_pose_pth, output_pc_pth = running_dust3r(exp_name)
    poses, pts_tor = load_pose_ptc(output_pose_pth, output_pc_pth)
    print(output_pose_pth, output_pc_pth)
    print(poses.shape, pts_tor.shape)