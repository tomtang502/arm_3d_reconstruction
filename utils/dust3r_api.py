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

def running_dust3r(exp_name, num_imgs, out_dir=data_config.dustr_out_pth, batch_size=4, 
                   schedule="cosine", lr=0.01, niter=380, device="cuda", model_path=std_model_pth):
    model = load_model(model_path, device)
    file_paths = data_config.get_images_paths(exp_name, num_imgs=num_imgs)
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
    print(type(pts3d), len(pts3d), type(pts3d[0]), pts3d[0].shape)
    confidence_masks = scene.get_masks()
    print(type(confidence_masks), len(confidence_masks), type(confidence_masks[0]), confidence_masks[0].shape)
    pts_list=[]

    pts3d_np = to_numpy(pts3d)
    masks_np = to_numpy(confidence_masks)
    print(scene.imgs[0].shape)
    pts = np.concatenate([p[m] for p, m in zip(pts3d_np, masks_np)])
    
    rgb_colors = np.concatenate([p[m] for p, m in zip(imgs, masks_np)])
    print(pts.shape, rgb_colors.shape)

    pt_loc = []
    h, w, _ = pts3d_np[0].shape
    for i in range(len(imgs)):
        sel_pts=pts3d[i][confidence_masks[i]]
        pts_list.append(sel_pts)
        # Generate coordinate matrices for rows and columns
        rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        # Stack rows and columns to create the new array
        loc_array = np.stack((rows, cols), axis=-1)
        loc_array = loc_array[masks_np[i]]
        loc_array = np.hstack((loc_array, np.ones((loc_array.shape[0], 1), dtype=int) * i))
        pt_loc.append(loc_array)

    imgs_tor = torch.tensor(np.array(scene.imgs)).detach().cpu()
    masks_tor = torch.tensor(np.array(masks_np))
    pt_loc_tor = torch.tensor(np.concatenate(pt_loc))
    print(pt_loc_tor.shape)
    pts_tor=torch.cat(pts_list).detach().cpu()
    poses = poses.detach().cpu()  

    tensors_to_save = {
        'poses': poses,
        'pts': pts_tor,
        'rgb_colors': rgb_colors,
        'loc_info' : pt_loc_tor, # row, col, img_idx
        'images': imgs_tor, 
        'masks': masks_tor,
    }
    saving_loc = os.path.join(out_dir, f'{exp_name}_{num_imgs}_raw.pth')
    torch.save(tensors_to_save, saving_loc)
    print("-"*10)
    print(f"dust3r initial output saved at {saving_loc}")
    print("-"*10)
    return saving_loc

def load_pose_ptc(dict_loc):
    d = torch.load(dict_loc)
    poses = d['poses']
    pts_tor = d['pts']
    rgb_colors = d['rgb_colors']
    loc_info = d['loc_info']
    # may not needed
    compressed_imgs = d['images']
    masks = d['masks']
    return poses, pts_tor, rgb_colors, loc_info

def load_pose_from_exp_name(exp_name, num_imgs, out_dir=data_config.dustr_out_pth):
    saving_loc = os.path.join(out_dir, f'{exp_name}_{num_imgs}_raw.pth')
    if os.path.isfile(saving_loc):
        return load_pose_ptc(saving_loc)
    else:
        print(f"{exp_name} has not been processed by dust3r yet, START processing...")
        if exp_name not in data_config.expnames:
            raise Exception(f"{exp_name} not configured, please configuring it in configs/experiments_data_config.py")
        else:
            saving_loc = running_dust3r(exp_name, num_imgs)
            return load_pose_ptc(saving_loc)

def extract_positions(transform_matrices):
    positions = transform_matrices[:, :3, 3]
    forward_directions = transform_matrices[:, :3, 2] # Negate to if cam face -Z direction
    return positions, forward_directions

if __name__ == "__main__":
    # This is for submodule test and can be treated as a sample usage
    exp_name = '7obj_backonly'
    output_pose_pth, output_pc_pth = running_dust3r(exp_name, 8, niter=10)
    poses, pts_tor = load_pose_ptc(output_pose_pth, output_pc_pth)
    print(output_pose_pth, output_pc_pth)
    print(poses.shape, pts_tor.shape)